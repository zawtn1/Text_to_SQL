"""Generate SQL predictions with a local model (base, or base + LoRA adapter).

Usage:
    # base model
    python -m text2sql.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out predictions/base_1.5b.jsonl
    # fine-tuned
    python -m text2sql.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter /kaggle/working/outputs/qwen2.5-coder-1.5b-spider/final_adapter --out predictions/ft_1.5b.jsonl
    # same, with vLLM (much faster if installed)
    python -m text2sql.generate --backend vllm ...

Decoding is greedy so runs are reproducible and comparable across setups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.sql_utils import extract_sql


def load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def generate_hf(args, rows: list[dict]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto", quantization_config=quant)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.load_in_4bit:
            model = model.merge_and_unload()  # faster inference; no effect on outputs
    model.eval()

    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
    # Sort by length so each batch has similar padding, then restore order.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outputs[i] = t
        print(f"  {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
    return outputs


def generate_vllm(args, rows: list[dict]) -> list[str]:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    bf16 = torch.cuda.get_device_capability()[0] >= 8
    llm = LLM(
        model=args.model,
        dtype="auto" if bf16 else "half",
        max_model_len=args.max_model_len,
        enable_lora=bool(args.adapter),
        max_lora_rank=64,
        gpu_memory_utilization=0.9,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    lora = LoRARequest("sql", 1, args.adapter) if args.adapter else None
    results = llm.chat([r["prompt"] for r in rows], params, lora_request=lora)
    return [r.outputs[0].text for r in results]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model name or path")
    ap.add_argument("--adapter", help="LoRA adapter directory (omit for the base model)")
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096, help="vLLM only")
    ap.add_argument("--load-in-4bit", action="store_true", help="hf only; for 7B on a 16 GB GPU")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    rows = load_rows(args.data, args.limit)
    print(f"Generating {len(rows)} predictions with {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    raw = generate_vllm(args, rows) if args.backend == "vllm" else generate_hf(args, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r, text in zip(rows, raw):
            f.write(json.dumps({
                "id": r["id"],
                "db_id": r["db_id"],
                "question": r["question"],
                "gold": r["query"],
                "pred": extract_sql(text),
                "raw": text,
                "model": args.model,
                "adapter": args.adapter,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
