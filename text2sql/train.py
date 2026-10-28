"""QLoRA supervised fine-tuning with TRL.

The model is loaded in 4-bit NF4 and only LoRA adapters are trained. Loss is
computed on the SQL completion only, not on the schema/question prompt.

Usage:
    python -m text2sql.train --config configs/qwen2.5-coder-1.5b.yaml
    python -m text2sql.train --config configs/qwen2.5-coder-1.5b.yaml --max-steps 5 --limit 64   # smoke test
    python -m text2sql.train --config configs/qwen2.5-coder-1.5b.yaml --resume                  # continue after a session timeout
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Kaggle's "T4 x2" exposes two GPUs, and Trainer would wrap the model in
# nn.DataParallel, which SFTTrainer rejects. Train on one GPU unless the caller
# chose devices explicitly. Must run before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


def supports_bf16() -> bool:
    # T4 / P100 (Kaggle) are pre-Ampere: no native bf16, so train in fp16 there.
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def fix_param_dtypes(model, compute_dtype: torch.dtype) -> None:
    """Keep trainable (LoRA) weights in fp32 and frozen weights in compute_dtype.

    Qwen ships bf16 weights, and some may stay bf16 even when fp16 is requested.
    The fp16 GradScaler cannot unscale bf16 gradients, and T4s have no native
    bf16, so cast: trainable -> fp32, frozen floating-point -> compute_dtype.
    4-bit quantized weights (uint8) are left alone.
    """
    from collections import Counter

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
        elif p.is_floating_point() and p.dtype != compute_dtype and p.dtype != torch.float32:
            p.data = p.data.to(compute_dtype)
    counts = Counter((str(p.dtype), p.requires_grad) for p in model.parameters())
    print("param dtypes (dtype, trainable): count ->", dict(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", help="override output_dir from the config")
    ap.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--limit", type=int, help="use only the first N training examples")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "train_config.yaml")

    bf16 = supports_bf16()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | precision: {'bf16' if bf16 else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    data_files = {"train": cfg["train_file"]}
    if cfg.get("val_file"):
        data_files["validation"] = cfg["val_file"]
    ds = load_dataset("json", data_files=data_files)
    ds = ds.select_columns(["prompt", "completion"])
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
        if "validation" in ds:
            ds["validation"] = ds["validation"].select(range(min(args.limit // 4 or 1, len(ds["validation"]))))
    print({k: len(v) for k, v in ds.items()})

    train_kwargs = dict(cfg["training"])
    if args.max_steps:
        train_kwargs.update(max_steps=args.max_steps, eval_steps=args.max_steps, save_steps=args.max_steps)
    if "validation" not in ds:
        train_kwargs["eval_strategy"] = "no"
    if cfg.get("hub_model_id"):
        # Push checkpoints to the Hub so training can resume in a new Kaggle session.
        train_kwargs.update(push_to_hub=True, hub_model_id=cfg["hub_model_id"],
                            hub_strategy="checkpoint", hub_private_repo=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=cfg["max_length"],
        completion_only_loss=True,
        packing=False,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **train_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    fix_param_dtypes(trainer.model, compute_dtype)
    trainer.model.print_trainable_parameters()

    resume_from = None
    if args.resume:
        resume_from = get_last_checkpoint(str(output_dir))
        print(f"Resuming from {resume_from}" if resume_from else "No checkpoint found, starting fresh")

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if "validation" in ds:
        print("Final eval:", trainer.evaluate())
    if cfg.get("hub_model_id"):
        trainer.push_to_hub()
    print(f"Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
