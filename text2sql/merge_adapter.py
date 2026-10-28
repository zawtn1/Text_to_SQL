"""Merge a LoRA adapter into the base model and save full weights.

vLLM can serve a base model + adapter directly, but a merged model is simpler
to deploy. Run this after training, on a machine with enough RAM/VRAM to hold
the base model in fp16 (~3.5 GB for 1.5B, ~15 GB for 7B).

Usage:
    python -m text2sql.merge_adapter --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter /kaggle/working/outputs/qwen2.5-coder-1.5b-spider/final_adapter \
        --out /kaggle/working/merged-1.5b [--push-to-hub your-user/qwen2.5-coder-1.5b-text2sql]
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push-to-hub", metavar="REPO_ID")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
