from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default="artifacts/hinglish-qwen25-3b-lora/final_adapter")
    parser.add_argument("--output-dir", default="artifacts/hinglish-qwen25-3b-merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
