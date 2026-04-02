from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for a Hinglish chatbot.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--validation-file", default="data/processed/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/hinglish-qwen25-3b-lora")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--assistant-only-loss", action="store_true")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    return parser.parse_args()


def build_model_kwargs(load_in_4bit: bool) -> dict[str, object]:
    model_kwargs: dict[str, object] = {"trust_remote_code": False}
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit fine-tuning ke liye CUDA GPU chahiye.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    return model_kwargs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **build_model_kwargs(args.load_in_4bit))
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        completion_only_loss=True,
        assistant_only_loss=args.assistant_only_loss,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_num_proc=args.dataset_num_proc,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    training_summary = {
        "model_name": args.model_name,
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]),
        "output_dir": str(final_adapter_dir),
        "load_in_4bit": args.load_in_4bit,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()

