from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.prompts import DEFAULT_SYSTEM_PROMPT
from app.utils.text import clip_history, normalize_text


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 180
    temperature: float = 0.85
    top_p: float = 0.92
    repetition_penalty: float = 1.08


class HinglishChatEngine:
    def __init__(
        self,
        *,
        base_model_name: str,
        adapter_path: str | None = None,
        merged_model_path: str | None = None,
        load_in_4bit: bool = True,
        max_history_turns: int = 6,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.merged_model_path = merged_model_path
        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt

        tokenizer_source = merged_model_path or adapter_path or base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = self._build_model_kwargs(load_in_4bit)
        if merged_model_path:
            self.model = AutoModelForCausalLM.from_pretrained(merged_model_path, **model_kwargs)
        elif adapter_path:
            self.model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _build_model_kwargs(self, load_in_4bit: bool) -> dict[str, object]:
        model_kwargs: dict[str, object] = {"trust_remote_code": False}

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = torch.float32

        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit loading ke liye CUDA GPU chahiye.")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs.pop("torch_dtype", None)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        return model_kwargs

    def build_messages(
        self,
        *,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": normalize_text(system_prompt or self.system_prompt)}]
        messages.extend(clip_history(history, self.max_history_turns))
        messages.append({"role": "user", "content": normalize_text(user_message)})
        return messages

    @torch.inference_mode()
    def chat(
        self,
        *,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        config = generation_config or GenerationConfig()
        messages = self.build_messages(
            user_message=user_message,
            history=history,
            system_prompt=system_prompt,
        )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=max(config.temperature, 1e-5),
            top_p=config.top_p,
            do_sample=config.temperature > 0,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = generated[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return normalize_text(response)

