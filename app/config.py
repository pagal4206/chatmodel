from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class Settings:
    base_model_name: str
    adapter_path: str | None
    merged_model_path: str | None
    load_in_4bit: bool
    api_host: str
    api_port: int
    chat_max_history_turns: int
    default_max_new_tokens: int
    default_temperature: float
    default_top_p: float
    telegram_bot_token: str | None
    telegram_api_url: str

    @property
    def active_model_reference(self) -> str:
        return self.merged_model_path or self.adapter_path or self.base_model_name


def load_settings() -> Settings:
    return Settings(
        base_model_name=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
        adapter_path=os.getenv("ADAPTER_PATH") or None,
        merged_model_path=os.getenv("MERGED_MODEL_PATH") or None,
        load_in_4bit=_get_bool("LOAD_IN_4BIT", False),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        chat_max_history_turns=int(os.getenv("CHAT_MAX_HISTORY_TURNS", "6")),
        default_max_new_tokens=int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "180")),
        default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.85")),
        default_top_p=float(os.getenv("DEFAULT_TOP_P", "0.92")),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_api_url=os.getenv("TELEGRAM_API_URL", "http://127.0.0.1:8000/chat"),
    )
