from __future__ import annotations

import re
from typing import Iterable


SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_text(text: object) -> str:
    if text is None:
        return ""

    normalized = str(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", " ")
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("“", '"').replace("”", '"')

    lines = [SPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(line for line in lines if line)
    normalized = MULTI_BLANK_RE.sub("\n\n", normalized)
    return normalized.strip()


def is_valid_text(text: object, *, min_chars: int = 2, max_chars: int = 1200) -> bool:
    cleaned = normalize_text(text)
    return min_chars <= len(cleaned) <= max_chars


def clip_history(messages: Iterable[dict[str, str]] | None, max_turns: int) -> list[dict[str, str]]:
    if not messages or max_turns <= 0:
        return []

    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = normalize_text(message.get("content", ""))
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})

    max_messages = max_turns * 2
    return cleaned[-max_messages:]


def dedupe_signature(prompt: list[dict[str, str]], completion: list[dict[str, str]]) -> str:
    last_user = ""
    for message in reversed(prompt):
        if message["role"] == "user":
            last_user = message["content"]
            break

    assistant = completion[0]["content"] if completion else ""
    return f"{last_user.lower()}||{assistant.lower()}"

