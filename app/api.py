from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import Settings, load_settings
from app.inference import GenerationConfig, HinglishChatEngine
from app.utils.text import clip_history, normalize_text


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=32, le=512)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    top_p: float | None = Field(default=None, ge=0.1, le=1.0)


class ChatResponse(BaseModel):
    response: str
    history: list[ChatMessage]
    model: str


app = FastAPI(title="ChatBaka Hinglish API", version="0.1.0")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_engine() -> HinglishChatEngine:
    settings = get_settings()
    return HinglishChatEngine(
        base_model_name=settings.base_model_name,
        adapter_path=settings.adapter_path,
        merged_model_path=settings.merged_model_path,
        load_in_4bit=settings.load_in_4bit,
        max_history_turns=settings.chat_max_history_turns,
    )


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.active_model_reference,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    history = [message.model_dump() for message in request.history]
    message = normalize_text(request.message)

    generation_config = GenerationConfig(
        max_new_tokens=request.max_new_tokens or settings.default_max_new_tokens,
        temperature=request.temperature if request.temperature is not None else settings.default_temperature,
        top_p=request.top_p if request.top_p is not None else settings.default_top_p,
    )
    response_text = get_engine().chat(
        user_message=message,
        history=history,
        system_prompt=request.system_prompt,
        generation_config=generation_config,
    )

    updated_history = clip_history(
        history + [{"role": "user", "content": message}, {"role": "assistant", "content": response_text}],
        settings.chat_max_history_turns,
    )
    return ChatResponse(
        response=response_text,
        history=[ChatMessage(**item) for item in updated_history],
        model=settings.active_model_reference,
    )
