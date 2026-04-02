from __future__ import annotations

import logging
from collections import defaultdict

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from app.config import load_settings
from app.utils.text import clip_history, normalize_text


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("chatbaka.telegram")

CHAT_SESSIONS: dict[int, list[dict[str, str]]] = defaultdict(list)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        "Namaste! Main aapka Hinglish bot hoon. Mujhe normal chat ki tarah message bhejo. "
        "Agar context reset karna ho to /reset likho."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    CHAT_SESSIONS[update.effective_chat.id] = []
    await update.effective_message.reply_text("Context reset ho gaya. Ab fresh chat start kar sakte ho.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = load_settings()
    if update.effective_chat is None or update.effective_message is None or not update.effective_message.text:
        return

    chat_id = update.effective_chat.id
    user_text = normalize_text(update.effective_message.text)
    history = clip_history(CHAT_SESSIONS[chat_id], settings.chat_max_history_turns)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                settings.telegram_api_url,
                json={
                    "message": user_text,
                    "history": history,
                },
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.exception("API call failed")
        await update.effective_message.reply_text(
            f"Reply lane mein issue aa gaya: {exc}. API server chal raha hai kya?"
        )
        return

    payload = response.json()
    CHAT_SESSIONS[chat_id] = payload.get("history", [])
    await update.effective_message.reply_text(payload["response"])


def main() -> None:
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing hai. Pehle .env set karo.")

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
