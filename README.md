# Hinglish Chatbot Starter

Ye repo ek practical starter hai jisse aap Hinglish chatbot fine-tune kar sakte ho, local API pe chala sakte ho, aur Telegram bot se connect kar sakte ho.

Important reality check: scratch se naya LLM train karna bahut mehenga hota hai. Isliye yahan fine-tuning flow diya gaya hai:

1. Public Hinglish datasets ko combine karo
2. `Qwen/Qwen2.5-3B-Instruct` par LoRA / QLoRA fine-tune karo
3. Adapter ko serve karo
4. Telegram bot ko API se connect karo

## Quick Start

Windows par direct chal sakta hai, lekin GPU fine-tuning ke liye WSL2 ya Linux generally smoother hota hai.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Default dataset build:

```powershell
python scripts/prepare_dataset.py
```

`data/custom/` ke andar agar `input,output` format wali CSV file hogi, jaise [hinglish_conversations.csv](D:/chatbaka-main/chatbot/data/custom/hinglish_conversations.csv), to script usse bhi automatically include karegi.

Agar [hinglish_conversations.csv](D:/chatbaka-main/chatbot/data/custom/hinglish_conversations.csv) present hai, to main synthetic dataset ke liye script isi local file ko prefer karegi. Matlab extra download ki zarurat nahi padegi.

Outputs:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/metadata.json`

Training:

```powershell
python scripts/train_lora.py --load-in-4bit --packing
```

Output adapter:

- `artifacts/hinglish-qwen25-3b-lora/final_adapter`

Optional merge:

```powershell
python scripts/merge_adapter.py
```

API run:

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Note: `.env.example` me `LOAD_IN_4BIT=false` safe default diya gaya hai. Agar GPU available hai aur 4-bit inference chahiye to isse `true` kar do.

Telegram bot run:

```powershell
python -m app.bot
```

## Dataset Mix

Default pipeline in files ko use karti hai:

- local `data/custom/hinglish_conversations.csv` if present, warna `Abhishekcr448/Hinglish-Everyday-Conversations-1M`
- `ankitdhiman/hinglish-conversations` turn pairs
- `ankitdhiman/hinglish-conversations` multi-turn chat format
- `sk-community/romanized_hindi`
- `data/custom/*.jsonl`
- `data/custom/*.csv`

Agar aur bada dataset chahiye to synthetic rows badha do:

```powershell
python scripts/prepare_dataset.py --synthetic-max 1000000 --turn-pairs-max 150000 --chat-format-max 20000
```

## Telegram Setup

1. Telegram me `@BotFather` open karo
2. `/newbot` chalao
3. Token copy karke `.env` me `TELEGRAM_BOT_TOKEN` me daalo
4. `TELEGRAM_API_URL=http://127.0.0.1:8000/chat` set rakho
5. API aur bot dono run karo

Bot commands:

- `/start`
- `/reset`

## Custom Data

Bot ko zyada human-like banana hai to `data/custom/*.jsonl` me apne examples daalo. Supported formats:

Prompt-completion format:

```json
{"prompt":[{"role":"system","content":"..."},{"role":"user","content":"kya kar rahe ho?"}],"completion":[{"role":"assistant","content":"bas tumse baat kar raha hoon"}]}
```

Simple input-output format:

```json
{"input":"aaj mood off hai","output":"samajh sakta hoon, kya hua?"}
```

Multi-turn format:

```json
{"messages":[{"role":"user","content":"kal exam hai"},{"role":"assistant","content":"revision ho gaya kya?"},{"role":"user","content":"aadha hua"}]}
```

CSV format:

```csv
input,output
kya kar rahe ho?,bas chill kar raha hoon
kal mil rahe ho?,haan shaam ko milte hain
```

Agar aap sirf local CSV / custom data par dataset banana chahte ho to:

```powershell
python scripts/prepare_dataset.py --synthetic-max 0 --turn-pairs-max 0 --chat-format-max 0 --romanized-max 0
```

## Better Quality Tips

- apni real chat style ke examples add karo
- funny, emotional, practical, supportive examples mix karo
- over-formal prompt mat rakho
- Telegram par bot test karke bad replies ko custom dataset me add karo aur re-train karo

## Practical Hardware

- QLoRA training ke liye 12-16 GB VRAM practical hota hai
- Agar local GPU nahi hai to Colab, Kaggle, RunPod, Lambda Labs, ya Paperspace use kar sakte ho
- `bitsandbytes` low-memory quantized workflows ke liye useful hota hai

## File Guide

- `scripts/prepare_dataset.py`: public + custom data ko prompt-completion SFT format me convert karta hai
- `scripts/train_lora.py`: LoRA / QLoRA training script
- `scripts/merge_adapter.py`: adapter merge utility
- `app/api.py`: local FastAPI chat server
- `app/inference.py`: model loading + generation
- `app/bot.py`: Telegram polling bot

## Sources

- Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
- Transformers chat templating docs: https://huggingface.co/docs/transformers/en/chat_templating
- Datasets docs: https://huggingface.co/docs/datasets/loading
- BitsAndBytes docs: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Hinglish 1M dataset: https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M
- Hinglish conversations dataset: https://huggingface.co/datasets/ankitdhiman/hinglish-conversations
- Romanized Hindi dataset: https://huggingface.co/datasets/sk-community/romanized_hindi
- python-telegram-bot docs: https://docs.python-telegram-bot.org/
