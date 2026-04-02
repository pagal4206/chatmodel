from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from app.prompts import DEFAULT_SYSTEM_PROMPT, ROMANIZATION_SYSTEM_PROMPT
from app.utils.text import dedupe_signature, is_valid_text, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a large Hinglish SFT dataset.")
    parser.add_argument("--output-dir", default="data/processed", help="Where train/validation JSONL files will be written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--synthetic-max", type=int, default=300000, help="Rows from Abhishekcr448/Hinglish-Everyday-Conversations-1M.")
    parser.add_argument("--local-synthetic-file", default="data/custom/hinglish_conversations.csv", help="Preferred local CSV with input/output columns for the main Hinglish pair dataset.")
    parser.add_argument("--turn-pairs-max", type=int, default=80000, help="Rows from ankitdhiman/hinglish-conversations turn_pairs config.")
    parser.add_argument("--chat-format-max", type=int, default=10000, help="Conversation rows from ankitdhiman/hinglish-conversations chat_format config.")
    parser.add_argument("--romanized-max", type=int, default=20000, help="Rows from sk-community/romanized_hindi for style support.")
    parser.add_argument("--max-context-messages", type=int, default=8, help="Previous user/assistant messages to keep in multi-turn examples.")
    parser.add_argument("--custom-dir", default="data/custom", help="Optional folder with custom .jsonl or input/output CSV training data.")
    return parser.parse_args()


def select_subset(dataset, max_rows: int, seed: int):
    if max_rows <= 0:
        return dataset.select([])
    if max_rows >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_rows))


def make_example(
    *,
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
    source: str,
) -> dict[str, object] | None:
    if not prompt or not completion:
        return None

    cleaned_prompt: list[dict[str, str]] = []
    for message in prompt:
        role = str(message.get("role", "")).strip().lower()
        content = normalize_text(message.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            continue
        cleaned_prompt.append({"role": role, "content": content})

    cleaned_completion: list[dict[str, str]] = []
    for message in completion:
        role = str(message.get("role", "")).strip().lower()
        content = normalize_text(message.get("content", ""))
        if role != "assistant" or not content:
            continue
        cleaned_completion.append({"role": role, "content": content})

    if not cleaned_prompt or not cleaned_completion:
        return None

    user_messages = [message for message in cleaned_prompt if message["role"] == "user"]
    if not user_messages:
        return None

    if not is_valid_text(user_messages[-1]["content"]) or not is_valid_text(cleaned_completion[0]["content"]):
        return None

    return {
        "prompt": cleaned_prompt,
        "completion": cleaned_completion,
        "source": source,
    }


def load_synthetic_pairs(system_prompt: str, limit: int, seed: int) -> list[dict[str, object]]:
    dataset = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M", split="train")
    dataset = select_subset(dataset, limit, seed)

    rows: list[dict[str, object]] = []
    for item in dataset:
        example = make_example(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["input"]},
            ],
            completion=[{"role": "assistant", "content": item["output"]}],
            source="Abhishekcr448/Hinglish-Everyday-Conversations-1M",
        )
        if example:
            rows.append(example)
    return rows


def load_local_input_output_csv(
    file_path: Path,
    *,
    system_prompt: str,
    limit: int,
    seed: int,
    source_name: str,
) -> list[dict[str, object]]:
    dataset = load_dataset("csv", data_files=str(file_path), split="train")
    dataset = select_subset(dataset, limit, seed)

    rows: list[dict[str, object]] = []
    for item in dataset:
        example = make_example(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["input"]},
            ],
            completion=[{"role": "assistant", "content": item["output"]}],
            source=source_name,
        )
        if example:
            rows.append(example)
    return rows


def load_turn_pairs(system_prompt: str, limit: int, seed: int) -> list[dict[str, object]]:
    dataset = load_dataset("ankitdhiman/hinglish-conversations", "turn_pairs", split="train")
    dataset = select_subset(dataset, limit, seed)

    rows: list[dict[str, object]] = []
    for item in dataset:
        example = make_example(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["user_message"]},
            ],
            completion=[{"role": "assistant", "content": item["assistant_message"]}],
            source="ankitdhiman/hinglish-conversations:turn_pairs",
        )
        if example:
            rows.append(example)
    return rows


def messages_to_examples(
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
    source: str,
    max_context_messages: int,
) -> list[dict[str, object]]:
    history: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = normalize_text(message.get("content", ""))
        if role not in {"user", "assistant"} or not content:
            continue

        if role == "assistant" and history and history[-1]["role"] == "user":
            prompt = [{"role": "system", "content": system_prompt}] + history[-max_context_messages:]
            example = make_example(
                prompt=prompt,
                completion=[{"role": "assistant", "content": content}],
                source=source,
            )
            if example:
                rows.append(example)

        history.append({"role": role, "content": content})

    return rows


def load_chat_format(system_prompt: str, limit: int, seed: int, max_context_messages: int) -> list[dict[str, object]]:
    dataset = load_dataset("ankitdhiman/hinglish-conversations", "chat_format", split="train")
    dataset = select_subset(dataset, limit, seed)

    rows: list[dict[str, object]] = []
    for item in dataset:
        rows.extend(
            messages_to_examples(
                item["messages"],
                system_prompt=system_prompt,
                source="ankitdhiman/hinglish-conversations:chat_format",
                max_context_messages=max_context_messages,
            )
        )
    return rows


def load_romanized_pairs(limit: int, seed: int) -> list[dict[str, object]]:
    dataset = load_dataset("sk-community/romanized_hindi", split="train")
    dataset = select_subset(dataset, limit, seed)

    rows: list[dict[str, object]] = []
    for item in dataset:
        user_text = f"Is Hindi sentence ko natural Roman script me likho:\n{item['hi']}"
        example = make_example(
            prompt=[
                {"role": "system", "content": ROMANIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            completion=[{"role": "assistant", "content": item["hi_rom"]}],
            source="sk-community/romanized_hindi",
        )
        if example:
            rows.append(example)
    return rows


def load_custom_rows(
    custom_dir: Path,
    system_prompt: str,
    max_context_messages: int,
    *,
    ignored_csv_names: set[str] | None = None,
    seed: int = 42,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not custom_dir.exists():
        return rows

    ignored_csv_names = ignored_csv_names or set()

    for file_path in sorted(custom_dir.glob("*.csv")):
        if file_path.name in ignored_csv_names:
            continue

        with file_path.open("r", encoding="utf-8-sig") as handle:
            header = handle.readline().strip().lower()

        if header == "input,output":
            rows.extend(
                load_local_input_output_csv(
                    file_path,
                    system_prompt=system_prompt,
                    limit=10**12,
                    seed=seed,
                    source_name=f"custom:{file_path.name}",
                )
            )
            continue

    for file_path in sorted(custom_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)

                if "prompt" in item and "completion" in item:
                    example = make_example(
                        prompt=item["prompt"],
                        completion=item["completion"],
                        source=f"custom:{file_path.name}",
                    )
                    if example:
                        rows.append(example)
                    continue

                if "input" in item and "output" in item:
                    example = make_example(
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item["input"]},
                        ],
                        completion=[{"role": "assistant", "content": item["output"]}],
                        source=f"custom:{file_path.name}",
                    )
                    if example:
                        rows.append(example)
                    continue

                if "user" in item and "assistant" in item:
                    example = make_example(
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item["user"]},
                        ],
                        completion=[{"role": "assistant", "content": item["assistant"]}],
                        source=f"custom:{file_path.name}",
                    )
                    if example:
                        rows.append(example)
                    continue

                if "messages" in item:
                    rows.extend(
                        messages_to_examples(
                            item["messages"],
                            system_prompt=system_prompt,
                            source=f"custom:{file_path.name}",
                            max_context_messages=max_context_messages,
                        )
                    )
    return rows


def deduplicate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for row in rows:
        signature = dedupe_signature(row["prompt"], row["completion"])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def save_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    custom_dir = Path(args.custom_dir)
    local_synthetic_file = Path(args.local_synthetic_file)
    system_prompt = DEFAULT_SYSTEM_PROMPT

    all_rows: list[dict[str, object]] = []
    ignored_csv_names: set[str] = set()

    if local_synthetic_file.exists() and args.synthetic_max > 0:
        all_rows.extend(
            load_local_input_output_csv(
                local_synthetic_file,
                system_prompt=system_prompt,
                limit=args.synthetic_max,
                seed=args.seed,
                source_name=f"local:{local_synthetic_file.name}",
            )
        )
        ignored_csv_names.add(local_synthetic_file.name)
    else:
        all_rows.extend(load_synthetic_pairs(system_prompt, args.synthetic_max, args.seed))

    all_rows.extend(load_turn_pairs(system_prompt, args.turn_pairs_max, args.seed))
    all_rows.extend(load_chat_format(system_prompt, args.chat_format_max, args.seed, args.max_context_messages))
    all_rows.extend(load_romanized_pairs(args.romanized_max, args.seed))
    all_rows.extend(
        load_custom_rows(
            custom_dir,
            system_prompt,
            args.max_context_messages,
            ignored_csv_names=ignored_csv_names,
            seed=args.seed,
        )
    )

    source_counts_before = Counter(row["source"] for row in all_rows)
    all_rows = deduplicate_rows(all_rows)
    random.Random(args.seed).shuffle(all_rows)

    if len(all_rows) < 2:
        raise RuntimeError("Dataset bahut chhota hai. Source counts ya custom data check karo.")

    validation_size = max(1, int(len(all_rows) * args.validation_ratio))
    validation_size = min(validation_size, max(1, len(all_rows) // 10))
    if validation_size >= len(all_rows):
        validation_size = len(all_rows) - 1

    validation_rows = all_rows[:validation_size]
    train_rows = all_rows[validation_size:]

    save_jsonl(output_dir / "train.jsonl", train_rows)
    save_jsonl(output_dir / "validation.jsonl", validation_rows)

    metadata = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "total_rows_after_dedup": len(all_rows),
        "source_counts_before_dedup": dict(source_counts_before),
        "source_counts_after_dedup": dict(Counter(row["source"] for row in all_rows)),
        "notes": [
            "Default dataset is a mix of large synthetic Hinglish chat, smaller multi-turn Hinglish chats, and Romanized Hindi support data.",
            "If data/custom/hinglish_conversations.csv exists, it is preferred over downloading the main synthetic dataset from Hugging Face.",
            "Raise --synthetic-max if you want to use more of the 1M-row synthetic corpus.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
