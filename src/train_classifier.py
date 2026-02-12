"""Fine-tune an Arabic text classifier for risk categories."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABELS = [
    "SAFE",
    "BULLYING",
    "SEXUAL",
    "VIOLENCE",
    "WEAPONS_TERRORISM",
    "DRUGS_CRIME",
    "POLITICS_INCITEMENT",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _map_label(label: str) -> int:
    normalized = label.strip().upper()
    mapping = {
        "SAFE": "SAFE",
        "SAFE_TRICKY": "SAFE",
        "BULLYING": "BULLYING",
        "INSULT": "BULLYING",
        "SEXUAL": "SEXUAL",
        "VIOLENCE": "VIOLENCE",
        "WEAPONS/TERRORISM": "WEAPONS_TERRORISM",
        "WEAPONS_TERRORISM": "WEAPONS_TERRORISM",
        "TERRORISM": "WEAPONS_TERRORISM",
        "DRUGS": "DRUGS_CRIME",
        "CRIME": "DRUGS_CRIME",
        "DRUGS_CRIME": "DRUGS_CRIME",
        "POLITICS": "POLITICS_INCITEMENT",
        "POLITICS_INCITEMENT": "POLITICS_INCITEMENT",
    }
    mapped = mapping.get(normalized, normalized)
    if mapped not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    return LABELS.index(mapped)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "training" / "train.jsonl"
    val_path = root / "data" / "training" / "val.jsonl"
    output_dir = root / "models" / "classifier"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path) if val_path.exists() else None

    def _prep(rows: list[dict]) -> list[dict]:
        processed = []
        for row in rows:
            processed.append(
                {
                    "text": row["text"],
                    "label": _map_label(row["label"]),
                }
            )
        return processed

    train_data = Dataset.from_list(_prep(train_rows))
    val_data = Dataset.from_list(_prep(val_rows)) if val_rows else None

    model_id = "UBC-NLP/MARBERTv2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_data = train_data.map(tokenize, batched=True)
    if val_data:
        val_data = val_data.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="steps" if val_data else "no",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
