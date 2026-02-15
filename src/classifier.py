"""Text classifier wrapper for Arabic risk categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .risk_engine import DEFAULT_WEIGHTS


LABEL_ALIASES = {
    "violence": "violence_and_killing",
    "violent": "violence_and_killing",
    "violence_threat": "violence_and_killing",
    "weapons": "weapons_and_war",
    "weaponsterrorism": "weapons_and_war",
    "weapons_terrorism": "weapons_and_war",
    "terrorism": "weapons_and_war",
    "drugs": "crime_theft_drugs",
    "crime": "crime_theft_drugs",
    "drugs_crime": "crime_theft_drugs",
    "sexual": "sexual_content_and_harassment",
    "harassment": "sexual_content_and_harassment",
    "bullying": "bullying_and_insults",
    "insult": "bullying_and_insults",
    "safe": "safe",
    "safe_tricky": "safe",
    "neutral": "safe",
}


@dataclass(frozen=True)
class ClassifierResult:
    top_label: str
    top_score: float
    category: str | None
    category_score: float
    category_scores: dict[str, float]
    raw_scores: list[dict]
    risk_score: float


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace(" ", "_").replace("-", "_")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("__", "_")
    return cleaned


def _map_to_category(label: str) -> str | None:
    normalized = _normalize_label(label)
    return LABEL_ALIASES.get(normalized, normalized if normalized in DEFAULT_WEIGHTS else None)


class TextClassifier:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier model path not found: {model_path}")

        device_index = 0 if (device == "auto" and torch.cuda.is_available()) else -1
        torch_dtype = torch.float16 if device_index >= 0 else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch_dtype if device_index >= 0 else None,
        )

        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
            top_k=None,
            dtype=torch_dtype if device_index >= 0 else None,
        )

    def predict(self, text: str) -> ClassifierResult:
        raw = self.pipe(text)
        scores: list[dict] = []
        if isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, list):
                scores = first
            elif isinstance(first, dict):
                # Transformers may return either [dict] (top-1) or [[dict, ...]] (all labels).
                scores = [first]

        top = max(scores, key=lambda item: item["score"]) if scores else {"label": "SAFE", "score": 0.0}
        top_label = top["label"]
        top_score = float(top["score"])

        category_scores: dict[str, float] = {}
        for item in scores:
            category = _map_to_category(item["label"])
            if category is None:
                continue
            category_scores[category] = max(category_scores.get(category, 0.0), float(item["score"]))

        category = _map_to_category(top_label)
        category_score = category_scores.get(category, top_score) if category else top_score

        risk_score = 0.0
        if category and category != "safe":
            weight = float(DEFAULT_WEIGHTS.get(category, 0.0))
            risk_score = round(weight * category_score, 4)

        return ClassifierResult(
            top_label=top_label,
            top_score=top_score,
            category=category,
            category_score=category_score,
            category_scores=category_scores,
            raw_scores=scores,
            risk_score=risk_score,
        )
