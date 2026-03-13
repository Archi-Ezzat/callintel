"""Arabic sentiment analysis using CAMeLBERT.

Wraps the ``CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment`` model
to provide per-text sentiment scores (positive / negative / neutral).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Label mapping for the CAMeLBERT sentiment model
_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


@dataclass(frozen=True)
class SentimentResult:
    """Result from sentiment analysis."""

    label: str  # positive | negative | neutral
    score: float  # confidence of the top label
    scores: dict[str, float]  # per-class scores
    risk_boost: float  # risk penalty derived from sentiment (0.0 – 0.15)


class SentimentAnalyzer:
    """Sentiment analyzer backed by a HuggingFace classification model."""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Sentiment model not found: {model_path}")

        device_index = 0 if (device == "auto" and torch.cuda.is_available()) else -1
        torch_dtype = torch.float16 if device_index >= 0 else torch.float32

        logger.info("Loading sentiment model from %s (device=%s)", model_path, device)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch_dtype,
        )

        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
            top_k=None,  # return all class scores
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment model loaded successfully")

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of the given text."""
        if not text or not text.strip():
            return SentimentResult(
                label="neutral",
                score=1.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                risk_boost=0.0,
            )

        # Truncate very long texts to avoid memory issues
        truncated = text[:2000]
        raw_results = self.pipe(truncated)

        # Parse pipeline output — top_k=None returns list of dicts per input
        if isinstance(raw_results, list) and raw_results:
            if isinstance(raw_results[0], list):
                raw_results = raw_results[0]

        scores: dict[str, float] = {}
        for item in raw_results:
            raw_label = item.get("label", "")
            mapped = _LABEL_MAP.get(raw_label, raw_label.lower())
            scores[mapped] = round(float(item.get("score", 0.0)), 4)

        # Determine top label
        top_label = max(scores, key=scores.get) if scores else "neutral"
        top_score = scores.get(top_label, 0.0)

        # Calculate risk boost from negative sentiment
        negative_score = scores.get("negative", 0.0)
        risk_boost = 0.0
        if negative_score > 0.5:
            # Scale: 0.5–1.0 negative score maps to 0.0–0.15 risk boost
            risk_boost = round(min(0.15, (negative_score - 0.5) * 0.3), 4)

        return SentimentResult(
            label=top_label,
            score=round(top_score, 4),
            scores=scores,
            risk_boost=risk_boost,
        )
