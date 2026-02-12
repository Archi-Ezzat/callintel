"""Rule-based risk scoring engine for Arabic content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RiskConfig:
    weights: dict[str, float]
    density_factor: float = 0.05


DEFAULT_WEIGHTS = {
    "violence_and_killing": 1.0,
    "weapons_and_war": 0.95,
    "sexual_content_and_harassment": 0.85,
    "crime_theft_drugs": 0.75,
    "treason_politics_extremism": 0.60,
    "bullying_and_insults": 0.35,
}


def _normalize_text(text: str) -> str:
    return text.lower()


def _word_pattern(word: str) -> re.Pattern[str]:
    # Use unicode-aware word boundaries. This is conservative for Arabic, but
    # still better than substring-only matching.
    escaped = re.escape(word)
    return re.compile(rf"(?<!\\w){escaped}(?!\\w)", re.IGNORECASE)


class RiskEvaluator:
    def __init__(self, dataset: dict[str, list[str]], config: RiskConfig | None = None) -> None:
        self.dataset = dataset
        self.config = config or RiskConfig(weights=DEFAULT_WEIGHTS)

    def analyze_text(self, text: str) -> dict:
        normalized = _normalize_text(text)
        detected: list[dict] = []
        categories_triggered: set[str] = set()
        counts_by_category: dict[str, int] = {key: 0 for key in self.dataset}

        max_severity = 0.0

        for category, words in self.dataset.items():
            weight = float(self.config.weights.get(category, 0.0))
            for word in words:
                pattern = _word_pattern(word)
                matches = list(pattern.finditer(normalized))
                if not matches:
                    continue
                counts_by_category[category] += len(matches)
                categories_triggered.add(category)
                for match in matches:
                    detected.append(
                        {
                            "word": word,
                            "category": category,
                            "weight": weight,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )
                if weight > max_severity:
                    max_severity = weight

        total_hits = len(detected)
        if total_hits == 0:
            return {
                "risk_percentage": "0.0%",
                "risk_score": 0.0,
                "risk_label": "SAFE (آمن)",
                "max_severity_category": None,
                "triggered_categories": [],
                "detected_words": [],
                "counts_by_category": counts_by_category,
            }

        accumulation_bonus = max(0.0, (total_hits - 1) * self.config.density_factor)
        total_risk = min(1.0, max_severity + accumulation_bonus)
        risk_percentage = round(total_risk * 100, 2)

        return {
            "risk_percentage": f"{risk_percentage}%",
            "risk_score": round(total_risk, 4),
            "risk_label": _risk_label(risk_percentage),
            "max_severity_category": _max_category(detected),
            "triggered_categories": sorted(categories_triggered),
            "detected_words": [d["word"] for d in detected],
            "detected_details": detected,
            "counts_by_category": counts_by_category,
        }


def _max_category(detected: Iterable[dict]) -> str | None:
    if not detected:
        return None
    return max(detected, key=lambda item: item["weight"])["category"]


def _risk_label(score: float) -> str:
    if score >= 90:
        return "CRITICAL (خطر حرج - إشعار أمني)"
    if score >= 75:
        return "HIGH (خطر مرتفع - حظر مؤقت)"
    if score >= 50:
        return "MODERATE (خطر متوسط - مراجعة يدوية)"
    if score >= 20:
        return "LOW (منخفض - تحذير)"
    return "SAFE (آمن)"
