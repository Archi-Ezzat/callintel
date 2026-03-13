"""Rule-based risk scoring engine for Arabic content."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


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


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify hamza/alef, remove tatweel."""
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)
    # Remove tatweel (kashida)
    text = text.replace('\u0640', '')
    # Normalize hamza variants to bare alef
    text = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)
    # Normalize alef maksura to ya
    text = text.replace('\u0649', '\u064A')
    # Normalize taa marbouta to haa
    text = text.replace('\u0629', '\u0647')
    return text.lower()


def _build_category_pattern(words: list[str]) -> re.Pattern[str]:
    """Compile a single alternation regex for all words in a category."""
    normalized_words = [_normalize_arabic(w) for w in words]
    escaped = [re.escape(w) for w in normalized_words]
    alternation = "|".join(escaped)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


class RiskEvaluator:
    def __init__(self, dataset: dict[str, list[str]], config: RiskConfig | None = None) -> None:
        self.dataset = dataset
        self.config = config or RiskConfig(weights=DEFAULT_WEIGHTS)
        # Pre-compile one regex per category for performance
        self._patterns: dict[str, re.Pattern[str]] = {}
        self._word_map: dict[str, dict[str, str]] = {}  # category -> {normalized: original}
        for category, words in dataset.items():
            self._patterns[category] = _build_category_pattern(words)
            self._word_map[category] = {
                _normalize_arabic(w): w for w in words
            }
        logger.debug("RiskEvaluator initialized with %d categories", len(dataset))

    def analyze_text(self, text: str) -> dict:
        normalized = _normalize_arabic(text)
        detected: list[dict] = []
        categories_triggered: set[str] = set()
        counts_by_category: dict[str, int] = {key: 0 for key in self.dataset}

        max_severity = 0.0

        for category in self.dataset:
            weight = float(self.config.weights.get(category, 0.0))
            pattern = self._patterns[category]
            matches = list(pattern.finditer(normalized))
            if not matches:
                continue
            counts_by_category[category] += len(matches)
            categories_triggered.add(category)
            for match in matches:
                matched_text = match.group()
                original_word = self._word_map[category].get(matched_text, matched_text)
                detected.append(
                    {
                        "word": original_word,
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
