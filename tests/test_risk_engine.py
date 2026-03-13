# -*- coding: utf-8 -*-
"""Unit tests for the risk engine."""
import pytest
from src.risk_engine import (
    RiskEvaluator,
    RiskConfig,
    DEFAULT_WEIGHTS,
    _normalize_arabic,
    _build_category_pattern,
    _risk_label,
    _max_category,
)


# ── Arabic normalisation ──────────────────────────────────────────────


class TestNormalizeArabic:
    def test_diacritics_removed(self):
        assert _normalize_arabic("\u0642\u064E\u062A\u0652\u0644\u064C") == _normalize_arabic("\u0642\u062A\u0644")

    def test_hamza_variants_unified(self):
        assert _normalize_arabic("\u0625\u0633\u0644\u0627\u0645") == _normalize_arabic("\u0627\u0633\u0644\u0627\u0645")
        assert _normalize_arabic("\u0623\u062D\u0645\u062F") == _normalize_arabic("\u0627\u062D\u0645\u062F")

    def test_taa_marbouta_to_haa(self):
        assert _normalize_arabic("\u0645\u062F\u0631\u0633\u0629") == _normalize_arabic("\u0645\u062F\u0631\u0633\u0647")

    def test_alef_maksura_to_ya(self):
        assert _normalize_arabic("\u0639\u0644\u0649") == _normalize_arabic("\u0639\u0644\u064A")

    def test_tatweel_removed(self):
        assert _normalize_arabic("\u0642\u0640\u0640\u062A\u0644") == _normalize_arabic("\u0642\u062A\u0644")

    def test_lowercased(self):
        assert _normalize_arabic("ABC") == "abc"


# ── Word boundary pattern ─────────────────────────────────────────────


class TestBuildCategoryPattern:
    def test_standalone_match(self):
        pattern = _build_category_pattern(["\u0642\u062A\u0644"])
        normalized = _normalize_arabic("\u0647\u0630\u0627 \u0642\u062A\u0644 \u062E\u0637\u064A\u0631")
        assert pattern.search(normalized) is not None

    def test_multiple_words(self):
        pattern = _build_category_pattern(["\u0642\u062A\u0644", "\u0630\u0628\u062D"])
        text = _normalize_arabic("\u0647\u0630\u0627 \u0630\u0628\u062D")
        assert pattern.search(text) is not None

    def test_empty_list(self):
        pattern = _build_category_pattern([])
        assert pattern.search("anything") is None


# ── Risk labels ───────────────────────────────────────────────────────


class TestRiskLabel:
    def test_safe(self):
        assert "SAFE" in _risk_label(0)
        assert "SAFE" in _risk_label(19.9)

    def test_low(self):
        assert "LOW" in _risk_label(20)
        assert "LOW" in _risk_label(49.9)

    def test_moderate(self):
        assert "MODERATE" in _risk_label(50)
        assert "MODERATE" in _risk_label(74.9)

    def test_high(self):
        assert "HIGH" in _risk_label(75)
        assert "HIGH" in _risk_label(89.9)

    def test_critical(self):
        assert "CRITICAL" in _risk_label(90)
        assert "CRITICAL" in _risk_label(100)


# ── Max category helper ──────────────────────────────────────────────


class TestMaxCategory:
    def test_returns_highest_weight(self):
        items = [
            {"category": "low", "weight": 0.3},
            {"category": "high", "weight": 0.9},
        ]
        assert _max_category(items) == "high"

    def test_empty_returns_none(self):
        assert _max_category([]) is None


# ── RiskEvaluator integration ────────────────────────────────────────


SMALL_DATASET = {
    "violence_and_killing": ["\u0642\u062A\u0644", "\u0630\u0628\u062D"],
    "bullying_and_insults": ["\u063A\u0628\u064A"],
}


class TestRiskEvaluator:
    def setup_method(self):
        self.evaluator = RiskEvaluator(SMALL_DATASET)

    def test_clean_text_returns_safe(self):
        result = self.evaluator.analyze_text("\u0645\u0631\u062D\u0628\u0627 \u0643\u064A\u0641 \u062D\u0627\u0644\u0643")
        assert result["risk_score"] == 0.0
        assert "SAFE" in result["risk_label"]
        assert result["detected_words"] == []

    def test_single_hit(self):
        result = self.evaluator.analyze_text("\u0647\u0630\u0627 \u0642\u062A\u0644 \u062E\u0637\u064A\u0631")
        assert result["risk_score"] > 0
        assert "\u0642\u062A\u0644" in result["detected_words"]
        assert "violence_and_killing" in result["triggered_categories"]

    def test_multiple_hits_increase_score(self):
        single = self.evaluator.analyze_text("\u0647\u0630\u0627 \u0642\u062A\u0644")
        double = self.evaluator.analyze_text("\u0647\u0630\u0627 \u0642\u062A\u0644 \u0648 \u0630\u0628\u062D")
        assert double["risk_score"] >= single["risk_score"]

    def test_score_capped_at_one(self):
        text = " ".join(["\u0642\u062A\u0644"] * 100)
        result = self.evaluator.analyze_text(text)
        assert result["risk_score"] <= 1.0

    def test_custom_weights(self):
        config = RiskConfig(weights={"violence_and_killing": 0.5, "bullying_and_insults": 0.1})
        evaluator = RiskEvaluator(SMALL_DATASET, config=config)
        result = evaluator.analyze_text("\u0647\u0630\u0627 \u0642\u062A\u0644")
        assert result["risk_score"] == 0.5

    def test_diacritized_text_still_matches(self):
        result = self.evaluator.analyze_text("\u0647\u0630\u0627 \u0642\u064E\u062A\u0652\u0644\u064C \u062E\u0637\u064A\u0631")
        assert len(result["detected_words"]) > 0
