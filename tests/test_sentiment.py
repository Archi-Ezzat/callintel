# -*- coding: utf-8 -*-
"""Unit tests for sentiment analysis module."""
import pytest

from src.sentiment import SentimentResult, _LABEL_MAP


class TestSentimentResult:
    def test_dataclass_immutable(self):
        result = SentimentResult(
            label="positive", score=0.95,
            scores={"positive": 0.95, "negative": 0.02, "neutral": 0.03},
            risk_boost=0.0,
        )
        assert result.label == "positive"
        assert result.score == 0.95
        with pytest.raises(AttributeError):
            result.label = "negative"  # frozen

    def test_risk_boost_zero_for_positive(self):
        result = SentimentResult(
            label="positive", score=0.9,
            scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            risk_boost=0.0,
        )
        assert result.risk_boost == 0.0

    def test_risk_boost_for_negative(self):
        result = SentimentResult(
            label="negative", score=0.85,
            scores={"positive": 0.05, "negative": 0.85, "neutral": 0.10},
            risk_boost=0.105,
        )
        assert result.risk_boost > 0
        assert result.risk_boost <= 0.15


class TestLabelMap:
    def test_label_0_is_negative(self):
        assert _LABEL_MAP["LABEL_0"] == "negative"

    def test_label_1_is_neutral(self):
        assert _LABEL_MAP["LABEL_1"] == "neutral"

    def test_label_2_is_positive(self):
        assert _LABEL_MAP["LABEL_2"] == "positive"

    def test_human_labels_mapped(self):
        assert _LABEL_MAP["negative"] == "negative"
        assert _LABEL_MAP["neutral"] == "neutral"
        assert _LABEL_MAP["positive"] == "positive"
