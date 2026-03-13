# -*- coding: utf-8 -*-
"""Unit tests for the LLM evaluation module."""
import pytest

from src.evaluate_llm import (
    _heuristic_summary,
    _build_prompt,
    format_llm_report_text,
    _extract_chat_text,
)


class TestHeuristicSummary:
    def test_no_triggers(self):
        result = _heuristic_summary("hello world foo bar", {"total_hits": 0, "counts": {}})
        assert result["mode"] == "heuristic"
        assert result["scores"]["risk_score"] == 15

    def test_triggers_increase_risk(self):
        result = _heuristic_summary("text", {"total_hits": 5, "counts": {"cancel": 5}})
        assert result["scores"]["risk_score"] > 15

    def test_risk_capped_at_100(self):
        result = _heuristic_summary("text", {"total_hits": 100, "counts": {}})
        assert result["scores"]["risk_score"] <= 100

    def test_arabic_output(self):
        result = _heuristic_summary("text words here", {"total_hits": 0, "counts": {}}, output_language="ar")
        # Summary should be in Arabic
        assert any("\u0627" <= c <= "\u064A" for c in result["summary"])

    def test_long_text_clarity(self):
        text = " ".join(["word"] * 50)
        result = _heuristic_summary(text, {"total_hits": 0, "counts": {}})
        assert result["scores"]["clarity_score"] == 80

    def test_short_text_clarity(self):
        result = _heuristic_summary("short text", {"total_hits": 0, "counts": {}})
        assert result["scores"]["clarity_score"] == 55


class TestBuildPrompt:
    def test_english_prompt(self):
        prompt = _build_prompt("my transcript", {"counts": {}}, "en")
        assert "Analyze" in prompt
        assert "my transcript" in prompt

    def test_arabic_prompt(self):
        prompt = _build_prompt("my transcript", {"counts": {}}, "ar")
        assert "Arabic" in prompt

    def test_truncates_long_text(self):
        long_text = "x" * 50000
        prompt = _build_prompt(long_text, {"counts": {}}, "en")
        assert len(prompt) < 50000

    def test_contains_real_newlines(self):
        prompt = _build_prompt("text", {"counts": {}}, "en")
        assert "\n" in prompt
        assert "\\n" not in prompt  # No escaped newlines


class TestFormatLlmReportText:
    def test_basic_format(self):
        report = {
            "summary": "Test summary",
            "key_points": ["point 1", "point 2"],
            "scores": {"risk_score": 50},
        }
        text = format_llm_report_text(report)
        assert "Test summary" in text
        assert "point 1" in text
        assert "risk_score" in text

    def test_contains_real_newlines(self):
        report = {"summary": "Test", "key_points": [], "scores": {}}
        text = format_llm_report_text(report)
        assert "\n" in text
        assert "\\n" not in text  # No escaped newlines

    def test_empty_report(self):
        report = {"summary": "", "key_points": None, "scores": None}
        text = format_llm_report_text(report)
        assert "No summary generated" in text


class TestExtractChatText:
    def test_normal_response(self):
        response = {"choices": [{"message": {"content": "Hello world"}}]}
        assert _extract_chat_text(response) == "Hello world"

    def test_empty_choices(self):
        assert _extract_chat_text({"choices": []}) == ""

    def test_no_choices_key(self):
        assert _extract_chat_text({}) == ""

    def test_list_content(self):
        response = {"choices": [{"message": {"content": [{"text": "part1"}, {"text": "part2"}]}}]}
        result = _extract_chat_text(response)
        assert "part1" in result
        assert "part2" in result
