# -*- coding: utf-8 -*-
"""Unit tests for trigger detection."""
import json
import pytest
from pathlib import Path

from src.triggers import find_triggers, write_triggers


class TestFindTriggers:
    def test_no_matches(self):
        result = find_triggers("hello world", ["cancel", "refund"])
        assert result["triggered"] is False
        assert result["total_hits"] == 0
        assert result["findings"] == []

    def test_single_match(self):
        result = find_triggers("I want to cancel my subscription", ["cancel"])
        assert result["triggered"] is True
        assert result["total_hits"] == 1
        assert result["counts"]["cancel"] == 1
        assert result["findings"][0]["term"] == "cancel"

    def test_multiple_matches_same_term(self):
        result = find_triggers("cancel this, cancel that", ["cancel"])
        assert result["total_hits"] == 2
        assert result["counts"]["cancel"] == 2

    def test_case_insensitive(self):
        result = find_triggers("I am ANGRY about this", ["angry"])
        assert result["total_hits"] == 1

    def test_snippet_context(self):
        text = "I am very unhappy and I want to cancel immediately"
        result = find_triggers(text, ["cancel"])
        finding = result["findings"][0]
        assert "cancel" in finding["snippet"]
        assert finding["start"] < finding["end"]

    def test_arabic_triggers(self):
        result = find_triggers("\u0623\u0631\u064A\u062F \u0625\u0644\u063A\u0627\u0621 \u0627\u0644\u0627\u0634\u062A\u0631\u0627\u0643", ["\u0625\u0644\u063A\u0627\u0621"])
        assert result["triggered"] is True
        assert result["total_hits"] == 1

    def test_empty_text(self):
        result = find_triggers("", ["cancel"])
        assert result["triggered"] is False

    def test_empty_terms(self):
        result = find_triggers("cancel refund angry", [])
        assert result["triggered"] is False
        assert result["total_hits"] == 0

    def test_findings_sorted_by_position(self):
        result = find_triggers("refund first then cancel later", ["cancel", "refund"])
        positions = [f["start"] for f in result["findings"]]
        assert positions == sorted(positions)


class TestWriteTriggers:
    def test_writes_json_file(self, tmp_path):
        output = tmp_path / "triggers.json"
        result = write_triggers("I want to cancel", ["cancel"], output)
        assert output.exists()
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["total_hits"] == 1
        assert result["total_hits"] == 1
