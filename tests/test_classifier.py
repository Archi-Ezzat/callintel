# -*- coding: utf-8 -*-
"""Unit tests for the classifier module."""
import pytest

from src.classifier import _normalize_label, _map_to_category, LABEL_ALIASES


class TestNormalizeLabel:
    def test_strips_whitespace(self):
        assert _normalize_label("  Violence  ") == "violence"

    def test_lowercased(self):
        assert _normalize_label("BULLYING") == "bullying"

    def test_spaces_to_underscores(self):
        assert _normalize_label("violence threat") == "violence_threat"

    def test_slashes_to_underscores(self):
        assert _normalize_label("weapons/terrorism") == "weapons_terrorism"

    def test_double_underscore_collapsed(self):
        assert _normalize_label("a__b") == "a_b"


class TestMapToCategory:
    def test_known_aliases(self):
        for alias, category in LABEL_ALIASES.items():
            result = _map_to_category(alias)
            assert result == category, f"{alias} -> expected {category}, got {result}"

    def test_safe_label(self):
        assert _map_to_category("safe") == "safe"
        assert _map_to_category("neutral") == "safe"
        assert _map_to_category("safe_tricky") == "safe"

    def test_unknown_label(self):
        result = _map_to_category("completely_unknown_label_xyz")
        assert result is None
