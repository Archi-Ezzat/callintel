# -*- coding: utf-8 -*-
"""Unit tests for configuration loading."""
import os
import pytest
from pathlib import Path

from src.config import AppConfig, _split_csv, _env_int


class TestSplitCsv:
    def test_normal_values(self):
        assert _split_csv("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert _split_csv(" a , b , c ") == ["a", "b", "c"]

    def test_empty_string(self):
        assert _split_csv("") == []

    def test_none(self):
        assert _split_csv(None) == []

    def test_trailing_comma(self):
        assert _split_csv("a,b,") == ["a", "b"]


class TestEnvInt:
    def test_valid_integer(self):
        os.environ["TEST_INT"] = "42"
        assert _env_int("TEST_INT", 0) == 42
        del os.environ["TEST_INT"]

    def test_default_value(self):
        assert _env_int("NONEXISTENT_KEY_XYZ", 99) == 99

    def test_invalid_raises(self):
        os.environ["TEST_INT"] = "abc"
        with pytest.raises(ValueError, match="Invalid integer"):
            _env_int("TEST_INT", 0)
        del os.environ["TEST_INT"]


class TestAppConfig:
    def test_from_env_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WHISPER_MODEL_PATH", raising=False)
        monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
        monkeypatch.delenv("TRIGGER_TERMS", raising=False)
        monkeypatch.delenv("LANGUAGE", raising=False)

        config = AppConfig.from_env(root_dir=tmp_path)
        assert config.sample_rate == 16000
        assert config.chunk_seconds == 30
        assert config.device in ("auto", "cpu", "cuda")
        assert len(config.trigger_terms) > 0

    def test_arabic_default_triggers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LANGUAGE", "ar")
        monkeypatch.delenv("TRIGGER_TERMS", raising=False)
        config = AppConfig.from_env(root_dir=tmp_path)
        # Should contain Arabic triggers, not English
        assert any("\u0627" <= c <= "\u064A" for term in config.trigger_terms for c in term)

    def test_custom_trigger_terms(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIGGER_TERMS", "foo,bar,baz")
        config = AppConfig.from_env(root_dir=tmp_path)
        assert config.trigger_terms == ["foo", "bar", "baz"]
