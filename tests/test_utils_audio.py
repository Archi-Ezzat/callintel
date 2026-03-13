# -*- coding: utf-8 -*-
"""Unit tests for audio utilities."""
import numpy as np
import pytest
from pathlib import Path

from src.utils_audio import slugify, validate_audio


class TestSlugify:
    def test_english_alphanumeric(self):
        assert slugify("my_test_file") == "my_test_file"

    def test_spaces_to_underscores(self):
        assert slugify("hello world") == "hello_world"

    def test_special_chars_stripped(self):
        assert slugify("file@name#123") == "file_name_123"

    def test_arabic_uses_hash_fallback(self):
        result = slugify("\u0645\u0644\u0641_\u0635\u0648\u062A\u064A")
        assert result.startswith("call_")
        assert len(result) == 13  # call_ + 8 hex chars

    def test_arabic_deterministic(self):
        a = slugify("\u0645\u0644\u0641_\u0635\u0648\u062A\u064A")
        b = slugify("\u0645\u0644\u0641_\u0635\u0648\u062A\u064A")
        assert a == b

    def test_mixed_keeps_ascii(self):
        result = slugify("test_123")
        assert result == "test_123"

    def test_empty_string(self):
        result = slugify("")
        assert result.startswith("call_")

    def test_all_special_characters(self):
        result = slugify("@#$%^&*()")
        assert result.startswith("call_")


class TestValidateAudio:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_audio(tmp_path / "doesnt_exist.wav")

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            validate_audio(f)

    def test_garbage_file_raises(self, tmp_path):
        f = tmp_path / "garbage.wav"
        f.write_bytes(b"this is not audio content at all " * 100)
        with pytest.raises(ValueError, match="Cannot read"):
            validate_audio(f)
