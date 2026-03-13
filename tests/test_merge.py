# -*- coding: utf-8 -*-
"""Unit tests for transcript merging."""
import json
import pytest
from pathlib import Path

from src.merge import merge_transcripts, _remove_overlap_prefix


class TestRemoveOverlapPrefix:
    def test_removes_repeated_suffix(self):
        previous = "hello world this is a test"
        current = "is a test and more content"
        result = _remove_overlap_prefix(current, previous)
        assert "and more content" in result
        assert not result.startswith("is a test")

    def test_no_overlap(self):
        previous = "hello world"
        current = "completely different text"
        result = _remove_overlap_prefix(current, previous)
        assert result == "completely different text"

    def test_empty_previous(self):
        result = _remove_overlap_prefix("some text", "")
        assert result == "some text"

    def test_empty_current(self):
        result = _remove_overlap_prefix("", "some text")
        assert result == ""


class TestMergeTranscripts:
    def _write_chunk(self, directory: Path, index: int, text: str,
                     start_sec: float = 0.0, end_sec: float = 30.0,
                     segments: list | None = None):
        payload = {
            "chunk_file": f"chunk_{index:03d}.wav",
            "text": text,
            "chunk_index": index,
            "chunk_start_sec": start_sec,
            "chunk_end_sec": end_sec,
            "segments": segments or [],
        }
        path = directory / f"chunk_{index:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_single_chunk(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        merged = tmp_path / "merged"
        transcripts.mkdir()

        self._write_chunk(transcripts, 0, "hello world")
        result = merge_transcripts(transcripts, merged)
        assert result["full_text"] == "hello world"

    def test_multiple_chunks_concatenated(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        merged = tmp_path / "merged"
        transcripts.mkdir()

        self._write_chunk(transcripts, 0, "first part", start_sec=0, end_sec=30)
        self._write_chunk(transcripts, 1, "second part", start_sec=28, end_sec=58)
        result = merge_transcripts(transcripts, merged, overlap_seconds=2.0)
        assert "first part" in result["full_text"]
        assert "second part" in result["full_text"]

    def test_output_files_created(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        merged = tmp_path / "merged"
        transcripts.mkdir()

        self._write_chunk(transcripts, 0, "test content")
        result = merge_transcripts(transcripts, merged)
        assert Path(result["txt_path"]).exists()
        assert Path(result["json_path"]).exists()

    def test_empty_chunks_handled(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        merged = tmp_path / "merged"
        transcripts.mkdir()

        self._write_chunk(transcripts, 0, "")
        result = merge_transcripts(transcripts, merged)
        assert result["full_text"] == ""
