# -*- coding: utf-8 -*-
"""Unit tests for speaker diarization module."""
import pytest

from src.diarize import (
    SpeakerSegment,
    DiarizationResult,
    align_transcript_with_speakers,
    write_diarization_output,
)


class TestSpeakerSegment:
    def test_immutable(self):
        seg = SpeakerSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=5.0, duration_sec=5.0)
        assert seg.speaker == "SPEAKER_00"
        with pytest.raises(AttributeError):
            seg.speaker = "SPEAKER_01"


class TestDiarizationResult:
    def test_empty(self):
        result = DiarizationResult(segments=[], num_speakers=0, speaker_durations={})
        assert result.num_speakers == 0

    def test_with_segments(self):
        segs = [
            SpeakerSegment("A", 0.0, 5.0, 5.0),
            SpeakerSegment("B", 5.0, 10.0, 5.0),
        ]
        result = DiarizationResult(segments=segs, num_speakers=2, speaker_durations={"A": 5.0, "B": 5.0})
        assert result.num_speakers == 2
        assert len(result.segments) == 2


class TestAlignTranscript:
    def test_basic_alignment(self):
        transcript = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": "Hello"},
            {"start_sec": 5.0, "end_sec": 10.0, "text": "How are you"},
        ]
        diarization = [
            SpeakerSegment("SPEAKER_00", 0.0, 6.0, 6.0),
            SpeakerSegment("SPEAKER_01", 6.0, 12.0, 6.0),
        ]
        aligned = align_transcript_with_speakers(transcript, diarization)
        assert aligned[0]["speaker"] == "SPEAKER_00"
        assert aligned[1]["speaker"] == "SPEAKER_01"

    def test_empty_diarization(self):
        transcript = [{"start_sec": 0.0, "end_sec": 5.0, "text": "Hello"}]
        aligned = align_transcript_with_speakers(transcript, [])
        assert aligned == transcript

    def test_no_timestamps(self):
        transcript = [{"text": "Hello"}]
        diarization = [SpeakerSegment("SPEAKER_00", 0.0, 5.0, 5.0)]
        aligned = align_transcript_with_speakers(transcript, diarization)
        assert aligned[0]["speaker"] == "UNKNOWN"

    def test_overlap_resolution(self):
        """When a transcript segment overlaps two speakers, assign the one with more overlap."""
        transcript = [{"start_sec": 3.0, "end_sec": 8.0, "text": "test"}]
        diarization = [
            SpeakerSegment("A", 0.0, 4.0, 4.0),   # 1s overlap (3-4)
            SpeakerSegment("B", 4.0, 10.0, 6.0),   # 4s overlap (4-8)
        ]
        aligned = align_transcript_with_speakers(transcript, diarization)
        assert aligned[0]["speaker"] == "B"


class TestWriteOutput:
    def test_writes_json(self, tmp_path):
        result = DiarizationResult(
            segments=[SpeakerSegment("A", 0.0, 5.0, 5.0)],
            num_speakers=1,
            speaker_durations={"A": 5.0},
        )
        path = write_diarization_output(result, tmp_path)
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["num_speakers"] == 1
        assert len(data["segments"]) == 1
