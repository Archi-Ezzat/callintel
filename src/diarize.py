"""Speaker diarization using pyannote.audio.

Identifies who spoke when in an audio file, producing timestamped
speaker segments that can be aligned with the transcript.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerSegment:
    """A single speaker turn."""

    speaker: str
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class DiarizationResult:
    """Full diarization output."""

    segments: list[SpeakerSegment]
    num_speakers: int
    speaker_durations: dict[str, float]  # speaker -> total seconds


class SpeakerDiarizer:
    """Speaker diarization via pyannote.audio pipeline."""

    def __init__(
        self,
        hf_token: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
    ) -> None:
        import torch
        from pyannote.audio import Pipeline

        use_device = "cpu"
        if device == "auto" and torch.cuda.is_available():
            use_device = "cuda"
        elif device.startswith("cuda") and torch.cuda.is_available():
            use_device = device

        logger.info("Loading diarization pipeline: %s (device=%s)", model_name, use_device)
        self.pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        if use_device != "cpu":
            import torch
            self.pipeline.to(torch.device(use_device))
        logger.info("Diarization pipeline loaded")

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Run diarization on an audio file.

        Parameters
        ----------
        audio_path : Path
            Path to a WAV file (16kHz mono recommended).
        min_speakers / max_speakers : int, optional
            Constrain the number of speakers if known.
        """
        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        logger.info("Running diarization on %s", audio_path.name)
        diarization = self.pipeline(str(audio_path), **kwargs)

        segments: list[SpeakerSegment] = []
        speaker_durations: dict[str, float] = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = round(turn.end - turn.start, 3)
            seg = SpeakerSegment(
                speaker=speaker,
                start_sec=round(turn.start, 3),
                end_sec=round(turn.end, 3),
                duration_sec=duration,
            )
            segments.append(seg)
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

        # Round durations
        speaker_durations = {k: round(v, 3) for k, v in speaker_durations.items()}

        result = DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_durations),
            speaker_durations=speaker_durations,
        )
        logger.info(
            "Diarization complete: %d segments, %d speakers",
            len(segments),
            result.num_speakers,
        )
        return result


def align_transcript_with_speakers(
    transcript_segments: list[dict],
    diarization_segments: list[SpeakerSegment],
) -> list[dict]:
    """Align transcript segments with speaker labels.

    For each transcript segment, find the diarization segment with the
    most overlap and assign that speaker label.
    """
    if not diarization_segments:
        return transcript_segments

    result = []
    for tseg in transcript_segments:
        t_start = tseg.get("start_sec")
        t_end = tseg.get("end_sec")
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        if t_start is not None and t_end is not None:
            for dseg in diarization_segments:
                overlap_start = max(t_start, dseg.start_sec)
                overlap_end = min(t_end, dseg.end_sec)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg.speaker

        aligned = dict(tseg)
        aligned["speaker"] = best_speaker
        result.append(aligned)

    return result


def write_diarization_output(
    result: DiarizationResult,
    output_dir: Path,
) -> Path:
    """Write diarization results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_speakers": result.num_speakers,
        "speaker_durations": result.speaker_durations,
        "segments": [
            {
                "speaker": s.speaker,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "duration_sec": s.duration_sec,
            }
            for s in result.segments
        ],
    }
    path = output_dir / "diarization.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Diarization saved to %s", path)
    return path
