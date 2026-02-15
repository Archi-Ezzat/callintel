from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import librosa
import soundfile as sf


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "call"


def copy_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def normalize_audio(input_file: Path, output_wav: Path, sample_rate: int = 16000) -> dict:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(str(input_file), sr=sample_rate, mono=True)
    sf.write(str(output_wav), audio, sample_rate)
    duration_sec = len(audio) / sample_rate if sample_rate else 0
    return {
        "path": str(output_wav),
        "sample_rate": sample_rate,
        "samples": int(len(audio)),
        "duration_sec": round(float(duration_sec), 3),
    }


def chunk_audio(
    normalized_wav: Path,
    chunks_dir: Path,
    chunk_seconds: int = 30,
    overlap_seconds: int = 2,
    sample_rate: int = 16000,
) -> list[dict]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(str(normalized_wav), sr=sample_rate, mono=True)

    chunk_size = int(chunk_seconds * sample_rate)
    step_size = int((chunk_seconds - overlap_seconds) * sample_rate)
    total_samples = len(audio)

    chunks: list[dict] = []
    idx = 0
    start = 0

    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        clip = audio[start:end]
        if len(clip) == 0:
            break

        chunk_path = chunks_dir / f"chunk_{idx:03d}.wav"
        sf.write(str(chunk_path), clip, sample_rate)

        chunks.append(
            {
                "index": idx,
                "path": str(chunk_path),
                "start_sec": round(start / sample_rate, 3),
                "end_sec": round(end / sample_rate, 3),
                "duration_sec": round((end - start) / sample_rate, 3),
            }
        )

        idx += 1
        start += step_size

    return chunks

