from __future__ import annotations

import hashlib
import logging
import re
import shutil
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Create a filesystem-safe slug from a filename.

    Handles Arabic filenames by falling back to a hash-based name.
    """
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    if slug:
        return slug
    # Fallback for non-ASCII filenames (e.g. Arabic)
    short_hash = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return f"call_{short_hash}"


def copy_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def normalize_audio(
    input_file: Path,
    output_wav: Path,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, dict]:
    """Normalize audio to mono WAV at the given sample rate.

    Returns the loaded numpy array alongside metadata so callers can
    avoid a redundant disk read.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(str(input_file), sr=sample_rate, mono=True)
    sf.write(str(output_wav), audio, sample_rate)
    duration_sec = len(audio) / sample_rate if sample_rate else 0
    logger.info(
        "Normalized %s -> %s (%.1fs, %d samples)",
        input_file.name,
        output_wav.name,
        duration_sec,
        len(audio),
    )
    metadata = {
        "path": str(output_wav),
        "sample_rate": sample_rate,
        "samples": int(len(audio)),
        "duration_sec": round(float(duration_sec), 3),
    }
    return audio, metadata


def chunk_audio(
    normalized_wav: Path,
    chunks_dir: Path,
    chunk_seconds: int = 30,
    overlap_seconds: int = 2,
    sample_rate: int = 16000,
    audio_array: np.ndarray | None = None,
) -> list[dict]:
    """Split audio into overlapping chunks.

    If ``audio_array`` is provided, the data is used directly instead of
    re-reading ``normalized_wav`` from disk (performance optimisation).
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    chunks_dir.mkdir(parents=True, exist_ok=True)

    if audio_array is not None:
        audio = audio_array
    else:
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

    logger.info("Created %d chunks from %.1fs audio", len(chunks), total_samples / sample_rate)
    return chunks


def validate_audio(path: Path) -> None:
    """Validate that a file looks like a readable audio file."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {path}")
    if size < 1024:
        logger.warning("Suspiciously small audio file: %s (%d bytes)", path, size)
    try:
        sf.info(str(path))
    except Exception:
        # sf.info may not support all formats (e.g. mp3); librosa handles them.
        # Only fail on truly unreadable files.
        try:
            audio, _ = librosa.load(str(path), sr=None, duration=1.0)
            if len(audio) == 0:
                raise ValueError(f"Audio file contains no samples: {path}")
        except Exception as exc:
            raise ValueError(f"Cannot read audio file {path}: {exc}") from exc
