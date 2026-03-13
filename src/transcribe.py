from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)


def _resolve_device(device_name: str) -> tuple[int, torch.dtype]:
    if device_name == "auto":
        if torch.cuda.is_available():
            return 0, torch.float16
        return -1, torch.float32
    if device_name == "cpu":
        return -1, torch.float32
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            return -1, torch.float32
        parts = device_name.split(":")
        index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        return index, torch.float16
    return -1, torch.float32


class WhisperTranscriber:
    def __init__(
        self,
        model_path: Path,
        batch_size: int = 8,
        language: str | None = None,
        device: str = "auto",
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Whisper model path not found: {model_path}")

        device_index, torch_dtype = _resolve_device(device)
        self.batch_size = batch_size
        self.language = language

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "local_files_only": True,
        }
        # Support fp32 variant shards produced by HF snapshot downloads.
        if (model_path / "model.safetensors.index.fp32.json").exists() or (
            model_path / "model.fp32-00001-of-00002.safetensors"
        ).exists():
            model_kwargs["variant"] = "fp32"
            model_kwargs["use_safetensors"] = True

        logger.info("Loading Whisper model from %s (device=%s)", model_path, device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_path), **model_kwargs)
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device_index,
            dtype=torch_dtype,
            batch_size=batch_size,
        )
        logger.info("Whisper model loaded successfully")

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"task": "transcribe"}
        if self.language:
            kwargs["language"] = self.language
        return kwargs

    def _parse_result(self, result: dict[str, Any], chunk_path: Path) -> dict[str, Any]:
        """Parse a single pipeline result into a transcript dict."""
        segments = []
        for segment in result.get("chunks", []):
            timestamp = segment.get("timestamp", (None, None))
            segments.append(
                {
                    "start": timestamp[0],
                    "end": timestamp[1],
                    "text": segment.get("text", "").strip(),
                }
            )
        return {
            "chunk_file": chunk_path.name,
            "text": result.get("text", "").strip(),
            "segments": segments,
        }

    def transcribe_chunk(self, chunk_path: Path) -> dict[str, Any]:
        result = self.pipe(
            str(chunk_path),
            return_timestamps=True,
            generate_kwargs=self._generate_kwargs(),
        )
        return self._parse_result(result, chunk_path)

    def transcribe_chunks(
        self,
        chunk_manifest: list[dict[str, Any]],
        transcripts_dir: Path,
        force: bool = False,
    ) -> list[Path]:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        # Separate already-done chunks from to-do chunks
        to_process: list[dict[str, Any]] = []
        for item in chunk_manifest:
            transcript_path = transcripts_dir / f"chunk_{item['index']:03d}.json"
            if transcript_path.exists() and not force:
                created.append(transcript_path)
            else:
                to_process.append(item)

        if not to_process:
            logger.info("All %d chunks already transcribed, skipping", len(chunk_manifest))
            return created

        logger.info(
            "Transcribing %d chunks (batch_size=%d, %d cached)",
            len(to_process),
            self.batch_size,
            len(created),
        )

        # Use batched pipeline for better GPU utilization (E-2)
        paths = [item["path"] for item in to_process]
        results = self.pipe(
            paths,
            return_timestamps=True,
            generate_kwargs=self._generate_kwargs(),
            batch_size=self.batch_size,
        )

        for item, result in zip(to_process, results):
            chunk_path = Path(item["path"])
            transcript = self._parse_result(result, chunk_path)
            transcript["chunk_index"] = item["index"]
            transcript["chunk_start_sec"] = item.get("start_sec")
            transcript["chunk_end_sec"] = item.get("end_sec")

            transcript_path = transcripts_dir / f"chunk_{item['index']:03d}.json"
            with transcript_path.open("w", encoding="utf-8") as fh:
                json.dump(transcript, fh, ensure_ascii=False, indent=2)
            created.append(transcript_path)

        logger.info("Transcription complete: %d chunks processed", len(to_process))
        return created
