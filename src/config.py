from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    input_calls_dir: Path
    work_dir: Path
    output_dir: Path
    whisper_model_path: Path
    llm_model_path: Path | None
    llm_quantization: str | None
    llm_output_language: str
    sample_rate: int
    chunk_seconds: int
    chunk_overlap_seconds: int
    batch_size: int
    device: str
    language: str | None
    trigger_terms: list[str]

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "AppConfig":
        load_dotenv()

        resolved_root = (root_dir or Path(__file__).resolve().parents[1]).resolve()
        input_calls_dir = resolved_root / os.getenv("INPUT_CALLS_DIR", "data/input_calls")
        work_dir = resolved_root / os.getenv("WORK_DIR", "data/work")
        output_dir = resolved_root / os.getenv("OUTPUT_DIR", "data/output")
        whisper_model_path = resolved_root / os.getenv(
            "WHISPER_MODEL_PATH", "models/whisper-large-v3"
        )

        llm_path_raw = os.getenv("LLM_MODEL_PATH", "models/llm").strip()
        llm_model_path = (resolved_root / llm_path_raw) if llm_path_raw else None
        llm_quantization = os.getenv("LLM_QUANTIZATION", "").strip().lower() or None
        language = os.getenv("LANGUAGE", "").strip()
        llm_output_language = (
            os.getenv("LLM_OUTPUT_LANGUAGE", "").strip().lower()
            or (language.lower() if language else "en")
        )

        trigger_terms = _split_csv(os.getenv("TRIGGER_TERMS"))
        if not trigger_terms:
            trigger_terms = [
                "cancel",
                "refund",
                "angry",
                "escalate",
                "complaint",
                "lawsuit",
                "supervisor",
            ]

        return cls(
            root_dir=resolved_root,
            input_calls_dir=input_calls_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            whisper_model_path=whisper_model_path,
            llm_model_path=llm_model_path,
            llm_quantization=llm_quantization,
            llm_output_language=llm_output_language,
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            chunk_seconds=int(os.getenv("CHUNK_SECONDS", "30")),
            chunk_overlap_seconds=int(os.getenv("CHUNK_OVERLAP_SECONDS", "2")),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
            device=os.getenv("DEVICE", "auto").strip().lower(),
            language=language or None,
            trigger_terms=trigger_terms,
        )


def ensure_base_dirs(config: AppConfig) -> None:
    config.input_calls_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
