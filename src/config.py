from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_TRIGGERS_BY_LANG: dict[str, list[str]] = {
    "ar": [
        "إلغاء",
        "استرداد",
        "زعلان",
        "غاضب",
        "تصعيد",
        "شكوى",
        "دعوى",
        "مدير",
        "مش راضي",
    ],
    "en": [
        "cancel",
        "refund",
        "angry",
        "escalate",
        "complaint",
        "lawsuit",
        "supervisor",
        "not happy",
    ],
}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_first(*keys: str) -> str | None:
    """Return the first non-empty environment value for the given keys."""
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _env_int(key: str, default: int) -> int:
    """Read an integer from the environment with a descriptive error on failure."""
    raw = os.getenv(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid integer value for {key}: '{raw}'") from None


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    input_calls_dir: Path
    work_dir: Path
    output_dir: Path
    whisper_model_path: Path
    llm_model_path: Path | None
    llm_quantization: str | None
    llm_api_base_url: str
    llm_api_model: str | None
    llm_output_language: str
    classifier_model_path: Path | None
    sentiment_model_path: Path | None
    hf_token: str | None
    diarization_model: str | None
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
        llm_api_base_url = os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:1234/v1").strip()
        llm_api_model = os.getenv("LLM_API_MODEL", "").strip() or None
        language = os.getenv("LANGUAGE", "").strip()
        llm_output_language = (
            os.getenv("LLM_OUTPUT_LANGUAGE", "").strip().lower()
            or (language.lower() if language else "en")
        )
        classifier_path_raw = os.getenv("CLASSIFIER_MODEL_PATH", "").strip()
        classifier_model_path = (resolved_root / classifier_path_raw) if classifier_path_raw else None
        sentiment_path_raw = os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment").strip()
        sentiment_model_path = (resolved_root / sentiment_path_raw) if sentiment_path_raw else None
        hf_token = _env_first("HF_TOKEN", "HUGGINGFACE_TOKEN")
        diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1").strip() or None

        trigger_terms = _split_csv(os.getenv("TRIGGER_TERMS"))
        if not trigger_terms:
            lang_key = language.lower() if language else "en"
            trigger_terms = DEFAULT_TRIGGERS_BY_LANG.get(
                lang_key, DEFAULT_TRIGGERS_BY_LANG["en"]
            )

        sample_rate = _env_int("SAMPLE_RATE", 16000)
        chunk_seconds = _env_int("CHUNK_SECONDS", 30)
        chunk_overlap = _env_int("CHUNK_OVERLAP_SECONDS", 2)
        batch_size = _env_int("BATCH_SIZE", 8)

        config = cls(
            root_dir=resolved_root,
            input_calls_dir=input_calls_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            whisper_model_path=whisper_model_path,
            llm_model_path=llm_model_path,
            llm_quantization=llm_quantization,
            llm_api_base_url=llm_api_base_url,
            llm_api_model=llm_api_model,
            llm_output_language=llm_output_language,
            classifier_model_path=classifier_model_path,
            sentiment_model_path=sentiment_model_path,
            hf_token=hf_token,
            diarization_model=diarization_model,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            chunk_overlap_seconds=chunk_overlap,
            batch_size=batch_size,
            device=os.getenv("DEVICE", "auto").strip().lower(),
            language=language or None,
            trigger_terms=trigger_terms,
        )
        logger.debug("Loaded config: language=%s, device=%s", config.language, config.device)
        return config


def ensure_base_dirs(config: AppConfig) -> None:
    config.input_calls_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
