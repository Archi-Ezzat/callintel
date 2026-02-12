from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig, ensure_base_dirs
from .evaluate_llm import evaluate_transcript, format_llm_report_text
from .merge import merge_transcripts
from .transcribe import WhisperTranscriber
from .triggers import write_triggers
from .utils_audio import chunk_audio, copy_audio, normalize_audio, slugify
from transformers.utils import logging as hf_logging

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
TRANSCRIPT_DIR_NAME = "Transcript"
LLM_DIR_NAME = "LLM_Justification"


def _next_work_index(work_dir: Path) -> int:
    max_seen = 0
    for item in work_dir.iterdir():
        if not item.is_dir():
            continue
        prefix = item.name.split("_", 1)[0]
        if prefix.isdigit():
            max_seen = max(max_seen, int(prefix))
    return max_seen + 1


def _create_call_workspace(work_dir: Path, call_name: str) -> Path:
    idx = _next_work_index(work_dir)
    folder_name = f"{idx:04d}_{slugify(call_name)}"
    call_dir = work_dir / folder_name
    for part in ("audio", "chunks", "transcripts", "merged", "analysis"):
        (call_dir / part).mkdir(parents=True, exist_ok=True)
    return call_dir


def _create_output_workspace(output_dir: Path, audio_file: Path) -> Path:
    call_dir = output_dir / audio_file.name
    (call_dir / TRANSCRIPT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (call_dir / LLM_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return call_dir


def _list_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )


def process_file(
    audio_file: Path,
    config: AppConfig,
    transcriber: WhisperTranscriber | None,
    skip_llm: bool,
    force: bool,
) -> Path:
    call_dir = _create_call_workspace(config.work_dir, audio_file.stem)
    output_dir = _create_output_workspace(config.output_dir, audio_file)
    transcript_output_dir = output_dir / TRANSCRIPT_DIR_NAME
    llm_output_dir = output_dir / LLM_DIR_NAME

    original_path = call_dir / "audio" / f"original{audio_file.suffix.lower()}"
    normalized_path = call_dir / "audio" / "normalized.wav"
    chunks_dir = call_dir / "chunks"
    transcripts_dir = call_dir / "transcripts"
    merged_dir = call_dir / "merged"
    analysis_dir = call_dir / "analysis"

    copy_audio(audio_file, original_path)
    normalize_audio(audio_file, normalized_path, sample_rate=config.sample_rate)

    chunk_manifest = chunk_audio(
        normalized_path,
        chunks_dir=chunks_dir,
        chunk_seconds=config.chunk_seconds,
        overlap_seconds=config.chunk_overlap_seconds,
        sample_rate=config.sample_rate,
    )
    (chunks_dir / "chunks_manifest.json").write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if transcriber is None:
        raise RuntimeError("Transcriber was not initialized")
    transcriber.transcribe_chunks(chunk_manifest, transcripts_dir, force=force)

    merged = merge_transcripts(transcripts_dir, merged_dir, chunk_manifest=chunk_manifest)
    triggers_path = analysis_dir / "triggers.json"
    write_triggers(merged["full_text"], config.trigger_terms, triggers_path)

    if not skip_llm:
        evaluate_transcript(
            Path(merged["json_path"]),
            triggers_path,
            analysis_dir,
            llm_model_path=config.llm_model_path,
            llm_quantization=config.llm_quantization,
            llm_output_language=config.llm_output_language,
        )

    # Write final deliverables to output directory
    transcript_text_path = transcript_output_dir / "transcript.txt"
    transcript_json_path = transcript_output_dir / "transcript.json"
    transcript_text_path.write_text(merged["full_text"], encoding="utf-8")
    transcript_json_path.write_text(
        Path(merged["json_path"]).read_text(encoding="utf-8"), encoding="utf-8"
    )

    llm_report_path = analysis_dir / "llm_report.json"
    score_path = analysis_dir / "score.json"
    llm_text_path = llm_output_dir / "llm_report.txt"

    if llm_report_path.exists():
        llm_payload = json.loads(llm_report_path.read_text(encoding="utf-8"))
        llm_text_path.write_text(
            format_llm_report_text(llm_payload, output_language=config.llm_output_language),
            encoding="utf-8",
        )
        (llm_output_dir / "llm_report.json").write_text(
            llm_report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        if score_path.exists():
            (llm_output_dir / "score.json").write_text(
                score_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
    else:
        llm_text_path.write_text(
            "LLM summary was skipped for this run.\n", encoding="utf-8"
        )

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CallIntel on local audio calls.")
    parser.add_argument("--file", type=str, default=None, help="Single audio file to process.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM/heuristic evaluation.")
    parser.add_argument("--force", action="store_true", help="Re-run transcription if output exists.")
    return parser.parse_args()


def main() -> None:
    hf_logging.set_verbosity_error()
    args = parse_args()
    config = AppConfig.from_env()
    ensure_base_dirs(config)

    if args.file:
        files = [Path(args.file).resolve()]
    else:
        files = _list_input_files(config.input_calls_dir)

    if not files:
        print(f"No input audio files found in {config.input_calls_dir}")
        return

    transcriber = WhisperTranscriber(
        model_path=config.whisper_model_path,
        batch_size=config.batch_size,
        language=config.language,
        device=config.device,
    )

    for audio_file in files:
        if not audio_file.exists():
            print(f"Skipping missing file: {audio_file}")
            continue
        out_dir = process_file(
            audio_file=audio_file,
            config=config,
            transcriber=transcriber,
            skip_llm=args.skip_llm,
            force=args.force,
        )
        print(f"Processed: {audio_file.name} -> {out_dir}")


if __name__ == "__main__":
    main()
