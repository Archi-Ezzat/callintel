from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig, ensure_base_dirs
from .evaluate_llm import evaluate_transcript, format_llm_report_text
from .merge import merge_transcripts
from .transcribe import WhisperTranscriber
from .bad_words_dataset import bad_words_dataset
from .classifier import TextClassifier
from .risk_engine import RiskEvaluator
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

    # Risk scoring (rule-based)
    evaluator = RiskEvaluator(bad_words_dataset)
    risk_payload = evaluator.analyze_text(merged["full_text"])
    risk_detail_path = analysis_dir / "risk_detail.json"
    risk_summary_path = analysis_dir / "risk_summary.txt"
    risk_detail_path.write_text(
        json.dumps(risk_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    risk_summary_lines = [
        "ملخص المخاطر",
        f"درجة الخطورة: {risk_payload.get('risk_percentage')}",
        f"التصنيف: {risk_payload.get('risk_label')}",
        f"أعلى فئة خطورة: {risk_payload.get('max_severity_category')}",
        f"الفئات المفعّلة: {', '.join(risk_payload.get('triggered_categories') or [])}",
        f"عدد الكلمات الخطرة: {len(risk_payload.get('detected_words') or [])}",
    ]
    risk_summary_path.write_text("\n".join(risk_summary_lines) + "\n", encoding="utf-8")

    classifier_payload = None
    classifier_summary_path = analysis_dir / "classifier_summary.txt"
    classifier_detail_path = analysis_dir / "classifier_detail.json"
    combined_risk_path = analysis_dir / "combined_risk.json"
    combined_summary_path = analysis_dir / "combined_risk.txt"

    if config.classifier_model_path and config.classifier_model_path.exists():
        classifier = TextClassifier(config.classifier_model_path, device=config.device)
        result = classifier.predict(merged["full_text"])
        classifier_payload = {
            "top_label": result.top_label,
            "top_score": result.top_score,
            "category": result.category,
            "category_score": result.category_score,
            "category_scores": result.category_scores,
            "risk_score": result.risk_score,
            "raw_scores": result.raw_scores,
        }
        classifier_detail_path.write_text(
            json.dumps(classifier_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        classifier_summary_path.write_text(
            "\n".join(
                [
                    "ملخص المصنف",
                    f"التصنيف الأعلى: {result.top_label} ({round(result.top_score, 4)})",
                    f"الفئة المطابقة: {result.category}",
                    f"درجة خطورة المصنف: {round(result.risk_score, 4)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    rule_score = float(risk_payload.get("risk_score", 0.0))
    classifier_score = float(classifier_payload.get("risk_score", 0.0)) if classifier_payload else 0.0
    final_score = max(rule_score, classifier_score)
    final_percentage = round(final_score * 100, 2)
    combined_payload = {
        "rule_risk_score": rule_score,
        "classifier_risk_score": classifier_score,
        "final_risk_score": final_score,
        "final_risk_percentage": f"{final_percentage}%",
    }
    combined_risk_path.write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combined_summary_path.write_text(
        "\n".join(
            [
                "ملخص المخاطر النهائي",
                f"خطر القواعد: {round(rule_score, 4)}",
                f"خطر المصنف: {round(classifier_score, 4)}",
                f"الخطر النهائي: {final_percentage}%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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

    # Risk outputs to LLM_Justification
    if risk_detail_path.exists():
        (llm_output_dir / "risk_detail.json").write_text(
            risk_detail_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if risk_summary_path.exists():
        (llm_output_dir / "risk_summary.txt").write_text(
            risk_summary_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if classifier_detail_path.exists():
        (llm_output_dir / "classifier_detail.json").write_text(
            classifier_detail_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if classifier_summary_path.exists():
        (llm_output_dir / "classifier_summary.txt").write_text(
            classifier_summary_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if combined_risk_path.exists():
        (llm_output_dir / "combined_risk.json").write_text(
            combined_risk_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if combined_summary_path.exists():
        (llm_output_dir / "combined_risk.txt").write_text(
            combined_summary_path.read_text(encoding="utf-8"), encoding="utf-8"
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
