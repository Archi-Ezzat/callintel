from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import AppConfig, ensure_base_dirs
from .diarize import SpeakerDiarizer, align_transcript_with_speakers, write_diarization_output
from .evaluate_llm import evaluate_transcript, format_llm_report_text
from .merge import merge_transcripts
from .pii_redact import redact_pii
from .sentiment import SentimentAnalyzer
from .transcribe import WhisperTranscriber
from .bad_words_dataset import bad_words_dataset
from .classifier import TextClassifier
from .risk_engine import RiskEvaluator
from .triggers import write_triggers
from .utils_audio import chunk_audio, copy_audio, normalize_audio, slugify, validate_audio
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
TRANSCRIPT_DIR_NAME = "Transcript"
LLM_DIR_NAME = "LLM_Justification"

# --- UI string translations (I-1) ---
TRANSLATIONS = {
    "ar": {
        "risk_summary_title": "ملخص المخاطر",
        "risk_score_label": "درجة الخطورة",
        "risk_class_label": "التصنيف",
        "max_category_label": "أعلى فئة خطورة",
        "triggered_cats_label": "الفئات المفعّلة",
        "detected_count_label": "عدد الكلمات الخطرة",
        "classifier_title": "ملخص المصنف",
        "classifier_top_label": "التصنيف الأعلى",
        "classifier_match_label": "الفئة المطابقة",
        "classifier_risk_label": "درجة خطورة المصنف",
        "combined_title": "ملخص المخاطر النهائي",
        "rule_risk_label": "خطر القواعد",
        "clf_risk_label": "خطر المصنف",
        "final_risk_label": "الخطر النهائي",
    },
    "en": {
        "risk_summary_title": "Risk Summary",
        "risk_score_label": "Risk Score",
        "risk_class_label": "Classification",
        "max_category_label": "Highest Severity Category",
        "triggered_cats_label": "Triggered Categories",
        "detected_count_label": "Detected Risky Words",
        "classifier_title": "Classifier Summary",
        "classifier_top_label": "Top Label",
        "classifier_match_label": "Matched Category",
        "classifier_risk_label": "Classifier Risk Score",
        "combined_title": "Final Risk Summary",
        "rule_risk_label": "Rule-Based Risk",
        "clf_risk_label": "Classifier Risk",
        "final_risk_label": "Final Risk",
    },
}


def _t(key: str, lang: str) -> str:
    """Get a translated string, falling back to English."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )


def _safe_console_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = message.encode(encoding, errors="backslashreplace").decode(
            encoding, errors="ignore"
        )
        print(safe)


def _next_work_index(work_dir: Path) -> int:
    max_seen = 0
    if not work_dir.exists():
        return 1
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


# ---------- Pipeline stage functions (MO-1) ----------

def stage_normalize(audio_file: Path, call_dir: Path, config: AppConfig) -> tuple:
    """Stage 1: Copy original and normalize audio. Returns (audio_array, metadata)."""
    original_path = call_dir / "audio" / f"original{audio_file.suffix.lower()}"
    normalized_path = call_dir / "audio" / "normalized.wav"
    copy_audio(audio_file, original_path)
    audio_array, metadata = normalize_audio(
        audio_file, normalized_path, sample_rate=config.sample_rate
    )
    return audio_array, normalized_path


def stage_chunk(audio_array, normalized_path: Path, call_dir: Path, config: AppConfig) -> list[dict]:
    """Stage 2: Split audio into overlapping chunks."""
    chunks_dir = call_dir / "chunks"
    chunk_manifest = chunk_audio(
        normalized_path,
        chunks_dir=chunks_dir,
        chunk_seconds=config.chunk_seconds,
        overlap_seconds=config.chunk_overlap_seconds,
        sample_rate=config.sample_rate,
        audio_array=audio_array,
    )
    (chunks_dir / "chunks_manifest.json").write_text(
        json.dumps(chunk_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chunk_manifest


def stage_transcribe(
    call_dir: Path,
    chunk_manifest: list[dict],
    transcriber: WhisperTranscriber,
    force: bool,
) -> None:
    """Stage 3: Transcribe each chunk with Whisper."""
    transcripts_dir = call_dir / "transcripts"
    transcriber.transcribe_chunks(chunk_manifest, transcripts_dir, force=force)


def stage_merge(call_dir: Path, chunk_manifest: list[dict], overlap_seconds: float) -> dict:
    """Stage 4: Merge all chunk transcripts into a single transcript."""
    transcripts_dir = call_dir / "transcripts"
    merged_dir = call_dir / "merged"
    return merge_transcripts(
        transcripts_dir, merged_dir,
        chunk_manifest=chunk_manifest,
        overlap_seconds=overlap_seconds,
    )


def stage_triggers(merged: dict, config: AppConfig, analysis_dir: Path) -> Path:
    """Stage 5: Detect trigger words in the transcript."""
    triggers_path = analysis_dir / "triggers.json"
    write_triggers(merged["full_text"], config.trigger_terms, triggers_path)
    return triggers_path


def stage_risk(
    merged: dict,
    risk_evaluator: RiskEvaluator,
    analysis_dir: Path,
    lang: str,
) -> dict:
    """Stage 6: Rule-based risk scoring."""
    risk_payload = risk_evaluator.analyze_text(merged["full_text"])
    risk_detail_path = analysis_dir / "risk_detail.json"
    risk_summary_path = analysis_dir / "risk_summary.txt"
    risk_detail_path.write_text(
        json.dumps(risk_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    risk_summary_lines = [
        _t("risk_summary_title", lang),
        f"{_t('risk_score_label', lang)}: {risk_payload.get('risk_percentage')}",
        f"{_t('risk_class_label', lang)}: {risk_payload.get('risk_label')}",
        f"{_t('max_category_label', lang)}: {risk_payload.get('max_severity_category')}",
        f"{_t('triggered_cats_label', lang)}: {', '.join(risk_payload.get('triggered_categories') or [])}",
        f"{_t('detected_count_label', lang)}: {len(risk_payload.get('detected_words') or [])}",
    ]
    risk_summary_path.write_text("\n".join(risk_summary_lines) + "\n", encoding="utf-8")
    return risk_payload


def stage_classify(
    merged: dict,
    classifier: TextClassifier | None,
    analysis_dir: Path,
    lang: str,
) -> dict | None:
    """Stage 7: Optional ML classification."""
    if classifier is None:
        return None
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
    (analysis_dir / "classifier_detail.json").write_text(
        json.dumps(classifier_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_dir / "classifier_summary.txt").write_text(
        "\n".join(
            [
                _t("classifier_title", lang),
                f"{_t('classifier_top_label', lang)}: {result.top_label} ({round(result.top_score, 4)})",
                f"{_t('classifier_match_label', lang)}: {result.category}",
                f"{_t('classifier_risk_label', lang)}: {round(result.risk_score, 4)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return classifier_payload


def stage_pii_redact(merged: dict, analysis_dir: Path) -> dict:
    """Stage 4b: Optionally redact PII from the transcript."""
    result = redact_pii(merged["full_text"])
    if result.total_redacted > 0:
        merged = dict(merged)  # shallow copy to avoid mutating caller's dict
        merged["full_text"] = result.redacted_text
        pii_report = {
            "total_redacted": result.total_redacted,
            "redacted_types": result.redacted_types,
        }
        (analysis_dir / "pii_redaction.json").write_text(
            json.dumps(pii_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("PII redaction: %d items redacted", result.total_redacted)
    return merged


def stage_diarize(
    diarizer: SpeakerDiarizer | None,
    normalized_path: Path,
    merged: dict,
    analysis_dir: Path,
) -> None:
    """Stage 4c: Optional speaker diarization.

    Runs pyannote on the full normalized audio, writes diarization.json,
    and enriches the merged transcript segments with speaker labels.
    """
    if diarizer is None:
        return

    result = diarizer.diarize(normalized_path)
    write_diarization_output(result, analysis_dir)

    # Align transcript segments with speakers
    segments = merged.get("segments", [])
    if segments and result.segments:
        merged["segments"] = align_transcript_with_speakers(segments, result.segments)
        logger.info("Aligned %d transcript segments with %d speakers", len(segments), result.num_speakers)


def stage_sentiment(
    merged: dict,
    analyzer: SentimentAnalyzer | None,
    analysis_dir: Path,
) -> dict | None:
    """Stage 7b: Optional sentiment analysis."""
    if analyzer is None:
        return None
    result = analyzer.analyze(merged["full_text"])
    payload = {
        "label": result.label,
        "score": result.score,
        "scores": result.scores,
        "risk_boost": result.risk_boost,
    }
    (analysis_dir / "sentiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Sentiment: %s (%.2f) risk_boost=%.4f", result.label, result.score, result.risk_boost)
    return payload


def stage_combined_risk(
    risk_payload: dict,
    classifier_payload: dict | None,
    sentiment_payload: dict | None,
    merged: dict,
    analysis_dir: Path,
    lang: str,
) -> dict:
    """Stage 8: Combine rule-based, classifier, and sentiment risk.

    Includes a position bonus for late-call detections and a sentiment
    penalty for negative sentiment.
    """
    rule_score = float(risk_payload.get("risk_score", 0.0))
    classifier_score = float(classifier_payload.get("risk_score", 0.0)) if classifier_payload else 0.0

    # Position-based bonus (F-4): check if detections cluster in the last third
    position_bonus = 0.0
    full_text = merged.get("full_text", "")
    text_len = len(full_text)
    if text_len > 0:
        detected_details = risk_payload.get("detected_details", [])
        last_third_start = text_len * 2 // 3
        late_hits = sum(1 for d in detected_details if d.get("start", 0) >= last_third_start)
        if late_hits > 0:
            position_bonus = min(0.05, late_hits * 0.01)

    # Sentiment-based boost (F-2): negative sentiment adds risk
    sentiment_boost = float(sentiment_payload.get("risk_boost", 0.0)) if sentiment_payload else 0.0

    final_score = min(1.0, max(rule_score, classifier_score) + position_bonus + sentiment_boost)
    final_percentage = round(final_score * 100, 2)
    combined_payload = {
        "rule_risk_score": rule_score,
        "classifier_risk_score": classifier_score,
        "sentiment_boost": round(sentiment_boost, 4),
        "position_bonus": round(position_bonus, 4),
        "final_risk_score": round(final_score, 4),
        "final_risk_percentage": f"{final_percentage}%",
    }
    if sentiment_payload:
        combined_payload["sentiment_label"] = sentiment_payload.get("label", "unknown")

    (analysis_dir / "combined_risk.json").write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_dir / "combined_risk.txt").write_text(
        "\n".join(
            [
                _t("combined_title", lang),
                f"{_t('rule_risk_label', lang)}: {round(rule_score, 4)}",
                f"{_t('clf_risk_label', lang)}: {round(classifier_score, 4)}",
                f"{_t('final_risk_label', lang)}: {final_percentage}%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return combined_payload


def stage_llm(merged: dict, triggers_path: Path, analysis_dir: Path, config: AppConfig) -> None:
    """Stage 9: Generate LLM or heuristic evaluation report."""
    evaluate_transcript(
        Path(merged["json_path"]),
        triggers_path,
        analysis_dir,
        llm_model_path=config.llm_model_path,
        llm_quantization=config.llm_quantization,
        llm_api_base_url=config.llm_api_base_url,
        llm_api_model=config.llm_api_model,
        llm_output_language=config.llm_output_language,
    )


def stage_export(call_dir: Path, output_dir: Path, merged: dict, config: AppConfig) -> None:
    """Stage 10: Copy final deliverables from workspace to output directory."""
    transcript_output_dir = output_dir / TRANSCRIPT_DIR_NAME
    llm_output_dir = output_dir / LLM_DIR_NAME
    analysis_dir = call_dir / "analysis"

    # Transcript
    (transcript_output_dir / "transcript.txt").write_text(merged["full_text"], encoding="utf-8")
    json_src = Path(merged["json_path"])
    if json_src.exists():
        shutil.copy2(str(json_src), str(transcript_output_dir / "transcript.json"))

    # Analysis outputs — copy all existing files
    export_files = [
        ("llm_report.json", llm_output_dir),
        ("llm_report.txt", llm_output_dir),
        ("score.json", llm_output_dir),
        ("risk_detail.json", llm_output_dir),
        ("risk_summary.txt", llm_output_dir),
        ("classifier_detail.json", llm_output_dir),
        ("classifier_summary.txt", llm_output_dir),
        ("combined_risk.json", llm_output_dir),
        ("combined_risk.txt", llm_output_dir),
        ("sentiment.json", llm_output_dir),
        ("pii_redaction.json", llm_output_dir),
        ("diarization.json", llm_output_dir),
    ]
    for filename, dest_dir in export_files:
        src = analysis_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(dest_dir / filename))

    # Format the LLM report as plaintext if we have it
    llm_report_src = analysis_dir / "llm_report.json"
    if llm_report_src.exists():
        llm_payload = json.loads(llm_report_src.read_text(encoding="utf-8"))
        (llm_output_dir / "llm_report.txt").write_text(
            format_llm_report_text(llm_payload, output_language=config.llm_output_language),
            encoding="utf-8",
        )
    else:
        (llm_output_dir / "llm_report.txt").write_text(
            "LLM summary was skipped for this run.\n", encoding="utf-8"
        )


# ---------- Main orchestration ----------

def process_file(
    audio_file: Path,
    config: AppConfig,
    transcriber: WhisperTranscriber | None,
    risk_evaluator: RiskEvaluator,
    classifier: TextClassifier | None,
    sentiment_analyzer: SentimentAnalyzer | None,
    diarizer: SpeakerDiarizer | None,
    skip_llm: bool,
    force: bool,
    redact: bool = False,
) -> Path:
    """Process a single audio file through all pipeline stages."""
    call_dir = _create_call_workspace(config.work_dir, audio_file.stem)
    output_dir = _create_output_workspace(config.output_dir, audio_file)
    analysis_dir = call_dir / "analysis"
    lang = config.llm_output_language or "en"

    logger.info("Processing %s -> workspace %s", audio_file.name, call_dir.name)

    # Stage 1: Normalize
    audio_array, normalized_path = stage_normalize(audio_file, call_dir, config)

    # Stage 2: Chunk
    chunk_manifest = stage_chunk(audio_array, normalized_path, call_dir, config)
    del audio_array  # Free memory

    # Stage 3: Transcribe
    if transcriber is None:
        raise RuntimeError("Transcriber was not initialized")
    stage_transcribe(call_dir, chunk_manifest, transcriber, force)

    # Stage 4: Merge
    merged = stage_merge(call_dir, chunk_manifest, config.chunk_overlap_seconds)

    # Stage 4b: PII redaction (optional)
    if redact:
        merged = stage_pii_redact(merged, analysis_dir)

    # Stage 4c: Speaker diarization (optional)
    stage_diarize(diarizer, normalized_path, merged, analysis_dir)

    # Stage 5: Triggers
    triggers_path = stage_triggers(merged, config, analysis_dir)

    # Stage 6: Risk scoring
    risk_payload = stage_risk(merged, risk_evaluator, analysis_dir, lang)

    # Stage 7: Classification
    classifier_payload = stage_classify(merged, classifier, analysis_dir, lang)

    # Stage 7b: Sentiment analysis
    sentiment_payload = stage_sentiment(merged, sentiment_analyzer, analysis_dir)

    # Stage 8: Combined risk
    stage_combined_risk(risk_payload, classifier_payload, sentiment_payload, merged, analysis_dir, lang)

    # Stage 9: LLM evaluation
    if not skip_llm:
        stage_llm(merged, triggers_path, analysis_dir, config)

    # Stage 10: Export
    stage_export(call_dir, output_dir, merged, config)

    logger.info("Completed %s -> %s", audio_file.name, output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CallIntel on local audio calls.")
    parser.add_argument("--file", type=str, default=None, help="Single audio file to process.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM/heuristic evaluation.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis.")
    parser.add_argument("--skip-diarize", action="store_true", help="Skip speaker diarization.")
    parser.add_argument("--force", action="store_true", help="Re-run transcription if output exists.")
    parser.add_argument("--redact-pii", action="store_true", help="Redact PII from transcripts before analysis.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    hf_logging.set_verbosity_error()
    args = parse_args()
    config = AppConfig.from_env()
    ensure_base_dirs(config)

    if args.file:
        files = [Path(args.file).resolve()]
    else:
        files = _list_input_files(config.input_calls_dir)

    if not files:
        _safe_console_print(f"No input audio files found in {config.input_calls_dir}")
        return

    # Validate all input files before loading expensive models (F-T3)
    for audio_file in files:
        if audio_file.exists():
            validate_audio(audio_file)

    # Initialize models ONCE (E-3)
    transcriber = WhisperTranscriber(
        model_path=config.whisper_model_path,
        batch_size=config.batch_size,
        language=config.language,
        device=config.device,
    )
    risk_evaluator = RiskEvaluator(bad_words_dataset)

    # F-T5: Check for model files, not just directory existence
    classifier: TextClassifier | None = None
    if (
        config.classifier_model_path
        and config.classifier_model_path.exists()
        and (config.classifier_model_path / "config.json").exists()
    ):
        logger.info("Loading classifier from %s", config.classifier_model_path)
        classifier = TextClassifier(config.classifier_model_path, device=config.device)
    elif config.classifier_model_path:
        logger.info(
            "Classifier path configured (%s) but no model files found; skipping classifier",
            config.classifier_model_path,
        )

    # Sentiment analyzer (optional)
    sentiment_analyzer: SentimentAnalyzer | None = None
    if (
        not args.skip_sentiment
        and config.sentiment_model_path
        and config.sentiment_model_path.exists()
        and (config.sentiment_model_path / "config.json").exists()
    ):
        logger.info("Loading sentiment model from %s", config.sentiment_model_path)
        sentiment_analyzer = SentimentAnalyzer(config.sentiment_model_path, device=config.device)
    elif not args.skip_sentiment and config.sentiment_model_path:
        logger.info(
            "Sentiment model path configured (%s) but no model files found; skipping sentiment",
            config.sentiment_model_path,
        )

    # Speaker diarizer (optional)
    diarizer: SpeakerDiarizer | None = None
    if (
        not args.skip_diarize
        and config.hf_token
        and config.diarization_model
    ):
        try:
            diarizer = SpeakerDiarizer(
                hf_token=config.hf_token,
                model_name=config.diarization_model,
                device=config.device,
            )
        except Exception:
            logger.warning("Failed to load diarization pipeline; skipping", exc_info=True)

    for audio_file in files:
        if not audio_file.exists():
            _safe_console_print(f"Skipping missing file: {audio_file}")
            continue
        try:
            out_dir = process_file(
                audio_file=audio_file,
                config=config,
                transcriber=transcriber,
                risk_evaluator=risk_evaluator,
                classifier=classifier,
                sentiment_analyzer=sentiment_analyzer,
                diarizer=diarizer,
                skip_llm=args.skip_llm,
                force=args.force,
                redact=args.redact_pii,
            )
            _safe_console_print(f"Processed: {audio_file.name} -> {out_dir}")
        except Exception:
            logger.error("Failed to process %s", audio_file.name, exc_info=True)
            _safe_console_print(f"ERROR: Failed to process {audio_file.name} (see log for details)")


if __name__ == "__main__":
    main()
