from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _iter_transcript_files(transcripts_dir: Path) -> list[Path]:
    return sorted(transcripts_dir.glob("chunk_*.json"))


def _remove_overlap_prefix(
    current_text: str,
    previous_text: str,
    overlap_ratio: float = 0.15,
) -> str:
    """Remove the overlapping prefix from current_text that repeats the end of previous_text.

    Uses character-level comparison to find the best overlap point.
    """
    if not previous_text or not current_text:
        return current_text

    # Look at the end of previous text and start of current text
    max_compare = min(
        int(len(current_text) * overlap_ratio) + 50,
        len(current_text),
        len(previous_text),
    )
    if max_compare < 5:
        return current_text

    prev_tail = previous_text[-max_compare:]
    curr_head = current_text[:max_compare]

    # Find the longest suffix of prev_tail that is a prefix of curr_head
    best_overlap = 0
    for length in range(5, max_compare + 1):
        suffix = prev_tail[-length:]
        if curr_head.startswith(suffix):
            best_overlap = length

    if best_overlap > 0:
        trimmed = current_text[best_overlap:].strip()
        logger.debug("Trimmed %d overlap characters from chunk", best_overlap)
        return trimmed

    return current_text


def _filter_overlap_segments(
    segments: list[dict],
    previous_end_sec: float | None,
    overlap_seconds: float,
) -> list[dict]:
    """Skip segments that fall entirely within the overlap window of the previous chunk."""
    if previous_end_sec is None or overlap_seconds <= 0:
        return segments

    threshold = previous_end_sec - (overlap_seconds * 0.5)
    filtered = []
    for seg in segments:
        seg_start = seg.get("start")
        if seg_start is not None and (seg_start + seg.get("chunk_offset", 0)) < threshold:
            continue
        filtered.append(seg)
    return filtered


def merge_transcripts(
    transcripts_dir: Path,
    merged_dir: Path,
    chunk_manifest: list[dict] | None = None,
    overlap_seconds: float = 2.0,
) -> dict:
    merged_dir.mkdir(parents=True, exist_ok=True)
    transcript_files = _iter_transcript_files(transcripts_dir)

    chunk_start_by_index: dict[int, float] = {}
    chunk_end_by_index: dict[int, float] = {}
    if chunk_manifest:
        for item in chunk_manifest:
            chunk_start_by_index[int(item["index"])] = float(item.get("start_sec", 0.0))
            chunk_end_by_index[int(item["index"])] = float(item.get("end_sec", 0.0))

    merged_chunks: list[dict] = []
    merged_segments: list[dict] = []
    text_parts: list[str] = []
    previous_text: str = ""
    previous_end_sec: float | None = None

    for transcript_file in transcript_files:
        with transcript_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        chunk_index = int(payload.get("chunk_index", 0))
        offset = float(payload.get("chunk_start_sec", chunk_start_by_index.get(chunk_index, 0.0)))
        chunk_text = (payload.get("text") or "").strip()

        # Deduplicate overlap: trim the repeated prefix from chunks after the first
        if chunk_index > 0 and previous_text and chunk_text:
            chunk_text = _remove_overlap_prefix(chunk_text, previous_text)

        if chunk_text:
            text_parts.append(chunk_text)
            previous_text = chunk_text

        merged_chunks.append(
            {
                "index": chunk_index,
                "file": transcript_file.name,
                "start_sec": offset,
                "end_sec": float(payload.get("chunk_end_sec", chunk_end_by_index.get(chunk_index, offset))),
                "text": chunk_text,
            }
        )

        for segment in payload.get("segments", []):
            local_start = segment.get("start")
            local_end = segment.get("end")
            abs_start = round(offset + float(local_start), 3) if local_start is not None else None
            abs_end = round(offset + float(local_end), 3) if local_end is not None else None

            # Skip segments in the overlap zone that duplicate previous chunk
            if (
                chunk_index > 0
                and previous_end_sec is not None
                and abs_start is not None
                and abs_start < previous_end_sec - (overlap_seconds * 0.3)
            ):
                continue

            merged_segments.append(
                {
                    "chunk_index": chunk_index,
                    "start_sec": abs_start,
                    "end_sec": abs_end,
                    "text": segment.get("text", "").strip(),
                }
            )

        previous_end_sec = float(payload.get("chunk_end_sec", chunk_end_by_index.get(chunk_index, offset)))

    full_text = "\n".join(text_parts).strip()
    full_payload = {
        "text": full_text,
        "total_chunks": len(merged_chunks),
        "chunks": sorted(merged_chunks, key=lambda item: item["index"]),
        "segments": sorted(
            merged_segments,
            key=lambda item: (
                item["start_sec"] if item["start_sec"] is not None else float("inf"),
                item["chunk_index"],
            ),
        ),
    }

    txt_path = merged_dir / "transcript_full.txt"
    json_path = merged_dir / "transcript_full.json"

    txt_path.write_text(full_text, encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(full_payload, fh, ensure_ascii=False, indent=2)

    logger.info("Merged %d chunks into transcript (%d chars)", len(merged_chunks), len(full_text))
    return {"txt_path": str(txt_path), "json_path": str(json_path), "full_text": full_text}
