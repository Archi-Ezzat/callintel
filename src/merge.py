from __future__ import annotations

import json
from pathlib import Path


def _iter_transcript_files(transcripts_dir: Path) -> list[Path]:
    return sorted(transcripts_dir.glob("chunk_*.json"))


def merge_transcripts(
    transcripts_dir: Path, merged_dir: Path, chunk_manifest: list[dict] | None = None
) -> dict:
    merged_dir.mkdir(parents=True, exist_ok=True)
    transcript_files = _iter_transcript_files(transcripts_dir)

    chunk_start_by_index: dict[int, float] = {}
    if chunk_manifest:
        for item in chunk_manifest:
            chunk_start_by_index[int(item["index"])] = float(item.get("start_sec", 0.0))

    merged_chunks: list[dict] = []
    merged_segments: list[dict] = []
    text_parts: list[str] = []

    for transcript_file in transcript_files:
        with transcript_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        chunk_index = int(payload.get("chunk_index", 0))
        offset = float(payload.get("chunk_start_sec", chunk_start_by_index.get(chunk_index, 0.0)))
        chunk_text = (payload.get("text") or "").strip()
        if chunk_text:
            text_parts.append(chunk_text)

        merged_chunks.append(
            {
                "index": chunk_index,
                "file": transcript_file.name,
                "start_sec": offset,
                "end_sec": float(payload.get("chunk_end_sec", offset)),
                "text": chunk_text,
            }
        )

        for segment in payload.get("segments", []):
            local_start = segment.get("start")
            local_end = segment.get("end")
            merged_segments.append(
                {
                    "chunk_index": chunk_index,
                    "start_sec": round(offset + float(local_start), 3)
                    if local_start is not None
                    else None,
                    "end_sec": round(offset + float(local_end), 3) if local_end is not None else None,
                    "text": segment.get("text", "").strip(),
                }
            )

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

    return {"txt_path": str(txt_path), "json_path": str(json_path), "full_text": full_text}

