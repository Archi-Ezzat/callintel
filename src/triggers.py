from __future__ import annotations

import json
import re
from pathlib import Path


def _snippet(text: str, start: int, end: int, radius: int = 60) -> str:
    left = max(start - radius, 0)
    right = min(end + radius, len(text))
    return text[left:right].replace("\n", " ").strip()


def find_triggers(text: str, terms: list[str]) -> dict:
    findings: list[dict] = []
    counts: dict[str, int] = {}

    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        counts[term] = len(matches)

        for match in matches:
            findings.append(
                {
                    "term": term,
                    "start": match.start(),
                    "end": match.end(),
                    "snippet": _snippet(text, match.start(), match.end()),
                }
            )

    findings.sort(key=lambda item: item["start"])
    return {
        "triggered": bool(findings),
        "total_hits": len(findings),
        "counts": counts,
        "findings": findings,
    }


def write_triggers(text: str, terms: list[str], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = find_triggers(text, terms)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload

