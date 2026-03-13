"""Regex-based PII redaction for transcripts.

This module provides pattern-based detection and redaction of common
personally identifiable information (PII) in text, supporting both
English and Arabic contexts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    total_redacted: int
    redacted_types: dict[str, int]


# ── Patterns ──────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Credit card numbers (4 groups of 4 digits, optional separators)
    ("CREDIT_CARD", re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    )),
    # Phone numbers — international format
    ("PHONE", re.compile(
        r"(?:\+?\d{1,3}[\s\-]?)?"
        r"(?:\(?\d{2,4}\)?[\s\-]?)?"
        r"\d{3,4}[\s\-]?\d{3,4}\b"
    )),
    # Email addresses
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    # Egyptian National ID (14 digits)
    ("NATIONAL_ID", re.compile(r"\b[23]\d{13}\b")),
    # US SSN (###-##-####)
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # IP addresses (v4)
    ("IP_ADDRESS", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )),
    # IBAN (2 letter country code + up to 34 alphanum)
    ("IBAN", re.compile(
        r"\b[A-Z]{2}\d{2}\s?[\dA-Z]{4}(?:\s?[\dA-Z]{4}){1,7}(?:\s?[\dA-Z]{1,4})?\b"
    )),
]


def redact_pii(text: str) -> RedactionResult:
    """Replace PII patterns in text with redaction placeholders.

    Returns a ``RedactionResult`` with the cleaned text, count of total
    redactions, and counts per PII type.
    """
    redacted = text
    counts: dict[str, int] = {}

    for pii_type, pattern in _PATTERNS:
        matches = pattern.findall(redacted)
        if matches:
            count = len(matches)
            counts[pii_type] = counts.get(pii_type, 0) + count
            placeholder = f"[REDACTED_{pii_type}]"
            redacted = pattern.sub(placeholder, redacted)

    total = sum(counts.values())
    if total > 0:
        logger.info("Redacted %d PII items: %s", total, counts)

    return RedactionResult(
        redacted_text=redacted,
        total_redacted=total,
        redacted_types=counts,
    )
