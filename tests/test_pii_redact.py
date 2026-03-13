# -*- coding: utf-8 -*-
"""Unit tests for PII redaction."""
import pytest

from src.pii_redact import redact_pii


class TestRedactPii:
    def test_credit_card(self):
        result = redact_pii("My card is 4111-1111-1111-1111 please charge")
        assert "4111" not in result.redacted_text
        assert "[REDACTED_CREDIT_CARD]" in result.redacted_text
        assert result.redacted_types.get("CREDIT_CARD", 0) >= 1

    def test_credit_card_no_separators(self):
        result = redact_pii("card 4111111111111111 done")
        assert "[REDACTED_CREDIT_CARD]" in result.redacted_text

    def test_email(self):
        result = redact_pii("email me at john.doe@example.com")
        assert "john.doe@example.com" not in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text

    def test_ssn(self):
        result = redact_pii("my SSN is 123-45-6789")
        assert "123-45-6789" not in result.redacted_text
        assert "[REDACTED_SSN]" in result.redacted_text

    def test_ip_address(self):
        result = redact_pii("server at 192.168.1.100")
        assert "192.168.1.100" not in result.redacted_text
        assert "[REDACTED_IP_ADDRESS]" in result.redacted_text

    def test_no_pii(self):
        text = "this is a safe message with no personal data"
        result = redact_pii(text)
        assert result.redacted_text == text
        assert result.total_redacted == 0

    def test_multiple_types(self):
        text = "Call 555-123-4567 or email test@example.com"
        result = redact_pii(text)
        assert result.total_redacted >= 2

    def test_arabic_text_preserved(self):
        text = "\u0645\u0631\u062D\u0628\u0627 \u0643\u064A\u0641 \u0627\u0644\u062D\u0627\u0644"
        result = redact_pii(text)
        assert result.redacted_text == text
        assert result.total_redacted == 0

    def test_egyptian_national_id(self):
        result = redact_pii("ID number: 29901011234567")
        assert "29901011234567" not in result.redacted_text
        assert result.total_redacted >= 1
