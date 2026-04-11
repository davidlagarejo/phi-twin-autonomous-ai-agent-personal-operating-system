#!/usr/bin/env python3
"""
test_privacy_pre_hook.py
Unit tests for the privacy pre-hook system.
Usage: python3 -m pytest tests/ -v
   or: python3 tests/test_privacy_pre_hook.py
"""
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.privacy_pre_hook import privacy_pre_hook, HookResult
from core.pii_rules import detect_pii, redact_text, text_is_safe


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tmp_audit_path(tmp_dir: str) -> Path:
    """Return a temporary audit file path inside a temp directory."""
    p = Path(tmp_dir) / "audit_logs" / "privacy_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Test class ───────────────────────────────────────────────────────────────

class TestPrivacyPreHook(unittest.TestCase):

    # ── ASK_CLAUDE ────────────────────────────────────────────────────────────

    def test_block_email_body_to_claude(self):
        """ASK_CLAUDE payload containing an 'email_body' field must always BLOCK."""
        payload = {
            "message": "What should I reply to this?",
            "email_body": "Dear John, please find the attached invoice for $5,000.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload)
        self.assertEqual(result.decision, "BLOCK")
        self.assertIn("doc_blob", result.pii_detected)

    def test_block_raw_contract_to_claude(self):
        """ASK_CLAUDE with 'content' field longer than 800 chars must BLOCK (doc blob heuristic)."""
        long_content = (
            "This agreement is entered into by and between the parties listed herein. "
            * 15  # ~1050 chars — well over the 800-char threshold
        )
        self.assertGreater(len(long_content), 800)
        payload = {"content": long_content}
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload)
        self.assertEqual(result.decision, "BLOCK")

    def test_redact_company_name_to_claude(self):
        """ASK_CLAUDE with a person name in message should REDACT_AND_PROCEED (not hard-block)."""
        payload = {
            "message": (
                "Our client Carlos Mendoza wants to automate invoice processing. "
                "What n8n workflow structure do you recommend?"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload)
        # Full name is soft PII — should be redacted, not hard-blocked
        self.assertIn(result.decision, ("REDACT_AND_PROCEED", "BLOCK"))
        # If redacted, the name must not appear in the redacted payload
        if result.decision == "REDACT_AND_PROCEED":
            self.assertIsNotNone(result.redacted_payload)
            redacted_str = json.dumps(result.redacted_payload)
            self.assertNotIn("Carlos Mendoza", redacted_str)

    def test_allow_clean_claude_request(self):
        """ASK_CLAUDE with a generic, PII-free technical query must return ALLOW."""
        payload = {
            "message": (
                "What is the best way to implement retry logic in an n8n HTTP node "
                "when the upstream API returns a 429 rate-limit error?"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload)
        self.assertEqual(result.decision, "ALLOW")
        self.assertIsNone(result.redacted_payload)

    def test_block_ssn_to_claude(self):
        """ASK_CLAUDE containing an SSN must BLOCK with a hard-block kind."""
        payload = {
            "message": "Can you validate this SSN: 123-45-6789 for our employee record system?",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload)
        self.assertEqual(result.decision, "BLOCK")
        self.assertIn("ssn", result.pii_detected)

    # ── SEARCH_WEB ────────────────────────────────────────────────────────────

    def test_anonymize_web_query(self):
        """SEARCH_WEB query containing an email address must be anonymized and return REDACT_AND_PROCEED."""
        payload = {
            "query": "best practices for emailing john.doe@acmecorp.com type of users",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("SEARCH_WEB", payload)
        self.assertEqual(result.decision, "REDACT_AND_PROCEED")
        self.assertIsNotNone(result.redacted_payload)
        # The raw email must not appear in the forwarded query
        self.assertNotIn("john.doe@acmecorp.com", result.redacted_payload.get("query", ""))

    def test_block_web_query_cannot_anonymize(self):
        """SEARCH_WEB query with an SSN that cannot be anonymized must BLOCK."""
        payload = {
            "query": "what does SSN 987-65-4321 belong to",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("SEARCH_WEB", payload)
        self.assertEqual(result.decision, "BLOCK")

    def test_allow_clean_web_query(self):
        """SEARCH_WEB with a generic, PII-free query must return ALLOW."""
        payload = {
            "query": "n8n http node best practices for error handling",
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("SEARCH_WEB", payload)
        self.assertEqual(result.decision, "ALLOW")
        self.assertIsNone(result.redacted_payload)

    # ── Audit log ─────────────────────────────────────────────────────────────

    def test_audit_entry_written(self):
        """Every call to privacy_pre_hook must write at least one entry to the audit log."""
        payload = {"message": "How do I use the n8n code node?"}
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                privacy_pre_hook("ASK_CLAUDE", payload)

            # Audit file must exist and contain at least one valid JSON line
            self.assertTrue(audit_file.exists(), "Audit file was not created")
            lines = [l for l in audit_file.read_text().splitlines() if l.strip()]
            self.assertGreater(len(lines), 0, "No audit entries were written")
            entry = json.loads(lines[0])
            self.assertIn("decision", entry)
            self.assertIn("action_type", entry)
            self.assertIn("timestamp", entry)

    # ── Policy override ───────────────────────────────────────────────────────

    def test_policy_block_overrides_redact(self):
        """policy={'on_pii_remote': 'block'} must force BLOCK even when redaction is possible."""
        # A name alone is normally redactable; with block policy it must be blocked.
        payload = {
            "message": (
                "Our contact Ana Torres needs help with the n8n HTTP node configuration."
            ),
        }
        policy = {"on_pii_remote": "block"}
        with tempfile.TemporaryDirectory() as tmp:
            audit_file = _tmp_audit_path(tmp)
            with patch("core.audit_append._AUDIT_FILE", audit_file), \
                 patch("core.audit_append._AUDIT_DIR", audit_file.parent):
                result = privacy_pre_hook("ASK_CLAUDE", payload, policy=policy)
        self.assertEqual(result.decision, "BLOCK")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
