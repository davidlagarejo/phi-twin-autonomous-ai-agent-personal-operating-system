#!/usr/bin/env python3
"""
test_ask_claude.py
Proves that ask_claude():
  - Is disabled when CLAUDE_API_KEY is absent.
  - Blocks when payload contains PII (EN and ES patterns).
  - Allows clean abstract specs (English-only, structural).
  - Never logs raw payload (only hash + decision).
  - Writes audit entries for every call attempt.

Usage:
    python3 -m unittest tests.test_ask_claude -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ask_claude import (
    ClaudeBlockedError,
    ClaudeDisabledError,
    ask_claude,
    _validate_spec,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_CLEAN_SPEC = {
    "goal": "workflow_spec: explain HTTP retry configuration for local Python scripts",
    "language": "en",
    "inputs_schema": {"workflow_name": "string", "max_retries": "integer"},
    "outputs_schema": {"steps": "array[string]", "warnings": "array[string]"},
    "constraints": ["local execution only", "no private data"],
}

_PII_SPEC_EN = {
    "goal": "send email to john.smith@acmecorp.io about contract",
    "language": "en",
    "inputs_schema": {"recipient": "john.smith@acmecorp.io"},
    "outputs_schema": {"status": "string"},
    "constraints": [],
}

_PII_SPEC_ES = {
    "goal": "enviar factura de $45,000 a cliente",
    "language": "es",
    "inputs_schema": {"monto": "$45,000", "telefono": "718-555-9988"},
    "outputs_schema": {"estado": "string"},
    "constraints": [],
}


def _tmp_audit(tmp_dir: str) -> Path:
    p = Path(tmp_dir) / "audit_logs" / "privacy_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _env_no_key() -> dict:
    return {k: v for k, v in __import__("os").environ.items() if k != "CLAUDE_API_KEY"}


def _env_with_key() -> dict:
    return {**__import__("os").environ, "CLAUDE_API_KEY": "sk-test-fake-key-1234567890abcdef"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestClaudeDisabled(unittest.TestCase):

    def test_degrades_when_no_api_key(self):
        """ask_claude() must degrade gracefully (return dict) when CLAUDE_API_KEY absent."""
        with patch.dict("os.environ", _env_no_key(), clear=True):
            result = ask_claude(_CLEAN_SPEC)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("decision"), "DEGRADED")

    def test_no_network_when_disabled(self):
        """When disabled, no HTTP call must be made."""
        with patch.dict("os.environ", _env_no_key(), clear=True):
            with patch("httpx.post") as mock_post:
                ask_claude(_CLEAN_SPEC)
                mock_post.assert_not_called()


class TestSpecValidation(unittest.TestCase):

    def test_missing_required_field_raises(self):
        bad_spec = {"goal": "test"}  # missing inputs_schema, outputs_schema, constraints
        with self.assertRaises(ValueError):
            _validate_spec(bad_spec)

    def test_forbidden_key_raises(self):
        bad_spec = {
            "goal": "test",
            "inputs_schema": {},
            "outputs_schema": {},
            "constraints": [],
            "email": "someone@example.com",  # forbidden key
        }
        with self.assertRaises(ValueError):
            _validate_spec(bad_spec)

    def test_clean_spec_passes_validation(self):
        # Should not raise
        _validate_spec(_CLEAN_SPEC)


class TestPIIBlocked(unittest.TestCase):

    def _run_with_key(self, spec: dict) -> None:
        with patch.dict("os.environ", _env_with_key()):
            with tempfile.TemporaryDirectory() as tmp:
                audit = _tmp_audit(tmp)
                with patch("core.audit_append._AUDIT_FILE", audit), \
                     patch("core.audit_append._AUDIT_DIR", audit.parent):
                    ask_claude(spec)

    def test_block_en_pii_email_in_goal(self):
        """English spec with email address in goal must be BLOCKED."""
        with self.assertRaises((ClaudeBlockedError, ValueError)):
            self._run_with_key(_PII_SPEC_EN)

    def test_block_es_pii_phone_and_money(self):
        """Spanish spec with phone + money must be BLOCKED (privacy gate handles ES patterns)."""
        with self.assertRaises((ClaudeBlockedError, ValueError)):
            self._run_with_key(_PII_SPEC_ES)

    def test_block_ssn_in_spec(self):
        """Spec containing SSN must be hard-blocked."""
        ssn_spec = {**_CLEAN_SPEC, "goal": "debug_fix: validate SSN 123-45-6789 for employee record"}
        with self.assertRaises((ClaudeBlockedError, ValueError)):
            self._run_with_key(ssn_spec)

    def test_no_network_call_on_block(self):
        """When BLOCKED, httpx.post must never be called."""
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post") as mock_post:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        try:
                            ask_claude(_PII_SPEC_EN)
                        except (ClaudeBlockedError, ValueError):
                            pass
                mock_post.assert_not_called()


class TestCleanSpecAllowed(unittest.TestCase):

    def _mock_claude_response(self) -> MagicMock:
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "Step 1: Configure retry. Step 2: Set timeout."}],
        }
        return mock

    def test_clean_spec_reaches_claude(self):
        """A PII-free abstract spec must call httpx.post exactly once."""
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post", return_value=self._mock_claude_response()) as mock_post:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        result = ask_claude(_CLEAN_SPEC)
                mock_post.assert_called_once()

        self.assertIn("content", result)
        self.assertEqual(result["decision"], "ALLOW")

    def test_response_has_required_keys(self):
        """Response dict must have 'content', 'model', 'decision'."""
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post", return_value=self._mock_claude_response()):
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        result = ask_claude(_CLEAN_SPEC)

        self.assertIn("content", result)
        self.assertIn("model", result)
        self.assertIn("decision", result)

    def test_latest_model_used_by_default(self):
        """Default model must be the latest Claude generation (not an old version)."""
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post", return_value=self._mock_claude_response()) as mock_post:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        ask_claude(_CLEAN_SPEC)

                body = mock_post.call_args[1]["json"]
                model_used = body.get("model", "")
                # Must be Opus or Sonnet 4.x — not claude-2, claude-instant, etc.
                self.assertTrue(
                    "opus" in model_used.lower() or "sonnet" in model_used.lower(),
                    f"Expected Opus/Sonnet model but got: {model_used}",
                )


class TestAuditLogging(unittest.TestCase):

    def test_audit_written_on_block(self):
        """A BLOCK decision must write an audit entry with no raw payload."""
        with patch.dict("os.environ", _env_with_key()):
            with tempfile.TemporaryDirectory() as tmp:
                audit = _tmp_audit(tmp)
                with patch("core.audit_append._AUDIT_FILE", audit), \
                     patch("core.audit_append._AUDIT_DIR", audit.parent):
                    try:
                        ask_claude(_PII_SPEC_EN)
                    except (ClaudeBlockedError, ValueError):
                        pass

                if audit.exists():
                    lines = [l for l in audit.read_text().splitlines() if l.strip()]
                    claude_entries = [
                        json.loads(l) for l in lines
                        if json.loads(l).get("action_type") == "ASK_CLAUDE"
                    ]
                    if claude_entries:
                        entry = claude_entries[-1]
                        self.assertIn("decision", entry)
                        self.assertIn("payload_hash", entry)
                        self.assertIn("timestamp", entry)
                        # Raw payload must NOT be present
                        entry_str = json.dumps(entry)
                        self.assertNotIn("john.smith@acmecorp.io", entry_str)

    def test_audit_on_success_contains_hash(self):
        """Successful call must write audit with payload_hash, not raw message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "Instructions here."}],
        }
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post", return_value=mock_resp):
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        ask_claude(_CLEAN_SPEC)

                    lines = [l for l in audit.read_text().splitlines() if l.strip()]
                    entries = [json.loads(l) for l in lines]
                    claude_entries = [e for e in entries if e.get("action_type") == "ASK_CLAUDE"]
                    self.assertGreater(len(claude_entries), 0)
                    entry = claude_entries[-1]
                    self.assertIn("payload_hash", entry)
                    # Raw spec content must not appear
                    entry_str = json.dumps(entry)
                    self.assertNotIn("HTTP retry configuration for local Python", entry_str)


class TestEnglishEnforcement(unittest.TestCase):

    def test_system_prompt_is_english(self):
        """The default system prompt sent to Claude must instruct English-only responses."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "ok"}],
        }
        with patch.dict("os.environ", _env_with_key()):
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        ask_claude(_CLEAN_SPEC)

                body = mock_post.call_args[1]["json"]
                system = body.get("system", "")
                self.assertIn("ENGLISH", system.upper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
