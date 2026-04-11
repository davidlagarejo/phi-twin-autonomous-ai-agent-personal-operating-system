"""
tests/test_research_smoke_contract.py
──────────────────────────────────────
Smoke-contract tests for privacy + audit invariants in the research pipeline.

Validates:
1. Web search is always routed through privacy_pre_hook BEFORE any HTTP call.
2. SEARXNG_URL missing → safe empty result, no network.
3. Non-local SEARXNG_URL → BLOCK, no network.
4. privacy_pre_hook BLOCK → _call_searxng never called.
5. Low-score library_fetch → SCORE_BELOW_THRESHOLD, no HTTP download.
6. SSRF guard blocks private-IP PDF URLs before download.
7. Audit entries contain payload_hash, never raw payload fields.
8. Audit _payload_hash is deterministic and collision-resistant.
9. literature.py BLOCK raises LiteratureBlockedError; REDACT uses sanitized query.

All tests are offline (no real SearXNG / Ollama / arXiv required).
Passes on macOS Python 3.9+.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. search_web — privacy gate + local-URL enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchWebPrivacyGate(unittest.TestCase):
    """search_web must gate every request through privacy_pre_hook before HTTP."""

    def _env_without_searxng(self) -> dict:
        return {k: v for k, v in os.environ.items() if k != "SEARXNG_URL"}

    # ── 1.1 Missing SEARXNG_URL ──────────────────────────────────────────────

    def test_no_searxng_url_returns_empty_no_network(self):
        """SEARXNG_URL not set → returns [] without any HTTP call."""
        with patch.dict(os.environ, self._env_without_searxng(), clear=True):
            with patch("tools.search._call_searxng") as mock_http:
                from tools.search import search_web
                results = search_web("test query")
        self.assertEqual(results, [])
        mock_http.assert_not_called()

    def test_no_searxng_url_writes_audit_block(self):
        """SEARXNG_URL not set → _audit_block is called (decision logged)."""
        with patch.dict(os.environ, self._env_without_searxng(), clear=True):
            with patch("tools.search._call_searxng"):
                with patch("tools.search._audit_block") as mock_audit:
                    from tools.search import search_web
                    search_web("test query")
        mock_audit.assert_called_once()
        reason_arg = mock_audit.call_args[0][0]
        self.assertIn("SEARXNG_URL", reason_arg)

    # ── 1.2 Non-local SEARXNG_URL ────────────────────────────────────────────

    def test_public_searxng_url_returns_empty_no_network(self):
        """Public domain SEARXNG_URL → BLOCK, returns [], _call_searxng not called."""
        with patch.dict(os.environ, {"SEARXNG_URL": "https://searxng.example.com"}):
            with patch("tools.search._call_searxng") as mock_http:
                from tools.search import search_web
                results = search_web("test query")
        self.assertEqual(results, [])
        mock_http.assert_not_called()

    def test_public_ip_searxng_url_blocked(self):
        """Public IP SEARXNG_URL → BLOCK, no network."""
        with patch.dict(os.environ, {"SEARXNG_URL": "http://8.8.8.8:8888"}):
            with patch("tools.search._call_searxng") as mock_http:
                from tools.search import search_web
                results = search_web("test query")
        self.assertEqual(results, [])
        mock_http.assert_not_called()

    # ── 1.3 privacy_pre_hook BLOCK prevents HTTP ─────────────────────────────

    def test_privacy_hook_block_prevents_http(self):
        """privacy_pre_hook returns BLOCK → _call_searxng never called."""
        with patch.dict(os.environ, {"SEARXNG_URL": "http://127.0.0.1:8888"}):
            hook_block = MagicMock()
            hook_block.decision = "BLOCK"
            with patch("tools.search.privacy_pre_hook", return_value=hook_block):
                with patch("tools.search._call_searxng") as mock_http:
                    from tools.search import search_web
                    results = search_web("private name query")
        self.assertEqual(results, [])
        mock_http.assert_not_called()

    # ── 1.4 Happy path: local URL + ALLOW → reaches SearXNG ─────────────────

    def test_local_url_allow_calls_searxng(self):
        """Local URL + ALLOW → _call_searxng is called exactly once."""
        with patch.dict(os.environ, {"SEARXNG_URL": "http://127.0.0.1:8888"}):
            hook_allow = MagicMock()
            hook_allow.decision = "ALLOW"
            hook_allow.redacted_payload = {"query": "test query"}
            with patch("tools.search.privacy_pre_hook", return_value=hook_allow):
                with patch("tools.search.text_is_safe", return_value=(True, [])):
                    with patch("tools.search._call_searxng", return_value=[]) as mock_http:
                        from tools.search import search_web
                        search_web("test query")
        mock_http.assert_called_once()

    def test_privacy_hook_called_before_http(self):
        """privacy_pre_hook must be invoked before _call_searxng (call order check)."""
        call_order = []
        with patch.dict(os.environ, {"SEARXNG_URL": "http://127.0.0.1:8888"}):
            hook_allow = MagicMock()
            hook_allow.decision = "ALLOW"
            hook_allow.redacted_payload = {"query": "safe query"}

            def record_hook(*args, **kwargs):
                call_order.append("hook")
                return hook_allow

            def record_http(*args, **kwargs):
                call_order.append("http")
                return []

            with patch("tools.search.privacy_pre_hook", side_effect=record_hook):
                with patch("tools.search.text_is_safe", return_value=(True, [])):
                    with patch("tools.search._call_searxng", side_effect=record_http):
                        from tools.search import search_web
                        search_web("safe query")

        self.assertIn("hook", call_order)
        self.assertIn("http", call_order)
        self.assertLess(
            call_order.index("hook"), call_order.index("http"),
            "privacy_pre_hook must be called BEFORE _call_searxng",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. library_fetch — score gate + SSRF guard (no network on rejection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLibraryFetchScoreGate(unittest.TestCase):
    """Low-score requests must short-circuit before any network call."""

    def test_low_score_returns_threshold_no_network(self):
        """score < 0.6 → SCORE_BELOW_THRESHOLD, _do_download never called."""
        with patch("tools.library_fetch._do_download") as mock_dl:
            from tools.library_fetch import fetch_library_document
            result = fetch_library_document(
                doi_or_url="https://doi.org/10.1/test",
                credibility_score=0.1,
                relevance_score=0.1,
                open_pdf_url="https://arxiv.org/pdf/test.pdf",
            )
        self.assertEqual(result.status, "SCORE_BELOW_THRESHOLD")
        mock_dl.assert_not_called()

    def test_low_score_provides_unlocking_question(self):
        """SCORE_BELOW_THRESHOLD response includes an unlocking_question."""
        from tools.library_fetch import fetch_library_document
        result = fetch_library_document(
            doi_or_url="https://doi.org/10.1/test",
            credibility_score=0.2,
            relevance_score=0.3,
        )
        self.assertEqual(result.status, "SCORE_BELOW_THRESHOLD")
        self.assertIsNotNone(result.unlocking_question)
        self.assertGreater(len(result.unlocking_question), 10)

    def test_score_exactly_at_threshold_passes_gate(self):
        """score == 0.6 → passes score gate (reaches cache check)."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_ws = MagicMock()
            cache_dir = Path(tmp) / "lib"
            cache_dir.mkdir()
            fake_ws.library_cache_path.return_value = cache_dir
            from tools.library_fetch import fetch_library_document
            result = fetch_library_document(
                doi_or_url="https://doi.org/10.1/at-threshold",
                credibility_score=0.6,
                relevance_score=0.6,
                workspace_state=fake_ws,
            )
        self.assertNotEqual(result.status, "SCORE_BELOW_THRESHOLD")

    def test_score_just_below_threshold_blocked(self):
        """score = 0.59 → SCORE_BELOW_THRESHOLD (boundary condition)."""
        from tools.library_fetch import fetch_library_document
        result = fetch_library_document(
            doi_or_url="https://doi.org/10.1/boundary",
            credibility_score=0.58,
            relevance_score=0.60,
        )
        # (0.58 + 0.60) / 2 = 0.59 < 0.60
        self.assertEqual(result.status, "SCORE_BELOW_THRESHOLD")

    def test_low_score_writes_audit_block(self):
        """Score gate fires → LIBRARY_FETCH BLOCK is written to audit."""
        with patch("tools.library_fetch._lib_audit") as mock_audit:
            from tools.library_fetch import fetch_library_document
            fetch_library_document(
                doi_or_url="https://doi.org/10.1/test",
                credibility_score=0.1,
                relevance_score=0.1,
            )
        mock_audit.assert_called_once()
        args = mock_audit.call_args[0]
        self.assertEqual(args[0], "BLOCK")
        self.assertIn("score_below_threshold", args[2])


class TestLibraryFetchSSRFGuard(unittest.TestCase):
    """SSRF guard blocks private/loopback PDF URLs before download."""

    def _fetch_with_url(self, pdf_url: str):
        with patch("tools.library_fetch._do_download") as mock_dl:
            from tools.library_fetch import fetch_library_document
            result = fetch_library_document(
                doi_or_url="https://arxiv.org/abs/2301.12345",
                credibility_score=0.9,
                relevance_score=0.9,
                open_pdf_url=pdf_url,
            )
        return result, mock_dl

    def test_private_ip_blocked_no_download(self):
        result, mock_dl = self._fetch_with_url("http://192.168.1.1/secret.pdf")
        self.assertEqual(result.status, "BLOCKED")
        mock_dl.assert_not_called()

    def test_loopback_blocked_no_download(self):
        result, mock_dl = self._fetch_with_url("http://127.0.0.1/internal.pdf")
        self.assertEqual(result.status, "BLOCKED")
        mock_dl.assert_not_called()

    def test_docker_private_range_blocked(self):
        result, mock_dl = self._fetch_with_url("http://172.17.0.1/data.pdf")
        self.assertEqual(result.status, "BLOCKED")
        mock_dl.assert_not_called()

    def test_ssrf_block_writes_audit(self):
        with patch("tools.library_fetch._lib_audit") as mock_audit:
            from tools.library_fetch import fetch_library_document
            fetch_library_document(
                doi_or_url="https://arxiv.org/abs/2301.12345",
                credibility_score=0.9,
                relevance_score=0.9,
                open_pdf_url="http://192.168.1.1/secret.pdf",
            )
        mock_audit.assert_called_once()
        args = mock_audit.call_args[0]
        self.assertEqual(args[0], "BLOCK")
        self.assertEqual(args[2], "ssrf_guard")

    def test_public_arxiv_url_passes_ssrf(self):
        """Public arXiv URL must pass SSRF guard (download attempted)."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_ws = MagicMock()
            cache_dir = Path(tmp) / "lib"
            cache_dir.mkdir()
            fake_ws.library_cache_path.return_value = cache_dir
            with patch("tools.library_fetch._do_download") as mock_dl:
                mock_dl.side_effect = Exception("mocked — no real network")
                from tools.library_fetch import fetch_library_document
                result = fetch_library_document(
                    doi_or_url="https://arxiv.org/abs/2301.12345",
                    credibility_score=0.9,
                    relevance_score=0.9,
                    open_pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
                    workspace_state=fake_ws,
                )
            # SSRF guard passed → download was attempted (even though it failed)
            mock_dl.assert_called_once()
            self.assertIn(result.status, ("NEEDS_USER_LOGIN", "DOWNLOADED"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Audit log integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditLogIntegrity(unittest.TestCase):
    """Audit entries must contain payload_hash, never raw payload text."""

    def _write_entry(self, payload: dict) -> dict:
        """Write one audit entry to a temp file and return the parsed result."""
        import core.audit_append as _aud
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp)
            real_file = real_dir / "audit.jsonl"
            orig_dir, orig_file = _aud._AUDIT_DIR, _aud._AUDIT_FILE
            try:
                _aud._AUDIT_DIR = real_dir
                _aud._AUDIT_FILE = real_file
                h = _aud._payload_hash(payload)
                _aud.append_audit(
                    action_type="SEARCH_WEB",
                    decision="ALLOW",
                    pii_detected=[],
                    redaction_notes=[],
                    payload_hash=h,
                    reason="test",
                )
            finally:
                _aud._AUDIT_DIR = orig_dir
                _aud._AUDIT_FILE = orig_file
            with open(real_file) as f:
                return json.loads(f.readline())

    def test_audit_contains_payload_hash(self):
        entry = self._write_entry({"query": "some query"})
        self.assertIn("payload_hash", entry)
        self.assertIsInstance(entry["payload_hash"], str)
        self.assertGreater(len(entry["payload_hash"]), 4)

    def test_audit_no_raw_payload_fields(self):
        """Audit entries must NOT store raw query/payload/text/content keys."""
        entry = self._write_entry({"query": "sensitive data here"})
        raw_keys = {"query", "payload", "text", "content", "raw"}
        overlap = raw_keys & set(entry.keys())
        self.assertEqual(overlap, set(), f"Raw payload keys leaked into audit: {overlap}")

    def test_audit_has_all_required_fields(self):
        entry = self._write_entry({"query": "test"})
        required = ("timestamp", "action_type", "decision", "payload_hash", "reason")
        for field in required:
            self.assertIn(field, entry, f"Missing required audit field: {field}")

    def test_payload_hash_is_deterministic(self):
        from core.audit_append import _payload_hash
        h1 = _payload_hash({"query": "hello world"})
        h2 = _payload_hash({"query": "hello world"})
        self.assertEqual(h1, h2)

    def test_payload_hash_differs_for_different_inputs(self):
        from core.audit_append import _payload_hash
        h1 = _payload_hash({"query": "hello world"})
        h2 = _payload_hash({"query": "different content"})
        self.assertNotEqual(h1, h2)

    def test_payload_hash_length_fixed(self):
        """Hash is always 16 hex chars (truncated SHA-256)."""
        from core.audit_append import _payload_hash
        h = _payload_hash({"query": "anything"})
        self.assertEqual(len(h), 16)
        self.assertRegex(h, r"^[0-9a-f]{16}$")

    def test_audit_is_valid_jsonl(self):
        """Each line written by append_audit must be valid JSON."""
        import core.audit_append as _aud
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp)
            real_file = real_dir / "audit.jsonl"
            orig_dir, orig_file = _aud._AUDIT_DIR, _aud._AUDIT_FILE
            try:
                _aud._AUDIT_DIR = real_dir
                _aud._AUDIT_FILE = real_file
                for i in range(3):
                    _aud.append_audit(
                        action_type="SEARCH_WEB",
                        decision="ALLOW",
                        pii_detected=[],
                        redaction_notes=[],
                        payload_hash=_aud._payload_hash({"n": i}),
                        reason=f"test_{i}",
                    )
            finally:
                _aud._AUDIT_DIR = orig_dir
                _aud._AUDIT_FILE = orig_file
            with open(real_file) as f:
                lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 3)
        for line in lines:
            parsed = json.loads(line)  # must not raise
            self.assertIsInstance(parsed, dict)

    def test_no_searxng_url_triggers_audit_block(self):
        """search_web with no SEARXNG_URL → _audit_block is called, no HTTP."""
        env = {k: v for k, v in os.environ.items() if k != "SEARXNG_URL"}
        with patch.dict(os.environ, env, clear=True):
            with patch("tools.search._call_searxng") as mock_http:
                with patch("tools.search._audit_block") as mock_audit:
                    from tools.search import search_web
                    results = search_web("some query")
        mock_http.assert_not_called()
        self.assertEqual(results, [])
        mock_audit.assert_called_once()
        # The reason string must mention SEARXNG_URL, not the raw query
        reason = mock_audit.call_args[0][0]
        self.assertNotIn("some query", reason)
        self.assertIn("SEARXNG_URL", reason)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. literature.py — privacy gate before fan-out
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiteraturePrivacyGate(unittest.TestCase):
    """search_literature must route through privacy_pre_hook before any source call."""

    def test_block_raises_literature_blocked_error(self):
        """privacy_pre_hook BLOCK → LiteratureBlockedError raised, no source calls."""
        from tools.literature import LiteratureBlockedError
        block_result = {"decision": "BLOCK", "reason": "PII detected"}
        with patch("tools.literature.privacy_pre_hook", return_value=block_result):
            with patch("tools.literature._SOURCE_FETCHERS", {"fake": MagicMock()}) as mock_src:
                from tools.literature import search_literature
                with self.assertRaises(LiteratureBlockedError):
                    search_literature("private name query", sources=["fake"])

    def test_redact_and_proceed_uses_sanitized_query(self):
        """REDACT_AND_PROCEED → sources receive the sanitized query, not the original."""
        received = []

        def fake_source(query, max_results):
            received.append(query)
            return []

        redact_result = {
            "decision": "REDACT_AND_PROCEED",
            "sanitized_payload": {"query": "[REDACTED]"},
        }
        with patch("tools.literature.privacy_pre_hook", return_value=redact_result):
            with patch("tools.literature._SOURCE_FETCHERS", {"fake": fake_source}):
                from tools.literature import search_literature
                search_literature("John Smith private query", sources=["fake"])

        self.assertTrue(len(received) > 0, "Source was never called")
        for q in received:
            self.assertNotIn("John Smith", q, "Raw PII found in query sent to source")
            self.assertIn("REDACTED", q)

    def test_allow_proceeds_to_sources(self):
        """privacy_pre_hook ALLOW → sources are called."""
        source_called = []

        def fake_source(query, max_results):
            source_called.append(query)
            return []

        allow_result = {"decision": "ALLOW"}
        with patch("tools.literature.privacy_pre_hook", return_value=allow_result):
            with patch("tools.literature._SOURCE_FETCHERS", {"fake": fake_source}):
                from tools.literature import search_literature
                search_literature("safe query", sources=["fake"])

        self.assertTrue(len(source_called) > 0, "Source was never called on ALLOW")

    def test_privacy_hook_called_before_sources(self):
        """Hook must be invoked before any source fetch (call-order invariant)."""
        call_order = []

        def hook_fn(*args, **kwargs):
            call_order.append("hook")
            return {"decision": "ALLOW"}

        def fake_source(query, max_results):
            call_order.append("source")
            return []

        with patch("tools.literature.privacy_pre_hook", side_effect=hook_fn):
            with patch("tools.literature._SOURCE_FETCHERS", {"fake": fake_source}):
                from tools.literature import search_literature
                search_literature("test", sources=["fake"])

        self.assertIn("hook", call_order)
        self.assertIn("source", call_order)
        self.assertLess(
            call_order.index("hook"), call_order.index("source"),
            "privacy_pre_hook must run BEFORE source fetches",
        )


if __name__ == "__main__":
    unittest.main()
