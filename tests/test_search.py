#!/usr/bin/env python3
"""
test_search.py
Proves that:
  - DDG is permanently disabled
  - No internet egress occurs without a valid local SEARXNG_URL
  - Privacy pre-hook gates every query before any network call
  - Public SEARXNG_URL is blocked
  - PII queries are blocked before requests.get() is called
  - Safe queries reach SearXNG (mocked)

Usage:
    python3 -m unittest tests.test_search -v
    # or
    python3 tests/test_search.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.search import (
    DDGDisabledError,
    SearchResult,
    _is_local_url,
    search_ddg,
    search_news_ddg,
    search_web,
    run_queries,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tmp_audit(tmp_dir: str) -> Path:
    p = Path(tmp_dir) / "audit_logs" / "privacy_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _fake_searxng_response(query: str = "test") -> MagicMock:
    """Mock requests.get() returning a valid SearXNG JSON body."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {
                "title": "Best practices for n8n error handling",
                "url": "https://docs.n8n.io/error-handling",
                "content": "How to handle errors in n8n workflows effectively.",
                "engine": "google",
            }
        ]
    }
    return mock_resp


# ── DDG disabled ─────────────────────────────────────────────────────────────

class TestDDGDisabled(unittest.TestCase):

    def test_search_ddg_raises(self):
        """search_ddg() must always raise DDGDisabledError — no network call possible."""
        with self.assertRaises(DDGDisabledError):
            search_ddg("industrial AI trends")

    def test_search_news_ddg_raises(self):
        """search_news_ddg() must always raise DDGDisabledError."""
        with self.assertRaises(DDGDisabledError):
            search_news_ddg("venture capital news")

    def test_ddg_module_not_imported(self):
        """duckduckgo_search must not be imported anywhere in tools.search."""
        import tools.search as search_module
        self.assertFalse(
            hasattr(search_module, "DDGS"),
            "DDGS symbol must not be present in tools.search",
        )


# ── Network guardrails ────────────────────────────────────────────────────────

class TestIsLocalUrl(unittest.TestCase):

    def test_localhost_allowed(self):
        self.assertTrue(_is_local_url("http://localhost:8080"))

    def test_127_allowed(self):
        self.assertTrue(_is_local_url("http://127.0.0.1:8888"))

    def test_searxng_hostname_allowed(self):
        self.assertTrue(_is_local_url("http://searxng:8080"))

    def test_docker_internal_allowed(self):
        self.assertTrue(_is_local_url("http://host.docker.internal:8080"))

    def test_rfc1918_10_allowed(self):
        self.assertTrue(_is_local_url("http://10.0.0.5:8080"))

    def test_rfc1918_172_allowed(self):
        self.assertTrue(_is_local_url("http://172.20.0.3:8080"))

    def test_rfc1918_192_allowed(self):
        self.assertTrue(_is_local_url("http://192.168.1.100:8080"))

    def test_public_ip_blocked(self):
        self.assertFalse(_is_local_url("http://8.8.8.8:8080"))

    def test_public_domain_blocked(self):
        self.assertFalse(_is_local_url("https://searxng.example.com"))

    def test_aws_ip_blocked(self):
        self.assertFalse(_is_local_url("http://54.23.100.5:8080"))

    def test_empty_url_blocked(self):
        self.assertFalse(_is_local_url(""))

    def test_malformed_url_blocked(self):
        self.assertFalse(_is_local_url("not_a_url"))


# ── SEARXNG_URL missing → BLOCK ───────────────────────────────────────────────

class TestMissingSearxngUrl(unittest.TestCase):

    def test_no_env_var_returns_empty(self):
        """If SEARXNG_URL is not set, search_web must return [] without any network call."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove SEARXNG_URL if present
            env = {k: v for k, v in __import__("os").environ.items() if k != "SEARXNG_URL"}
            with patch.dict("os.environ", env, clear=True):
                with patch("requests.get") as mock_get:
                    with tempfile.TemporaryDirectory() as tmp:
                        audit = _tmp_audit(tmp)
                        with patch("core.audit_append._AUDIT_FILE", audit), \
                             patch("core.audit_append._AUDIT_DIR", audit.parent):
                            results = search_web("industrial AI best practices")
                    mock_get.assert_not_called()
                self.assertEqual(results, [])


# ── Public SEARXNG_URL → BLOCK ────────────────────────────────────────────────

class TestPublicSearxngUrl(unittest.TestCase):

    def test_public_url_blocked(self):
        """SEARXNG_URL pointing to a public domain must return [] without any network call."""
        with patch.dict("os.environ", {"SEARXNG_URL": "https://searx.be"}):
            with patch("requests.get") as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web("open source automation tools")
                mock_get.assert_not_called()
            self.assertEqual(results, [])

    def test_public_ip_url_blocked(self):
        """SEARXNG_URL with a public IP must be blocked."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://203.0.113.5:8080"}):
            with patch("requests.get") as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web("n8n workflow patterns")
                mock_get.assert_not_called()
            self.assertEqual(results, [])


# ── PII queries blocked before network ───────────────────────────────────────

class TestPIIQueryBlocked(unittest.TestCase):

    def _run(self, query: str) -> tuple[list, MagicMock]:
        with patch.dict("os.environ", {"SEARXNG_URL": "http://127.0.0.1:8080"}):
            with patch("requests.get") as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web(query)
                return results, mock_get

    def test_email_in_query_not_forwarded_raw(self):
        """Raw email address must never reach SearXNG — either BLOCK or redacted before call."""
        raw_email = "john.smith@privatecorp.io"
        results, mock_get = self._run(f"contact {raw_email} about industrial AI")
        if mock_get.called:
            # REDACT_AND_PROCEED path: verify the raw email is NOT in the forwarded query
            forwarded_query = mock_get.call_args[1].get("params", {}).get("q", "")
            self.assertNotIn(raw_email, forwarded_query,
                "Raw email address must not appear in the query sent to SearXNG")
        # Either path is acceptable; raw PII must not leave

    def test_ssn_in_query_blocked(self):
        """SSN is a hard-block kind — must BLOCK and never call requests.get()."""
        results, mock_get = self._run("what does 123-45-6789 map to")
        mock_get.assert_not_called()
        self.assertEqual(results, [])

    def test_phone_in_query_not_forwarded_raw(self):
        """Raw phone number must never reach SearXNG — either BLOCK or redacted before call."""
        raw_phone = "718-555-1234"
        results, mock_get = self._run(f"who owns phone {raw_phone} in New York")
        if mock_get.called:
            forwarded_query = mock_get.call_args[1].get("params", {}).get("q", "")
            self.assertNotIn(raw_phone, forwarded_query,
                "Raw phone number must not appear in the query sent to SearXNG")
        # Either BLOCK or REDACT_AND_PROCEED is acceptable; raw PII must not leave

    def test_money_in_query_varies(self):
        """Query with a money amount may be blocked or redacted — either way no raw PII leaves."""
        results, mock_get = self._run("contract worth $4,500,000 best practices")
        # Either blocked (no call) or redacted (call with clean query)
        if mock_get.called:
            call_args = mock_get.call_args
            forwarded_query = call_args[1].get("params", {}).get("q", "")
            self.assertNotIn("$4,500,000", forwarded_query)
        # No assertion on results — redacted proceed is also acceptable


# ── Safe query reaches SearXNG (mocked) ──────────────────────────────────────

class TestSafeQueryAllowed(unittest.TestCase):

    def test_benign_query_calls_searxng(self):
        """A PII-free query must call requests.get() exactly once against local SearXNG."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://127.0.0.1:8080"}):
            with patch("requests.get", return_value=_fake_searxng_response()) as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web("n8n http node error handling best practices")

                mock_get.assert_called_once()
                # Verify destination is local
                called_url = mock_get.call_args[0][0]
                self.assertIn("127.0.0.1", called_url)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)

    def test_results_have_required_fields(self):
        """Each returned SearchResult must have source_id, title, url, snippet."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://localhost:8080"}):
            with patch("requests.get", return_value=_fake_searxng_response()):
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web("industrial AI workflow automation")

        for r in results:
            self.assertTrue(r.source_id.startswith("src_"))
            self.assertIsInstance(r.title, str)
            self.assertIsInstance(r.url, str)
            self.assertIsInstance(r.snippet, str)

    def test_run_queries_uses_search_web(self):
        """run_queries must route through search_web, not DDG."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://127.0.0.1:8080"}):
            with patch("tools.search.search_web", return_value=[]) as mock_sw:
                run_queries(["n8n best practices", "workflow automation"])
                self.assertEqual(mock_sw.call_count, 2)

    def test_docker_hostname_allowed(self):
        """SEARXNG_URL with docker hostname 'searxng' must be accepted."""
        with patch.dict("os.environ", {"SEARXNG_URL": "http://searxng:8080"}):
            with patch("requests.get", return_value=_fake_searxng_response()) as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        results = search_web("predictive maintenance best practices")
                mock_get.assert_called_once()


# ── Exfil impossibility proof ─────────────────────────────────────────────────

class TestNoExfilPath(unittest.TestCase):

    def test_cannot_import_ddgs(self):
        """duckduckgo_search.DDGS must not be importable via tools.search namespace."""
        import tools.search as m
        self.assertFalse(
            any("duckduckgo" in str(v) for v in vars(m).values()),
            "duckduckgo symbol found in tools.search namespace",
        )

    def test_search_web_no_call_without_env(self):
        """search_web() with no SEARXNG_URL must never call requests.get."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "SEARXNG_URL"}
        with patch.dict("os.environ", env, clear=True):
            with patch("requests.get") as mock_get:
                with tempfile.TemporaryDirectory() as tmp:
                    audit = _tmp_audit(tmp)
                    with patch("core.audit_append._AUDIT_FILE", audit), \
                         patch("core.audit_append._AUDIT_DIR", audit.parent):
                        search_web("anything")
                mock_get.assert_not_called()

    def test_audit_written_on_block(self):
        """Every BLOCK decision must produce an audit entry."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "SEARXNG_URL"}
        with patch.dict("os.environ", env, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                audit = _tmp_audit(tmp)
                with patch("core.audit_append._AUDIT_FILE", audit), \
                     patch("core.audit_append._AUDIT_DIR", audit.parent):
                    search_web("test query without SEARXNG_URL")
                lines = [l for l in audit.read_text().splitlines() if l.strip()]
                self.assertGreater(len(lines), 0)
                entry = json.loads(lines[0])
                self.assertEqual(entry["decision"], "BLOCK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
