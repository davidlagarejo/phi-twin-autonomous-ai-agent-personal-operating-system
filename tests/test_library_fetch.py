"""Tests for tools/library_fetch.py — no external HTTP calls."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.library_fetch import (
    fetch_library_document, _normalize_doc_id, _is_safe_external_url,
    LIBRARY_FETCH_MIN_SCORE,
)


class TestNormalizeDocId(unittest.TestCase):
    def test_doi_prefix_stripped(self):
        self.assertTrue(_normalize_doc_id("https://doi.org/10.1000/test.paper").startswith("doi_"))

    def test_doi_no_slashes(self):
        doc_id = _normalize_doc_id("https://doi.org/10.1000/test/paper")
        self.assertNotIn("/", doc_id)

    def test_arxiv_url(self):
        self.assertTrue(_normalize_doc_id("https://arxiv.org/abs/2301.12345").startswith("arxiv_"))

    def test_pubmed_url(self):
        self.assertTrue(_normalize_doc_id("https://pubmed.ncbi.nlm.nih.gov/12345678/").startswith("pmid_"))

    def test_unknown_uses_hash(self):
        self.assertTrue(_normalize_doc_id("https://unknown.com/paper?x=1").startswith("url_"))

    def test_filesystem_safe(self):
        for url in ["https://doi.org/10.1000/test", "https://arxiv.org/abs/2301.12345v1"]:
            doc_id = _normalize_doc_id(url)
            self.assertFalse(any(c in doc_id for c in '/\\:*?"<>|'))


class TestSSRFGuard(unittest.TestCase):
    def test_localhost_blocked(self):
        self.assertFalse(_is_safe_external_url("http://localhost/evil"))
        self.assertFalse(_is_safe_external_url("http://127.0.0.1/evil"))

    def test_private_ip_blocked(self):
        self.assertFalse(_is_safe_external_url("http://192.168.1.1/paper.pdf"))
        self.assertFalse(_is_safe_external_url("http://10.0.0.1/paper.pdf"))

    def test_public_url_allowed(self):
        self.assertTrue(_is_safe_external_url("https://arxiv.org/pdf/2301.pdf"))

    def test_non_http_blocked(self):
        self.assertFalse(_is_safe_external_url("ftp://files.example.com/paper.pdf"))


class TestScoreGate(unittest.TestCase):
    def test_below_threshold(self):
        r = fetch_library_document("https://doi.org/10.1/low",
                                   credibility_score=0.1, relevance_score=0.1)
        self.assertEqual(r.status, "SCORE_BELOW_THRESHOLD")
        self.assertIsNotNone(r.unlocking_question)

    def test_at_threshold_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tools.state_manager import WorkspaceState
            ws = WorkspaceState(base_dir=tmp)
            r = fetch_library_document("https://doi.org/10.1/ok",
                                       credibility_score=0.6, relevance_score=0.6,
                                       open_pdf_url=None, workspace_state=ws)
            self.assertNotEqual(r.status, "SCORE_BELOW_THRESHOLD")


class TestCacheHit(unittest.TestCase):
    def test_already_cached_returns_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tools.state_manager import WorkspaceState
            ws = WorkspaceState(base_dir=tmp)
            doc_id = _normalize_doc_id("https://doi.org/10.1/cached")
            cache_dir = ws.library_cache_path(doc_id)
            dummy = cache_dir / "paper.pdf"
            dummy.write_bytes(b"%PDF-1.4 " + b"x" * 200)
            r = fetch_library_document("https://doi.org/10.1/cached",
                                       credibility_score=0.9, relevance_score=0.9,
                                       workspace_state=ws)
            self.assertEqual(r.status, "CACHED")
            self.assertEqual(r.local_path, str(dummy))


class TestNoOpenPDF(unittest.TestCase):
    def test_no_pdf_url_returns_needs_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tools.state_manager import WorkspaceState
            ws = WorkspaceState(base_dir=tmp)
            r = fetch_library_document("https://doi.org/10.1/paywall",
                                       credibility_score=0.9, relevance_score=0.9,
                                       open_pdf_url=None, workspace_state=ws)
            self.assertEqual(r.status, "NEEDS_USER_LOGIN")
            self.assertIn("log in", r.unlocking_question.lower())


class TestPrivacyBlock(unittest.TestCase):
    def test_privacy_gate_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tools.state_manager import WorkspaceState
            ws = WorkspaceState(base_dir=tmp)
            with patch("tools.library_fetch.privacy_pre_hook",
                       return_value={"decision": "BLOCK", "reason": "PII"}):
                r = fetch_library_document("https://doi.org/10.1/pii",
                                           credibility_score=0.9, relevance_score=0.9,
                                           open_pdf_url="https://arxiv.org/pdf/test.pdf",
                                           workspace_state=ws)
                self.assertEqual(r.status, "BLOCKED")


class TestOpenPDFDownload(unittest.TestCase):
    def test_download_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            from tools.state_manager import WorkspaceState
            ws = WorkspaceState(base_dir=tmp)
            fake_pdf = b"%PDF-1.4 " + b"x" * 2000
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_pdf
            mock_resp.headers = {"Content-Type": "application/pdf"}
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            with patch("tools.library_fetch.privacy_pre_hook",
                       return_value={"decision": "ALLOW"}):
                with patch("urllib.request.urlopen", return_value=mock_resp):
                    r = fetch_library_document(
                        "https://arxiv.org/abs/2301.12345",
                        credibility_score=0.9, relevance_score=0.9,
                        open_pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
                        workspace_state=ws,
                    )
            self.assertEqual(r.status, "DOWNLOADED")
            self.assertIsNotNone(r.local_path)


if __name__ == "__main__":
    unittest.main()
