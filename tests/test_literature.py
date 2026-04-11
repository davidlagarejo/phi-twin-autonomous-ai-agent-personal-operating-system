"""Tests for tools/literature.py — mocked HTTP, no external calls."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_result(doi=None, source_api="openalex", title="Test Paper", open_pdf=None):
    from tools.literature import LiteratureResult
    return LiteratureResult(
        source_id="lit_test", source_api=source_api, title=title,
        authors=["Smith J"], year=2024, venue="Journal", doi=doi,
        url="https://example.com", open_pdf_url=open_pdf,
        abstract="Energy and sustainability research.", citations_count=10,
        relevance_score=0.0, credibility_score=0.0,
    )


class TestScoring(unittest.TestCase):
    def test_credibility_doi_is_1_0(self):
        from tools.literature import _credibility_score
        self.assertEqual(_credibility_score("10.1/test", "openalex", None), 1.0)

    def test_credibility_arxiv_no_doi_is_0_7(self):
        from tools.literature import _credibility_score
        self.assertEqual(_credibility_score(None, "arxiv", "https://arxiv.org/pdf/x.pdf"), 0.7)

    def test_credibility_no_doi_no_pdf_is_0_5(self):
        from tools.literature import _credibility_score
        self.assertEqual(_credibility_score(None, "openalex", None), 0.5)

    def test_relevance_zero_for_unrelated(self):
        from tools.literature import _relevance_score
        score = _relevance_score("quantum computing photonics", "Gardening tips", "How to grow roses.")
        self.assertEqual(score, 0.0)

    def test_relevance_nonzero_for_overlap(self):
        from tools.literature import _relevance_score
        score = _relevance_score("energy sustainability", "Renewable energy sustainability study", "")
        self.assertGreater(score, 0.0)


class TestDedup(unittest.TestCase):
    def test_keeps_higher_credibility_for_same_doi(self):
        from tools.literature import _dedup_by_doi_then_title
        r1 = _make_result(doi="10.1/x", source_api="crossref")
        r1.credibility_score = 1.0
        r2 = _make_result(doi="10.1/x", source_api="semantic_scholar")
        r2.credibility_score = 0.7
        deduped = _dedup_by_doi_then_title([r2, r1])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].credibility_score, 1.0)

    def test_dedup_by_title_when_no_doi(self):
        from tools.literature import _dedup_by_doi_then_title
        r1 = _make_result(doi=None, title="Same Title Study")
        r2 = _make_result(doi=None, title="Same Title Study")
        r1.credibility_score = 0.5
        r2.credibility_score = 0.7
        deduped = _dedup_by_doi_then_title([r1, r2])
        self.assertEqual(len(deduped), 1)


class TestOutputFormats(unittest.TestCase):
    def test_results_to_evidence_records_shape(self):
        from tools.literature import results_to_evidence_records
        r = _make_result(doi="10.1/r1")
        r.relevance_score = 0.8
        r.credibility_score = 1.0
        records = results_to_evidence_records([r], hypothesis_ids=["hyp_001"])
        self.assertEqual(len(records), 1)
        rec = records[0]
        for f in ("type", "title", "url", "source_id", "snippet",
                  "credibility_score", "relevance_score", "hypothesis_ids", "tags"):
            self.assertIn(f, rec)
        self.assertEqual(rec["hypothesis_ids"], ["hyp_001"])

    def test_format_for_prompt_non_empty(self):
        from tools.literature import format_literature_for_prompt
        r = _make_result(doi="10.1/fmt")
        r.relevance_score = 0.5
        r.credibility_score = 1.0
        text = format_literature_for_prompt([r])
        self.assertIn("LITERATURE SEARCH RESULTS", text)
        self.assertIn("source_id=", text)

    def test_format_empty_list_returns_empty_string(self):
        from tools.literature import format_literature_for_prompt
        self.assertEqual(format_literature_for_prompt([]), "")


class TestPrivacyGate(unittest.TestCase):
    def test_blocked_raises_literature_blocked_error(self):
        from tools.literature import search_literature, LiteratureBlockedError
        with patch("tools.literature.privacy_pre_hook",
                   return_value={"decision": "BLOCK", "reason": "PII"}):
            with self.assertRaises(LiteratureBlockedError):
                search_literature("private name query", _policy={})

    def test_redact_and_proceed_uses_sanitized_query(self):
        from tools.literature import search_literature
        with patch("tools.literature.privacy_pre_hook", return_value={
            "decision": "REDACT_AND_PROCEED",
            "sanitized_payload": {"query": "clean query"},
        }):
            with patch("tools.literature._fetch_openalex", return_value=[]) as mock_oa:
                with patch("tools.literature._fetch_crossref", return_value=[]):
                    with patch("tools.literature._fetch_semantic_scholar", return_value=[]):
                        with patch("tools.literature._fetch_arxiv", return_value=[]):
                            with patch("tools.literature._fetch_pubmed", return_value=[]):
                                search_literature("private name query", _policy={})
                                if mock_oa.call_args:
                                    self.assertEqual(mock_oa.call_args[0][0], "clean query")


class TestSourceFailureIsolation(unittest.TestCase):
    def test_one_source_failure_does_not_abort(self):
        from tools.literature import search_literature, LiteratureAPIError, LiteratureResult
        good = _make_result(doi="10.1/good", title="Good Paper")
        good.credibility_score = 1.0
        good.relevance_score = 0.5

        def fail_oa(*a, **kw): raise LiteratureAPIError("openalex", "timeout")
        def ok_cr(*a, **kw): return [good]

        with patch("tools.literature.privacy_pre_hook", return_value={"decision": "ALLOW"}):
            with patch("tools.literature._fetch_openalex", side_effect=fail_oa):
                with patch("tools.literature._fetch_crossref", side_effect=ok_cr):
                    with patch("tools.literature._fetch_semantic_scholar", return_value=[]):
                        with patch("tools.literature._fetch_arxiv", return_value=[]):
                            with patch("tools.literature._fetch_pubmed", return_value=[]):
                                results = search_literature("energy", min_relevance=0.0)
                                self.assertGreaterEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
