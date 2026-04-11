"""Tests for tools/state_manager.py — no external dependencies."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.state_manager import (
    WorkspaceState, DuplicateError, StateWriteError,
    fingerprint_opportunity, fingerprint_task, fingerprint_evidence,
)


def _ws(tmp):
    return WorkspaceState(base_dir=tmp)


class TestFingerprints(unittest.TestCase):
    def test_opportunity_deterministic(self):
        fp1 = fingerprint_opportunity("ORG", "Acme", "Grant X", "https://acme.com")
        fp2 = fingerprint_opportunity("ORG", "Acme", "Grant X", "https://acme.com")
        self.assertEqual(fp1, fp2)

    def test_opportunity_case_insensitive(self):
        fp1 = fingerprint_opportunity("ORG", "Acme Corp", "Grant X", "https://x.com")
        fp2 = fingerprint_opportunity("ORG", "ACME CORP", "GRANT X", "https://x.com")
        self.assertEqual(fp1, fp2)

    def test_task_fingerprint_deterministic(self):
        fp1 = fingerprint_task("DISCOVER", {"entity_id": None, "query_hint": "bio"})
        fp2 = fingerprint_task("DISCOVER", {"entity_id": None, "query_hint": "bio"})
        self.assertEqual(fp1, fp2)

    def test_evidence_fingerprint_trailing_slash_normalized(self):
        fp1 = fingerprint_evidence("https://x.com/p", "Paper")
        fp2 = fingerprint_evidence("https://x.com/p/", "Paper")
        self.assertEqual(fp1, fp2)


class TestAtomicWrite(unittest.TestCase):
    def test_atomic_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            p = s.STATE_DIR / "test.json"
            s._write_json_atomic(p, {"k": "v"})
            self.assertEqual(json.loads(p.read_text())["k"], "v")

    def test_no_tmp_file_left_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            p = s.STATE_DIR / "test2.json"
            s._write_json_atomic(p, {"k": "v"})
            self.assertFalse(p.with_suffix(".json.tmp").exists())


class TestDedup(unittest.TestCase):
    def test_register_opportunity_blocks_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            fp = fingerprint_opportunity("ORG", "Co", "T", "https://co.com")
            s.register_opportunity(fp)
            with self.assertRaises(DuplicateError):
                s.register_opportunity(fp)

    def test_is_duplicate_false_before_register(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            fp = fingerprint_opportunity("ORG", "New", "T", "https://new.co")
            self.assertFalse(s.is_duplicate_opportunity(fp))

    def test_register_task_blocks_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            fp = fingerprint_task("DISCOVER", {"entity_id": None, "query_hint": "test"})
            s.register_task(fp, "task_aaa")
            with self.assertRaises(DuplicateError):
                s.register_task(fp, "task_bbb")

    def test_register_evidence_blocks_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            fp = fingerprint_evidence("https://ex.com", "Paper")
            s.register_evidence(fp, "ev_aaa")
            with self.assertRaises(DuplicateError):
                s.register_evidence(fp, "ev_bbb")


class TestQueue(unittest.TestCase):
    def _task(self, hint, priority=4):
        return {"strategy": "DISCOVER", "priority": priority,
                "payload": {"entity_id": None, "query_hint": hint, "hypothesis_ids": None}}

    def test_enqueue_assigns_task_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            tid = s.enqueue_task(self._task("bio"))
            self.assertTrue(tid.startswith("task_"))

    def test_enqueue_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.enqueue_task(self._task("bio"))
            with self.assertRaises(DuplicateError):
                s.enqueue_task(self._task("bio"))

    def test_peek_respects_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.enqueue_task(self._task("low", priority=4))
            s.enqueue_task({"strategy": "VALIDATE", "priority": 1,
                            "payload": {"entity_id": "ent_x", "query_hint": None, "hypothesis_ids": None}})
            tasks = s.peek_next_tasks(n=2)
            self.assertEqual(tasks[0]["priority"], 1)

    def test_mark_task_status_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            tid = s.enqueue_task(self._task("test"))
            s.mark_task_status(tid, "IN_PROGRESS")
            in_prog = s.read_queue(status_filter=["IN_PROGRESS"])
            self.assertEqual(len(in_prog), 1)
            self.assertEqual(in_prog[0]["task_id"], tid)


class TestEvidence(unittest.TestCase):
    def _rec(self, url, title="P"):
        return {"type": "web", "title": title, "url": url,
                "source_id": "s1", "snippet": "x",
                "credibility_score": 0.5, "relevance_score": 0.5,
                "hypothesis_ids": [], "tags": []}

    def test_append_returns_ev_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            eid = s.append_evidence(self._rec("https://x.com/1"))
            self.assertTrue(eid.startswith("ev_"))

    def test_append_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.append_evidence(self._rec("https://x.com/d", "Dup"))
            with self.assertRaises(DuplicateError):
                s.append_evidence(self._rec("https://x.com/d", "Dup"))

    def test_read_evidence_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.append_evidence(self._rec("https://a.com/1", "P1"))
            s.append_evidence(self._rec("https://a.com/2", "P2"))
            self.assertEqual(len(s.read_evidence()), 2)


class TestStrategyState(unittest.TestCase):
    def test_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            st = s.load_strategy_state()
            self.assertEqual(st["run_counter"], 0)
            self.assertEqual(st["hypotheses"], [])

    def test_update_merges_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.update_strategy_state({"run_counter": 5, "consecutive_empty_runs": 2})
            st = s.load_strategy_state()
            self.assertEqual(st["run_counter"], 5)
            self.assertEqual(st["consecutive_empty_runs"], 2)

    def test_hypotheses_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            h = {"id": "hyp_001", "statement": "test", "confidence": 0.5,
                 "falsifiers": [], "evidence_ids": [], "status": "ACTIVE"}
            s.update_strategy_state({"hypotheses": [h]})
            h2 = dict(h, confidence=0.8)
            s.update_strategy_state({"hypotheses": [h2]})
            st = s.load_strategy_state()
            self.assertEqual(len(st["hypotheses"]), 1)
            self.assertEqual(st["hypotheses"][0]["confidence"], 0.8)


class TestDossiers(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            s.save_dossier({"entity_id": "ent_abc", "name": "TestOrg",
                            "status": "DRAFT", "type": "ORG"})
            d = s.load_dossier("ent_abc")
            self.assertEqual(d["name"], "TestOrg")

    def test_missing_dossier_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_ws(tmp).load_dossier("ent_ghost"))


class TestConcurrency(unittest.TestCase):
    def test_concurrent_evidence_appends_no_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _ws(tmp)
            errors = []

            def append(i):
                try:
                    s.append_evidence({"type": "web", "title": f"P{i}",
                                       "url": f"https://c.com/{i}", "source_id": f"s{i}",
                                       "snippet": "x", "credibility_score": 0.5,
                                       "relevance_score": 0.5,
                                       "hypothesis_ids": [], "tags": []})
                except DuplicateError:
                    pass
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=append, args=(i,)) for i in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()
            self.assertEqual(errors, [])
            for r in s.read_evidence():
                self.assertIn("title", r)


class TestStateSummary(unittest.TestCase):
    def test_summary_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = _ws(tmp).get_state_summary()
            for key in ("run_counter", "queue_summary", "hypothesis_summary",
                        "dossier_summary", "pending_approvals"):
                self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
