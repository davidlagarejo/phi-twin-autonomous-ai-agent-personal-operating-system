"""Tests for tools/research_engine.py — mocked LLM and search."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.state_manager import WorkspaceState, DuplicateError, fingerprint_opportunity
from tools.research_engine import (
    RunBudget, execute_research_cycle,
    _run_discover, _run_due_diligence, _run_correlate, _run_validate,
    _make_checkpoint,
)


def _ws(tmp):
    return WorkspaceState(base_dir=tmp)


def _task(strategy, hint=None, entity_id=None, priority=4):
    return {
        "task_id": f"task_{strategy[:4].lower()}_{hint or entity_id or 'x'}",
        "strategy": strategy, "priority": priority,
        "payload": {"entity_id": entity_id, "query_hint": hint, "hypothesis_ids": None},
    }


EMPTY_ENTITIES = '[]'
GOOD_ENTITIES = json.dumps([
    {"name": "GreenCo SA", "type": "ORG", "description": "Cleantech startup", "signal": "funding"}
])
GOOD_DOSSIER = json.dumps({
    "name": "TestCo", "description": "A test company.",
    "profile": {"website": None, "country": "ES", "sector": "cleantech",
                "size_signal": "seed", "funding_stage": "pre-seed", "key_people": []},
    "fit_assessment": {"fit_score": 65, "profile_match": 60, "timing": 70,
                       "effort_vs_reward": 60, "risk": 40, "why_yes": ["good fit"], "why_not": []},
    "next_actions": [],
})
HIGH_FIT_DOSSIER = json.dumps({
    "fit_assessment": {"fit_score": 85, "why_yes": ["strong match"], "why_not": []},
    "description": "High fit.", "profile": {}, "next_actions": [],
})
GOOD_HYPOTHESES = json.dumps([
    {"id": "hyp_NEW", "statement": "Market X is growing fast",
     "confidence": 0.6, "falsifiers": ["no data", "declining"], "status": "ACTIVE"}
])
GOOD_VALIDATION = json.dumps({
    "validated": True, "fit_score_final": 75,
    "key_findings": ["Strong market [src_001]"],
    "risks": ["Competition"], "recommended_next_action": "Contact CEO",
    "draft_outreach": "We can help you achieve X with our technology.",
    "evidence_coverage": "high", "gate_failures": [],
})


class TestDiscover(unittest.TestCase):
    def test_enqueues_due_diligence_for_new_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("DISCOVER", hint="cleantech")
            with patch("tools.research_engine._call_phi_sync",
                       side_effect=['["q1","q2","q3"]', GOOD_ENTITIES]):
                with patch("tools.research_engine._safe_search", return_value=[
                    {"title": "GreenCo news", "snippet": "GreenCo raises money",
                     "url": "https://news.com/gc", "source_id": "src_001"}
                ]):
                    r = _run_discover(task, RunBudget(max_seconds=60, max_web_queries=10),
                                      ws, [0], time.monotonic())
            self.assertEqual(r.status, "DONE")
            queue = ws.read_queue(status_filter=["PENDING"])
            self.assertTrue(any(t["strategy"] == "DUE_DILIGENCE" for t in queue))

    def test_skips_duplicate_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            fp = fingerprint_opportunity("ORG", "greenco sa", "greenco sa", "")
            ws.register_opportunity(fp)
            task = _task("DISCOVER", hint="cleantech")
            with patch("tools.research_engine._call_phi_sync",
                       side_effect=['["q1"]', GOOD_ENTITIES]):
                with patch("tools.research_engine._safe_search", return_value=[
                    {"title": "t", "snippet": "s", "url": "https://x.com", "source_id": "s1"}
                ]):
                    _run_discover(task, RunBudget(max_seconds=60, max_web_queries=10),
                                  ws, [0], time.monotonic())
            dd = [t for t in ws.read_queue() if t["strategy"] == "DUE_DILIGENCE"]
            self.assertEqual(len(dd), 0)

    def test_consecutive_empty_runs_increases(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            ws.update_strategy_state({"consecutive_empty_runs": 1})
            task = _task("DISCOVER", hint="x")
            with patch("tools.research_engine._call_phi_sync", return_value='[]'):
                with patch("tools.research_engine._safe_search", return_value=[]):
                    _run_discover(task, RunBudget(max_seconds=60, max_web_queries=10),
                                  ws, [0], time.monotonic())
            st = ws.load_strategy_state()
            self.assertEqual(st["consecutive_empty_runs"], 2)

    def test_budget_exceeded_returns_in_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("DISCOVER", hint="test")
            with patch("tools.research_engine._call_phi_sync",
                       return_value='["q1","q2","q3"]'):
                with patch("tools.research_engine._safe_search", return_value=[]):
                    r = _run_discover(task, RunBudget(max_seconds=1, max_web_queries=100),
                                      ws, [0], time.monotonic() - 1000)
            self.assertEqual(r.status, "IN_PROGRESS")
            self.assertIsNotNone(r.checkpoint.checkpoint_id)


class TestDueDiligence(unittest.TestCase):
    def test_creates_dossier(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("DUE_DILIGENCE", entity_id="ent_testco", hint="TestCo", priority=2)
            with patch("tools.research_engine._call_phi_sync", return_value=GOOD_DOSSIER):
                with patch("tools.research_engine._safe_search", return_value=[
                    {"title": "TestCo overview", "snippet": "cleantech",
                     "url": "https://tc.com", "source_id": "src_tc"}
                ]):
                    with patch("tools.research_engine.search_literature", return_value=[]):
                        r = _run_due_diligence(task, RunBudget(max_seconds=60, max_web_queries=10),
                                               ws, [0], time.monotonic())
            self.assertEqual(r.status, "DONE")
            d = ws.load_dossier("ent_testco")
            self.assertIsNotNone(d)

    def test_enqueues_validate_when_fit_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("DUE_DILIGENCE", entity_id="ent_highfit", hint="HighFit", priority=2)
            with patch("tools.research_engine._call_phi_sync", return_value=HIGH_FIT_DOSSIER):
                with patch("tools.research_engine._safe_search", return_value=[]):
                    with patch("tools.research_engine.search_literature", return_value=[]):
                        _run_due_diligence(task, RunBudget(max_seconds=60, max_web_queries=10),
                                           ws, [0], time.monotonic())
            val_tasks = [t for t in ws.read_queue() if t["strategy"] == "VALIDATE"]
            self.assertEqual(len(val_tasks), 1)


class TestCorrelate(unittest.TestCase):
    def test_generates_hypotheses(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            ws.save_dossier({"entity_id": "ent_a", "name": "OrgA", "status": "DRAFT",
                              "type": "ORG", "fit_assessment": {"fit_score": 60}})
            ws.save_dossier({"entity_id": "ent_b", "name": "OrgB", "status": "DRAFT",
                              "type": "ORG", "fit_assessment": {"fit_score": 70}})
            task = _task("CORRELATE", priority=3)
            task["payload"]["hypothesis_ids"] = []
            with patch("tools.research_engine._call_phi_sync", return_value=GOOD_HYPOTHESES):
                r = _run_correlate(task, RunBudget(max_seconds=60, max_web_queries=10),
                                   ws, [0], time.monotonic())
            self.assertEqual(r.status, "DONE")
            st = ws.load_strategy_state()
            self.assertGreater(len(st["hypotheses"]), 0)

    def test_skip_with_insufficient_dossiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("CORRELATE", priority=3)
            task["payload"]["hypothesis_ids"] = []
            r = _run_correlate(task, RunBudget(max_seconds=60, max_web_queries=10),
                               ws, [0], time.monotonic())
            self.assertEqual(r.status, "DONE")
            self.assertIn("insufficient", r.result_summary)


class TestValidate(unittest.TestCase):
    def test_passes_marks_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            ws.save_dossier({"entity_id": "ent_val", "name": "ValCo", "status": "DRAFT",
                              "type": "ORG", "fit_assessment": {"fit_score": 80},
                              "evidence_ids": [], "approval_status": "NONE"})
            ws.append_evidence({"type": "web", "title": "ValCo report",
                                 "url": "https://val.co/r", "source_id": "src_001",
                                 "snippet": "Strong evidence", "credibility_score": 0.9,
                                 "relevance_score": 0.8, "hypothesis_ids": [], "tags": []})
            task = _task("VALIDATE", entity_id="ent_val", priority=1)
            with patch("tools.research_engine._call_phi_sync", return_value=GOOD_VALIDATION):
                with patch("tools.research_engine.privacy_pre_hook",
                           return_value={"decision": "ALLOW"}):
                    r = _run_validate(task, RunBudget(max_seconds=60, max_web_queries=10),
                                      ws, [0], time.monotonic())
            self.assertEqual(r.status, "DONE")
            d = ws.load_dossier("ent_val")
            self.assertEqual(d["approval_status"], "PENDING")

    def test_fails_when_dossier_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            task = _task("VALIDATE", entity_id="ent_ghost", priority=1)
            r = _run_validate(task, RunBudget(), ws, [0], time.monotonic())
            self.assertEqual(r.status, "FAILED")


class TestExecuteCycle(unittest.TestCase):
    def test_max_tasks_is_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            for i in range(4):
                try:
                    ws.enqueue_task({"strategy": "DISCOVER", "priority": 4,
                                     "payload": {"entity_id": None, "query_hint": f"q{i}",
                                                 "hypothesis_ids": None}})
                except DuplicateError:
                    pass
            with patch("tools.research_engine._call_phi_sync", return_value='[]'):
                with patch("tools.research_engine._safe_search", return_value=[]):
                    r = execute_research_cycle(ws, budget=RunBudget(max_tasks=2,
                                                                     max_seconds=60,
                                                                     max_web_queries=20))
            self.assertLessEqual(r.tasks_run, 2)

    def test_frozen_when_in_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            from datetime import datetime, timezone, timedelta
            future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
            ws.update_strategy_state({"backoff_until": future})
            r = execute_research_cycle(ws)
            self.assertEqual(r.status, "FROZEN")

    def test_result_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            with patch("tools.research_engine._call_phi_sync", return_value='[]'):
                with patch("tools.research_engine._safe_search", return_value=[]):
                    r = execute_research_cycle(ws)
            d = r.to_dict()
            for f in ("status", "checkpoint", "result_summary", "artifacts",
                      "tasks_run", "queries_used", "elapsed_seconds"):
                self.assertIn(f, d)


class TestCheckpoint(unittest.TestCase):
    def test_id_length_12(self):
        cp = _make_checkpoint("DISCOVER", "step", "task_01", 5, {})
        self.assertEqual(len(cp.checkpoint_id), 12)

    def test_deterministic(self):
        cp1 = _make_checkpoint("DISCOVER", "step", "task_01", 5, {})
        cp2 = _make_checkpoint("DISCOVER", "step", "task_01", 5, {})
        self.assertEqual(cp1.checkpoint_id, cp2.checkpoint_id)


if __name__ == "__main__":
    unittest.main()
