"""
tests/test_background_runner.py
────────────────────────────────
Verifies the background job runner, enqueue endpoint, and performance contracts.

All tests are offline — no Ollama, no SearXNG needed.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))

REPO = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Job runner data structures
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobDataStructures(unittest.TestCase):
    """_ResearchJob must be orderable by priority (lower = higher priority)."""

    def test_import_job_classes(self):
        """JOB_INTERACTIVE and JOB_BACKGROUND must exist and be ints."""
        import importlib, sys
        # Minimal import of constants without running server
        server_text = (REPO / "web" / "server.py").read_text(encoding="utf-8")
        self.assertIn("JOB_INTERACTIVE = 0", server_text)
        self.assertIn("JOB_BACKGROUND   = 1", server_text)

    def test_budget_profiles_defined(self):
        server_text = (REPO / "web" / "server.py").read_text(encoding="utf-8")
        self.assertIn("BUDGET_INTERACTIVE", server_text)
        self.assertIn("BUDGET_BACKGROUND", server_text)
        self.assertIn("BUDGET_SCHEDULED", server_text)

    def test_interactive_budget_is_capped(self):
        """Interactive budget must be <= 120s to avoid blocking Mac."""
        server_text = (REPO / "web" / "server.py").read_text(encoding="utf-8")
        # Find BUDGET_INTERACTIVE line
        for line in server_text.splitlines():
            if "BUDGET_INTERACTIVE" in line and "RunBudget" in line and "max_seconds" in line:
                import re
                m = re.search(r"max_seconds=(\d+\.?\d*)", line)
                if m:
                    val = float(m.group(1))
                    self.assertLessEqual(val, 120.0,
                        "BUDGET_INTERACTIVE.max_seconds must be <= 120s to protect Mac performance")
                    return
        # If we reach here, look for the block form
        import re
        m = re.search(r"BUDGET_INTERACTIVE\s*=\s*RunBudget\([^)]*max_seconds=(\d+\.?\d*)", server_text)
        self.assertIsNotNone(m, "BUDGET_INTERACTIVE must set max_seconds")
        self.assertLessEqual(float(m.group(1)), 120.0)

    def test_background_budget_is_smaller_than_interactive(self):
        """Background budget must be <= interactive budget."""
        server_text = (REPO / "web" / "server.py").read_text(encoding="utf-8")
        import re
        mi = re.search(r"BUDGET_INTERACTIVE\s*=\s*RunBudget\([^)]*max_seconds=(\d+\.?\d*)", server_text)
        mb = re.search(r"BUDGET_BACKGROUND\s*=\s*RunBudget\([^)]*max_seconds=(\d+\.?\d*)", server_text)
        if mi and mb:
            self.assertLessEqual(float(mb.group(1)), float(mi.group(1)))

    def test_scheduler_interval_reasonable(self):
        """Scheduler must not fire more often than every 5 minutes."""
        server_text = (REPO / "web" / "server.py").read_text(encoding="utf-8")
        import re
        m = re.search(r"SCHEDULER_INTERVAL_SEC\s*=\s*(\d+)", server_text)
        self.assertIsNotNone(m, "SCHEDULER_INTERVAL_SEC must be defined")
        interval = int(m.group(1))
        self.assertGreaterEqual(interval, 300,
            "Scheduler must not run more than once per 5 minutes (300s)")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Server routing — /api/enqueue and /api/diagnostics exist
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewEndpointsExist(unittest.TestCase):
    """Server must expose /api/enqueue and /api/diagnostics."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def _extract_fn(self, marker: str) -> str:
        start = self.server.find(marker)
        if start < 0:
            return ""
        next_route = self.server.find("\n@app.", start + len(marker))
        return self.server[start:next_route] if next_route > 0 else self.server[start:]

    def test_enqueue_route_exists(self):
        self.assertIn('"/api/enqueue"', self.server)

    def test_enqueue_returns_queued_status(self):
        body = self._extract_fn('"/api/enqueue"')
        self.assertIn('"queued"', body, "/api/enqueue must return status=queued")

    def test_enqueue_returns_job_id(self):
        body = self._extract_fn('"/api/enqueue"')
        self.assertIn("job_id", body, "/api/enqueue must return job_id")

    def test_enqueue_puts_to_queue(self):
        body = self._extract_fn('"/api/enqueue"')
        self.assertIn("_job_queue", body)
        self.assertIn("put", body)

    def test_diagnostics_route_exists(self):
        self.assertIn('"/api/diagnostics"', self.server)

    def test_diagnostics_includes_job_runner_state(self):
        body = self._extract_fn('"/api/diagnostics"')
        self.assertIn("job_running", body)
        self.assertIn("queue_depth", body)

    def test_diagnostics_includes_audit(self):
        body = self._extract_fn('"/api/diagnostics"')
        self.assertIn("audit", body)

    def test_diagnostics_includes_searxng_status(self):
        body = self._extract_fn('"/api/diagnostics"')
        self.assertIn("searxng", body.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Chat handler performance contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatPerformanceContract(unittest.TestCase):
    """Chat handler must use reduced context window and non-blocking memory_store."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def _chat_body(self) -> str:
        start = self.server.find("async def chat(")
        next_route = self.server.find("\n@app.", start + 1)
        return self.server[start:next_route] if next_route > 0 else self.server[start:]

    def test_chat_uses_reduced_num_ctx(self):
        """Chat must pass num_ctx=4096 to call_phi (interactive budget)."""
        body = self._chat_body()
        self.assertIn("num_ctx=4096", body,
                      "chat handler must pass num_ctx=4096 to limit inference time")

    def test_memory_store_is_non_blocking(self):
        """memory_store must be offloaded (asyncio.to_thread or create_task)."""
        body = self._chat_body()
        self.assertTrue(
            "asyncio.to_thread" in body or "create_task" in body,
            "memory_store must not block the SSE stream — use asyncio.to_thread or create_task"
        )

    def test_call_phi_accepts_num_ctx(self):
        """call_phi signature must accept num_ctx parameter."""
        start = self.server.find("async def call_phi(")
        end = self.server.find("\n    async with", start)
        sig = self.server[start:end]
        self.assertIn("num_ctx", sig, "call_phi must accept num_ctx parameter")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Lifespan / startup hooks
# ═══════════════════════════════════════════════════════════════════════════════

class TestLifespanHooks(unittest.TestCase):
    """Server must use lifespan context manager, not deprecated on_event."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def test_uses_lifespan_not_on_event(self):
        """Prefer lifespan over deprecated @app.on_event('startup')."""
        self.assertIn("lifespan", self.server)
        # on_event is allowed for other things but startup should use lifespan
        self.assertNotIn("@app.on_event(\"startup\")", self.server,
                         "Use lifespan context manager, not deprecated @app.on_event('startup')")

    def test_job_consumer_started_in_lifespan(self):
        start = self.server.find("async def lifespan(")
        end = self.server.find("yield", start) + 10
        body = self.server[start:end]
        self.assertIn("_job_consumer", body, "lifespan must start _job_consumer task")

    def test_scheduler_started_in_lifespan(self):
        start = self.server.find("async def lifespan(")
        end = self.server.find("yield", start) + 10
        body = self.server[start:end]
        self.assertIn("_periodic_scheduler", body, "lifespan must start _periodic_scheduler task")

    def test_searxng_validated_on_startup(self):
        start = self.server.find("async def lifespan(")
        end = self.server.find("yield", start) + 10
        body = self.server[start:end]
        self.assertTrue(
            "searxng" in body.lower() or "SEARXNG" in body,
            "lifespan must validate SEARXNG_URL on startup"
        )

    def test_fastapi_uses_lifespan_param(self):
        """FastAPI app must be instantiated with lifespan=lifespan."""
        self.assertIn("lifespan=lifespan", self.server)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Background consumer logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobConsumerLogic(unittest.TestCase):
    """_job_consumer must write timeline card when artifacts are produced."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def _consumer_body(self) -> str:
        start = self.server.find("async def _job_consumer(")
        end = self.server.find("\nasync def _periodic_scheduler", start)
        return self.server[start:end] if end > 0 else self.server[start:start+3000]

    def test_consumer_writes_timeline_card_on_artifacts(self):
        body = self._consumer_body()
        self.assertIn("timeline_cards.json", body,
                      "_job_consumer must write timeline card when artifacts are produced")

    def test_consumer_logs_job_completion(self):
        body = self._consumer_body()
        self.assertTrue(
            "_log.info" in body or "logging" in body,
            "_job_consumer must log job completion"
        )

    def test_consumer_handles_exceptions(self):
        body = self._consumer_body()
        self.assertIn("except Exception", body,
                      "_job_consumer must catch exceptions so the loop continues")

    def test_consumer_releases_lock_in_finally(self):
        body = self._consumer_body()
        self.assertIn("finally:", body,
                      "_job_consumer must set _job_running=False in finally block")

    def test_single_consumer_enforces_max_1_concurrent(self):
        """Concurrency is enforced by having a single consumer coroutine."""
        # Verify there is exactly one _job_consumer task started
        lifespan_start = self.server.find("async def lifespan(")
        lifespan_end = self.server.find("yield", lifespan_start) + 200
        lifespan_body = self.server[lifespan_start:lifespan_end]
        count = lifespan_body.count("_job_consumer")
        self.assertEqual(count, 1, "Exactly one _job_consumer task must be started")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mac-lag guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestMacLagGuard(unittest.TestCase):
    """Enforce budget limits that protect Mac from runaway inference."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def test_research_executor_max_workers_1(self):
        """ThreadPoolExecutor for research must have max_workers=1."""
        import re
        m = re.search(r"_research_executor\s*=\s*ThreadPoolExecutor\(max_workers=(\d+)", self.server)
        self.assertIsNotNone(m, "_research_executor must be defined with max_workers")
        self.assertEqual(int(m.group(1)), 1,
                         "_research_executor must have max_workers=1 to prevent concurrent heavy inference")

    def test_scheduled_budget_max_seconds_le_120(self):
        """Scheduled budget must be capped to prevent runaway background cycles."""
        import re
        m = re.search(r"BUDGET_SCHEDULED\s*=\s*RunBudget\([^)]*max_seconds=(\d+\.?\d*)", self.server)
        if m:
            self.assertLessEqual(float(m.group(1)), 120.0)

    def test_scheduler_has_startup_delay(self):
        """Scheduler must not fire immediately on startup — needs a delay."""
        start = self.server.find("async def _periodic_scheduler(")
        end = self.server.find("\nasync def ", start + 1)
        body = self.server[start:end] if end > 0 else self.server[start:start+500]
        self.assertIn("asyncio.sleep", body,
                      "Scheduler must have an initial sleep delay before first cycle")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UI wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestUIEnqueueWiring(unittest.TestCase):
    """UI must expose enqueueResearch() and poll diagnostics for job status."""

    @classmethod
    def setUpClass(cls):
        cls.html = (REPO / "web" / "davidsan.html").read_text(encoding="utf-8")

    def test_enqueue_function_exists(self):
        self.assertIn("async function enqueueResearch(", self.html)

    def test_enqueue_calls_api_enqueue(self):
        start = self.html.find("async function enqueueResearch(")
        end = self.html.find("\n  }", start) + 4
        body = self.html[start:end]
        self.assertIn("/api/enqueue", body)

    def test_job_status_indicator_in_html(self):
        self.assertIn("jobStatus", self.html,
                      "UI must have a job status indicator element")

    def test_poll_diagnostics_implemented(self):
        self.assertIn("pollJobStatus", self.html)
        self.assertIn("/api/diagnostics", self.html)

    def test_poll_interval_reasonable(self):
        """Diagnostics polling must not be more frequent than every 10s."""
        import re
        m = re.search(r"setInterval\(pollJobStatus,\s*(\d+)\)", self.html)
        self.assertIsNotNone(m, "setInterval for pollJobStatus must be present")
        interval_ms = int(m.group(1))
        self.assertGreaterEqual(interval_ms, 10000,
                                "Diagnostics polling must not be more frequent than 10s")


if __name__ == "__main__":
    unittest.main()
