"""
api/context.py — Shared singletons for the phi-twin API layer.

Single source of truth for objects that must be instantiated exactly once per process:
  - BASE_DIR        : root of the phi-twin project (phi-twin/)
  - workspace       : WorkspaceState (dossier index, task state)
  - request_executor: ThreadPoolExecutor(1) for blocking LLM + research calls

Both server.py and all router files import from here. This prevents duplicate
instantiation and ensures the job consumer, route handlers, and routers all share
the same executor and workspace state.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Resolve phi-twin/ root regardless of where this file lives (phi-twin/api/context.py)
BASE_DIR = Path(__file__).parent.parent

# Ensure phi-twin/ is importable — idempotent if server.py already added it
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.state_manager import WorkspaceState  # noqa: E402

workspace = WorkspaceState(BASE_DIR)
request_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
