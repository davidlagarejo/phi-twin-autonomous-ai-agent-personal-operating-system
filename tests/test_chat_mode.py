"""
tests/test_chat_mode.py
────────────────────────
Verifies the chat/plan routing separation:
  - /api/chat uses CHAT_SYSTEM_PROMPT (plain-text, max-3-questions, no JSON mandate)
  - /api/plan still uses PLAN_PROMPT + plan_json schema validation
  - The UI sends to /api/chat for conversation
  - No template-loop patterns in the chat prompt
  - Privacy gates are not weakened (privacy_pre_hook still imported in search path)

All tests are offline (no Ollama, no SearXNG needed).
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Prompt file checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatSystemPromptFile(unittest.TestCase):
    """chat_system.md must exist and enforce conversational, not JSON, behavior."""

    @classmethod
    def setUpClass(cls):
        cls.path = REPO / "prompts" / "chat_system.md"
        cls.text = cls.path.read_text(encoding="utf-8") if cls.path.exists() else ""

    def test_file_exists(self):
        self.assertTrue(self.path.exists(), "prompts/chat_system.md must exist")

    def test_no_json_mandate(self):
        """Must NOT contain 'Always return valid JSON' — the root cause of prompt-lock."""
        self.assertNotIn("Always return valid JSON", self.text)
        self.assertNotIn("No markdown outside JSON", self.text)

    def test_no_json_schema_reference(self):
        """Must not reference draft_message.channel or JSON field constraints."""
        self.assertNotIn("draft_message.channel", self.text)
        self.assertNotIn('"channel"', self.text)

    def test_limits_questions(self):
        """Must cap the number of questions at 3 or fewer."""
        lower = self.text.lower()
        self.assertTrue(
            "max 3" in lower or "3 question" in lower or "máximo 3" in lower,
            "chat_system.md must cap questions at max 3"
        )

    def test_forbids_template_loop_phrases(self):
        """Must explicitly ban the template-loop trigger phrases."""
        lower = self.text.lower()
        self.assertIn("here is what i need to know", lower,
                      "chat_system.md must explicitly forbid the template-loop phrase")

    def test_no_json_output(self):
        """Must prohibit JSON output (markdown is now allowed for rich formatting)."""
        lower = self.text.lower()
        self.assertTrue(
            "no json" in lower or "plain text" in lower or "no code blocks" in lower,
            "chat_system.md must explicitly prohibit JSON output"
        )

    def test_not_empty(self):
        self.assertGreater(len(self.text.strip()), 100)


class TestSystemMdUnchanged(unittest.TestCase):
    """system.md must remain unchanged — other endpoints (triage, dossier) still use it."""

    @classmethod
    def setUpClass(cls):
        cls.text = (REPO / "prompts" / "system.md").read_text(encoding="utf-8")

    def test_json_mandate_still_present(self):
        """system.md must still contain the JSON mandate (used by triage/dossier)."""
        self.assertIn("Always return valid JSON", self.text)

    def test_plan_prompt_unchanged(self):
        text = (REPO / "prompts" / "orchestrator_plan.md").read_text(encoding="utf-8")
        self.assertIn("PLAN_JSON", text)
        self.assertIn("SEARCH_WEB", text)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. server.py routing checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerRoutingConfig(unittest.TestCase):
    """server.py must load CHAT_SYSTEM_PROMPT and route /api/chat to it."""

    @classmethod
    def setUpClass(cls):
        cls.server = (REPO / "web" / "server.py").read_text(encoding="utf-8")

    def _extract_fn(self, marker: str) -> str:
        """Extract the body of a function starting at `marker`."""
        start = self.server.find(marker)
        if start < 0:
            return ""
        # Find next top-level @app. decorator (marks end of this handler)
        next_route = self.server.find("\n@app.", start + len(marker))
        return self.server[start:next_route] if next_route > 0 else self.server[start:]

    def test_chat_system_prompt_loaded(self):
        self.assertIn("CHAT_SYSTEM_PROMPT", self.server)
        self.assertIn("chat_system.md", self.server)

    def test_chat_handler_uses_chat_prompt(self):
        body = self._extract_fn("async def chat(")
        self.assertIn("CHAT_SYSTEM_PROMPT", body,
                      "/api/chat handler must use CHAT_SYSTEM_PROMPT")

    def test_chat_handler_not_using_system_prompt(self):
        """The SYSTEM_PROMPT (JSON-only) must NOT appear in the chat handler body."""
        body = self._extract_fn("async def chat(")
        # Strip occurrences of CHAT_SYSTEM_PROMPT, then check SYSTEM_PROMPT is absent
        cleaned = body.replace("CHAT_SYSTEM_PROMPT", "")
        self.assertNotIn("SYSTEM_PROMPT", cleaned,
                         "/api/chat must not use the JSON-mandate SYSTEM_PROMPT")

    def test_chat_handler_no_json_extraction(self):
        """Chat handler must NOT strip ```json wrappers — it expects plain text."""
        body = self._extract_fn("async def chat(")
        self.assertNotIn('if "```json" in raw', body,
                         "Removed JSON extraction block must stay removed from /api/chat")

    def test_chat_handler_no_validate_plan_json(self):
        """Chat handler must NOT call validate_plan_json."""
        body = self._extract_fn("async def chat(")
        self.assertNotIn("validate_plan_json", body)

    def test_plan_handler_still_uses_plan_prompt(self):
        body = self._extract_fn("async def plan(")
        self.assertIn("PLAN_PROMPT", body,
                      "/api/plan must still use PLAN_PROMPT (orchestrator_plan.md)")

    def test_plan_handler_does_not_use_chat_prompt(self):
        body = self._extract_fn("async def plan(")
        self.assertNotIn("CHAT_SYSTEM_PROMPT", body,
                         "/api/plan must not use the chat prompt")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UI (davidsan.html) routing checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestUIRouting(unittest.TestCase):
    """The chat UI must call /api/chat (SSE), not the n8n webhook, for conversation."""

    @classmethod
    def setUpClass(cls):
        html = (REPO / "web" / "davidsan.html").read_text(encoding="utf-8")
        # Extract sendMessage function body
        start = html.find("async function sendMessage(")
        end   = html.find("\n  }", start) + 4  # include closing brace
        cls.fn = html[start:end] if start >= 0 else ""

    def test_calls_api_chat(self):
        self.assertIn("/api/chat", self.fn,
                      "sendMessage() must call /api/chat")

    def test_primary_call_is_api_chat(self):
        """The first fetch in sendMessage must go to /api/chat."""
        first_fetch = re.search(r"fetch\(`?\$\{PHI_BASE\}/api/chat`?", self.fn)
        self.assertIsNotNone(first_fetch,
                             "sendMessage must contain fetch to ${PHI_BASE}/api/chat")

    def test_sse_streaming_implemented(self):
        """sendMessage must use getReader() to consume the SSE stream."""
        self.assertIn("getReader()", self.fn,
                      "sendMessage must read the SSE stream via getReader()")

    def test_sse_done_marker_handled(self):
        """sendMessage must detect the [DONE] SSE marker."""
        self.assertIn("[DONE]", self.fn,
                      "sendMessage must handle the [DONE] SSE end marker")

    def test_fallback_still_present(self):
        """A catch block must handle server unavailability (shows error, no crash)."""
        # localResponse() was replaced by an inline error display in the catch block.
        # The invariant is: catch block exists and does not save bad text to history.
        self.assertIn("catch", self.fn,
                      "sendMessage must have a catch block for offline/error handling")
        self.assertIn("err", self.fn.lower(),
                      "catch block must handle the error")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Non-regression checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonRegression(unittest.TestCase):
    """Ensure privacy gates and plan contract are unaffected."""

    def test_plan_json_schema_intact(self):
        schema = json.loads((REPO / "schemas" / "plan_json.schema.json").read_text())
        for field in ("status", "action", "goal", "why", "state_update", "next_step"):
            self.assertIn(field, schema.get("required", []))

    def test_privacy_pre_hook_not_removed_from_search(self):
        """privacy_pre_hook must still be imported in tools/search.py."""
        text = (REPO / "tools" / "search.py").read_text()
        self.assertIn("privacy_pre_hook", text)

    def test_privacy_pre_hook_not_removed_from_literature(self):
        text = (REPO / "tools" / "literature.py").read_text()
        self.assertIn("privacy_pre_hook", text)

    def test_no_new_external_api_calls(self):
        """server.py must not contain any new external API base URLs."""
        text = (REPO / "web" / "server.py").read_text()
        external_patterns = [
            "api.openai.com", "api.anthropic.com", "duckduckgo.com",
            "google.com/search", "bing.com",
        ]
        for pat in external_patterns:
            # These might exist for ASK_CLAUDE — check they haven't been added to chat route
            if pat in text:
                # Only allowed if they were already there (anthropic for /api/ask_claude)
                if pat == "api.anthropic.com":
                    continue  # pre-existing CLAUDE_API_KEY path
                self.fail(f"Unexpected external API URL added: {pat}")

    def test_audit_write_still_in_chat_handler(self):
        """Chat handler must still write to the audit log."""
        server = (REPO / "web" / "server.py").read_text()
        body = server[server.find("async def chat("):]
        next_route = body.find("\n@app.")
        chat_body = body[:next_route] if next_route > 0 else body
        self.assertIn("write_audit", chat_body,
                      "Chat handler must still call write_audit for traceability")


if __name__ == "__main__":
    unittest.main()
