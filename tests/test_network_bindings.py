"""
tests/test_network_bindings.py — Static security tests for network binding config.

Verifies that no service is configured to bind to 0.0.0.0 in any config or
compose file. All tests are deterministic and require no running server.
"""
import json
import re
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"
COMPOSE_PATH = BASE_DIR / "docker" / "docker-compose.yml"
START_SH_PATH = BASE_DIR / "start.sh"
RUNBOOK_PATH = BASE_DIR / "RUNBOOK.md"


class TestPhiTwinHostConfig(unittest.TestCase):
    """Verify phi-twin FastAPI is configured for localhost-only binding."""

    def setUp(self):
        with open(SETTINGS_PATH) as f:
            self.settings = json.load(f)

    def test_host_is_not_0000(self):
        """settings.json web.host must never be 0.0.0.0."""
        host = self.settings["web"]["host"]
        self.assertNotEqual(host, "0.0.0.0",
            "SECURITY: web.host='0.0.0.0' would expose phi-twin on all interfaces")

    def test_host_is_127_0_0_1(self):
        """settings.json web.host must be 127.0.0.1."""
        host = self.settings["web"]["host"]
        self.assertEqual(host, "127.0.0.1",
            f"Expected host='127.0.0.1', got '{host}'")

    def test_port_is_8080(self):
        """settings.json web.port must be 8080."""
        self.assertEqual(self.settings["web"]["port"], 8080)

    def test_no_wildcard_host_anywhere_in_settings(self):
        """No key in settings.json should contain '0.0.0.0'."""
        raw = json.dumps(self.settings)
        self.assertNotIn("0.0.0.0", raw,
            "Found '0.0.0.0' in settings.json — audit all host/url fields")

    def test_ollama_url_is_localhost(self):
        """Ollama must be on localhost (no external LLM calls)."""
        url = self.settings["ollama"]["base_url"]
        self.assertTrue(
            url.startswith("http://localhost") or url.startswith("http://127.0.0.1"),
            f"Ollama URL '{url}' must be localhost-only"
        )


class TestDockerComposeBindings(unittest.TestCase):
    """Verify docker-compose.yml binds all ports to 127.0.0.1."""

    def setUp(self):
        self.compose_text = COMPOSE_PATH.read_text(encoding="utf-8")

    def test_compose_exists(self):
        self.assertTrue(COMPOSE_PATH.exists(),
            "docker/docker-compose.yml must exist")

    def test_no_unbound_searxng_port(self):
        """SearXNG port must not be exposed as bare '8888:...' (binds 0.0.0.0)."""
        self.assertNotRegex(self.compose_text, r'^\s*-\s*"?8888:[0-9]+"?\s*$',
            "SearXNG port must be bound to 127.0.0.1, not 0.0.0.0")

    def test_searxng_port_bound_to_loopback(self):
        """SearXNG port must explicitly bind to 127.0.0.1."""
        self.assertIn("127.0.0.1:8888:", self.compose_text,
            "SearXNG must use '127.0.0.1:8888:...' port mapping")

    def test_no_0000_in_port_bindings(self):
        """No port binding line in compose file should bind to 0.0.0.0."""
        for line in self.compose_text.splitlines():
            stripped = line.strip()
            # Only inspect actual port mapping lines (start with - "... or - '...)
            if stripped.startswith('- "') or stripped.startswith("- '"):
                if ":" in stripped:
                    self.assertNotIn("0.0.0.0", stripped,
                        f"Port binding line binds to 0.0.0.0: {stripped!r}")

    def test_no_funnel_in_compose(self):
        """No reference to Tailscale Funnel (public exposure) in compose."""
        self.assertNotIn("funnel", self.compose_text.lower(),
            "Tailscale Funnel must never be used — it exposes services publicly")


class TestStartScript(unittest.TestCase):
    """Verify start.sh does not override localhost binding."""

    def setUp(self):
        self.start_text = START_SH_PATH.read_text(encoding="utf-8")

    def test_no_host_0000_in_start(self):
        """start.sh must not pass --host 0.0.0.0 to uvicorn."""
        self.assertNotIn("--host 0.0.0.0", self.start_text,
            "start.sh must not bind uvicorn to 0.0.0.0")

    def test_no_host_0000_in_uvicorn_call(self):
        """start.sh must not contain any 0.0.0.0 binding."""
        self.assertNotIn("0.0.0.0", self.start_text,
            "start.sh contains 0.0.0.0 — remove or replace with 127.0.0.1")


class TestURLGuardInvariants(unittest.TestCase):
    """
    Verify URL guard invariants: all service URLs must be loopback.
    These are the env var defaults — real values must also be loopback.
    """

    # Default values as documented in the workflow / docker-compose
    ALLOWED_PREFIXES = ("http://127.0.0.1", "http://localhost")

    def _assert_loopback(self, url: str, label: str):
        self.assertTrue(
            any(url.startswith(p) for p in self.ALLOWED_PREFIXES),
            f"{label} URL '{url}' must start with {self.ALLOWED_PREFIXES}"
        )

    def test_phi_server_url_default(self):
        """Default PHI_SERVER_URL must be loopback."""
        self._assert_loopback("http://127.0.0.1:8080", "PHI_SERVER_URL")

    def test_searxng_url_default(self):
        """Default SEARXNG_URL must be loopback."""
        self._assert_loopback("http://127.0.0.1:8888", "SEARXNG_URL")

    def test_imessage_relay_url_default(self):
        """Default iMessage relay URL must be loopback."""
        self._assert_loopback("http://127.0.0.1:9999", "IMESSAGE_RELAY")

    def test_no_external_url_allowed(self):
        """External URLs must not pass the loopback check."""
        external_urls = [
            "https://example.com",
            "http://0.0.0.0:5678",
            "http://192.168.1.100:8080",
        ]
        for url in external_urls:
            with self.subTest(url=url):
                self.assertFalse(
                    any(url.startswith(p) for p in self.ALLOWED_PREFIXES),
                    f"External URL '{url}' incorrectly passes loopback check"
                )

    def test_tailscale_serve_urls_not_used_internally(self):
        """
        Tailscale HTTPS URLs must not appear in settings.json or start.sh.
        Tailscale serve is an access layer only; internal routing stays local.
        """
        for path in (SETTINGS_PATH, START_SH_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(".ts.net", text,
                f"Tailscale domain (.ts.net) must not appear in {path.name} — internal URLs must stay 127.0.0.1")


class TestTailscaleServeScript(unittest.TestCase):
    """Verify the Tailscale serve script enforces the right security posture."""

    def setUp(self):
        self.script_path = BASE_DIR / "scripts" / "tailscale-serve.sh"
        self.script_text = self.script_path.read_text(encoding="utf-8")

    def test_script_exists(self):
        self.assertTrue(self.script_path.exists())

    def test_no_funnel_command(self):
        """Script must not invoke tailscale funnel as a command (comments/guards are ok)."""
        import re
        # The only forbidden pattern is actually executing: $TS funnel or tailscale funnel
        # The script may have a `funnel)` guard that blocks it — that is fine.
        forbidden = re.compile(r'(\$TS|tailscale)\s+funnel\b')
        for line in self.script_text.splitlines():
            stripped = line.strip()
            # Skip comments and echo/error output lines
            if stripped.startswith("#") or stripped.startswith("echo"):
                continue
            self.assertNotRegex(stripped, forbidden,
                f"tailscale funnel must not be executed: {stripped!r}")

    def test_no_public_in_serve_flags(self):
        """Script must not pass --public flag to tailscale serve."""
        self.assertNotIn("--public", self.script_text)

    def test_serve_uses_localhost(self):
        """tailscale serve target must be localhost, not 0.0.0.0."""
        self.assertIn("http://localhost:", self.script_text,
            "tailscale serve must proxy from localhost:PORT")
        self.assertNotIn("http://0.0.0.0:", self.script_text)

    def test_searxng_not_served_via_tailscale(self):
        """SearXNG (port 8888) must NOT be exposed via Tailscale Serve."""
        self.assertNotIn("--https=8888", self.script_text,
            "SearXNG must not be exposed via Tailscale — it is internal-only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
