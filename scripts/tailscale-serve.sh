#!/bin/zsh
# tailscale-serve.sh — Variant B: Tailscale Serve (tailnet-only, no Funnel)
#
# Exposes phi-twin (:8080) on the Tailscale tailnet ONLY.
# Services remain bound to 127.0.0.1 — zero 0.0.0.0 or LAN exposure.
#
# What this does NOT do:
#   - Does NOT use tailscale funnel (no public internet exposure)
#   - Does NOT open router/firewall ports
#   - Does NOT touch SearXNG (internal-only, no remote access needed)
#   - Does NOT touch the iMessage relay (local-only by design)
#
# Prerequisites:
#   1. Tailscale installed and connected: `tailscale status` shows "Running"
#   2. MagicDNS enabled in Tailscale admin → https://login.tailscale.com/admin/dns
#   3. HTTPS certificates enabled in Tailscale admin (same DNS settings page)
#   4. phi-twin running on 127.0.0.1:8080
#
# Usage:
#   chmod +x scripts/tailscale-serve.sh
#   ./scripts/tailscale-serve.sh setup      # configure Tailscale Serve
#   ./scripts/tailscale-serve.sh teardown   # remove all Serve rules
#   ./scripts/tailscale-serve.sh status     # show current Serve config

set -euo pipefail

# ── Detect tailscale CLI ──────────────────────────────────────────────────────
_find_tailscale() {
  if command -v tailscale &>/dev/null; then
    echo "tailscale"
  elif [[ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ]]; then
    echo "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
  else
    echo ""
  fi
}

TS=$(_find_tailscale)
if [[ -z "$TS" ]]; then
  echo "ERROR: tailscale CLI not found." >&2
  echo "  Option A: Install Tailscale from https://tailscale.com/download" >&2
  echo "  Option B: Add Tailscale app CLI to PATH:" >&2
  echo "    sudo ln -s /Applications/Tailscale.app/Contents/MacOS/Tailscale /usr/local/bin/tailscale" >&2
  exit 1
fi

# ── Explicitly block Funnel ───────────────────────────────────────────────────
_no_funnel() {
  echo "ERROR: Tailscale Funnel is explicitly forbidden in this setup." >&2
  echo "  Funnel exposes services to the public internet." >&2
  echo "  Use 'setup' to configure tailnet-only Serve instead." >&2
  exit 1
}

# ── Verify Tailscale is running ───────────────────────────────────────────────
_check_running() {
  local state
  state=$($TS status --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('BackendState',''))" 2>/dev/null || echo "")
  if [[ "$state" != "Running" ]]; then
    echo "ERROR: Tailscale is not running (state: '${state:-unknown}')." >&2
    echo "  Start Tailscale and run: tailscale status" >&2
    exit 1
  fi
}

# ── Get machine's Tailscale hostname ─────────────────────────────────────────
_ts_hostname() {
  $TS status --json 2>/dev/null \
    | python3 -c "
import sys, json
s = json.load(sys.stdin)
self_node = s.get('Self', {})
dns = self_node.get('DNSName', '')
# DNSName is like 'machine.tailnet.ts.net.' — strip trailing dot
print(dns.rstrip('.') or 'your-machine.your-tailnet.ts.net')
" 2>/dev/null || echo "your-machine.your-tailnet.ts.net"
}

case "${1:-help}" in

  # ── setup ──────────────────────────────────────────────────────────────────
  setup)
    echo "=== Tailscale Serve — Option 3 Variant B ==="
    echo "    (tailnet-only HTTPS proxy, no Funnel, no router ports)"
    echo ""
    _check_running

    TS_HOST=$(_ts_hostname)

    # phi-twin API: https://<machine>.ts.net:8080 → http://localhost:8080
    echo "[1/1] Configuring phi-twin: https://${TS_HOST}:8080 → localhost:8080"
    $TS serve --bg --https=8080 http://localhost:8080
    echo "      OK"

    echo ""
    echo "=== Active Serve rules ==="
    $TS serve status
    echo ""
    echo "Remote access URLs (reachable only from your tailnet):"
    echo "  phi-twin → https://${TS_HOST}:8080/health"
    echo ""
    echo "SearXNG (port 8888) is intentionally NOT exposed — internal only."
    echo "iMessage relay (port 9999) is intentionally NOT exposed — local Mac process."
    ;;

  # ── teardown ───────────────────────────────────────────────────────────────
  teardown)
    echo "=== Removing Tailscale Serve rules ==="
    $TS serve --https=8080 off 2>/dev/null && echo "  removed :8080" || echo "  (no :8080 rule active)"
    echo ""
    echo "Services are now localhost-only. Tailscale Serve rules cleared."
    $TS serve status 2>/dev/null || true
    ;;

  # ── status ─────────────────────────────────────────────────────────────────
  status)
    echo "=== Tailscale Serve Status ==="
    $TS serve status
    echo ""
    echo "=== Tailscale Node ==="
    $TS status | head -5
    ;;

  # ── funnel guard ───────────────────────────────────────────────────────────
  funnel)
    _no_funnel
    ;;

  # ── help ───────────────────────────────────────────────────────────────────
  help|--help|-h)
    echo "Usage: $0 [setup|teardown|status]"
    echo ""
    echo "  setup     Configure Tailscale Serve for phi-twin (:8080)"
    echo "  teardown  Remove all Serve rules (revert to localhost-only)"
    echo "  status    Show current Serve config and Tailscale node status"
    echo ""
    echo "NOTE: 'funnel' is explicitly blocked. Services must NOT be publicly exposed."
    ;;

  *)
    echo "ERROR: Unknown command '${1}'" >&2
    echo "Usage: $0 [setup|teardown|status]" >&2
    exit 1
    ;;
esac
