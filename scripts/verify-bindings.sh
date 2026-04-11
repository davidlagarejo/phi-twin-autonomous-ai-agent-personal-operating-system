#!/bin/zsh
# verify-bindings.sh — Network binding security verification
#
# Verifies that NO service is bound to 0.0.0.0.
# Run after any stack restart to confirm the security posture.
#
# Exit code: 0 = all PASS, 1 = at least one FAIL

set -uo pipefail

PASS=0
FAIL=0
ERRORS=()

check() {
  local label="$1"
  local cmd="$2"
  local expect="$3"
  local result
  result=$(eval "$cmd" 2>/dev/null || true)
  if echo "$result" | grep -qF "$expect"; then
    echo "  PASS  $label"
    ((PASS++)) || true
  else
    echo "  FAIL  $label"
    echo "        expected: $expect"
    echo "        got: ${result:0:200}"
    ((FAIL++)) || true
    ERRORS+=("$label")
  fi
}

check_absent() {
  local label="$1"
  local cmd="$2"
  local absent="$3"
  local result
  result=$(eval "$cmd" 2>/dev/null || true)
  if echo "$result" | grep -qF "$absent"; then
    echo "  FAIL  $label — found FORBIDDEN pattern: $absent"
    ((FAIL++)) || true
    ERRORS+=("$label")
  else
    echo "  PASS  $label"
    ((PASS++)) || true
  fi
}

echo "=============================="
echo " phi-twin Network Binding Audit"
echo "=============================="
echo ""

# ── 1. phi-twin (port 8080) ──────────────────────────────────────────────────
echo "[phi-twin :8080]"
check_absent \
  "8080 not on 0.0.0.0" \
  "lsof -iTCP:8080 -sTCP:LISTEN -nP 2>/dev/null" \
  "0.0.0.0:8080"
check \
  "8080 bound to 127.0.0.1" \
  "lsof -iTCP:8080 -sTCP:LISTEN -nP 2>/dev/null" \
  "127.0.0.1:8080"
check \
  "8080 health responds" \
  "curl -sf --max-time 3 http://127.0.0.1:8080/health" \
  '"status"'
check_absent \
  "8080 NOT reachable on tailscale without serve" \
  "curl -sf --max-time 2 http://0.0.0.0:8080/health" \
  '"status"'

echo ""

# ── 2. SearXNG (port 8888) ───────────────────────────────────────────────────
echo "[SearXNG :8888]"
check_absent \
  "8888 not on 0.0.0.0" \
  "lsof -iTCP:8888 -sTCP:LISTEN -nP 2>/dev/null" \
  "0.0.0.0:8888"
check \
  "8888 bound to 127.0.0.1" \
  "lsof -iTCP:8888 -sTCP:LISTEN -nP 2>/dev/null" \
  "127.0.0.1:8888"
check \
  "8888 JSON search responds" \
  "curl -sf --max-time 5 'http://127.0.0.1:8888/search?q=test&format=json'" \
  '"results"'

echo ""

# ── 4. Docker port bindings ──────────────────────────────────────────────────
echo "[Docker port bindings]"
if command -v docker &>/dev/null; then
  check_absent \
    "searxng docker not on 0.0.0.0" \
    "docker inspect searxng --format '{{json .HostConfig.PortBindings}}' 2>/dev/null" \
    '"HostIp":""'
  check \
    "searxng docker bound to 127.0.0.1" \
    "docker inspect searxng --format '{{json .HostConfig.PortBindings}}' 2>/dev/null" \
    '"HostIp":"127.0.0.1"'
else
  echo "  SKIP  docker not available in this shell"
fi

echo ""

# ── 5. settings.json host check ──────────────────────────────────────────────
echo "[phi-twin config]"
check_absent \
  "settings.json host is not 0.0.0.0" \
  "python3 -c \"import json; s=json.load(open('/Users/davidlagarejo/phi-twin/config/settings.json')); print(s['web']['host'])\"" \
  "0.0.0.0"
check \
  "settings.json host is 127.0.0.1" \
  "python3 -c \"import json; s=json.load(open('/Users/davidlagarejo/phi-twin/config/settings.json')); print(s['web']['host'])\"" \
  "127.0.0.1"

echo ""

# ── 6. Tailscale serve status ────────────────────────────────────────────────
echo "[Tailscale Serve]"
if command -v tailscale &>/dev/null; then
  TS_SERVE=$(tailscale serve status 2>/dev/null || echo "")
  if echo "$TS_SERVE" | grep -q "8080"; then
    echo "  PASS  tailscale serve :8080 active"
    ((PASS++)) || true
  else
    echo "  WARN  tailscale serve :8080 not active — run: ./scripts/tailscale-serve.sh setup"
  fi
else
  echo "  SKIP  tailscale CLI not found"
fi

echo ""
echo "=============================="
echo " Results: ${PASS} PASS, ${FAIL} FAIL"
echo "=============================="

if [[ $FAIL -gt 0 ]]; then
  echo "FAILED checks:"
  for e in "${ERRORS[@]}"; do echo "  - $e"; done
  exit 1
fi
echo "All security binding checks PASSED."
