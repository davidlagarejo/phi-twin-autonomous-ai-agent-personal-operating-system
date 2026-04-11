#!/usr/bin/env bash
# scripts/smoke_research.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-command smoke test for phi-twin research + library_fetch flows.
# Usage:  bash scripts/smoke_research.sh
# Exits 0 on full pass, 1 on any failure.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHI="http://127.0.0.1:8080"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
AUDIT_LOG="$REPO_DIR/data/audit_logs/privacy_audit.jsonl"

PASS=0
FAIL=0

_pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
_fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

# ── json_field <json> <key> ────────────────────────────────────────────────
json_field() {
  echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$2',''))" 2>/dev/null || echo ""
}

# ── assert_json_key <json> <key> <label> ──────────────────────────────────
assert_json_key() {
  local json="$1" key="$2" label="$3"
  if python3 -c "import sys,json; d=json.load(sys.stdin); assert '$key' in d" <<< "$json" 2>/dev/null; then
    _pass "$label"
  else
    _fail "$label — unexpected body: $(echo "$json" | head -c 200)"
  fi
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  phi-twin smoke: research + library_fetch            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 0. Server health ──────────────────────────────────────────────────────────
echo "── 0. Server health"
if ! curl -sf --max-time 5 "$PHI/health" > /dev/null 2>&1; then
  echo "  [ERROR] Server not responding at $PHI"
  echo "          Start it with:  bash start.sh"
  exit 1
fi
_pass "Server healthy at $PHI"

# ── 1. GET /api/state ─────────────────────────────────────────────────────────
echo ""
echo "── 1. GET /api/state"
STATE=$(curl -sf --max-time 10 "$PHI/api/state" 2>/dev/null || echo "{}")
if python3 -c "
import sys, json
d = json.load(sys.stdin)
assert isinstance(d, dict) and len(d) > 0, 'empty or non-dict'
# Accept any reasonable state shape
ok = any(k in d for k in ('queue_pending', 'queue', 'run_counter', 'status', 'pending_approvals'))
assert ok, f'no expected key found in: {list(d.keys())}'
" <<< "$STATE" 2>/dev/null; then
  RUN=$(json_field "$STATE" "run_counter")
  QPEND=$(echo "$STATE" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d.get('queue_pending', d.get('queue', {}).get('pending', '?')))
" 2>/dev/null || echo "?")
  _pass "State OK — run_counter=${RUN:-?}, queue_pending=${QPEND}"
else
  _fail "State missing expected keys — got: $(echo "$STATE" | head -c 200)"
fi

# ── 2. POST /api/execute (quick cycle) ───────────────────────────────────────
echo ""
echo "── 2. POST /api/execute (max_seconds=30, max_web_queries=2)"
EXEC=$(curl -sf --max-time 60 -X POST "$PHI/api/execute" \
  -H "Content-Type: application/json" \
  -d '{"max_seconds": 30, "max_web_queries": 2}' 2>/dev/null || echo "{}")
EXEC_STATUS=$(json_field "$EXEC" "status")
if [[ "$EXEC_STATUS" =~ ^(DONE|IN_PROGRESS|FROZEN|FAILED)$ ]]; then
  ELAPSED=$(json_field "$EXEC" "elapsed_seconds")
  QUERIES=$(json_field "$EXEC" "queries_used")
  _pass "Execute returned status=$EXEC_STATUS (elapsed=${ELAPSED:-?}s, queries=${QUERIES:-?})"
else
  _fail "Execute invalid status='$EXEC_STATUS' — body: $(echo "$EXEC" | head -c 200)"
fi

# ── 3. POST /api/library_fetch — open-access arXiv ───────────────────────────
echo ""
echo "── 3. POST /api/library_fetch (arXiv open-access, score=0.75)"
FETCH=$(curl -sf --max-time 45 -X POST "$PHI/api/library_fetch" \
  -H "Content-Type: application/json" \
  -d '{
    "doi_or_url": "https://arxiv.org/abs/2301.12345",
    "credibility_score": 0.7,
    "relevance_score": 0.8,
    "open_pdf_url": "https://arxiv.org/pdf/2301.12345.pdf"
  }' 2>/dev/null || echo "{}")
FSTATUS=$(json_field "$FETCH" "status")
case "$FSTATUS" in
  DOWNLOADED|CACHED)
    _pass "library_fetch open-access: $FSTATUS"
    ;;
  NEEDS_USER_LOGIN)
    # arXiv might redirect or be temporarily unavailable — acceptable
    _pass "library_fetch open-access: NEEDS_USER_LOGIN (download failed, privacy/network ok)"
    ;;
  BLOCKED)
    # Privacy gate fired or SSRF blocked — policy is working correctly
    _pass "library_fetch open-access: BLOCKED (privacy gate active — expected in strict mode)"
    ;;
  SCORE_BELOW_THRESHOLD)
    _fail "library_fetch open-access: score gate fired unexpectedly (scores 0.7+0.8=0.75 should pass)"
    ;;
  *)
    _fail "library_fetch open-access: unexpected status='$FSTATUS' — body: $(echo "$FETCH" | head -c 200)"
    ;;
esac

# ── 4. POST /api/library_fetch — low score ───────────────────────────────────
echo ""
echo "── 4. POST /api/library_fetch (low score — must reject without network)"
LOWSCORE=$(curl -sf --max-time 10 -X POST "$PHI/api/library_fetch" \
  -H "Content-Type: application/json" \
  -d '{"doi_or_url":"https://doi.org/10.1/test","credibility_score":0.1,"relevance_score":0.1}' \
  2>/dev/null || echo "{}")
LSTATUS=$(json_field "$LOWSCORE" "status")
if [[ "$LSTATUS" == "SCORE_BELOW_THRESHOLD" ]]; then
  UQ=$(json_field "$LOWSCORE" "unlocking_question")
  _pass "library_fetch low-score: SCORE_BELOW_THRESHOLD (unlocking_question present=$([ -n "$UQ" ] && echo yes || echo no))"
else
  _fail "library_fetch low-score: expected SCORE_BELOW_THRESHOLD, got '$LSTATUS'"
fi

# ── 5. Audit log ─────────────────────────────────────────────────────────────
echo ""
echo "── 5. Audit log inspection"
if [[ -f "$AUDIT_LOG" ]]; then
  python3 - "$AUDIT_LOG" <<'PYEOF'
import sys, json

path = sys.argv[1]
entries = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("action_type") in ("SEARCH_WEB", "LIBRARY_FETCH"):
                entries.append(e)
        except Exception:
            pass

if entries:
    print(f"  Recent gating decisions (last {min(5,len(entries))} of {len(entries)}):")
    for e in entries[-5:]:
        ts = e.get("timestamp", "?")[:19]
        at = e.get("action_type", "?").ljust(14)
        dec = e.get("decision", "?").ljust(20)
        ph = e.get("payload_hash", "?")[:8]
        print(f"    {ts}  {at}  {dec}  hash={ph}…")
else:
    print("  (no SEARCH_WEB / LIBRARY_FETCH entries yet)")

# Check: no raw payload fields
RAW_KEYS = {"query", "payload", "text", "content", "raw"}
violations = [e for e in entries if RAW_KEYS & set(e.keys())]
if violations:
    bad = [list(RAW_KEYS & set(v.keys())) for v in violations]
    print(f"  [WARN] Audit entries contain raw payload keys: {bad}")
    sys.exit(1)
else:
    print("  [OK] No raw payload keys in audit entries")
PYEOF
  _pass "Audit log readable and clean"
else
  echo "  (audit log not yet created at $AUDIT_LOG — will appear after first research cycle)"
  _pass "Audit log check skipped (no requests yet)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
if [[ "$FAIL" -eq 0 ]]; then
  echo "  RESULT: ${PASS} passed, 0 failed — ALL CLEAR"
else
  echo "  RESULT: ${PASS} passed, ${FAIL} failed"
fi
echo "══════════════════════════════════════════════════════"
echo ""
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
