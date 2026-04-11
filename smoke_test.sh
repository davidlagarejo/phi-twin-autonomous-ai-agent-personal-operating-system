#!/usr/bin/env bash
# smoke_test.sh — Phi-Twin production smoke test
# Usage: cd /Users/davidlagarejo/phi-twin && bash smoke_test.sh
# Prerequisites: server running on 127.0.0.1:8080

set -euo pipefail

BASE="http://127.0.0.1:8080"
PASS=0
FAIL=0
AUDIT="data/audit_logs/audit.jsonl"

check() {
  local n="$1" desc="$2"
  if [ "$3" = "PASS" ]; then
    echo "  [PASS] $n: $desc"
    PASS=$((PASS+1))
  else
    echo "  [FAIL] $n: $desc — $4"
    FAIL=$((FAIL+1))
  fi
}

echo "========================================"
echo "PHI-TWIN SMOKE TEST  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================"

# ── S0: health check ──────────────────────────────────────────────────────────
echo ""
echo "S0: Health"
STATUS=$(curl -s "$BASE/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
PHI=$(curl -s "$BASE/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('phi_ready',''))" 2>/dev/null)
[ "$STATUS" = "ok" ] && check S0 "health=ok" PASS || check S0 "health endpoint" FAIL "status=$STATUS"
[ "$PHI" = "True" ] && check S0b "phi_ready=true" PASS || check S0b "phi_ready" FAIL "phi_ready=$PHI"

# ── S1: simple signal (no context_json) ───────────────────────────────────────
echo ""
echo "S1: Simple signal (no context_json)"
RESP1=$(curl -s -X POST "$BASE/api/triage" \
  -H "Content-Type: application/json" \
  -d '{"signal":"Early-stage industrial AI startup seeking seed investor","source":"email"}')
VALID1=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gate',{}).get('valid',''))" 2>/dev/null)
DEC1=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); out=d.get('output',{}); print(out.get('decision','') if isinstance(out,dict) else 'FAIL')" 2>/dev/null)
BODY1=$(echo "$RESP1" | python3 -c "import sys,json,re; d=json.load(sys.stdin); out=d.get('output',{}); dm=out.get('draft_message',{}) if isinstance(out,dict) else {}; body=dm.get('body','') if isinstance(dm,dict) else ''; ph=re.findall(r'[\[{<](FIRST_NAME|LAST_NAME|NAME)[\]}>]',body,re.I); print('NO_PH' if not ph else ','.join(ph))" 2>/dev/null)
[ "$VALID1" = "True" ] && check S1a "gate.valid=true" PASS || check S1a "gate.valid" FAIL "$VALID1"
[ -n "$DEC1" ] && [ "$DEC1" != "FAIL" ] && check S1b "decision=$DEC1" PASS || check S1b "decision field" FAIL "$DEC1"
[ "$BODY1" = "NO_PH" ] && check S1c "no placeholders" PASS || check S1c "placeholder found" FAIL "$BODY1"

# ── S2: context_json with contact ─────────────────────────────────────────────
echo ""
echo "S2: With context_json"
RESP2=$(curl -s -X POST "$BASE/api/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "signal":"Industrial AI pilot program with allocated budget and named contact at Ecopetrol Ventures",
    "source":"linkedin",
    "context_json":{
      "title":"Industrial AI Pilot Grant",
      "company":"Ecopetrol Ventures",
      "signal":"Industrial AI pilot program with allocated budget and named contact"
    }
  }')
VALID2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gate',{}).get('valid',''))" 2>/dev/null)
SUBJ2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); out=d.get('output',{}); dm=out.get('draft_message',{}) if isinstance(out,dict) else {}; print('HAS_SUBJ' if isinstance(dm,dict) and dm.get('subject') else 'NO_SUBJ')" 2>/dev/null)
MODEL2=$(tail -1 "$AUDIT" 2>/dev/null | python3 -c "import sys,json; e=json.load(sys.stdin); print(e.get('model',''))" 2>/dev/null)
[ "$VALID2" = "True" ] && check S2a "gate.valid=true" PASS || check S2a "gate.valid (may fail on schema drift)" FAIL "valid=$VALID2"
[ "$SUBJ2" = "HAS_SUBJ" ] && check S2b "subject auto-filled" PASS || check S2b "subject missing" FAIL "$SUBJ2"
[ "$MODEL2" = "mlx" ] && check S2c "audit model=mlx" PASS || check S2c "audit model" FAIL "$MODEL2"

# ── S3: audit log integrity ───────────────────────────────────────────────────
echo ""
echo "S3: Audit log"
AUDIT_COUNT=$(python3 -c "
lines = open('$AUDIT').readlines()
recent = [l for l in lines[-10:] if l.strip()]
mlx_ok = sum(1 for l in recent if '\"model\": \"mlx\"' in l or '\"model\":\"mlx\"' in l)
print(mlx_ok)
" 2>/dev/null)
[ "${AUDIT_COUNT:-0}" -gt 0 ] && check S3 "audit entries with model=mlx: $AUDIT_COUNT" PASS || check S3 "no mlx entries in audit" FAIL "count=$AUDIT_COUNT"

# ── S4: adapter path ──────────────────────────────────────────────────────────
echo ""
echo "S4: Adapter resolution"
ADAPTER=$(python3 -c "
import sys; sys.path.insert(0,'web')
from web.mlx_runner import _resolve_adapter_dir
print(_resolve_adapter_dir())
" 2>/dev/null)
echo "$ADAPTER" | grep -q "adapters_best" && check S4 "adapter=adapters_best" PASS || check S4 "adapter not adapters_best" FAIL "$ADAPTER"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
TOTAL=$((PASS+FAIL))
echo "RESULT: $PASS/$TOTAL PASS"
if [ "$FAIL" -eq 0 ]; then
  echo "VERDICT: GO"
else
  echo "VERDICT: NO-GO — $FAIL check(s) failed"
fi
echo "========================================"
exit $FAIL
