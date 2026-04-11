#!/bin/zsh
cd "$(dirname "$0")"

echo "Starting Phi Twin..."

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Starting Ollama..."
  ollama serve &
  sleep 3
fi

# Install deps if needed
pip install -q -r requirements.txt

# Wire SearXNG (local-only, privacy gate requires 127.0.0.1)
export SEARXNG_URL="http://127.0.0.1:8888"

# Claude API (for person extraction and tool calling)
[ -f "$(dirname "$0")/.env" ] && source "$(dirname "$0")/.env"
[ -z "$CLAUDE_API_KEY" ] && source ~/.zshrc 2>/dev/null || true

# Kill stale background processes if any
pkill -f "proactive_loop.py" 2>/dev/null || true
pkill -f "imessage_relay.py" 2>/dev/null || true
pkill -f "job_tracker.py"   2>/dev/null || true

# Crear directorio de logs si no existe
mkdir -p logs

# Start proactive loop daemon (reemplaza n8n workflows)
python3 scripts/proactive_loop.py >> logs/proactive_loop.log 2>&1 &
echo "Proactive loop PID: $!"

# Start iMessage relay (inbound: iMessage → /api/chat directo, sin n8n)
if [ -f scripts/imessage_relay.py ]; then
  python3 scripts/imessage_relay.py >> logs/imessage_relay.log 2>&1 &
  echo "iMessage relay PID: $!"
fi

# Start job tracker daemon (obligatorio — escanea ofertas cada 4h)
python3 scripts/job_tracker.py >> logs/job_tracker.log 2>&1 &
echo "Job tracker PID: $!"

# Start server (foreground — bloquea hasta Ctrl+C)
python3 -m uvicorn web.server:app --host 127.0.0.1 --port 8080 --log-level warning
