#!/usr/bin/env bash
# Production RAG local demo: one FastAPI process serves the built UI and the API on
#   http://127.0.0.1:${RAG_DEMO_PORT:-8000}
# It does not start a tunnel (see the README's "Sharing the demo").
#
# Usage:
#   deploy/local-demo/demo.sh start    # start in demo mode, wait until /ready (model loaded)
#   deploy/local-demo/demo.sh warm     # pre-answer the presets so they're served from the cache
#   deploy/local-demo/demo.sh status   # process, /health, /ready, queue
#   deploy/local-demo/demo.sh stop
#
# Overrides (environment):
#   RAG_DEMO_PORT        port (default 8000)
#   RAG_DEMO_STATE_DIR   pid and log directory (default ~/.local/state/production-rag-demo)
#   RAG_CACHE_TTL_S      answer cache lifetime (default here 28800 = 8 h, so warmed presets last)
# Everything else comes from .env (RAG_MODELS_DIR, ...).
#
# The bind address is fixed to 127.0.0.1: reachable only from this machine and from a
# tunnel that connects locally. Demo mode is always on (design 9.4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST="127.0.0.1"
PORT="${RAG_DEMO_PORT:-8000}"
RAG_BIN="$REPO_ROOT/.venv/bin/rag"
STATE_DIR="${RAG_DEMO_STATE_DIR:-$HOME/.local/state/production-rag-demo}"
PID_FILE="$STATE_DIR/demo.pid"
LOG_FILE="$STATE_DIR/server.log"
BASE_URL="http://$HOST:$PORT"
READY_TIMEOUT_S=420  # a cold model load took 217 s on this laptop

die() { echo "ERROR: $*" >&2; exit 1; }
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL$1" || true; }
read_pid() { [ -f "$PID_FILE" ] && cat "$PID_FILE" || true; }
is_port_listening() { (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; }
pid_is_demo_server() {
  local pid="$1"
  [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "rag serve" || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "--port $PORT" || return 1
}

cmd_start() {
  command -v curl >/dev/null || die "curl is required"
  [ -x "$RAG_BIN" ] || die "no virtual environment: run 'uv sync' in $REPO_ROOT"
  [ -f "$REPO_ROOT/frontend/dist/index.html" ] || \
    die "the UI isn't built: run (cd frontend && npm ci && npm run build)"
  [ -n "$(ls -A "$REPO_ROOT/indexes" 2>/dev/null)" ] || die "no index: run 'uv run rag ingest'"

  mkdir -p "$STATE_DIR"
  local existing; existing="$(read_pid)"
  if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
    die "the demo is already running (pid $existing). Run: deploy/local-demo/demo.sh stop"
  fi
  rm -f "$PID_FILE"
  is_port_listening && die "port $PORT on $HOST is already in use"

  echo "Starting Production RAG demo on $BASE_URL (log: $LOG_FILE)"
  # exec keeps $! equal to the server's pid; nohup survives closing the terminal.
  (
    cd "$REPO_ROOT"
    exec env \
      RAG_DEMO_MODE=true \
      RAG_TRUSTED_PROXY_HEADER=cf-connecting-ip \
      RAG_CACHE_TTL_S="${RAG_CACHE_TTL_S:-28800}" \
      PYTHONUNBUFFERED=1 \
      nohup "$RAG_BIN" serve --demo --host "$HOST" --port "$PORT" \
        >>"$LOG_FILE" 2>&1 < /dev/null
  ) &
  echo "$!" > "$PID_FILE"

  local waited=0 code
  echo -n "Loading the index, reranker and Qwen3-1.7B"
  while :; do
    code="$(http_code /ready)"
    [ "$code" = "200" ] && break
    kill -0 "$(read_pid)" 2>/dev/null || { rm -f "$PID_FILE"; echo; die "the server exited; see $LOG_FILE"; }
    if curl -s --max-time 5 "$BASE_URL/ready" | grep -q startup_failed; then
      echo; die "startup failed (the UI's Replay mode still works); see $LOG_FILE"
    fi
    [ "$waited" -ge "$READY_TIMEOUT_S" ] && { echo; die "not ready after ${READY_TIMEOUT_S}s; see $LOG_FILE"; }
    sleep 2; waited=$((waited + 2)); echo -n "."
  done
  echo
  echo "Ready after ~${waited}s (pid $(read_pid)). Open: $BASE_URL/"
  echo "Next: 'demo.sh warm' to pre-answer the presets, then start the tunnel (README)."
}

cmd_warm() {
  [ "$(http_code /ready)" = "200" ] || die "the demo isn't ready; run: demo.sh start"
  command -v jq >/dev/null || die "jq is required"
  # Answer every single-view preset once, so the interview starts from the cache (D5).
  # Recorded replays are separate: 'uv run rag demo record' (after a fresh start).
  curl -s "$BASE_URL/v1/demo/presets" \
    | jq -r '.presets[] | select(.view != "compare" and .id != "injection" and .id != "repeat") | .question' \
    | while IFS= read -r q; do
        echo -n "warming: $q ... "
        curl -s -X POST "$BASE_URL/v1/query" -H 'Content-Type: application/json' \
          --data "$(jq -n --arg q "$q" '{question: $q, include_trace: false}')" \
          | jq -r '"\(if .cached then "already cached" else "\(.latency_ms / 1000 | floor) s" end)\(if .abstained then ", declined" else "" end)"'
        sleep 4  # stay under the per-minute question limit
      done
  echo "Warm. Presets now answer from the cache (labelled 'cached' in the UI) for RAG_CACHE_TTL_S."
}

cmd_status() {
  local pid; pid="$(read_pid)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "process: running (pid $pid)"; else echo "process: not running"; fi
  echo "/health: $(http_code /health)"
  echo "/ready:  $(http_code /ready)"
  if [ "$(http_code /metrics)" = "200" ] && command -v jq >/dev/null; then
    curl -s "$BASE_URL/metrics" | jq -c '{requests, queue, cache: .cache.hit_rate, answers: .answers.generated}'
  fi
}

cmd_stop() {
  local pid; pid="$(read_pid)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "the demo is not running"; return 0; fi
  pid_is_demo_server "$pid" || die "pid $pid is not this demo's server on port $PORT; refusing to signal it"
  echo "Stopping the demo server (pid $pid)..."
  kill -TERM "$pid"
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge 20 ]; then echo "no exit after 20s; sending SIGKILL" >&2; kill -KILL "$pid" 2>/dev/null || true; break; fi
    sleep 1; waited=$((waited + 1))
  done
  rm -f "$PID_FILE"
  echo "stopped"
}

case "${1:-}" in
  start) cmd_start ;;
  warm) cmd_warm ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) echo "usage: $(basename "$0") {start|warm|status|stop}" >&2; exit 2 ;;
esac
