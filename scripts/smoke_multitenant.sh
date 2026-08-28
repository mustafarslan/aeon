#!/bin/bash
# Live two-identity multi-tenant smoke test against a REAL uvicorn process.
#
# WHY THIS EXISTS ALONGSIDE tests/test_multitenant_http.py. That suite drives the same
# dependency graph through FastAPI's TestClient, which runs the app in-process. It cannot
# catch anything that only goes wrong across a real process boundary: env vars read at
# import time, the lifespan hooks firing under uvicorn rather than TestClient, auth with no
# `dependency_overrides` available, or real sockets. This is the end-to-end gate.
#
# ENV MUST BE SET BEFORE THE PROCESS STARTS. `dependencies.py` reads every AEON_* var at
# import time and warns that setting one later silently does nothing -- a subprocess makes
# that natural instead of requiring cache_clear() gymnastics.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${AEON_SMOKE_PORT:-8077}"
SERVER_PID=""

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/shell"   # set -u: PYTHONPATH may be unset
export AEON_RECORDS_DIR="$TMP/records"      # required: server.py returns early from startup without it
export AEON_ATLAS_PATH="$TMP/atlas.aeon"
export AEON_TRACE_PATH="$TMP/trace.bin"
export AEON_AUDIT_LOG_PATH="$TMP/audit.jsonl"
export AEON_AUTH_MODE="insecure_dev_no_verify"   # otherwise the first /chat raises: auth fails closed
export AEON_CONSOLIDATION_INTERVAL_SECONDS="1"   # default 30s would outlive the test
# AEON_USE_OLLAMA deliberately unset -> MockProvider. This asserts isolation and plumbing,
# never answer quality; with the mock encoder the vectors are random by design.

echo "smoke: records dir $AEON_RECORDS_DIR, port $PORT"
"$ROOT/.venv/bin/python" -m uvicorn aeon_py.server:app --host 127.0.0.1 --port "$PORT" \
  > "$TMP/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited during startup"; tail -20 "$TMP/server.log"; exit 1
  fi
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || {
  echo "FAIL: server never became ready"; tail -20 "$TMP/server.log"; exit 1; }
echo "  server ready (pid $SERVER_PID)"

chat() {  # $1 = identity, $2 = text
  curl -sf -X POST "http://127.0.0.1:$PORT/chat" \
    -H "Content-Type: application/json" -H "X-User-ID: $1" \
    -d "{\"text\": \"$2\"}" > /dev/null
}

chat alice "my salary is 400000" || { echo "FAIL: alice chat"; tail -20 "$TMP/server.log"; exit 1; }
chat bob   "my salary is 90000"  || { echo "FAIL: bob chat";   tail -20 "$TMP/server.log"; exit 1; }
echo "  two identities chatted over real sockets"

FAILED=0
check() { if [ "$1" = "0" ]; then echo "  ok    $2"; else echo "  FAIL  $2"; FAILED=1; fi }

# 1. Per-tenant files exist and are distinct. This is the isolation boundary itself: a
#    shared record file would put every tenant's records into every tenant's prompt.
[ -f "$AEON_RECORDS_DIR/alice.atlas" ]; check $? "alice.atlas exists"
[ -f "$AEON_RECORDS_DIR/bob.atlas" ];   check $? "bob.atlas exists"

# 2. No third file appeared -- a stray shared store would defeat the boundary silently.
COUNT=$(ls "$AEON_RECORDS_DIR" 2>/dev/null | grep -c '\.atlas$')
[ "$COUNT" = "2" ]; check $? "exactly 2 record files (found $COUNT)"

# 3. No crosstalk in the raw bytes. Crude on purpose: it reads the files rather than asking
#    the API, so it cannot be fooled by a filter that happens to be applied at read time.
! grep -qa "400000" "$AEON_RECORDS_DIR/bob.atlas"; check $? "bob's store does not contain alice's figure"
! grep -qa "90000" "$AEON_RECORDS_DIR/alice.atlas"; check $? "alice's store does not contain bob's figure"

# 4. The lifespan hook actually ran under uvicorn -- TestClient cannot prove this.
grep -q "Aeon consolidation worker started" "$TMP/server.log"; check $? "background worker started under uvicorn"

# 5. Auth is real: an unidentified request must be rejected, not defaulted.
UNAUTH=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/chat" \
  -H "Content-Type: application/json" -d '{"text":"who am i"}')
[ "$UNAUTH" = "401" ]; check $? "request without an identity is rejected (got $UNAUTH)"

if [ "$FAILED" = "0" ]; then
  echo "SMOKE PASSED: two identities, real sockets, isolated stores"
  exit 0
fi
echo "SMOKE FAILED"; echo "--- server log ---"; tail -40 "$TMP/server.log"; exit 1
