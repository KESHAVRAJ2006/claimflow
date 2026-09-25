#!/bin/sh
# Production start sequence (the runtime image's CMD). docker-compose runs its own, equivalent dev sequence.
#
#   1. migrate:   required; a schema problem must stop the deploy, so there is no fallback.
#   2. provision: the agent tools' SELECT-only login. On failure the API still starts, and the tools fail closed
#                 (they cannot connect), so a provisioning problem disables the agents instead of weakening them.
#   3. seed:      only when SEED_DEMO_DATA=true (a demo deployment); never wipes data the app has written.
#   4. ingest:    index the policy PDFs; skips documents already indexed unchanged.
#   5. uvicorn:   ONE worker. The triage runner and the SSE event broker live in the process; a second worker
#                 would not see the first one's runs or events. Scale up (a bigger instance), not out.
set -eu

python -m scripts.migrate
python -m scripts.provision_readonly_role \
    || echo "WARNING: read-only role provisioning failed (see above); agent tools will be unavailable" >&2
if [ "${SEED_DEMO_DATA:-false}" = "true" ]; then
    python -m scripts.seed || echo "WARNING: seeding failed (see above); starting the API anyway" >&2
fi
python -m scripts.ingest_policies || echo "WARNING: policy ingestion failed (see above)" >&2

# --proxy-headers: behind Render's load balancer, trust X-Forwarded-For/-Proto so logs show the client's address.
# The instance is only reachable through that proxy, hence '*'.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips '*' \
    --no-server-header
