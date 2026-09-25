"""Production start sequence (the runtime image's CMD), in ONE Python process.

Free hosts give the API a fraction of a CPU (Render's free instances: 0.1), and a free instance runs this
sequence again every time it wakes from sleep. Each step used to be its own process, and each paid again for
importing the application: 1.3-1.6 CPU-seconds, so 15-20 seconds at 0.1 CPU. Ingestion also loaded the embedding
model separately from the API. In one process the imports happen once, and ingestion and the API share
get_embedder's cached model.

    1. migrate:   required; a schema problem must stop the deploy, so there is no fallback.
    2. provision: the agent tools' SELECT-only login. On failure the API still starts, and the tools fail closed
                  (they cannot connect), so a provisioning problem disables the agents instead of weakening them.
    3. seed:      only when SEED_DEMO_DATA=true (a demo deployment); never wipes data the app has written.
    4. ingest:    index the policy PDFs; skips documents already indexed unchanged.
    5. uvicorn:   ONE worker. The triage runner and the SSE event broker live in the process; a second worker
                  would not see the first one's runs or events. Scale up (a bigger instance), not out.

docker-compose runs its own, equivalent development sequence.

Usage:
    python -m scripts.serve
"""

import asyncio
import os
import sys

import uvicorn

from app.core.config import get_settings
from scripts import ingest_policies, migrate, provision_readonly_role, seed


def prepare() -> bool:
    """Run steps 1-4.

    Returns:
        False if the schema could not be migrated, in which case the API must not start.
    """
    if migrate.main() != 0:
        return False
    if provision_readonly_role.main() != 0:
        print("WARNING: read-only role provisioning failed (see above); agent tools are unavailable", file=sys.stderr)
    if get_settings().seed_demo_data:
        try:
            seeded = asyncio.run(seed.run(reset=False, seed=seed.DEFAULT_SEED)) == 0
        except Exception as error:  # noqa: BLE001 — demo data is optional; the API starts without it
            print(f"seed: FAILED: {error!r}", file=sys.stderr)
            seeded = False
        if not seeded:
            print("WARNING: seeding failed (see above); starting the API anyway", file=sys.stderr)
    try:
        asyncio.run(ingest_policies.run())
    except Exception as error:  # noqa: BLE001 — search and triage report their own errors without the index
        print(f"WARNING: policy ingestion failed: {error!r}", file=sys.stderr)
    return True


def main() -> int:
    """Prepare the database and the policy index, then serve the API until stopped.

    Returns:
        Process exit code.
    """
    if not prepare():
        return 1
    # proxy_headers: behind Render's load balancer, trust X-Forwarded-For/-Proto so logs show the client's
    # address. The instance is only reachable through that proxy, hence '*'.
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # noqa: S104 — inside a container, reachable only through the host's proxy
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
