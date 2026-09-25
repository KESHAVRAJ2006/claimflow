"""Check every moving part of a ClaimFlow setup and print the exact fix for anything wrong.

    docker compose exec backend python -m scripts.doctor             # everything, including 1 tiny LLM call each
    docker compose exec backend python -m scripts.doctor --offline   # no LLM calls, no n8n ping
    python -m scripts.doctor                                          # on the host: also compares the console's key

Most setup problems are one of a few: a stale container that never saw an edited .env, the console and the API
holding different keys, a read-only login that lost its grants, an empty vector index, a rate-limited or retired
LLM model, or n8n with a different webhook secret. Each check below names its problem and the command that fixes
it, instead of leaving you to decode a 401 or a 503 three layers away.

Exit code: 0 when nothing failed (warnings allowed), 1 otherwise.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from alembic.config import Config
from alembic.script import ScriptDirectory
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError
from sqlalchemy import func, pool, select
from sqlalchemy.ext.asyncio import create_async_engine
from transformers.utils import logging as transformers_logging

from app.agents.llm import provider_models
from app.agents.llm_errors import explain_llm_error
from app.core.config import Settings, get_settings
from app.db.models import Claim, Policy
from app.db.readonly import ReadOnlyDatabase, ReadOnlyViolationError, create_readonly_engine
from app.db.vector import create_qdrant_client
from app.retrieval.embeddings import get_embedder
from app.services.webhooks import SIGNATURE_HEADER, sign
from scripts.migrate import ALEMBIC_INI, inspect_schema

Status = Literal["ok", "warn", "fail", "skip"]
# Each network check gets this long; a check that hangs is itself a finding.
CHECK_TIMEOUT_S = 20.0
REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISION = "docker compose exec backend python -m scripts.provision_readonly_role"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check."""

    name: str
    status: Status
    detail: str
    fix: str | None = None


def _short(error: BaseException) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return message if len(message) <= 160 else f"{message[:157]}..."


# ---- checks -------------------------------------------------------------------------------------------------------


async def check_database(settings: Settings) -> CheckResult:
    """Postgres answers and the schema is at the latest migration."""
    name = "Postgres"
    try:
        state = await inspect_schema(settings.database_url.get_secret_value(), attempts=1)
    except Exception as error:  # noqa: BLE001 — any failure to connect is the finding
        return CheckResult(name, "fail", f"cannot connect: {_short(error)}", "docker compose up -d postgres")
    head = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()
    if state.current_revision != head:
        return CheckResult(
            name, "fail", f"schema at {state.current_revision or 'nothing'}, latest is {head}",
            "docker compose exec backend python -m scripts.migrate",
        )  # fmt: skip
    return CheckResult(name, "ok", f"reachable, schema at the latest migration ({head})")


async def check_seed_data(settings: Settings) -> CheckResult:
    """There is data to triage."""
    name = "Data"
    engine = create_async_engine(settings.database_url.get_secret_value(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            policies = (await connection.execute(select(func.count()).select_from(Policy))).scalar_one()
            claims = (await connection.execute(select(func.count()).select_from(Claim))).scalar_one()
    except Exception as error:  # noqa: BLE001
        return CheckResult(name, "skip", f"not checked ({_short(error)})")
    finally:
        await engine.dispose()
    if not policies:
        return CheckResult(name, "warn", "no policies yet", "docker compose exec backend python -m scripts.seed")
    return CheckResult(name, "ok", f"{policies} policies, {claims} claims")


async def check_readonly_login(settings: Settings) -> CheckResult:
    """The agents' login connects and can read the four agent tables and nothing more."""
    name = "Read-only login"
    if settings.tools_database_url is None:
        return CheckResult(
            name, "fail", "not configured, so the agents cannot run",
            "set AGENT_DB_USER and AGENT_DB_PASSWORD (or TOOLS_DATABASE_URL) in .env, then `docker compose up -d`",
        )  # fmt: skip
    database = ReadOnlyDatabase(create_readonly_engine(settings))
    try:
        async with database.connect():  # the first connection runs the privilege probe
            pass
    except ReadOnlyViolationError as error:
        return CheckResult(name, "fail", f"privileges are wrong: {_short(error)}", PROVISION)
    except Exception as error:  # noqa: BLE001 — wrong password, missing role, missing grants, database down
        return CheckResult(name, "fail", f"cannot use the login: {_short(error)}", PROVISION)
    finally:
        await database.dispose()
    return CheckResult(name, "ok", "connects; SELECT on the four agent tables only")


async def check_qdrant(settings: Settings) -> CheckResult:
    """Qdrant answers and the policy wordings are indexed."""
    name = "Qdrant"
    client = create_qdrant_client(settings)
    try:
        if not await client.collection_exists(settings.qdrant_collection):
            return CheckResult(
                name, "fail", f"collection {settings.qdrant_collection!r} does not exist",
                "docker compose exec backend python -m scripts.ingest_policies",
            )  # fmt: skip
        points = (await client.count(settings.qdrant_collection, exact=True)).count
    except Exception as error:  # noqa: BLE001
        fix = "docker compose up -d qdrant (or check QDRANT_URL / QDRANT_API_KEY)"
        return CheckResult(name, "fail", f"cannot reach {settings.qdrant_url}: {_short(error)}", fix)
    finally:
        await client.close()
    if points == 0:
        return CheckResult(
            name,
            "fail",
            "the policy collection is empty",
            "docker compose exec backend python -m scripts.ingest_policies",
        )
    return CheckResult(name, "ok", f"{points} policy passages indexed")


async def check_embedding_model(settings: Settings) -> CheckResult:
    """The embedding model loads from the local cache (the image bakes it in)."""
    name = "Embedding model"
    try:
        embedder = await asyncio.to_thread(get_embedder, settings.embedding_model_name)
        dimensions = len((await asyncio.to_thread(embedder.embed, ["waiting period"]))[0])
    except Exception as error:  # noqa: BLE001
        return CheckResult(
            name, "fail", f"cannot load {settings.embedding_model_name}: {_short(error)}",
            "rebuild the image: docker compose build backend",
        )  # fmt: skip
    return CheckResult(name, "ok", f"{settings.embedding_model_name} loaded ({dimensions} dimensions)")


async def check_llm_providers(
    settings: Settings, *, live: bool, models: Sequence[tuple[str, BaseChatModel]] | None = None
) -> list[CheckResult]:
    """Each configured provider accepts its key and model (one tiny call each when ``live``)."""
    configured = list(models) if models is not None else provider_models(settings)
    if not configured:
        return [
            CheckResult(
                "LLM",
                "fail",
                "no provider key is set, so triage is off",
                "set GROQ_API_KEY and/or GOOGLE_API_KEY in .env, then `docker compose up -d`",
            )  # fmt: skip
        ]
    results: list[CheckResult] = []
    for provider, model in configured:
        name = f"LLM {provider}"
        model_name = getattr(model, "model_name", None) or getattr(model, "model", "?")
        if not live:
            results.append(CheckResult(name, "skip", f"key set for {model_name}; not called (--offline)"))
            continue
        started = time.perf_counter()
        try:
            async with asyncio.timeout(CHECK_TIMEOUT_S):
                await model.ainvoke("Reply with the single word OK.")
        except Exception as error:  # noqa: BLE001
            advice = explain_llm_error(error, single_provider=len(configured) == 1)
            results.append(CheckResult(name, "fail", f"{model_name}: call failed", advice))
            continue
        latency = int((time.perf_counter() - started) * 1000)
        results.append(CheckResult(name, "ok", f"{model_name} answered in {latency} ms"))
    if len(configured) == 1:
        results.append(
            CheckResult(
                "LLM fallback",
                "warn",
                "only one provider is configured, so a rate limit fails the run",
                "set the other key (GROQ_API_KEY or GOOGLE_API_KEY) to enable automatic fallback",
            )  # fmt: skip
        )
    return results


def check_api_key(settings: Settings) -> CheckResult:
    """The API has a key (without one it refuses every request except health)."""
    name = "API key"
    if settings.api_key is None:
        return CheckResult(
            name, "fail", "API_KEY is not set, so the API answers 401 to every request",
            "set API_KEY in .env (and the same value in frontend/.env.local), then `docker compose up -d`",
        )  # fmt: skip
    return CheckResult(name, "ok", f"set ({len(settings.api_key.get_secret_value())} characters)")


def _env_value(path: Path, key: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}=") and not stripped.startswith("#"):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def check_console_key(settings: Settings, env_local: Path = REPO_ROOT / "frontend" / ".env.local") -> CheckResult:
    """The console's API_KEY equals the API's (the cause of "every call returns 401" in the console)."""
    name = "Console key"
    if not env_local.is_file():
        return CheckResult(name, "skip", "frontend/.env.local not visible here (run the doctor on the host to compare)")
    console_key = _env_value(env_local, "API_KEY")
    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    if not console_key:
        return CheckResult(name, "fail", "frontend/.env.local has no API_KEY", "copy API_KEY from .env into it")
    if console_key != api_key:
        return CheckResult(
            name, "fail", "frontend/.env.local API_KEY differs from the API's",
            "make them equal, then restart `npm run dev` (Next.js reads env files only at startup)",
        )  # fmt: skip
    return CheckResult(name, "ok", "frontend/.env.local matches the API")


async def check_webhook(settings: Settings, *, live: bool, client: httpx.AsyncClient | None = None) -> CheckResult:
    """n8n accepts a signed ping (so real events will get through too)."""
    name = "n8n webhook"
    if settings.n8n_webhook_url is None:
        return CheckResult(name, "skip", "N8N_WEBHOOK_URL not set; notifications are off")
    if not live:
        return CheckResult(name, "skip", "not pinged (--offline)")
    secret = settings.webhook_secret.get_secret_value() if settings.webhook_secret else ""
    body = json.dumps(
        {"event": "claimflow.ping", "sent_at": datetime.now(UTC).isoformat()}, separators=(",", ":")
    ).encode()
    headers = {"Content-Type": "application/json", SIGNATURE_HEADER: sign(body, secret)}
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=CHECK_TIMEOUT_S)
    try:
        response = await client.post(settings.n8n_webhook_url, content=body, headers=headers)
    except httpx.HTTPError as error:
        return CheckResult(
            name, "fail", f"cannot reach {settings.n8n_webhook_url}: {_short(error)}",
            "start n8n: docker compose --profile notifications up -d",
        )  # fmt: skip
    finally:
        if owns_client:
            await client.aclose()
    if response.status_code == httpx.codes.UNAUTHORIZED:
        return CheckResult(
            name, "fail", f"n8n rejected the signature ({response.text[:80]})",
            "WEBHOOK_SECRET must equal n8n's CLAIMFLOW_WEBHOOK_SECRET; after editing .env run "
            "`docker compose --profile notifications up -d --force-recreate`",
        )  # fmt: skip
    if response.status_code == httpx.codes.NOT_FOUND:
        return CheckResult(
            name, "fail", "n8n has no published workflow at this URL",
            "import and publish n8n/claimflow-notifications.json (compose does this on start)",
        )  # fmt: skip
    if response.is_error:
        return CheckResult(name, "fail", f"n8n answered {response.status_code}")
    return CheckResult(name, "ok", f"signed ping accepted ({response.status_code})")


# ---- runner -------------------------------------------------------------------------------------------------------


async def _bounded(name: str, check: Awaitable[CheckResult]) -> CheckResult:
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_S + 5):
            return await check
    except TimeoutError:
        return CheckResult(name, "fail", f"no answer within {CHECK_TIMEOUT_S + 5:.0f}s")


async def run_checks(settings: Settings, *, live: bool) -> list[CheckResult]:
    """Run every check in order.

    Args:
        settings: Application settings.
        live: Whether to make one tiny call per LLM provider and ping n8n.

    Returns:
        One result per check (several for the LLM providers).
    """
    results = [
        await _bounded("Postgres", check_database(settings)),
        await _bounded("Data", check_seed_data(settings)),
        await _bounded("Read-only login", check_readonly_login(settings)),
        await _bounded("Qdrant", check_qdrant(settings)),
        await _bounded("Embedding model", check_embedding_model(settings)),
    ]
    results.extend(await check_llm_providers(settings, live=live))
    results.append(check_api_key(settings))
    results.append(check_console_key(settings))
    results.append(await _bounded("n8n webhook", check_webhook(settings, live=live)))
    return results


TAGS: dict[Status, str] = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]", "skip": "[skip]"}


def render(results: Sequence[CheckResult], write: Callable[[str], object] = print) -> int:
    """Print the report.

    Args:
        results: Check results.
        write: Output function (print by default).

    Returns:
        The exit code: 1 if any check failed, else 0.
    """
    width = max(len(result.name) for result in results)
    for result in results:
        # ASCII tags, not check marks: a Windows console with a legacy code page cannot print those.
        write(f"  {TAGS[result.status]} {result.name.ljust(width)}  {result.detail}")
        if result.fix and result.status in ("fail", "warn"):
            write(f"  {' ' * 6} {' ' * width}  fix: {result.fix}")
    failed = sum(result.status == "fail" for result in results)
    warned = sum(result.status == "warn" for result in results)
    write("")
    write(f"  {failed} failed, {warned} warnings." if failed or warned else "  Everything checks out.")
    return 1 if failed else 0


def main() -> int:
    """Parse arguments, load settings (reporting their errors plainly) and run the checks."""
    parser = argparse.ArgumentParser(description="Diagnose a ClaimFlow setup and print fixes.")
    parser.add_argument("--offline", action="store_true", help="skip the LLM calls and the n8n ping")
    args = parser.parse_args()
    try:
        settings = get_settings()
    except ValidationError as error:
        print("ClaimFlow doctor: the configuration is invalid, so nothing else can be checked.\n")
        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"]) or "settings"
            print(f"  [FAIL] {field.upper()}: {item['msg']}")
        print("\n  fix: correct .env, then `docker compose up -d` so the containers get the new values.")
        return 1
    # Library chatter (a weight-loading progress bar, an SDK usage notice) would bury the report.
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    transformers_logging.disable_progress_bar()
    print(f"ClaimFlow doctor (environment: {settings.environment})\n")
    return render(asyncio.run(run_checks(settings, live=not args.offline)))


if __name__ == "__main__":
    sys.exit(main())
