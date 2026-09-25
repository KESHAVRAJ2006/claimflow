"""Wipe the development Postgres database, rebuild the schema and reload seed data.

Safer than ``docker compose down -v``, which also deletes the Qdrant volume (the indexed policy documents).
Refuses to run when ENVIRONMENT=production.

Usage:
    python -m scripts.reset_db --yes              # wipe, migrate, seed
    python -m scripts.reset_db --yes --no-seed    # wipe, migrate
"""

import argparse
import asyncio
import sys

from alembic import command
from alembic.config import Config
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from scripts.migrate import ALEMBIC_INI
from scripts.provision_readonly_role import provision
from scripts.seed import run as run_seed


async def drop_all_objects(database_url: str) -> None:
    """Drop and recreate the ``public`` schema, removing every table, trigger and function in it.

    Args:
        database_url: asyncpg connection URL.
    """
    engine = create_async_engine(database_url, poolclass=pool.NullPool)
    try:
        async with engine.begin() as connection:
            # Static statements only — nothing user-supplied is ever interpolated into this SQL.
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
            # Restore the default privilege that Postgres grants on a freshly created database.
            await connection.execute(text("GRANT USAGE ON SCHEMA public TO PUBLIC"))
    finally:
        await engine.dispose()


def main() -> int:
    """Parse arguments and reset the database.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Wipe and rebuild the development database.")
    parser.add_argument("--yes", action="store_true", help="confirm that all Postgres data will be deleted")
    parser.add_argument("--no-seed", action="store_true", help="leave the database empty after migrating")
    args = parser.parse_args()

    settings = get_settings()
    if settings.is_production:
        print("reset_db: refusing to run with ENVIRONMENT=production.", file=sys.stderr)
        return 1
    if not args.yes:
        print("reset_db: this deletes ALL Postgres data (Qdrant is untouched). Re-run with --yes to confirm.")
        return 2

    database_url = settings.database_url.get_secret_value()
    print("reset_db: dropping all tables...")
    asyncio.run(drop_all_objects(database_url))
    print("reset_db: applying migrations...")
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    # Dropping the tables dropped their grants too, while the agents' login role (cluster-wide) survived. Without
    # re-granting, every agent tool and read-only test would fail with "permission denied" until the next restart.
    if settings.tools_database_url is not None:
        print("reset_db: re-granting the agents' read-only login...")
        username = asyncio.run(provision(database_url, settings.tools_database_url.get_secret_value()))
        print(f"reset_db: '{username}' can read the agent tables and write nothing.")
    if args.no_seed:
        print("reset_db: done (database is empty).")
        return 0
    print("reset_db: seeding...")
    return asyncio.run(run_seed(reset=False, seed=42))


if __name__ == "__main__":
    sys.exit(main())
