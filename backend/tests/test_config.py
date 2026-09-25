"""Tests for settings validation."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_rejects_sync_database_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pw@localhost/db")
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(_env_file=None)


def test_database_password_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:supersecret@localhost/db")
    settings = Settings(_env_file=None)
    assert "supersecret" not in repr(settings)
    assert settings.database_url.get_secret_value().endswith("@localhost/db")


def test_rejects_unreasonable_health_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_S", "30")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
