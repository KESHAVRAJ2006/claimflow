"""Application settings loaded from environment variables and .env files."""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

# Schemes managed Postgres providers hand out (Render: postgres://, most others: postgresql://).
_PLAIN_POSTGRES_SCHEMES = ("postgres://", "postgresql://")
# libpq options asyncpg has no parameter for; passed through, they crash connect() with "unexpected keyword".
# Neon's connection strings carry channel_binding=require; asyncpg still authenticates with SCRAM over TLS.
_LIBPQ_ONLY_OPTIONS = ("channel_binding", "gssencmode")


def to_asyncpg_url(url: str) -> str:
    """Rewrite a plain Postgres URL to use the asyncpg driver.

    Render (and Heroku-style providers) give ``postgres://user:pw@host/db``. SQLAlchemy needs the driver in the
    scheme, and asyncpg spells the TLS option ``ssl`` where libpq URLs say ``sslmode``. Options only libpq
    understands, such as Neon's ``channel_binding``, are dropped.

    Args:
        url: A database URL.

    Returns:
        The URL with a ``postgresql+asyncpg://`` scheme; any other scheme is returned unchanged (and then rejected
        by ``require_async_driver`` if it names a different driver).
    """
    if not url.startswith(_PLAIN_POSTGRES_SCHEMES):
        return url
    parsed = make_url("postgresql+asyncpg://" + url.split("://", 1)[1])
    query = {key: value for key, value in parsed.query.items() if key not in _LIBPQ_ONLY_OPTIONS}
    if "sslmode" in query:
        query["ssl"] = query.pop("sslmode")
    return parsed.set(query=query).render_as_string(hide_password=False)


class Settings(BaseSettings):
    """Typed, validated configuration.

    Precedence (highest first): real environment variables, backend/.env, repo-root .env, defaults.
    A missing or invalid required value stops the app at startup instead of failing on the first request.
    """

    model_config = SettingsConfigDict(
        # Later files win, so backend/.env can override the shared repo-root .env.
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        # The root .env also holds docker-compose-only keys (POSTGRES_PORT, ...); ignore them here.
        extra="ignore",
    )

    app_name: str = "ClaimFlow"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # A demo deployment's opt-in to the synthetic data set: scripts/serve.py seeds on start when it is true.
    # Production refuses to seed without it, and refuses --reset even with it.
    seed_demo_data: bool = False

    # SecretStr keeps the embedded password out of repr(), logs and tracebacks.
    database_url: SecretStr
    # A separate login with SELECT-only grants for the agent tools (see scripts/provision_readonly_role.py).
    # Optional so the API, tests and scripts that don't run agents work without it.
    tools_database_url: SecretStr | None = None
    # 5s: every tool query is a keyed lookup on a small table; anything slower is a problem to surface, not wait on.
    # On hosts where TOOLS_DATABASE_URL cannot be written out (render.yaml cannot join strings), give the login's
    # name and password instead and the URL is derived from DATABASE_URL; see derive_tools_database_url.
    agent_db_user: str | None = None
    agent_db_password: SecretStr | None = None
    tools_statement_timeout_ms: int = Field(default=5000, gt=0, le=30000)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "policy_chunks"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Off only in tests that exercise the API without needing retrieval; loading the model takes seconds.
    load_embedding_model: bool = True
    policy_documents_dir: Path = Path(__file__).resolve().parents[2] / "data" / "policies"

    # --- LLMs: Groq is primary; Gemini is used when Groq errors or rate-limits. At least one key is needed to
    # run agents; the API, rules and tools work without either.
    groq_api_key: SecretStr | None = None
    # The spec named llama-3.3-70b-versatile and gemini-2.0-flash; both were retired by the providers (Sept 2026).
    # gpt-oss-120b is the largest tool-calling model Groq serves; gemini-2.5-flash is the nearest stable Flash.
    groq_model: str = "openai/gpt-oss-120b"
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    # 60s: a 70B model answering a long investigation prompt can take 10-20s; anything past 60s is an outage.
    llm_timeout_s: float = Field(default=60.0, gt=0, le=300)
    # Provider SDK retries (with backoff) before falling back to the other provider.
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    # LangGraph checkpoints (SQLite). Temp dir by default so it works on any OS without configuration.
    checkpoint_path: Path = Path(tempfile.gettempdir()) / "claimflow_checkpoints.sqlite"

    # --- API (Phase 7) ---
    # Shared secret for every endpoint except /api/health. Required in production (see require_api_key_in_production).
    api_key: SecretStr | None = None
    # Browsers are allowed to call the API from these origins (the Next.js dev server by default).
    cors_origins: list[str] = ["http://localhost:3000"]
    # 10 MB, per the spec: a claim document is a few scanned pages; anything larger is almost certainly wrong.
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    # pypdf decompresses every page; a cap stops a small file with thousands of pages from eating the CPU.
    max_pdf_pages: int = Field(default=50, gt=0)
    upload_dir: Path = Path(tempfile.gettempdir()) / "claimflow_uploads"
    # Concurrent agent runs. Each run makes ~10 LLM calls; 2 keeps a free-tier Groq key under its rate limit.
    max_concurrent_runs: int = Field(default=2, ge=1, le=16)
    # n8n (Phase 11) receives an HMAC-signed POST for every triage result and human decision.
    n8n_webhook_url: str | None = None
    webhook_secret: SecretStr | None = None

    # --- MCP server (Phase 10) ---
    # Only for the optional HTTP transport; stdio (Claude Desktop) is authenticated by being a local subprocess.
    mcp_auth_token: SecretStr | None = None
    # Loopback by default: exposing the tools on a network is a deliberate choice, not an accident of defaults.
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = Field(default=8765, ge=1, le=65535)

    # Upper bound of 10s: a health endpoint slower than that is useless to an orchestrator's probe.
    health_check_timeout_s: float = Field(default=2.0, gt=0, le=10)

    @field_validator(
        "tools_database_url", "groq_api_key", "google_api_key", "qdrant_api_key", "api_key", "n8n_webhook_url",
        "webhook_secret", "mcp_auth_token", "agent_db_user", "agent_db_password", mode="before",
    )  # fmt: skip
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        """Treat an empty value as not configured.

        docker-compose passes ``KEY: ${KEY:-}`` as an empty string when the variable is unset.

        Args:
            value: The raw value from the environment.

        Returns:
            None for an empty or whitespace-only string, otherwise the value unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url", "tools_database_url", mode="before")
    @classmethod
    def accept_plain_postgres_urls(cls, value: object) -> object:
        """Let a provider's ``postgres://`` URL be pasted as-is.

        Args:
            value: The raw value from the environment.

        Returns:
            The URL rewritten for asyncpg, or the value unchanged if it is not a plain Postgres URL.
        """
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        return to_asyncpg_url(raw) if isinstance(raw, str) else value

    @field_validator("database_url", "tools_database_url")
    @classmethod
    def require_async_driver(cls, value: SecretStr | None) -> SecretStr | None:
        """Reject database URLs that would use a blocking driver.

        Args:
            value: The configured database URL, or None for an unset optional URL.

        Returns:
            The unchanged value when it is None or uses asyncpg.

        Raises:
            ValueError: If the URL does not start with ``postgresql+asyncpg://``.
        """
        # A sync driver (psycopg2) would block the event loop and stall every concurrent request.
        if value is not None and not value.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("database URLs must start with 'postgresql+asyncpg://'")
        return value

    @model_validator(mode="after")
    def derive_tools_database_url(self) -> "Settings":
        """Build TOOLS_DATABASE_URL from DATABASE_URL and AGENT_DB_USER/AGENT_DB_PASSWORD when it is not set.

        Same host, port and database as the owner; only the credentials differ. The password is percent-encoded
        by SQLAlchemy, so any character in it is safe in the URL.

        Returns:
            The settings, with ``tools_database_url`` filled in when both parts are given.
        """
        if self.tools_database_url is None and self.agent_db_user and self.agent_db_password:
            owner = make_url(self.database_url.get_secret_value())
            tools = owner.set(username=self.agent_db_user, password=self.agent_db_password.get_secret_value())
            self.tools_database_url = SecretStr(tools.render_as_string(hide_password=False))
        return self

    @field_validator("qdrant_url")
    @classmethod
    def require_qdrant_scheme(cls, value: str) -> str:
        """Catch a Qdrant URL pasted without its scheme (a common slip with Qdrant Cloud URLs).

        Args:
            value: The configured URL.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: If it does not start with http:// or https://.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"QDRANT_URL must start with http:// or https:// (got {value!r})")
        return value

    @model_validator(mode="after")
    def require_webhook_secret(self) -> "Settings":
        """Refuse a webhook URL without a signing secret.

        The n8n workflow rejects every unsigned event, so this combination can only ever produce silent 401s.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If N8N_WEBHOOK_URL is set and WEBHOOK_SECRET is missing or shorter than 16 characters.
        """
        secret = self.webhook_secret.get_secret_value() if self.webhook_secret else ""
        if self.n8n_webhook_url and len(secret) < 16:
            raise ValueError(
                "WEBHOOK_SECRET (16+ characters) is required when N8N_WEBHOOK_URL is set; n8n rejects unsigned events"
            )
        return self

    @model_validator(mode="after")
    def require_api_key_in_production(self) -> "Settings":
        """Refuse to start a production API that anyone could call.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``environment`` is production and API_KEY is unset or shorter than 32 characters.
        """
        # 32 characters of a random token is ~190 bits; shorter keys are usually a placeholder someone forgot.
        if self.is_production and (self.api_key is None or len(self.api_key.get_secret_value()) < 32):
            raise ValueError("API_KEY must be set to at least 32 characters in production")
        return self

    @property
    def is_production(self) -> bool:
        """Whether the app is running in production.

        Returns:
            True when ``environment`` is ``"production"``.
        """
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so .env is parsed once; tests call ``get_settings.cache_clear()`` after changing env vars.

    Returns:
        The validated Settings object.
    """
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
