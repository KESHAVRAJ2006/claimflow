# ClaimFlow API image, multi-stage.
#
#   requirements  the dependency lists, extracted from pyproject.toml
#   base          Python + CPU-only torch in a virtualenv owned by the app user
#   deps          + the runtime dependencies, + the embedding model baked in
#   dev           + dev tools (pytest, ruff), the project installed editable, tests. docker-compose builds this.
#   runtime       a clean python:3.11-slim with only the venv, the model and the application code. The LAST stage,
#                 so a plain `docker build` (Render builds without --target) produces the production image.

# Why a stage just for this: layers are cached by the files they copy. Copying pyproject.toml itself into the slow
# install layer meant that editing a TOOL setting there ([tool.ruff], [tool.pytest]) reinstalled every dependency
# and re-downloaded the model: minutes, and long enough for Claude Desktop to give up starting the MCP server.
# Copying only the extracted lists keeps those layers cached until a dependency really changes.
FROM python:3.11-slim AS requirements
WORKDIR /src
COPY pyproject.toml ./
RUN python -c "import tomllib; project = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
open('requirements.txt', 'w').write('\n'.join(project['dependencies']) + '\n'); \
open('requirements-dev.txt', 'w').write('\n'.join(project['optional-dependencies']['dev']) + '\n')"


FROM python:3.11-slim AS base

# No .pyc clutter, logs flushed immediately (important for JSON log shipping), no pip cache in the layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

# Run as a non-root user; a compromised process should not own the container.
RUN useradd --create-home --uid 1000 claimflow \
    && mkdir /app /opt/venv /opt/hf-cache \
    && chown claimflow /app /opt/venv /opt/hf-cache
USER claimflow

# A virtualenv owned by the app user, so the dev entrypoint can install new dependencies without root.
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache
WORKDIR /app

# CPU-only torch first. The default PyPI wheel for Linux bundles ~2 GB of CUDA libraries this app never uses;
# once this torch is installed, sentence-transformers accepts it and pip does not replace it.
ARG TORCH_VERSION=2.14.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"


FROM base AS deps

# Only [project].dependencies: the application itself is not installed as a package in production, and the dev
# extras (pytest, ruff) stay out.
COPY --from=requirements /src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Bake the embedding model into the image: startup needs no network and never waits for a download.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"


FROM deps AS dev

COPY --from=requirements /src/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt
# Register the project as an editable install, against an empty package so editing application code never
# invalidates it. --no-deps: everything it needs is installed above, so this takes seconds even after
# pyproject.toml changes.
COPY --chown=claimflow pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install --no-deps -e . \
    && sha256sum pyproject.toml > /opt/venv/.pyproject.sha256
# From here on, never contact the Hugging Face hub (the model is already cached).
ENV HF_HUB_OFFLINE=1

COPY --chown=claimflow app ./app
COPY --chown=claimflow alembic.ini docker-entrypoint.sh ./
COPY --chown=claimflow migrations ./migrations
COPY --chown=claimflow scripts ./scripts
COPY --chown=claimflow data ./data
COPY --chown=claimflow tests ./tests
# Strip Windows line endings in case the scripts were checked out with CRLF.
RUN sed -i 's/\r$//' docker-entrypoint.sh scripts/start.sh

EXPOSE 8000
# Invoked through `sh` so the script needs no executable bit (Windows checkouts often lose it).
ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python -m scripts.migrate && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Apply Debian's security fixes released since the base image was published (a scan flagged 4 high-severity
# issues in the stock python:3.11-slim packages). Rebuilding the image picks up newer fixes.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 claimflow
# The venv's interpreter symlinks point at /usr/local/bin/python3.11, which this image has at the same path.
COPY --from=deps --chown=claimflow /opt/venv /opt/venv
COPY --from=deps --chown=claimflow /opt/hf-cache /opt/hf-cache

WORKDIR /app
# Application code is owned by root and read-only to the app user: a compromised process cannot rewrite it.
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY data ./data
RUN sed -i 's/\r$//' scripts/start.sh
USER claimflow

EXPOSE 8000
# Render probes healthCheckPath itself; this is for any other orchestrator (and `docker ps`). Python, not curl:
# the slim image has no curl, and adding it only for this would grow the attack surface.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"
CMD ["sh", "/app/scripts/start.sh"]
