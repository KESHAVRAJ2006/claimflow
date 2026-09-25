#!/bin/sh
# Container entrypoint: keep installed Python packages in sync with pyproject.toml, then run the command.
#
# pyproject.toml is bind-mounted in development, so adding a dependency and restarting the container is
# enough; without this check the old image would crash later with "ModuleNotFoundError".
set -eu

STAMP=/opt/venv/.pyproject.sha256

# The stamp holds the hash of the pyproject.toml the virtualenv was built from (written in the Dockerfile).
if ! sha256sum -c --status "$STAMP" 2>/dev/null; then
    echo "entrypoint: pyproject.toml changed since the image was built; installing dependencies..." >&2
    pip install --quiet -e ".[dev]"
    sha256sum pyproject.toml > "$STAMP"
    echo "entrypoint: dependencies are up to date" >&2
fi

exec "$@"
