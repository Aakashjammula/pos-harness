# Build context is the repo root.
#
# There is no Node stage: the UI is built ahead of time with
# `npm run build:app` and committed to backend/src/pos/static/, so this image
# only has to install Python dependencies and copy the source.
#
# This image is for deploying the app somewhere, not for daily local use:
# a container has no display, so the native folder picker cannot open, and
# `execute` runs inside the container rather than on your machine.

FROM python:3.14-slim AS builder
# Pinned rather than :latest, per Astral's Docker guide -- a moving uv is a
# moving build.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

# The cache and the target venv are on different filesystems once a cache
# mount is involved, so hardlinking fails; copying silences the warning.
ENV UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first, without the project: they change rarely, so this layer
# is reused across source edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=backend/README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev --compile-bytecode

COPY backend/ /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --compile-bytecode


FROM python:3.14-slim

# git because the agent shells out to it; the document skills need nothing
# beyond what pip installed.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uid 1000 matches a typical host user, so files the agent writes into the
# mounted workspace belong to you rather than to root.
RUN useradd --create-home --uid 1000 app

COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    POS_DATA_DIR=/data \
    POS_ROOT_DIR=/workspace

RUN mkdir -p /data /workspace && chown app:app /data /workspace
USER app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "pos.app:app", "--host", "0.0.0.0", "--port", "8000"]
