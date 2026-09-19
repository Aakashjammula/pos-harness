# Build context is the repo root (see docker-compose.yml).
FROM python:3.14-slim AS base

# libsndfile/portaudio for sounddevice + onnxruntime's native deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN uv sync --frozen --no-dev

COPY backend/src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Run as an unprivileged user. uid 1000 matches a typical host user, so model
# files written into the bind-mounted ./models belong to you, not to root (which
# would make them undeletable without sudo). The mount targets are created up
# front so Docker doesn't create them as root.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface /home/app/nltk_data \
    && chown -R app:app /app /home/app
USER app
ENV HOME=/home/app

EXPOSE 8000

CMD ["pos-server"]
