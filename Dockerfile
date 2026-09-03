# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Resolve and install exactly the dependencies recorded in uv.lock. Copying the
# metadata first keeps this layer reusable when only application code changes.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY main.py ./main.py
RUN uv sync --locked --no-dev --no-editable

FROM builder AS test

COPY tests ./tests
COPY scripts ./scripts
CMD ["uv", "run", "--locked", "--no-dev", "python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM ${PYTHON_IMAGE} AS production

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 uma-bct \
    && useradd --system --uid 10001 --gid uma-bct \
        --create-home --home-dir /home/uma-bct uma-bct \
    && mkdir -p /data \
    && chown uma-bct:uma-bct /data

COPY --from=builder --chown=uma-bct:uma-bct /app/.venv /app/.venv

USER uma-bct
WORKDIR /data

ENTRYPOINT ["uma-bct"]
CMD []
