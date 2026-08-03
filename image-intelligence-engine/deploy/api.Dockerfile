# API image. Deliberately lean — no browser, no OCR, no NLP models.
# Those belong to the worker (ARCHITECTURE §3).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tooling for wheels that lack a manylinux build, removed in the same
# layer so it never reaches the final image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY shared ./shared
COPY database ./database
COPY api ./api
COPY worker ./worker
COPY scripts ./scripts
COPY alembic.ini ./

RUN pip install --no-cache-dir . \
 && apt-get purge -y build-essential && apt-get autoremove -y

# The forbidden-dependency guard runs at build time as well as in CI, so an
# image can never be produced containing a face-recognition library.
RUN python scripts/check_forbidden_deps.py

RUN useradd --create-home --uid 10001 iie && chown -R iie:iie /app
USER iie

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
