# Worker image. Carries the heavy runtime: Playwright from Phase 6, Tesseract
# and spaCy from Phase 9. Expect ~2 GB once those land — which is precisely why
# it is separate from the api.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

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

RUN pip install --no-cache-dir .

# Phase 6 adds:  RUN playwright install --with-deps chromium
# Phase 9 adds:  RUN apt-get install -y tesseract-ocr && python -m spacy download en_core_web_lg

RUN python scripts/check_forbidden_deps.py

RUN useradd --create-home --uid 10001 iie && chown -R iie:iie /app
USER iie

CMD ["arq", "worker.main.WorkerSettings"]
