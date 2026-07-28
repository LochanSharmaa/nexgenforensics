# NexGen iMATCH API image.
#
# Build from the backend/ directory:
#   docker build -f deployment/Dockerfile -t nexgen-imatch-api:1.0.0 .
#
# The recognition model pack (~350 MB) is NOT baked in. It downloads on first
# start into /models, which docker-compose mounts as a named volume so the
# download happens once rather than on every container start. For a fully
# offline image, copy a pre-downloaded pack into /models at build time and set
# NEXGEN_ENGINE_MODE=real so a missing pack fails the container instead of
# silently degrading to the stub.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime needs libgomp; opencv-headless needs libglib. Both are runtime
# libraries, so they stay in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-engine.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-engine.txt

COPY nexgen_engine /app/nexgen_engine
COPY imatch_api /app/imatch_api
COPY scripts /app/scripts
COPY pyproject.toml /app/

# Run unprivileged. A container that decodes untrusted image uploads should not
# be able to write outside its own data directories.
RUN useradd --system --create-home --uid 10001 imatch \
    && mkdir -p /models /data/storage \
    && chown -R imatch:imatch /app /models /data
USER imatch

ENV NEXGEN_MODEL_ROOT=/models \
    NEXGEN_STORAGE_ROOT=/data/storage \
    NEXGEN_AUDIT_LOG_PATH=/data/audit.jsonl

EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8443/api/health || exit 1

# One worker by default: each worker loads its own copy of the model (hundreds
# of MB) and keeps its own in-memory gallery. Scale with replicas, and move the
# gallery to a shared vector store before scaling workers inside a container.
CMD ["uvicorn", "imatch_api.main:app", "--host", "0.0.0.0", "--port", "8443", "--workers", "1"]
