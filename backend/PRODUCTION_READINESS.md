# Production Readiness Checklist

## Complete in This Repository

- Runnable FastAPI service.
- JSON/base64 API support without multipart dependency.
- Tenant-scoped enrollment and identification.
- Eight-backbone engine interface with local deterministic fallback.
- Quality scoring, TTA, fusion, cohort normalization, vector search.
- Template encryption and hash-chain audit logging.
- Training losses, curriculum, hard-negative mining, benchmark utilities.
- Export manifests, client package manifests, analytics reports.
- Docker, Compose, nginx, Kubernetes, and hybrid deployment configs.
- Local validation script: `python scripts/validate_system.py`.
- Optional Torch/timm and FAISS adapter hooks with deterministic fallback mode.
- CLI/runtime capability detection for production dependency readiness.

## Required Before Real Commercial Claims

- Replace deterministic fallback backbones with trained PyTorch/timm/InsightFace weights.
- Install and validate production dependencies: PyTorch, timm, InsightFace, FAISS GPU, CUDA/TensorRT.
- Run independent benchmarks on approved datasets.
- Complete privacy, legal, and security review.
- Validate demographic performance and failure modes.
- Configure production secrets, TLS, storage, monitoring, and incident response.
- Document customer-specific data retention and deletion policies.
