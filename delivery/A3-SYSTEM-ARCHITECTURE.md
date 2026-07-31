# A3 — System Architecture

**Generated:** 2026-07-31 19:37 UTC ·
**Repository state:** `ea56818763c9`

| | |
|---|---|
| Python modules in tree | 102 |
| **Reachable from the running service** | **56** |
| Present but not reached | 46 |
| HTTP endpoints | 36 |
| Persisted tables | 9 |

---

## How to read this document

Module reachability is **computed** by walking the import graph from
`imatch_api.main`, not asserted from the directory listing.

That distinction is load-bearing. The source tree contains more modules than the
service uses: some are instrumentation, some are approaches that were superseded
and left in place, and some implement capabilities the product does not claim.
Listing every file as part of the delivered system would let a reader conclude
that a capability exists because a file named after it is present — for a
forensic system, that is exactly the wrong inference to invite.

**Both halves are published.** For a maintainer the unreachable list is the more
useful one, and for anyone assessing what the system actually does it is the
half that matters.

---

# Part I — Runtime architecture

## Process model

```
  Browser (React SPA, Vite)
        |  HTTPS, Bearer token or HTTPOnly cookie
        v
  FastAPI application  (imatch_api.main:app)
        |
        +-- middleware, outermost first
        |     CORSMiddleware            headers on every response, incl. refusals
        |     csrf_guard                double-submit on cookie-borne changes
        |     request_context           correlation id + security headers
        |
        +-- routers   auth, account, cases, subjects, search, audit, reports, admin, health
        |
        +-- services  engine_service, audit_service, accounts, mail, report_pdf
        |
        +-- db        SQLModel over SQLite or PostgreSQL
        |
        v
  nexgen_engine   (in-process, not a separate service)
        detection -> alignment -> quality -> embedding -> matching
        |
        v
  ONNX Runtime, CUDA execution provider
```

The recognition engine runs **in-process**. There is no model server: a request
thread calls the ONNX session directly. This bounds throughput (see A5,
concurrency: threading saturates at about four workers) and is the reason
batching, rather than more workers, is the scaling lever.

## Request path — 1:1 verification

1. `POST /api/imatch/verify` with two base64 images and a lawful basis.
2. Authentication resolves a principal from `Authorization` or `X-API-Key`.
3. Rate limit checked against the principal.
4. Lawful basis required — refused if absent when enforcement is on.
5. Each image: decode → **geometry guard** (min 16px edge, max 50:1 aspect) →
   detect → align to 112×112 → quality assessment → embed.
6. Cosine similarity, compared against the configured threshold.
7. Audit record written and hash-chained before the response is returned.

The geometry guard at step 5 exists because malformed input previously reached
the detector and raised from inside OpenCV, producing a 500 (A2, F-08).

## Request path — 1:N identification

As above through step 5, then the probe embedding is compared against the
tenant's gallery shard via exact inner-product search, ranked, and truncated to
`top_k`. Exact search is deliberate: approximate indexing was measured and its
recall loss was not judged acceptable for the lead-generation task (A5, ANN).

---

# Part II — Module inventory

## Reachable from the running service

These modules are executed, directly or transitively, by the delivered service.

| Module | File |
|---|---|
| `imatch_api` | `backend/imatch_api/__init__.py` |
| `imatch_api.api` | `backend/imatch_api/api/__init__.py` |
| `imatch_api.api.routes` | `backend/imatch_api/api/routes/__init__.py` |
| `imatch_api.api.routes.account` | `backend/imatch_api/api/routes/account.py` |
| `imatch_api.api.routes.admin` | `backend/imatch_api/api/routes/admin.py` |
| `imatch_api.api.routes.audit` | `backend/imatch_api/api/routes/audit.py` |
| `imatch_api.api.routes.auth` | `backend/imatch_api/api/routes/auth.py` |
| `imatch_api.api.routes.cases` | `backend/imatch_api/api/routes/cases.py` |
| `imatch_api.api.routes.health` | `backend/imatch_api/api/routes/health.py` |
| `imatch_api.api.routes.reports` | `backend/imatch_api/api/routes/reports.py` |
| `imatch_api.api.routes.search` | `backend/imatch_api/api/routes/search.py` |
| `imatch_api.api.routes.subjects` | `backend/imatch_api/api/routes/subjects.py` |
| `imatch_api.api.schemas` | `backend/imatch_api/api/schemas.py` |
| `imatch_api.core` | `backend/imatch_api/core/__init__.py` |
| `imatch_api.core.config` | `backend/imatch_api/core/config.py` |
| `imatch_api.core.csrf` | `backend/imatch_api/core/csrf.py` |
| `imatch_api.core.dependencies` | `backend/imatch_api/core/dependencies.py` |
| `imatch_api.core.rate_limit` | `backend/imatch_api/core/rate_limit.py` |
| `imatch_api.core.security` | `backend/imatch_api/core/security.py` |
| `imatch_api.db` | `backend/imatch_api/db/__init__.py` |
| `imatch_api.db.models` | `backend/imatch_api/db/models.py` |
| `imatch_api.db.session` | `backend/imatch_api/db/session.py` |
| `imatch_api.main` | `backend/imatch_api/main.py` |
| `imatch_api.services` | `backend/imatch_api/services/__init__.py` |
| `imatch_api.services.accounts` | `backend/imatch_api/services/accounts.py` |
| `imatch_api.services.audit_service` | `backend/imatch_api/services/audit_service.py` |
| `imatch_api.services.engine_service` | `backend/imatch_api/services/engine_service.py` |
| `imatch_api.services.mail` | `backend/imatch_api/services/mail.py` |
| `imatch_api.services.mail_templates` | `backend/imatch_api/services/mail_templates.py` |
| `imatch_api.services.report_pdf` | `backend/imatch_api/services/report_pdf.py` |
| `imatch_api.services.report_service` | `backend/imatch_api/services/report_service.py` |
| `imatch_api.services.storage_service` | `backend/imatch_api/services/storage_service.py` |
| `nexgen_engine` | `backend/nexgen_engine/__init__.py` |
| `nexgen_engine.config` | `backend/nexgen_engine/config.py` |
| `nexgen_engine.data` | `backend/nexgen_engine/data/__init__.py` |
| `nexgen_engine.data.quality_filter` | `backend/nexgen_engine/data/quality_filter.py` |
| `nexgen_engine.detection` | `backend/nexgen_engine/detection/__init__.py` |
| `nexgen_engine.detection.alignment` | `backend/nexgen_engine/detection/alignment.py` |
| `nexgen_engine.detection.detector` | `backend/nexgen_engine/detection/detector.py` |
| `nexgen_engine.detection.types` | `backend/nexgen_engine/detection/types.py` |
| `nexgen_engine.inference` | `backend/nexgen_engine/inference/__init__.py` |
| `nexgen_engine.inference.pipeline` | `backend/nexgen_engine/inference/pipeline.py` |
| `nexgen_engine.inference.score_fusion` | `backend/nexgen_engine/inference/score_fusion.py` |
| `nexgen_engine.models` | `backend/nexgen_engine/models/__init__.py` |
| `nexgen_engine.models.arcface` | `backend/nexgen_engine/models/arcface.py` |
| `nexgen_engine.models.cuda_runtime` | `backend/nexgen_engine/models/cuda_runtime.py` |
| `nexgen_engine.observability` | `backend/nexgen_engine/observability.py` |
| `nexgen_engine.runtime` | `backend/nexgen_engine/runtime.py` |
| `nexgen_engine.search` | `backend/nexgen_engine/search/__init__.py` |
| `nexgen_engine.search.gallery_index` | `backend/nexgen_engine/search/gallery_index.py` |
| `nexgen_engine.security` | `backend/nexgen_engine/security/__init__.py` |
| `nexgen_engine.security.deepfake_detector` | `backend/nexgen_engine/security/deepfake_detector.py` |
| `nexgen_engine.security.liveness` | `backend/nexgen_engine/security/liveness.py` |
| `nexgen_engine.security.morphing_detector` | `backend/nexgen_engine/security/morphing_detector.py` |
| `nexgen_engine.security.template_encryption` | `backend/nexgen_engine/security/template_encryption.py` |
| `nexgen_engine.utils` | `backend/nexgen_engine/utils.py` |

## Present in the tree but NOT reached from the service

**46 modules.** Reasons vary and are not interchangeable:

- **Instrumentation** — benchmark and audit scripts invoked directly from the
  command line rather than by the service. These are part of the delivery and
  their outputs are in A5; they simply are not in the request path.
- **Superseded approaches** — earlier implementations retained in history.
- **Unclaimed capability** — a module whose name suggests a feature the product
  does not claim. The presence of a file is not a claim that the capability is
  delivered, and CLAIMS.md is the authority on what is claimed.

A reader assessing what this system does should treat this list as **not part of
the running system** unless a specific module is shown to be invoked by an
operator-facing workflow.

### 17 module(s)

Training and dataset tooling — used to produce and audit models offline. Never in a request path by design.

| Module | File |
|---|---|
| `nexgen_engine.data.augmentation` | `backend/nexgen_engine/data/augmentation.py` |
| `nexgen_engine.data.dataset_builder` | `backend/nexgen_engine/data/dataset_builder.py` |
| `nexgen_engine.data.ingestion_validator` | `backend/nexgen_engine/data/ingestion_validator.py` |
| `nexgen_engine.data.loader` | `backend/nexgen_engine/data/loader.py` |
| `nexgen_engine.data.manifest` | `backend/nexgen_engine/data/manifest.py` |
| `nexgen_engine.data.synthetic_gen` | `backend/nexgen_engine/data/synthetic_gen.py` |
| `nexgen_engine.losses` | `backend/nexgen_engine/losses/__init__.py` |
| `nexgen_engine.losses.combined` | `backend/nexgen_engine/losses/combined.py` |
| `nexgen_engine.losses.metric_losses` | `backend/nexgen_engine/losses/metric_losses.py` |
| `nexgen_engine.training` | `backend/nexgen_engine/training/__init__.py` |
| `nexgen_engine.training.arcface_loss` | `backend/nexgen_engine/training/arcface_loss.py` |
| `nexgen_engine.training.curriculum` | `backend/nexgen_engine/training/curriculum.py` |
| `nexgen_engine.training.dataset` | `backend/nexgen_engine/training/dataset.py` |
| `nexgen_engine.training.hard_negative_miner` | `backend/nexgen_engine/training/hard_negative_miner.py` |
| `nexgen_engine.training.scheduler` | `backend/nexgen_engine/training/scheduler.py` |
| `nexgen_engine.training.train_pipeline` | `backend/nexgen_engine/training/train_pipeline.py` |
| `nexgen_engine.training.trainer` | `backend/nexgen_engine/training/trainer.py` |

### 13 module(s)

Instrumentation — invoked from the command line, not the service. Part of the delivery; outputs are in A5.

| Module | File |
|---|---|
| `nexgen_engine.analytics` | `backend/nexgen_engine/analytics/__init__.py` |
| `nexgen_engine.analytics.accuracy_tracker` | `backend/nexgen_engine/analytics/accuracy_tracker.py` |
| `nexgen_engine.analytics.metrics` | `backend/nexgen_engine/analytics/metrics.py` |
| `nexgen_engine.analytics.report_generator` | `backend/nexgen_engine/analytics/report_generator.py` |
| `nexgen_engine.benchmarks` | `backend/nexgen_engine/benchmarks/__init__.py` |
| `nexgen_engine.benchmarks.aging_eval` | `backend/nexgen_engine/benchmarks/aging_eval.py` |
| `nexgen_engine.benchmarks.ijbc_eval` | `backend/nexgen_engine/benchmarks/ijbc_eval.py` |
| `nexgen_engine.benchmarks.lfw_eval` | `backend/nexgen_engine/benchmarks/lfw_eval.py` |
| `nexgen_engine.benchmarks.metrics` | `backend/nexgen_engine/benchmarks/metrics.py` |
| `nexgen_engine.benchmarks.nist_eval` | `backend/nexgen_engine/benchmarks/nist_eval.py` |
| `nexgen_engine.benchmarks.report_generator` | `backend/nexgen_engine/benchmarks/report_generator.py` |
| `nexgen_engine.benchmarks.speed_benchmark` | `backend/nexgen_engine/benchmarks/speed_benchmark.py` |
| `nexgen_engine.benchmarks.verification` | `backend/nexgen_engine/benchmarks/verification.py` |

### 7 module(s)

Not reached from the entry point; no specific role identified.

| Module | File |
|---|---|
| `nexgen_engine.cli` | `backend/nexgen_engine/cli.py` |
| `nexgen_engine.inference.cohort_normalizer` | `backend/nexgen_engine/inference/cohort_normalizer.py` |
| `nexgen_engine.inference.embedding_extractor` | `backend/nexgen_engine/inference/embedding_extractor.py` |
| `nexgen_engine.inference.tta` | `backend/nexgen_engine/inference/tta.py` |
| `nexgen_engine.models.backbones` | `backend/nexgen_engine/models/backbones.py` |
| `nexgen_engine.models.insightface_backbone` | `backend/nexgen_engine/models/insightface_backbone.py` |
| `nexgen_engine.search.persistence` | `backend/nexgen_engine/search/persistence.py` |

### 4 module(s)

Packaging and model-export tooling — offline use.

| Module | File |
|---|---|
| `nexgen_engine.export` | `backend/nexgen_engine/export/__init__.py` |
| `nexgen_engine.export.export_onnx` | `backend/nexgen_engine/export/export_onnx.py` |
| `nexgen_engine.export.export_trt` | `backend/nexgen_engine/export/export_trt.py` |
| `nexgen_engine.export.package_for_client` | `backend/nexgen_engine/export/package_for_client.py` |

### 3 module(s)

Superseded — an earlier in-engine API layer, replaced by `imatch_api`. Retained in history.

| Module | File |
|---|---|
| `nexgen_engine.api` | `backend/nexgen_engine/api/__init__.py` |
| `nexgen_engine.api.schemas` | `backend/nexgen_engine/api/schemas.py` |
| `nexgen_engine.api.service` | `backend/nexgen_engine/api/service.py` |

### 1 module(s)

Superseded — the service uses `imatch_api.services.audit_service` instead.

| Module | File |
|---|---|
| `nexgen_engine.security.audit_logger` | `backend/nexgen_engine/security/audit_logger.py` |

### 1 module(s)

**Unclaimed capability — deliberately not wired in.** The product does NOT claim presentation-attack detection; the liveness signal it does report is a heuristic marked `certified: false`. This module being unreached is consistent with that claim, not an oversight.

| Module | File |
|---|---|
| `nexgen_engine.security.presentation_attack` | `backend/nexgen_engine/security/presentation_attack.py` |


---

# Part III — HTTP surface

Extracted from the FastAPI route decorators.

| Method | Path | Handler |
|---|---|---|
| GET | `/api/admin/api-keys` | `list_api_keys` |
| POST | `/api/admin/api-keys` | `create_api_key` |
| DELETE | `/api/admin/api-keys/{key_id}` | `revoke_api_key` |
| GET | `/api/audit` | `list_audit_records` |
| GET | `/api/audit/verify` | `verify_chain` |
| GET | `/api/auth/csrf` | `csrf_token` |
| POST | `/api/auth/forgot-password` | `forgot_password` |
| POST | `/api/auth/login` | `login` |
| POST | `/api/auth/logout` | `logout` |
| GET | `/api/auth/me` | `me` |
| POST | `/api/auth/refresh` | `refresh` |
| POST | `/api/auth/register` | `register` |
| POST | `/api/auth/resend-otp` | `resend_otp` |
| POST | `/api/auth/reset-password` | `reset_password` |
| POST | `/api/auth/users` | `create_user` |
| POST | `/api/auth/verify-email` | `verify_email` |
| GET | `/api/cases` | `list_cases` |
| POST | `/api/cases` | `create_case` |
| GET | `/api/cases/{case_id}` | `get_case` |
| PATCH | `/api/cases/{case_id}` | `update_case` |
| GET | `/api/cases/{case_id}/report` | `export_report` |
| GET | `/api/health` | `health` |
| POST | `/api/imatch/batch` | `batch` |
| POST | `/api/imatch/candidates/{candidate_id}/adjudicate` | `adjudicate` |
| GET | `/api/imatch/engine/metrics` | `engine_metrics` |
| GET | `/api/imatch/engine/status` | `engine_status` |
| POST | `/api/imatch/search` | `search` |
| GET | `/api/imatch/searches` | `list_searches` |
| GET | `/api/imatch/searches/{search_id}/candidates` | `list_candidates` |
| POST | `/api/imatch/verify` | `verify` |
| GET | `/api/subjects` | `list_subjects` |
| POST | `/api/subjects` | `enrol` |
| DELETE | `/api/subjects/{subject_id}` | `delete_subject` |
| GET | `/api/subjects/{subject_id}` | `get_subject` |
| GET | `/api/subjects/{subject_id}/templates` | `list_templates` |
| DELETE | `/api/subjects/{subject_id}/templates/{template_id}` | `delete_template` |

---

# Part IV — Persisted data model

## `Tenant`

| Field |
|---|
| `id` |
| `slug` |
| `name` |
| `active` |
| `created_at` |

## `User`

| Field |
|---|
| `id` |
| `tenant_id` |
| `email` |
| `full_name` |
| `password_hash` |
| `role` |
| `active` |
| `badge_number` |
| `last_login_at` |
| `created_at` |
| `email_verified` |
| `otp_hash` |
| `otp_expires_at` |
| `otp_attempts` |
| `otp_sent_count` |
| `otp_window_started_at` |
| `reset_token_hash` |
| `reset_token_expires_at` |
| `failed_login_attempts` |
| `locked_until` |
| `refresh_token_hash` |
| `refresh_token_expires_at` |
| `session_epoch` |
| `last_login_ip` |
| `updated_at` |

## `ApiKey`

| Field |
|---|
| `id` |
| `tenant_id` |
| `name` |
| `prefix` |
| `key_hash` |
| `role` |
| `active` |
| `created_by` |
| `last_used_at` |
| `expires_at` |
| `created_at` |

## `Case`

| Field |
|---|
| `id` |
| `tenant_id` |
| `reference` |
| `title` |
| `description` |
| `status` |
| `lawful_basis` |
| `owner_id` |
| `created_at` |
| `updated_at` |
| `closed_at` |

## `Subject`

| Field |
|---|
| `id` |
| `tenant_id` |
| `external_ref` |
| `display_name` |
| `notes` |
| `case_id` |
| `enrolled_by` |
| `active` |
| `created_at` |

## `Template`

| Field |
|---|
| `id` |
| `tenant_id` |
| `subject_id` |
| `nonce` |
| `ciphertext` |
| `dimensions` |
| `recognizer_backend` |
| `recognizer_pack` |
| `quality_score` |
| `detector` |
| `image_sha256` |
| `image_path` |
| `created_by` |
| `created_at` |

## `SearchRun`

| Field |
|---|
| `id` |
| `tenant_id` |
| `case_id` |
| `operator_id` |
| `mode` |
| `lawful_basis` |
| `purpose` |
| `decision` |
| `top_score` |
| `margin` |
| `gallery_size` |
| `candidate_count` |
| `quality_score` |
| `liveness_score` |
| `deepfake_risk` |
| `faces_detected` |
| `review_required` |
| `recognition_capable` |
| `reasons` |
| `explanation` |
| `recognizer_backend` |
| `recognizer_pack` |
| `match_threshold` |
| `review_threshold` |
| `probe_sha256` |
| `probe_path` |
| `duration_ms` |
| `audit_hash` |
| `created_at` |

## `Candidate`

| Field |
|---|
| `id` |
| `tenant_id` |
| `search_run_id` |
| `subject_id` |
| `template_id` |
| `rank` |
| `score` |
| `normalized_score` |
| `adjudication` |
| `adjudicated_by` |
| `adjudicated_at` |
| `examiner_notes` |

## `AuditRecord`

| Field |
|---|
| `id` |
| `tenant_id` |
| `sequence` |
| `actor_id` |
| `actor_label` |
| `action` |
| `resource_type` |
| `resource_id` |
| `outcome` |
| `lawful_basis` |
| `detail` |
| `ip_address` |
| `user_agent` |
| `previous_hash` |
| `entry_hash` |
| `created_at` |


---

# Part V — Core implementation

The modules on which the recognition result depends, in full.

## `backend/nexgen_engine/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class QualityConfig:
    """Gates applied to a probe before it is allowed to reach the matcher."""

    min_quality_score: float = 0.35
    min_detection_confidence: float = 0.70
    min_face_pixels: int = 60
    max_yaw: float = 35.0
    max_pitch: float = 30.0
    max_roll: float = 25.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast_score: float = 0.30
    min_sharpness_score: float = 0.30
    keep_top_fraction: float = 0.65


@dataclass(frozen=True)
class ThresholdConfig:
    """Cosine-similarity operating points on L2-normalized ArcFace templates.

    Defaults are the widely used ArcFace/buffalo_l starting points, not
    guarantees. Every deployment must recalibrate against its own imagery --
    see ``scripts/calibrate_threshold.py`` -- because the false-match rate at a
    given threshold depends heavily on gallery size and capture conditions.
    """

    # CALIBRATED FOR FALSE-MATCH CONTROL, not for peak accuracy.
    # Decision record and full measured tradeoff: BENCHMARKS.md section 5c.
    #
    # 0.2871 is the FMR=0.1% operating point for w600k_r50 (the pack this
    # service loads), measured over 40,098 AgeDB pairs.
    #
    # It is deliberately NOT the accuracy-maximising value. That was 0.20, the
    # 10-fold cross-validated optimum, and it carried FMR = 1.19% -- roughly
    # one false match per 84 impostor comparisons. That was demonstrated in
    # practice: two different people scored 0.2405 and the system reported the
    # comparison as supporting the same person.
    #
    # For forensic use the objective is not accuracy. A missed lead costs a
    # re-check; a false match points investigators at an innocent person. The
    # measured trade is a 12x reduction in false matches (1.19% -> 0.10%) for
    # roughly double the miss rate (FNMR 3.30% -> 6.32%).
    #
    # Model-specific: re-derive with backend/scripts/benchmark_demographics.py
    # (omit --threshold to get the FMR=0.1% point) after ANY change to the
    # model pack or embedding pipeline. Do not copy this number to another pack.
    match: float = 0.2871
    # Review band kept at the previous 0.75 ratio to the match threshold, so
    # near-miss candidates still reach a human. This ratio was not separately
    # optimised and is a carried-forward convention, not a measured value.
    review: float = 0.2153
    verify: float = 0.2871


@dataclass(frozen=True)
class SecurityConfig:
    # Passive single-image screens. These are heuristics, not certified
    # presentation-attack detection; see nexgen_engine/security/liveness.py.
    liveness_threshold: float = 0.45
    deepfake_threshold: float = 0.65
    morphing_threshold: float = 0.55
    template_key_bytes: int = 32
    audit_hash_algorithm: str = "sha256"


@dataclass(frozen=True)
class SearchConfig:
    embedding_dim: int = 512
    top_k: int = 20
    # Candidates below this are never returned, regardless of top_k.
    min_candidate_score: float = 0.20


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the recognition engine.

    There is no "mode" any more. The engine either loads real weights or raises
    ``EngineUnavailableError``; earlier revisions had an ``auto`` mode that fell
    back to hashing pixels into a vector, which kept the API answering while
    making every score meaningless.

    ``device`` is a request. CUDA is used when the CUDA execution provider is
    actually registered, otherwise the engine runs on CPU and says so. The model
    and the arithmetic are identical either way, so results do not depend on it.
    """

    model_pack: str = "buffalo_l"
    model_root: str | None = None
    # "auto" picks CUDA when it genuinely binds (verified with a real ONNX
    # session, not the build-time provider list) and CPU otherwise.
    device: str = "auto"
    # Upper bound for the detector input. The detector picks the smallest size
    # that fits each image, so this is a ceiling and not a fixed cost.
    detection_size: tuple[int, int] = (640, 640)
    min_detection_confidence: float = 0.5
    embedding_dim: int = 512
    # Average the template over the crop and its mirror. Horizontal flip is the
    # only augmentation that is safe here: brightness and sharpening shifts move
    # the template off the manifold ArcFace was trained on and cost accuracy.
    use_flip_tta: bool = True

    quality: QualityConfig = field(default_factory=QualityConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"Unknown device {self.device!r}; expected cpu, cuda, or auto.")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0, 1].")

    def with_overrides(self, **changes: object) -> "EngineConfig":
        return replace(self, **changes)  # type: ignore[arg-type]


__all__ = [
    "EngineConfig",
    "QualityConfig",
    "SearchConfig",
    "SecurityConfig",
    "ThresholdConfig",
]

```

## `backend/nexgen_engine/inference/pipeline.py`

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import EngineConfig
from ..data.quality_filter import ImageQualityFilter, QualityReport
from ..detection.alignment import FaceAligner
from ..detection.types import DetectedFace
from ..runtime import EngineRuntime
from ..security.deepfake_detector import DeepfakeDetector
from ..security.liveness import LivenessDetector, LivenessReport
from ..utils import l2_normalize


class NoFaceDetectedError(ValueError):
    """Raised when an image contains no usable face."""


class InvalidImageError(ValueError):
    """Raised when the supplied bytes are not a decodable image."""


@dataclass(frozen=True)
class StageTimings:
    """Per-stage wall-clock cost, in milliseconds."""

    decode_ms: float = 0.0
    detect_ms: float = 0.0
    align_ms: float = 0.0
    embed_ms: float = 0.0
    quality_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "decode_ms": self.decode_ms,
            "detect_ms": self.detect_ms,
            "align_ms": self.align_ms,
            "embed_ms": self.embed_ms,
            "quality_ms": self.quality_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class RecognitionResult:
    """Everything derived from one probe image.

    ``embedding`` is the L2-normalized template used for matching: the raw model
    output averaged over flip-TTA and renormalized. It is deliberately not
    adjusted by any query-dependent state, so the same image always produces the
    same template regardless of what was searched before it.
    """

    embedding: np.ndarray
    face: DetectedFace
    quality: QualityReport
    liveness: LivenessReport
    deepfake_risk: float
    faces_detected: int
    detector_name: str
    padded_detection: bool
    timings: StageTimings
    review_required: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    # Retained for API compatibility. The engine cannot start without a real
    # model, so reaching this object at all means recognition is live.
    recognition_capable: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "quality": self.quality.as_dict(),
            "liveness": self.liveness.as_dict(),
            "deepfake_risk": self.deepfake_risk,
            "faces_detected": self.faces_detected,
            "recognition_capable": True,
            "detector": self.detector_name,
            "padded_detection": self.padded_detection,
            "timings": self.timings.as_dict(),
            "review_required": self.review_required,
            "reasons": list(self.reasons),
            "box": {
                "left": self.face.box.left,
                "top": self.face.box.top,
                "right": self.face.box.right,
                "bottom": self.face.box.bottom,
            },
            "pose": {
                "yaw": self.face.pose.yaw,
                "pitch": self.face.pose.pitch,
                "roll": self.face.pose.roll,
            },
        }


class FacialRecognitionPipeline:
    """Image bytes in, comparable biometric template out.

    Stages: decode -> detect -> select face -> quality -> align -> embed.

    Two things this deliberately does NOT do, because both corrupt matching:

    * It does not fuse unrelated backbones through a random projection. A
      projection never trained jointly with the encoder destroys the metric
      structure ArcFace learned.
    * It does not adjust the stored template using statistics from previous
      queries. Cohort normalization belongs at the score level, where it cannot
      make a template depend on search history.
    """

    def __init__(self, config: EngineConfig | None = None, runtime: EngineRuntime | None = None) -> None:
        self.config = config or EngineConfig()
        self.runtime = runtime or EngineRuntime(self.config)
        self.quality_filter = ImageQualityFilter(self.config.quality)
        self.aligner = FaceAligner()
        self.liveness = LivenessDetector(self.config.security.liveness_threshold)
        self.deepfake = DeepfakeDetector(self.config.security.deepfake_threshold)

    # ------------------------------------------------------------------ api --

    def encode_bytes(self, image_bytes: bytes) -> RecognitionResult:
        started = time.perf_counter()
        image = decode_image(image_bytes)
        decode_ms = (time.perf_counter() - started) * 1000
        return self.encode_image(image, decode_ms=decode_ms, started=started)

    def encode_image(
        self,
        image: Image.Image,
        decode_ms: float = 0.0,
        started: float | None = None,
    ) -> RecognitionResult:
        started = started if started is not None else time.perf_counter()
        image = image.convert("RGB")

        outcome = self.runtime.detector.detect_detailed(image)
        if not outcome.faces:
            raise NoFaceDetectedError(
                "No face was detected. The image may not contain a face, the face may be too "
                "small, or the detection confidence threshold may be too high."
            )

        face = self._select_face(list(outcome.faces))

        mark = time.perf_counter()
        quality = self.quality_filter.evaluate(image, face)
        quality_ms = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        crop = self.aligner.align(image, face)
        align_ms = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        embedding = self._embed(crop)
        embed_ms = (time.perf_counter() - mark) * 1000

        liveness = self.liveness.analyze(crop)
        deepfake_risk = self.deepfake.risk_score(crop)

        reasons = list(quality.reasons)
        if not liveness.passed:
            reasons.append("liveness_below_threshold")
            reasons.extend(liveness.reasons)
        if deepfake_risk >= self.config.security.deepfake_threshold:
            reasons.append("synthetic_media_risk")
        if len(outcome.faces) > 1:
            reasons.append("multiple_faces_detected")

        return RecognitionResult(
            embedding=embedding,
            face=face,
            quality=quality,
            liveness=liveness,
            deepfake_risk=deepfake_risk,
            faces_detected=len(outcome.faces),
            detector_name=self.runtime.detector.name,
            padded_detection=outcome.padded,
            timings=StageTimings(
                decode_ms=round(decode_ms, 2),
                detect_ms=outcome.elapsed_ms,
                align_ms=round(align_ms, 2),
                embed_ms=round(embed_ms, 2),
                quality_ms=round(quality_ms, 2),
                total_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
            review_required=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def encode_all_faces(self, image: Image.Image) -> list[RecognitionResult]:
        """Encode every detected face. Used by group-photo and batch intake.

        All crops go through the recognizer in a single batch, which is
        substantially cheaper than one call per face.
        """
        image = image.convert("RGB")
        outcome = self.runtime.detector.detect_detailed(image)
        if not outcome.faces:
            return []

        faces = list(outcome.faces)
        crops = [self.aligner.align(image, face) for face in faces]

        mark = time.perf_counter()
        embeddings = self._embed_many(crops)
        embed_ms = (time.perf_counter() - mark) * 1000

        results: list[RecognitionResult] = []
        for face, crop, embedding in zip(faces, crops, embeddings):
            quality = self.quality_filter.evaluate(image, face)
            liveness = self.liveness.analyze(crop)
            reasons = list(quality.reasons)
            if not liveness.passed:
                reasons.append("liveness_below_threshold")
            results.append(
                RecognitionResult(
                    embedding=embedding,
                    face=face,
                    quality=quality,
                    liveness=liveness,
                    deepfake_risk=self.deepfake.risk_score(crop),
                    faces_detected=len(faces),
                    detector_name=self.runtime.detector.name,
                    padded_detection=outcome.padded,
                    timings=StageTimings(
                        detect_ms=outcome.elapsed_ms,
                        embed_ms=round(embed_ms / max(len(faces), 1), 2),
                    ),
                    review_required=bool(reasons),
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        return results

    # -------------------------------------------------------------- internal --

    def _select_face(self, faces: list[DetectedFace]) -> DetectedFace:
        """Pick the subject face: the largest that clears the confidence gate."""
        for face in faces:
            if face.confidence >= self.config.quality.min_detection_confidence:
                return face
        return faces[0]

    def _embed(self, crop: Image.Image) -> np.ndarray:
        """Template for one aligned crop, averaged with its mirror.

        Flip-TTA is a genuine small gain for ArcFace: a face and its mirror
        should encode to the same identity, so averaging cancels part of the
        pose-specific noise. Both crops go through in one batch.
        """
        crops = [crop, ImageOps.mirror(crop)] if self.config.use_flip_tta else [crop]
        embeddings = self.runtime.recognizer.embed_batch(crops)
        return l2_normalize(embeddings.mean(axis=0).astype(np.float32))

    def _embed_many(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """Batch-embed several crops, applying flip-TTA in the same batch."""
        if not crops:
            return []
        if not self.config.use_flip_tta:
            return list(self.runtime.recognizer.embed_batch(crops))

        batch = list(crops) + [ImageOps.mirror(crop) for crop in crops]
        embeddings = self.runtime.recognizer.embed_batch(batch)
        count = len(crops)
        return [
            l2_normalize(((embeddings[i] + embeddings[i + count]) / 2.0).astype(np.float32))
            for i in range(count)
        ]


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes to RGB, honouring EXIF orientation.

    Phone photos routinely carry an EXIF rotation flag. Ignoring it feeds the
    detector a sideways face and silently tanks the match rate.
    """
    if not image_bytes:
        raise InvalidImageError("Empty image payload.")
    try:
        with Image.open(BytesIO(image_bytes)) as handle:
            handle.load()
            image = ImageOps.exif_transpose(handle).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    # Reject degenerate geometry HERE, where it can still be a typed error.
    #
    # SCRFD scales the image to fit the detector input while preserving aspect
    # ratio: new_height = int(new_width * height / width). For a 4000x1 strip
    # that rounds to 0, and cv2.resize then raises a bare cv2.error
    # ("inv_scale_x > 0") from deep inside insightface. That exception is not
    # one the API maps to a 4xx, so it surfaced as a 500 -- an unreadable
    # failure for the operator and, in a batch, one that killed the request.
    #
    # Found by tests_engine/test_adversarial_input.py. A 1px-tall strip is a
    # realistic accident from a bad crop, not just a synthetic attack.
    #
    # The floor is the detector's minimum useful edge, not 1px: an image
    # smaller than this cannot contain a detectable face, so rejecting it with
    # a clear message beats letting the detector return nothing.
    min_edge = 16
    if image.width < min_edge or image.height < min_edge:
        raise InvalidImageError(
            f"Image is {image.width}x{image.height}; each side must be at least "
            f"{min_edge}px for face detection to be possible."
        )

    # Guard the ratio itself as well as the sides: 4000x20 clears the floor
    # above but still scales to a sub-pixel height at detector input size.
    longest, shortest = max(image.width, image.height), min(image.width, image.height)
    max_ratio = 50
    if longest / shortest > max_ratio:
        raise InvalidImageError(
            f"Image aspect ratio is {longest / shortest:.0f}:1 "
            f"({image.width}x{image.height}); the maximum supported is {max_ratio}:1. "
            "This usually means the image was cropped incorrectly."
        )

    return image


__all__ = [
    "FacialRecognitionPipeline",
    "InvalidImageError",
    "NoFaceDetectedError",
    "RecognitionResult",
    "StageTimings",
    "decode_image",
]

```

## `backend/nexgen_engine/search/gallery_index.py`

```python
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import l2_normalize

try:  # pragma: no cover - depends on host packages
    import faiss as _faiss  # noqa: F401

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAISS_AVAILABLE = False


def faiss_available() -> bool:
    """Whether exhaustive search runs through FAISS or numpy.

    Both are exact, so this affects speed only. Exposed so the status endpoint
    can report which path is live.
    """
    return _FAISS_AVAILABLE


@dataclass(frozen=True)
class MatchResult:
    """One candidate returned by a search."""

    template_id: str
    subject_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "subject_id": self.subject_id,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchOutcome:
    """Ranked candidates plus the statistics needed to judge them.

    ``all_scores`` is the full score vector over the searched tenant's gallery.
    The decision layer needs it to compute the runner-up margin and the impostor
    distribution; without it a top score cannot be told apart from a near-tie.
    """

    matches: tuple[MatchResult, ...]
    all_scores: np.ndarray
    gallery_size: int

    @property
    def top_score(self) -> float:
        return float(self.matches[0].score) if self.matches else 0.0

    @property
    def margin(self) -> float:
        """Gap between best and second-best *subject* (not template)."""
        if len(self.matches) < 2:
            return 0.0
        return float(self.matches[0].score - self.matches[1].score)


class _TenantShard:
    """All enrolled templates for a single tenant."""

    __slots__ = (
        "dimensions",
        "template_ids",
        "subject_ids",
        "metadata",
        "vectors",
        "_positions",
        "_index",
        "_index_rows",
    )

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.template_ids: list[str] = []
        self.subject_ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.vectors = np.empty((0, dimensions), dtype=np.float32)
        self._positions: dict[str, int] = {}
        # Rebuilt lazily whenever the row count changes.
        self._index = None
        self._index_rows = -1

    def add(self, template_id: str, subject_id: str, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        self._invalidate()
        if template_id in self._positions:
            self.remove(template_id)
        self._positions[template_id] = len(self.template_ids)
        self.template_ids.append(template_id)
        self.subject_ids.append(subject_id)
        self.metadata.append(metadata)
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)]) if self.vectors.size else vector.reshape(1, -1)

    def add_many(self, rows: list[tuple[str, str, np.ndarray, dict[str, Any]]]) -> None:
        """Bulk insert. Rebuilding a gallery row by row is O(n^2) in copies."""
        if not rows:
            return
        self._invalidate()
        stacked = np.vstack([vector.reshape(1, -1) for _, _, vector, _ in rows])
        start = len(self.template_ids)
        for offset, (template_id, subject_id, _, metadata) in enumerate(rows):
            self._positions[template_id] = start + offset
            self.template_ids.append(template_id)
            self.subject_ids.append(subject_id)
            self.metadata.append(metadata)
        self.vectors = np.vstack([self.vectors, stacked]) if self.vectors.size else stacked

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Similarity of ``query`` against every template in this shard.

        Uses FAISS ``IndexFlatIP`` when available and falls back to a numpy
        matmul otherwise. Both compute an exhaustive inner product over
        L2-normalized vectors, so the two paths are numerically equivalent and
        return the same ranking -- FAISS is a speed optimisation, never a
        different answer. That matters here: an approximate index would silently
        drop true candidates, and a missed lead is invisible to the examiner.
        """
        if self.vectors.size == 0:
            return np.empty(0, dtype=np.float32)

        index = self._faiss_index()
        if index is not None:
            total = len(self.template_ids)
            distances, ids = index.search(np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), total)
            # FAISS returns results ranked by score. Scatter them back into
            # gallery order so callers can index by position, exactly as the
            # numpy path does.
            ordered = np.zeros(total, dtype=np.float32)
            valid = ids[0] >= 0
            ordered[ids[0][valid]] = distances[0][valid]
            return ordered

        return (self.vectors @ query).astype(np.float32)

    def _faiss_index(self):  # noqa: ANN202 - faiss types are optional
        """Build (and cache) a FAISS flat index for this shard.

        DELIBERATE CHOICE, MEASURED -- read before "optimising" this.

        faiss is NOT installed and NOT declared in any requirements file, so
        this path is dormant and every search runs the numpy matmul in
        ``scores()``. That is intentional, not an oversight.

        Measured on an RTX A3000 host (BENCHMARKS.md 7b), brute-force cosine:

            1,000 templates    0.207 ms p50
           10,000 templates    1.087 ms p50
          100,000 templates   15.981 ms p50

        Below ~10k the search is under 8% of the ~14.7 ms it costs to encode
        the probe image, i.e. free. Adding a dependency to optimise 8% of the
        request would be premature.

        Note what this branch would and would not buy: ``IndexFlatIP`` is an
        EXACT inner-product index -- brute force with better SIMD. It is not an
        approximate-nearest-neighbour structure. Enabling faiss here would win a
        constant factor, NOT a change in complexity; the 100k cost would still
        grow linearly. Genuine scaling past ~100k needs an approximate index
        (IVF-PQ or HNSW) and the recall loss that comes with it, which must be
        measured and accepted explicitly rather than assumed.
        """
        if not _FAISS_AVAILABLE:
            return None
        if self._index is not None and self._index_rows == len(self.template_ids):
            return self._index
        import faiss

        index = faiss.IndexFlatIP(self.dimensions)
        index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
        self._index = index
        self._index_rows = len(self.template_ids)
        return index

    def _invalidate(self) -> None:
        self._index = None
        self._index_rows = -1

    def remove(self, template_id: str) -> bool:
        position = self._positions.pop(template_id, None)
        if position is None:
            return False
        self._invalidate()
        self.template_ids.pop(position)
        self.subject_ids.pop(position)
        self.metadata.pop(position)
        self.vectors = np.delete(self.vectors, position, axis=0)
        # Positions after the removed row all shift down by one.
        for key, value in self._positions.items():
            if value > position:
                self._positions[key] = value - 1
        return True

    def __len__(self) -> int:
        return len(self.template_ids)


class GalleryIndex:
    """In-memory vector gallery, partitioned by tenant.

    Tenant isolation is structural rather than advisory: each tenant's vectors
    live in a separate matrix, and ``search`` takes the tenant id as a required
    argument, so there is no code path that can compare a probe against another
    tenant's templates. A filter applied after a global search would be one
    forgotten predicate away from a cross-tenant biometric leak.

    Cosine similarity on L2-normalized templates reduces to a dot product, so a
    brute-force matmul is exact. That is fast enough well past 10^5 templates per
    tenant on CPU; beyond that, swap in an ANN backend behind this same
    interface and accept the recall/latency trade-off explicitly.

    Every method is guarded by a lock because the API server is multi-threaded
    and numpy array replacement is not atomic.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._shards: dict[str, _TenantShard] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------- mutation ---

    def add(
        self,
        tenant_id: str,
        template_id: str,
        subject_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vector = self._prepare(embedding)
        with self._lock:
            self._shard(tenant_id).add(template_id, subject_id, vector, metadata or {})

    def add_many(
        self,
        tenant_id: str,
        rows: list[tuple[str, str, np.ndarray, dict[str, Any]]],
    ) -> None:
        prepared = [(t_id, s_id, self._prepare(vec), meta) for t_id, s_id, vec, meta in rows]
        with self._lock:
            self._shard(tenant_id).add_many(prepared)

    def remove(self, tenant_id: str, template_id: str) -> bool:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return shard.remove(template_id) if shard else False

    def remove_subject(self, tenant_id: str, subject_id: str) -> int:
        """Delete every template belonging to one subject. Returns the count."""
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None:
                return 0
            doomed = [
                template_id
                for template_id, owner in zip(shard.template_ids, shard.subject_ids)
                if owner == subject_id
            ]
            for template_id in doomed:
                shard.remove(template_id)
            return len(doomed)

    def clear(self, tenant_id: str | None = None) -> None:
        with self._lock:
            if tenant_id is None:
                self._shards.clear()
            else:
                self._shards.pop(tenant_id, None)

    # -------------------------------------------------------------- query ---

    def size(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return len(shard) if shard else 0

    def subject_count(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return len(set(shard.subject_ids)) if shard else 0

    def tenants(self) -> list[str]:
        with self._lock:
            return sorted(self._shards)

    def search(
        self,
        tenant_id: str,
        embedding: np.ndarray,
        top_k: int = 20,
        min_score: float = -1.0,
        collapse_subjects: bool = True,
    ) -> SearchOutcome:
        """Rank the tenant's gallery against a probe.

        With ``collapse_subjects`` (the default) a subject enrolled from several
        photographs appears once, at its best-scoring template. Otherwise a
        well-enrolled subject would fill the entire candidate list and hide every
        other lead from the examiner.
        """
        query = self._prepare(embedding)
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None or len(shard) == 0:
                return SearchOutcome(matches=(), all_scores=np.empty(0, dtype=np.float32), gallery_size=0)

            scores = shard.scores(query)
            template_ids = list(shard.template_ids)
            subject_ids = list(shard.subject_ids)
            metadata = list(shard.metadata)
            gallery_size = len(shard)

        order = np.argsort(scores)[::-1]
        matches: list[MatchResult] = []
        seen_subjects: set[str] = set()

        for index in order:
            score = float(scores[index])
            if score < min_score:
                break
            subject_id = subject_ids[index]
            if collapse_subjects:
                if subject_id in seen_subjects:
                    continue
                seen_subjects.add(subject_id)
            matches.append(
                MatchResult(
                    template_id=template_ids[index],
                    subject_id=subject_id,
                    score=round(score, 6),
                    metadata=metadata[index],
                )
            )
            if len(matches) >= top_k:
                break

        return SearchOutcome(matches=tuple(matches), all_scores=scores, gallery_size=gallery_size)

    # ----------------------------------------------------------- internal ---

    def _shard(self, tenant_id: str) -> _TenantShard:
        shard = self._shards.get(tenant_id)
        if shard is None:
            shard = _TenantShard(self.dimensions)
            self._shards[tenant_id] = shard
        return shard

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimensions:
            raise ValueError(f"Expected a {self.dimensions}-d template, got {vector.shape[0]}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Template contains NaN or infinite values.")
        return l2_normalize(vector)


__all__ = ["GalleryIndex", "MatchResult", "SearchOutcome"]

```

## `backend/nexgen_engine/models/cuda_runtime.py`

```python
"""
CUDA runtime discovery and onnxruntime execution-provider assertions.

WHY THIS MODULE EXISTS
----------------------
onnxruntime-gpu fails *silently*. When it cannot load
`onnxruntime_providers_cuda.dll` (missing CUDA 12 / cuDNN 9 DLLs, or a
CUDA-13-built wheel on a CUDA-12 host) it logs a warning to stderr that is
easy to miss, drops `CUDAExecutionProvider` from the session, and runs the
model on CPU at roughly 20x the latency. Nothing raises. Nothing crashes.

Worse, `onnxruntime.get_available_providers()` still reports
`CUDAExecutionProvider` in that state -- it lists providers the wheel was
*built* with, not providers that actually loaded. Any GPU check based on that
call is a false positive. The only trustworthy signal is
`InferenceSession.get_providers()` on a session that has already been created.

This project has lost the GPU configuration twice to exactly that failure
mode, so the rule here is: resolve the DLLs explicitly, then assert on the
*post-construction* session providers and raise if CUDA was expected but CPU
was bound.

ENVIRONMENT FLAGS
-----------------
NEXGEN_FORCE_CPU=1    Skip CUDA entirely and expect CPU. No assertion failure.
NEXGEN_REQUIRE_GPU=1  Hard-require CUDA even if no GPU is detected (CI guard).
"""

from __future__ import annotations

import logging
import os
import site
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class GpuBindingError(RuntimeError):
    """Raised when CUDA was expected but onnxruntime bound the CPU provider."""


def _force_cpu() -> bool:
    return os.environ.get("NEXGEN_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}


def _require_gpu() -> bool:
    return os.environ.get("NEXGEN_REQUIRE_GPU", "").strip().lower() in {"1", "true", "yes"}


def _candidate_dll_dirs() -> list[Path]:
    """Directories that may hold the CUDA 12 / cuDNN 9 runtime DLLs.

    onnxruntime-gpu does not vendor the CUDA runtime; it resolves
    cudart64_12.dll / cublasLt64_12.dll / cudnn64_9.dll through the normal
    Windows DLL search order. In this project those DLLs are supplied by the
    PyTorch cu121 wheel (torch/lib), and optionally by the standalone
    nvidia-*-cu12 wheels (nvidia/<component>/bin).
    """
    dirs: list[Path] = []

    try:
        import torch  # noqa: PLC0415 - intentionally lazy

        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:  # pragma: no cover - torch is optional for CPU-only hosts
        pass

    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = getattr(site, "getusersitepackages", None)
    if callable(user_site):
        try:
            roots.append(user_site())
        except Exception:
            pass

    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for component in sorted(nvidia_root.iterdir()):
            bin_dir = component / "bin"
            if bin_dir.is_dir():
                dirs.append(bin_dir)

    return dirs


@lru_cache(maxsize=1)
def prepare_cuda_dlls() -> list[str]:
    """Expose the vendored CUDA DLLs to onnxruntime's loader. Idempotent.

    Both mechanisms are needed on Windows: `os.add_dll_directory` covers
    LoadLibraryEx calls that use LOAD_LIBRARY_SEARCH_* flags, while prepending
    PATH covers the legacy search order that resolves the *transitive*
    dependencies of onnxruntime_providers_cuda.dll.
    """
    if not sys.platform.startswith("win"):
        return []

    added: list[str] = []
    for directory in _candidate_dll_dirs():
        resolved = str(directory)
        try:
            os.add_dll_directory(resolved)
        except (OSError, AttributeError):
            pass
        if resolved not in os.environ.get("PATH", ""):
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
        added.append(resolved)

    if added:
        logger.debug("CUDA DLL search paths registered: %s", added)
    return added


@lru_cache(maxsize=1)
def gpu_hardware_present() -> bool:
    """True when a usable CUDA device exists, independent of onnxruntime."""
    prepare_cuda_dlls()
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def init_cuda() -> bool:
    """Initialize the CUDA context before any onnxruntime session is built.

    Creating the torch CUDA context first forces the CUDA driver and the
    cu121 runtime DLLs to be loaded into the process. onnxruntime then finds
    an already-initialized runtime instead of trying (and on a cold process,
    sometimes failing) to bootstrap it from its own provider DLL.
    """
    if _force_cpu():
        logger.info("NEXGEN_FORCE_CPU set - skipping CUDA initialization")
        return False

    prepare_cuda_dlls()
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        torch.cuda.init()
        logger.info("CUDA initialized: %s", torch.cuda.get_device_name(0))
        return True
    except Exception as exc:
        logger.warning("CUDA initialization failed, continuing on CPU: %s", exc)
        return False


def detect_duplicate_onnxruntime() -> list[str]:
    """Return installed onnxruntime distributions.

    `onnxruntime` and `onnxruntime-gpu` install into the same import
    namespace. With both present the import resolves to whichever wrote the
    files last, which is a silent, machine-dependent CPU fallback.
    """
    from importlib.metadata import distributions  # noqa: PLC0415

    found = set()
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"}:
            found.add(name)
    return sorted(found)


def resolve_providers() -> tuple[list[str], int]:
    """Return (provider_list, ctx_id) for InsightFace `prepare()`.

    ctx_id is InsightFace's device selector: 0 = first GPU, -1 = CPU.
    """
    if _force_cpu():
        return [CPU_PROVIDER], -1

    init_cuda()

    import onnxruntime as ort  # noqa: PLC0415

    # NOTE: this is a build-time capability list, NOT proof CUDA will bind.
    # The real check happens in assert_session_provider() after load.
    if CUDA_PROVIDER in ort.get_available_providers():
        return [CUDA_PROVIDER, CPU_PROVIDER], 0
    return [CPU_PROVIDER], -1


def cuda_expected() -> bool:
    """Whether this host is supposed to be running on GPU."""
    if _force_cpu():
        return False
    if _require_gpu():
        return True
    return gpu_hardware_present()


def session_provider(model_obj: object) -> str | None:
    """Best-effort read of the execution provider an InsightFace model bound.

    InsightFace model wrappers expose the underlying
    onnxruntime.InferenceSession as `.session`.
    """
    session = getattr(model_obj, "session", None)
    if session is None:
        return None
    try:
        providers = session.get_providers()
    except Exception:
        return None
    return providers[0] if providers else None


def assert_face_analysis_providers(app: object, label: str) -> dict[str, str]:
    """Assert every sub-model of a FaceAnalysis app bound the expected provider.

    Returns {model_name: provider}. Raises GpuBindingError when CUDA was
    expected but any sub-model fell back to CPU.
    """
    bound: dict[str, str] = {}
    for name, model_obj in getattr(app, "models", {}).items():
        provider = session_provider(model_obj)
        if provider:
            bound[name] = provider

    if not bound:
        logger.warning("[%s] could not introspect any onnxruntime session", label)
        return bound

    if not cuda_expected():
        logger.info("[%s] CPU mode (expected). providers=%s", label, bound)
        return bound

    on_cpu = sorted(n for n, p in bound.items() if p != CUDA_PROVIDER)
    if on_cpu:
        raise GpuBindingError(
            f"[{label}] CUDA was expected but these sub-models bound "
            f"{CPU_PROVIDER}: {on_cpu}. Full binding map: {bound}.\n"
            "This is the silent-CPU-fallback failure mode. Diagnose with:\n"
            "  python scripts/verify_gpu.py\n"
            "Most likely causes:\n"
            "  1. torch is a +cpu build, so no CUDA 12 DLLs exist in the venv.\n"
            "     Fix: pip install -r backend/requirements-gpu.txt\n"
            "  2. plain `onnxruntime` is installed alongside `onnxruntime-gpu` "
            "and is shadowing it.\n"
            "     Fix: pip uninstall -y onnxruntime\n"
            "  3. onnxruntime-gpu >= 1.22 (CUDA 13 build) on this CUDA 12 host.\n"
            "     Fix: pin onnxruntime-gpu==1.20.1 per backend/requirements-gpu.txt\n"
            "Set NEXGEN_FORCE_CPU=1 to intentionally run on CPU."
        )

    logger.info("[%s] all sub-models on %s", label, CUDA_PROVIDER)
    return bound

```

## `backend/imatch_api/main.py`

```python
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from .core.csrf import CSRF_COOKIE, CSRF_HEADER, request_is_exempt, tokens_match, validate_csrf_token
from fastapi.responses import JSONResponse

from .api.routes import account, admin, audit, auth, cases, health, reports, search, subjects
from .core.config import get_settings
from .db.session import init_database
from .services.engine_service import get_engine_service

logger = logging.getLogger(__name__)

DESCRIPTION = """
NexGen iMATCH -- facial recognition for forensic investigation.

**What this system does:** given a probe image, it ranks visually similar faces
from a gallery your organisation enrolled, and records who searched for what, when,
and on what stated authority.

**What it does not do:** it does not identify people. A similarity score is not a
probability that two images show the same person. Every result is an investigative
lead requiring examiner verification before it is relied upon.

The service will not start without real recognition weights. There is no fallback
mode: a substitute embedding would produce numbers that look like similarity
scores and mean nothing, which is worse in an investigation than an outage.
Inspect `GET /api/imatch/engine/status` for the loaded model, device, and
thresholds actually in effect.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    init_database()
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.audit_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine_service()
    # Load models during startup rather than inside the first user request, which
    # would otherwise hang for the seconds an ONNX pack takes to load. A failure
    # here aborts startup by design: the service must not accept biometric
    # searches it cannot actually perform.
    engine.warm_up()

    info = engine.runtime.recognizer.info
    logger.info(
        "iMATCH ready: %s (%s), %s-d templates on %s via %s.",
        info.model_pack,
        info.recognition_network,
        info.embedding_dim,
        engine.runtime.device,
        ", ".join(info.providers),
    )

    yield

    logger.info("iMATCH shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="NexGen iMATCH",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        """Attach a correlation id and baseline security headers."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # API responses containing biometric findings must not sit in a shared
        # or browser cache.
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
            "interest-cohort=()"
        )
        # Isolate the browsing context so a cross-origin opener cannot reach
        # this window, and so no other site can embed a response as a resource.
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-DNS-Prefetch-Control"] = "off"

        # Content-Security-Policy.
        #
        # This service returns JSON, not markup, so the strictest possible
        # policy is also the correct one: nothing should ever be loaded or
        # executed from an API response. `frame-ancestors 'none'` is the
        # modern form of X-Frame-Options and is kept alongside it because
        # older browsers honour only the header.
        #
        # /docs is the exception and gets a relaxed policy rather than none:
        # Swagger UI is served from a CDN and uses inline styles, so the strict
        # policy would leave an interactive page that silently renders blank.
        # It is unavailable in production anyway (docs_url is None there).
        if request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data: https:; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                "form-action 'none'; sandbox"
            )

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):  # noqa: ANN001, ANN202
        """Reject state-changing requests that rely on cookies without a token.

        Requests presenting Authorization or X-API-Key are exempt: a
        cross-origin page cannot set those headers, so they are not forgeable
        and demanding a token would break every API client to no benefit. What
        remains is the unauthenticated auth endpoints, where this prevents
        login CSRF -- an attacker forcing a victim into an account they control
        so the victim's later searches are audited against the wrong person.
        """
        if settings.csrf_enabled:
            has_auth_header = bool(
                request.headers.get("authorization") or request.headers.get("x-api-key")
            )
            if not request_is_exempt(request.method, has_auth_header):
                cookie = request.cookies.get(CSRF_COOKIE)
                header = request.headers.get(CSRF_HEADER)
                secret = settings.resolved_jwt_secret()
                if not tokens_match(cookie, header) or not validate_csrf_token(header, secret):
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "Missing or invalid CSRF token.",
                            "hint": f"GET /api/auth/csrf, then send {CSRF_HEADER}.",
                        },
                    )
        return await call_next(request)

    # Registered LAST so it is the OUTERMOST middleware.
    #
    # Starlette runs the most recently added middleware first, and that
    # ordering is load-bearing here: with CORS on the inside, a 403 from the
    # CSRF guard short-circuits before CORS can attach its headers, and the
    # browser reports a legitimate rejection as an opaque "Failed to fetch"
    # with no way for the client to see why. Outermost, every response —
    # including refusals raised by middleware — carries the CORS headers.
    app.add_middleware(
        CORSMiddleware,
        # Never "*": these endpoints carry biometric data behind credentialed
        # requests, and a wildcard origin with credentials is both invalid and
        # dangerous.
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-CSRF-Token"],
        max_age=600,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return none of it.

        Stack traces and driver errors routinely leak schema, file paths, and
        occasionally credentials. The request id is the bridge between what the
        caller sees and what the operator can look up.
        """
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(account.router)
    app.include_router(cases.router)
    app.include_router(reports.router)
    app.include_router(subjects.router)
    app.include_router(search.router)
    app.include_router(audit.router)
    app.include_router(admin.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "NexGen iMATCH",
            "version": "1.0.0",
            "documentation": "/docs" if not settings.is_production else "disabled in production",
            "health": "/api/health",
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]

```

## `backend/imatch_api/core/dependencies.py`

```python
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request, status
from sqlmodel import Session, select

from ..db.models import ApiKey, Role, Tenant, User, utcnow
from ..db.session import get_session
from .config import Settings, get_settings
from .rate_limit import SlidingWindowRateLimiter, reset_auth_limiters
from .security import TokenError, api_key_prefix, decode_token, verify_api_key

logger = logging.getLogger(__name__)

_ROLE_RANK = {Role.INVESTIGATOR: 1, Role.SUPERVISOR: 2, Role.ADMIN: 3}

_general_limiter: SlidingWindowRateLimiter | None = None
_search_limiter: SlidingWindowRateLimiter | None = None


@dataclass(frozen=True)
class Principal:
    """The authenticated caller.

    ``tenant_id`` comes from the verified credential, never from a request body
    or header the client controls. Every repository query is scoped by it, so a
    caller cannot reach another tenant's biometric data even by guessing ids.
    """

    id: str
    tenant_id: str
    role: Role
    label: str
    credential: str  # "session" or "api_key"

    def has_role(self, minimum: Role) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[minimum]

    @property
    def rate_limit_key(self) -> str:
        return f"{self.credential}:{self.id}"


def get_general_limiter(settings: Settings = Depends(get_settings)) -> SlidingWindowRateLimiter:
    global _general_limiter
    if _general_limiter is None:
        _general_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)
    return _general_limiter


def get_search_limiter(settings: Settings = Depends(get_settings)) -> SlidingWindowRateLimiter:
    global _search_limiter
    if _search_limiter is None:
        _search_limiter = SlidingWindowRateLimiter(settings.search_rate_limit_per_minute)
    return _search_limiter


def reset_limiters() -> None:
    """Used by tests, which would otherwise trip the limiter across cases."""
    global _general_limiter, _search_limiter
    _general_limiter = None
    _search_limiter = None
    # The unauthenticated auth limiters are process-global (there is no
    # principal to key on before sign-in), so a test session's repeated logins
    # would otherwise exhaust the login budget and fail unrelated cases.
    reset_auth_limiters()


def get_current_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """Authenticate via bearer token or API key.

    Failures return a single generic message. Distinguishing "no such user" from
    "wrong password" here would let an unauthenticated caller enumerate accounts.
    """
    principal = None
    if x_api_key:
        principal = _principal_from_api_key(x_api_key, session)
    elif authorization:
        principal = _principal_from_bearer(authorization, session, settings)

    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.principal = principal
    return principal


def _principal_from_bearer(authorization: str, session: Session, settings: Settings) -> Principal | None:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None

    try:
        payload = decode_token(token, settings.resolved_jwt_secret(), settings.jwt_algorithm, "access")
    except TokenError:
        return None

    user = session.get(User, payload.subject_id)
    if user is None or not user.active:
        return None
    # The token carries a tenant claim, but the database is authoritative: a user
    # moved between tenants must not keep acting under the old one until expiry.
    if user.tenant_id != payload.tenant_id:
        logger.warning("Token tenant claim does not match stored user %s; rejecting.", user.id)
        return None
    if not _tenant_active(session, user.tenant_id):
        return None

    return Principal(
        id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
        label=user.email,
        credential="session",
    )


def _principal_from_api_key(raw_key: str, session: Session) -> Principal | None:
    record = session.exec(select(ApiKey).where(ApiKey.prefix == api_key_prefix(raw_key))).first()
    if record is None or not record.active:
        return None
    if not verify_api_key(raw_key, record.key_hash):
        return None
    if record.expires_at is not None and _as_utc(record.expires_at) < datetime.now(timezone.utc):
        return None
    if not _tenant_active(session, record.tenant_id):
        return None

    record.last_used_at = utcnow()
    session.add(record)

    return Principal(
        id=record.id,
        tenant_id=record.tenant_id,
        role=record.role,
        label=f"api-key:{record.name}",
        credential="api_key",
    )


def _tenant_active(session: Session, tenant_id: str) -> bool:
    tenant = session.get(Tenant, tenant_id)
    return tenant is not None and tenant.active


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; compare them as UTC, not local time."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def require_role(minimum: Role) -> Callable[[Principal], Principal]:
    def dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.has_role(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the {minimum.value} role or higher.",
            )
        return principal

    return dependency


require_investigator = require_role(Role.INVESTIGATOR)
require_supervisor = require_role(Role.SUPERVISOR)
require_admin = require_role(Role.ADMIN)


def enforce_rate_limit(
    principal: Principal = Depends(get_current_principal),
    limiter: SlidingWindowRateLimiter = Depends(get_general_limiter),
) -> Principal:
    result = limiter.check(principal.rate_limit_key)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(int(result.retry_after) + 1)},
        )
    return principal


def enforce_search_rate_limit(
    principal: Principal = Depends(get_current_principal),
    limiter: SlidingWindowRateLimiter = Depends(get_search_limiter),
) -> Principal:
    """Tighter limit on biometric search specifically.

    Search is the expensive and the privacy-sensitive path: a bulk scrape of a
    gallery looks exactly like a burst of ordinary searches, so it gets its own
    lower ceiling rather than sharing the general budget.
    """
    result = limiter.check(principal.rate_limit_key)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Biometric search rate limit exceeded.",
            headers={"Retry-After": str(int(result.retry_after) + 1)},
        )
    return principal


def client_context(request: Request) -> tuple[str, str]:
    """Best-effort caller identification for the audit record.

    X-Forwarded-For is client-controllable unless a trusted proxy overwrites it,
    so it is recorded as a claim, not as fact.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")
    return ip, request.headers.get("user-agent", "")[:512]


__all__ = [
    "Principal",
    "client_context",
    "enforce_rate_limit",
    "enforce_search_rate_limit",
    "get_current_principal",
    "get_general_limiter",
    "get_search_limiter",
    "require_admin",
    "require_investigator",
    "require_role",
    "require_supervisor",
    "reset_limiters",
]

```

