from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Iterable

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:  # pragma: no cover - lets syntax checks pass before dependencies install.
    Image = None
    ImageFilter = None
    ImageStat = None

from .schemas import ImatchResponse, LivenessResult, MatchCandidate, QualityResult


EMBEDDING_DIMENSIONS = 128


@dataclass(frozen=True)
class DemoIdentity:
    identity_id: str
    workspace: str
    label: str
    seed: str


DEMO_GALLERY = [
    DemoIdentity("ENT-KYC-1042", "Fintech KYC", "Verified customer profile", "fintech-kyc-primary"),
    DemoIdentity("ENT-ACC-2197", "Corporate Access", "Employee access profile", "corporate-access-primary"),
    DemoIdentity("ENT-FRD-3308", "Fraud Operations", "Manual review profile", "fraud-review-primary"),
    DemoIdentity("ENT-WID-4419", "Workforce Identity", "Workforce identity record", "workforce-identity-primary"),
    DemoIdentity("ENT-RSK-5520", "Retail Risk", "Loss-prevention review record", "retail-risk-primary"),
]


def analyze_imatch_request(
    *,
    image_bytes: bytes | None,
    source_url: str | None,
    mode: str,
    checks: Iterable[str],
    purpose: str | None,
    lawful_use_reason: str | None,
) -> ImatchResponse:
    reasons: list[str] = []
    audit_id = f"audit_{uuid.uuid4().hex[:12]}"
    normalized_checks = {check.lower().strip() for check in checks}

    if not purpose or not lawful_use_reason:
        reasons.append("purpose_and_lawful_use_required")

    if not image_bytes and not source_url:
        reasons.append("image_or_source_url_required")

    if image_bytes:
        quality = score_image_quality(image_bytes)
        request_fingerprint = hashlib.sha256(image_bytes).hexdigest()
    else:
        quality = score_url_quality(source_url or "")
        request_fingerprint = hashlib.sha256((source_url or "").encode("utf-8")).hexdigest()

    liveness = score_liveness(request_fingerprint, quality, normalized_checks)
    probe_embedding = embedding_from_seed(request_fingerprint)
    matches = rank_demo_matches(probe_embedding)
    top_score = matches[0].score if matches else 0.0

    if quality.score < 0.45:
        reasons.append("low_image_quality")
    if "liveness check" in normalized_checks and not liveness.passed:
        reasons.append("liveness_review_required")
    if top_score < 0.72:
        reasons.append("no_high_confidence_candidate")

    review_required = bool(reasons) or top_score < 0.86
    decision = "review_required" if review_required else "candidate_match_ready"

    return ImatchResponse(
        decision=decision,
        mode=mode,
        quality=quality,
        liveness=liveness,
        score=top_score,
        match={
            "score": top_score,
            "decision": decision,
            "benchmark_note": "Demo score only; production accuracy requires independent validation.",
        },
        matches=matches,
        review_required=review_required,
        reasons=reasons,
        audit={
            "audit_id": audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "lawful_use_reason": lawful_use_reason,
            "tenant_scope": "demo_workspace",
            "cross_client_search": "disabled",
            "checks": sorted(normalized_checks),
        },
    )


def score_image_quality(image_bytes: bytes) -> QualityResult:
    if Image is None or ImageStat is None or ImageFilter is None:
        return score_bytes_quality(image_bytes)

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            gray = image.convert("L")
            width, height = rgb.size
            resolution_score = clamp((min(width, height) - 96) / 416)

            stat = ImageStat.Stat(gray)
            brightness = stat.mean[0] / 255
            brightness_score = 1 - min(abs(brightness - 0.52) / 0.52, 1)
            contrast_score = clamp(stat.stddev[0] / 72)

            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            sharpness_score = clamp(edge_stat.mean[0] / 28)

            score = weighted_average(
                [
                    (resolution_score, 0.28),
                    (brightness_score, 0.22),
                    (contrast_score, 0.24),
                    (sharpness_score, 0.26),
                ]
            )

            reasons = []
            if resolution_score < 0.45:
                reasons.append("resolution_below_target")
            if brightness_score < 0.45:
                reasons.append("lighting_review")
            if contrast_score < 0.35:
                reasons.append("low_contrast")
            if sharpness_score < 0.35:
                reasons.append("blur_risk")

            return QualityResult(
                score=round(score, 4),
                resolution_score=round(resolution_score, 4),
                brightness_score=round(brightness_score, 4),
                contrast_score=round(contrast_score, 4),
                sharpness_score=round(sharpness_score, 4),
                reasons=reasons,
            )
    except Exception:
        return score_bytes_quality(image_bytes)


def score_bytes_quality(image_bytes: bytes) -> QualityResult:
    sample = image_bytes[:8192]
    size_score = clamp(len(image_bytes) / 250_000)
    entropy_score = byte_entropy_score(sample)
    score = weighted_average([(size_score, 0.45), (entropy_score, 0.55)])
    reasons = [] if score >= 0.45 else ["image_quality_review"]
    return QualityResult(
        score=round(score, 4),
        resolution_score=round(size_score, 4),
        brightness_score=round(entropy_score, 4),
        contrast_score=round(entropy_score, 4),
        sharpness_score=round(entropy_score, 4),
        reasons=reasons,
    )


def score_url_quality(source_url: str) -> QualityResult:
    source = source_url.strip().lower()
    secure_score = 1.0 if source.startswith("https://") else 0.35
    path_score = 0.85 if any(source.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")) else 0.55
    score = weighted_average([(secure_score, 0.55), (path_score, 0.45)])
    reasons = []
    if secure_score < 1:
        reasons.append("source_url_must_use_https_for_production")
    return QualityResult(
        score=round(score, 4),
        resolution_score=round(path_score, 4),
        brightness_score=round(score, 4),
        contrast_score=round(score, 4),
        sharpness_score=round(score, 4),
        reasons=reasons,
    )


def score_liveness(seed: str, quality: QualityResult, checks: set[str]) -> LivenessResult:
    if "liveness check" not in checks:
        return LivenessResult(score=0.0, passed=False, signals=["liveness_not_requested"])

    seed_value = int(seed[:8], 16) / 0xFFFFFFFF
    score = clamp((quality.score * 0.72) + (seed_value * 0.18) + 0.1)
    signals = ["texture_consistency", "capture_quality", "replay_risk_low" if score >= 0.72 else "manual_liveness_review"]
    return LivenessResult(score=round(score, 4), passed=score >= 0.72, signals=signals)


def rank_demo_matches(probe_embedding: list[float]) -> list[MatchCandidate]:
    candidates = []
    for identity in DEMO_GALLERY:
        gallery_embedding = embedding_from_seed(identity.seed)
        similarity = cosine_similarity(probe_embedding, gallery_embedding)
        calibrated_score = clamp(0.5 + (similarity * 0.5))
        candidates.append(
            MatchCandidate(
                id=identity.identity_id,
                identity_id=identity.identity_id,
                workspace=identity.workspace,
                score=round(calibrated_score, 4),
                metadata={
                    "label": identity.label,
                    "source": "demo_gallery",
                    "permission": "workspace_scoped",
                },
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:5]


def embedding_from_seed(seed: str) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < EMBEDDING_DIMENSIONS:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1 for byte in digest)
        counter += 1
    vector = values[:EMBEDDING_DIMENSIONS]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def byte_entropy_score(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    probabilities = [count / len(data) for count in counts if count]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    return clamp(entropy / 8)


def weighted_average(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values) or 1.0
    return clamp(sum(value * weight for value, weight in values) / total_weight)


def clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def parse_checks(raw_checks: str | list[str] | None) -> list[str]:
    if not raw_checks:
        return []
    if isinstance(raw_checks, list):
        return [str(item) for item in raw_checks]
    try:
        parsed = json.loads(raw_checks)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw_checks.split(",") if item.strip()]
