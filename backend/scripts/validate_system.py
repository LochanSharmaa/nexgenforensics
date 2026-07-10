from __future__ import annotations

import base64
import json
import py_compile
import sys
import uuid
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from nexgen_engine.analytics import AccuracyTracker, AnalyticsReportGenerator, UsageMetrics
from nexgen_engine.auth import AuthService, Principal, require_role
from nexgen_engine.api.routes.admin import engine_status
from nexgen_engine.api.routes.verify import verify_embeddings
from nexgen_engine.api.service import EngineService
from nexgen_engine.benchmarks import evaluate_verification_scores
from nexgen_engine.benchmarks.report_generator import BenchmarkReportGenerator
from nexgen_engine.export import export_onnx_manifest, export_trt_manifest, package_for_client
from nexgen_engine.inference import EmbeddingExtractor, ScoreFusion, TTAProcessor
from nexgen_engine.models import (
    AgeInvariantBackbone,
    BEiTLBackbone,
    EfficientNetXLBackbone,
    IRFaceNetBackbone,
    PoseNetBackbone,
    ResNet100IRBackbone,
    SwinLBackbone,
    ViTH14Backbone,
)
from nexgen_engine.data.ingestion_validator import DatasetIngestionValidator
from nexgen_engine.data.image_archive import ImageArchiveCataloger
from nexgen_engine.data.manifest import DatasetManifest
from nexgen_engine.data.recordio import RecordIOArchiveCataloger
from nexgen_engine.dependencies import DependencyVerifier
from nexgen_engine.detection import DetectorConfig, DetectorRegistry, FaceAligner
from nexgen_engine.detection.smoke import detector_smoke
from nexgen_engine.quality import FaceQNetScorer, ProductionQualityScorer
from nexgen_engine.quality.cli import quality_check
from nexgen_engine.runtime import detect_runtime_capabilities
from nexgen_engine.search import OptionalFaissIndex
from nexgen_engine.jobs import JobQueue, JobWorker
from nexgen_engine.monitoring import MetricsRegistry
from nexgen_engine.settings import Settings
from nexgen_engine.storage import Database, UserRecord
from nexgen_engine.security.presentation_attack import PresentationAttackDetector


ROOT = Path(__file__).resolve().parents[1]


def make_image(color: tuple[int, int, int] = (120, 130, 150)) -> tuple[Image.Image, bytes]:
    image = Image.new("RGB", (180, 180), color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return image, buffer.getvalue()


def compile_backend() -> int:
    count = 0
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        py_compile.compile(str(path), doraise=True)
        count += 1
    return count


def run_engine_smoke() -> dict[str, object]:
    image, payload = make_image()
    wrappers = [
        ViTH14Backbone(),
        SwinLBackbone(),
        BEiTLBackbone(),
        ResNet100IRBackbone(),
        EfficientNetXLBackbone(),
        AgeInvariantBackbone(),
        PoseNetBackbone(),
        IRFaceNetBackbone(),
    ]
    wrapper_dims = [wrapper.encode(image).embedding.shape[0] for wrapper in wrappers]
    embedding = EmbeddingExtractor().from_image(image)
    service = EngineService(audit_path=ROOT.parent / "runtime" / "validation_audit.jsonl")
    service.enroll(payload, "validation-subject", {"workspace": "validation"})
    identification = service.identify(payload, operator_id="validator")

    return {
        "runtime_mode": detect_runtime_capabilities().mode,
        "wrapper_dims": wrapper_dims,
        "tta_count": len(TTAProcessor().apply(image)),
        "embedding_dim": int(embedding.shape[0]),
        "self_verify": verify_embeddings(embedding, embedding)["verified"],
        "engine_backbones": engine_status()["backbone_count"],
        "score_fusion": ScoreFusion().fuse(0.8, 0.9, 0.95),
        "presentation_attack_keys": sorted(PresentationAttackDetector().assess(image, 0.8).keys()),
        "top_match": identification.matches[0].identity_id if identification.matches else None,
        "audit_valid": service.audit.verify(),
    }


def run_api_smoke(payload: bytes) -> dict[str, object]:
    client = TestClient(app)
    image_b64 = base64.b64encode(payload).decode("ascii")
    suffix = uuid.uuid4().hex[:8]
    enroll = client.post(
        "/api/v1/engine/enroll",
        json={"identity_id": "api-validation-subject", "workspace": "api", "image_base64": image_b64},
    )
    identify = client.post(
        "/api/v1/engine/identify",
        json={"operator_id": "api-validator", "image_base64": image_b64},
    )
    imatch = client.post(
        "/api/imatch/search",
        json={
            "mode": "single",
            "purpose": "authorized_imatch_demo",
            "lawful_use_reason": "local validation",
            "checks": ["Liveness Check", "Quality Assessment"],
            "image_base64": image_b64,
        },
    )
    register = client.post(
        "/api/v1/auth/register",
        json={
            "tenant_id": f"api-tenant-{suffix}",
            "tenant_name": "API Tenant",
            "email": f"admin-{suffix}@example.com",
            "password": "secret123",
            "role": "tenant_admin",
        },
    )
    token = register.json().get("access_token") if register.status_code == 200 else ""
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    job = client.post("/api/v1/jobs", json={"job_type": "dataset_ingestion", "payload": {"path": "demo"}}, headers={"Authorization": f"Bearer {token}"})
    return {
        "health": client.get("/api/v1/health").json()["status"],
        "enroll_status": enroll.status_code,
        "identify_status": identify.status_code,
        "identify_matches": len(identify.json().get("matches", [])),
        "imatch_status": imatch.status_code,
        "register_status": register.status_code,
        "me_status": me.status_code,
        "job_status": job.status_code,
    }


def run_artifact_smoke() -> dict[str, object]:
    output_root = ROOT.parent / "runtime" / "validation"
    metrics = evaluate_verification_scores(
        scores=__import__("numpy").asarray([0.95, 0.88, 0.22, 0.12]),
        same_identity=__import__("numpy").asarray([True, True, False, False]),
    )
    benchmark_path = BenchmarkReportGenerator().write(metrics, output_root / "benchmark.json", "validation-demo")
    onnx = export_onnx_manifest(output_root / "onnx")
    trt = export_trt_manifest(output_root / "trt")
    package = package_for_client("validation-client", output_root / "packages")
    tracker = AccuracyTracker()
    tracker.add(0.95, True)
    tracker.add(0.20, False)
    analytics_path = AnalyticsReportGenerator().write(UsageMetrics(1, 2, 1, 12.5), output_root / "analytics.json")
    return {
        "benchmark_report": benchmark_path.exists(),
        "onnx_dim": onnx.embedding_dim,
        "trt_precision": trt.precision,
        "package_client": package.client_id,
        "accuracy_samples": tracker.summary()["samples"],
        "analytics_report": analytics_path.exists(),
    }


def run_dataset_smoke(payload: bytes) -> dict[str, object]:
    output_root = ROOT.parent / "runtime" / "dataset_smoke"
    image_dir = output_root / "person_001"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "img_001.jpg"
    image_path.write_bytes(payload)
    manifest_path = output_root / "manifest.csv"
    DatasetManifest.write_template(manifest_path)
    report = DatasetIngestionValidator().validate(output_root, manifest_path)
    recordio_zip = output_root / "sample_recordio.zip"
    with ZipFile(recordio_zip, "w") as archive:
        archive.writestr("sample/train.rec", b"placeholder")
        archive.writestr("sample/train.idx", b"placeholder")
        archive.writestr("sample/train.lst", "0\tperson_001\tperson_001/img_001.jpg\n1\tperson_001\tperson_001/img_002.jpg\n")
        archive.writestr("sample/lfw.bin", b"placeholder")
    image_zip = output_root / "sample_images.zip"
    with ZipFile(image_zip, "w") as archive:
        archive.writestr("WIDER_val/images/0--Parade/sample.jpg", payload)
    recordio_report = RecordIOArchiveCataloger().catalog_zip(
        recordio_zip,
        output_root / "recordio_catalog",
        workspace="validation",
        lawful_basis="validation_fixture",
        consent=True,
    )
    image_archive_report = ImageArchiveCataloger().catalog_zip(
        image_zip,
        output_root / "image_archive_catalog",
        workspace="validation",
        lawful_basis="validation_fixture",
        consent=True,
    )
    return {
        "manifest_template": manifest_path.exists(),
        "records": report.total_records,
        "accepted": report.accepted_records,
        "rejected": report.rejected_records,
        "recordio_ready": recordio_report.ready_for_catalog,
        "recordio_records": recordio_report.listed_records,
        "recordio_identities": recordio_report.listed_identities,
        "image_archive_ready": image_archive_report.ready_for_catalog,
        "image_archive_records": image_archive_report.image_records,
    }


def run_detector_quality_smoke() -> dict[str, object]:
    image, payload = make_image()
    output_root = ROOT.parent / "runtime" / "detector_quality_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / "face.jpg"
    image_path.write_bytes(payload)
    detector = DetectorRegistry().create(DetectorConfig(backend="center_crop", allow_fallback=True))
    boxes = detector.detect(image)
    aligned = FaceAligner(detector=detector).align(image)
    quality = ProductionQualityScorer().evaluate(image, detector_confidence=boxes[0].confidence, landmark_confidence=1.0)
    missing_faceqnet_error = False
    try:
        FaceQNetScorer("missing-faceqnet.pt", required=True)
    except FileNotFoundError:
        missing_faceqnet_error = True
    production_fallback_blocked = False
    try:
        DetectorRegistry().create(DetectorConfig(backend="center_crop", allow_fallback=False))
    except RuntimeError:
        production_fallback_blocked = True
    return {
        "detector_backend": boxes[0].backend,
        "detector_smoke_file": detector_smoke(image_path, "center_crop", True, output_root / "detector.json").exists(),
        "quality_cli_has_brisque": quality_check(image_path).get("brisque_score") is not None,
        "aligned_size": aligned.size,
        "quality_accepts": quality.accepted,
        "brisque_present": quality.brisque_score is not None,
        "faceqnet_present": quality.faceqnet_score is not None,
        "missing_faceqnet_error": missing_faceqnet_error,
        "production_fallback_blocked": production_fallback_blocked,
        "dependency_count": len(DependencyVerifier().statuses()),
    }


def run_optional_backend_smoke() -> dict[str, object]:
    import numpy as np

    index = OptionalFaissIndex(512)
    vector = np.ones(512, dtype=np.float32)
    index.add("optional-faiss-subject", vector, {"workspace": "validation"})
    result = index.search(vector, 1)
    return {
        "faiss_mode": "production" if index.production_loaded else "fallback",
        "top_id": result[0].identity_id if result else None,
    }


def run_production_layer_smoke() -> dict[str, object]:
    runtime = ROOT.parent / "runtime" / "production_smoke"
    db = Database(runtime / "nexgen.db")
    db.migrate()
    suffix = uuid.uuid4().hex[:8]
    tenant_id = f"tenant-{suffix}"
    email = f"user-{suffix}@example.com"
    db.upsert_tenant(tenant_id, "Tenant A")
    auth = AuthService("validation-secret")
    password_hash = auth.hash_password("secret", "salt")
    db.create_user(UserRecord(f"user-{suffix}", tenant_id, email, "tenant_admin", password_hash))
    user = db.get_user_by_email(email)
    assert user is not None
    token = auth.issue_token(Principal(user.user_id, user.tenant_id, user.role, user.email))
    principal = auth.verify_token(token)
    require_role(principal.role, "operator")

    queue = JobQueue(db)
    job_id = queue.enqueue(tenant_id, "noop", {"ok": True})
    worker = JobWorker(db, {"noop": lambda payload: None})
    worker.run_job(job_id)
    metrics = MetricsRegistry()
    metrics.increment("validation")
    with metrics.timer("validation_latency"):
        pass
    settings = Settings.from_env()
    return {
        "user_loaded": user.email,
        "principal_role": principal.role,
        "job_status": queue.status(job_id)["status"],
        "metrics_has_counter": "nexgen_validation_total" in metrics.prometheus_text(),
        "settings_env": settings.env,
    }


def main() -> int:
    image, payload = make_image()
    result = {
        "compiled_files": compile_backend(),
        "engine": run_engine_smoke(),
        "api": run_api_smoke(payload),
        "artifacts": run_artifact_smoke(),
        "dataset": run_dataset_smoke(payload),
        "detector_quality": run_detector_quality_smoke(),
        "optional_backends": run_optional_backend_smoke(),
        "production_layers": run_production_layer_smoke(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        result["api"]["health"] != "ok",
        result["api"]["enroll_status"] != 200,
        result["api"]["identify_status"] != 200,
        result["api"]["imatch_status"] != 200,
        result["api"]["register_status"] != 200,
        result["api"]["me_status"] != 200,
        result["api"]["job_status"] != 200,
        result["engine"]["embedding_dim"] != 512,
        result["engine"]["engine_backbones"] != 8,
        not result["engine"]["audit_valid"],
        result["production_layers"]["job_status"] != "succeeded",
        not result["detector_quality"]["missing_faceqnet_error"],
        not result["detector_quality"]["production_fallback_blocked"],
    ]
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
