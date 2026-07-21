from __future__ import annotations

import base64
import json
import py_compile
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from nexgen_engine.analytics import AccuracyTracker, AnalyticsReportGenerator, UsageMetrics
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
from nexgen_engine.data.manifest import DatasetManifest
from nexgen_engine.runtime import detect_runtime_capabilities
from nexgen_engine.search import OptionalFaissIndex
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
    return {
        "health": client.get("/api/v1/health").json()["status"],
        "enroll_status": enroll.status_code,
        "identify_status": identify.status_code,
        "identify_matches": len(identify.json().get("matches", [])),
        "imatch_status": imatch.status_code,
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
    return {
        "manifest_template": manifest_path.exists(),
        "records": report.total_records,
        "accepted": report.accepted_records,
        "rejected": report.rejected_records,
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


def main() -> int:
    image, payload = make_image()
    result = {
        "compiled_files": compile_backend(),
        "engine": run_engine_smoke(),
        "api": run_api_smoke(payload),
        "artifacts": run_artifact_smoke(),
        "dataset": run_dataset_smoke(payload),
        "optional_backends": run_optional_backend_smoke(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        result["api"]["health"] != "ok",
        result["api"]["enroll_status"] != 200,
        result["api"]["identify_status"] != 200,
        result["api"]["imatch_status"] != 200,
        result["engine"]["embedding_dim"] != 512,
        result["engine"]["engine_backbones"] != 8,
        not result["engine"]["audit_valid"],
    ]
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
