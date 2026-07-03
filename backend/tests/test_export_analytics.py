from __future__ import annotations

from nexgen_engine.analytics import AccuracyTracker, AnalyticsReportGenerator, UsageMetrics
from nexgen_engine.export import export_onnx_manifest, export_trt_manifest, package_for_client


def test_export_and_analytics_outputs(tmp_path):
    onnx = export_onnx_manifest(tmp_path / "onnx")
    trt = export_trt_manifest(tmp_path / "trt")
    package = package_for_client("client-a", tmp_path / "packages")
    assert onnx.embedding_dim == 512
    assert trt.precision == "fp16"
    assert package.validation_required

    tracker = AccuracyTracker()
    tracker.add(0.91, True)
    tracker.add(0.42, False)
    assert tracker.summary()["samples"] == 2.0
    report = AnalyticsReportGenerator().write(UsageMetrics(enrollments=1, identifications=2, review_required=1), tmp_path / "analytics.json")
    assert report.exists()
