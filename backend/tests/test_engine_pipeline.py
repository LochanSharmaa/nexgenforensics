from __future__ import annotations

from io import BytesIO

from PIL import Image

from nexgen_engine.api.service import EngineService
from nexgen_engine.api.routes.admin import engine_status
from nexgen_engine.api.routes.verify import verify_embeddings
from nexgen_engine.benchmarks import evaluate_verification_scores
from nexgen_engine.inference import EmbeddingExtractor, ScoreFusion, TTAProcessor
from nexgen_engine.losses import CombinedMetricLoss
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
from nexgen_engine.security.presentation_attack import PresentationAttackDetector
from nexgen_engine.security.template_encryption import TemplateEncryptor
from nexgen_engine.training import NumpyMetricTrainer, TrainingBatch
import numpy as np


def make_image(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (180, 180), color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def test_enroll_identify_round_trip(tmp_path):
    service = EngineService(audit_path=tmp_path / "audit.jsonl")
    enrollment = service.enroll(make_image((120, 130, 150)), "person-001", {"workspace": "test"})
    assert enrollment.decision == "enrolled"
    result = service.identify(make_image((120, 130, 150)), operator_id="tester")
    assert result.matches
    assert result.matches[0].identity_id == "person-001"
    assert result.audit_hash
    assert service.audit.verify()


def test_template_encryption_round_trip():
    service = EngineService()
    result = service.pipeline.encode_bytes(make_image((90, 120, 160)))
    encryptor = TemplateEncryptor(iterations=10_000)
    encrypted = encryptor.encrypt(result.embedding, "local-test-secret")
    decrypted = encryptor.decrypt(encrypted, "local-test-secret")
    assert decrypted.shape == result.embedding.shape


def test_losses_training_and_benchmark_metrics():
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(8, 512)).astype(np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    class_weights = rng.normal(size=(4, 512)).astype(np.float32)
    loss = CombinedMetricLoss()(embeddings, labels, class_weights, stage=3)
    assert loss["total"] > -10

    trainer = NumpyMetricTrainer(class_count=4)
    report = trainer.evaluate_batch(TrainingBatch(embeddings, labels), epoch=35, global_step=1200)
    assert report.stage == 3
    assert report.learning_rate > 0

    metrics = evaluate_verification_scores(
        np.asarray([0.95, 0.87, 0.22, 0.11], dtype=np.float32),
        np.asarray([True, True, False, False]),
    )
    assert metrics.accuracy >= 0.5


def test_named_wrappers_and_auxiliary_modules():
    image = Image.new("RGB", (180, 180), (130, 120, 110))
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
    assert all(wrapper.encode(image).embedding.shape[0] == 2048 for wrapper in wrappers)
    assert len(TTAProcessor().apply(image)) == 6
    extractor = EmbeddingExtractor()
    embedding = extractor.from_image(image)
    assert embedding.shape[0] == 512
    assert ScoreFusion().fuse(0.8, 0.9, 0.95) > 0.7
    assert verify_embeddings(embedding, embedding)["verified"]
    assert engine_status()["backbone_count"] == 8
    assert "flagged" in PresentationAttackDetector().assess(image, quality_score=0.8)
