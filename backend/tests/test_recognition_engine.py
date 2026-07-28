"""Tests that prove the recognition engine actually recognizes people.

Everything else in the suite can pass while the system is incapable of its one
job. These run the real model on real photographs and assert on measured
behaviour: same-person pairs must score higher than different-person pairs, and
identification must find the right subject in a gallery.

They skip when the model or the face dataset is unavailable, and they never
assert a fixed accuracy figure -- the numbers reported here are properties of
this model on this dataset, not a claim about the product.
"""

from __future__ import annotations

import itertools
import random
import time

import numpy as np
import pytest

from nexgen_engine.inference.pipeline import (
    FacialRecognitionPipeline,
    InvalidImageError,
    NoFaceDetectedError,
)
from nexgen_engine.models.arcface import EngineUnavailableError
from nexgen_engine.search.gallery_index import GalleryIndex, faiss_available

from .conftest import flat_image, image_bytes, noise_image


@pytest.fixture(scope="session")
def pipeline(engine_runtime) -> FacialRecognitionPipeline:
    return FacialRecognitionPipeline(engine_runtime.config, engine_runtime)


@pytest.fixture(scope="session")
def templates(pipeline, face_paths) -> dict[str, list[np.ndarray]]:
    """Encode up to three images for each identity, once for the whole session."""
    encoded: dict[str, list[np.ndarray]] = {}
    for name, paths in face_paths.items():
        vectors = []
        for path in paths[:3]:
            try:
                vectors.append(pipeline.encode_bytes(path.read_bytes()).embedding)
            except (NoFaceDetectedError, InvalidImageError, OSError):
                continue
        if len(vectors) >= 2:
            encoded[name] = vectors

    if len(encoded) < 5:
        pytest.skip("Too few identities survived detection to measure anything.")
    return encoded


@pytest.fixture(scope="session")
def score_distributions(templates) -> tuple[np.ndarray, np.ndarray]:
    genuine = [
        float(np.dot(a, b))
        for vectors in templates.values()
        for a, b in itertools.combinations(vectors, 2)
    ]
    rng = random.Random(20260728)
    names = list(templates)
    impostor = [
        float(np.dot(rng.choice(templates[a]), rng.choice(templates[b])))
        for a, b in (rng.sample(names, 2) for _ in range(2000))
    ]
    return np.asarray(genuine), np.asarray(impostor)


# ----------------------------------------------------------------- engine ---


class TestEngineLoads:
    def test_a_real_model_is_loaded(self, engine_runtime):
        info = engine_runtime.recognizer.info
        assert info.backend == "insightface_arcface"
        assert info.embedding_dim == 512
        assert info.recognition_network  # the actual network name, not a label

    def test_detector_produces_landmarks(self, engine_runtime):
        """Without landmarks there is no proper alignment, and accuracy collapses."""
        assert engine_runtime.detector.produces_landmarks is True

    def test_device_is_reported_honestly(self, engine_runtime):
        status = engine_runtime.status()
        assert status["device"]["effective"] in {"cpu", "cuda"}
        assert status["device"]["providers"]


# ------------------------------------------------------------- embeddings ---


class TestEmbeddings:
    def test_embedding_is_512d_and_unit_length(self, pipeline, face_paths):
        path = next(iter(face_paths.values()))[0]
        result = pipeline.encode_bytes(path.read_bytes())
        assert result.embedding.shape == (512,)
        assert float(np.linalg.norm(result.embedding)) == pytest.approx(1.0, abs=1e-4)

    def test_encoding_is_deterministic(self, pipeline, face_paths):
        """The same bytes must always give the same template, or one person
        enrolled twice becomes two people."""
        payload = next(iter(face_paths.values()))[0].read_bytes()
        first = pipeline.encode_bytes(payload).embedding
        second = pipeline.encode_bytes(payload).embedding
        assert float(np.dot(first, second)) == pytest.approx(1.0, abs=1e-5)

    def test_encoding_does_not_drift_with_use(self, pipeline, face_paths):
        """Regression: query-dependent state used to leak into stored templates."""
        paths = list(face_paths.values())
        payload = paths[0][0].read_bytes()
        baseline = pipeline.encode_bytes(payload).embedding
        for other in paths[1:6]:
            pipeline.encode_bytes(other[0].read_bytes())
        assert float(np.dot(pipeline.encode_bytes(payload).embedding, baseline)) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_embeddings_are_not_hashes(self, pipeline, face_paths):
        """A hash of pixels would make two images of one person unrelated.

        This is the specific failure this codebase previously shipped, so it is
        asserted directly rather than only implied by the separation tests.
        """
        paths = next(iter(face_paths.values()))
        a = pipeline.encode_bytes(paths[0].read_bytes()).embedding
        b = pipeline.encode_bytes(paths[1].read_bytes()).embedding
        similarity = float(np.dot(a, b))
        # Independent 512-d unit vectors sit near 0 with sd ~0.044, so anything
        # above 0.15 is far outside what any hash could produce.
        assert similarity > 0.15, (
            f"Same-person similarity {similarity:.4f} is indistinguishable from random. "
            "The embedding is not encoding identity."
        )


# ------------------------------------------------------------- separation ---


class TestSeparation:
    def test_same_person_scores_higher_than_different_people(self, score_distributions):
        """The single claim this product rests on."""
        genuine, impostor = score_distributions
        assert genuine.size and impostor.size
        assert genuine.mean() > impostor.mean() + 0.15, (
            f"genuine mean {genuine.mean():.4f} vs impostor mean {impostor.mean():.4f}: "
            "the engine is not separating identities."
        )

    def test_impostor_scores_cluster_near_zero(self, score_distributions):
        _, impostor = score_distributions
        assert abs(float(impostor.mean())) < 0.20

    def test_false_match_rate_is_low_at_the_default_threshold(self, score_distributions, engine_runtime):
        genuine, impostor = score_distributions
        threshold = engine_runtime.config.thresholds.match
        false_match_rate = float((impostor >= threshold).mean())
        true_match_rate = float((genuine >= threshold).mean())
        assert false_match_rate < 0.02, (
            f"FAR {false_match_rate:.2%} at threshold {threshold}. "
            "Recalibrate with scripts/calibrate_threshold.py."
        )
        assert true_match_rate > 0.40, f"TAR only {true_match_rate:.2%} at threshold {threshold}."

    def test_report_measured_distributions(self, score_distributions, capsys):
        """Not an assertion so much as a record of what this build measured."""
        genuine, impostor = score_distributions
        with capsys.disabled():
            print(
                f"\n  genuine  n={genuine.size} mean={genuine.mean():.4f} sd={genuine.std():.4f}"
                f"\n  impostor n={impostor.size} mean={impostor.mean():.4f} sd={impostor.std():.4f}"
                f"\n  separation={genuine.mean() - impostor.mean():.4f}"
            )
        assert genuine.mean() > impostor.mean()


# --------------------------------------------------------------- gallery ----


class TestGallerySearch:
    def test_rank1_identification_finds_the_right_person(self, templates):
        """End-to-end through the index, not just pairwise similarity."""
        index = GalleryIndex(512)
        probes: list[tuple[str, np.ndarray]] = []

        for name, vectors in templates.items():
            index.add("tenant", f"{name}-enrolled", name, vectors[0])
            probes.append((name, vectors[1]))

        correct = sum(
            1
            for name, probe in probes
            if (matches := index.search("tenant", probe, top_k=1).matches)
            and matches[0].subject_id == name
        )
        accuracy = correct / len(probes)
        assert accuracy > 0.70, (
            f"Rank-1 accuracy {accuracy:.1%} over {len(probes)} probes "
            f"in a {len(templates)}-subject gallery."
        )

    def test_search_is_tenant_isolated_with_real_templates(self, templates):
        vectors = next(iter(templates.values()))
        index = GalleryIndex(512)
        index.add("tenant-a", "t1", "s1", vectors[0])
        index.add("tenant-b", "t2", "s2", vectors[0])

        outcome = index.search("tenant-a", vectors[1], top_k=10)
        assert len(outcome.matches) == 1
        assert outcome.matches[0].template_id == "t1"

    def test_faiss_and_numpy_agree(self, templates):
        """FAISS is a speed optimisation, never a different answer."""
        if not faiss_available():
            pytest.skip("faiss is not installed.")

        index = GalleryIndex(512)
        for name, vectors in templates.items():
            index.add("tenant", f"{name}-0", name, vectors[0])

        probe = next(iter(templates.values()))[1]
        shard = index._shards["tenant"]  # noqa: SLF001 - comparing the two backends

        faiss_scores = shard.scores(probe / np.linalg.norm(probe))
        numpy_scores = (shard.vectors @ (probe / np.linalg.norm(probe))).astype(np.float32)
        np.testing.assert_allclose(faiss_scores, numpy_scores, atol=1e-5)

    def test_search_latency_is_reasonable(self, templates):
        index = GalleryIndex(512)
        for name, vectors in templates.items():
            for position, vector in enumerate(vectors):
                index.add("tenant", f"{name}-{position}", name, vector)

        probe = next(iter(templates.values()))[0]
        started = time.perf_counter()
        for _ in range(20):
            index.search("tenant", probe, top_k=10)
        per_search_ms = (time.perf_counter() - started) / 20 * 1000
        assert per_search_ms < 100, f"{per_search_ms:.1f} ms per search on {index.size('tenant')} templates."


# ----------------------------------------------------------- negative path --


class TestRefusals:
    def test_flat_image_has_no_face(self, pipeline):
        with pytest.raises(NoFaceDetectedError):
            pipeline.encode_bytes(image_bytes(flat_image()))

    def test_noise_has_no_face(self, pipeline):
        with pytest.raises(NoFaceDetectedError):
            pipeline.encode_bytes(image_bytes(noise_image(seed=3)))

    def test_non_image_bytes_are_rejected(self, pipeline):
        with pytest.raises(InvalidImageError):
            pipeline.encode_bytes(b"this is not an image")

    def test_empty_payload_is_rejected(self, pipeline):
        with pytest.raises(InvalidImageError):
            pipeline.encode_bytes(b"")

    def test_unknown_model_pack_fails_loudly(self, engine_runtime):
        """A bad configuration must raise, not silently substitute something."""
        from nexgen_engine.config import EngineConfig
        from nexgen_engine.runtime import EngineRuntime

        runtime = EngineRuntime(EngineConfig(model_pack="does-not-exist"))
        with pytest.raises(EngineUnavailableError, match="Unknown model pack"):
            runtime.warm_up()


# --------------------------------------------------------------- cropping ---


class TestPreCroppedFaces:
    def test_tightly_cropped_faces_are_detected(self, pipeline, face_paths):
        """AgeDB images are 112x112 crops with no margin.

        Without pad-and-retry the detector finds nothing in any of them, which
        is the common case for mugshots and database thumbnails.
        """
        detected = 0
        for paths in list(face_paths.values())[:10]:
            try:
                pipeline.encode_bytes(paths[0].read_bytes())
                detected += 1
            except NoFaceDetectedError:
                pass
        assert detected >= 9, f"only {detected}/10 pre-cropped faces detected."

    def test_padding_is_reported(self, pipeline, face_paths):
        """The examiner should be able to tell how a detection was obtained."""
        result = pipeline.encode_bytes(next(iter(face_paths.values()))[0].read_bytes())
        assert result.padded_detection is True
        assert result.timings.total_ms > 0
