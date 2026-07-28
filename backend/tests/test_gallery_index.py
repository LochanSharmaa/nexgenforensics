from __future__ import annotations

import numpy as np
import pytest

from nexgen_engine.search.gallery_index import GalleryIndex


def unit(seed: int, dimensions: int = 512) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dimensions).astype(np.float32)
    return vector / np.linalg.norm(vector)


class TestTenantIsolation:
    """The property that must never regress: a probe cannot reach another
    tenant's templates. Everything else in the system depends on this."""

    def test_search_never_crosses_tenants(self):
        index = GalleryIndex(512)
        shared = unit(1)
        index.add("tenant-a", "template-a", "subject-a", shared)
        index.add("tenant-b", "template-b", "subject-b", shared)

        outcome = index.search("tenant-a", shared, top_k=10)

        assert len(outcome.matches) == 1
        assert outcome.matches[0].template_id == "template-a"
        assert outcome.gallery_size == 1

    def test_unknown_tenant_returns_nothing(self):
        index = GalleryIndex(512)
        index.add("tenant-a", "template-a", "subject-a", unit(2))
        outcome = index.search("tenant-does-not-exist", unit(2), top_k=10)
        assert outcome.matches == ()
        assert outcome.gallery_size == 0

    def test_clearing_one_tenant_leaves_others_intact(self):
        index = GalleryIndex(512)
        index.add("tenant-a", "t1", "s1", unit(3))
        index.add("tenant-b", "t2", "s2", unit(4))
        index.clear("tenant-a")
        assert index.size("tenant-a") == 0
        assert index.size("tenant-b") == 1


class TestSearch:
    def test_identical_template_scores_one(self):
        index = GalleryIndex(512)
        vector = unit(5)
        index.add("t", "template-1", "subject-1", vector)
        assert index.search("t", vector, top_k=1).top_score == pytest.approx(1.0, abs=1e-5)

    def test_results_are_ranked_descending(self):
        index = GalleryIndex(512)
        for i in range(10):
            index.add("t", f"template-{i}", f"subject-{i}", unit(100 + i))
        scores = [match.score for match in index.search("t", unit(105), top_k=10).matches]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_subjects_collapse_to_their_best_template(self):
        """One well-enrolled subject must not fill the whole candidate list."""
        index = GalleryIndex(512)
        base = unit(6)
        for i in range(5):
            index.add("t", f"template-{i}", "subject-shared", base)
        index.add("t", "template-other", "subject-other", unit(7))

        matches = index.search("t", base, top_k=5).matches
        assert [match.subject_id for match in matches] == ["subject-shared", "subject-other"]

    def test_collapse_can_be_disabled(self):
        index = GalleryIndex(512)
        base = unit(8)
        for i in range(3):
            index.add("t", f"template-{i}", "subject-shared", base)
        matches = index.search("t", base, top_k=5, collapse_subjects=False).matches
        assert len(matches) == 3

    def test_min_score_truncates_results(self):
        index = GalleryIndex(512)
        for i in range(20):
            index.add("t", f"template-{i}", f"subject-{i}", unit(200 + i))
        outcome = index.search("t", unit(205), top_k=20, min_score=0.99)
        assert all(match.score >= 0.99 for match in outcome.matches)

    def test_top_k_is_respected(self):
        index = GalleryIndex(512)
        for i in range(50):
            index.add("t", f"template-{i}", f"subject-{i}", unit(300 + i))
        assert len(index.search("t", unit(305), top_k=7).matches) == 7

    def test_margin_reflects_the_top_two_gap(self):
        index = GalleryIndex(512)
        probe = unit(9)
        index.add("t", "near", "subject-near", probe)
        far = unit(10)
        index.add("t", "far", "subject-far", far)
        outcome = index.search("t", probe, top_k=2)
        assert outcome.margin == pytest.approx(outcome.matches[0].score - outcome.matches[1].score)


class TestMutation:
    def test_remove_drops_the_template(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(11))
        assert index.remove("t", "template-1") is True
        assert index.size("t") == 0

    def test_remove_unknown_template_is_false(self):
        assert GalleryIndex(512).remove("t", "nope") is False

    def test_removal_keeps_remaining_rows_aligned(self):
        """Deleting from the middle must not shift ids away from their vectors."""
        index = GalleryIndex(512)
        vectors = {f"template-{i}": unit(400 + i) for i in range(5)}
        for template_id, vector in vectors.items():
            index.add("t", template_id, f"subject-{template_id}", vector)

        index.remove("t", "template-2")

        for template_id in ("template-0", "template-1", "template-3", "template-4"):
            outcome = index.search("t", vectors[template_id], top_k=1)
            assert outcome.matches[0].template_id == template_id
            assert outcome.matches[0].score == pytest.approx(1.0, abs=1e-5)

    def test_remove_subject_drops_every_template(self):
        index = GalleryIndex(512)
        for i in range(4):
            index.add("t", f"template-{i}", "subject-x", unit(500 + i))
        index.add("t", "keep", "subject-y", unit(600))

        assert index.remove_subject("t", "subject-x") == 4
        assert index.size("t") == 1

    def test_re_adding_the_same_id_replaces_it(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(12))
        index.add("t", "template-1", "subject-1", unit(13))
        assert index.size("t") == 1
        assert index.search("t", unit(13), top_k=1).top_score == pytest.approx(1.0, abs=1e-5)

    def test_add_many_matches_repeated_add(self):
        rows = [(f"template-{i}", f"subject-{i}", unit(700 + i), {}) for i in range(6)]
        bulk = GalleryIndex(512)
        bulk.add_many("t", rows)
        single = GalleryIndex(512)
        for template_id, subject_id, vector, meta in rows:
            single.add("t", template_id, subject_id, vector, meta)

        probe = unit(703)
        assert [m.template_id for m in bulk.search("t", probe, top_k=6).matches] == [
            m.template_id for m in single.search("t", probe, top_k=6).matches
        ]

    def test_subject_count_is_distinct(self):
        index = GalleryIndex(512)
        for i in range(3):
            index.add("t", f"template-{i}", "subject-1", unit(800 + i))
        index.add("t", "template-x", "subject-2", unit(900))
        assert index.size("t") == 4
        assert index.subject_count("t") == 2


class TestValidation:
    def test_wrong_dimension_is_rejected(self):
        with pytest.raises(ValueError, match="512-d"):
            GalleryIndex(512).add("t", "template-1", "subject-1", np.ones(128, dtype=np.float32))

    def test_nan_template_is_rejected(self):
        broken = np.full(512, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            GalleryIndex(512).add("t", "template-1", "subject-1", broken)

    def test_unnormalized_input_is_normalized(self):
        index = GalleryIndex(512)
        index.add("t", "template-1", "subject-1", unit(14) * 42.0)
        assert index.search("t", unit(14), top_k=1).top_score == pytest.approx(1.0, abs=1e-5)
