"""Tests for the forensic evidence layer.

Several tests encode failures that actually happened on 2026-08-02 rather than
hypothetical ones. Those are marked REGRESSION and exist so that a future change
reintroducing the bug fails loudly instead of producing a plausible wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexgen_engine.forensics.calibration import (
    ConditionalCalibrator,
    LogisticCalibrator,
    cross_validated_log10_lr,
)
from nexgen_engine.forensics.evidence import EvidenceReport, Provenance, verbal_scale
from nexgen_engine.forensics.information import capacity_from_pools, gallery_for_rank1, identity_bits
from nexgen_engine.forensics.memory import IdentityLog, Observation, Status
from nexgen_engine.forensics.metrics import cllr, cllr_report, pav, pav_log10_lr, tippett
from nexgen_engine.forensics.openset import ConformalCalibrator, posterior_over_hypotheses
from nexgen_engine.forensics.population import (
    PopulationPurityError,
    ReferencePopulation,
    audit_population,
)
from nexgen_engine.forensics.provenance import Artifact, LineageLedger, canonical_digest


def _separable(n=2000, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(sep, 1.0, n)
    i = rng.normal(0.0, 1.0, n)
    scores = np.empty(2 * n)
    labels = np.zeros(2 * n, dtype=bool)
    scores[0::2], labels[0::2] = g, True
    scores[1::2] = i
    return scores, labels


# --------------------------------------------------------------------- metrics


class TestCllr:
    def test_uninformative_system_scores_one(self):
        """LR = 1 everywhere delivers nothing, and Cllr must say exactly 1.0."""
        assert cllr(np.zeros(500), np.zeros(500)) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_system_approaches_zero(self):
        assert cllr(np.full(500, 20.0), np.full(500, -20.0)) < 1e-6

    def test_backwards_system_scores_worse_than_silence(self):
        """A system with inverted evidence must score above 1, not merely badly."""
        assert cllr(np.full(200, -3.0), np.full(200, 3.0)) > 1.0

    def test_class_balance_does_not_change_cllr(self):
        """The property that makes Cllr reportable: it is prior-independent."""
        g, i = np.full(100, 2.0), np.full(900, -2.0)
        assert cllr(g, i) == pytest.approx(cllr(np.full(900, 2.0), np.full(100, -2.0)))

    def test_decomposition_is_exact_and_min_is_a_floor(self):
        scores, labels = _separable()
        lr = cross_validated_log10_lr(scores, labels)
        r = cllr_report(lr, labels, scores=scores)
        assert r.cllr == pytest.approx(r.cllr_min + r.cllr_cal, abs=1e-12)
        assert r.cllr_min <= r.cllr + 1e-9
        assert r.cllr_cal >= -1e-9


class TestPAV:
    def test_output_is_monotone(self):
        scores, labels = _separable(500)
        fitted = pav(scores, labels)
        order = np.argsort(scores)
        assert np.all(np.diff(fitted[order]) >= -1e-12)

    def test_pav_is_optimal_so_it_cannot_be_beaten(self):
        scores, labels = _separable(1000)
        oracle = cllr_report(pav_log10_lr(scores, labels), labels, scores=scores)
        fitted = cllr_report(cross_validated_log10_lr(scores, labels), labels, scores=scores)
        assert oracle.cllr <= fitted.cllr + 1e-9


class TestTippett:
    def test_misleading_rates_are_directional(self):
        scores, labels = _separable(1000, sep=3.0)
        t = tippett(cross_validated_log10_lr(scores, labels), labels)
        assert 0.0 <= t.rate_misleading_same_source < 0.2
        assert 0.0 <= t.rate_misleading_different_source < 0.2


# ----------------------------------------------------------------- calibration


class TestCalibration:
    def test_llr_is_prior_independent(self):
        """Fitted on 10:1 imbalance, the LLR must match the balanced fit.

        This is what makes the output a likelihood ratio rather than a posterior.
        """
        rng = np.random.default_rng(0)
        g = rng.normal(2.0, 1.0, 2000)
        i = rng.normal(0.0, 1.0, 2000)
        bal = LogisticCalibrator().fit(np.r_[g, i], np.r_[np.ones(2000), np.zeros(2000)].astype(bool))
        imb = LogisticCalibrator().fit(
            np.r_[g[:200], i], np.r_[np.ones(200), np.zeros(2000)].astype(bool)
        )
        probe = np.linspace(-2, 4, 25)
        assert np.allclose(bal.log10_lr(probe), imb.log10_lr(probe), atol=0.25)

    def test_monotone_in_score(self):
        scores, labels = _separable()
        cal = LogisticCalibrator().fit(scores, labels)
        out = cal.log10_lr(np.linspace(-3, 5, 50))
        assert np.all(np.diff(out) > 0)

    def test_perfect_separation_does_not_produce_infinite_lr(self):
        """REGRESSION: without the ridge term a separable fit sends |a| to
        infinity and every reported LR becomes arbitrary."""
        s = np.r_[np.full(200, 5.0), np.full(200, -5.0)]
        y = np.r_[np.ones(200), np.zeros(200)].astype(bool)
        out = LogisticCalibrator().fit(s, y).log10_lr(s)
        assert np.all(np.isfinite(out))

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            LogisticCalibrator().fit(np.arange(10.0), np.ones(10, dtype=bool))

    def test_cross_validated_is_not_better_than_oracle(self):
        scores, labels = _separable()
        held = cllr_report(cross_validated_log10_lr(scores, labels), labels, scores=scores)
        oracle_cal = LogisticCalibrator().fit(scores, labels).log10_lr(scores)
        oracle = cllr_report(oracle_cal, labels, scores=scores)
        assert held.cllr >= oracle.cllr - 1e-6

    def test_conditional_falls_back_when_a_bin_is_too_small(self):
        scores, labels = _separable(1000)
        cond = np.r_[np.zeros(1990), np.ones(10)]
        cal = ConditionalCalibrator(bin_edges=[0.5], min_per_bin=200).fit(scores, labels, cond)
        assert 1 not in cal.bins  # underpopulated bin refused
        assert np.all(np.isfinite(cal.log10_lr(scores, cond)))


# ------------------------------------------------------------------ population


class TestPopulation:
    def test_unlabelled_pool_is_always_refused(self):
        """REGRESSION: a pool built by random cross-pairing without identity
        labels contained 0.2% same-person pairs -- twice the FAR being measured."""
        a = audit_population(np.random.default_rng(0).normal(0, 0.1, 50_000), False)
        assert a.verdict == "unlabelled"
        assert not a.usable

    def test_underpowered_pool_is_refused(self):
        a = audit_population(np.random.default_rng(0).normal(0, 0.1, 500), True, target_far=1e-3)
        assert a.verdict == "underpowered"
        assert not a.usable

    def test_probe_is_condition_relative(self):
        """REGRESSION: a fixed 0.5 probe refused a VALID TinyFace population,
        because on degraded imagery impostors genuinely reach 0.5. The probe must
        derive from the corpus's own genuine distribution."""
        rng = np.random.default_rng(0)
        imp = rng.normal(0.25, 0.12, 200_000)  # degraded: heavy tail past 0.5
        gen = rng.normal(0.45, 0.15, 5_000)
        fixed = audit_population(imp, True, contamination_probe=0.5)
        relative = audit_population(imp, True, genuine_reference=gen)
        assert fixed.suspect_fraction > relative.suspect_fraction
        assert relative.usable

    def test_from_labelled_rejects_same_identity_pairs(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(0, 1, (400, 32))
        ids = np.repeat(np.arange(40), 10)
        pop = ReferencePopulation.from_labelled(
            "t", "test", emb, ids, n_samples=200_000, strict=False
        )
        assert pop.audit.identity_labels_present

    def test_typicality_never_returns_zero(self):
        """A zero tail would produce an infinite LR."""
        rng = np.random.default_rng(0)
        emb = rng.normal(0, 1, (300, 16))
        ids = np.arange(300)
        pop = ReferencePopulation.from_labelled("t", "d", emb, ids, n_samples=100_000, strict=False)
        assert float(pop.typicality(10.0)[0]) > 0.0

    def test_unusable_population_raises_on_query(self):
        pop = ReferencePopulation.unverified("bad", "no labels", np.zeros(100))
        with pytest.raises(PopulationPurityError):
            pop.typicality(0.5)


# ----------------------------------------------------------------- information


class TestCapacity:
    def test_censoring_is_detected_and_reported(self):
        """REGRESSION: LFW came back 99.7% censored against 3,000 impostors and
        the ceiling was reported as if it were a measurement."""
        rng = np.random.default_rng(0)
        cap = capacity_from_pools("t", rng.normal(5.0, 0.1, 500), rng.normal(0.0, 1.0, 3000))
        assert cap.censored_fraction > 0.9
        assert cap.bits_median <= cap.censoring_ceiling_bits + 1e-9

    def test_bits_increase_with_separation(self):
        rng = np.random.default_rng(0)
        imp = rng.normal(0.0, 1.0, 200_000)
        weak, _ = identity_bits(rng.normal(1.0, 0.5, 1000), imp)
        strong, _ = identity_bits(rng.normal(3.0, 0.5, 1000), imp)
        assert np.median(strong) > np.median(weak)

    def test_gallery_bound_is_monotone_in_target(self):
        bits = np.random.default_rng(0).normal(10, 2, 1000)
        assert gallery_for_rank1(bits, 0.5) > gallery_for_rank1(bits, 0.9)

    def test_gallery_matches_the_two_to_the_bits_rule(self):
        bits = np.full(1000, 10.0)
        assert gallery_for_rank1(bits, 0.5, tolerance=1.0) == pytest.approx(1024.0)


# ---------------------------------------------------------------------- openset


class TestOpenSet:
    def test_posterior_sums_to_one_including_unknown(self):
        post, unk = posterior_over_hypotheses(np.array([2.0, 1.0, 0.0]), prior_unknown=0.5)
        assert post.sum() + unk == pytest.approx(1.0)

    def test_weak_evidence_leaves_mass_on_unknown(self):
        _, unk = posterior_over_hypotheses(np.zeros(10), prior_unknown=0.5)
        assert unk > 0.4

    def test_strong_evidence_moves_mass_off_unknown(self):
        _, unk = posterior_over_hypotheses(np.array([8.0, 0.0, 0.0]), prior_unknown=0.5)
        assert unk < 0.01

    def test_conformal_threshold_admits_the_expected_share(self):
        g = np.random.default_rng(0).normal(0.6, 0.1, 5000)
        cal = ConformalCalibrator(alpha=0.05).fit(g)
        held = np.random.default_rng(1).normal(0.6, 0.1, 5000)
        assert cal.admit(held).mean() == pytest.approx(0.95, abs=0.03)

    def test_conformal_needs_enough_calibration_data(self):
        with pytest.raises(ValueError):
            ConformalCalibrator().fit(np.zeros(5))


# --------------------------------------------------------------------- evidence


class TestEvidence:
    def _prov(self, verdict="valid"):
        return Provenance(
            model="w600k_r50", model_sha256="a" * 64, device="cpu",
            calibration_source="agedb_30", calibration_cllr=0.11,
            reference_population="test", population_verdict=verdict, engine_version="test",
        )

    def test_verbal_scale_is_symmetric(self):
        assert "same source" in verbal_scale(5.0)
        assert "different sources" in verbal_scale(-5.0)
        assert "no meaningful support" in verbal_scale(0.05)

    def test_capacity_guard_fires_when_lr_exceeds_available_bits(self):
        r = EvidenceReport(score=0.9, log10_lr=8.0, provenance=self._prov(), bits_available=2.0)
        assert r.capacity_flag
        assert not r.reportable

    def test_capacity_guard_silent_when_lr_is_supported(self):
        r = EvidenceReport(score=0.9, log10_lr=1.0, provenance=self._prov(), bits_available=20.0)
        assert not r.capacity_flag
        assert r.reportable

    def test_missing_population_makes_the_lr_unreportable(self):
        r = EvidenceReport(score=0.5, log10_lr=1.0, provenance=self._prov(), bits_available=None)
        assert r.capacity_flag and not r.reportable

    def test_invalid_population_is_the_first_limitation(self):
        r = EvidenceReport(score=0.5, log10_lr=2.0, provenance=self._prov("unlabelled"), bits_available=30.0)
        assert not r.reportable
        assert "NOT VALID" in r.limitations[0] or "CAPACITY" in r.limitations[0]

    def test_text_report_never_shows_a_number_without_limitations(self):
        text = EvidenceReport(
            score=0.7, log10_lr=3.0, provenance=self._prov(), bits_available=30.0
        ).to_text()
        assert "LIMITATIONS" in text and "PROVENANCE" in text
        assert "investigative leads" in text


# ------------------------------------------------------------------- provenance


class TestLineage:
    def test_chain_verifies_when_intact(self):
        led = LineageLedger()
        for k in range(5):
            led.record(Artifact(digest=canonical_digest({"k": k}), kind="frame", operation="op"))
        assert led.verify()[0]

    def test_tampering_is_detected(self):
        led = LineageLedger()
        for k in range(5):
            led.record(Artifact(digest=canonical_digest({"k": k}), kind="frame", operation="op"))
        led.entries[2]["artifact"]["operation"] = "altered"
        ok, msg = led.verify()
        assert not ok and "entry 2" in msg

    def test_deletion_is_detected(self):
        led = LineageLedger()
        for k in range(5):
            led.record(Artifact(digest=canonical_digest({"k": k}), kind="frame", operation="op"))
        del led.entries[2]
        assert not led.verify()[0]

    def test_ancestors_traverses_the_dag(self):
        led = LineageLedger()
        a = Artifact(digest="a" * 64, kind="source_media", operation="ingest")
        led.record(a)
        b = Artifact(digest="b" * 64, kind="crop", operation="align", parents=(a.digest,))
        led.record(b)
        assert [x["digest"] for x in led.ancestors(b.digest)] == [a.digest, b.digest]

    def test_digest_is_order_independent(self):
        assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})


# ----------------------------------------------------------------------- memory


def _obs(oid, emb, lr=2.0, status=Status.QUARANTINED):
    return Observation(
        observation_id=oid, artifact_digest=oid * 8, embedding=emb,
        log10_lr=lr, condition="clean", status=status,
    )


class TestMemory:
    def test_automated_writes_never_reach_the_model(self):
        """The control that prevents the system bootstrapping its own errors."""
        log = IdentityLog("s1")
        log.propose(_obs("o1", np.ones(8)))
        assert log.replay().n_admitted == 0
        assert log.replay().n_quarantined == 1

    def test_only_adjudication_admits(self):
        log = IdentityLog("s1")
        log.propose(_obs("o1", np.ones(8)))
        assert log.adjudicate("o1", True, "examiner")
        assert log.replay().n_admitted == 1

    def test_weak_evidence_is_refused_at_intake(self):
        log = IdentityLog("s1", min_log10_lr=1.0)
        ok, msg = log.propose(_obs("o1", np.ones(8), lr=0.2))
        assert not ok and "below intake floor" in msg

    def test_state_is_a_pure_function_of_the_log(self):
        rng = np.random.default_rng(0)
        log = IdentityLog("s1")
        for k in range(4):
            log.propose(_obs(f"o{k}", rng.normal(0, 1, 8)))
            log.adjudicate(f"o{k}", True, "examiner")
        assert np.allclose(log.replay().centroid, log.replay().centroid)

    def test_retraction_is_exactly_reversible(self):
        """The requirement: recompute conclusions without a disproven exhibit."""
        rng = np.random.default_rng(0)
        log = IdentityLog("s1")
        for k in range(4):
            log.propose(_obs(f"o{k}", rng.normal(0, 1, 8)))
            log.adjudicate(f"o{k}", True, "examiner")
        without = log.replay(exclude={"o3"})
        log.retract("o3", "examiner", "disproven at trial")
        assert np.allclose(log.replay().centroid, without.centroid)
        assert log.replay().n_retracted == 1

    def test_counterfactual_reports_the_change(self):
        rng = np.random.default_rng(0)
        log = IdentityLog("s1")
        for k in range(3):
            log.propose(_obs(f"o{k}", rng.normal(0, 1, 8)))
            log.adjudicate(f"o{k}", True, "examiner")
        cf = log.counterfactual("o1")
        assert cf["before"]["n_admitted"] == 3
        assert cf["after"]["n_admitted"] == 2
        assert cf["conclusion_changed"]

    def test_two_people_in_one_dossier_are_flagged(self):
        log = IdentityLog("s1", coherence_floor=0.5)
        a = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
        for k, e in enumerate([a, a, b, b]):
            log.propose(_obs(f"o{k}", e))
            log.adjudicate(f"o{k}", True, "examiner")
        st = log.replay()
        assert st.bimodal_flag and "more than" in st.bimodal_detail
