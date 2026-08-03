"""Prometheus metrics.

Registered once at import. Metric names follow the `iie_` prefix convention so a
shared Prometheus instance can scope them.

Cardinality discipline: labels are bounded vocabularies only — stage names,
provider names, status values. Never `investigation_id`, `url` or `domain`.
Unbounded labels are the standard way to melt a Prometheus server, and an OSINT
platform generates unbounded domains by design.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# -- pipeline ---------------------------------------------------------------

STAGE_DURATION = Histogram(
    "iie_pipeline_stage_duration_seconds",
    "Wall-clock duration of a pipeline stage",
    labelnames=("stage", "status"),
    registry=REGISTRY,
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 900),
)

STAGE_ITEMS = Counter(
    "iie_pipeline_stage_items_total",
    "Items processed by a pipeline stage",
    labelnames=("stage", "outcome"),   # outcome: ok | failed | skipped
    registry=REGISTRY,
)

RUNS_ACTIVE = Gauge(
    "iie_pipeline_runs_active",
    "Pipeline runs currently executing",
    registry=REGISTRY,
)

# -- providers --------------------------------------------------------------

PROVIDER_REQUESTS = Counter(
    "iie_provider_requests_total",
    "Discovery provider calls",
    labelnames=("provider", "capability", "outcome"),
    registry=REGISTRY,
)

PROVIDER_LATENCY = Histogram(
    "iie_provider_latency_seconds",
    "Discovery provider round-trip latency",
    labelnames=("provider",),
    registry=REGISTRY,
    buckets=(0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

PROVIDER_RESULTS = Counter(
    "iie_provider_results_total",
    "Results returned by providers, by local verification outcome",
    labelnames=("provider", "verification"),   # accepted | rejected | unverified
    registry=REGISTRY,
)

# -- crawler ----------------------------------------------------------------

PAGES_FETCHED = Counter(
    "iie_pages_fetched_total",
    "Pages fetched",
    labelnames=("outcome",),
    registry=REGISTRY,
)

FETCH_BLOCKED = Counter(
    "iie_fetch_blocked_total",
    "Fetches refused by the SSRF guard or robots policy",
    labelnames=("reason",),
    registry=REGISTRY,
)

# -- evidence ---------------------------------------------------------------

OBSERVATIONS_EXTRACTED = Counter(
    "iie_observations_extracted_total",
    "Observations written, by extraction method",
    labelnames=("method",),
    registry=REGISTRY,
)

FACTS_BY_STATUS = Gauge(
    "iie_facts_by_status",
    "Current fact count by evidential status",
    labelnames=("status",),
    registry=REGISTRY,
)

REVIEW_QUEUE_DEPTH = Gauge(
    "iie_review_queue_depth",
    "Review items awaiting a human decision",
    labelnames=("kind",),
    registry=REGISTRY,
)

# -- LLM --------------------------------------------------------------------

LLM_TOKENS = Counter(
    "iie_llm_tokens_total",
    "LLM tokens consumed",
    labelnames=("model", "purpose", "direction"),   # direction: input | output
    registry=REGISTRY,
)

LLM_REJECTIONS = Counter(
    "iie_llm_claim_rejections_total",
    "Copilot claims rejected by citation validation",
    labelnames=("reason",),
    registry=REGISTRY,
)
"""A rising rejection rate means the assistant is drifting toward unsupported
claims — a correctness signal, not just an operational one."""

# -- helpers ----------------------------------------------------------------


@contextmanager
def time_stage(stage: str) -> Iterator[dict[str, str]]:
    """Time a stage and record its terminal status.

    The yielded dict lets the caller set the outcome:
    ``with time_stage("CRAWL") as m: ...; m["status"] = "OK"``
    """
    import time as _time

    marker = {"status": "FAILED"}
    started = _time.perf_counter()
    try:
        yield marker
    finally:
        STAGE_DURATION.labels(stage=stage, status=marker["status"]).observe(
            _time.perf_counter() - started
        )


def render() -> bytes:
    return generate_latest(REGISTRY)


__all__ = [
    "CONTENT_TYPE",
    "FACTS_BY_STATUS",
    "FETCH_BLOCKED",
    "LLM_REJECTIONS",
    "LLM_TOKENS",
    "OBSERVATIONS_EXTRACTED",
    "PAGES_FETCHED",
    "PROVIDER_LATENCY",
    "PROVIDER_REQUESTS",
    "PROVIDER_RESULTS",
    "REGISTRY",
    "REVIEW_QUEUE_DEPTH",
    "RUNS_ACTIVE",
    "STAGE_DURATION",
    "STAGE_ITEMS",
    "render",
    "time_stage",
]
