"""Discovery providers and the findings view."""

from __future__ import annotations

import io
import random

from PIL import Image as PILImage

from providers.base import KIND_PAGE, KIND_SIMILAR_IMAGE, Appearance, DiscoveryResult
from providers.registry import load_providers, manifests, run_discovery
from shared.config import Settings
from shared.enums import ProviderCapability

CASE = {"case_id": "DISC-1", "title": "Discovery", "lawful_basis": "Engagement"}


def _png(seed: int = 5) -> bytes:
    rng = random.Random(seed)  # noqa: S311 - test fixture, not cryptography
    image = PILImage.new("RGB", (64, 64))
    for bx in range(8):
        for by in range(8):
            colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for x in range(bx * 8, bx * 8 + 8):
                for y in range(by * 8, by * 8 + 8):
                    image.putpixel((x, y), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------- plugin host --


def test_every_plugin_is_discovered():
    names = {m["name"] for m in manifests(Settings())}
    assert {"google", "manual", "tineye"} <= names


def test_manual_needs_no_credentials():
    """The provider that works on day one — and the most defensible use of the
    platform, since the operator already holds the lead."""
    entry = next(m for m in manifests(Settings()) if m["name"] == "manual")
    assert entry["requires_credentials"] is False
    assert entry["configured"] is True


def test_keyed_providers_report_unconfigured_not_empty():
    """"No matches" and "no API key" lead to opposite next steps."""
    for name in ("google", "tineye"):
        entry = next(m for m in manifests(Settings()) if m["name"] == name)
        assert entry["configured"] is False
        assert entry["status"] == "unconfigured"
        assert entry["config_keys"], "an unconfigured provider must say what would enable it"


def test_capability_filter_separates_the_two_kinds():
    """The split is the whole reason capabilities exist: an archive provider
    must never be handed image bytes, and a discovery provider must never be
    asked to date a URL."""
    discovery = {
        p.name
        for p in load_providers(Settings(), capability=ProviderCapability.IMAGE_DISCOVERY)
    }
    archive = {
        p.name
        for p in load_providers(Settings(), capability=ProviderCapability.ARCHIVE_LOOKUP)
    }
    assert {"google", "manual", "tineye"} <= discovery
    assert archive == {"wayback"}
    assert not (discovery & archive)


def test_manifests_expose_cost_before_a_run():
    google = next(m for m in manifests(Settings()) if m["name"] == "google")
    assert google["cost_per_1k"] == 3.50


# ------------------------------------------------------------ merging rules --


class _Stub:
    def __init__(self, name, appearances=(), error="", available=True):
        from providers.base import ProviderManifest

        self.manifest = ProviderManifest(
            name=name, title=name, capabilities=(ProviderCapability.IMAGE_DISCOVERY,)
        )
        self._appearances = appearances
        self._error = error
        self._available = available

    def available(self):
        return self._available

    async def discover(self, image, *, max_results=50):
        return DiscoveryResult(
            provider=self.manifest.name,
            available=self._available,
            appearances=tuple(self._appearances),
            error=self._error,
        )


class _Exploding(_Stub):
    async def discover(self, image, *, max_results=50):
        raise RuntimeError("provider on fire")


def _loaded(stub):
    from providers.registry import LoadedProvider

    return LoadedProvider(manifest=stub.manifest, instance=stub)


def _hit(url, kind=KIND_PAGE, provider="a", score=None):
    return Appearance(url=url, kind=kind, provider=provider, score=score)


async def test_results_merge_and_deduplicate():
    a = _loaded(_Stub("a", [_hit("https://x.test/1"), _hit("https://x.test/2")]))
    b = _loaded(_Stub("b", [_hit("https://x.test/2", provider="b"), _hit("https://x.test/3")]))
    summary = await run_discovery([a, b], b"img")
    assert sorted(h.url for h in summary.appearances) == [
        "https://x.test/1", "https://x.test/2", "https://x.test/3",
    ]


async def test_same_url_different_kind_is_two_facts():
    from providers.base import KIND_FULL_IMAGE

    a = _loaded(
        _Stub("a", [_hit("https://x.test/1"), _hit("https://x.test/1", kind=KIND_FULL_IMAGE)])
    )
    summary = await run_discovery([a], b"img")
    assert len(summary.appearances) == 2


async def test_exact_matches_rank_above_merely_similar():
    """Burying an exact hit under forty 'visually similar' ones would hide the
    finding that matters."""
    a = _loaded(
        _Stub("a", [_hit("https://x.test/sim", kind=KIND_SIMILAR_IMAGE), _hit("https://x.test/page")])
    )
    summary = await run_discovery([a], b"img")
    assert summary.appearances[0].kind == KIND_PAGE


async def test_one_broken_provider_does_not_suppress_another():
    good = _loaded(_Stub("good", [_hit("https://x.test/1")]))
    summary = await run_discovery([_loaded(_Exploding("boom")), good], b"img")
    assert [h.url for h in summary.appearances] == ["https://x.test/1"]
    assert "boom" in summary.failed


async def test_unconfigured_is_distinguished_from_empty():
    summary = await run_discovery(
        [_loaded(_Stub("nokey", [], error="no credentials", available=False))], b"img"
    )
    assert summary.appearances == ()
    assert summary.unconfigured == ("nokey",)
    assert summary.configured == ()


# ------------------------------------------------------------- manual leads --


async def test_manual_provider_accepts_operator_urls():
    from providers.manual.provider import ManualProvider

    provider = ManualProvider().with_urls(
        ["https://example.test/a", "https://example.test/b"]
    )
    result = await provider.discover(b"img")
    assert [a.url for a in result.appearances] == [
        "https://example.test/a", "https://example.test/b",
    ]


async def test_manual_provider_reports_rejected_urls():
    """The operator must learn which of their URLs was unusable, and why,
    rather than finding a silent gap in the results."""
    from providers.manual.provider import ManualProvider

    result = await ManualProvider().with_urls(
        ["https://ok.test/a", "javascript:alert(1)", "not a url"]
    ).discover(b"img")
    assert len(result.appearances) == 1
    assert "2 URL(s) ignored" in result.error


async def test_with_urls_returns_a_copy():
    """The registry caches providers; one investigation's leads must never leak
    into another's."""
    from providers.manual.provider import ManualProvider

    base = ManualProvider()
    scoped = base.with_urls(["https://x.test/1"])
    assert base.urls == ()
    assert scoped is not base


# ------------------------------------------------------- end to end via API --


async def _case_with_image(auth_client) -> str:
    case_id = (await auth_client.post("/api/v1/investigations", json=CASE)).json()["id"]
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("probe.png", _png(), "image/png")},
    )
    return case_id


async def test_providers_endpoint_reports_state_before_searching(auth_client):
    listing = (await auth_client.get("/api/v1/providers")).json()
    by_name = {p["name"]: p for p in listing}
    assert by_name["manual"]["configured"] is True
    assert by_name["google"]["configured"] is False
    assert "IIE_GOOGLE_VISION_API_KEY" in by_name["google"]["config_keys"]


async def test_findings_before_any_search_say_so(auth_client):
    case_id = await _case_with_image(auth_client)
    body = (await auth_client.get(f"/api/v1/investigations/{case_id}/findings")).json()
    assert body["searched"] is False
    assert body["summary"]["sources_found"] == 0
    # The investigator is told which providers would need a key.
    assert {p["name"] for p in body["providers"]["unconfigured"]} == {"google", "tineye"}


async def test_discovery_with_operator_urls_produces_findings(auth_client):
    case_id = await _case_with_image(auth_client)
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover",
        json={"urls": ["https://newsroom.example/story", "https://registry.example/profile"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["searched"] is True
    assert body["summary"]["sources_found"] == 2
    assert body["summary"]["distinct_sites"] == 2
    assert {f["site"] for f in body["findings"]} == {
        "newsroom.example", "registry.example",
    }
    assert all(f["verification"] == "PENDING" for f in body["findings"])


async def test_discovery_requires_a_probe_image(auth_client):
    case_id = (await auth_client.post("/api/v1/investigations", json=CASE)).json()["id"]
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover", json={"urls": []}
    )
    assert response.status_code == 422
    assert "nothing to search for" in response.json()["detail"]


async def test_running_discovery_twice_does_not_duplicate(auth_client):
    case_id = await _case_with_image(auth_client)
    payload = {"urls": ["https://newsroom.example/story"]}
    await auth_client.post(f"/api/v1/investigations/{case_id}/discover", json=payload)
    second = await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover", json=payload
    )
    assert second.json()["summary"]["sources_found"] == 1


async def test_discovery_records_how_evidence_was_found(auth_client, session_factory):
    """'We asked and got nothing' is a different fact from 'we never asked',
    and only the first can be relied on later."""
    from sqlalchemy import select

    from database.models import DiscoveryRequest

    case_id = await _case_with_image(auth_client)
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover",
        json={"urls": ["https://newsroom.example/story"]},
    )
    async with session_factory() as session:
        rows = (await session.execute(select(DiscoveryRequest))).scalars().all()

    providers = {r.provider for r in rows}
    assert "manual" in providers
    # Unconfigured providers are recorded as asked-and-unavailable too.
    assert {"google", "tineye"} <= providers


async def test_discovery_is_audited(auth_client):
    case_id = await _case_with_image(auth_client)
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover",
        json={"urls": ["https://newsroom.example/story"]},
    )
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    runs = [e for e in entries if e["action"] == "discovery.run"]
    assert runs
    assert runs[-1]["detail"]["found"] == 1
    assert "google" in runs[-1]["detail"]["providers_unconfigured"]


# ------------------------------------------------------ archive lookup ------


def test_wayback_is_free_and_configured_by_default():
    """The one provider besides `manual` that works with no key at all."""
    entry = next(m for m in manifests(Settings()) if m["name"] == "wayback")
    assert entry["requires_credentials"] is False
    assert entry["configured"] is True
    assert entry["cost_per_1k"] is None
    assert entry["capabilities"] == ["ARCHIVE_LOOKUP"]


def test_wayback_is_not_an_image_discovery_provider():
    """It takes a URL, not image bytes. The capability split exists so it is
    never asked a question it cannot answer."""
    discovery = {
        p.name
        for p in load_providers(Settings(), capability=ProviderCapability.IMAGE_DISCOVERY)
    }
    assert "wayback" not in discovery

    archive = {
        p.name
        for p in load_providers(Settings(), capability=ProviderCapability.ARCHIVE_LOOKUP)
    }
    assert archive == {"wayback"}


def test_cdx_timestamps_render_as_dates():
    """A capture *time* reflects when the crawler passed, not a fact about the
    page, so showing seconds would imply precision that does not exist."""
    from providers.wayback.provider import _readable

    assert _readable("20020120142510") == "2002-01-20"
    assert _readable("garbage") == "garbage"[:8]


class _StubArchive:
    def __init__(self, record=None, raises=False):
        from providers.base import ProviderManifest

        self.manifest = ProviderManifest(
            name="stub-archive", title="stub",
            capabilities=(ProviderCapability.ARCHIVE_LOOKUP,),
            requires_credentials=False,
        )
        self._record = record
        self._raises = raises

    def available(self):
        return True

    async def lookup(self, url):
        from providers.base import ArchiveRecord

        if self._raises:
            raise RuntimeError("archive unreachable")
        return self._record or ArchiveRecord(url=url, available=True)


async def test_archive_lookups_run_for_every_url():
    from providers.base import ArchiveRecord
    from providers.registry import run_archive_lookups

    stub = _loaded(
        _StubArchive(ArchiveRecord(url="x", available=True, first_seen="2019-04-02"))
    )
    records = await run_archive_lookups([stub], ["https://a.test/1", "https://b.test/2"])
    assert set(records) == {"https://a.test/1", "https://b.test/2"}
    assert all(r.first_seen == "2019-04-02" for r in records.values())


async def test_a_failing_archive_lookup_does_not_break_the_others():
    from providers.registry import run_archive_lookups

    records = await run_archive_lookups(
        [_loaded(_StubArchive(raises=True))], ["https://a.test/1"]
    )
    assert records["https://a.test/1"].error
    assert "archive unreachable" in records["https://a.test/1"].error


async def test_archive_lookups_are_capped():
    """The archive is a free courtesy service; hammering it would be both rude
    and self-defeating."""
    from providers.registry import run_archive_lookups

    urls = [f"https://a.test/{i}" for i in range(60)]
    records = await run_archive_lookups([_loaded(_StubArchive())], urls, limit=5)
    assert len(records) == 5


async def test_no_archive_provider_is_not_an_error():
    from providers.registry import run_archive_lookups

    assert await run_archive_lookups([], ["https://a.test/1"]) == {}


# ------------------------------------------------------ source identity -----


def test_public_suffixes_are_not_mistaken_for_sources():
    """`co.uk` is a public suffix, not a source. Naive last-two-labels gets this
    visibly wrong, and it matters beyond display: one site must count as one
    source when independence is scored."""
    from image_discovery.service import registrable_domain

    assert registrable_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert registrable_domain("https://www.itv.co.uk/x") == "itv.co.uk"


def test_subdomains_collapse_to_one_source():
    """`example.com` and `blog.example.com` are one operator, so one source."""
    from image_discovery.service import registrable_domain

    assert registrable_domain("https://blog.example.com/x") == "example.com"
    assert registrable_domain("https://example.com/") == "example.com"


def test_private_suffixes_keep_authors_apart():
    """Two GitHub Pages sites are two unrelated authors; collapsing them to
    `github.io` would score them as a single source."""
    from image_discovery.service import registrable_domain

    assert registrable_domain("https://alice.github.io/p") == "alice.github.io"
    assert registrable_domain("https://bob.github.io/p") == "bob.github.io"


def test_unknown_suffixes_do_not_merge_distinct_sites():
    """The dangerous direction to be wrong in.

    With no recognised public suffix, falling back to the domain *label* makes
    `newsroom.example` and `registry.example` both reduce to `example` — two
    unrelated sites scored as one source, understating independence. Falling
    back to the hostname over-counts instead, which is the safe error.
    """
    from image_discovery.service import registrable_domain

    first = registrable_domain("https://newsroom.example/story")
    second = registrable_domain("https://registry.example/profile")
    assert first != second
    assert first == "newsroom.example"


async def test_several_urls_on_one_site_insert_that_domain_once(auth_client):
    """Rows added earlier in the batch are not yet flushed, so a plain
    SELECT-then-INSERT would try to insert the same domain twice.

    Uses a real TLD deliberately: with a recognised public suffix, subdomains
    collapse and three pages on one operator count as one source.
    """
    case_id = await _case_with_image(auth_client)
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/discover",
        json={
            "urls": [
                "https://newsroom.test-site.com/story-one",
                "https://newsroom.test-site.com/story-two",
                "https://blog.test-site.com/story-three",
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"]["sources_found"] == 3
    # Three pages, one operator — one source.
    assert body["summary"]["distinct_sites"] == 1


async def test_unrecognised_tlds_over_count_rather_than_under_count(auth_client):
    """The documented consequence of the hostname fallback.

    Without a recognised suffix, subdomains cannot be collapsed safely, so each
    host counts separately. That over-states independence, which is the
    direction that makes a finding look *weaker* than it is — the opposite
    error would inflate confidence.
    """
    case_id = await _case_with_image(auth_client)
    body = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/discover",
            json={
                "urls": [
                    "https://newsroom.example/a",
                    "https://blog.newsroom.example/b",
                ]
            },
        )
    ).json()
    assert body["summary"]["distinct_sites"] == 2


async def test_a_slow_latest_query_never_misdates_the_finding():
    """`limit=-1` makes the archive scan its whole index and times out on busy
    URLs. Falling back to the first capture would present the *oldest* snapshot
    as the most recent — a misdated finding is worse than a missing field.
    """
    import httpx

    from providers.wayback.provider import WaybackProvider

    provider = WaybackProvider(timeout=1.0)
    calls = {"n": 0}

    async def fake_cdx(client, url, limit):
        calls["n"] += 1
        if limit == "1":
            return [["20050317120000", url, "200"]]
        raise httpx.ReadTimeout("index scan too slow")

    provider._cdx = fake_cdx
    record = await provider.lookup("https://busy.test/news")

    assert record.first_seen == "2005-03-17"
    assert record.last_seen == "", "an unknown latest capture must stay empty"
    assert record.snapshot_count == 1
    assert record.archived_url.endswith("/20050317120000/https://busy.test/news")
    assert calls["n"] == 2, "the earliest query must still have been made"
