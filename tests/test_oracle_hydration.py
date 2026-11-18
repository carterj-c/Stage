from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from stage.domain import Company, DetailFetch, Platform
from stage.http import HttpClient, RequestRefusedError, profile
from stage.services.screening import Screener, normalize_batch
from stage.services.sync import _persist_source_batch
from stage.sources.oracle_cloud import DETAIL_PATH, MAX_PAGES, OracleCloudAdapter, _detail_body
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 11, 18, tzinfo=UTC)
COMPANY = Company(
    name="Macy's",
    platform=Platform.ORACLE_CLOUD,
    slug="macys",
    oracle_host="ebwh.fa.us2.oraclecloud.com",
    oracle_site="CX_1001",
)
ADAPTER = OracleCloudAdapter()
DETAIL_URL = f"https://{COMPANY.oracle_host}{DETAIL_PATH}"
BODY = "<p>Develop software. Must be a US citizen.</p><p>GPA &gt; 3, x&lt;y and y&gt;z.</p>"


def _posting(identifier: str = "91731") -> dict[str, object]:
    return {
        "Id": identifier,
        "Title": "Software Engineer Intern Summer 2027",
        "PrimaryLocation": "New York, NY",
        "ShortDescriptionStr": "Build systems",
    }


def _listing(*rows: dict[str, object], total: int | None = None) -> dict[str, object]:
    return {
        "items": [
            {"TotalJobsCount": len(rows) if total is None else total, "requisitionList": list(rows)}
        ]
    }


def _detail(identifier: str = "91731", body: str = BODY) -> dict[str, object]:
    return {
        "items": [
            {
                "Id": identifier,
                "ExternalDescriptionStr": body,
                "ExternalResponsibilitiesStr": "Responsibilities: build services",
                "ExternalQualificationsStr": "Qualifications: Python",
                "CorporateDescriptionStr": "Corporate information",
                "OrganizationDescriptionStr": "Organization information",
            }
        ]
    }


def _id(identifier: str = "91731") -> str:
    return ADAPTER._to_job(
        COMPANY, ADAPTER._validate(COMPANY, _listing(_posting(identifier)))[0][0], NOW
    ).id


def _client(**kwargs: Any) -> HttpClient:
    return HttpClient(
        allowed_hosts=ADAPTER.hosts_for((COMPANY,)),
        posture=replace(profile(ADAPTER.rate_profile), min_interval_s=0),
        jitter=False,
        **kwargs,
    )


@respx.mock
async def test_summary_full_detail_and_next_sync_preserve_and_rescreen(db_path: Path) -> None:
    listing = respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client, open_repository(db_path) as repo:
        summary = await ADAPTER.fetch(COMPANY, client, NOW)
        assert summary.jobs[0].description == ""
        await _persist_source_batch(
            repo, Screener(), SourceBatch(source=ADAPTER.name, run_started_at=NOW), summary.jobs
        )
        assert await repo.detail_queue(ADAPTER.name, 20) == [_id()]
        full = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert full.authoritative and full.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        assert detail.calls[0].request.url.params["finder"] == "ById;Id=91731,siteNumber=CX_1001"
        body = full.jobs[0].description
        assert "x<y and y>z" in body
        for section in ("Responsibilities", "Qualifications", "Corporate", "Organization"):
            assert section in body
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(hours=1),
                detail_fetches=full.detail_fetches,
            ),
            full.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 20) == []
        later = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=2))
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(source=ADAPTER.name, run_started_at=NOW + timedelta(hours=2)),
            later.jobs,
        )
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == body
        assert stored.work_auth_flag is True
        # Even a direct stale empty-body write cannot erase the body.
        await repo.apply_source_batch(
            SourceBatch(source=ADAPTER.name, run_started_at=NOW, jobs=later.jobs)
        )
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == body
    assert listing.call_count == 3 and detail.call_count == 1


@respx.mock
async def test_classified_legacy_short_summary_is_eligible_and_failures_retry(
    db_path: Path,
) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(404)
    async with _client() as client, open_repository(db_path) as repo:
        raw = (await ADAPTER.fetch(COMPANY, client, NOW)).jobs[0]
        jobs, rejected = normalize_batch(
            [replace(raw, description="Build systems. No sponsorship needed.")]
        )
        assert not rejected and jobs[0].work_auth_flag is False and jobs[0].term != "unknown"
        await repo.apply_source_batch(
            SourceBatch(source=ADAPTER.name, run_started_at=NOW, jobs=jobs)
        )
        for attempt in range(3):
            assert await repo.detail_queue(ADAPTER.name, 20) == [_id()]
            result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
            assert result.authoritative and result.detail_fetches == (
                DetailFetch(id=_id(), failed=True),
            )
            await repo.apply_source_batch(
                SourceBatch(
                    source=ADAPTER.name,
                    run_started_at=NOW + timedelta(hours=attempt),
                    detail_fetches=result.detail_fetches,
                )
            )
        assert await repo.detail_queue(ADAPTER.name, 20) == []
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == jobs[0].description
    assert detail.call_count == 3


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"items": []},
        {"items": [{}, {}]},
        {"items": [{}]},
        {"items": [{"Id": "91731", "ExternalDescriptionStr": 42}]},
        _detail("other"),
        _detail(body=" "),
        _detail(body="<b>Build</b> systems"),
        {
            "items": [
                {
                    "Id": "91731",
                    "ExternalDescriptionStr": "build &amp; test",
                    "ShortDescriptionStr": "<p>build &amp; test</p>",
                }
            ]
        },
    ],
)
@respx.mock
async def test_invalid_or_summary_only_detail_never_resolves(payload: object) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    respx.get(DETAIL_URL).respond(200, json=payload)
    async with _client() as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert result.authoritative
    assert result.jobs[0].description == ""
    assert result.detail_fetches == (DetailFetch(id=_id(), failed=True),)


@respx.mock
async def test_budget_deduplicates_and_ignores_foreign_or_absent_ids() -> None:
    rows = [_posting(str(n)) for n in range(ADAPTER.detail_budget + 2)]
    rows.append(rows[0])
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(*rows))
    detail = respx.get(DETAIL_URL).mock(
        side_effect=lambda req: httpx.Response(
            200, json=_detail(req.url.params["finder"].split("Id=")[1].split(",")[0])
        )
    )
    ids = [_id(str(n)) for n in range(ADAPTER.detail_budget + 2)]
    async with _client() as client:
        result = await ADAPTER.fetch(
            COMPANY, client, NOW, details=["oracle_cloud:foreign:1", _id("999"), *ids, *ids]
        )
    assert detail.call_count == ADAPTER.detail_budget
    assert len(result.detail_fetches) == ADAPTER.detail_budget
    assert result.jobs[0].description == result.jobs[-1].description
    assert ADAPTER.max_requests_per_company == MAX_PAGES + ADAPTER.detail_budget


@respx.mock
async def test_refused_request_does_not_spend_attempt_or_lose_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    async with _client() as client:
        original = client.get_json

        async def refused(url: str, **kwargs: Any) -> Any:
            if url == DETAIL_URL:
                raise RequestRefusedError("request cap")
            return await original(url, **kwargs)

        monkeypatch.setattr(client, "get_json", refused)
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert result.authoritative and len(result.jobs) == 1
    assert not result.detail_fetches


@respx.mock
async def test_hydration_does_not_make_partial_listing_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stage.sources.oracle_cloud.MAX_PAGES", 1)
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting(), total=2))
    respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert not result.authoritative and result.jobs[0].description
    assert result.detail_fetches == (DetailFetch(id=_id(), resolved=True),)


def test_public_sections_are_complete_without_arbitrary_length_threshold() -> None:
    assert _detail_body(_detail(body="A short complete posting."), "91731", "summary")
    body = _detail_body(
        _detail(body="<p>first</p>" + "<p>required qualification</p>" * 2000 + "<p>last</p>"),
        "91731",
        "summary",
    )
    assert body.count("required qualification") == 2000 and "last" in body


@respx.mock
async def test_real_http_cap_defers_detail_without_spending_attempt(db_path: Path) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    posture = replace(profile(ADAPTER.rate_profile), min_interval_s=0, max_requests_per_run=1)
    async with HttpClient(
        allowed_hosts=ADAPTER.hosts_for((COMPANY,)), posture=posture, jitter=False
    ) as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert client.request_count == 1
    assert result.authoritative and result.detail_fetches == () and not detail.called
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW,
                jobs=result.jobs,
                detail_fetches=result.detail_fetches,
            )
        )
        assert await repo.detail_queue(ADAPTER.name, 20) == [_id()]


@respx.mock
async def test_queued_details_bypass_listing_and_detail_validators() -> None:
    listing = respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client:
        listing_url = str(
            httpx.URL(ADAPTER.url_for(COMPANY), params=ADAPTER.params_for(COMPANY, 0))
        )
        detail_url = str(
            httpx.URL(
                DETAIL_URL,
                params={"onlyData": "true", "finder": "ById;Id=91731,siteNumber=CX_1001"},
            )
        )
        for url in (listing_url, detail_url):
            client.cache.record(url, httpx.Headers({"etag": '"old"'}), NOW)
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert result.jobs[0].description
        assert "if-none-match" not in listing.calls[-1].request.headers
        assert "if-none-match" not in detail.calls[-1].request.headers
