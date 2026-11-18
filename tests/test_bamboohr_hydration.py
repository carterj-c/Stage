import json
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from bs4 import BeautifulSoup

from stage.domain import Company, DetailFetch, JobFilters, Platform, RoleCategory, job_id
from stage.http import (
    BreakerOpenError,
    BucketBlockedError,
    HostBudgetExceededError,
    HttpClient,
    JsonResponse,
    RatePosture,
    RequestRefusedError,
)
from stage.services import sync as sync_module
from stage.services.screening import normalize_batch
from stage.sources.bamboohr import BambooHrAdapter
from stage.sources.platforms import SlugRejectedError
from stage.storage import SourceBatch, open_repository
from stage.storage.sqlite_repo import SqliteRepository

FIXTURES = Path(__file__).parent / "fixtures"
BOARD = json.loads((FIXTURES / "bamboohr_activategames_board.json").read_text())
DETAIL = json.loads((FIXTURES / "bamboohr_activategames_detail.json").read_text())
COMPANY = Company(name="Activate Games", platform=Platform.BAMBOOHR, slug="activategames")
NOW = datetime(2026, 11, 18, tzinfo=UTC)
IDENT = job_id("bamboohr", COMPANY.slug, "884")
BASE = "https://activategames.bamboohr.com/careers"
UNPACED = RatePosture(min_interval_s=0, max_requests_per_run=100, concurrency=1)


def _at(when: datetime) -> Callable[[], datetime]:
    return lambda: when


def _client() -> HttpClient:
    return HttpClient(
        allowed_hosts=BambooHrAdapter().hosts_for([COMPANY]),
        posture=UNPACED,
        jitter=False,
    )


async def _fetch(details: Sequence[str] = (IDENT,), board: Any = BOARD) -> Any:
    respx.get(f"{BASE}/list").respond(200, json=board)
    async with _client() as client:
        return await BambooHrAdapter().fetch(COMPANY, client, NOW, details=details)


def _detail(body: str, title: str = "Quality Assurance Analyst") -> dict[str, Any]:
    return {"result": {"jobOpening": {"jobOpeningName": title, "description": body}}}


@respx.mock
async def test_full_live_body_preserves_every_section_and_tail() -> None:
    route = respx.get(f"{BASE}/884/detail").respond(200, json=DETAIL)
    result = await _fetch()
    job = next(j for j in result.jobs if j.id == IDENT)
    assert route.call_count == 1
    assert result.authoritative and not result.degraded
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)
    for text in (
        "The Duties and Responsibilities",
        "The Qualifications",
        "4-5+ years",
        "The Perks",
        "only those candidates selected for interviews will be contacted.",
    ):
        assert text in job.description
    raw = DETAIL["result"]["jobOpening"]["description"]
    assert len(raw) == 5892
    assert "".join(job.description.split()) == "".join(
        BeautifulSoup(raw, "html.parser").get_text().split()
    )
    assert len(job.description) == 3167
    assert all(not j.description for j in result.jobs if j.id != IDENT)


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"result": None},
        {"result": []},
        {"result": {}},
        {"result": {"jobOpening": []}},
        _detail(""),
        _detail("   "),
        _detail("<p><br>&nbsp;</p>"),
        _detail("<script>tracking()</script><style>p{color:red}</style>"),
        _detail("Body", title="Wrong job"),
        _detail("Body") | {"result": {"jobOpening": {"description": "Body"}}},
        _detail("Body")
        | {
            "result": {
                "jobOpening": {
                    "jobOpeningName": "Quality Assurance Analyst",
                    "description": {"text": "Body"},
                }
            }
        },
        {
            "result": {
                "jobOpening": DETAIL["result"]["jobOpening"]
                | {
                    "jobOpeningShareUrl": "https://evil.example/careers/884",
                }
            }
        },
    ],
)
async def test_malformed_missing_and_empty_detail_never_resolves(payload: Any) -> None:
    respx.get(f"{BASE}/884/detail").respond(200, content=json.dumps(payload))
    result = await _fetch()
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)
    assert not any(j.description for j in result.jobs)


@respx.mock
@pytest.mark.parametrize("status", [403, 404, 410, 304])
async def test_http_failure_keeps_listing_authority(status: int) -> None:
    respx.get(f"{BASE}/884/detail").respond(status)
    result = await _fetch()
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)


@respx.mock
@pytest.mark.parametrize("failure", ["invalid-json", "transport"])
async def test_unreadable_or_transport_failure(failure: str) -> None:
    route = respx.get(f"{BASE}/884/detail")
    if failure == "transport":
        route.mock(side_effect=httpx.ReadError("lost connection"))
    else:
        route.respond(200, text="{truncated")
    result = await _fetch()
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)


@respx.mock
async def test_only_board_scoped_queue_ids_bypass_stale_validators() -> None:
    listing = respx.get(f"{BASE}/list").respond(200, json=BOARD)
    detail = respx.get(f"{BASE}/884/detail").respond(200, json=DETAIL)
    async with _client() as client:
        for url in (f"{BASE}/list", f"{BASE}/884/detail"):
            client.cache.record(url, httpx.Headers({"etag": '"old"'}), NOW)
        await BambooHrAdapter().fetch(COMPANY, client, NOW, details=[IDENT])
        assert "if-none-match" not in listing.calls[-1].request.headers
        assert "if-none-match" not in detail.calls[-1].request.headers
        listing.respond(304)
        result = await BambooHrAdapter().fetch(COMPANY, client, NOW, details=["bamboohr:other:884"])
        assert result.not_modified
        assert listing.calls[-1].request.headers["if-none-match"] == '"old"'
        assert detail.call_count == 1


@respx.mock
async def test_budget_deduplicates_and_ignores_foreign_and_stale_ids() -> None:
    adapter = BambooHrAdapter()
    ids = [job_id("bamboohr", COMPANY.slug, str(n)) for n in range(1, 24)]
    board = {"result": [{"id": n, "jobOpeningName": "QA"} for n in range(1, 24)]}
    route = respx.get(url__regex=rf"{BASE}/[0-9]+/detail").respond(404)
    result = await _fetch(["bamboohr:other:884", *ids, *ids], board)
    assert route.call_count == adapter.detail_budget == 20
    assert len(result.jobs) == 23
    assert len(result.detail_fetches) == 20
    assert not (await _fetch(())).detail_fetches
    assert not (await _fetch([f"bamboohr:{COMPANY.slug}:99999"])).detail_fetches
    assert route.call_count == 20


@respx.mock
@pytest.mark.parametrize("ident", [True, 1.5, 0, -1, "../884", "884?x=1", "８８４", "884\n"])
async def test_invalid_listing_id_cannot_rewrite_fixed_urls(ident: Any) -> None:
    result = await _fetch(board={"result": [{"id": ident, "jobOpeningName": "QA"}]})
    assert not result.jobs and not result.authoritative
    assert not result.detail_fetches and len(respx.calls) == 1


@pytest.mark.parametrize("slug", ["evil.com/../x", "a@evil.com", "a?x=1", "../a"])
async def test_bad_board_slug_is_refused_before_network(slug: str) -> None:
    async with _client() as client:
        with pytest.raises(SlugRejectedError):
            await BambooHrAdapter().fetch(replace(COMPANY, slug=slug), client, NOW, details=[IDENT])


@pytest.mark.parametrize("error", [HostBudgetExceededError, BreakerOpenError, BucketBlockedError])
async def test_refused_details_do_not_increment_attempts(
    db_path: Path, error: type[RequestRefusedError]
) -> None:
    client = AsyncMock(spec=HttpClient)
    client.get_json.side_effect = [
        value
        for _ in range(4)
        for value in (JsonResponse(status=200, payload=BOARD, not_modified=False), error("refused"))
    ]
    async with open_repository(db_path) as repo:
        jobs = BambooHrAdapter().result(COMPANY, BOARD, NOW).jobs
        await repo.apply_source_batch(SourceBatch(source="bamboohr", run_started_at=NOW, jobs=jobs))
        for _ in range(4):
            result = await BambooHrAdapter().fetch(COMPANY, client, NOW, details=[IDENT])
            assert not result.detail_fetches
            assert IDENT in await repo.detail_queue("bamboohr", 100)
    with closing(SqliteRepository.connect(db_path)) as repo:
        assert repo._conn.execute("SELECT count(*) FROM detail_fetches").fetchone()[0] == 0


@respx.mock
async def test_failure_attempts_are_capped_and_success_removes_queue(db_path: Path) -> None:
    respx.get(f"{BASE}/884/detail").respond(200, json=_detail(""))
    async with open_repository(db_path) as repo:
        jobs = BambooHrAdapter().result(COMPANY, BOARD, NOW).jobs
        await repo.apply_source_batch(SourceBatch(source="bamboohr", run_started_at=NOW, jobs=jobs))
        for attempt in range(3):
            assert IDENT in await repo.detail_queue("bamboohr", 100)
            result = await _fetch()
            await repo.apply_source_batch(
                SourceBatch(
                    source="bamboohr",
                    run_started_at=NOW + timedelta(hours=attempt),
                    detail_fetches=result.detail_fetches,
                )
            )
        assert IDENT not in await repo.detail_queue("bamboohr", 100)
    with closing(SqliteRepository.connect(db_path)) as repo:
        row = repo._conn.execute(
            "SELECT attempts, resolved, failed FROM detail_fetches WHERE id = ?", (IDENT,)
        ).fetchone()
        assert tuple(row) == (3, 0, 1)


@respx.mock
async def test_multiple_boards_share_budget_rescreen_and_preserve_later_bodies(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BambooHrAdapter, "detail_budget", 2)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: UNPACED)
    companies = [replace(COMPANY, slug=slug, name=slug) for slug in ("acme", "beta")]
    title = "Summer 2027 Software Engineer Intern"
    board = {
        "result": [
            {"id": n, "jobOpeningName": title, "location": {"city": "Toronto", "country": "Canada"}}
            for n in (1, 2)
        ]
    }
    for company in companies:
        respx.get(f"https://{company.slug}.bamboohr.com/careers/list").respond(200, json=board)
    body = "Build software. Must be a US citizen. Requirements: Python. End of full posting."
    detail = respx.get(
        url__regex=r"https://(acme|beta)\.bamboohr\.com/careers/[12]/detail"
    ).respond(200, json=_detail(body, title))
    async with open_repository(db_path) as repo:
        raw = tuple(j for c in companies for j in BambooHrAdapter().result(c, board, NOW).jobs)
        kept, rejected = normalize_batch(raw)
        assert not rejected
        # Known term, role and authorization must not exclude missing bodies from the queue.
        kept = tuple(replace(j, work_auth_flag=False) for j in kept)
        assert all(j.role is RoleCategory.SWE and j.term == "summer-2027" for j in kept)
        await repo.apply_source_batch(SourceBatch(source="bamboohr", run_started_at=NOW, jobs=kept))
        assert len(await repo.detail_queue("bamboohr", 10)) == 4
        for day in range(3):
            when = NOW + timedelta(days=day + 1)
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    companies,
                    sources=["bamboohr"],
                    force_refresh=True,
                    now_fn=_at(when),
                    still_open=False,
                )
            ]
            jobs = await repo.list_jobs(JobFilters(limit=None, with_duplicates=True))
            assert len(jobs) == 4
            assert sum(bool(j.description) for j in jobs) == min(4, (day + 1) * 2)
            assert detail.call_count == min(4, (day + 1) * 2)
            for job in jobs:
                if job.description:
                    assert job.description == body and job.work_auth_flag is True
            if day == 0:
                assert {call.request.url.host for call in detail.calls} == {
                    "acme.bamboohr.com",
                    "beta.bamboohr.com",
                }
        assert await repo.detail_queue("bamboohr", 10) == []


@respx.mock
async def test_failed_detail_retries_then_succeeds_without_double_counting(db_path: Path) -> None:
    route = respx.get(f"{BASE}/884/detail").respond(200, json=_detail(""))
    async with open_repository(db_path) as repo:
        first = await _fetch(details=())
        job = next(j for j in first.jobs if j.id == IDENT)
        await repo.apply_source_batch(
            SourceBatch(
                source="bamboohr",
                run_started_at=NOW,
                jobs=(job,),
            )
        )
        for attempt in range(2):
            assert await repo.detail_queue("bamboohr", 10) == [IDENT]
            if attempt:
                route.respond(200, json=DETAIL)
            result = await _fetch(await repo.detail_queue("bamboohr", 10))
            await repo.apply_source_batch(
                SourceBatch(
                    source="bamboohr",
                    run_started_at=NOW + timedelta(hours=attempt),
                    jobs=tuple(j for j in result.jobs if j.id == IDENT),
                    detail_fetches=result.detail_fetches,
                )
            )
        assert IDENT not in await repo.detail_queue("bamboohr", 100)
        assert route.call_count == 2
    with closing(SqliteRepository.connect(db_path)) as repo:
        row = repo._conn.execute(
            "SELECT attempts, resolved, failed FROM detail_fetches WHERE id = ?", (IDENT,)
        ).fetchone()
        assert tuple(row) == (2, 1, 0)


@respx.mock
async def test_hydration_does_not_make_partial_listing_authoritative() -> None:
    respx.get(f"{BASE}/884/detail").respond(200, json=DETAIL)
    result = await _fetch(board={"result": [BOARD["result"][0], {"id": "broken"}]})
    assert not result.authoritative and result.degraded
    assert result.jobs[0].description
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)
