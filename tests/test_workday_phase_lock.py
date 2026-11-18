import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from stage.domain import Company, JobFilters, JobStatus, Platform, job_id
from stage.http import RatePosture
from stage.services import sync as sync_module
from stage.sources.workday import WorkdayAdapter, WorkdayPosting, _to_job
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


def _at(when: datetime) -> Callable[[], datetime]:
    return lambda: when


def _company(tenant: str = "acme") -> Company:
    return Company(
        name=f"Company {tenant}",
        slug=tenant,
        platform=Platform.WORKDAY,
        workday_tenant=tenant,
        workday_site="careers",
        workday_dc="wd3",
    )


def _posting(req: str, title: str) -> dict[str, object]:
    return {
        "title": title,
        "externalPath": f"/job/Montreal/x_{req}",
        "locationsText": "Montreal, Quebec, Canada",
        "bulletFields": [req],
    }


@respx.mock
async def test_two_page_crawl_hydrates_current_page_despite_opposite_queue_phase(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = WorkdayAdapter()
    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(
        concurrency=2, min_interval_s=0, max_requests_per_run=42, requests_per_board=0
    )
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    company = _company()
    title_a = "Summer 2027 Software Engineer Intern Backend"
    title_b = "Summer 2027 Software Engineer Intern Frontend"
    rows = [_posting("R10001", title_a), _posting("R10002", title_b)]
    pages = [
        [
            rows[index],
            *[_posting(f"M{index}{n:04d}", f"Marketing Manager {index} {n}") for n in range(19)],
        ]
        for index in range(2)
    ]
    offsets: list[int] = []
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"

    def listing(request: httpx.Request) -> httpx.Response:
        offset = int(json.loads(request.content)["offset"])
        offsets.append(offset)
        return httpx.Response(
            200, json={"total": 40, "facets": [], "jobPostings": pages[offset // 20]}
        )

    respx.post(f"{base}/jobs").mock(side_effect=listing)
    detail_a = respx.get(f"{base}/job/Montreal/x_R10001").respond(
        200, json={"jobPostingInfo": {"jobDescription": "Backend full description"}}
    )
    detail_b = respx.get(f"{base}/job/Montreal/x_R10002").respond(
        200, json={"jobPostingInfo": {"jobDescription": "Frontend full description"}}
    )
    seed_a = _to_job(company, WorkdayPosting.model_validate(rows[0]), NOW)
    seed_b = _to_job(company, WorkdayPosting.model_validate(rows[1]), NOW + timedelta(seconds=1))
    assert seed_a.id == job_id("workday", "acme-careers", "R10001")
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                jobs=(seed_a, seed_b),
            )
        )
        assert await repository.detail_queue("workday", 1, [adapter.board_key(company)]) == [
            seed_b.id
        ]
        for index in range(2):
            when = NOW + timedelta(minutes=index + 1)
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [company],
                    sources=["workday"],
                    force_refresh=True,
                    now_fn=_at(when),
                    still_open=False,
                )
            ]
            assert len(detail_a.calls) + len(detail_b.calls) == index + 1
        stored = {job.id: job for job in await repository.list_jobs(JobFilters(status=None))}
        assert stored[seed_a.id].description == "Backend full description"
        assert stored[seed_b.id].description == "Frontend full description"
        assert stored[seed_a.id].status is JobStatus.OPEN
        assert stored[seed_b.id].status is JobStatus.OPEN
        assert await repository.detail_queue("workday", 10, ids=[seed_a.id, seed_b.id]) == []
    assert offsets == [0, 20]
    assert detail_a.call_count == detail_b.call_count == 1


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/job/",
        "https://evil.test/job/x",
        "//evil.test/job/x",
        "/job/../../other",
        "/job/%2e%2e/other",
        "/job/%252e%252e/other",
        "/job/x?target=other",
        "/job/x#other",
        "/job/x\\..\\other",
        "/job/x\nother",
        "/other/job/x",
    ],
)
async def test_workday_rejects_unsafe_detail_paths_without_spending_attempts(path: str) -> None:
    from typing import cast
    from unittest.mock import AsyncMock

    from stage.http import HttpClient
    from stage.sources.workday import _attach_descriptions

    company = _company()
    posting = WorkdayPosting(title="Software Intern", externalPath=path, bulletFields=["R10001"])
    job = _to_job(company, posting, NOW)
    client = AsyncMock(spec=HttpClient)
    merged, outcomes = await _attach_descriptions(
        company, cast(HttpClient, client), [(posting, job)], {job.id}
    )
    assert merged == [(posting, job)]
    assert outcomes == []
    client.get_json.assert_not_awaited()


@respx.mock
async def test_present_board_uses_shared_slot_when_newer_queue_item_is_absent_elsewhere(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(
        concurrency=2, min_interval_s=0, max_requests_per_run=43, requests_per_board=0
    )
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    a, b = _company("a"), _company("b")
    title = "Summer 2027 Software Engineer Intern"
    row_a = _posting("R10001", title + " Backend")
    row_b = _posting("R10002", title + " Frontend")
    job_a = _to_job(a, WorkdayPosting.model_validate(row_a), NOW + timedelta(seconds=1))
    job_b = _to_job(b, WorkdayPosting.model_validate(row_b), NOW)
    base_a = "https://a.wd3.myworkdayjobs.com/wday/cxs/a/careers"
    base_b = "https://b.wd3.myworkdayjobs.com/wday/cxs/b/careers"
    listing_a = respx.post(f"{base_a}/jobs").respond(
        200,
        json={
            "total": 2,
            "facets": [],
            "jobPostings": [
                _posting("R90001", "Marketing Manager"),
                {"externalPath": "/job/x_R00000"},
            ],
        },
    )
    listing_b = respx.post(f"{base_b}/jobs").respond(
        200, json={"total": 1, "facets": [], "jobPostings": [row_b]}
    )
    details = respx.get(f"{base_b}/job/Montreal/x_R10002").respond(
        200, json={"jobPostingInfo": {"jobDescription": "Full frontend description"}}
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                jobs=(job_a, job_b),
            )
        )
        assert await repository.detail_queue("workday", 1) == [job_a.id]
        _ = [
            event
            async for event in sync_module.sync(
                repository,
                [a, b],
                sources=["workday"],
                force_refresh=True,
                now_fn=lambda: NOW + timedelta(minutes=1),
                still_open=False,
            )
        ]
        assert listing_a.call_count == listing_b.call_count == 1
        assert details.call_count == 1
        stored = {job.id: job for job in await repository.list_jobs(JobFilters(status=None))}
        assert stored[job_b.id].description == "Full frontend description"
        assert not stored[job_a.id].description
        assert await repository.detail_queue("workday", 10, ids=[job_a.id, job_b.id]) == [job_a.id]


@respx.mock
async def test_failed_listing_does_not_block_other_board_selection(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(
        concurrency=2, min_interval_s=0, max_requests_per_run=43, requests_per_board=0
    )
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    a, b = _company("a"), _company("b")
    row = _posting("R10002", "Summer 2027 Software Engineer Intern")
    job = _to_job(b, WorkdayPosting.model_validate(row), NOW)
    base_a = "https://a.wd3.myworkdayjobs.com/wday/cxs/a/careers"
    base_b = "https://b.wd3.myworkdayjobs.com/wday/cxs/b/careers"
    respx.post(f"{base_a}/jobs").respond(404)
    respx.post(f"{base_b}/jobs").respond(200, json={"total": 1, "facets": [], "jobPostings": [row]})
    details = respx.get(f"{base_b}/job/Montreal/x_R10002").respond(
        200, json={"jobPostingInfo": {"jobDescription": "Full description"}}
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=(job,))
        )

        async def run() -> None:
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [a, b],
                    sources=["workday"],
                    force_refresh=True,
                    now_fn=lambda: NOW + timedelta(minutes=1),
                    still_open=False,
                )
            ]

        await asyncio.wait_for(run(), timeout=5)
        assert details.call_count == 1
        stored = await repository.list_jobs(JobFilters(status=None))
        assert any(item.id == job.id and item.description == "Full description" for item in stored)


@respx.mock
async def test_page_selected_quarantine_row_is_screened_and_promoted(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.services.screening import normalize_batch

    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    company = _company()
    row = _posting("R10001", "Systems Engineer Intern")
    waiting = _to_job(company, WorkdayPosting.model_validate(row), NOW)
    kept, rejected = normalize_batch([waiting])
    assert not kept and len(rejected) == 1
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"
    respx.post(f"{base}/jobs").respond(200, json={"total": 1, "facets": [], "jobPostings": [row]})
    detail = respx.get(f"{base}/job/Montreal/x_R10001").respond(
        200,
        json={
            "jobPostingInfo": {"jobDescription": "Open to Computer Science majors. Build software."}
        },
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                quarantined=tuple(rejected),
            )
        )
        assert await repository.detail_queue("workday", 1) == [waiting.id]
        _ = [
            event
            async for event in sync_module.sync(
                repository,
                [company],
                sources=["workday"],
                force_refresh=True,
                now_fn=lambda: NOW + timedelta(minutes=1),
                still_open=False,
            )
        ]
        assert detail.call_count == 1
        stored = await repository.list_jobs(JobFilters(status=None))
        assert any(job.id == waiting.id and "Computer Science" in job.description for job in stored)
        assert await repository.detail_queue("workday", 1) == []


@respx.mock
async def test_page_selected_detail_failure_remains_retryable_on_next_crawl(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    company = _company()
    row = _posting("R10001", "Summer 2027 Software Engineer Intern")
    job = _to_job(company, WorkdayPosting.model_validate(row), NOW)
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"
    respx.post(f"{base}/jobs").respond(200, json={"total": 1, "facets": [], "jobPostings": [row]})
    detail = respx.get(f"{base}/job/Montreal/x_R10001").mock(
        side_effect=[
            httpx.Response(404),
            httpx.Response(200, json={"jobPostingInfo": {"jobDescription": "Full description"}}),
        ]
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=(job,))
        )
        for index in range(2):
            when = NOW + timedelta(minutes=index + 1)
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [company],
                    sources=["workday"],
                    force_refresh=True,
                    now_fn=_at(when),
                    still_open=False,
                )
            ]
            if index == 0:
                assert await repository.detail_queue("workday", 1) == [job.id]
                with sqlite3.connect(db_path) as conn:
                    assert conn.execute(
                        "SELECT attempts, failed FROM detail_fetches WHERE id = ?", (job.id,)
                    ).fetchone() == (1, 1)
        assert detail.call_count == 2
        stored = await repository.list_jobs(JobFilters(status=None))
        assert any(item.id == job.id and item.description == "Full description" for item in stored)
        assert await repository.detail_queue("workday", 1) == []


async def test_shared_selector_never_assigns_duplicate_id_to_two_concurrent_reads(
    db_path: Path,
) -> None:
    import asyncio

    company = _company()
    row = _posting("R10001", "Summer 2027 Software Engineer Intern")
    job = _to_job(company, WorkdayPosting.model_validate(row), NOW)
    board = WorkdayAdapter().board_key(company)
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=(job,))
        )
        selector = sync_module.WorkdayDetails(repository, 1, {0: board, 1: board}, {0, 1})
        first, second = await asyncio.gather(
            selector.select(0, [job.id, job.id]),
            selector.select(1, [job.id]),
        )
        assert first == [job.id]
        assert second == []
        assert selector.selected == [job.id]


async def test_empty_page_filter_and_empty_selection_preserve_board_cursor(db_path: Path) -> None:
    import sqlite3

    company = _company()
    row = _posting("R10001", "Summer 2027 Software Engineer Intern")
    job = _to_job(company, WorkdayPosting.model_validate(row), NOW)
    board = WorkdayAdapter().board_key(company)
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=(job,))
        )
        assert await repository.detail_queue("workday", 1, boards=[board], ids=[]) == []
        assert (
            await repository.detail_queue("workday", 1, boards=[board], ids=["workday:other:r1"])
            == []
        )
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                detail_queue_cursor=job.id,
                detail_queue_selected=(),
            )
        )
        assert await repository.detail_queue("workday", 1, boards=[board], ids=[job.id]) == [job.id]
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT cursor FROM detail_queue_cursors WHERE source = ?", (board,)
            ).fetchone()
            is None
        )


@respx.mock
async def test_two_present_boards_share_one_detail_slot_across_runs(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=2, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    companies = [_company("a"), _company("b")]
    seeds = []
    details = []
    for index, company in enumerate(companies):
        req = f"R1000{index + 1}"
        row = _posting(req, f"Summer 2027 Software Engineer Intern {index}")
        seeds.append(_to_job(company, WorkdayPosting.model_validate(row), NOW))
        base = f"https://{company.workday_tenant}.wd3.myworkdayjobs.com/wday/cxs/{company.workday_tenant}/careers"
        respx.post(f"{base}/jobs").respond(
            200, json={"total": 1, "facets": [], "jobPostings": [row]}
        )
        details.append(
            respx.get(f"{base}/job/Montreal/x_{req}").respond(
                200, json={"jobPostingInfo": {"jobDescription": f"Full description {index}"}}
            )
        )

    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=tuple(seeds))
        )
        for index in range(2):
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    companies,
                    sources=["workday"],
                    force_refresh=True,
                    now_fn=_at(NOW + timedelta(minutes=index + 1)),
                    still_open=False,
                )
            ]
            assert sum(route.call_count for route in details) == index + 1
        stored = {job.id: job for job in await repository.list_jobs(JobFilters(status=None))}
        assert [stored[job.id].description for job in seeds] == [
            "Full description 0",
            "Full description 1",
        ]
        assert await repository.detail_queue("workday", 10) == []


@respx.mock
async def test_page_selected_budget_refusal_does_not_spend_a_retry(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from stage.http import HostBudgetExceededError, HttpClient, JsonResponse

    monkeypatch.setattr(WorkdayAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    monkeypatch.setattr(
        sync_module,
        "resolve",
        lambda *_, **__: RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100),
    )
    company = _company()
    row = _posting("R10001", "Summer 2027 Software Engineer Intern")
    job = _to_job(company, WorkdayPosting.model_validate(row), NOW)
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"
    respx.post(f"{base}/jobs").respond(200, json={"total": 1, "facets": [], "jobPostings": [row]})
    calls = 0

    async def get_detail(*_: object, **__: object) -> JsonResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HostBudgetExceededError("deferred")
        return JsonResponse(
            status=200,
            not_modified=False,
            payload={"jobPostingInfo": {"jobDescription": "Full description"}},
        )

    monkeypatch.setattr(HttpClient, "get_json", get_detail)
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                jobs=(job,),
            )
        )
        for index in range(2):
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [company],
                    sources=["workday"],
                    force_refresh=True,
                    now_fn=_at(NOW + timedelta(minutes=index + 1)),
                    still_open=False,
                )
            ]
            if index == 0:
                assert await repository.detail_queue("workday", 1) == [job.id]
                with sqlite3.connect(db_path) as conn:
                    assert (
                        conn.execute(
                            "SELECT attempts FROM detail_fetches WHERE id = ?", (job.id,)
                        ).fetchone()
                        is None
                    )
        stored = await repository.list_jobs(JobFilters(status=None))
        assert any(item.id == job.id and item.description == "Full description" for item in stored)
        assert calls == 2
