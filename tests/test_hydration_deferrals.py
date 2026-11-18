from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
import respx

from stage.domain import Company, Job, JobFilters, Platform, RoleCategory
from stage.http import (
    BreakerOpenError,
    BucketBlockedError,
    HostBudgetExceededError,
    HttpClient,
    RatePosture,
    RequestRefusedError,
)
from stage.services import sync as sync_module
from stage.sources.base import FetchResult
from stage.sources.jobvite import JobviteAdapter
from stage.sources.smartrecruiters import SmartRecruitersAdapter, SmartRecruitersPosting
from stage.sources.workday import WorkdayPosting, _attach_descriptions
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


def _at(when: datetime) -> Callable[[], datetime]:
    return lambda: when


def _job(source: str, board: str, ident: str, seen: datetime = NOW) -> Job:
    return Job(
        id=f"{source}:{board}:{ident}",
        source=source,
        company="Acme",
        title_raw="Summer 2027 Software Engineer Intern",
        title_normalized="summer 2027 software engineer intern",
        apply_url_raw=(f"https://jobs.jobvite.com/acme/job/{ident}" if source == "jobvite" else ""),
        description="",
        term="summer-2027",
        role=RoleCategory.SWE,
        first_seen=seen,
        last_seen=seen,
    )


@pytest.mark.parametrize("source", ["workday", "smartrecruiters", "jobvite"])
@pytest.mark.parametrize("error", [HostBudgetExceededError, BreakerOpenError, BucketBlockedError])
async def test_refused_details_do_not_spend_retry_attempts(
    db_path: Path, source: str, error: type[RequestRefusedError]
) -> None:
    board = "acme-careers" if source == "workday" else "acme"
    job = _job(source, board, "p1")
    company = Company(
        name="Acme",
        slug="acme",
        platform=Platform(source),
        workday_tenant="acme" if source == "workday" else None,
        workday_site="careers" if source == "workday" else None,
        workday_dc="wd3" if source == "workday" else None,
    )
    client = AsyncMock(spec=HttpClient)
    client.get_json.side_effect = error("deferred")
    client.get_text.side_effect = error("deferred")
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source=source, run_started_at=NOW, jobs=(job,))
        )
        for index in range(4):
            if source == "workday":
                workday_posting = WorkdayPosting(
                    title=job.title_raw, externalPath="/job/montreal/x_p1"
                )
                workday_merged, outcomes = await _attach_descriptions(
                    company, cast(HttpClient, client), [(workday_posting, job)], {job.id}
                )
                assert workday_merged == [(workday_posting, job)]
            elif source == "smartrecruiters":
                smart_posting = SmartRecruitersPosting(id="p1", name=job.title_raw)
                smart_merged, outcomes = await SmartRecruitersAdapter()._attach_descriptions(
                    company, cast(HttpClient, client), [(smart_posting, job)], {job.id}
                )
                assert smart_merged == [(smart_posting, job)]
            else:
                result = await JobviteAdapter()._attach_descriptions(
                    company, cast(HttpClient, client), FetchResult(jobs=(job,)), {job.id}
                )
                assert result.jobs == (job,)
                assert not result.detail_fetches
                outcomes = []
            assert not outcomes
            await repository.apply_source_batch(
                SourceBatch(
                    source=source,
                    run_started_at=NOW + timedelta(minutes=index),
                    detail_fetches=tuple(outcomes),
                )
            )
            assert await repository.detail_queue(source, 1) == [job.id]


@respx.mock
async def test_absent_listing_ids_rotate_across_syncs_until_present_job_is_hydrated(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SmartRecruitersAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    company = Company(name="Acme", slug="acme", platform=Platform.SMARTRECRUITERS)
    listing = "https://api.smartrecruiters.com/v1/companies/acme/postings"
    respx.get(listing).respond(
        200,
        json={
            "totalFound": 2,
            "content": [
                {
                    "id": "p3",
                    "name": "Summer 2027 Software Engineer Intern",
                    "location": {"city": "Montreal"},
                },
                {"id": "broken"},
            ],
        },
    )
    detail = respx.get(f"{listing}/p3").respond(
        200, json={"jobAd": {"sections": {"body": {"text": "Full job description"}}}}
    )
    jobs = (
        _job("smartrecruiters", "acme", "p1", NOW + timedelta(minutes=2)),
        _job("smartrecruiters", "acme", "p2", NOW + timedelta(minutes=1)),
        _job("smartrecruiters", "acme", "p3", NOW),
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="smartrecruiters",
                run_started_at=NOW,
                jobs=jobs,
            )
        )
        for day in range(3):
            when = NOW + timedelta(days=day + 1)
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [company],
                    sources=["smartrecruiters"],
                    force_refresh=True,
                    now_fn=_at(when),
                    still_open=False,
                )
            ]
            if day < 2:
                assert detail.call_count == 0
        assert detail.call_count == 1
        stored = await repository.list_jobs(JobFilters(status=None))
        assert any(
            job.id == jobs[2].id and job.description == "Full job description" for job in stored
        )
        assert set(await repository.detail_queue("smartrecruiters", 10)) == {jobs[0].id, jobs[1].id}
    async with open_repository(db_path) as reopened:
        assert await reopened.detail_queue("smartrecruiters", 1) == [jobs[0].id]


async def test_cursor_cycles_from_quarantine_to_retained_without_losing_board_scope(
    db_path: Path,
) -> None:
    from dataclasses import replace

    from stage.services.screening import normalize_batch

    waiting = replace(
        _job("smartrecruiters", "acme", "z"),
        title_raw="Systems Engineer Intern",
        title_normalized="systems engineer intern",
    )
    kept, rejected = normalize_batch([waiting])
    assert not kept and len(rejected) == 1
    retained = _job("smartrecruiters", "acme", "a")
    other = _job("smartrecruiters", "other", "a", NOW + timedelta(days=1))
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="smartrecruiters",
                run_started_at=NOW,
                jobs=(retained, other),
                quarantined=tuple(rejected),
            )
        )
        scope = ["smartrecruiters:acme"]
        first = await repository.detail_queue("smartrecruiters", 1, scope)
        assert first == [waiting.id]
        await repository.apply_source_batch(
            SourceBatch(
                source="smartrecruiters",
                run_started_at=NOW,
                detail_queue_cursor=first[-1],
            )
        )
        second = await repository.detail_queue("smartrecruiters", 1, scope)
        assert second == [retained.id]
        await repository.apply_source_batch(
            SourceBatch(
                source="smartrecruiters",
                run_started_at=NOW,
                detail_queue_cursor=second[-1],
            )
        )
        assert await repository.detail_queue("smartrecruiters", 1, scope) == [waiting.id]


def test_job_writes_preserve_migrated_authorization_state(db_path: Path) -> None:
    from dataclasses import replace

    from stage.services.screening import normalize_batch
    from stage.storage.sqlite_repo import SqliteRepository

    (initial,), rejected = normalize_batch([_job("workday", "acme-careers", "p1")])
    assert not rejected and initial.work_auth_flag is None
    repo = SqliteRepository.connect(db_path)
    try:
        repo.apply_source_batch(SourceBatch(source="workday", run_started_at=NOW, jobs=(initial,)))
        row = repo._conn.execute(
            "SELECT work_auth_known FROM jobs WHERE id = ?", (initial.id,)
        ).fetchone()
        assert row[0] == 0
        (described,), rejected = normalize_batch(
            [replace(initial, description="Full posting text")]
        )
        assert not rejected and described.work_auth_flag is False
        repo.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, jobs=(described,))
        )
        row = repo._conn.execute(
            "SELECT work_auth_known FROM jobs WHERE id = ?", (initial.id,)
        ).fetchone()
        assert row[0] == 1
        flagged = replace(initial, work_auth_flag=True)
        repo.apply_source_batch(SourceBatch(source="workday", run_started_at=NOW, jobs=(flagged,)))
        row = repo._conn.execute(
            "SELECT work_auth_known FROM jobs WHERE id = ?", (initial.id,)
        ).fetchone()
        assert row[0] == 1
    finally:
        repo.close()


@respx.mock
async def test_alternating_scheduled_boards_each_reach_present_jobs_with_one_slot(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    monkeypatch.setattr(SmartRecruitersAdapter, "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    companies = {
        board: Company(name=f"Company {board}", slug=board, platform=Platform.SMARTRECRUITERS)
        for board in ("a", "b")
    }
    details = {}
    jobs = []
    for board, company in companies.items():
        listing = f"https://api.smartrecruiters.com/v1/companies/{board}/postings"
        respx.get(listing).respond(
            200,
            json={
                "totalFound": 2,
                "content": [
                    {
                        "id": "p2",
                        "name": "Summer 2027 Software Engineer Intern",
                        "location": {"city": "Montreal"},
                    },
                    {"id": "broken"},
                ],
            },
        )
        details[board] = respx.get(f"{listing}/p2").respond(
            200, json={"jobAd": {"sections": {"body": {"text": f"Full {board} description"}}}}
        )
        jobs.extend(
            [
                replace(
                    _job("smartrecruiters", board, "p1", NOW + timedelta(minutes=1)),
                    company=company.name,
                ),
                replace(_job("smartrecruiters", board, "p2"), company=company.name),
            ]
        )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="smartrecruiters",
                run_started_at=NOW,
                jobs=tuple(jobs),
            )
        )
    for index, board in enumerate(("a", "b", "a", "b")):
        async with open_repository(db_path) as repository:
            before = sum(route.call_count for route in details.values())
            when = NOW + timedelta(days=index + 1)
            _ = [
                event
                async for event in sync_module.sync(
                    repository,
                    [companies[board]],
                    sources=["smartrecruiters"],
                    force_refresh=True,
                    close_orphans=False,
                    now_fn=_at(when),
                    still_open=False,
                )
            ]
            assert sum(route.call_count for route in details.values()) - before <= 1
    assert details["a"].call_count == details["b"].call_count == 1
    async with open_repository(db_path) as repository:
        stored = {job.id: job for job in await repository.list_jobs(JobFilters(status=None))}
        for board in ("a", "b"):
            assert stored[f"smartrecruiters:{board}:p2"].description == f"Full {board} description"
            assert f"smartrecruiters:{board}:p1" in await repository.detail_queue(
                "smartrecruiters", 1, [f"smartrecruiters:{board}"]
            )


@pytest.mark.parametrize("limit", [1, 4])
async def test_simultaneous_boards_share_budget_and_save_every_board_cursor(
    db_path: Path, limit: int
) -> None:
    source = "smartrecruiters"
    boards = [f"{source}:{board}" for board in ("a", "b", "c")]
    jobs = tuple(
        _job(source, board, posting) for board in ("a", "b", "c") for posting in ("p1", "p2")
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source=source, run_started_at=NOW, jobs=jobs)
        )
        selected: list[str] = []
        for _ in range((6 + limit - 1) // limit):
            picked = await repository.detail_queue(source, limit, boards)
            assert len(picked) == limit
            selected.extend(picked)
            await repository.apply_source_batch(
                SourceBatch(
                    source=source,
                    run_started_at=NOW,
                    detail_queue_selected=tuple(picked),
                )
            )
        assert len(set(selected[:6])) == 6
        assert selected[:3] == [f"{source}:{board}:p1" for board in ("a", "b", "c")]
        assert selected[3:6] == [f"{source}:{board}:p2" for board in ("a", "b", "c")]
    async with open_repository(db_path) as repository:
        next_queue = await repository.detail_queue(source, 1, [boards[2]])
        assert next_queue == [f"{source}:c:p1"]
