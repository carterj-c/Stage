from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import respx

from stage.domain import Company, CompanyFailed, Job, JobFilters, Platform, RoleCategory
from stage.http import RatePosture
from stage.services import sync as sync_module
from stage.sources.smartrecruiters import SmartRecruitersAdapter
from stage.sources.workday import WorkdayAdapter
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


def _job(ident: str, *, source: str, first_seen: datetime, term: str, role: RoleCategory) -> Job:
    return Job(
        id=ident,
        source=source,
        company="Acme",
        title_raw="Software Engineer Intern",
        title_normalized="software engineer intern",
        apply_url_raw="",
        description="",
        first_seen=first_seen,
        last_seen=NOW,
        term=term,
        role=role,
    )


@pytest.mark.parametrize("source", ["workday", "smartrecruiters"])
@respx.mock
async def test_seed_then_hydrate_classified_jobs_with_a_bounded_scheduled_queue(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    adapter = WorkdayAdapter() if source == "workday" else SmartRecruitersAdapter()
    monkeypatch.setattr(type(adapter), "detail_budget", 1)
    monkeypatch.setattr(sync_module, "get_feeds", dict)
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=100)
    monkeypatch.setattr(sync_module, "resolve", lambda *_, **__: posture)
    company = Company(
        name="Acme",
        slug="acme",
        platform=Platform(source),
        workday_tenant="acme" if source == "workday" else None,
        workday_site="careers" if source == "workday" else None,
        workday_dc="wd3" if source == "workday" else None,
    )
    title = "Summer 2027 Software Engineer Intern"
    body = "Build software with our engineering team. " * 12
    details = []
    if source == "workday":
        base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"
        listing = respx.post(f"{base}/jobs").respond(
            200,
            json={
                "total": 2,
                "facets": [],
                "jobPostings": [
                    {
                        "title": f"{title} {index}",
                        "externalPath": f"/job/Montreal/x_R{index}",
                        "locationsText": "Montreal, Quebec, Canada",
                        "bulletFields": [f"R{index}"],
                    }
                    for index in range(2)
                ],
            },
        )
        for index in range(2):
            details.append(
                respx.get(f"{base}/job/Montreal/x_R{index}").respond(
                    200, json={"jobPostingInfo": {"jobDescription": f"<p>{body}</p>"}}
                )
            )
    else:
        base = "https://api.smartrecruiters.com/v1/companies/acme/postings"
        listing = respx.get(base).respond(
            200,
            json={
                "totalFound": 2,
                "content": [
                    {
                        "id": f"P{index}",
                        "name": f"{title} {index}",
                        "location": {"city": "Montreal"},
                    }
                    for index in range(2)
                ],
            },
        )
        for index in range(2):
            details.append(
                respx.get(f"{base}/P{index}").respond(
                    200, json={"jobAd": {"sections": {"description": {"text": body}}}}
                )
            )

    async with open_repository(db_path) as repository:

        async def run(day: int) -> None:
            events = [
                event
                async for event in sync_module.sync(
                    repository,
                    [company],
                    sources=[source],
                    force_refresh=True,
                    now_fn=lambda: NOW + timedelta(days=day),
                    still_open=False,
                )
            ]

            assert not [event for event in events if isinstance(event, CompanyFailed)]

        await run(0)
        seeded = await repository.list_jobs(JobFilters(status=None))
        assert len(seeded) == 2
        assert all(not job.description for job in seeded)
        assert all(job.term != "unknown" and job.role != RoleCategory.UNKNOWN for job in seeded)
        assert not any(route.called for route in details)

        await repository.apply_source_batch(
            SourceBatch(
                source=source,
                run_started_at=NOW,
                jobs=tuple(
                    _job(
                        f"{source}:unscheduled:{index}",
                        source=source,
                        first_seen=NOW + timedelta(hours=1),
                        term="summer-2027",
                        role=RoleCategory.SWE,
                    )
                    for index in range(3)
                ),
            )
        )
        await run(1)
        hydrated = await repository.list_jobs(JobFilters(status=None))
        assert sum(bool(job.description) for job in hydrated) == 1
        assert sum(route.call_count for route in details) == 1
        await run(2)
        hydrated = await repository.list_jobs(JobFilters(status=None))
        assert sum(len(job.description) > 300 for job in hydrated) == 2
        assert sum(route.call_count for route in details) == 2
        await run(3)
        refreshed = await repository.list_jobs(JobFilters(status=None))
        assert sum(bool(job.description) for job in refreshed) == 2
        assert sum(route.call_count for route in details) == 2
        assert listing.call_count >= 4
        assert len(respx.calls) <= 4 * posture.max_requests_per_run


@pytest.mark.parametrize("source", ["workday", "smartrecruiters", "jobvite"])
async def test_scoped_queue_filters_before_limit_and_is_safe_when_empty(
    db_path: Path, source: str
) -> None:
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source=source,
                run_started_at=NOW,
                jobs=tuple(
                    _job(
                        f"{source}:{board}:1",
                        source=source,
                        first_seen=NOW + timedelta(days=offset),
                        term="summer-2027",
                        role=RoleCategory.SWE,
                    )
                    for offset, board in enumerate(["scheduled", "other", "scheduled-extra"])
                ),
            )
        )
        assert await repository.detail_queue(source, 1, [f"{source}:scheduled"]) == [
            f"{source}:scheduled:1"
        ]
        assert await repository.detail_queue(source, 1, []) == []
        assert await repository.detail_queue(source, 0) == []
        assert await repository.detail_queue(source, -1) == []
        assert await repository.detail_queue_size(source) == 3


@pytest.mark.parametrize("source", ["workday", "smartrecruiters"])
async def test_planner_scopes_queue_after_rotation_refresh_and_rest(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    import random

    from stage.domain import CompanyVisit

    adapter = WorkdayAdapter() if source == "workday" else SmartRecruitersAdapter()
    monkeypatch.setattr(type(adapter), "detail_budget", 1)
    monkeypatch.setattr(type(adapter), "rotation_slice", 3)
    companies = [
        Company(
            name=name,
            slug=name,
            platform=Platform(source),
            workday_tenant=name if source == "workday" else None,
            workday_site="careers" if source == "workday" else None,
            workday_dc="wd3" if source == "workday" else None,
        )
        for name in ["a-due", "b-fresh", "c-resting", "d-rotated"]
    ]
    boards = [adapter.board_key(company) for company in companies]
    posture = RatePosture(
        min_interval_s=0, max_requests_per_run=100, refresh_interval_h=24, busy_refresh_h=24
    )
    bucket = adapter.bucket_key or next(iter(adapter.hosts))
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source=source,
                run_started_at=NOW,
                jobs=tuple(
                    _job(
                        f"{board}:1",
                        source=source,
                        first_seen=NOW + timedelta(minutes=index),
                        term="summer-2027",
                        role=RoleCategory.SWE,
                    )
                    for index, board in enumerate(boards)
                ),
                visits=(CompanyVisit(board=boards[1], succeeded=True),),
            )
        )
        for _ in range(3):
            await repository.apply_source_batch(
                SourceBatch(
                    source=source,
                    run_started_at=NOW,
                    visits=(CompanyVisit(board=boards[2], succeeded=False, error="returned 403"),),
                )
            )
        plan = await sync_module._plan_boards(
            repository,
            adapter,
            companies,
            NOW + timedelta(minutes=5),
            random.Random(0),
            {},
            {bucket: posture},
            sync_module.Pacing(),
            False,
        )
        assert plan.ordered == [companies[0]]
        assert plan.refreshed == 1
        assert plan.resting == 1
        assert plan.rotation.deferred == (companies[3].registry_key,)
        assert list(plan.detail_queue) == [f"{boards[0]}:1"]


@pytest.mark.parametrize("spare", [0, 1, 12, 180])
@pytest.mark.parametrize("queued", [0, 1, 60])
def test_crowded_workday_allocation_reserves_details_without_losing_board_pages(
    spare: int, queued: int
) -> None:
    from stage.domain import WorkdayCrawl, WorkdayFacet

    adapter = WorkdayAdapter()
    companies = [
        Company(
            name=f"Board {index}",
            slug=f"b{index}",
            platform=Platform.WORKDAY,
            workday_tenant=f"b{index}",
            workday_site="careers",
            workday_dc="wd3",
        )
        for index in range(12)
    ]
    facets = {
        (company.workday_tenant or "", "careers"): WorkdayFacet(
            tenant=company.workday_tenant or "",
            site="careers",
            parameter="timeType",
            facet_ids=("i",),
        )
        for company in companies
    }
    crawls = {
        adapter.board_key(company): WorkdayCrawl(
            board=adapter.board_key(company),
            next_offset=100,
            total=2000,
            facet_parameter="timeType",
            facet_ids=("i",),
        )
        for company in companies
    }
    ceiling = adapter.retry_reserve + len(companies) + spare
    pages, details = adapter.crawl_budgets(
        companies, crawls, facets, ceiling, queued_details=queued
    )
    assert len(pages) == len(companies)
    assert min(pages.values()) >= 1
    assert sum(pages.values()) + details + adapter.retry_reserve <= ceiling
    assert details <= adapter.detail_budget
    if queued and spare:
        assert details == min(queued, adapter.detail_budget, (spare + 1) // 2)
        assert sum(pages.values()) > len(companies) or spare == 1
    else:
        assert details == 0


async def test_workday_planner_passes_scheduled_backlog_to_crowded_allocator(db_path: Path) -> None:
    import random

    adapter = WorkdayAdapter()
    companies = [
        Company(
            name=f"Board {index}",
            slug=f"b{index}",
            platform=Platform.WORKDAY,
            workday_tenant=f"b{index}",
            workday_site="careers",
            workday_dc="wd3",
        )
        for index in range(12)
    ]
    ceiling = adapter.retry_reserve + len(companies) + 12
    posture = RatePosture(min_interval_s=0, max_requests_per_run=ceiling)
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="workday",
                run_started_at=NOW,
                jobs=tuple(
                    _job(
                        f"{adapter.board_key(company)}:r1",
                        source="workday",
                        first_seen=NOW,
                        term="summer-2027",
                        role=RoleCategory.SWE,
                    )
                    for company in companies
                ),
            )
        )
        plan = await sync_module._plan_boards(
            repository,
            adapter,
            companies,
            NOW,
            random.Random(0),
            {},
            {"workday": posture},
            sync_module.Pacing(),
            False,
        )
        assert len(plan.ordered) == len(companies)
        assert plan.detail_budget == len(plan.detail_queue) == 6
        assert min(plan.page_budgets.values()) >= 1
        assert (
            sum(plan.page_budgets.values()) + plan.detail_budget + adapter.retry_reserve <= ceiling
        )


def test_reserved_workday_details_are_charged_once_before_full_reads() -> None:
    adapter = WorkdayAdapter()
    companies = [
        Company(
            name=f"Board {index}",
            slug=f"b{index}",
            platform=Platform.WORKDAY,
            workday_tenant=f"b{index}",
            workday_site="careers",
            workday_dc="wd3",
        )
        for index in range(3)
    ]
    extra_pages = 20
    ceiling = (
        adapter.retry_reserve
        + len(companies) * adapter.crawl_page_cap
        + adapter.detail_budget
        + adapter.check_reserve
        + extra_pages
    )
    pages, details = adapter.crawl_budgets(
        companies,
        {},
        {},
        ceiling,
        full_reads=[companies[0].registry_key],
        queued_details=adapter.detail_budget,
    )
    assert details == adapter.detail_budget
    assert pages[companies[0].registry_key] == adapter.crawl_page_cap + extra_pages
    assert sum(pages.values()) + details + adapter.retry_reserve + adapter.check_reserve == ceiling
