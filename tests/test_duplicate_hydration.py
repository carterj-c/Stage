import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
import respx

from stage.domain import Company, DetailFetch, Platform, job_id
from stage.http import HostBudgetExceededError, HttpClient, RatePosture
from stage.sources.smartrecruiters import SmartRecruitersAdapter, SmartRecruitersPosting
from stage.sources.workday import WorkdayAdapter, WorkdayPosting, _attach_descriptions, _to_job
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


@pytest.mark.parametrize("source", ["workday", "smartrecruiters"])
@pytest.mark.parametrize("status", [200, 404])
@pytest.mark.parametrize("budget", [2, 10])
@respx.mock
async def test_duplicate_listing_rows_make_one_detail_request_and_record_one_outcome(
    db_path: Path, source: str, status: int, budget: int
) -> None:
    company = Company(
        name="Acme",
        slug="acme",
        platform=Platform(source),
        workday_tenant="acme" if source == "workday" else None,
        workday_site="careers" if source == "workday" else None,
        workday_dc="wd3" if source == "workday" else None,
    )
    title = "Summer 2027 Software Engineer Intern"
    adapter: WorkdayAdapter | SmartRecruitersAdapter
    posting: dict[str, object]
    if source == "workday":
        adapter = WorkdayAdapter()
        base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/careers"
        posting = {
            "title": title,
            "externalPath": "/job/Montreal/x_p1",
            "locationsText": "2 Locations",
            "bulletFields": ["p1"],
        }
        respx.post(f"{base}/jobs").respond(
            200, json={"total": 2, "facets": [], "jobPostings": [posting, posting]}
        )
        detail = respx.get(f"{base}/job/Montreal/x_p1").respond(
            status,
            json={
                "jobPostingInfo": {
                    "jobDescription": "<p>Full description</p>",
                    "location": "Montreal",
                }
            },
        )
        ident = job_id(source, "acme-careers", "p1")
    else:
        adapter = SmartRecruitersAdapter()
        base = "https://api.smartrecruiters.com/v1/companies/acme/postings"
        posting = {"id": "p1", "name": title, "location": {"city": "Montreal"}}
        respx.get(base).respond(200, json={"totalFound": 2, "content": [posting, posting]})
        detail = respx.get(f"{base}/p1").respond(
            status, json={"jobAd": {"sections": {"body": {"text": "Full description"}}}}
        )
        ident = job_id(source, "acme", "p1")
    posture = RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=budget)
    async with HttpClient(
        allowed_hosts=adapter.hosts_for([company]),
        posture=posture,
        bucket_key=adapter.bucket_key,
        jitter=False,
    ) as client:
        if isinstance(adapter, WorkdayAdapter):
            result = await adapter.fetch(company, client, NOW, details=[ident], page_budget=1)
        else:
            result = await adapter.fetch(company, client, NOW, details=[ident])
    assert detail.call_count == 1
    assert len(result.jobs) == 2
    assert result.detail_fetches == (
        DetailFetch(id=ident, resolved=status == 200, failed=status != 200),
    )
    assert all(
        job.description == ("Full description" if status == 200 else "") for job in result.jobs
    )
    if source == "workday" and status == 200:
        assert all(job.location_raw == "Montreal" for job in result.jobs)
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source=source,
                run_started_at=NOW,
                jobs=result.jobs,
                detail_fetches=cast(tuple[DetailFetch, ...], result.detail_fetches),
            )
        )
        if status == 404:
            assert await repository.detail_queue(source, 1) == [ident]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT attempts FROM detail_fetches WHERE id = ?", (ident,)
        ).fetchone() == (1,)


@pytest.mark.parametrize("source", ["workday", "smartrecruiters"])
async def test_duplicate_refused_detail_is_deferred_once_per_id(source: str) -> None:
    company = Company(
        name="Acme",
        slug="acme",
        platform=Platform(source),
        workday_tenant="acme" if source == "workday" else None,
        workday_site="careers" if source == "workday" else None,
        workday_dc="wd3" if source == "workday" else None,
    )
    client = AsyncMock(spec=HttpClient)
    client.get_json.side_effect = HostBudgetExceededError("budget")
    if source == "workday":
        posting = WorkdayPosting(title="Intern", externalPath="/job/Montreal/x_p1")
        job = _to_job(company, posting, NOW)
        paired, outcomes = await _attach_descriptions(
            company, cast(HttpClient, client), [(posting, job), (posting, job)], {job.id}
        )
    else:
        smart_posting = SmartRecruitersPosting(id="p1", name="Intern")
        adapter = SmartRecruitersAdapter()
        job = adapter._to_job(company, smart_posting, NOW)
        smart_paired, outcomes = await adapter._attach_descriptions(
            company,
            cast(HttpClient, client),
            [(smart_posting, job), (smart_posting, job)],
            {job.id},
        )
        assert len(smart_paired) == 2
    if source == "workday":
        assert len(paired) == 2
    assert outcomes == []
    assert client.get_json.await_count == 1


async def test_workday_reuses_body_for_pathless_copies_and_preserves_named_locations() -> None:
    from dataclasses import replace

    from stage.http import JsonResponse

    company = Company(
        name="Acme",
        slug="acme",
        platform=Platform.WORKDAY,
        workday_tenant="acme",
        workday_site="careers",
        workday_dc="wd3",
    )
    posting = WorkdayPosting(title="Intern", externalPath="/job/Montreal/x_p1", bulletFields=["p1"])
    missing_path = WorkdayPosting(title="Intern", bulletFields=["p1"])
    base = _to_job(company, posting, NOW)
    named = replace(base, location_raw="Toronto")
    counted = replace(base, location_raw="2 Locations")
    client = AsyncMock(spec=HttpClient)
    client.get_json.return_value = JsonResponse(
        status=200,
        not_modified=False,
        payload={
            "jobPostingInfo": {"jobDescription": "<p>Full description</p>", "location": "Montreal"}
        },
    )
    merged, outcomes = await _attach_descriptions(
        company,
        cast(HttpClient, client),
        [(missing_path, named), (posting, counted)],
        {base.id},
    )
    assert client.get_json.await_count == 1
    assert outcomes == [DetailFetch(id=base.id, resolved=True)]
    assert [job.description for _, job in merged] == ["Full description", "Full description"]
    assert [job.location_raw for _, job in merged] == ["Toronto", "Montreal"]
