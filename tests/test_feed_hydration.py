import random
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
import respx

from stage.domain import (
    Job,
    JobStatus,
    QuarantinedJob,
    QuarantineFilters,
    RateState,
    SourceFresh,
    SyncFinished,
)
from stage.domain.visits import FEED_DETAIL_PREFIX, FEED_REJECTION_PREFIX
from stage.http import HostBudget, RatePosture
from stage.services import feed_hydration
from stage.services import sync as sync_module
from stage.services.screening import Screener
from stage.sources.base import FetchResult
from stage.storage import AsyncRepository, SourceBatch, open_repository

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
GH = "https://boards-api.greenhouse.io/v1/boards/acme/jobs/"
LEVER_ID = "337314ee-4de5-4b96-a419-1b9287a7df68"
FEED_URL = "https://feed.example.test/jobs"


class Feed:
    name: ClassVar[str] = "test-feed"
    rate_profile: ClassVar[str] = "feeds"
    hosts: ClassVar[frozenset[str]] = frozenset({"feed.example.test"})
    bucket_key: ClassVar[str] = ""

    def season_year(self, now: datetime) -> int:
        return now.year

    def plan(self, now: datetime) -> tuple[str, ...]:
        return (FEED_URL,)

    async def fetch(self, client: object, now: datetime) -> FetchResult:
        from stage.http import HttpClient

        assert isinstance(client, HttpClient)
        response = await client.get_json(FEED_URL)
        if response.not_modified:
            return FetchResult(not_modified=True)
        return FetchResult(jobs=(replace(job(), first_seen=now, last_seen=now),))


@pytest.fixture(autouse=True)
def unpaced(monkeypatch: pytest.MonkeyPatch) -> None:
    def posture(*_: object) -> RatePosture:
        return RatePosture(concurrency=1, min_interval_s=0, max_requests_per_run=300)

    monkeypatch.setattr(feed_hydration, "resolve", posture)
    monkeypatch.setattr(sync_module, "resolve", posture)
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"test-feed": Feed()})


def job(number: int = 123) -> Job:
    return Job(
        id=f"test-feed:acme:{number}",
        source="test-feed",
        company="Acme",
        title_raw="Software Engineer Intern",
        title_normalized="Software Engineer Intern",
        apply_url_raw=f"https://boards.greenhouse.io/acme/jobs/{number}",
        description="",
        location_raw="Toronto, ON, Canada",
        first_seen=NOW - timedelta(days=2),
        last_seen=NOW - timedelta(days=1),
    )


def payload(
    number: int = 123, body: str = "Build software systems. x &lt; 5 years."
) -> dict[str, object]:
    return {"id": number, "title": "Software Engineer Intern", "content": f"<p>{body}</p>"}


def attempt(path: Path, number: int = 123) -> tuple[int, int, int] | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT resolved, failed, attempts FROM detail_fetches WHERE id = ?",
            (FEED_DETAIL_PREFIX + job(number).id,),
        ).fetchone()
        return (int(row[0]), int(row[1]), int(row[2])) if row else None


async def hydrate(
    repo: AsyncRepository,
    now: datetime = NOW,
    budgets: dict[str, HostBudget] | None = None,
) -> feed_hydration.FeedHydrationResult:
    screener = Screener()
    try:
        return await feed_hydration.hydrate_feeds(
            repo,
            ["test-feed"],
            screener,
            now=now,
            budgets={} if budgets is None else budgets,
            postures={},
            rng=random.Random(1),
        )
    finally:
        screener.close()


async def seed(repo: AsyncRepository, *jobs: Job) -> None:
    await repo.apply_source_batch(SourceBatch(source="test-feed", run_started_at=NOW, jobs=jobs))


async def sync(repo: AsyncRepository, now: datetime = NOW, dry_run: bool = False) -> list[object]:
    return [
        event
        async for event in sync_module.sync(
            repo,
            [],
            sources=["test-feed"],
            now_fn=lambda: now,
            close_orphans=False,
            still_open=False,
            dry_run=dry_run,
        )
    ]


@respx.mock
async def test_existing_feed_304_still_hydrates(db_path: Path) -> None:
    respx.get(FEED_URL).respond(304)
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        events = await sync(repo)
        stored = await repo.get_job(job().id)
        assert stored is not None and "x < 5 years" in stored.description
        assert stored.first_seen == job().first_seen
        assert stored.status is JobStatus.OPEN
    assert route.call_count == 1
    assert attempt(db_path) == (1, 0, 1)
    finished = events[-1]
    assert isinstance(finished, SyncFinished)
    assert finished.requests == 2


@respx.mock
async def test_blank_feed_ingest_is_hydrated_and_survives_future_blank_feed(db_path: Path) -> None:
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await sync(repo)
        first = await repo.get_job(job().id)
        assert first is not None and first.description
        await sync(repo, NOW + timedelta(hours=2))
        second = await repo.get_job(job().id)
        assert second is not None and second.description == first.description
    assert route.call_count == 1


@respx.mock
async def test_recently_skipped_feed_still_hydrates(db_path: Path) -> None:
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await sync(repo)
        await seed(repo, job(124))
        respx.get(GH + "124").respond(200, json=payload(124))
        events = await sync(repo, NOW + timedelta(minutes=1))
        stored = await repo.get_job(job(124).id)
        assert stored is not None and stored.description
    assert route.call_count == 1
    assert any(isinstance(event, SourceFresh) for event in events)


@respx.mock
async def test_full_detail_is_rescreened_and_rejected_through_batch_writer(db_path: Path) -> None:
    respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        result = await hydrate(repo)
        assert await repo.get_job(job().id) is None
        rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(rejected) == 1
        assert rejected[0].verified_description == "Applicants must be enrolled in a PhD program."
        assert rejected[0].feed_summary == ""
        assert rejected[0].first_seen == job().first_seen
        assert rejected[0].last_seen == job().last_seen
        assert sum(batch.quarantined for batch in result.batches) == 1
    assert attempt(db_path) == (1, 0, 1)


@respx.mock
async def test_hydration_preserves_times_closed_rows_and_duplicate_links(db_path: Path) -> None:
    main, duplicate, closed = (
        job(),
        replace(job(124), duplicate_of=job().id),
        replace(job(125), status=JobStatus.CLOSED),
    )
    for number in (123, 124):
        respx.get(GH + str(number)).respond(200, json=payload(number))
    closed_route = respx.get(GH + "125").respond(200, json=payload(125))
    async with open_repository(db_path) as repo:
        await seed(repo, main, duplicate, closed)
        before = [await repo.get_job(row.id) for row in (main, duplicate, closed)]
        await hydrate(repo)
        after = [await repo.get_job(row.id) for row in (main, duplicate, closed)]
        for old, new in zip(before, after, strict=True):
            assert old is not None and new is not None
            assert (new.id, new.first_seen, new.last_seen, new.status, new.duplicate_of) == (
                old.id,
                old.first_seen,
                old.last_seen,
                old.status,
                old.duplicate_of,
            )
        assert after[2] == before[2]
    assert closed_route.call_count == 0


@respx.mock
@pytest.mark.parametrize("failure", ["identity", "error", "empty", "redirect"])
async def test_failed_endpoints_have_bounded_attempts_and_cooldown(
    db_path: Path, failure: str
) -> None:
    response = {
        "identity": httpx.Response(200, json=payload(456)),
        "error": httpx.Response(404, json={"error": "missing"}),
        "empty": httpx.Response(200, json=payload(body="")),
        "redirect": httpx.Response(302, headers={"location": GH + "456"}),
    }[failure]
    route = respx.get(GH + "123").mock(return_value=response)
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await hydrate(repo)
        await hydrate(repo, NOW + timedelta(minutes=10))
        assert route.call_count == 1
        for hour in (2, 4, 6):
            await hydrate(repo, NOW + timedelta(hours=hour))
        stored = await repo.get_job(job().id)
        assert stored is not None and not stored.description
    assert route.call_count == 3
    assert attempt(db_path) == (0, 1, 3)


@respx.mock
async def test_unsupported_and_ashby_never_resolve_or_fetch(db_path: Path) -> None:
    async with open_repository(db_path) as repo:
        await seed(
            repo,
            replace(job(), apply_url_raw="https://evil.test/job"),
            replace(job(124), apply_url_raw=f"https://jobs.ashbyhq.com/acme/{LEVER_ID}"),
        )
        result = await hydrate(repo)
    assert not result.requests
    assert attempt(db_path) == (0, 1, 1)
    assert attempt(db_path, 124) == (0, 1, 1)


@respx.mock
async def test_missing_lever_fields_and_detail_304_are_not_success(db_path: Path) -> None:
    respx.get(f"https://api.lever.co/v0/postings/acme/{LEVER_ID}").respond(
        200,
        json={"id": LEVER_ID, "text": "Intern", "descriptionPlain": "Summary"},
    )
    respx.get(GH + "124").respond(304)
    async with open_repository(db_path) as repo:
        await seed(
            repo, replace(job(), apply_url_raw=f"https://jobs.lever.co/acme/{LEVER_ID}"), job(124)
        )
        await hydrate(repo)
        for number in (123, 124):
            stored = await repo.get_job(job(number).id)
            assert stored is not None and not stored.description
            assert attempt(db_path, number) == (0, 1, 1)


@respx.mock
async def test_persisted_provider_block_does_not_spend_attempt(db_path: Path) -> None:
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                rate_state=(
                    RateState(
                        bucket="boards-api.greenhouse.io",
                        updated_at=NOW,
                        blocked_until=NOW + timedelta(hours=2),
                        reason="cooldown",
                    ),
                ),
            )
        )
        await hydrate(repo)
    assert route.call_count == 0
    assert attempt(db_path) is None


@respx.mock
async def test_429_preserves_pushback_and_defers_other_rows(db_path: Path) -> None:
    route = respx.route(host="boards-api.greenhouse.io").respond(
        429, headers={"Retry-After": "3600"}
    )
    async with open_repository(db_path) as repo:
        await seed(repo, job(), job(124))
        await hydrate(repo)
        states = await repo.load_rate_state()
        assert states["boards-api.greenhouse.io"].blocked_until is not None
    assert route.call_count == 1
    assert sorted(value for number in (123, 124) if (value := attempt(db_path, number))) == [
        (0, 1, 1)
    ]


@respx.mock
async def test_shared_provider_budget_is_not_reset(db_path: Path) -> None:
    posture = RatePosture(min_interval_s=0, max_requests_per_run=1)
    budget = HostBudget(posture=posture)
    budget.requests = 1
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await hydrate(repo, budgets={"boards-api.greenhouse.io": budget})
    assert not route.called
    assert attempt(db_path) is None


@respx.mock
async def test_run_is_bounded_to_30_actual_requests(db_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        number = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, json=payload(number))

    route = respx.route(host="boards-api.greenhouse.io").mock(side_effect=respond)
    async with open_repository(db_path) as repo:
        await seed(repo, *(job(number) for number in range(100, 140)))
        result = await hydrate(repo)
    assert route.call_count == 30
    assert len(result.requests) == 30
    assert sum(batch.updated for batch in result.batches) == 30


@respx.mock
@pytest.mark.parametrize("change", ["text", "closed", "url", "deleted", "reject-race"])
async def test_concurrent_updates_win_atomic_snapshot_guard(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    original_screen = Screener.__call__
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        respx.get(GH + "123").respond(
            200,
            json=payload(
                body="Applicants must be enrolled in a PhD program."
                if change == "reject-race"
                else "Build systems",
            ),
        )

        async def race(
            self: Screener, jobs: list[Job]
        ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            screened = await original_screen(self, jobs)
            current = await repo.get_job(job().id)
            assert current is not None
            if change == "deleted":
                with sqlite3.connect(db_path) as conn:
                    conn.execute("DELETE FROM jobs WHERE id = ?", (current.id,))
            else:
                update = {
                    "text": replace(current, description="Fresher description"),
                    "closed": replace(current, status=JobStatus.CLOSED),
                    "url": replace(
                        current, apply_url_raw="https://boards.greenhouse.io/acme/jobs/999"
                    ),
                    "reject-race": replace(current, description="Fresher description"),
                }[change]
                await seed(repo, update)
            return screened

        monkeypatch.setattr(Screener, "__call__", race)
        await hydrate(repo)
        stored = await repo.get_job(job().id)
        if change == "deleted":
            assert stored is None
        else:
            assert stored is not None
            if change in {"text", "reject-race"}:
                assert stored.description == "Fresher description"
            if change == "closed":
                assert stored.status is JobStatus.CLOSED and stored.description == ""
            if change == "url":
                assert stored.apply_url_raw.endswith("/999") and not stored.description
        assert not await repo.list_quarantined(QuarantineFilters())
    assert attempt(db_path) is None
    assert rejection_marker(db_path) is None


@respx.mock
async def test_dry_run_does_not_hydrate_or_spend_attempts(db_path: Path) -> None:
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await sync(repo, dry_run=True)
        stored = await repo.get_job(job().id)
        assert stored is not None and not stored.description
    assert attempt(db_path) is None


@respx.mock
async def test_tracking_record_is_unchanged(db_path: Path) -> None:
    from stage.domain.tracker import ApplicationStatus
    from stage.services.tracker import record_status, tracked_for

    respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await record_status(repo, [job().id], ApplicationStatus.SAVED, note="Keep this", now=NOW)
        before = await tracked_for(repo, [job().id])
        await hydrate(repo)
        assert await tracked_for(repo, [job().id]) == before


async def test_request_limit_counts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from tenacity import wait_none

    from stage.http import HostBudgetExceededError, HttpClient
    from stage.http import client as http_module

    monkeypatch.setattr(http_module, "wait_exponential_jitter", lambda **_: wait_none())
    calls = 0

    def failing(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    async with HttpClient(
        allowed_hosts=frozenset({"boards-api.greenhouse.io"}),
        posture=RatePosture(min_interval_s=0),
        jitter=False,
        request_limit=2,
        transport=httpx.MockTransport(failing),
    ) as client:
        with pytest.raises(HostBudgetExceededError):
            await client.get_json(GH + "123", revalidate=True)
        assert calls == client.request_count == 2


async def test_request_limit_is_shared_across_hosts_and_concurrent_requests() -> None:
    import asyncio

    from stage.http import HostBudgetExceededError, HttpClient

    async with HttpClient(
        allowed_hosts=frozenset({"api.lever.co", "boards-api.greenhouse.io"}),
        posture=RatePosture(min_interval_s=0),
        request_limit=3,
        jitter=False,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    ) as client:
        results = await asyncio.gather(
            *(
                client.get_json(f"https://{host}/job/{index}")
                for index, host in enumerate(["api.lever.co", "boards-api.greenhouse.io"] * 5)
            ),
            return_exceptions=True,
        )
        assert client.request_count == 3
    assert sum(isinstance(result, HostBudgetExceededError) for result in results) == 7


@respx.mock
async def test_hydration_rejection_survives_repeated_blank_feed_refresh(db_path: Path) -> None:
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        await sync(repo)
        assert await repo.get_job(job().id) is None
        original = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(original) == 1
        for hours in (6, 12):
            await sync(repo, NOW + timedelta(hours=hours))
            assert await repo.get_job(job().id) is None
            rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert len(rejected) == 1
            assert rejected[0].reason == original[0].reason
            assert rejected[0].matched_phrase == original[0].matched_phrase
            assert rejected[0].first_seen == original[0].first_seen
            assert rejected[0].last_seen == NOW + timedelta(hours=hours)
    assert route.call_count == 1
    assert attempt(db_path) == (1, 0, 1)


@respx.mock
@pytest.mark.parametrize("unavailable", ["persisted", "shared-budget", "429"])
async def test_unavailable_provider_cannot_starve_later_provider(
    db_path: Path,
    unavailable: str,
) -> None:
    route = respx.route(host="boards-api.greenhouse.io").respond(
        429, headers={"Retry-After": "3600"}
    )
    lever = respx.get(f"https://api.lever.co/v0/postings/acme/{LEVER_ID}").respond(
        200,
        json={
            "id": LEVER_ID,
            "text": "Software Engineer Intern",
            "descriptionPlain": "Build systems",
            "descriptionBodyPlain": "",
            "salaryDescriptionPlain": "",
            "additionalPlain": "",
            "lists": [],
        },
    )
    budgets: dict[str, HostBudget] = {}
    async with open_repository(db_path) as repo:
        await seed(
            repo,
            *(job(number) for number in range(100, 165)),
            replace(
                job(999),
                apply_url_raw=f"https://jobs.lever.co/acme/{LEVER_ID}",
                first_seen=NOW - timedelta(days=3),
            ),
        )
        if unavailable == "persisted":
            await repo.apply_source_batch(
                SourceBatch(
                    source="test-feed",
                    run_started_at=NOW,
                    rate_state=(
                        RateState(
                            bucket="boards-api.greenhouse.io",
                            updated_at=NOW,
                            blocked_until=NOW + timedelta(hours=2),
                            reason="cooldown",
                        ),
                    ),
                )
            )
        if unavailable == "shared-budget":
            budget = HostBudget(posture=RatePosture(min_interval_s=0, max_requests_per_run=1))
            budget.requests = 1
            budgets["boards-api.greenhouse.io"] = budget
        result = await hydrate(repo, budgets=budgets)
        stored = await repo.get_job(job(999).id)
        assert stored is not None and stored.description == "Build systems"
    assert lever.call_count == 1
    assert route.call_count == (1 if unavailable == "429" else 0)
    assert len(result.requests) <= feed_hydration.REQUESTS_PER_SYNC
    assert attempt(db_path, 999) == (1, 0, 1)


@respx.mock
async def test_unverified_nonblank_update_cannot_promote_hydration_rejection(db_path: Path) -> None:
    respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await hydrate(repo)
        before = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        await seed(repo, replace(job(), description="A new short summary of our team"))
        assert await repo.get_job(job().id) is None
        assert await repo.list_quarantined(QuarantineFilters(source="test-feed")) == before


@respx.mock
async def test_provider_paging_stops_at_time_ceiling(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    route = respx.route(host="boards-api.greenhouse.io").respond(200, json=payload())
    ticks = iter([0.0, 61.0, 61.0])
    monkeypatch.setattr(feed_hydration, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    async with open_repository(db_path) as repo:
        await seed(repo, *(job(number) for number in range(100, 165)))
        result = await hydrate(repo)
    assert not route.called
    assert not result.requests
    assert attempt(db_path, 100) is None


@respx.mock
async def test_held_changed_metadata_signals_and_live_requirements_do_not_release(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import SourceSignals

    respx.get(FEED_URL).respond(200, json={})
    original = respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    changed_url = "https://boards.greenhouse.io/acme/jobs/456"
    replacement = respx.get(GH + "456").respond(200, json=payload(456))
    async with open_repository(db_path) as repo:
        await sync(repo)
        changed = replace(
            job(),
            title_raw="Senior Designer",
            title_normalized="Senior Designer",
            location_raw="Paris, France",
            apply_url_raw=changed_url,
            signals=SourceSignals(
                terms=("Summer 2028",),
                degrees=("PhD",),
                sponsorship="no",
                season="Summer",
                category="Design",
                employment_type="full-time",
            ),
            source_posted_at=NOW + timedelta(days=2),
        )

        async def refreshed(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(jobs=(replace(changed, last_seen=now),))

        monkeypatch.setattr(Feed, "fetch", refreshed)
        original.respond(200, json=payload(body="Bachelor's students welcome."))
        for elapsed in (7, 20, 100):
            await sync(repo, NOW + timedelta(days=elapsed))
            assert await repo.get_job(job().id) is None
            rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert len(rejected) == 1 and "phd" in rejected[0].matched_phrase
        assert attempt(db_path) == (1, 0, 1)
    assert original.call_count == 1
    assert replacement.call_count == 0


@respx.mock
async def test_removed_feed_posting_cannot_be_reopened(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        await sync(repo)
        route.respond(200, json=payload(body="Bachelor's students welcome."))

        async def absent(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(jobs=())

        monkeypatch.setattr(Feed, "fetch", absent)
        await sync(repo, NOW + timedelta(days=100))
        assert await repo.get_job(job().id) is None
        rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(rejected) == 1
        assert await repo.preview_purge(NOW + timedelta(days=101)) == await repo.purge(
            NOW + timedelta(days=101)
        )
        assert len(await repo.list_quarantined(QuarantineFilters(source="test-feed"))) == 1
    assert route.call_count == 1


@respx.mock
async def test_duplicate_restoration_does_not_release_a_held_copy(db_path: Path) -> None:
    from stage.domain import DetailFetch, RejectionReason

    held = QuarantinedJob(
        id=job(124).id,
        source="test-feed",
        company="Acme",
        title_raw=job(124).title_raw,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase=f"copy of {job().id} rejected for phd",
        first_seen=job(124).first_seen,
        last_seen=job(124).last_seen,
        apply_url_raw=job(124).apply_url_raw,
        location_raw=job(124).location_raw,
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(held,),
                detail_fetches=(DetailFetch(FEED_REJECTION_PREFIX + held.id, resolved=True),),
            )
        )
        await seed(repo, job())
        assert await repo.get_job(held.id) is None
        assert (await repo.list_quarantined(QuarantineFilters(source="test-feed")))[0] == held
        assert await repo.get_job(job().id) is not None


@respx.mock
async def test_held_rows_cannot_starve_ordinary_hydration(db_path: Path) -> None:
    from stage.domain import DetailFetch, RejectionReason

    held = tuple(
        QuarantinedJob(
            id=job(number).id,
            source="test-feed",
            company="Acme",
            title_raw=job(number).title_raw,
            reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
            matched_phrase="phd required",
            first_seen=job(number).first_seen,
            last_seen=job(number).last_seen,
            apply_url_raw=job(number).apply_url_raw,
            location_raw=job(number).location_raw,
        )
        for number in range(100, 140)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        number = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, json=payload(number))

    route = respx.route(host="boards-api.greenhouse.io").mock(side_effect=respond)
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=held,
                detail_fetches=tuple(
                    DetailFetch(FEED_DETAIL_PREFIX + entry.id, resolved=True) for entry in held
                ),
            )
        )
        await seed(repo, *(job(number) for number in range(200, 230)))
        result = await hydrate(repo, NOW + timedelta(days=10))
        assert sum(batch.updated for batch in result.batches) == 30
        assert (
            len(await repo.list_quarantined(QuarantineFilters(source="test-feed", limit=None)))
            == 40
        )
    assert route.call_count == 30
    assert all(int(call.request.url.path.rsplit("/", 1)[1]) >= 200 for call in route.calls)


@respx.mock
async def test_batch_guard_cannot_promote_a_quarantined_row_during_a_race(db_path: Path) -> None:
    respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        snapshot = await repo.get_job(job().id)
        assert snapshot is not None
        await hydrate(repo)
        proposed = replace(snapshot, description="Bachelor's students welcome.")
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW + timedelta(days=8),
                jobs=(proposed,),
                hydration_expected=(snapshot,),
            )
        )
        assert await repo.get_job(job().id) is None
        assert len(await repo.list_quarantined(QuarantineFilters(source="test-feed"))) == 1


@pytest.mark.parametrize("marker", ["failed", "unsupported", "successful"])
async def test_inherited_rejection_with_attempt_marker_reopens_when_canonical_is_eligible(
    db_path: Path,
    marker: str,
) -> None:
    from stage.domain import DetailFetch, RejectionReason

    copy = QuarantinedJob(
        id=job(124).id,
        source="test-feed",
        company="Acme",
        title_raw=job(124).title_raw,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase=f"copy of {job().id} rejected for phd",
        first_seen=job(124).first_seen,
        last_seen=job(124).last_seen,
        apply_url_raw=job(124).apply_url_raw,
    )
    outcome = DetailFetch(
        FEED_DETAIL_PREFIX + copy.id,
        resolved=marker == "successful",
        failed=marker != "successful",
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(copy,),
                detail_fetches=(outcome,),
            )
        )
        await seed(repo, job())
        restored = await repo.get_job(copy.id)
        assert restored is not None and restored.duplicate_of == job().id
        assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))


async def test_failed_marker_does_not_keep_stale_inherited_quarantine(db_path: Path) -> None:
    from stage.domain import DetailFetch, RejectionReason

    copy = QuarantinedJob(
        id=job(124).id,
        source="test-feed",
        company="Acme",
        title_raw=job(124).title_raw,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase=f"copy of {job().id} rejected for phd",
        first_seen=job(124).first_seen,
        last_seen=job(124).last_seen,
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(copy,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + copy.id, failed=True),),
            )
        )
        preview = await repo.preview_purge(NOW + timedelta(days=100))
        result = await repo.purge(NOW + timedelta(days=100))
        assert preview.forgotten == result.forgotten == 1
        assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))


def rejection_marker(path: Path, number: int = 123) -> tuple[int, int] | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT resolved, failed FROM detail_fetches WHERE id = ?",
            (FEED_REJECTION_PREFIX + job(number).id,),
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else None


@respx.mock
@pytest.mark.parametrize("result", ["rejected", "accepted", "failed", "unsupported"])
async def test_only_hydrated_text_that_is_rejected_records_verified_hold(
    db_path: Path,
    result: str,
) -> None:
    if result == "rejected":
        response = httpx.Response(
            200,
            json=payload(
                body="Applicants must be enrolled in a PhD program.",
            ),
        )
    elif result == "accepted":
        response = httpx.Response(200, json=payload())
    else:
        response = httpx.Response(404, json={"error": "missing"})
    route = respx.get(GH + "123").mock(return_value=response)
    row = (
        replace(job(), apply_url_raw="https://unknown.example/job")
        if result == "unsupported"
        else job()
    )
    async with open_repository(db_path) as repo:
        await seed(repo, row)
        await hydrate(repo)
        if result == "rejected":
            assert await repo.get_job(row.id) is None
            assert len(await repo.list_quarantined(QuarantineFilters(source="test-feed"))) == 1
            assert rejection_marker(db_path) == (1, 0)
            await seed(repo, replace(row, description="unverified summary"))
            assert await repo.get_job(row.id) is None
            assert (await repo.purge(NOW + timedelta(days=100))).forgotten == 0
        else:
            assert rejection_marker(db_path) is None
    assert route.call_count == (0 if result == "unsupported" else 1)


async def test_legacy_successful_direct_rejection_remains_conservatively_held(
    db_path: Path,
) -> None:
    from stage.domain import DetailFetch, RejectionReason

    legacy = QuarantinedJob(
        id=job().id,
        source="test-feed",
        company="Acme",
        title_raw=job().title_raw,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="phd required",
        first_seen=job().first_seen,
        last_seen=job().last_seen,
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(legacy,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + legacy.id, resolved=True),),
            )
        )
        assert rejection_marker(db_path) is None
        await seed(repo, replace(job(), description="unverified summary"))
        assert await repo.get_job(legacy.id) is None
        assert (await repo.purge(NOW + timedelta(days=100))).forgotten == 0


@respx.mock
async def test_accepted_hydration_is_not_later_mistaken_for_a_verified_rejection(
    db_path: Path,
) -> None:
    from stage.domain import RejectionReason

    respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        await hydrate(repo)
        assert rejection_marker(db_path) is None
        later = QuarantinedJob(
            id=job().id,
            source="test-feed",
            company="Acme",
            title_raw=job().title_raw,
            reason=RejectionReason.NOT_AN_INTERNSHIP,
            matched_phrase="full time",
            first_seen=job().first_seen,
            last_seen=NOW,
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(later,),
            )
        )
        await seed(repo, job())
        assert await repo.get_job(job().id) is not None
        assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))


@respx.mock
@pytest.mark.parametrize("path", ["refresh", "rescreen"])
async def test_later_screening_rejection_preserves_verified_body_hold(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    from stage.classify.scope import Rejection, to_quarantined
    from stage.domain import RejectionReason
    from stage.domain.quarantine import QuarantinedJob as _QuarantinedJob
    from stage.services import maintenance
    from stage.services.screening import normalize_batch

    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await sync(repo)
        hydrated = await repo.get_job(job().id)
        assert hydrated is not None and hydrated.description
        assert rejection_marker(db_path) is None
        original = normalize_batch

        def reject_body(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[_QuarantinedJob, ...]]:
            kept, rejected = original(jobs)
            return (), rejected + tuple(
                to_quarantined(entry, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "new rule"))
                for entry in kept
            )

        if path == "refresh":
            monkeypatch.setattr("stage.services.screening.normalize_batch", reject_body)
            await sync(repo, NOW + timedelta(hours=6))
            monkeypatch.setattr("stage.services.screening.normalize_batch", original)
        else:
            original_reclassifications = maintenance._reclassifications

            def rules_reject(
                jobs: Sequence[Job],
            ) -> tuple[tuple[Job, ...], tuple[_QuarantinedJob, ...]]:
                candidates, rejected = original_reclassifications(jobs)
                return candidates, rejected + tuple(
                    to_quarantined(
                        entry, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "new rule")
                    )
                    for entry in candidates
                )

            monkeypatch.setattr(maintenance, "_reclassifications", rules_reject)
            await maintenance.rescreen(repo, now=NOW + timedelta(hours=6))
            monkeypatch.setattr(maintenance, "_reclassifications", original_reclassifications)
        assert await repo.get_job(job().id) is None
        assert rejection_marker(db_path) == (1, 0)
        await sync(repo, NOW + timedelta(hours=12))
        assert await repo.get_job(job().id) is None
        assert len(await repo.list_quarantined(QuarantineFilters(source="test-feed"))) == 1
        assert (await repo.purge(NOW + timedelta(days=100))).forgotten == 0
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("path", ["refresh", "rescreen"])
async def test_later_rejection_does_not_override_concurrent_body_or_close_it(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    from stage.classify.scope import Rejection, to_quarantined
    from stage.domain import RejectionReason
    from stage.services import maintenance
    from stage.services.screening import normalize_batch

    respx.get(FEED_URL).respond(200, json={})
    respx.get(GH + "123").respond(200, json=payload())
    changed = False

    def update_after_snapshot() -> None:
        nonlocal changed
        if not changed:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE jobs SET description = ?, title_raw = ? WHERE id = ?",
                    ("Fresher description", "Updated software intern", job().id),
                )
            changed = True

    async with open_repository(db_path) as repo:
        await sync(repo)
        if path == "refresh":

            def screen(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
                kept, rejected = normalize_batch(jobs)
                update_after_snapshot()
                return (), rejected + tuple(
                    to_quarantined(
                        entry, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "new rule")
                    )
                    for entry in kept
                )

            monkeypatch.setattr("stage.services.screening.normalize_batch", screen)
            await sync(repo, NOW + timedelta(hours=6))
        else:
            reclassify = maintenance._reclassifications

            def screen_rules(
                jobs: Sequence[Job],
            ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
                candidates, rejected = reclassify(jobs)
                if not changed:
                    update_after_snapshot()
                    rejected += tuple(
                        to_quarantined(
                            entry, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "new rule")
                        )
                        for entry in candidates
                    )
                return candidates, rejected

            monkeypatch.setattr(maintenance, "_reclassifications", screen_rules)
            await maintenance.rescreen(repo, now=NOW + timedelta(hours=6))
        current = await repo.get_job(job().id)
        assert current is not None and current.description == "Fresher description"
        assert current.title_raw == "Updated software intern"
        assert current.status is JobStatus.OPEN
        assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert rejection_marker(db_path) is None


@respx.mock
@pytest.mark.parametrize("decision", ["accept", "reject"])
@pytest.mark.parametrize("initial", ["blank", "verified"])
async def test_conflicted_feed_batch_retries_without_validator_or_recent_success(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    initial: str,
) -> None:
    from stage.classify.scope import Rejection, to_quarantined
    from stage.domain import CompanyVisit, DetailFetch, HttpValidator, RejectionReason
    from stage.services.screening import normalize_batch
    from stage.storage.sqlite_repo import SqliteRepository

    requests: list[httpx.Request] = []

    def feed_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("if-none-match") == '"latest"':
            return httpx.Response(304)
        return httpx.Response(200, json={}, headers={"ETag": '"latest"'})

    respx.get(FEED_URL).mock(side_effect=feed_response)
    respx.get(GH + "123").respond(200, json=payload())
    changed = False
    async with open_repository(db_path) as repo:
        await seed(repo, job())
        if initial == "verified":
            await hydrate(repo)
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW - timedelta(days=1),
                validators=(HttpValidator(url=FEED_URL, etag='"old"'),),
                visits=(CompanyVisit(board="test-feed", succeeded=True),),
            )
        )

        def concurrent_screen(
            jobs: Sequence[Job],
        ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            kept, rejected = normalize_batch(jobs)
            if not changed:
                other = SqliteRepository.connect(db_path)
                try:
                    snapshot = other.get_job(job().id)
                    assert snapshot is not None
                    other.apply_source_batch(
                        SourceBatch(
                            source="test-feed",
                            run_started_at=NOW,
                            jobs=(replace(snapshot, description="Newly hydrated full body"),),
                            hydration_expected=(snapshot,),
                            detail_fetches=(
                                DetailFetch(FEED_DETAIL_PREFIX + snapshot.id, resolved=True),
                            ),
                        )
                    )
                finally:
                    other.close()
                changed = True
            if decision == "reject":
                return (), rejected + tuple(
                    to_quarantined(
                        entry, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "new rule")
                    )
                    for entry in kept
                )
            return kept, rejected

        monkeypatch.setattr("stage.services.screening.normalize_batch", concurrent_screen)
        events = await sync(repo)
        current = await repo.get_job(job().id)
        assert current is not None and current.description == "Newly hydrated full body"
        assert current.status is JobStatus.OPEN
        assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert not await repo.load_validators("test-feed")
        assert not [visit for visit in await repo.all_visits() if visit.source == "test-feed"]
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert len(requests) == 1 and requests[0].headers.get("if-none-match") == '"old"'
        await sync(repo, NOW + timedelta(minutes=1))
        assert len(requests) == 2 and "if-none-match" not in requests[1].headers
        current = await repo.get_job(job().id)
        if decision == "reject":
            assert current is None
            assert rejection_marker(db_path) == (1, 0)
        else:
            assert current is not None and current.description == "Newly hydrated full body"


@respx.mock
@pytest.mark.parametrize("decision", ["accepted", "rejected"])
async def test_changed_feed_summary_is_screened_without_replacing_verified_body(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload(body="Full verified job description."))
    async with open_repository(db_path) as repo:
        await sync(repo)
        original = await repo.get_job(job().id)
        assert original is not None and original.description == "Full verified job description."
        summary = (
            "Applicants must be enrolled in a PhD program."
            if decision == "rejected"
            else "A short feed summary."
        )

        async def changed(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(
                jobs=(
                    replace(
                        job(),
                        title_raw="Software Developer Intern",
                        description=summary,
                        first_seen=now,
                        last_seen=now,
                    ),
                )
            )

        monkeypatch.setattr(Feed, "fetch", changed)
        events = await sync(repo, NOW + timedelta(hours=6))
        assert isinstance(events[-1], SyncFinished)
        assert events[-1].outcome.value == "success"
        stored = await repo.get_job(job().id)
        if decision == "accepted":
            assert stored is not None
            assert stored.description == original.description
            assert stored.title_raw == "Software Developer Intern"
            assert rejection_marker(db_path) is None
        else:
            assert stored is None
            entries = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert len(entries) == 1 and "phd" in entries[0].matched_phrase
            assert rejection_marker(db_path) == (1, 0)
            await sync(repo, NOW + timedelta(hours=12))
            assert await repo.get_job(job().id) is None
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("decision", ["accepted", "rejected"])
async def test_changed_summary_retries_after_concurrent_verified_body_update(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    from stage.http import HttpClient
    from stage.services.screening import normalize_batch

    requests: list[httpx.Request] = []

    def feed_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={}, headers={"ETag": '"changed"'})

    respx.get(FEED_URL).mock(side_effect=feed_response)
    respx.get(GH + "123").respond(200, json=payload(body="Original verified body."))
    async with open_repository(db_path) as repo:
        await sync(repo)
        summary = (
            "Applicants must be enrolled in a PhD program."
            if decision == "rejected"
            else "Updated summary."
        )

        async def changed(self: Feed, client: object, now: datetime) -> FetchResult:
            assert isinstance(client, HttpClient)
            response = await client.get_json(FEED_URL)
            if response.not_modified:
                return FetchResult(not_modified=True)
            return FetchResult(jobs=(replace(job(), description=summary, last_seen=now),))

        changed_once = False

        def concurrent_screen(
            jobs: Sequence[Job],
        ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed_once
            result = normalize_batch(jobs)
            if not changed_once:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE jobs SET description = ? WHERE id = ?",
                        ("Fresher verified body.", job().id),
                    )
                changed_once = True
            return result

        monkeypatch.setattr(Feed, "fetch", changed)
        monkeypatch.setattr("stage.services.screening.normalize_batch", concurrent_screen)
        events = await sync(repo, NOW + timedelta(hours=6))
        current = await repo.get_job(job().id)
        assert current is not None and current.description == "Fresher verified body."
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert rejection_marker(db_path) is None
        assert not await repo.load_validators("test-feed")
        events = await sync(repo, NOW + timedelta(hours=6, minutes=1))
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "success"
        assert "if-none-match" not in requests[-1].headers
        current = await repo.get_job(job().id)
        if decision == "accepted":
            assert current is not None and current.description == "Fresher verified body."
        else:
            assert current is None and rejection_marker(db_path) == (1, 0)


@respx.mock
@pytest.mark.parametrize("restriction", ["body", "summary"])
async def test_distinct_summary_and_body_flags_survive_blank_refresh_and_rescreen(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restriction: str,
) -> None:
    from stage.services.maintenance import rescreen

    body = "Must be a US citizen." if restriction == "body" else "Build software systems."
    summary = "Must be a US citizen." if restriction == "summary" else "No travel required."
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload(body=body))
    async with open_repository(db_path) as repo:
        await sync(repo)

        async def changed(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(jobs=(replace(job(), description=summary, last_seen=now),))

        monkeypatch.setattr(Feed, "fetch", changed)
        await sync(repo, NOW + timedelta(hours=6))
        current = await repo.get_job(job().id)
        assert current is not None
        assert (current.description, current.feed_summary, current.work_auth_flag) == (
            body,
            summary,
            True,
        )

        async def blank(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(jobs=(replace(job(), last_seen=now),))

        monkeypatch.setattr(Feed, "fetch", blank)
        await sync(repo, NOW + timedelta(hours=12))
        await rescreen(repo, now=NOW + timedelta(hours=13))
        current = await repo.get_job(job().id)
        assert current is not None
        assert (current.description, current.feed_summary, current.work_auth_flag) == (
            body,
            summary,
            True,
        )
    async with open_repository(db_path) as reopened:
        persisted = await reopened.get_job(job().id)
        assert persisted is not None
        assert (persisted.description, persisted.feed_summary, persisted.work_auth_flag) == (
            body,
            summary,
            True,
        )
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("restricted", ["body", "summary"])
async def test_independent_text_screening_prevents_cross_text_phd_masking(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restricted: str,
) -> None:
    from stage.services.maintenance import rescreen

    phd = "Applicants must be enrolled in a PhD program."
    undergrad = "Open to undergraduate"
    body = phd if restricted == "body" else undergrad
    summary = phd if restricted == "summary" else undergrad
    respx.get(FEED_URL).respond(200, json={})
    route = respx.get(GH + "123").respond(200, json=payload(body=undergrad))
    async with open_repository(db_path) as repo:
        await sync(repo)
        if restricted == "body":
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE jobs SET description = ? WHERE id = ?", (body, job().id))

        async def changed(self: Feed, client: object, now: datetime) -> FetchResult:
            return FetchResult(jobs=(replace(job(), description=summary, last_seen=now),))

        monkeypatch.setattr(Feed, "fetch", changed)
        await sync(repo, NOW + timedelta(hours=6))
        assert await repo.get_job(job().id) is None
        rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(rejected) == 1 and "phd" in rejected[0].matched_phrase
        assert rejected[0].feed_summary == summary
        assert rejected[0].verified_description == body
        assert rejection_marker(db_path) == (1, 0)
        await rescreen(repo, now=NOW + timedelta(hours=7))
        await sync(repo, NOW + timedelta(hours=12))
        assert await repo.get_job(job().id) is None
        assert (await repo.list_quarantined(QuarantineFilters(source="test-feed")))[
            0
        ].feed_summary == summary
    async with open_repository(db_path) as reopened:
        persisted = await reopened.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(persisted) == 1
        assert (persisted[0].verified_description, persisted[0].feed_summary) == (body, summary)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("decision", ["accepted", "rejected"])
async def test_concurrent_summary_update_prevents_stale_screening_and_retries(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    from stage.http import HttpClient
    from stage.services.screening import normalize_batch

    requests: list[httpx.Request] = []

    def feed_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={}, headers={"ETag": '"summary"'})

    respx.get(FEED_URL).mock(side_effect=feed_response)
    respx.get(GH + "123").respond(200, json=payload(body="Original verified body."))
    async with open_repository(db_path) as repo:
        await sync(repo)
        summary = (
            "Applicants must be enrolled in a PhD program."
            if decision == "rejected"
            else "Updated feed summary."
        )

        async def changed(self: Feed, client: object, now: datetime) -> FetchResult:
            assert isinstance(client, HttpClient)
            response = await client.get_json(FEED_URL)
            if response.not_modified:
                return FetchResult(not_modified=True)
            return FetchResult(jobs=(replace(job(), description=summary, last_seen=now),))

        changed_once = False

        def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed_once
            result = normalize_batch(jobs)
            if not changed_once:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE jobs SET feed_summary = ? WHERE id = ?",
                        (
                            "Concurrent summary",
                            job().id,
                        ),
                    )
                changed_once = True
            return result

        monkeypatch.setattr(Feed, "fetch", changed)
        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = await sync(repo, NOW + timedelta(hours=6))
        current = await repo.get_job(job().id)
        assert current is not None
        assert (current.description, current.feed_summary) == (
            "Original verified body.",
            "Concurrent summary",
        )
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert rejection_marker(db_path) is None
        assert not await repo.load_validators("test-feed")
        events = await sync(repo, NOW + timedelta(hours=6, minutes=1))
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "success"
        assert "if-none-match" not in requests[-1].headers
        current = await repo.get_job(job().id)
        if decision == "accepted":
            assert current is not None and current.description == "Original verified body."
            assert current.feed_summary == summary
        else:
            assert current is None and rejection_marker(db_path) == (1, 0)
            rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert len(rejected) == 1 and rejected[0].feed_summary == summary


@respx.mock
@pytest.mark.parametrize("restricted", ["body", "summary"])
async def test_rescreen_independently_rejects_preexisting_phd_evidence(
    db_path: Path,
    restricted: str,
) -> None:
    from stage.services.maintenance import rescreen

    phd = "Applicants must be enrolled in a PhD program."
    undergrad = "Open to undergraduate"
    respx.get(FEED_URL).respond(200, json={})
    respx.get(GH + "123").respond(200, json=payload(body=undergrad))
    async with open_repository(db_path) as repo:
        await sync(repo)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE jobs SET description = ?, feed_summary = ? WHERE id = ?",
                (
                    phd if restricted == "body" else undergrad,
                    phd if restricted == "summary" else undergrad,
                    job().id,
                ),
            )
        result = await rescreen(repo, now=NOW + timedelta(hours=6))
        assert result.quarantined == 1
        assert await repo.get_job(job().id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(entries) == 1 and "phd" in entries[0].matched_phrase
        assert entries[0].feed_summary == (phd if restricted == "summary" else undergrad)
        assert entries[0].verified_description == (phd if restricted == "body" else undergrad)
        assert rejection_marker(db_path) == (1, 0)
        await sync(repo, NOW + timedelta(hours=12))
        assert await repo.get_job(job().id) is None


@respx.mock
@pytest.mark.parametrize("change", ["reject_to_accept", "accept_to_reject"])
async def test_rescreen_retries_when_summary_changes_during_classification(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from stage.services import maintenance

    phd = "Applicants must be enrolled in a PhD program."
    eligible = "Must be a US citizen."
    respx.get(FEED_URL).respond(200, json={})
    respx.get(GH + "123").respond(200, json=payload(body="Build software systems."))
    async with open_repository(db_path) as repo:
        await sync(repo)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE jobs SET feed_summary = ? WHERE id = ?",
                (
                    phd if change == "reject_to_accept" else eligible,
                    job().id,
                ),
            )
        classify = maintenance._reclassifications
        changed = False

        def racing(
            jobs: Sequence[Job],
        ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            result = classify(jobs)
            if not changed:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE jobs SET feed_summary = ? WHERE id = ?",
                        (
                            eligible if change == "reject_to_accept" else phd,
                            job().id,
                        ),
                    )
                changed = True
            return result

        monkeypatch.setattr(maintenance, "_reclassifications", racing)
        await maintenance.rescreen(repo, now=NOW + timedelta(hours=6))
        current = await repo.get_job(job().id)
        if change == "reject_to_accept":
            assert current is not None and current.description == "Build software systems."
            assert current.feed_summary == eligible and current.work_auth_flag
            assert not await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert rejection_marker(db_path) is None
        else:
            assert current is None and rejection_marker(db_path) == (1, 0)
            rejected = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
            assert len(rejected) == 1 and rejected[0].feed_summary == phd


async def test_rescreen_does_not_count_held_durable_evidence_as_released(db_path: Path) -> None:
    from stage.domain import DetailFetch, RejectionReason
    from stage.services.maintenance import rescreen

    held = QuarantinedJob(
        id=job().id,
        source="test-feed",
        company="Acme",
        title_raw=job().title_raw,
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="old rule",
        first_seen=job().first_seen,
        last_seen=job().last_seen,
        apply_url_raw=job().apply_url_raw,
        location_raw=job().location_raw,
        verified_description="Build software systems.",
        feed_summary="Must be a US citizen.",
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(held,),
                detail_fetches=(DetailFetch(FEED_REJECTION_PREFIX + held.id, resolved=True),),
            )
        )
        result = await rescreen(repo, now=NOW + timedelta(hours=1))
        assert result.released == 0
        assert await repo.get_job(held.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(entries) == 1
        assert (entries[0].verified_description, entries[0].feed_summary) == (
            held.verified_description,
            held.feed_summary,
        )
