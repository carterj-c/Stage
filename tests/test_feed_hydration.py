import random
import sqlite3
from collections.abc import Callable, Sequence
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


def clock_at(moment: datetime) -> Callable[[], datetime]:
    def now_at() -> datetime:
        return moment

    return now_at


class Feed:
    name: ClassVar[str] = "test-feed"
    description_is_summary: ClassVar[bool] = True
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
async def test_unsupported_never_resolves_or_fetches(db_path: Path) -> None:
    async with open_repository(db_path) as repo:
        await seed(repo, replace(job(), apply_url_raw="https://evil.test/job"))
        result = await hydrate(repo)
    assert not result.requests
    assert attempt(db_path) == (0, 1, 1)


@respx.mock
async def test_ashby_feed_link_hydrates_exact_board_posting(db_path: Path) -> None:
    url = f"https://jobs.ashbyhq.com/acme/{LEVER_ID}"
    respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").respond(
        200,
        json={
            "jobs": [
                {
                    "id": LEVER_ID,
                    "jobUrl": url,
                    "title": "Software Engineer Intern",
                    "isListed": True,
                    "descriptionPlain": "Build software systems.",
                }
            ]
        },
    )
    async with open_repository(db_path) as repo:
        await seed(repo, replace(job(), apply_url_raw=url))
        result = await hydrate(repo)
        stored = await repo.get_job(job().id)
    assert len(result.requests) == 1
    assert stored is not None and stored.description == "Build software systems."
    assert attempt(db_path) == (1, 0, 1)


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
async def test_inherited_rejection_with_attempt_marker_waits_for_fresh_copy_evidence(
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
        assert await repo.get_job(copy.id) is None
        assert await repo.list_quarantined(QuarantineFilters(source="test-feed")) == [copy]


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
        assert rejection_marker(db_path) is None
        assert await repo.get_job(job().id) is None
        assert len(await repo.list_quarantined(QuarantineFilters(source="test-feed"))) == 1


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


@pytest.mark.parametrize("technical", ["body", "summary"])
def test_role_qualification_combines_positive_evidence_without_masking_requirements(
    technical: str,
) -> None:
    from stage.services.screening import normalize_batch

    engineering = "Develop backend software engineering systems."
    neutral = "Hybrid position."
    row = replace(
        job(),
        title_raw="Systems Engineer Intern",
        title_normalized="Systems Engineer Intern",
        description=engineering if technical == "body" else neutral,
        feed_summary=engineering if technical == "summary" else neutral,
    )
    kept, rejected = normalize_batch((row,))
    assert len(kept) == 1 and not rejected
    assert kept[0].description == row.description and kept[0].feed_summary == row.feed_summary
    phd = "Applicants must be enrolled in a PhD program."
    restricted = (
        replace(row, feed_summary=phd) if technical == "body" else replace(row, description=phd)
    )
    _, rejected = normalize_batch((restricted,))
    assert len(rejected) == 1 and "phd" in rejected[0].matched_phrase
    auth = "Must be a US citizen."
    restricted = (
        replace(row, feed_summary=auth) if technical == "body" else replace(row, description=auth)
    )
    kept, rejected = normalize_batch((restricted,))
    assert len(kept) == 1 and not rejected and kept[0].work_auth_flag


@respx.mock
@pytest.mark.parametrize("target", ["greenhouse", "failed", "unsupported"])
async def test_fresh_hanzili_summary_is_never_a_complete_posting(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    apply_url = (
        "https://example.com/acme/job"
        if target == "unsupported"
        else "https://boards.greenhouse.io/acme/jobs/123"
    )
    body = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Software Engineer Intern | Acme | Hybrid position | Company | Intern | "
        f"Toronto, ON | [Apply]({apply_url}) |\n"
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=body)
    route = respx.get(GH + "123").mock(
        return_value=(
            httpx.Response(404, json={"error": "missing"})
            if target == "failed"
            else httpx.Response(200, json=payload(body="Build software. Must be a US citizen."))
        )
    )
    async with open_repository(db_path) as repo:
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        rows = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(rows) == 1
        assert rows[0].feed_summary == "Hybrid position"
        if target == "greenhouse":
            assert rows[0].description == "Build software. Must be a US citizen."
            assert rows[0].work_auth_flag
        else:
            assert rows[0].description == ""
        assert isinstance(events[-1], SyncFinished)
    assert route.call_count == (0 if target == "unsupported" else 1)


@respx.mock
async def test_fresh_ambiguous_hanzili_row_can_be_hydrated_before_role_decision(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    body = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Systems Engineer Intern | Acme | Hybrid position | Company | Intern | "
        "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=body)
    route = respx.get(GH + "123").respond(
        200,
        json=payload(body="Develop backend software engineering systems."),
    )
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        rows = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(rows) == 1
        assert rows[0].description == "Develop backend software engineering systems."
        assert rows[0].feed_summary == "Hybrid position"
        assert not await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        from stage.services.maintenance import rescreen

        await rescreen(repo, now=NOW + timedelta(hours=1))
        rows = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(rows) == 1 and rows[0].feed_summary == "Hybrid position"
    assert route.call_count == 1


def ambiguous_hanzili_markdown(summary: str = "Hybrid position") -> str:
    return (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        f"| Systems Engineer Intern | Acme | {summary} | Company | Intern | "
        "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
    )


@respx.mock
async def test_pending_hanzili_retry_after_feed_304_keeps_summary_hidden_until_complete(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    feed_route = respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=ambiguous_hanzili_markdown(), headers={"ETag": '"first"'}),
            httpx.Response(304),
        ]
    )
    detail = respx.get(GH + "123").mock(
        side_effect=[
            httpx.Response(404, json={"error": "not found"}),
            httpx.Response(200, json=payload(body="Develop backend software engineering systems.")),
        ]
    )
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        pending = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(pending) == 1 and pending[0].feed_summary == "Hybrid position"
        assert "waiting for its job page" in pending[0].matched_phrase
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=2),
                close_orphans=False,
                still_open=False,
                force_refresh=True,
            )
        ]
        rows = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(rows) == 1 and rows[0].feed_summary == "Hybrid position"
        assert rows[0].description == "Develop backend software engineering systems."
        assert not await repo.list_quarantined(QuarantineFilters(source="hanzili"))
    assert detail.call_count == 2
    assert feed_route.calls[1].request.headers["if-none-match"] == '"first"'


@respx.mock
async def test_pending_hanzili_removed_from_feed_is_never_promoted(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=ambiguous_hanzili_markdown()),
            httpx.Response(200, text="_No roles currently available_"),
        ]
    )
    detail = respx.get(GH + "123").respond(404, json={"error": "not found"})
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        detail.respond(200, json=payload(body="Develop backend software engineering systems."))
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=2),
                close_orphans=False,
                still_open=False,
                force_refresh=True,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        pending = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(pending) == 1 and pending[0].feed_summary == "Hybrid position"
    assert detail.call_count == 1


@respx.mock
@pytest.mark.parametrize("change", ["membership", "summary", "source_failure"])
async def test_pending_hanzili_membership_change_during_fetch_blocks_promotion(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from stage.domain import JobFilters
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).respond(200, text=ambiguous_hanzili_markdown())

    def remove_during_fetch(request: httpx.Request) -> httpx.Response:
        with sqlite3.connect(db_path) as conn:
            if change == "membership":
                conn.execute(
                    "UPDATE detail_fetches SET resolved = 0 WHERE id LIKE ?",
                    (FEED_PENDING_PREFIX + "%",),
                )
            elif change == "summary":
                conn.execute(
                    "UPDATE quarantine SET feed_summary = ? WHERE source = ?",
                    ("Applicants must be enrolled in a PhD program.", "hanzili"),
                )
            else:
                conn.execute(
                    "UPDATE source_visits SET consecutive_failures = 1 WHERE source = ?",
                    ("hanzili",),
                )
        return httpx.Response(
            200, json=payload(body="Develop backend software engineering systems.")
        )

    route = respx.get(GH + "123").mock(side_effect=remove_during_fetch)
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        pending = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(pending) == 1
        assert pending[0].feed_summary == (
            "Applicants must be enrolled in a PhD program."
            if change == "summary"
            else "Hybrid position"
        )
        with sqlite3.connect(db_path) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM detail_fetches WHERE id = ?",
                    (FEED_REJECTION_PREFIX + pending[0].id,),
                ).fetchone()[0]
                == 0
            )
    assert route.call_count == 1


@respx.mock
async def test_ambiguous_hanzili_summary_phd_rejects_before_hydration(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).respond(
        200,
        text=ambiguous_hanzili_markdown(
            "Applicants must be enrolled in a PhD program.",
        ),
    )
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        rejected = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(rejected) == 1 and rejected[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        assert "PhD" in rejected[0].feed_summary
    assert route.call_count == 0


@respx.mock
async def test_removed_pending_hanzili_cannot_be_released_by_rules_rescreen(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.services import maintenance
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=ambiguous_hanzili_markdown()),
            httpx.Response(200, text="_No roles currently available_"),
        ]
    )
    respx.get(GH + "123").respond(404, json={"error": "not found"})
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=2),
                close_orphans=False,
                still_open=False,
                force_refresh=True,
            )
        ]
        summary = await repo.list_quarantined(
            QuarantineFilters(
                source="hanzili",
                reason=RejectionReason.UNKNOWN_CS_ROLE,
            )
        )
        assert len(summary) == 1

        def permissive(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            return normalize_batch(
                tuple(
                    replace(
                        entry,
                        title_raw="Software Engineer Intern",
                        title_normalized="Software Engineer Intern",
                    )
                    for entry in jobs
                )
            )

        monkeypatch.setattr("stage.services.screening.normalize_batch", permissive)
        result = await maintenance.rescreen(repo, now=NOW + timedelta(hours=3))
        assert result.released == 0
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(await repo.list_quarantined(QuarantineFilters(source="hanzili"))) == 1


def test_positive_role_evidence_cannot_override_a_hard_title_exclusion() -> None:
    from stage.services.screening import normalize_batch

    row = replace(
        job(),
        title_raw="Marketing Intern",
        title_normalized="Marketing Intern",
        description="Software engineering and computer science.",
        feed_summary="Hybrid position.",
    )
    kept, rejected = normalize_batch((row,))
    assert not kept and len(rejected) == 1


@respx.mock
async def test_pending_hanzili_is_not_promoted_when_feed_read_fails(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=ambiguous_hanzili_markdown()),
            httpx.Response(404),
        ]
    )
    detail = respx.get(GH + "123").respond(404, json={"error": "not found"})
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        detail.respond(200, json=payload(body="Develop backend software engineering systems."))
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=2),
                close_orphans=False,
                still_open=False,
                force_refresh=True,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        pending = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(pending) == 1 and pending[0].feed_summary == "Hybrid position"
    assert detail.call_count == 1


@respx.mock
async def test_pending_and_regular_feed_hydration_share_budget_fairly(db_path: Path) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX

    active = tuple(
        replace(
            job(number),
            id=f"hanzili:acme:{number}",
            source="hanzili",
            first_seen=NOW - timedelta(days=10),
        )
        for number in range(100, 140)
    )
    pending = tuple(
        QuarantinedJob(
            id=f"hanzili:acme:{number}",
            source="hanzili",
            company="Acme",
            title_raw="Systems Engineer Intern",
            reason=RejectionReason.UNKNOWN_CS_ROLE,
            matched_phrase="systems engineer, waiting for its job page",
            first_seen=NOW,
            last_seen=NOW,
            apply_url_raw=f"https://boards.greenhouse.io/acme/jobs/{number}",
            feed_summary="Hybrid position",
            employment_type="internship",
        )
        for number in range(200, 240)
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=active,
                quarantined=pending,
                detail_fetches=tuple(
                    DetailFetch(FEED_PENDING_PREFIX + entry.id, resolved=True) for entry in pending
                ),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )
        selected = await repo.feed_hydration_queue(["hanzili"], NOW, 30)
        assert sum(isinstance(entry, QuarantinedJob) for entry in selected) == 15
        assert sum(isinstance(entry, Job) for entry in selected) == 15
        route = respx.route(host="boards-api.greenhouse.io").respond(404)
        screener = Screener()
        try:
            result = await feed_hydration.hydrate_feeds(
                repo,
                ["hanzili"],
                screener,
                now=NOW,
                budgets={},
                postures={},
                rng=random.Random(1),
            )
        finally:
            screener.close()
        assert route.call_count == len(result.requests) == 30
        with sqlite3.connect(db_path) as conn:
            attempted = {
                row[0]
                for row in conn.execute(
                    "SELECT id FROM detail_fetches WHERE id LIKE ?",
                    (FEED_DETAIL_PREFIX + "%",),
                )
            }
        assert sum(FEED_DETAIL_PREFIX + entry.id in attempted for entry in active) == 15
        assert sum(FEED_DETAIL_PREFIX + entry.id in attempted for entry in pending) == 15


@respx.mock
@pytest.mark.parametrize(
    "body", ["Electrical hardware testing.", "Applicants must be enrolled in a PhD program."]
)
async def test_pending_hanzili_detail_rejection_stays_hidden_and_stores_full_evidence(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).respond(200, text=ambiguous_hanzili_markdown())
    route = respx.get(GH + "123").respond(200, json=payload(body=body))
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        rejected = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(rejected) == 1
        assert rejected[0].verified_description == body
        assert rejected[0].feed_summary == "Hybrid position"
        with sqlite3.connect(db_path) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM detail_fetches WHERE id = ?",
                    (FEED_REJECTION_PREFIX + rejected[0].id,),
                ).fetchone()[0]
                == 1
            )
        await hydrate(repo, NOW + timedelta(hours=3))
    assert route.call_count == 1


@respx.mock
async def test_pending_hanzili_retains_summary_restriction_after_hydration_and_rescreen(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.services.maintenance import rescreen
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    summary = "Hybrid position. Must be a US citizen."
    respx.get(feed.plan(NOW)[0]).respond(200, text=ambiguous_hanzili_markdown(summary))
    respx.get(GH + "123").respond(
        200, json=payload(body="Develop backend software engineering systems.")
    )
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        await rescreen(repo, now=NOW + timedelta(hours=1))
        rows = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(rows) == 1 and rows[0].work_auth_flag
        assert rows[0].feed_summary == summary
        assert rows[0].description == "Develop backend software engineering systems."


@respx.mock
async def test_existing_unrelated_quarantine_is_not_treated_as_fresh_pending(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    markdown = ambiguous_hanzili_markdown()
    parsed, _ = feed.rows(markdown)
    candidate = feed._to_job(parsed[0], NOW)
    old = QuarantinedJob(
        id=candidate.id,
        source="hanzili",
        company=candidate.company,
        title_raw=candidate.title_raw,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="phd required",
        first_seen=NOW - timedelta(days=10),
        last_seen=NOW,
        apply_url_raw=candidate.apply_url_raw,
        location_raw=candidate.location_raw,
        feed_summary="Applicants must be enrolled in a PhD program.",
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=markdown)
    route = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(source="hanzili", run_started_at=NOW, quarantined=(old,))
        )
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert await repo.get_job(old.id) is None
    assert route.call_count == 0


@respx.mock
async def test_pending_hanzili_failed_detail_attempts_remain_bounded(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).respond(200, text=ambiguous_hanzili_markdown())
    detail = respx.get(GH + "123").respond(404, json={"error": "missing"})

    def now_at(moment: datetime) -> Callable[[], datetime]:
        def current() -> datetime:
            return moment

        return current

    async with open_repository(db_path) as repo:
        for hours in (0, 2, 4, 6, 8):
            moment = NOW + timedelta(hours=hours)
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=now_at(moment),
                    close_orphans=False,
                    still_open=False,
                )
            ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        pending = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(pending) == 1 and pending[0].feed_summary == "Hybrid position"
        with sqlite3.connect(db_path) as conn:
            outcome = conn.execute(
                "SELECT resolved, failed, attempts FROM detail_fetches WHERE id = ?",
                (FEED_DETAIL_PREFIX + pending[0].id,),
            ).fetchone()
        assert outcome == (0, 1, 3)
    assert detail.call_count == 3


@respx.mock
async def test_unverified_hanzili_summary_race_defers_feed_and_preserves_newer_evidence(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    markdown = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Software Engineer Intern | Acme | Hybrid position | Company | Intern | "
        "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
    )
    feed_route = respx.get(feed.plan(NOW)[0]).respond(200, text=markdown, headers={"ETag": '"v1"'})
    respx.get(GH + "123").respond(404, json={"error": "missing"})
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        initial = (await repo.list_jobs(JobFilters(source="hanzili", limit=None)))[0]
        changed = False

        def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            result = normalize_batch(jobs)
            if not changed:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE jobs SET feed_summary = ?, work_auth_flag = 1, "
                        "work_auth_known = 1 WHERE id = ?",
                        ("Must be a US citizen.", initial.id),
                    )
                changed = True
            return result

        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=6),
                close_orphans=False,
                still_open=False,
                force_refresh=True,
            )
        ]
        stored = await repo.get_job(initial.id)
        assert stored is not None and stored.feed_summary == "Must be a US citizen."
        assert stored.work_auth_flag and stored.description == ""
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert not await repo.load_validators("hanzili")
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW + timedelta(hours=6, minutes=1),
                close_orphans=False,
                still_open=False,
            )
        ]
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "success"
        assert "if-none-match" not in feed_route.calls[-1].request.headers


@respx.mock
async def test_concurrent_quarantine_is_not_replaced_by_fresh_pending_hanzili(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    markdown = ambiguous_hanzili_markdown()
    parsed, _ = feed.rows(markdown)
    identity = feed._to_job(parsed[0], NOW).id
    respx.get(feed.plan(NOW)[0]).respond(200, text=markdown, headers={"ETag": '"latest"'})
    route = respx.get(GH + "123").respond(200, json=payload())
    changed = False

    def concurrent_screen(
        jobs: Sequence[Job],
    ) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
        nonlocal changed
        result = normalize_batch(jobs)
        if not changed:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO quarantine (id, source, company, title_raw, reason, "
                    "matched_phrase, "
                    "first_seen, last_seen, feed_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identity,
                        "hanzili",
                        "Acme",
                        "Systems Engineer Intern",
                        "out-of-scope-degree",
                        "phd required",
                        NOW.isoformat(),
                        NOW.isoformat(),
                        "PhD required.",
                    ),
                )
            changed = True
        return result

    async with open_repository(db_path) as repo:
        monkeypatch.setattr("stage.services.screening.normalize_batch", concurrent_screen)
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: NOW,
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        rejected = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(rejected) == 1 and rejected[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        assert rejected[0].feed_summary == "PhD required."
        assert not await repo.load_validators("hanzili")
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("target", ["greenhouse", "unsupported"])
async def test_hanzili_rejected_summary_survives_later_blank_feed_refresh(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    apply_url = (
        "https://boards.greenhouse.io/acme/jobs/123"
        if target == "greenhouse"
        else "https://unknown.example/job"
    )

    def page(summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| Software Engineer Intern | Acme | {summary} | Company | Intern | "
            f"Toronto, ON | [Apply]({apply_url}) |\n"
        )

    summary = "Applicants must be enrolled in a PhD program."
    feed_route = respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page(summary)),
            httpx.Response(200, text=page("")),
            httpx.Response(200, text=page("")),
        ]
    )
    detail = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        for hours in (0, 6, 12):
            moment = NOW + timedelta(hours=hours)

            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(moment),
                    close_orphans=False,
                    still_open=False,
                )
            ]
            assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
            entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
            assert len(entries) == 1 and entries[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
            assert entries[0].feed_summary == summary
    assert feed_route.call_count == 3
    assert detail.call_count == 0


@respx.mock
@pytest.mark.parametrize("target", ["muse", "greenhouse"])
async def test_muse_full_contents_remain_full_on_ingest_and_refresh(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    from stage.domain import JobFilters
    from stage.sources.themuse import TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})
    apply_url = (
        "https://www.themuse.com/jobs/123"
        if target == "muse"
        else "https://boards.greenhouse.io/acme/jobs/123"
    )
    body = {
        "results": [
            {
                "id": 123,
                "name": "Software Engineer Intern",
                "company": {"name": "Acme"},
                "locations": [{"name": "Toronto, Canada"}],
                "refs": {"landing_page": apply_url},
                "contents": "<p>Build software systems.</p><p>Must be a US citizen.</p>",
                "categories": [{"name": "Software Engineering"}],
            }
        ],
        "page_count": 1,
    }
    route = respx.get(url__startswith="https://www.themuse.com/api/public/jobs").respond(
        200, json=body
    )
    detail = respx.get(GH + "123").respond(200, json=payload())
    async with open_repository(db_path) as repo:
        for hours in (0, 6):
            moment = NOW + timedelta(hours=hours)

            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["themuse"],
                    now_fn=clock_at(moment),
                    close_orphans=False,
                    still_open=False,
                )
            ]
            rows = await repo.list_jobs(JobFilters(source="themuse", limit=None))
            assert len(rows) == 1
            assert rows[0].description == "Build software systems.\nMust be a US citizen."
            assert rows[0].feed_summary == "" and rows[0].work_auth_flag
    assert route.call_count == 2
    assert detail.call_count == 0


@respx.mock
async def test_blank_hanzili_refresh_retries_if_quarantine_summary_changes_concurrently(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    url = "https://boards.greenhouse.io/acme/jobs/123"

    def page(summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| Software Engineer Intern | Acme | {summary} | Company | Intern | "
            f"Toronto, ON | [Apply]({url}) |\n"
        )

    first_summary = "Applicants must be enrolled in a PhD program."
    updated_summary = "PhD required. Must be a US citizen."
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, text=page(first_summary), headers={"ETag": '"v1"'})
        if request.headers.get("if-none-match") == '"v2"':
            return httpx.Response(304)
        return httpx.Response(200, text=page(""), headers={"ETag": '"v2"'})

    respx.get(feed.plan(NOW)[0]).mock(side_effect=answer)
    detail = respx.get(GH + "123").respond(200, json=payload())
    changed = False

    def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
        nonlocal changed
        result = normalize_batch(jobs)
        if not changed:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE quarantine SET feed_summary = ? WHERE source = ?",
                    (
                        updated_summary,
                        "hanzili",
                    ),
                )
            changed = True
        return result

    async def run(repo: AsyncRepository, moment: datetime) -> list[object]:
        return [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=lambda: moment,
                close_orphans=False,
                still_open=False,
            )
        ]

    async with open_repository(db_path) as repo:
        await run(repo, NOW)
        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = await run(repo, NOW + timedelta(hours=6))
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        held = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(held) == 1 and held[0].feed_summary == updated_summary
        assert not await repo.load_validators("hanzili")
        events = await run(repo, NOW + timedelta(hours=6, minutes=1))
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "success"
        assert "if-none-match" not in requests[-1].headers
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        held = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(held) == 1 and held[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        assert held[0].feed_summary == updated_summary
    assert detail.call_count == 0


@respx.mock
async def test_muse_fresh_full_body_replaces_older_hydrated_body(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.domain import DetailFetch, JobFilters
    from stage.sources.themuse import MuseListing, TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})
    raw = {
        "id": 123,
        "name": "Software Engineer Intern",
        "company": {"name": "Acme"},
        "locations": [{"name": "Toronto, Canada"}],
        "refs": {"landing_page": "https://boards.greenhouse.io/acme/jobs/123"},
        "contents": "<p>New Muse full posting.</p><p>Must be a US citizen.</p>",
        "categories": [{"name": "Software Engineering"}],
    }
    respx.get(url__startswith="https://www.themuse.com/api/public/jobs").respond(
        200,
        json={"results": [raw], "page_count": 1},
    )
    async with open_repository(db_path) as repo:
        source_row = feed._to_job(MuseListing.model_validate(raw), NOW)
        await repo.apply_source_batch(
            SourceBatch(
                source="themuse",
                run_started_at=NOW,
                jobs=(replace(source_row, description=""),),
            )
        )
        snapshot = await repo.get_job(source_row.id)
        assert snapshot is not None
        await repo.apply_source_batch(
            SourceBatch(
                source="themuse",
                run_started_at=NOW,
                jobs=(replace(snapshot, description="Old hydrated body."),),
                hydration_expected=(snapshot,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + snapshot.id, resolved=True),),
            )
        )
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["themuse"],
                now_fn=lambda: NOW + timedelta(hours=6),
                close_orphans=False,
                still_open=False,
            )
        ]
        rows = await repo.list_jobs(JobFilters(source="themuse", limit=None))
        assert len(rows) == 1
        assert rows[0].description == "New Muse full posting.\nMust be a US citizen."
        assert rows[0].feed_summary == "" and rows[0].work_auth_flag
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "success"


@respx.mock
async def test_rejected_hanzili_keeps_old_evidence_and_current_summary(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.services.maintenance import rescreen
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| Software Engineer Intern | Acme | {summary} | Company | Intern | "
            "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
        )

    original = "Applicants must be enrolled in a PhD program."
    current = "Hybrid position."
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page(original)),
            httpx.Response(200, text=page(current)),
            httpx.Response(200, text=page("")),
        ]
    )
    detail = respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:
        for hours in (0, 6, 12):
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]
            assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
            entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
            assert len(entries) == 1 and entries[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
            assert entries[0].rejection_evidence == original
            assert entries[0].feed_summary == (original if hours == 0 else current)
        await rescreen(repo, now=NOW + timedelta(hours=13))
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert (await repo.purge(NOW + timedelta(days=101))).forgotten == 0
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 1 and entries[0].rejection_evidence == original
    assert detail.call_count == 0


@respx.mock
async def test_rejected_muse_body_survives_blank_refresh_and_fresh_body_can_recover(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.services.maintenance import rescreen
    from stage.sources.themuse import TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})

    def listing(contents: str) -> dict[str, object]:
        return {
            "results": [
                {
                    "id": 123,
                    "name": "Software Engineer Intern",
                    "company": {"name": "Acme"},
                    "locations": [{"name": "Toronto, Canada"}],
                    "refs": {"landing_page": "https://www.themuse.com/jobs/123"},
                    "contents": contents,
                    "categories": [{"name": "Software Engineering"}],
                }
            ],
            "page_count": 1,
        }

    body = "Applicants must be enrolled in a PhD program."
    respx.get(url__startswith="https://www.themuse.com/api/public/jobs").mock(
        side_effect=[
            httpx.Response(200, json=listing(f"<p>{body}</p>")),
            httpx.Response(200, json=listing("")),
            httpx.Response(200, json=listing("<p>Build software systems.</p>")),
        ]
    )
    async with open_repository(db_path) as repo:

        async def run(hours: int) -> None:
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["themuse"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]

        for hours in (0, 6):
            await run(hours)
            assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
            entries = await repo.list_quarantined(QuarantineFilters(source="themuse"))
            assert len(entries) == 1 and entries[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
            assert entries[0].verified_description == body
            assert entries[0].rejection_evidence == body
        await rescreen(repo, now=NOW + timedelta(hours=7))
        assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
        await run(12)
        jobs = await repo.list_jobs(JobFilters(source="themuse", limit=None))
        assert len(jobs) == 1 and jobs[0].description == "Build software systems."
        assert not await repo.list_quarantined(QuarantineFilters(source="themuse"))


@respx.mock
async def test_changed_summary_race_does_not_overwrite_quarantine_or_record_success(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| Software Engineer Intern | Acme | {summary} | Company | Intern | "
            "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
        )

    original = "Applicants must be enrolled in a PhD program."
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page(original), headers={"ETag": '"v1"'}),
            httpx.Response(200, text=page("Hybrid position."), headers={"ETag": '"v2"'}),
        ]
    )
    respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:

        async def run(hours: int) -> list[object]:
            return [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]

        await run(0)
        changed = False

        def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            result = normalize_batch(jobs)
            if not changed:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE quarantine SET rejection_evidence = ? WHERE source = ?",
                        ("Newer independent evidence", "hanzili"),
                    )
                changed = True
            return result

        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = await run(6)
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 1 and entries[0].rejection_evidence == "Newer independent evidence"
        assert not await repo.load_validators("hanzili")


async def test_independent_rejection_evidence_prevents_duplicate_restoration(db_path: Path) -> None:
    from stage.domain import RejectionReason

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
        rejection_evidence="Independently requires a PhD.",
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="test-feed",
                run_started_at=NOW,
                quarantined=(held,),
            )
        )
        await seed(repo, job())
        assert await repo.get_job(held.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="test-feed"))
        assert len(entries) == 1 and entries[0].rejection_evidence == held.rejection_evidence
        assert (await repo.purge(NOW + timedelta(days=100))).forgotten == 0


@respx.mock
async def test_muse_blank_refresh_race_preserves_newer_full_evidence(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.services.screening import normalize_batch
    from stage.sources.themuse import TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})

    def listing(contents: str) -> dict[str, object]:
        return {
            "results": [
                {
                    "id": 123,
                    "name": "Software Engineer Intern",
                    "company": {"name": "Acme"},
                    "locations": [{"name": "Toronto, Canada"}],
                    "refs": {"landing_page": "https://www.themuse.com/jobs/123"},
                    "contents": contents,
                    "categories": [{"name": "Software Engineering"}],
                }
            ],
            "page_count": 1,
        }

    respx.get(url__startswith="https://www.themuse.com/api/public/jobs").mock(
        side_effect=[
            httpx.Response(200, json=listing("<p>PhD required.</p>"), headers={"ETag": '"v1"'}),
            httpx.Response(200, json=listing(""), headers={"ETag": '"v2"'}),
        ]
    )
    async with open_repository(db_path) as repo:

        async def run(hours: int) -> list[object]:
            return [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["themuse"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]

        await run(0)
        changed = False

        def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            result = normalize_batch(jobs)
            if not changed:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE quarantine SET verified_description = ? WHERE source = ?",
                        ("Newer PhD required body.", "themuse"),
                    )
                changed = True
            return result

        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = await run(6)
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="themuse"))
        assert len(entries) == 1 and entries[0].verified_description == "Newer PhD required body."
        assert entries[0].rejection_evidence == "PhD required."
        assert not await repo.load_validators("themuse")


@respx.mock
async def test_hanzili_rejection_follows_same_url_when_title_changes(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(title: str, summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| {title} | Acme | {summary} | Company | Intern | "
            "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
        )

    original = "Applicants must be enrolled in a PhD program."
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page("Software Engineer Intern", original)),
            httpx.Response(200, text=page("Software Developer Intern", "Hybrid position.")),
        ]
    )
    detail = respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:
        for hours in (0, 6):
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        assert {entry.rejection_evidence for entry in entries} == {original}
        assert {entry.feed_summary for entry in entries} == {original, "Hybrid position."}
    assert detail.call_count == 0


@respx.mock
async def test_hanzili_new_matching_quarantine_during_screening_defers_feed(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.services.screening import normalize_batch
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    page = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Software Engineer Intern | Acme | Hybrid position. | Company | Intern | "
        "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123?utm_source=new) |\n"
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=page, headers={"ETag": '"v1"'})
    respx.get(GH + "123").respond(503)
    changed = False

    def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
        nonlocal changed
        result = normalize_batch(jobs)
        if not changed:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO quarantine (id, source, company, title_raw, apply_url_raw, "
                    "reason, matched_phrase, first_seen, last_seen, rejection_evidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "hanzili:older",
                        "hanzili",
                        "Acme",
                        "Developer Intern",
                        "https://boards.greenhouse.io/acme/jobs/123?utm_source=old",
                        "out-of-scope-degree",
                        "PhD",
                        NOW.isoformat(),
                        NOW.isoformat(),
                        "PhD required.",
                    ),
                )
            changed = True
        return result

    monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
    async with open_repository(db_path) as repo:
        events = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=clock_at(NOW),
                close_orphans=False,
                still_open=False,
            )
        ]
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert not await repo.load_validators("hanzili")


@respx.mock
async def test_hanzili_hydrated_rejection_follows_same_url_when_title_changes(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(title: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| {title} | Acme | Hybrid position. | Company | Intern | "
            "Toronto, ON | [Apply](https://boards.greenhouse.io/acme/jobs/123) |\n"
        )

    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page("Software Engineer Intern")),
            httpx.Response(200, text=page("Software Developer Intern")),
        ]
    )
    detail = respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:
        for hours in (0, 6):
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        assert all("PhD" in entry.rejection_evidence for entry in entries)
    assert detail.call_count == 1


@respx.mock
async def test_hanzili_tracking_query_variants_retain_same_posting_hold(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(query: str, summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| Software Engineer Intern | Acme | {summary} | Company | Intern | Toronto, ON | "
            f"[Apply](https://boards.greenhouse.io/acme/jobs/123?{query}) |\n"
        )

    evidence = "Applicants must be enrolled in a PhD program."
    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page("utm_source=first", evidence)),
            httpx.Response(200, text=page("utm_source=second", "Hybrid position.")),
        ]
    )
    detail = respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:
        for hours in (0, 6):
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        assert {entry.rejection_evidence for entry in entries} == {evidence}
    assert detail.call_count == 0


@respx.mock
async def test_hanzili_neutral_quarantine_does_not_mask_matching_rejection(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    page = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Software Engineer Intern | Acme | Hybrid position. | Company | Intern | Toronto, ON | "
        "[Apply](https://boards.greenhouse.io/acme/jobs/123?utm_source=new) |\n"
    )
    rows, _ = feed.rows(page)
    incoming = feed._to_job(rows[0], NOW)
    neutral = QuarantinedJob(
        id=incoming.id,
        source="hanzili",
        company="Acme",
        title_raw=incoming.title_raw,
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw=incoming.apply_url_raw,
        location_raw=incoming.location_raw,
    )
    held = replace(
        neutral,
        id="hanzili:prior",
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="PhD",
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=old",
        rejection_evidence="PhD required.",
        feed_summary="PhD required.",
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=page)
    respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=(neutral, held),
            )
        )
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=clock_at(NOW + timedelta(hours=6)),
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        current = next(entry for entry in entries if entry.id == incoming.id)
        assert current.rejection_evidence == "PhD required."
        assert current.feed_summary == "Hybrid position."


@respx.mock
async def test_hanzili_shared_generic_careers_url_does_not_transfer_hold(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})

    def page(title: str, summary: str) -> str:
        return (
            "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| {title} | Acme | {summary} | Company | Intern | Toronto, ON | "
            "[Apply](https://employer.example/careers) |\n"
        )

    respx.get(feed.plan(NOW)[0]).mock(
        side_effect=[
            httpx.Response(200, text=page("Software Engineer Intern", "PhD required.")),
            httpx.Response(200, text=page("Software Developer Intern", "Hybrid position.")),
        ]
    )
    async with open_repository(db_path) as repo:
        for hours in (0, 6):
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["hanzili"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 1 and entries[0].rejection_evidence == "PhD required."
        jobs = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        assert len(jobs) == 1 and jobs[0].title_raw == "Software Developer Intern"


@respx.mock
async def test_pending_hydration_cannot_promote_after_matching_hold_arrives_during_fetch(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    respx.get(feed.plan(NOW)[0]).respond(200, text=ambiguous_hanzili_markdown())
    evidence = "Applicants must be enrolled in a PhD program."

    def hold_during_fetch(request: httpx.Request) -> httpx.Response:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO quarantine (id, source, company, title_raw, apply_url_raw, reason, "
                "matched_phrase, first_seen, last_seen, verified_description, rejection_evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "hanzili:concurrent",
                    "hanzili",
                    "Acme",
                    "Systems Engineer Intern",
                    "https://job-boards.greenhouse.io/acme/jobs/123?utm_source=other",
                    "out-of-scope-degree",
                    "PhD",
                    NOW.isoformat(),
                    NOW.isoformat(),
                    evidence,
                    evidence,
                ),
            )
        return httpx.Response(
            200, json=payload(body="Develop backend software engineering systems.")
        )

    detail = respx.get(GH + "123").mock(side_effect=hold_during_fetch)
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=clock_at(NOW),
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        prior = next(entry for entry in entries if entry.id == "hanzili:concurrent")
        pending = next(entry for entry in entries if entry.id != prior.id)
        assert prior.reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        assert prior.rejection_evidence == evidence
        assert pending.reason is RejectionReason.UNKNOWN_CS_ROLE
        with sqlite3.connect(db_path) as conn:
            assert conn.execute(
                "SELECT resolved, failed FROM detail_fetches WHERE id = ?",
                (FEED_DETAIL_PREFIX + pending.id,),
            ).fetchone() == (0, 1)
    assert detail.call_count == 1


@respx.mock
async def test_pending_hydration_same_batch_rejection_blocks_matching_promotion(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    page = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Systems Engineer Intern | Acme | Hybrid position | Company | Intern | Toronto, ON | "
        "[Apply](https://boards.greenhouse.io/acme/jobs/123?utm_source=one) |\n"
        "| Systems Engineer Intern | Acme | Hybrid position | Company | Intern | New York, NY | "
        "[Apply](https://boards.greenhouse.io/acme/jobs/123?utm_source=two) |\n"
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=page)
    detail = respx.get(GH + "123").mock(
        side_effect=[
            httpx.Response(200, json=payload(body="Develop backend software engineering systems.")),
            httpx.Response(200, json=payload(body="Applicants must be enrolled in a PhD program.")),
        ]
    )
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=clock_at(NOW),
                close_orphans=False,
                still_open=False,
            )
        ]
        assert not await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        held = next(
            entry for entry in entries if entry.reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        )
        pending = next(entry for entry in entries if entry.id != held.id)
        assert "PhD" in held.rejection_evidence
        with sqlite3.connect(db_path) as conn:
            assert conn.execute(
                "SELECT resolved, failed FROM detail_fetches WHERE id = ?",
                (FEED_DETAIL_PREFIX + pending.id,),
            ).fetchone() == (0, 1)
    assert detail.call_count == 2


@pytest.mark.parametrize("same_batch", [False, True])
async def test_pending_identity_guard_does_not_link_generic_careers_urls(
    db_path: Path, same_batch: bool
) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.storage.copies import as_job

    pending = QuarantinedJob(
        id="hanzili:pending",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://employer.example/careers",
        feed_summary="Hybrid position",
    )
    held = replace(
        pending,
        id="hanzili:other",
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="PhD",
        rejection_evidence="PhD required.",
    )
    candidate = replace(as_job(pending), description="Develop backend software systems.")
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=(pending,) if same_batch else (pending, held),
                detail_fetches=(DetailFetch(FEED_PENDING_PREFIX + pending.id, resolved=True),),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )
        if same_batch:
            await repo.apply_source_batch(
                SourceBatch(
                    source="hanzili",
                    run_started_at=NOW,
                    jobs=(replace(as_job(held), description=""),),
                )
            )
            other = await repo.get_job(held.id)
            assert other is not None
        else:
            other = None
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(candidate,),
                quarantined=(held,) if same_batch else (),
                hydration_expected=(other,) if other is not None else (),
                pending_expected=(pending,),
                rejection_screened=(candidate,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + pending.id, resolved=True),)
                + (
                    (DetailFetch(FEED_DETAIL_PREFIX + held.id, resolved=True),)
                    if same_batch
                    else ()
                ),
            )
        )
        assert await repo.get_job(pending.id) is not None
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 1 and entries[0].id == held.id


@pytest.mark.parametrize("marker", ["evidence", "verified", "failed"])
async def test_pending_identity_guard_distinguishes_verified_hold_from_failed_marker(
    db_path: Path, marker: str
) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.storage.copies import as_job

    pending = QuarantinedJob(
        id="hanzili:pending",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=one",
        feed_summary="Hybrid position",
    )
    other = replace(
        pending,
        id="hanzili:other",
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="PhD",
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/123",
        rejection_evidence="PhD required." if marker == "evidence" else "",
    )
    original = replace(as_job(pending), description="Develop backend software systems.")
    mark = (
        DetailFetch(FEED_REJECTION_PREFIX + other.id, resolved=True)
        if marker == "verified"
        else DetailFetch(FEED_DETAIL_PREFIX + other.id, failed=True)
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=(pending, other),
                detail_fetches=(DetailFetch(FEED_PENDING_PREFIX + pending.id, resolved=True), mark),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(original,),
                hydration_expected=(),
                pending_expected=(pending,),
                rejection_screened=(original,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + pending.id, resolved=True),),
            )
        )
        assert (await repo.get_job(pending.id) is not None) is (marker == "failed")
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert any(entry.id == other.id for entry in entries)
        with sqlite3.connect(db_path) as conn:
            result = conn.execute(
                "SELECT resolved, failed FROM detail_fetches WHERE id = ?",
                (FEED_DETAIL_PREFIX + pending.id,),
            ).fetchone()
        assert result == ((1, 0) if marker == "failed" else (0, 1))


async def test_later_hydration_rejection_retracts_already_promoted_matching_pending(
    db_path: Path,
) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.storage.copies import as_job

    pending = QuarantinedJob(
        id="hanzili:first",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=one",
        feed_summary="Hybrid position",
    )
    sibling = replace(
        pending,
        id="hanzili:second",
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/123?utm_source=two",
    )
    accepted = replace(as_job(pending), description="Develop backend software systems.")
    phd_text = "Applicants must be enrolled in a PhD program."
    rejected_input = replace(as_job(sibling), description=phd_text)
    rejected = replace(
        sibling,
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="PhD",
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=(pending, sibling),
                detail_fetches=tuple(
                    DetailFetch(FEED_PENDING_PREFIX + entry.id, resolved=True)
                    for entry in (pending, sibling)
                ),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(accepted,),
                hydration_expected=(),
                pending_expected=(pending,),
                rejection_screened=(accepted,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + pending.id, resolved=True),),
            )
        )
        assert await repo.get_job(pending.id) is not None
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=(rejected,),
                hydration_expected=(),
                pending_expected=(sibling,),
                rejection_screened=(rejected_input,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + sibling.id, resolved=True),),
            )
        )
        assert await repo.get_job(pending.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        previous = next(entry for entry in entries if entry.id == pending.id)
        assert previous.rejection_evidence == phd_text
        assert previous.verified_description == accepted.description
        assert previous.feed_summary == pending.feed_summary
        assert previous.reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        other = next(entry for entry in entries if entry.id == sibling.id)
        assert other.rejection_evidence == phd_text


async def test_same_batch_rejections_keep_each_postings_own_evidence(db_path: Path) -> None:
    from stage.domain import DetailFetch, RejectionReason
    from stage.domain.visits import FEED_DECISION_PREFIX
    from stage.storage.copies import as_job

    first = QuarantinedJob(
        id="hanzili:first",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=one",
        feed_summary="Hybrid position",
    )
    second = replace(
        first,
        id="hanzili:second",
        apply_url_raw=("https://job-boards.greenhouse.io/acme/jobs/123?utm_source=two"),
    )
    bodies = ("PhD required in systems.", "PhD required in computing.")
    candidates = tuple(
        replace(as_job(entry), description=body)
        for entry, body in zip((first, second), bodies, strict=True)
    )
    rejected = tuple(
        replace(entry, reason=RejectionReason.OUT_OF_SCOPE_DEGREE, matched_phrase="PhD")
        for entry in (first, second)
    )
    active = tuple(
        replace(as_job(entry), description="Previously accepted full software detail.")
        for entry in (first, second)
    )
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=active,
                detail_fetches=tuple(
                    DetailFetch(prefix + entry.id, resolved=True)
                    for entry in active
                    for prefix in (FEED_DETAIL_PREFIX, FEED_DECISION_PREFIX)
                ),
            )
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                quarantined=rejected,
                hydration_expected=active,
                rejection_screened=candidates,
                detail_fetches=tuple(
                    DetailFetch(FEED_DETAIL_PREFIX + entry.id, resolved=True)
                    for entry in (first, second)
                ),
            )
        )
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert {entry.id: entry.rejection_evidence for entry in entries} == dict(
            zip((first.id, second.id), bodies, strict=True)
        )


@pytest.mark.parametrize("reject_first", [True, False])
async def test_active_hydration_and_matching_rejection_are_held_in_either_order(
    db_path: Path, reject_first: bool
) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.storage.copies import as_job

    active = replace(
        job(),
        id="hanzili:active",
        source="hanzili",
        description="Old unverified listing",
        feed_summary="Hybrid position",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=active",
    )
    pending = QuarantinedJob(
        id="hanzili:pending",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/123?utm_source=pending",
        feed_summary="Hybrid position",
    )
    accepted = replace(active, description="Develop backend software systems.")
    phd = "Applicants must be enrolled in a PhD program."
    rejected_input = replace(as_job(pending), description=phd)
    rejected = replace(pending, reason=RejectionReason.OUT_OF_SCOPE_DEGREE, matched_phrase="PhD")
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(active,),
                quarantined=(pending,),
                detail_fetches=(DetailFetch(FEED_PENDING_PREFIX + pending.id, resolved=True),),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )

        async def accept() -> None:
            await repo.apply_source_batch(
                SourceBatch(
                    source="hanzili",
                    run_started_at=NOW,
                    jobs=(accepted,),
                    hydration_expected=(active,),
                    rejection_screened=(accepted,),
                    detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + active.id, resolved=True),),
                )
            )

        async def reject() -> None:
            await repo.apply_source_batch(
                SourceBatch(
                    source="hanzili",
                    run_started_at=NOW,
                    quarantined=(rejected,),
                    hydration_expected=(),
                    pending_expected=(pending,),
                    rejection_screened=(rejected_input,),
                    detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + pending.id, resolved=True),),
                )
            )

        if reject_first:
            await reject()
            await accept()
        else:
            await accept()
            await reject()
        assert await repo.get_job(active.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(entries) == 2
        held = next(entry for entry in entries if entry.id == active.id)
        assert held.rejection_evidence == phd and held.feed_summary == "Hybrid position"
        if reject_first:
            assert held.unverified_description == "Old unverified listing"
            assert held.verified_description == ""
        else:
            assert held.verified_description == accepted.description
        assert any(entry.id == pending.id and entry.rejection_evidence == phd for entry in entries)
        from stage.services.maintenance import rescreen

        await rescreen(repo, now=NOW + timedelta(hours=1))
        await repo.purge(NOW + timedelta(days=101))
        persisted = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        assert len(persisted) == 2
        if reject_first:
            assert next(
                entry for entry in persisted if entry.id == active.id
            ).unverified_description == ("Old unverified listing")


@respx.mock
@pytest.mark.parametrize("generic", [False, True])
async def test_new_same_feed_rows_share_only_verified_posting_rejections(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, generic: bool
) -> None:
    from stage.domain import JobFilters
    from stage.sources.curated_markdown import HanziliFeed

    feed = HanziliFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"hanzili": feed})
    url = (
        "https://employer.example/careers"
        if generic
        else "https://boards.greenhouse.io/acme/jobs/123"
    )
    page = (
        "| Title | Company | Role | Company Info | Details | Location | Apply |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| Software Engineer Intern | Acme | Hybrid position. | Company | Intern | Toronto, ON | "
        f"[Apply]({url}?utm_source=a) |\n"
        "| Software Engineer Intern | Acme | PhD required. | Company | Intern | New York, NY | "
        f"[Apply]({url}?utm_source=b) |\n"
    )
    respx.get(feed.plan(NOW)[0]).respond(200, text=page)
    detail = respx.get(GH + "123").respond(503)
    async with open_repository(db_path) as repo:
        _ = [
            event
            async for event in sync_module.sync(
                repo,
                [],
                sources=["hanzili"],
                now_fn=clock_at(NOW),
                close_orphans=False,
                still_open=False,
            )
        ]
        jobs = await repo.list_jobs(JobFilters(source="hanzili", limit=None))
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        if generic:
            assert len(jobs) == 1 and len(entries) == 1
        else:
            assert not jobs and len(entries) == 2
            assert all(entry.rejection_evidence == "PhD required." for entry in entries)
            assert {entry.feed_summary for entry in entries} == {
                "Hybrid position.",
                "PhD required.",
            }
            assert detail.call_count == 0


@pytest.mark.parametrize("order", ["same_batch", "rejection_first", "stale_first"])
async def test_stale_hydration_evidence_never_replaces_newer_verified_body(
    db_path: Path, order: str
) -> None:
    from stage.domain import CompanyVisit, DetailFetch, RejectionReason
    from stage.domain.visits import FEED_PENDING_PREFIX
    from stage.storage.copies import as_job

    original = replace(
        job(),
        id="hanzili:active",
        source="hanzili",
        description="",
        feed_summary="Hybrid position",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123?utm_source=active",
    )
    pending = QuarantinedJob(
        id="hanzili:pending",
        source="hanzili",
        company="Acme",
        title_raw="Systems Engineer Intern",
        reason=RejectionReason.UNKNOWN_CS_ROLE,
        matched_phrase="systems engineer, waiting for its job page",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/123?utm_source=pending",
        feed_summary="Hybrid position",
    )
    newer = replace(original, description="Newest verified software engineering detail.")
    stale = replace(original, description="Stale software engineering detail.")
    phd = "Applicants must be enrolled in a PhD program."
    rejected_input = replace(as_job(pending), description=phd)
    rejected = replace(pending, reason=RejectionReason.OUT_OF_SCOPE_DEGREE, matched_phrase="PhD")
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(original,),
                quarantined=(pending,),
                detail_fetches=(DetailFetch(FEED_PENDING_PREFIX + pending.id, resolved=True),),
                visits=(CompanyVisit(board="hanzili", succeeded=True),),
            )
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="hanzili",
                run_started_at=NOW,
                jobs=(newer,),
                hydration_expected=(original,),
                rejection_screened=(newer,),
                detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + original.id, resolved=True),),
            )
        )
        assert (await repo.get_job(original.id)) == newer
        stale_batch = SourceBatch(
            source="hanzili",
            run_started_at=NOW,
            jobs=(stale,),
            hydration_expected=(original,),
            rejection_screened=(stale,),
            detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + original.id, resolved=True),),
        )
        reject_batch = SourceBatch(
            source="hanzili",
            run_started_at=NOW,
            quarantined=(rejected,),
            hydration_expected=(),
            pending_expected=(pending,),
            rejection_screened=(rejected_input,),
            detail_fetches=(DetailFetch(FEED_DETAIL_PREFIX + pending.id, resolved=True),),
        )
        if order == "same_batch":
            await repo.apply_source_batch(
                replace(
                    stale_batch,
                    quarantined=reject_batch.quarantined,
                    pending_expected=reject_batch.pending_expected,
                    rejection_screened=(stale, rejected_input),
                    detail_fetches=(*stale_batch.detail_fetches, *reject_batch.detail_fetches),
                )
            )
        elif order == "rejection_first":
            await repo.apply_source_batch(reject_batch)
            await repo.apply_source_batch(stale_batch)
        else:
            await repo.apply_source_batch(stale_batch)
            await repo.apply_source_batch(reject_batch)
        assert await repo.get_job(original.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="hanzili"))
        held = next(entry for entry in entries if entry.id == original.id)
        assert held.verified_description == newer.description
        assert held.feed_summary == newer.feed_summary
        assert held.rejection_evidence == phd
        assert held.unverified_description == ""


@respx.mock
async def test_muse_fresh_full_body_releases_hydrated_rejection_only_after_screening(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobFilters, RejectionReason
    from stage.sources.themuse import TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})

    def page(contents: str) -> dict[str, object]:
        return {
            "results": [
                {
                    "id": 123,
                    "name": "Software Engineer Intern",
                    "company": {"name": "Acme"},
                    "locations": [{"name": "Toronto, Canada"}],
                    "refs": {"landing_page": "https://boards.greenhouse.io/acme/jobs/123"},
                    "contents": contents,
                    "categories": [{"name": "Software Engineering"}],
                }
            ],
            "page_count": 1,
        }

    respx.get(url__startswith="https://www.themuse.com/api/public/jobs").mock(
        side_effect=[
            httpx.Response(200, json=page("")),
            httpx.Response(200, json=page("")),
            httpx.Response(200, json=page("<p>Build software systems. PhD required.</p>")),
            httpx.Response(
                200, json=page("<p>Build software systems. Bachelor's degree welcome.</p>")
            ),
            httpx.Response(200, json=page("")),
        ]
    )
    detail = respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:

        async def run(hours: int) -> None:
            _ = [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["themuse"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]

        for hours in (0, 6, 12):
            await run(hours)
            assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
            entries = await repo.list_quarantined(QuarantineFilters(source="themuse"))
            assert len(entries) == 1 and entries[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
            assert "PhD" in entries[0].verified_description
        await run(18)
        jobs = await repo.list_jobs(JobFilters(source="themuse", limit=None))
        assert len(jobs) == 1
        assert jobs[0].description == "Build software systems. Bachelor's degree welcome."
        assert not await repo.list_quarantined(QuarantineFilters(source="themuse"))
        with sqlite3.connect(db_path) as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM detail_fetches WHERE id = ?",
                    (FEED_REJECTION_PREFIX + jobs[0].id,),
                ).fetchone()
                is None
            )
        await run(24)
        jobs = await repo.list_jobs(JobFilters(source="themuse", limit=None))
        assert (
            len(jobs) == 1
            and jobs[0].description == "Build software systems. Bachelor's degree welcome."
        )
    assert detail.call_count == 1


@respx.mock
@pytest.mark.parametrize("concurrent", ["quarantine", "job", "marker"])
async def test_muse_fresh_full_release_defers_on_concurrent_snapshot_change(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent: str
) -> None:
    from stage.domain import JobFilters
    from stage.services.screening import normalize_batch
    from stage.sources.themuse import TheMuseFeed

    feed = TheMuseFeed()
    monkeypatch.setattr(sync_module, "get_feeds", lambda: {"themuse": feed})

    def page(contents: str) -> dict[str, object]:
        return {
            "results": [
                {
                    "id": 123,
                    "name": "Software Engineer Intern",
                    "company": {"name": "Acme"},
                    "locations": [{"name": "Toronto, Canada"}],
                    "refs": {"landing_page": "https://boards.greenhouse.io/acme/jobs/123"},
                    "contents": contents,
                    "categories": [{"name": "Software Engineering"}],
                }
            ],
            "page_count": 1,
        }

    respx.get(url__startswith="https://www.themuse.com/api/public/jobs").mock(
        side_effect=[
            httpx.Response(200, json=page(""), headers={"ETag": '"v1"'}),
            httpx.Response(
                200, json=page("<p>Build software systems.</p>"), headers={"ETag": '"v2"'}
            ),
        ]
    )
    respx.get(GH + "123").respond(
        200, json=payload(body="Applicants must be enrolled in a PhD program.")
    )
    async with open_repository(db_path) as repo:

        async def run(hours: int) -> list[object]:
            return [
                event
                async for event in sync_module.sync(
                    repo,
                    [],
                    sources=["themuse"],
                    now_fn=clock_at(NOW + timedelta(hours=hours)),
                    close_orphans=False,
                    still_open=False,
                )
            ]

        await run(0)
        before = (await repo.list_quarantined(QuarantineFilters(source="themuse")))[0]
        changed = False

        def racing(jobs: Sequence[Job]) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
            nonlocal changed
            result = normalize_batch(jobs)
            if not changed:
                with sqlite3.connect(db_path) as conn:
                    if concurrent == "quarantine":
                        conn.execute(
                            "UPDATE quarantine SET verified_description = ? WHERE id = ?",
                            ("Newer PhD requirements.", before.id),
                        )
                    elif concurrent == "marker":
                        conn.execute(
                            "UPDATE detail_fetches SET attempts = attempts + 1 WHERE id = ?",
                            (FEED_REJECTION_PREFIX + before.id,),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                            "apply_url_raw, description, first_seen, last_seen) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                before.id,
                                "themuse",
                                "Acme",
                                before.title_raw,
                                before.title_raw,
                                before.apply_url_raw,
                                "Newer concurrently stored body.",
                                NOW.isoformat(),
                                NOW.isoformat(),
                            ),
                        )
                changed = True
            return result

        monkeypatch.setattr("stage.services.screening.normalize_batch", racing)
        events = await run(6)
        assert isinstance(events[-1], SyncFinished) and events[-1].outcome.value == "partial"
        entries = await repo.list_quarantined(QuarantineFilters(source="themuse"))
        assert len(entries) == 1
        if concurrent == "quarantine":
            assert entries[0].verified_description == "Newer PhD requirements."
            assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
        elif concurrent == "marker":
            assert entries[0].verified_description == before.verified_description
            assert not await repo.list_jobs(JobFilters(source="themuse", limit=None))
        else:
            assert entries[0].verified_description == before.verified_description
            rows = await repo.list_jobs(JobFilters(source="themuse", limit=None))
            assert len(rows) == 1 and rows[0].description == "Newer concurrently stored body."
        with sqlite3.connect(db_path) as conn:
            assert conn.execute(
                "SELECT resolved, failed FROM detail_fetches WHERE id = ?",
                (FEED_REJECTION_PREFIX + before.id,),
            ).fetchone() == (1, 0)
        assert not await repo.load_validators("themuse")


@pytest.mark.parametrize("fresh", [None, "", "  ", "Different body."])
async def test_full_source_hold_cannot_release_without_matching_fresh_screened_body(
    db_path: Path, fresh: str | None
) -> None:
    from stage.domain import DetailFetch, RejectionReason
    from stage.storage.copies import as_job

    held = QuarantinedJob(
        id="themuse:acme:123",
        source="themuse",
        company="Acme",
        title_raw="Software Engineer Intern",
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        matched_phrase="PhD",
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw="https://boards.greenhouse.io/acme/jobs/123",
        verified_description="PhD required.",
    )
    accepted = replace(as_job(held), description="Build software systems.")
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source="themuse",
                run_started_at=NOW,
                quarantined=(held,),
                detail_fetches=(DetailFetch(FEED_REJECTION_PREFIX + held.id, resolved=True),),
            )
        )
        await repo.apply_source_batch(
            SourceBatch(
                source="themuse",
                run_started_at=NOW,
                jobs=(accepted,),
                feed_source=True,
                feed_full_source=True,
                feed_expected=(),
                feed_quarantine_expected=(held,),
                rejection_expected=(),
                rejection_screened=(accepted,),
                feed_fresh_bodies=() if fresh is None else ((held.id, fresh),),
                feed_marker_expected=await repo.feed_marker_snapshot([held.id]),
            )
        )
        assert await repo.get_job(held.id) is None
        entries = await repo.list_quarantined(QuarantineFilters(source="themuse"))
        assert len(entries) == 1 and entries[0].verified_description == held.verified_description
