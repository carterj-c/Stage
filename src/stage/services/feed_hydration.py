import asyncio
import random
import time
from collections.abc import AsyncIterator, Collection, Mapping
from dataclasses import dataclass, replace
from datetime import datetime

import httpx

from stage.domain import DetailFetch, Job, RequestLogged
from stage.domain.visits import FEED_DETAIL_PREFIX
from stage.http import HostBudget, HttpClient, HttpError, RatePosture, RequestRefusedError, resolve
from stage.services.hydration_targets import HOSTS, fetch_target, target_for_url
from stage.services.screening import Screener
from stage.sources import get_adapter, platform_bucket, platform_front
from stage.storage import AsyncRepository, SourceBatch, SourceBatchResult

REQUESTS_PER_SYNC = 30
HYDRATION_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class FeedHydrationResult:
    batches: tuple[SourceBatchResult, ...] = ()
    requests: tuple[RequestLogged, ...] = ()


async def _queued_jobs(
    repository: AsyncRepository,
    sources: Collection[str],
    now: datetime,
    first_page: list[Job],
    deadline: float,
) -> AsyncIterator[Job]:
    page = first_page
    offset = 0
    while page and time.monotonic() < deadline:
        for job in page:
            if time.monotonic() >= deadline:
                return
            yield job
        offset += len(page)
        if len(page) < REQUESTS_PER_SYNC or time.monotonic() >= deadline:
            return
        page = await repository.feed_hydration_queue(
            sources,
            now,
            REQUESTS_PER_SYNC,
            offset=offset,
        )


async def hydrate_feeds(
    repository: AsyncRepository,
    sources: Collection[str],
    screener: Screener,
    *,
    now: datetime,
    budgets: dict[str, HostBudget],
    postures: Mapping[str, RatePosture],
    rng: random.Random,
) -> FeedHydrationResult:
    deadline = time.monotonic() + HYDRATION_SECONDS
    queued = await repository.feed_hydration_queue(sources, now, REQUESTS_PER_SYNC)
    if not queued:
        return FeedHydrationResult()
    states = await repository.load_rate_state()
    host_buckets = {host: platform_bucket(host) for host in HOSTS}
    profiles = dict(postures)
    for provider, host in (
        ("greenhouse", "boards-api.greenhouse.io"),
        ("lever", "api.lever.co"),
        ("workable", "apply.workable.com"),
        ("ashby", "jobs.ashbyhq.com"),
    ):
        posture = resolve(get_adapter(provider).rate_profile, [])
        bucket = host_buckets[host]
        profiles[bucket] = profiles.get(bucket, posture).strictest(posture)
    originals: dict[str, list[Job]] = {}
    collected: dict[str, list[Job]] = {}
    outcomes: dict[str, list[DetailFetch]] = {}
    requests: list[RequestLogged] = []
    attempted = 0
    blocked: set[str] = set()
    async with HttpClient(
        allowed_hosts=HOSTS,
        budgets=budgets,
        postures=profiles,
        rate_state=states,
        now=now,
        rng=rng,
        host_buckets=host_buckets,
        host_fronts={host: platform_front(host) for host in HOSTS},
        request_limit=REQUESTS_PER_SYNC,
        allow_redirects=False,
    ) as client:
        async for job in _queued_jobs(repository, sources, now, queued, deadline):
            if (
                client.request_count >= REQUESTS_PER_SYNC
                or attempted >= REQUESTS_PER_SYNC
                or time.monotonic() >= deadline
            ):
                break
            target = target_for_url(job.apply_url_raw)
            detail = None
            if target is not None and target.provider != "ashby":
                host = next(iter(target.hosts))
                if host in blocked:
                    continue
                before = client.request_count
                try:
                    detail = await asyncio.wait_for(
                        fetch_target(target, client),
                        max(0.001, deadline - time.monotonic()),
                    )
                except RequestRefusedError:
                    blocked.add(host)
                    if client.request_count == before:
                        continue
                except (HttpError, httpx.HTTPError, TimeoutError):
                    pass
                finally:
                    requests.extend(client.drain_requests(job.source))
            attempted += 1
            originals.setdefault(job.source, []).append(job)
            complete = detail is not None and detail.complete and bool(detail.description.strip())
            if complete and detail is not None:
                collected.setdefault(job.source, []).append(
                    replace(job, description=detail.description)
                )
            outcomes.setdefault(job.source, []).append(
                DetailFetch(FEED_DETAIL_PREFIX + job.id, resolved=complete, failed=not complete)
            )
        settled = client.rate_state(now)
    batches: list[SourceBatchResult] = []
    pending_states = settled
    for source in sorted(set(originals) | {queued[0].source}):
        kept, rejected = await screener(collected.get(source, []))
        counts = await repository.apply_source_batch(
            SourceBatch(
                source=source,
                run_started_at=now,
                jobs=kept,
                quarantined=rejected,
                detail_fetches=tuple(outcomes.get(source, [])),
                rate_state=pending_states,
                hydration_expected=tuple(originals.get(source, [])),
                rejection_screened=tuple(collected.get(source, [])),
            )
        )
        pending_states = ()
        batches.append(counts)
        if counts.successors:
            from stage.services.tracker import follow_recorded

            await follow_recorded(repository)
    return FeedHydrationResult(tuple(batches), tuple(requests))
