import asyncio
import multiprocessing
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast, overload

from stage.classify import (
    classify_role,
    resolve_eligibility,
    resolve_level,
    screen_degree_scope,
    screen_is_cs_role,
    screen_location,
    to_quarantined,
)
from stage.classify.scope import Rejection
from stage.domain import DegreeRequirement, Job, QuarantinedJob, RejectionReason, RoleCategory
from stage.normalize import (
    canonical_apply_url,
    detect_language,
    listed_place,
    resolve_listing,
    resolve_term,
    shown_company,
)

POOL_THRESHOLD = 500
CHUNK_SIZE = 1_000
WORKERS = 2

Screened = tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]
_PARSE_WAITERS = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="stage-parse")


EARLIEST_POSTED = datetime(1990, 1, 1, tzinfo=UTC)


def _plausible(posted: datetime | None, seen: datetime) -> datetime | None:
    if posted is None:
        return None
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=UTC)
    latest = (seen if seen.tzinfo else seen.replace(tzinfo=UTC)) + timedelta(days=366)
    return posted if EARLIEST_POSTED <= posted <= latest else None


_DEGREE_RANK = {
    DegreeRequirement.UNKNOWN: 0,
    DegreeRequirement.NONE: 0,
    DegreeRequirement.BACHELORS: 1,
    DegreeRequirement.MASTERS: 2,
    DegreeRequirement.PHD: 3,
}


def _normalize_one(job: Job) -> tuple[Job, Rejection | None]:
    job = replace(
        job,
        company=shown_company(job.company),
        location_raw=listed_place(job.location_raw, job.apply_url_raw, job.title_raw),
        source_posted_at=_plausible(job.source_posted_at, job.first_seen),
    )
    location = resolve_listing(job.location_raw, job.title_raw)
    term = resolve_term(
        title=job.title_raw,
        description=job.description,
        structured_terms=job.signals.terms,
        structured_season=job.signals.season,
        pivot_year=(job.source_posted_at or job.first_seen).year,
        posted=job.source_posted_at or job.first_seen,
    )
    role = classify_role(job.title_raw, job.description, job.signals.category)
    language = detect_language(job.title_raw, job.description)
    eligibility = resolve_eligibility(job)
    normalized = replace(
        job,
        apply_url_canonical=canonical_apply_url(job.apply_url_raw),
        location=location.bucket,
        remote_scope=location.remote_scope,
        term=term.term,
        role=role.role,
        roles=role.roles,
        language=language.language,
        degree_requirement=eligibility.degree_requirement,
        work_auth_flag=eligibility.work_auth_flag,
    )
    level, rejection = resolve_level(normalized)
    normalized = replace(normalized, level=level)
    rejection = (
        rejection
        or screen_location(normalized)
        or screen_degree_scope(normalized)
        or screen_is_cs_role(normalized)
    )
    return normalized, rejection


@overload
def combine_work_auth(left: bool, right: bool) -> bool: ...


@overload
def combine_work_auth(left: bool | None, right: bool | None) -> bool | None: ...


def combine_work_auth(left: bool | None, right: bool | None) -> bool | None:
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def combine_summary_evidence(job: Job, rejection: Rejection | None) -> tuple[Job, Rejection | None]:
    if not job.feed_summary:
        return job, rejection
    summary, summary_rejection = _normalize_one(
        replace(job, description=job.feed_summary, feed_summary="")
    )
    role_reasons = {RejectionReason.NOT_A_CS_ROLE, RejectionReason.UNKNOWN_CS_ROLE}
    full_role = screen_is_cs_role(job)
    summary_role = screen_is_cs_role(summary)
    if rejection is not None and rejection.reason in role_reasons and summary_role is None:
        rejection = None
    if (
        summary_rejection is not None
        and summary_rejection.reason in role_reasons
        and full_role is None
    ):
        summary_rejection = None
    restrictive = next(
        (
            item
            for item in (rejection, summary_rejection)
            if item is not None and item.reason not in role_reasons
        ),
        None,
    )
    role = job.role
    if summary_role is None and (full_role is not None or role is RoleCategory.UNKNOWN):
        role = summary.role
    degree = job.degree_requirement
    if _DEGREE_RANK[summary.degree_requirement] > _DEGREE_RANK[degree]:
        degree = summary.degree_requirement
    return replace(
        job,
        role=role,
        roles=tuple(dict.fromkeys((*job.roles, *(summary.roles if summary_role is None else ())))),
        degree_requirement=degree,
        work_auth_flag=combine_work_auth(job.work_auth_flag, summary.work_auth_flag),
    ), restrictive or rejection or summary_rejection


def normalize_batch(jobs: Sequence[Job]) -> Screened:
    kept: list[Job] = []
    rejected: list[QuarantinedJob] = []
    for job in jobs:
        normalized, rejection = _normalize_one(job)
        normalized, rejection = combine_summary_evidence(normalized, rejection)
        if rejection is None:
            kept.append(normalized)
        else:
            rejected.append(to_quarantined(normalized, rejection))
    return tuple(kept), tuple(rejected)


def _ignore_interrupts() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_exit_with, args=(parent,), daemon=True).start()


def _exit_with(parent: multiprocessing.process.BaseProcess) -> None:
    parent.join()
    os._exit(0)


def _screen_in(pool: ProcessPoolExecutor, chunks: list[list[Job]]) -> list[Screened]:
    with _interrupts_held():
        results = pool.map(normalize_batch, chunks)
    return list(results)


def _outcome[R](call: Callable[[], R]) -> tuple[R | None, Exception | None]:
    try:
        return call(), None
    except Exception as exc:
        return None, exc


def _parse_in[R](
    pool: ProcessPoolExecutor, call: Callable[[], R]
) -> tuple[R | None, Exception | None]:
    with _interrupts_held():
        future = pool.submit(_outcome, call)
    return future.result()


if sys.platform == "win32":

    @contextmanager
    def _interrupts_held() -> Iterator[None]:
        yield

else:

    @contextmanager
    def _interrupts_held() -> Iterator[None]:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@contextmanager
def _terminal_stderr() -> Iterator[None]:
    captured = sys.stderr
    if sys.__stderr__ is not None:
        sys.stderr = sys.__stderr__
    try:
        yield
    finally:
        sys.stderr = captured


class Screener:
    def __init__(self, workers: int = WORKERS, threshold: int = POOL_THRESHOLD) -> None:
        self.workers = max(1, min(workers, os.cpu_count() or 1))
        self.threshold = threshold
        self.broken = False
        self._pool: ProcessPoolExecutor | None = None

    async def __call__(self, jobs: Sequence[Job]) -> Screened:
        from stage.sources.base import off_loop

        batch = list(jobs)
        if self.broken or len(batch) < self.threshold:
            return await off_loop(normalize_batch, batch)
        chunks = [batch[start : start + CHUNK_SIZE] for start in range(0, len(batch), CHUNK_SIZE)]
        try:
            results = await asyncio.to_thread(_screen_in, self._started(), chunks)
        except Exception:
            self._give_up()
            return await off_loop(normalize_batch, batch)
        return (
            tuple(job for kept, _ in results for job in kept),
            tuple(entry for _, rejected in results for entry in rejected),
        )

    async def parse[R](self, call: Callable[[], R]) -> R:
        from stage.sources.base import off_loop

        if not self.broken:
            loop = asyncio.get_running_loop()
            try:
                value, error = await loop.run_in_executor(
                    _PARSE_WAITERS, _parse_in, self._started(), call
                )
            except Exception:
                self._give_up()
            else:
                if error is not None:
                    raise error
                return cast(R, value)
        return await off_loop(call)

    def _give_up(self) -> None:
        self.broken = True
        self.close()

    def _started(self) -> ProcessPoolExecutor:
        if self._pool is None:
            with _terminal_stderr():
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_ignore_interrupts,
                )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
