from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from stage.dedup import resolve_duplicates
from stage.domain import (
    UNKNOWN_TERM,
    IntegrityRepair,
    Job,
    PurgeResult,
    QuarantinedJob,
    RateState,
    RoleCategory,
)
from stage.storage import AsyncRepository, SourceBatch

RESCREEN_LIMIT = 100_000
RESCREEN_PASSES = 8


@dataclass(frozen=True, slots=True)
class RateStateView:
    cleared: int
    states: tuple[RateState, ...]
    validators_cleared: int = 0


async def repair_integrity(repository: AsyncRepository) -> tuple[IntegrityRepair, ...]:
    return tuple(await repository.repair_integrity())


async def purge_expired(repository: AsyncRepository, *, now: datetime | None = None) -> PurgeResult:
    from stage.services.tracker import follow_recorded

    result = await repository.purge(now or datetime.now(UTC))
    if await follow_recorded(repository) is None:
        return replace(result, tracker_busy=True)
    return result


async def preview_purge(repository: AsyncRepository, *, now: datetime | None = None) -> PurgeResult:
    return await repository.preview_purge(now or datetime.now(UTC))


async def rate_state(
    repository: AsyncRepository,
    *,
    bucket: str | None = None,
    clear_all: bool = False,
    clear_cache: str | None = None,
    clear_cache_all: bool = False,
) -> RateStateView:
    cleared = 0
    if clear_all:
        cleared = await repository.clear_rate_state()
    elif bucket is not None:
        cleared = await repository.clear_rate_state(bucket)
    validators = 0
    if clear_cache_all:
        validators = await repository.clear_validators()
    elif clear_cache is not None:
        validators = await repository.clear_validators(clear_cache)
    states = await repository.load_rate_state()
    return RateStateView(
        cleared=cleared, states=tuple(states.values()), validators_cleared=validators
    )


@dataclass(frozen=True, slots=True)
class RescreenResult:
    examined: int
    quarantined: int
    total: int = 0
    updated: int = 0
    released: int = 0
    relabelled: int = 0
    relocated: int = 0
    relinked: int = 0
    quarantine_waiting: int = 0
    complete: bool = True

    @property
    def recovery_complete(self) -> bool:
        return self.complete and not self.quarantine_waiting

    @property
    def changed(self) -> bool:
        return bool(
            self.updated
            or self.quarantined
            or self.released
            or self.relabelled
            or self.relocated
            or self.relinked
        )

    @property
    def skipped(self) -> int:
        return max(0, self.total - self.examined)


RECHECK_NOTICE = "Checking your stored postings first. This can take a few minutes."


async def upgrade_pending(repository: AsyncRepository) -> bool:
    from stage.paths import rules_stamp

    return await repository.screened_with() != rules_stamp()


def upgrade_note(result: RescreenResult) -> str:
    from stage.domain.text import counted

    return (
        "This version of Stage screens postings differently, so it checked the ones already "
        f"stored: {counted(result.quarantined, 'posting')} now rejected, "
        f"{counted(result.released, 'posting')} brought back. "
        f"{counted(result.quarantine_waiting, 'quarantined posting')} await a fresh "
        "source response."
    )


async def rescreen(repository: AsyncRepository, *, now: datetime | None = None) -> RescreenResult:
    from stage.domain import JobFilters, QuarantineFilters
    from stage.normalize import canonical_apply_url
    from stage.paths import rules_stamp
    from stage.services.tracker import follow_recorded

    moment = now or datetime.now(UTC)
    examined = 0
    quarantined = 0
    updated = 0
    total = await repository.count_jobs(
        JobFilters(status=None, limit=RESCREEN_LIMIT, with_duplicates=True)
    )
    settled: dict[str, Job] = {}

    complete = False
    for pass_number in range(RESCREEN_PASSES):
        after_id = ""
        changed_this_pass = False
        while True:
            stored = await repository.list_rescreen_jobs(after_id, RESCREEN_LIMIT)
            if not stored:
                break
            after_id = stored[-1].id
            if not pass_number:
                examined += len(stored)
            pending = [job for job in stored if settled.get(job.id) != job]
            protected = (
                {
                    entry.id
                    for entry in await repository.verified_feed_jobs([job.id for job in pending])
                }
                if pending
                else set()
            )
            candidates, rejected = _reclassifications(pending) if pending else ((), ())
            rejected_ids = {entry.id for entry in rejected}
            changed: list[Job] = []
            for job, candidate in zip(pending, candidates, strict=True):
                if job != candidate:
                    changed.append(candidate)
                elif job.id not in rejected_ids:
                    settled[job.id] = job
            if not changed and not rejected:
                continue
            changed_this_pass = True
            updates = tuple(entry for entry in changed if entry.id not in rejected_ids)
            sources = {entry.source for entry in updates} | {entry.source for entry in rejected}
            for source in sorted(sources):
                applied = await repository.apply_source_batch(
                    SourceBatch(
                        source=source,
                        run_started_at=moment,
                        jobs=tuple(entry for entry in updates if entry.source == source),
                        quarantined=tuple(entry for entry in rejected if entry.source == source),
                        rejection_expected=tuple(
                            entry
                            for entry in pending
                            if entry.source == source and entry.id in protected
                        ),
                        rejection_screened=tuple(
                            entry for entry in pending if entry.source == source
                        ),
                    )
                )
                if applied.successors:
                    await follow_recorded(repository)
                quarantined += applied.quarantined
            updated += len(updates)
        if not changed_this_pass:
            complete = True
            break
    relabelled = await _reclassify_quarantine(repository)
    relocated = await repository.refresh_quarantine_locations(_place_resolver())
    relinked = await repository.relink_duplicates(resolve_duplicates, canonical_apply_url)
    waiting = await repository.count_quarantined(QuarantineFilters())
    if complete:
        await repository.mark_screened(rules_stamp(), at=moment)
    return RescreenResult(
        examined=examined,
        quarantined=quarantined,
        total=total,
        updated=updated,
        relabelled=relabelled,
        relocated=relocated,
        relinked=relinked,
        quarantine_waiting=waiting,
        complete=complete,
    )


async def _reclassify_quarantine(repository: AsyncRepository) -> int:
    from stage.domain import RejectionReason, SourceSignals
    from stage.lexicon import fold
    from stage.services.screening import normalize_batch

    relabelled = 0
    after_id = ""
    while entries := await repository.list_rescreen_quarantine(after_id, RESCREEN_LIMIT):
        after_id = entries[-1].id
        candidates = tuple(
            Job(
                id=entry.id,
                source=entry.source,
                company=entry.company,
                title_raw=entry.title_raw,
                title_normalized=fold(entry.title_raw),
                apply_url_raw=entry.apply_url_raw,
                description=entry.verified_description,
                feed_summary=entry.feed_summary,
                first_seen=entry.first_seen,
                last_seen=entry.last_seen,
                location_raw=entry.location_raw,
                location=entry.location,
                remote_scope=entry.remote_scope,
                signals=SourceSignals(
                    employment_type=entry.employment_type, category=entry.category
                ),
            )
            for entry in entries
            if entry.reason is not RejectionReason.OUT_OF_SCOPE_DEGREE
        )
        _, rejected = normalize_batch(candidates)
        previous = {entry.id: entry for entry in entries}
        relabelled += await repository.relabel_quarantine(
            tuple(
                entry
                for entry in rejected
                if (entry.reason, entry.matched_phrase)
                != (previous[entry.id].reason, previous[entry.id].matched_phrase)
            )
        )
    return relabelled


def _current_term(job: Job) -> str:
    from stage.normalize import resolve_term
    from stage.normalize.terms import ended_before, fits_title, graduation_only

    since = job.source_posted_at or job.first_seen
    titled = resolve_term(title=job.title_raw, pivot_year=since.year, posted=since)
    if titled.conflict:
        return UNKNOWN_TERM
    if titled.term != UNKNOWN_TERM:
        return titled.term
    if job.term == UNKNOWN_TERM:
        written = resolve_term(
            title=job.title_raw, description=job.description, pivot_year=since.year, posted=since
        ).term
        return written if fits_title(written, job.title_raw) else UNKNOWN_TERM
    if ended_before(job.term, since) or not fits_title(job.term, job.title_raw):
        return UNKNOWN_TERM
    if graduation_only(job.term, job.title_raw, job.description, since.year):
        return UNKNOWN_TERM
    return job.term


def _place_resolver() -> Callable[[str, str, str], tuple[str, str | None]]:
    from stage.domain import LocationBucket
    from stage.normalize import listed_place, resolve_listing, resolve_location

    def resolve(raw: str, url: str, title: str) -> tuple[str, str | None]:
        resolved = resolve_location(raw)
        if resolved.bucket is LocationBucket.UNKNOWN:
            resolved = resolve_listing(listed_place(raw, url, title), title)
        scope = resolved.remote_scope
        return resolved.bucket.value, None if scope is None else scope.value

    return resolve


def _reclassifications(
    jobs: Sequence[Job],
) -> tuple[tuple[Job, ...], tuple[QuarantinedJob, ...]]:
    from stage.classify import (
        classify_role,
        resolve_eligibility,
        resolve_level,
        screen_degree_scope,
        screen_is_cs_role,
        screen_location,
    )
    from stage.classify.scope import to_quarantined
    from stage.normalize import (
        canonical_apply_url,
        detect_language,
        listed_place,
        resolve_listing,
        shown_company,
    )
    from stage.services.screening import combine_summary_evidence

    candidates: list[Job] = []
    rejected: list[QuarantinedJob] = []
    for job in jobs:
        place = listed_place(job.location_raw, job.apply_url_raw, job.title_raw)
        location = resolve_listing(place, job.title_raw)
        verdict = classify_role(job.title_raw, job.description, job.signals.category)
        title_role = verdict.role
        candidate = replace(
            job,
            company=shown_company(job.company),
            location_raw=place,
            apply_url_canonical=canonical_apply_url(job.apply_url_raw),
            location=location.bucket,
            remote_scope=location.remote_scope,
            role=title_role if title_role is not RoleCategory.UNKNOWN else job.role,
            roles=verdict.roles,
            term=_current_term(job),
        )
        eligibility = resolve_eligibility(candidate)
        candidate = replace(
            candidate,
            language=detect_language(job.title_raw, job.description).language,
            degree_requirement=eligibility.degree_requirement,
            work_auth_flag=eligibility.work_auth_flag,
        )
        level, rejection = resolve_level(candidate)
        candidate = replace(candidate, level=level)
        rejection = (
            rejection
            or screen_location(candidate)
            or screen_degree_scope(candidate)
            or screen_is_cs_role(candidate)
        )
        candidate, rejection = combine_summary_evidence(candidate, rejection)
        if rejection is not None:
            rejected.append(to_quarantined(candidate, rejection))
        candidates.append(candidate)
    return tuple(candidates), tuple(rejected)
