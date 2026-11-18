from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stage.domain import (
    Job,
    JobFilters,
    JobLevel,
    JobStatus,
    LocationBucket,
    QuarantinedJob,
    QuarantineFilters,
    RejectionReason,
    RoleCategory,
    SourceSignals,
)
from stage.services import maintenance
from stage.services.maintenance import rescreen
from stage.storage import open_repository
from stage.storage.repository import SourceBatch

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
NO_ROLE = "no matching CS role or source category"


def _entry(
    identifier: str,
    title: str,
    reason: RejectionReason = RejectionReason.NOT_AN_INTERNSHIP,
    phrase: str = "no internship marker in title",
    *,
    source: str = "greenhouse",
    employment_type: str = "",
    category: str = "",
    location: str = "Montreal, QC",
) -> QuarantinedJob:
    return QuarantinedJob(
        id=identifier,
        source=source,
        company="Acme",
        title_raw=title,
        reason=reason,
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw=f"https://jobs.example.test/{identifier}",
        location_raw=location,
        matched_phrase=phrase,
        employment_type=employment_type,
        category=category,
    )


async def _store(repository: object, entries: list[QuarantinedJob]) -> None:
    await repository.apply_source_batch(  # type: ignore[attr-defined]
        SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=tuple(entries))
    )


async def _levels(repository: object) -> dict[str, JobLevel]:
    filters = JobFilters(status=None, limit=None)
    return {job.id: job.level for job in await repository.list_jobs(filters)}  # type: ignore[attr-defined]


async def _reasons(repository: object) -> dict[str, tuple[str, str]]:
    rows = await repository.list_quarantined(QuarantineFilters(limit=None))  # type: ignore[attr-defined]
    return {row.id: (row.reason.value, row.matched_phrase) for row in rows}


async def test_not_an_internship_rows_wait_for_a_source_refresh(db_path: Path) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry("grad", "Software Engineer, New Grad"),
                _entry("entry", "Software Engineer I, Backend (Continuous Delivery)"),
                _entry("senior", "Senior Software Engineer"),
                _entry("plain", "Software Engineer"),
                _entry("generalist", "Startup Generalist (New Grad)"),
            ],
        )

        result = await rescreen(repository, now=NOW)

        assert result.released == 0 and result.quarantine_waiting == 5
        assert await _levels(repository) == {}
        assert len(await _reasons(repository)) == 5


async def test_a_second_pass_finds_nothing_to_release_or_relabel(db_path: Path) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [_entry("grad", "Software Engineer, New Grad"), _entry("plain", "Software Engineer")],
        )
        await rescreen(repository, now=NOW)

        again = await rescreen(repository, now=NOW)

        assert (again.released, again.relabelled) == (0, 0)


async def test_an_ended_internship_term_keeps_its_not_an_internship_reason(db_path: Path) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry(
                    "old", "Software Engineer Intern, Summer 2025", phrase="summer 2025 has ended"
                )
            ],
        )

        result = await rescreen(repository, now=NOW)

        assert (result.released, result.relabelled) == (0, 0)
        assert await _reasons(repository) == {"old": ("not-an-internship", "summer 2025 has ended")}


async def test_quarantined_rows_do_not_consume_the_stored_posting_page_limit(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(maintenance, "RESCREEN_LIMIT", 2)
    stale = [
        _entry(f"role{number}", title, RejectionReason.UNKNOWN_CS_ROLE, NO_ROLE)
        for number, title in enumerate(("Software Engineer Intern", "Data Engineer Intern"))
    ]
    async with open_repository(db_path) as repository:
        await _store(repository, [*stale, _entry("grad", "Software Engineer, New Grad")])

        result = await rescreen(repository, now=NOW)

        assert result.released == 0 and result.quarantine_waiting == 3
        assert await _levels(repository) == {}
        assert set(await _reasons(repository)) == {"role0", "role1", "grad"}


async def test_a_quarantined_row_stays_held_after_a_lexicon_change(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.lexicon import level_lexicon

    async with open_repository(db_path) as repository:
        await _store(repository, [_entry("trainee", "Software Engineering Trainee")])
        await rescreen(repository, now=NOW)
        assert len(await _reasons(repository)) == 1
        assert await _levels(repository) == {}

        real = level_lexicon()
        monkeypatch.setattr(
            "stage.classify.level.level_lexicon",
            lambda: replace(real, entry=real.entry | {"trainee"}),
        )
        result = await rescreen(repository, now=NOW)

        assert result.updated == 0 and result.quarantine_waiting == 1
        assert await _levels(repository) == {}
        assert len(await _reasons(repository)) == 1


async def test_quarantined_reasons_wait_independently_of_the_page_limit(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(maintenance, "RESCREEN_LIMIT", 2)
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry("first", "Software Engineer, New Grad", RejectionReason.NOT_EARLY_CAREER),
                _entry("second", "Junior Software Engineer", RejectionReason.NOT_EARLY_CAREER),
                _entry(
                    "older", "Software Engineer Intern", RejectionReason.UNKNOWN_CS_ROLE, NO_ROLE
                ),
                _entry("middle", "Software Engineer, New Grad"),
            ],
        )

        result = await rescreen(repository, now=NOW)

        assert result.released == 0 and result.quarantine_waiting == 4
        assert await _levels(repository) == {}


async def test_a_markerless_new_grad_feed_row_keeps_its_reason_through_a_rescreen(
    db_path: Path,
) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry(
                    "simplify-new-grad:acme:1",
                    "Software Engineer",
                    RejectionReason.OUT_OF_SCOPE_LOCATION,
                    "Berlin, Germany",
                    source="simplify-new-grad",
                    employment_type="new grad",
                    location="Berlin, Germany",
                ),
                _entry(
                    "greenhouse:acme:2",
                    "Software Engineer",
                    RejectionReason.OUT_OF_SCOPE_LOCATION,
                    "Berlin, Germany",
                    location="Berlin, Germany",
                ),
            ],
        )

        result = await rescreen(repository, now=NOW)
        rows = await repository.list_quarantined(QuarantineFilters(limit=None))

        assert result.released == 0
        assert {row.id: (row.reason.value, row.employment_type) for row in rows} == {
            "simplify-new-grad:acme:1": ("out-of-scope-location", "new grad"),
            "greenhouse:acme:2": ("out-of-scope-location", ""),
        }


async def test_a_markerless_new_grad_feed_row_needs_a_fresh_body(
    db_path: Path,
) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry(
                    "simplify-new-grad:acme:1",
                    "Software Engineer",
                    RejectionReason.OUT_OF_SCOPE_LOCATION,
                    "Berlin, Germany",
                    source="simplify-new-grad",
                    employment_type="new grad",
                    location="Montreal, QC",
                )
            ],
        )

        result = await rescreen(repository, now=NOW)

        assert result.released == 0 and result.quarantine_waiting == 1
        assert await _levels(repository) == {}
        assert len(await _reasons(repository)) == 1


async def test_a_quarantined_row_with_conflicting_source_category_needs_a_fresh_body(
    db_path: Path,
) -> None:
    async with open_repository(db_path) as repository:
        await _store(
            repository,
            [
                _entry(
                    "custom:acme:1",
                    "Software Developer Intern",
                    RejectionReason.NOT_EARLY_CAREER,
                    "new graduates",
                    employment_type="internship",
                    category="New Graduates",
                )
            ],
        )

        result = await rescreen(repository, now=NOW)
        rows = await repository.list_quarantined(QuarantineFilters(limit=None))

        assert (result.released, result.relabelled) == (0, 0)
        assert len(rows) == 1
        assert await _levels(repository) == {}


async def test_a_stored_posting_with_conflicting_category_keeps_unknown_level(
    db_path: Path,
) -> None:
    stored = Job(
        id="custom:acme:1",
        source="greenhouse",
        company="Acme",
        title_raw="Software Developer Intern",
        title_normalized="software developer intern",
        apply_url_raw="https://jobs.example.test/1",
        description="",
        first_seen=NOW,
        last_seen=NOW,
        location_raw="Montreal, QC",
        location=LocationBucket.MONTREAL,
        role=RoleCategory.SWE,
        status=JobStatus.OPEN,
        level=JobLevel.INTERNSHIP,
        signals=SourceSignals(employment_type="internship", category="New Graduates"),
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=(stored,))
        )

        result = await rescreen(repository, now=NOW)
        rows = await repository.list_quarantined(QuarantineFilters(limit=None))

        assert (result.quarantined, result.released) == (0, 0)
        assert rows == []
        assert await _levels(repository) == {"custom:acme:1": JobLevel.UNKNOWN}


async def test_a_stored_posting_is_rescreened_to_its_own_level(db_path: Path) -> None:
    stored = [
        Job(
            id=f"greenhouse:acme:{name}",
            source="greenhouse",
            company="Acme",
            title_raw=title,
            title_normalized=title.lower(),
            apply_url_raw=f"https://jobs.example.test/{name}",
            description="",
            first_seen=NOW,
            last_seen=NOW,
            location_raw="Montreal, QC",
            location=LocationBucket.MONTREAL,
            role=RoleCategory.SWE,
            status=JobStatus.OPEN,
            level=level,
        )
        for name, title, level in (
            ("intern", "Software Engineer Intern", JobLevel.INTERNSHIP),
            ("grad", "Software Engineer, New Grad", JobLevel.NEW_GRAD),
            ("old", "Senior Software Engineer", JobLevel.INTERNSHIP),
        )
    ]
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=tuple(stored))
        )

        result = await rescreen(repository, now=NOW)

        assert result.quarantined == 0
        assert await _levels(repository) == {
            "greenhouse:acme:intern": JobLevel.INTERNSHIP,
            "greenhouse:acme:grad": JobLevel.NEW_GRAD,
            "greenhouse:acme:old": JobLevel.SENIOR,
        }
        assert await _reasons(repository) == {}
