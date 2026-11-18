from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stage.classify.level import screen_level
from stage.classify.scope import Rejection, to_quarantined
from stage.domain import (
    DegreeRequirement,
    Job,
    JobFilters,
    JobLevel,
    JobStatus,
    QuarantineFilters,
    RejectionReason,
    RoleCategory,
    SourceSignals,
)
from stage.services import maintenance
from stage.services.screening import normalize_batch
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 11, 18, tzinfo=UTC)


def posting(ident: str, title: str, body: str = "", source: str = "greenhouse") -> Job:
    return Job(
        id=ident,
        source=source,
        company="Acme",
        title_raw=title,
        title_normalized=title.lower(),
        apply_url_raw=f"https://example.test/{ident}",
        description=body,
        location_raw="Montreal, QC",
        first_seen=NOW,
        last_seen=NOW,
    )


@pytest.mark.parametrize("reason", list(RejectionReason))
async def test_old_quarantine_without_a_body_stays_held_until_complete_refresh(
    tmp_path: Path,
    reason: RejectionReason,
) -> None:
    unsafe = posting(
        "greenhouse:acme:one", "Software Engineer Intern", "PhD required. Must be a US citizen."
    )
    held = to_quarantined(unsafe, Rejection(reason, "old evidence"))
    async with open_repository(tmp_path / "held.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=(held,))
        )
        result = await maintenance.rescreen(repo, now=NOW)
        assert result.quarantine_waiting == 1 and result.released == 0
        assert await repo.get_job(unsafe.id) is None
        blank, _ = normalize_batch([replace(unsafe, description="")])
        assert len(blank) == 1
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=blank)
        )
        assert await repo.get_job(unsafe.id) is None
        _, rejected = normalize_batch([unsafe])
        assert rejected[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=rejected)
        )
        assert (await repo.list_quarantined(QuarantineFilters(limit=None)))[0].reason is (
            RejectionReason.OUT_OF_SCOPE_DEGREE
        )


async def test_rescreen_paginates_all_stored_rows_and_reports_quarantine_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maintenance, "RESCREEN_LIMIT", 2)
    jobs = tuple(posting(f"greenhouse:acme:{i}", "Software Engineer Intern") for i in range(5))
    held = to_quarantined(
        posting("greenhouse:acme:held", "Financial Analyst"),
        Rejection(RejectionReason.NOT_EARLY_CAREER, "old"),
    )
    async with open_repository(tmp_path / "pages.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=jobs, quarantined=(held,))
        )
        result = await maintenance.rescreen(repo, now=NOW)
        assert result.examined == result.total == 5
        assert result.skipped == 0 and result.complete
        assert result.quarantine_waiting == 1 and result.released == 0
        assert not result.recovery_complete
        assert {job.level for job in await repo.list_jobs(JobFilters(limit=None))} == {
            JobLevel.INTERNSHIP
        }


async def test_incomplete_rescreen_does_not_mark_rules_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maintenance, "RESCREEN_PASSES", 1)
    job = posting("greenhouse:acme:1", "Software Engineer Intern")
    async with open_repository(tmp_path / "incomplete.db") as repo:
        await repo.mark_screened(0)
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=(job,))
        )
        result = await maintenance.rescreen(repo, now=NOW)
        assert not result.complete
        assert await maintenance.upgrade_pending(repo)


async def test_duplicate_role_and_text_search_obey_status_and_other_filters(tmp_path: Path) -> None:
    main = replace(
        posting("greenhouse:acme:1", "Software Engineer", "backend systems"),
        role=RoleCategory.SWE,
        level=JobLevel.SENIOR,
    )
    copy = replace(
        posting("lever:acme:2", "Risk Manager", "derivatives research", "lever"),
        role=RoleCategory.RISK,
        level=JobLevel.SENIOR,
        duplicate_of=main.id,
    )
    filters = JobFilters(role=RoleCategory.RISK, level=JobLevel.SENIOR)
    async with open_repository(tmp_path / "copies.db") as repo:
        for job in (main, copy):
            await repo.apply_source_batch(
                SourceBatch(source=job.source, run_started_at=NOW, jobs=(job,))
            )
        assert [job.id for job in await repo.list_jobs(filters)] == [main.id]
        assert await repo.count_jobs(filters) == 1
        assert [job.id for job in await repo.search_jobs("derivatives", filters)] == [main.id]
        assert await repo.count_search("derivatives", filters) == 1
        other = replace(
            posting("ashby:acme:3", "Game Producer", "animation planning", "ashby"),
            role=RoleCategory.PRODUCTION,
            level=JobLevel.SENIOR,
            duplicate_of=main.id,
        )
        await repo.apply_source_batch(
            SourceBatch(source="ashby", run_started_at=NOW, jobs=(other,))
        )
        assert await repo.count_search("animation", filters) == 0
        assert await repo.count_search("animation", JobFilters(level=JobLevel.SENIOR)) == 1
        assert await repo.count_search("derivatives", replace(filters, level=JobLevel.MID)) == 0
        await repo.apply_source_batch(
            SourceBatch(
                source="lever", run_started_at=NOW, jobs=(replace(copy, status=JobStatus.CLOSED),)
            )
        )
        assert await repo.count_jobs(filters) == 0
        assert await repo.count_search("derivatives", filters) == 0


@pytest.mark.parametrize(
    "title", ["Risk Manager", "Finance Manager", "Project Manager", "Program Manager"]
)
def test_structured_intern_for_manager_occupations(title: str) -> None:
    job = replace(posting("job", title), signals=SourceSignals(employment_type="Intern"))
    kept, rejected = normalize_batch([job])
    assert not rejected
    assert kept[0].level is JobLevel.INTERNSHIP


@pytest.mark.parametrize(
    "title", ["Risk Manager of Interns", "Intern Manager", "Senior Risk Manager"]
)
def test_structured_intern_does_not_override_disqualifiers(title: str) -> None:
    assert screen_level(title, employment_type="Intern").level is not JobLevel.INTERNSHIP


@pytest.mark.parametrize("source", ["simplify-new-grad", "speedyapply-new-grad", "workday"])
async def test_descriptionless_quant_feed_refresh_preserves_unknown_evidence(
    tmp_path: Path,
    source: str,
) -> None:
    job = posting(f"{source}:acme:1", "Quantitative Researcher", source=source)
    held = to_quarantined(job, Rejection(RejectionReason.NOT_EARLY_CAREER, "old"))
    async with open_repository(tmp_path / "feed.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source=source, run_started_at=NOW, quarantined=(held,))
        )
        result = await maintenance.rescreen(repo, now=NOW)
        assert result.quarantine_waiting == 1
        kept, rejected = normalize_batch([job])
        assert len(kept) == 1 and not rejected
        await repo.apply_source_batch(SourceBatch(source=source, run_started_at=NOW, jobs=kept))
        restored = await repo.get_job(job.id)
        assert restored is not None
        assert restored.degree_requirement is DegreeRequirement.UNKNOWN
        assert restored.work_auth_flag is None
        assert await repo.list_quarantined(QuarantineFilters(limit=None)) == []


async def test_rescreen_pages_beyond_100000_without_hiding_held_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from stage.storage import AsyncRepository

    limit = 10_000
    total = 100_001
    template = posting("000000", "Software Engineer Intern")
    repo = AsyncMock(spec=AsyncRepository)
    repo.count_jobs.return_value = total
    repo.count_quarantined.return_value = total
    repo.refresh_quarantine_locations.return_value = 0
    repo.relink_duplicates.return_value = 0
    visited: list[int] = []

    async def page(after_id: str, size: int) -> list[Job]:
        start = int(after_id) + 1 if after_id else 0
        visited.append(start)
        return [
            replace(template, id=f"{index:06d}") for index in range(start, min(total, start + size))
        ]

    repo.list_rescreen_jobs.side_effect = page
    monkeypatch.setattr(maintenance, "RESCREEN_LIMIT", limit)
    monkeypatch.setattr(maintenance, "_reclassifications", lambda jobs: (tuple(jobs), ()))
    result = await maintenance.rescreen(repo, now=NOW)
    assert result.examined == result.total == total and result.complete
    assert result.quarantine_waiting == total and result.skipped == 0
    assert not result.recovery_complete
    assert visited[-2:] == [100_000, 100_001]
