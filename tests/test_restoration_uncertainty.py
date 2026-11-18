import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stage.ai.shape import posting_detail
from stage.classify import resolve_eligibility
from stage.classify.scope import Rejection, to_quarantined
from stage.cli.serialize import jobs_to_json
from stage.domain import DegreeRequirement, Job, JobFilters, JobLevel, RejectionReason
from stage.services.query import PostingDetail
from stage.services.screening import normalize_batch
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 11, 18, tzinfo=UTC)


def posting(title: str) -> Job:
    return Job(
        id="simplify-new-grad:feed:1",
        source="simplify-new-grad",
        company="Acme",
        title_raw=title,
        title_normalized=title.lower(),
        apply_url_raw="https://example.test/1",
        description="",
        location_raw="Montreal, QC",
        first_seen=NOW,
        last_seen=NOW,
    )


@pytest.mark.parametrize("title", ["Quantitative Researcher", "Project Manager", "Risk Manager"])
@pytest.mark.parametrize(
    "reason",
    [
        RejectionReason.NOT_EARLY_CAREER,
        RejectionReason.OUT_OF_SCOPE_LOCATION,
        RejectionReason.OUT_OF_SCOPE_DEGREE,
    ],
)
async def test_fresh_descriptionless_eligible_roles_restore_with_unknown_evidence(
    tmp_path: Path,
    title: str,
    reason: RejectionReason,
) -> None:
    job = posting(title)
    entry = to_quarantined(job, Rejection(reason, "old rejection"))
    path = tmp_path / "feed.db"
    async with open_repository(path) as repo:
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, quarantined=(entry,))
        )
        assert resolve_eligibility(job).work_auth_flag is None
        kept, rejected = normalize_batch([job])
        assert not rejected
        await repo.apply_source_batch(SourceBatch(source=job.source, run_started_at=NOW, jobs=kept))
    async with open_repository(path) as repo:
        restored = await repo.get_job(job.id)
        assert restored is not None
        assert restored.degree_requirement is DegreeRequirement.UNKNOWN
        assert restored.work_auth_flag is None
        assert json.loads(jobs_to_json([restored]))[0]["work_auth_flag"] is None
        assert (
            posting_detail(PostingDetail(restored, (), None), None)["work_auth_restricted"] is None
        )
        assert len(await repo.list_jobs(JobFilters(level=JobLevel.UNKNOWN))) == 1


@pytest.mark.parametrize(
    "reason",
    [
        RejectionReason.NOT_EARLY_CAREER,
        RejectionReason.OUT_OF_SCOPE_DEGREE,
        RejectionReason.OUT_OF_SCOPE_LOCATION,
    ],
)
async def test_fresh_descriptionless_legacy_intern_stays_held(
    tmp_path: Path,
    reason: RejectionReason,
) -> None:
    job = posting("Software Engineer Intern")
    entry = to_quarantined(job, Rejection(reason, "old"))
    async with open_repository(tmp_path / "held.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, quarantined=(entry,))
        )
        kept, _ = normalize_batch([job])
        assert kept[0].work_auth_flag is None
        await repo.apply_source_batch(SourceBatch(source=job.source, run_started_at=NOW, jobs=kept))
        assert await repo.get_job(job.id) is None
        refreshed = replace(job, description="PhD required. Must be a US citizen.")
        kept, rejected = normalize_batch([refreshed])
        assert not kept and rejected[0].reason is RejectionReason.OUT_OF_SCOPE_DEGREE


def test_migration_keeps_existing_restrictions_and_marks_missing_bodies_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.storage import migrations
    from stage.storage.sqlite_repo import SqliteRepository

    path = tmp_path / "legacy.db"
    shipped = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: shipped[:3])
        conn = sqlite3.connect(path)
        migrations.migrate(conn, path)
        for number, body, restricted in [(1, "", 0), (2, "", 1), (3, "body", 0)]:
            conn.execute(
                "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                "apply_url_raw, first_seen, last_seen, description, work_auth_flag) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(number),
                    "greenhouse",
                    "Acme",
                    "Engineer",
                    "engineer",
                    "",
                    NOW.isoformat(),
                    NOW.isoformat(),
                    body,
                    restricted,
                ),
            )
        conn.commit()
        conn.close()
    repo = SqliteRepository.connect(path)
    try:
        assert [repo.get_job(str(i)).work_auth_flag for i in (1, 2, 3)] == [None, True, False]  # type: ignore[union-attr]
    finally:
        repo.close()
    assert len(list(tmp_path.glob("legacy.db.bak-*"))) == 1


async def test_raw_descriptionless_refresh_cannot_fabricate_auth_status(tmp_path: Path) -> None:
    job = posting("Project Manager")
    kept, _ = normalize_batch([job])
    raw = replace(kept[0], work_auth_flag=False)
    held = to_quarantined(job, Rejection(RejectionReason.NOT_EARLY_CAREER, "old"))
    async with open_repository(tmp_path / "raw.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, quarantined=(held,))
        )
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, jobs=(raw,))
        )
        restored = await repo.get_job(job.id)
        assert restored is not None and restored.work_auth_flag is None


@pytest.mark.parametrize("title", ["Senior Software Engineer", "Junior Software Engineer"])
@pytest.mark.parametrize(
    "reason", [RejectionReason.NOT_EARLY_CAREER, RejectionReason.OUT_OF_SCOPE_LOCATION]
)
async def test_fresh_nonintern_legacy_prefilter_rejection_restores_with_unknown_evidence(
    tmp_path: Path,
    title: str,
    reason: RejectionReason,
) -> None:
    job = posting(title)
    entry = to_quarantined(job, Rejection(reason, "old"))
    async with open_repository(tmp_path / "nonintern.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, quarantined=(entry,))
        )
        kept, rejected = normalize_batch([job])
        assert len(kept) == 1 and not rejected
        await repo.apply_source_batch(SourceBatch(source=job.source, run_started_at=NOW, jobs=kept))
        restored = await repo.get_job(job.id)
        assert restored is not None
        assert restored.degree_requirement is DegreeRequirement.UNKNOWN
        assert restored.work_auth_flag is None


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("Senior Software Engineer", RejectionReason.OUT_OF_SCOPE_DEGREE),
        ("Software Engineer", RejectionReason.NOT_EARLY_CAREER),
    ],
)
async def test_fresh_legacy_listing_preserves_known_degree_or_uncertain_internship(
    tmp_path: Path,
    title: str,
    reason: RejectionReason,
) -> None:
    job = posting(title)
    entry = to_quarantined(job, Rejection(reason, "old evidence"))
    async with open_repository(tmp_path / "sensitive.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source=job.source, run_started_at=NOW, quarantined=(entry,))
        )
        kept, rejected = normalize_batch([job])
        assert len(kept) == 1 and not rejected
        await repo.apply_source_batch(SourceBatch(source=job.source, run_started_at=NOW, jobs=kept))
        assert await repo.get_job(job.id) is None
