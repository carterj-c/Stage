from datetime import UTC, datetime
from pathlib import Path

import pytest

from stage.classify import classify_role
from stage.classify.level import screen_level
from stage.domain import Job, JobFilters, JobLevel, RoleCategory
from stage.domain.filters import with_default_levels
from stage.domain.preferences import parse_levels
from stage.services.maintenance import rescreen
from stage.services.screening import normalize_batch
from stage.storage import open_repository
from stage.storage.repository import SourceBatch

NOW = datetime(2026, 11, 17, tzinfo=UTC)


def posting(title: str, description: str = "") -> Job:
    return Job(
        id=f"greenhouse:acme:{title}",
        source="greenhouse",
        company="Acme",
        title_raw=title,
        title_normalized=title.lower(),
        apply_url_raw=f"https://example.test/{title.replace(' ', '-')}",
        description=description,
        first_seen=NOW,
        last_seen=NOW,
        location_raw="Montreal, QC",
    )


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Project Manager", JobLevel.UNKNOWN),
        ("Program Manager", JobLevel.UNKNOWN),
        ("Senior Project Manager", JobLevel.SENIOR),
        ("Intermediate Game Producer", JobLevel.MID),
        ("Mid-Level Financial Analyst", JobLevel.MID),
        ("Project Management Intern", JobLevel.INTERNSHIP),
        ("New Grad Financial Analyst", JobLevel.NEW_GRAD),
    ],
)
def test_title_levels(title: str, level: JobLevel) -> None:
    assert screen_level(title).level is level
    kept, rejected = normalize_batch([posting(title)])
    assert not rejected
    assert kept[0].level is level


@pytest.mark.parametrize(
    ("title", "role"),
    [
        ("Project Manager", RoleCategory.PROJECT_MANAGEMENT),
        ("Technical Program Manager", RoleCategory.PROGRAM_MANAGEMENT),
        ("Senior Game Producer", RoleCategory.PRODUCTION),
        ("Financial Analyst", RoleCategory.FINANCE),
        ("Equity Research Analyst", RoleCategory.INVESTMENT_RESEARCH),
        ("Investment Banking Analyst", RoleCategory.INVESTMENT_BANKING),
        ("Market Risk Analyst", RoleCategory.RISK),
    ],
)
def test_occupations(title: str, role: RoleCategory) -> None:
    verdict = classify_role(title)
    assert role in verdict.roles


def test_industry_and_product_do_not_imply_occupation() -> None:
    assert classify_role("Product Manager").role is RoleCategory.UNKNOWN
    assert (
        classify_role("Software Engineer", "Game production in investment banking").role
        is RoleCategory.SWE
    )
    assert classify_role("Production Line Operator").role is not RoleCategory.PRODUCTION


@pytest.mark.parametrize(
    "title",
    [
        "Quantitative Researcher",
        "Quantitative Research Intern",
        "Senior Financial Analyst",
        "Project Manager",
    ],
)
def test_graduate_degrees_are_metadata(title: str) -> None:
    kept, rejected = normalize_batch([posting(title, "PhD required. Master's degree required.")])
    assert not rejected
    assert len(kept) == 1


async def test_durable_mixed_roles_default_and_rescreen(tmp_path: Path) -> None:
    path = tmp_path / "broader.db"
    jobs, rejected = normalize_batch(
        [
            posting("Senior Project Manager / Program Manager"),
            posting("Software Engineering Intern"),
            posting("Project Manager"),
        ]
    )
    assert not rejected
    async with open_repository(path) as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=jobs)
        )
    async with open_repository(path) as repo:
        for role in (RoleCategory.PROJECT_MANAGEMENT, RoleCategory.PROGRAM_MANAGEMENT):
            found = await repo.list_jobs(JobFilters(role=role, level=JobLevel.SENIOR))
            assert len(found) == 1
            assert len(await repo.search_jobs("Senior", JobFilters(role=role))) == 1
            assert len(await repo.list_jobs(JobFilters(roles=frozenset({role})))) >= 1
        default = await repo.list_jobs(with_default_levels(JobFilters()))
        assert [job.level for job in default] == [JobLevel.INTERNSHIP]
        assert len(await repo.list_jobs(JobFilters(level=JobLevel.UNKNOWN))) == 1
        await rescreen(repo, now=NOW)
        assert len(await repo.list_jobs(JobFilters(role=RoleCategory.PROGRAM_MANAGEMENT))) == 1
        assert len(await repo.list_jobs(JobFilters())) == 3
    assert parse_levels(["mid,senior,unknown"]) == (JobLevel.MID, JobLevel.SENIOR, JobLevel.UNKNOWN)


async def test_old_degree_quarantine_reconsidered(tmp_path: Path) -> None:
    from stage.classify.scope import Rejection, to_quarantined
    from stage.domain import RejectionReason

    job = posting("Quantitative Researcher")
    entry = to_quarantined(job, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "phd"))
    async with open_repository(tmp_path / "release.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=(entry,))
        )
        result = await rescreen(repo, now=NOW)
        assert result.released == 1
        assert (await repo.list_jobs(JobFilters()))[0].level is JobLevel.UNKNOWN


@pytest.mark.parametrize("baseline", [1, 2])
def test_upgrade_preserves_legacy_default_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: int
) -> None:
    import sqlite3

    from stage.storage import migrations
    from stage.storage.sqlite_repo import SqliteRepository

    path = tmp_path / "legacy.db"
    shipped = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: shipped[:baseline])
        conn = sqlite3.connect(path)
        migrations.migrate(conn, path)
        conn.execute(
            "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
            "apply_url_raw, role, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "greenhouse",
                "Acme",
                "Software Engineer Intern",
                "software intern",
                "https://example.test/legacy",
                "swe",
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    repo = SqliteRepository.connect(path)
    try:
        jobs = repo.list_jobs(with_default_levels(JobFilters()))
        assert len(jobs) == 1
        assert jobs[0].roles == ()
        assert jobs[0].role is RoleCategory.SWE
        assert jobs[0].level is JobLevel.INTERNSHIP
        assert len(repo.search_jobs("Software", with_default_levels(JobFilters()))) == 1
        assert migrations.migrate(repo._conn, path) == ()
    finally:
        repo.close()
    assert len(list(tmp_path.glob("legacy.db.bak-*"))) == 1
