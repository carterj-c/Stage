from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stage.classify import classify_role
from stage.classify.level import screen_level
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


async def test_degree_quarantine_waits_for_complete_source_refresh(tmp_path: Path) -> None:
    from stage.classify.scope import Rejection, to_quarantined

    unsafe = posting("Software Engineer Intern", "PhD required. Must be a US citizen.")
    eligible = posting("Quantitative Research Intern", "PhD required. Must be a US citizen.")
    entries = tuple(
        to_quarantined(job, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "phd required"))
        for job in (unsafe, eligible)
    )
    async with open_repository(tmp_path / "release.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=entries)
        )
        result = await rescreen(repo, now=NOW)
        assert result.released == 0
        assert await repo.list_jobs(JobFilters()) == []
        held = await repo.list_quarantined(QuarantineFilters(limit=None))
        assert {entry.id for entry in held} == {job.id for job in (unsafe, eligible)}
        assert all(entry.reason is RejectionReason.OUT_OF_SCOPE_DEGREE for entry in held)
        incomplete, _ = normalize_batch([replace(eligible, description="")])
        assert len(incomplete) == 1
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=incomplete)
        )
        assert await repo.get_job(eligible.id) is None
        assert len(await repo.list_quarantined(QuarantineFilters(limit=None))) == 2
        accepted, rejected = normalize_batch([unsafe, eligible])
        assert [entry.id for entry in rejected] == [unsafe.id]
        assert [job.id for job in accepted] == [eligible.id]
        await repo.apply_source_batch(
            SourceBatch(
                source="greenhouse", run_started_at=NOW, jobs=accepted, quarantined=rejected
            )
        )
        refreshed = await repo.get_job(eligible.id)
        assert refreshed is not None
        assert refreshed.description == eligible.description
        assert refreshed.degree_requirement is DegreeRequirement.PHD
        assert refreshed.work_auth_flag
        assert [
            entry.id for entry in await repo.list_quarantined(QuarantineFilters(limit=None))
        ] == [unsafe.id]


@pytest.mark.parametrize("category", ["Software", "Quant"])
def test_title_occupation_wins_source_category(category: str) -> None:
    job = replace(posting("Project Manager"), signals=SourceSignals(category=category))
    accepted, rejected = normalize_batch([job])
    assert not rejected
    assert accepted[0].role is RoleCategory.PROJECT_MANAGEMENT


def test_exempt_occupation_survives_as_nonprimary_role() -> None:
    job = posting("Software Engineering Project Manager", "PhD required.")
    accepted, rejected = normalize_batch([job])
    assert not rejected
    assert accepted[0].role is RoleCategory.SWE
    assert RoleCategory.PROJECT_MANAGEMENT in accepted[0].roles
    assert accepted[0].degree_requirement is DegreeRequirement.PHD


@pytest.mark.parametrize("title", ["Project Manager", "Program Manager"])
def test_structured_intern_manager_occupations(title: str) -> None:
    job = replace(posting(title), signals=SourceSignals(employment_type="Intern"))
    accepted, rejected = normalize_batch([job])
    assert not rejected
    assert accepted[0].level is JobLevel.INTERNSHIP


@pytest.mark.parametrize(
    "title",
    [
        "Manager of Interns",
        "Internship Program Manager",
        "Senior Project Manager",
        "Engineering Manager",
    ],
)
def test_structured_intern_does_not_override_manager_disqualifiers(title: str) -> None:
    assert screen_level(title, employment_type="Intern").level is not JobLevel.INTERNSHIP


@pytest.mark.parametrize("primary", [RoleCategory.PROJECT_MANAGEMENT, RoleCategory.SWE])
async def test_duplicate_roles_find_only_representatives_and_respect_other_filters(
    tmp_path: Path,
    primary: RoleCategory,
) -> None:
    canonical = replace(
        posting("Software Engineering Project Manager"),
        id="greenhouse:acme:1",
        role=RoleCategory.SWE,
        roles=(RoleCategory.SWE,),
        level=JobLevel.SENIOR,
    )
    duplicate = replace(
        canonical,
        id="lever:acme:2",
        source="lever",
        duplicate_of=canonical.id,
        role=primary,
        roles=(RoleCategory.PROJECT_MANAGEMENT,),
    )
    async with open_repository(tmp_path / "duplicates.db") as repo:
        await repo.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=(canonical,))
        )
        await repo.apply_source_batch(
            SourceBatch(source="lever", run_started_at=NOW, jobs=(duplicate,))
        )
        filters = JobFilters(role=RoleCategory.PROJECT_MANAGEMENT, level=JobLevel.SENIOR)
        assert [job.id for job in await repo.list_jobs(filters)] == [canonical.id]
        assert await repo.count_jobs(filters) == 1
        assert [job.id for job in await repo.search_jobs("Project", filters)] == [canonical.id]
        assert await repo.count_search("Project", filters) == 1
        assert await repo.count_jobs(replace(filters, level=JobLevel.MID)) == 0
        assert await repo.count_search("Project", replace(filters, source="lever")) == 0
        assert await repo.count_jobs(replace(filters, status=JobStatus.CLOSED)) == 0
        assert [job.id for job in await repo.list_jobs(replace(filters, with_duplicates=True))] == [
            duplicate.id
        ]
        choices = JobFilters(roles=frozenset({RoleCategory.PROJECT_MANAGEMENT}))
        assert await repo.count_jobs(choices) == 1
        assert await repo.count_search("Project", choices) == 1


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
