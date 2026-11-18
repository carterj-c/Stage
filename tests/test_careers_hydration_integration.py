import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stage.classify.scope import Rejection, to_quarantined
from stage.domain import (
    DetailFetch,
    Job,
    JobFilters,
    JobLevel,
    QuarantineFilters,
    RejectionReason,
    RoleCategory,
)
from stage.domain.industry import Industry
from stage.services import maintenance
from stage.services.screening import normalize_batch
from stage.storage import SourceBatch, migrations, open_repository
from stage.storage.sqlite_repo import SqliteRepository

NOW = datetime(2026, 11, 18, tzinfo=UTC)


def _job(source: str, board: str, ident: str) -> Job:
    return Job(
        id=f"{source}:{board}:{ident}",
        source=source,
        company="Royal Bank of Canada",
        title_raw="Senior Investment Banking Analyst",
        title_normalized="senior investment banking analyst",
        apply_url_raw=f"https://example.test/{source}/{board}/{ident}",
        description="",
        location_raw="Montreal, QC",
        first_seen=NOW,
        last_seen=NOW,
    )


@pytest.mark.parametrize("source", ["workday", "smartrecruiters", "jobvite"])
async def test_finance_senior_hydration_preserves_roles_scope_cursor_and_auth(
    db_path: Path, source: str
) -> None:
    raw = (_job(source, "rbc-careers", "a"), _job(source, "rbc-careers", "b"))
    jobs, rejected = normalize_batch(raw)
    assert not rejected
    scope = [f"{source}:rbc-careers"]
    filters = JobFilters(
        industries=frozenset({Industry.FINANCE}),
        role=RoleCategory.INVESTMENT_BANKING,
        level=JobLevel.SENIOR,
        limit=None,
    )
    async with open_repository(db_path) as repo:
        other, rejected = normalize_batch(
            [replace(_job(source, "other", "z"), first_seen=NOW + timedelta(hours=1))]
        )
        assert not rejected
        await repo.apply_source_batch(
            SourceBatch(source=source, run_started_at=NOW, jobs=(*jobs, *other))
        )
        assert await repo.detail_queue(source, 1, scope) == [jobs[0].id]
        await repo.apply_source_batch(
            SourceBatch(source=source, run_started_at=NOW, detail_queue_cursor=jobs[0].id)
        )
    async with open_repository(db_path) as repo:
        assert await repo.detail_queue(source, 1, scope) == [jobs[1].id]
        stored = await repo.get_job(jobs[0].id)
        assert stored is not None and stored.work_auth_flag is None
        assert stored.roles == (RoleCategory.FINANCE, RoleCategory.INVESTMENT_BANKING)
        assert stored.level is JobLevel.SENIOR
        for job, body, auth in (
            (jobs[0], "Manage investments. Must be a US citizen.", True),
            (jobs[1], "Manage investments and advise clients.", False),
        ):
            hydrated, rejected = normalize_batch([replace(job, description=body)])
            assert not rejected and hydrated[0].work_auth_flag is auth
            await repo.apply_source_batch(
                SourceBatch(
                    source=source,
                    run_started_at=NOW + timedelta(hours=2),
                    jobs=hydrated,
                    detail_fetches=(DetailFetch(id=job.id, resolved=True),),
                    detail_queue_cursor=job.id,
                )
            )
        assert await repo.detail_queue(source, 10, scope) == []
        assert await repo.detail_queue(source, 1, [f"{source}:other"]) == [other[0].id]
        assert {j.id for j in await repo.search_jobs("investments", filters)} == {
            j.id for j in jobs
        }
        assert await repo.count_search("investments", filters) == 2
        assert await repo.description_coverage() == {source: (3, 2)}
    async with open_repository(db_path) as repo:
        for job, auth in zip(jobs, (True, False), strict=True):
            stored = await repo.get_job(job.id)
            assert stored is not None and stored.work_auth_flag is auth
            assert stored.roles == job.roles and stored.level is JobLevel.SENIOR
        assert await repo.count_jobs(filters) == 3


@pytest.mark.parametrize("baseline", [1, 2, 3, 4])
async def test_shared_legacy_database_retains_quarantine_and_rearms_scoped_queue(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: int
) -> None:
    shipped = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: shipped[:baseline])
        with closing(sqlite3.connect(db_path)) as conn:
            migrations.migrate(conn, db_path)
            conn.execute(
                "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                "apply_url_raw, location_raw, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "workday:rbc-careers:a",
                    "workday",
                    "Royal Bank of Canada",
                    "Senior Investment Banking Analyst",
                    "senior investment banking analyst",
                    "https://example.test/a",
                    "Montreal, QC",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
            if baseline == 4:
                conn.execute("UPDATE jobs SET work_auth_known = 0")
                conn.execute(
                    "INSERT INTO detail_queue_cursors VALUES (?, ?)",
                    ("workday", "workday:rbc-careers:a"),
                )
            conn.commit()
    unsafe = replace(
        _job("workday", "rbc-careers", "held"),
        title_raw="Software Engineer Intern",
        title_normalized="software engineer intern",
    )
    held = to_quarantined(unsafe, Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "PhD required"))
    scope = ["workday:rbc-careers"]
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW, quarantined=(held,))
        )
        for _ in range(3):
            await repo.apply_source_batch(
                SourceBatch(
                    source="workday",
                    run_started_at=NOW,
                    detail_fetches=(DetailFetch(id=held.id, resolved=True),),
                )
            )
        await repo.mark_screened(0)
        result = await maintenance.rescreen(repo, now=NOW + timedelta(hours=1))
        assert result.complete and result.released == 0 and result.quarantine_waiting == 1
        fresh, rejected = normalize_batch([unsafe])
        assert not rejected
        await repo.apply_source_batch(
            SourceBatch(source="workday", run_started_at=NOW + timedelta(hours=2), jobs=fresh)
        )
        assert await repo.get_job(held.id) is None
        (saved,) = await repo.list_quarantined(QuarantineFilters(limit=None))
        assert (saved.id, saved.reason, saved.matched_phrase, saved.first_seen) == (
            held.id,
            held.reason,
            held.matched_phrase,
            held.first_seen,
        )
        assert set(await repo.detail_queue("workday", 10, scope)) == {
            "workday:rbc-careers:a",
            held.id,
        }
        senior = await repo.get_job("workday:rbc-careers:a")
        assert senior is not None and senior.work_auth_flag is None
        assert senior.level is JobLevel.SENIOR
        assert senior.roles == (RoleCategory.FINANCE, RoleCategory.INVESTMENT_BANKING)
    with closing(SqliteRepository.connect(db_path)) as repo:
        assert migrations.applied_versions(repo._conn) == tuple(
            range(1, migrations.latest_version() + 1)
        )
        assert migrations.migrate(repo._conn, db_path) == ()
        assert repo._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        row = repo._conn.execute(
            "SELECT attempts FROM detail_fetches WHERE id = ?", (held.id,)
        ).fetchone()
        assert row[0] == 3
        if baseline == 4:
            assert (
                repo._conn.execute(
                    "SELECT cursor FROM detail_queue_cursors WHERE source = 'workday'"
                ).fetchone()[0]
                == "workday:rbc-careers:a"
            )
