import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest

from stage.storage.migrations import (
    SchemaVersionError,
    applied_versions,
    latest_version,
    migrate,
    snapshot_path,
)
from stage.storage.sqlite_repo import SqliteRepository, read_only_uri


def _snapshots(db_path: Path) -> list[Path]:
    return sorted(db_path.parent.glob(f"{db_path.name}.bak-*"))


def test_fresh_database_migrates_without_snapshotting(db_path: Path) -> None:
    repository = SqliteRepository.connect(db_path)
    try:
        assert applied_versions(repository._conn) == tuple(range(1, latest_version() + 1))
    finally:
        repository.close()

    assert _snapshots(db_path) == []


def test_migrating_twice_is_a_no_op(db_path: Path) -> None:
    SqliteRepository.connect(db_path).close()
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        assert migrate(conn, db_path) == ()
    finally:
        conn.close()

    assert _snapshots(db_path) == []


def test_populated_database_is_snapshotted_before_a_new_migration(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.storage import migrations

    SqliteRepository.connect(db_path).close()
    assert _snapshots(db_path) == []

    shipped = migrations.discover()
    next_version = shipped[-1].version + 1
    extra = tmp_path / f"{next_version:04d}_add_note.sql"
    extra.write_text("CREATE TABLE note (id INTEGER PRIMARY KEY);", encoding="utf-8")
    pending = (
        *shipped,
        migrations.Migration(version=next_version, name=extra.stem, path=extra),
    )
    monkeypatch.setattr(migrations, "discover", lambda: pending)

    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        assert migrate(conn, db_path) == (next_version,)
        assert applied_versions(conn) == tuple(range(1, next_version + 1))
        conn.execute("SELECT id FROM note").fetchall()
    finally:
        conn.close()

    assert len(_snapshots(db_path)) == 1


def test_database_newer_than_this_build_is_refused(db_path: Path) -> None:
    SqliteRepository.connect(db_path).close()
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (9999, "from_the_future", datetime.now(UTC).isoformat()),
        )
        with pytest.raises(SchemaVersionError, match="newer version of Stage"):
            migrate(conn, db_path)
    finally:
        conn.close()


def test_a_read_only_connection_refuses_every_write(db_path: Path) -> None:
    from stage.domain import SyncOutcome, SyncRun

    SqliteRepository.connect(db_path).close()
    reader = SqliteRepository.connect(db_path, read_only=True)
    try:
        assert reader._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        moment = datetime.now(UTC)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.record_sync_run(
                SyncRun(started_at=moment, finished_at=moment, outcome=SyncOutcome.SUCCESS)
            )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._conn.execute("DELETE FROM jobs")
    finally:
        reader.close()


def test_a_read_only_open_of_an_older_database_migrates_it_first(db_path: Path) -> None:
    sqlite3.connect(db_path).close()
    repository = SqliteRepository.connect(db_path, read_only=True)
    try:
        assert applied_versions(repository._conn) == tuple(range(1, latest_version() + 1))
    finally:
        repository.close()
    reader = SqliteRepository.connect(db_path, read_only=True)
    try:
        assert reader._conn.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        reader.close()


def test_a_read_only_open_refuses_a_newer_database(db_path: Path) -> None:
    SqliteRepository.connect(db_path).close()
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (9999, "from_the_future", datetime.now(UTC).isoformat()),
        )
    finally:
        conn.close()
    with pytest.raises(SchemaVersionError, match="newer version of Stage"):
        SqliteRepository.connect(db_path, read_only=True)


def test_a_read_only_open_reads_beside_a_writer_and_a_space_in_the_path(tmp_path: Path) -> None:
    path = tmp_path / "Application Support" / "stage.db"
    writer = SqliteRepository.connect(path)
    try:
        writer._conn.execute("BEGIN IMMEDIATE")
        reader = SqliteRepository.connect(path, read_only=True)
        try:
            assert reader._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        finally:
            reader.close()
        writer._conn.rollback()
    finally:
        writer.close()


def test_snapshot_path_is_derived_from_the_database_name(db_path: Path) -> None:
    when = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    assert snapshot_path(db_path, when).name == f"{db_path.name}.bak-20260731T120000"


@pytest.mark.parametrize(
    ("table", "drop"),
    (
        ("coverage_classifications", "DROP TABLE coverage_classifications"),
        ("workday_crawls", "DROP TABLE workday_crawls"),
        ("workday_crawl_seen", "DROP TABLE workday_crawl_seen"),
    ),
)
def test_the_console_reports_a_prebaseline_database_without_a_traceback(
    db_path: Path, table: str, drop: str
) -> None:
    SqliteRepository.connect(db_path).close()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(drop)
        conn.commit()
    finally:
        conn.close()

    code = "import sys\nfrom stage.cli.app import main\nsys.argv = ['stage', 'coverage']\nmain()\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "STAGE_DB": str(db_path)},
        check=False,
    )

    assert result.returncode == 2
    assert table in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("path", "uri"),
    [
        (PurePosixPath("/home/a b/stage.db"), "file:///home/a%20b/stage.db?mode=ro"),
        (PureWindowsPath("C:/Users/a/stage.db"), "file:///C:/Users/a/stage.db?mode=ro"),
        (PureWindowsPath("//server/share/stage.db"), "file:////server/share/stage.db?mode=ro"),
    ],
)
def test_a_read_only_uri_suits_sqlite_on_every_system(path: PurePath, uri: str) -> None:
    assert read_only_uri(path) == uri, "SQLite refuses a host in a file URI, so a share needs ////"


def test_a_locked_database_is_reported_rather_than_opened_for_writing() -> None:
    from typing import cast

    from stage.storage.migrations import is_current

    class Locked:
        def execute(self, *_: object) -> None:
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        is_current(cast(sqlite3.Connection, Locked()), Path("stage.db"))


def test_a_baseline_another_process_applied_first_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import closing

    from stage.storage import migrations
    from stage.storage.sqlite_repo import SqliteRepository

    target = tmp_path / "stage.db"
    SqliteRepository.connect(target).close()
    real = migrations.applied_versions
    seen: list[int] = []

    def stale(conn: sqlite3.Connection) -> tuple[int, ...]:
        seen.append(1)
        return () if len(seen) == 1 else real(conn)

    monkeypatch.setattr(migrations, "applied_versions", stale)
    with closing(sqlite3.connect(target, isolation_level=None)) as conn:
        migrations.migrate(conn, target)

    assert target.exists() and len(seen) > 1


def test_a_database_that_is_not_stages_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    import hashlib
    from contextlib import closing

    from stage.storage.migrations import SchemaVersionError
    from stage.storage.sqlite_repo import SqliteRepository

    other = tmp_path / "other.sqlite"
    with closing(sqlite3.connect(other)) as conn:
        conn.execute("CREATE TABLE users (name TEXT)")
        conn.commit()
    other.chmod(0o644)
    before = hashlib.sha256(other.read_bytes()).hexdigest()

    with pytest.raises(SchemaVersionError, match="not a Stage database"):
        SqliteRepository.connect(other)

    assert hashlib.sha256(other.read_bytes()).hexdigest() == before
    assert other.stat().st_mode & 0o777 == 0o644 or os.name == "nt"
    assert not other.with_name("other.sqlite-wal").exists()


def test_a_hidden_file_beside_the_migrations_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from stage.storage import migrations

    expected = [migration.name for migration in migrations.discover()]
    folder = tmp_path / "migrations"
    shutil.copytree(migrations._MIGRATIONS_DIR, folder)
    (folder / "._0001_initial.sql").write_bytes(b"\x00\x05\x16\x07Mac OS X")
    monkeypatch.setattr(migrations, "_MIGRATIONS_DIR", folder)

    assert [migration.name for migration in migrations.discover()] == expected


def test_feed_summary_migration_preserves_existing_jobs_and_quarantine(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stage.storage import migrations

    shipped = migrations.discover()
    previous = tuple(entry for entry in shipped if entry.version < 7)
    assert [entry.version for entry in previous] == [1, 2, 3, 4, 5, 6]
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: previous)
        old = SqliteRepository.connect(db_path)
        old._conn.execute(
            "INSERT INTO jobs (id, source, company, title_raw, title_normalized, apply_url_raw, "
            "description, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feed:acme:1",
                "feed",
                "Acme",
                "Intern",
                "intern",
                "https://example.com/1",
                "Full original detail",
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ),
        )
        old._conn.execute(
            "INSERT INTO quarantine (id, source, company, title_raw, reason, "
            "first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "feed:acme:2",
                "feed",
                "Acme",
                "Intern",
                "out-of-scope-degree",
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ),
        )
        old.close()
    upgraded = SqliteRepository.connect(db_path)
    try:
        job = upgraded.get_job("feed:acme:1")
        assert (
            job is not None and job.description == "Full original detail" and job.feed_summary == ""
        )
        summary = upgraded._conn.execute(
            "SELECT feed_summary, verified_description FROM quarantine WHERE id = ?",
            ("feed:acme:2",),
        ).fetchone()
        assert summary is not None and tuple(summary) == ("", "")
        assert applied_versions(upgraded._conn) == tuple(range(1, 9))
    finally:
        upgraded.close()
    assert len(_snapshots(db_path)) == 1


def test_quarantine_evidence_migration_preserves_direct_hanzili_rejections(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.storage import migrations

    shipped = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: tuple(m for m in shipped if m.version < 8))
        old = SqliteRepository.connect(db_path)
        for ident, matched, summary in (
            ("hanzili:1", "PhD", "PhD required"),
            ("hanzili:2", "copy of another rejected for phd", "PhD required"),
            ("hanzili:3", "PhD", ""),
        ):
            old._conn.execute(
                "INSERT INTO quarantine (id, source, company, title_raw, reason, matched_phrase, "
                "feed_summary, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ident,
                    "hanzili",
                    "Acme",
                    "Intern",
                    "out-of-scope-degree",
                    matched,
                    summary,
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-02T00:00:00+00:00",
                ),
            )
        old._conn.execute(
            "INSERT INTO quarantine (id, source, company, title_raw, reason, matched_phrase, "
            "feed_summary, verified_description, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "hanzili:4",
                "hanzili",
                "Acme",
                "Intern",
                "out-of-scope-degree",
                "PhD",
                "Hybrid",
                "Verified PhD required",
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
            ),
        )
        old.close()
    upgraded = SqliteRepository.connect(db_path)
    try:
        rows = upgraded._conn.execute(
            "SELECT id, feed_summary, rejection_evidence FROM quarantine ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("hanzili:1", "PhD required", "PhD required"),
            ("hanzili:2", "PhD required", ""),
            ("hanzili:3", "", ""),
            ("hanzili:4", "Hybrid", "Verified PhD required"),
        ]
    finally:
        upgraded.close()


def test_version_six_upgrade_preserves_oracle_provenance_career_and_queue_state(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.domain import JobLevel, RoleCategory
    from stage.storage import migrations

    shipped = migrations.discover()
    ident = "oracle_cloud:acme:verified"
    when = "2026-11-18T12:00:00+00:00"
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: tuple(m for m in shipped if m.version <= 6))
        old = SqliteRepository.connect(db_path)
        try:
            old._conn.execute(
                "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                "apply_url_raw, description, role, roles, level, work_auth_flag, "
                "work_auth_known, first_seen, "
                "last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ident,
                    "oracle_cloud",
                    "Acme",
                    "Senior Finance Analyst",
                    "senior finance analyst",
                    "https://example.com/1",
                    "Verified full detail",
                    "finance",
                    '["finance"]',
                    "senior",
                    1,
                    1,
                    when,
                    when,
                ),
            )
            old._conn.execute("INSERT INTO oracle_full_details (id) VALUES (?)", (ident,))
            old._conn.execute(
                "INSERT INTO detail_fetches (id, source, fetched_at, resolved, failed) "
                "VALUES (?, ?, ?, 1, 0)",
                (ident, "oracle_cloud", when),
            )
            old._conn.execute(
                "INSERT INTO detail_queue_cursors (source, cursor) VALUES (?, ?)",
                ("oracle_cloud:acme", ident),
            )
            old._conn.execute(
                "INSERT INTO quarantine (id, source, company, title_raw, reason, matched_phrase, "
                "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "bamboohr:acme:held",
                    "bamboohr",
                    "Acme",
                    "Engineer Intern",
                    "out-of-scope-degree",
                    "PhD required",
                    when,
                    when,
                ),
            )
        finally:
            old.close()
    repo = SqliteRepository.connect(db_path)
    try:
        stored = repo.get_job(ident)
        assert stored is not None and stored.description == "Verified full detail"
        assert stored.role is RoleCategory.FINANCE and stored.roles == (RoleCategory.FINANCE,)
        assert stored.level is JobLevel.SENIOR and stored.work_auth_flag is True
        assert stored.feed_summary == ""
        assert repo._conn.execute("SELECT id FROM oracle_full_details").fetchone()[0] == ident
        assert repo._conn.execute("SELECT cursor FROM detail_queue_cursors").fetchone()[0] == ident
        held = repo._conn.execute(
            "SELECT matched_phrase, feed_summary, verified_description, rejection_evidence "
            "FROM quarantine WHERE id = ?",
            ("bamboohr:acme:held",),
        ).fetchone()
        assert tuple(held) == ("PhD required", "", "", "")
        assert applied_versions(repo._conn) == tuple(range(1, 9))
        assert repo._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrations.migrate(repo._conn, db_path) == ()
    finally:
        repo.close()
