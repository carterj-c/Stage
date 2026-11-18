import json
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from stage.domain import (
    UNKNOWN_TERM,
    Job,
    QuarantinedJob,
    RateState,
    RejectionReason,
    RoleCategory,
    WorkdayCrawl,
    WorkdayFacet,
)
from stage.domain.hydration_target import posting_identity
from stage.domain.liveness import (
    AWAITING_PAGE,
    BOARD_SOURCES,
    CHECK_PREFIX,
    Liveness,
    LivenessCheck,
)
from stage.domain.text import LOCATION_COUNT_GLOB
from stage.domain.visits import (
    FEED_DECISION_PREFIX,
    FEED_DETAIL_PREFIX,
    FEED_PENDING_PREFIX,
    FEED_REJECTION_PREFIX,
    ORACLE_DETAIL_REFRESH_DAYS,
)
from stage.storage.rows import (
    BAMBOO_QUARANTINE_REFRESH_DAYS,
    CLOSED,
    OPEN,
    UPSERT_CHECK,
    UPSERT_PUSHBACK,
    job_from,
    quarantined_from,
    rate_state_params,
)
from stage.storage.sql import (
    EACH,
    Store,
    count,
    feed_marker_snapshot,
    from_text,
    link_of,
    listed,
    to_text,
    verified_feed_hold,
)
from stage.storage.upkeep import promote_open_copies, remember_heirs

MAX_DETAIL_ATTEMPTS = 3
REREAD_ATTEMPTS = 2
BAMBOO_QUARANTINE_REFRESH = timedelta(days=BAMBOO_QUARANTINE_REFRESH_DAYS)

_QUEUED = (
    "FROM jobs j LEFT JOIN detail_fetches d ON d.id = j.id "
    "WHERE j.source = ? AND j.status = ? AND ("
    "((j.description = '' OR j.source = 'oracle_cloud') AND (j.source IN "
    "('workday', 'smartrecruiters', 'jobvite', 'bamboohr', 'oracle_cloud') "
    "OR j.term = ? OR j.role = ? OR j.work_auth_known = 0) "
    "AND (j.source <> 'oracle_cloud' OR d.id IS NULL OR d.resolved = 0 "
    "OR d.fetched_at < (SELECT fetched_at FROM detail_fetches WHERE id = 'rules:changed')) "
    "AND (d.id IS NULL OR (d.failed = 1 AND d.attempts < ?) "
    "OR (d.resolved = 1 AND d.attempts < ?) "
    "OR d.fetched_at < (SELECT fetched_at FROM detail_fetches WHERE id = 'rules:changed'))) "
    "OR (j.location_raw GLOB ? AND (d.id IS NULL OR d.attempts < ?)) "
    "OR (j.source = 'oracle_cloud' AND d.id IS NOT NULL "
    f"AND julianday(d.fetched_at) <= julianday(?) - {ORACLE_DETAIL_REFRESH_DAYS}))"
)


_AWAITING = (
    "FROM quarantine q LEFT JOIN detail_fetches d ON d.id = q.id "
    "WHERE q.source = ? AND (q.matched_phrase GLOB ? OR "
    "(q.source IN ('workday', 'smartrecruiters', 'bamboohr', 'oracle_cloud') "
    "AND q.reason IN (?, ?, ?, ?, ?, ?))) "
    "AND (d.id IS NULL OR (d.failed = 1 AND d.attempts < ?) "
    "OR (d.fetched_at < (SELECT fetched_at FROM detail_fetches WHERE id = ?)) "
    "OR (q.source = 'bamboohr' AND d.fetched_at <= ?) "
    "OR (q.source = 'oracle_cloud' "
    f"AND julianday(d.fetched_at) <= julianday(?) - {ORACLE_DETAIL_REFRESH_DAYS}))"
)
_QUEUE = (
    f"SELECT j.id, 1 AS rank, j.first_seen {_QUEUED} "
    f"UNION ALL SELECT q.id, 0 AS rank, q.first_seen {_AWAITING}"
)


def _queued(source: str, now: datetime) -> tuple[object, ...]:
    return (
        source,
        OPEN,
        UNKNOWN_TERM,
        RoleCategory.UNKNOWN.value,
        MAX_DETAIL_ATTEMPTS,
        REREAD_ATTEMPTS,
        LOCATION_COUNT_GLOB,
        REREAD_ATTEMPTS,
        to_text(now),
        source,
        f"*{AWAITING_PAGE}",
        RejectionReason.OUT_OF_SCOPE_DEGREE.value,
        RejectionReason.NOT_EARLY_CAREER.value,
        RejectionReason.NOT_AN_INTERNSHIP.value,
        RejectionReason.NOT_A_CS_ROLE.value,
        RejectionReason.UNKNOWN_CS_ROLE.value,
        RejectionReason.OUT_OF_SCOPE_LOCATION.value,
        MAX_DETAIL_ATTEMPTS,
        "rules:changed",
        to_text(now - BAMBOO_QUARANTINE_REFRESH),
        to_text(now),
    )


class DetailFetches(Store):
    def feed_hydration_queue(
        self,
        sources: Collection[str],
        now: datetime,
        limit: int,
        *,
        offset: int = 0,
        pending_sources: Collection[str] | None = None,
    ) -> list[Job | QuarantinedJob]:
        rows = self._conn.execute(
            "SELECT id, kind FROM (SELECT id, kind, row_number() OVER "
            "(PARTITION BY kind ORDER BY attempts, first_seen DESC, id) AS position FROM ("
            "SELECT j.id, 0 AS kind, coalesce(d.attempts, 0) AS attempts, j.first_seen "
            "FROM jobs j LEFT JOIN detail_fetches d ON d.id = ? || j.id "
            f"WHERE j.source IN {EACH} AND j.status = ? AND j.description = '' "
            "AND (d.id IS NULL OR (d.resolved = 0 AND d.attempts < ? AND d.fetched_at <= ?)) "
            "UNION ALL SELECT q.id, 1 AS kind, coalesce(d.attempts, 0) AS attempts, q.first_seen "
            "FROM quarantine q JOIN detail_fetches p ON p.id = ? || q.id "
            "JOIN source_visits v ON v.source = q.source AND v.board = q.source "
            "LEFT JOIN detail_fetches d ON d.id = ? || q.id "
            f"WHERE q.source IN {EACH} AND q.source = 'hanzili' "
            "AND p.resolved = 1 AND p.failed = 0 "
            "AND v.last_success_at IS NOT NULL AND v.consecutive_failures = 0 "
            "AND (d.id IS NULL OR (d.resolved = 0 AND d.attempts < ? AND d.fetched_at <= ?)) "
            f"AND q.rejection_evidence = '' AND NOT {verified_feed_hold('q')})) "
            "ORDER BY position, kind LIMIT ? OFFSET ?",
            (
                FEED_DETAIL_PREFIX,
                listed(sources),
                OPEN,
                MAX_DETAIL_ATTEMPTS,
                to_text(now - timedelta(hours=1)),
                FEED_PENDING_PREFIX,
                FEED_DETAIL_PREFIX,
                listed(sources if pending_sources is None else pending_sources),
                MAX_DETAIL_ATTEMPTS,
                to_text(now - timedelta(hours=1)),
                FEED_REJECTION_PREFIX,
                FEED_DECISION_PREFIX,
                FEED_DETAIL_PREFIX,
                max(0, limit),
                max(0, offset),
            ),
        ).fetchall()
        result: list[Job | QuarantinedJob] = []
        for row in rows:
            table = "jobs" if row["kind"] == 0 else "quarantine"
            item = self._conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (row["id"],)
            ).fetchone()
            if item is not None:
                result.append(job_from(item) if row["kind"] == 0 else quarantined_from(item))
        return result

    def detail_queue(
        self,
        source: str,
        limit: int,
        boards: Collection[str] | None = None,
        ids: Collection[str] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        if limit <= 0 or (boards is not None and not boards) or (ids is not None and not ids):
            return []
        filters: list[str] = []
        params: list[object] = []
        if boards is not None:
            filters.append(f"stage_board_of(id, ?) IN {EACH}")
            params.extend((source, listed(boards)))
        if ids is not None:
            filters.append(f"id IN {EACH}")
            params.append(listed(ids))
        scope = f"WHERE {' AND '.join(filters)} " if filters else ""
        rows = self._conn.execute(
            f"WITH candidates AS (SELECT id, rank, first_seen, stage_board_of(id, ?) AS board "
            f"FROM ({_QUEUE}) {scope}), "
            "positions AS (SELECT candidates.*, b.cursor AS board_cursor, "
            "s.cursor AS source_cursor, "
            "CASE WHEN EXISTS (SELECT 1 FROM quarantine WHERE id = b.cursor) "
            "THEN 0 ELSE 1 END AS cursor_rank "
            "FROM candidates LEFT JOIN detail_queue_cursors b ON b.source = candidates.board "
            "LEFT JOIN detail_queue_cursors s ON s.source = ?), "
            "ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY board ORDER BY "
            "CASE WHEN board_cursor IS NULL OR rank > cursor_rank "
            "OR (rank = cursor_rank AND id > board_cursor) THEN 0 ELSE 1 END, rank, "
            "CASE WHEN board_cursor IS NULL THEN first_seen ELSE '' END DESC, id) AS position "
            "FROM positions) "
            "SELECT id FROM ranked ORDER BY position, "
            "CASE WHEN source_cursor IS NULL THEN rank ELSE 0 END, "
            "CASE WHEN source_cursor IS NULL OR board > stage_board_of(source_cursor, ?) "
            "THEN 0 ELSE 1 END, board LIMIT ?",
            (source, *_queued(source, now or datetime.now(UTC)), *params, source, source, limit),
        )
        return [str(row["id"]) for row in rows]

    def detail_queue_size(self, source: str) -> int:
        return count(
            self._conn, f"SELECT COUNT(*) FROM ({_QUEUE})", _queued(source, datetime.now(UTC))
        )

    def feed_quarantined(
        self, ids: Collection[str], source: str = "", identities: Collection[str] = ()
    ) -> list[QuarantinedJob]:
        identity_set = set(identities)
        if not identity_set:
            rows = self._conn.execute(
                f"SELECT * FROM quarantine WHERE id IN {EACH}", (listed(ids),)
            )
            return [quarantined_from(row) for row in rows]
        rows = self._conn.execute(
            f"SELECT * FROM quarantine WHERE id IN {EACH} OR "
            "(source = ? AND rejection_evidence <> '')",
            (listed(ids), source),
        )
        wanted = set(ids)
        return [
            entry
            for entry in map(quarantined_from, rows)
            if entry.id in wanted or posting_identity(entry.apply_url_raw) in identity_set
        ]

    def feed_marker_snapshot(
        self, ids: Collection[str]
    ) -> tuple[tuple[str, str, str, int, int, int], ...]:
        return feed_marker_snapshot(self._conn, ids)

    def feed_pending_ids(self, ids: Collection[str]) -> set[str]:
        rows = self._conn.execute(
            "SELECT q.id FROM quarantine q JOIN detail_fetches p ON p.id = ? || q.id "
            f"WHERE q.id IN {EACH}",
            (FEED_PENDING_PREFIX, listed(ids)),
        )
        return {str(row["id"]) for row in rows}

    def feed_jobs(self, ids: Collection[str]) -> list[Job]:
        rows = self._conn.execute(f"SELECT * FROM jobs WHERE id IN {EACH}", (listed(ids),))
        return [job_from(row) for row in rows]

    def verified_feed_jobs(self, ids: Collection[str]) -> list[Job]:
        rows = self._conn.execute(
            "SELECT j.* FROM jobs j JOIN detail_fetches d ON d.id = ? || j.id "
            "JOIN detail_fetches f ON f.id = ? || j.id "
            f"WHERE j.id IN {EACH} AND j.description <> '' "
            "AND d.resolved = 1 AND d.failed = 0 AND f.resolved = 1 AND f.failed = 0",
            (FEED_DECISION_PREFIX, FEED_DETAIL_PREFIX, listed(ids)),
        )
        return [job_from(row) for row in rows]

    def stored_details(self, ids: Collection[str]) -> dict[str, tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT j.id, CASE WHEN j.source = 'oracle_cloud' AND NOT EXISTS "
            "(SELECT 1 FROM oracle_full_details f WHERE f.id = j.id) THEN '' "
            "ELSE j.description END AS description, j.location_raw FROM jobs j "
            f"WHERE j.id IN {EACH} "
            "AND (j.description <> '' OR NOT j.location_raw GLOB ?)",
            (listed(ids), LOCATION_COUNT_GLOB),
        )
        return {str(row["id"]): (str(row["description"]), str(row["location_raw"])) for row in rows}

    def load_workday_facets(self) -> Mapping[tuple[str, str], WorkdayFacet]:
        rows = self._conn.execute(
            "SELECT tenant, site, parameter, facet_id, descriptor, resolved_at FROM workday_facets"
        )
        return {
            (str(row["tenant"]), str(row["site"])): WorkdayFacet(
                tenant=str(row["tenant"]),
                site=str(row["site"]),
                parameter=str(row["parameter"]),
                facet_ids=tuple(str(row["facet_id"]).split()),
                descriptor=str(row["descriptor"]),
                resolved_at=from_text(row["resolved_at"]),
            )
            for row in rows
        }

    def load_workday_crawls(self) -> Mapping[str, WorkdayCrawl]:
        rows = self._conn.execute(
            "SELECT board, next_offset, total, facet_parameter, facet_ids FROM workday_crawls"
        )
        return {
            str(row["board"]): WorkdayCrawl(
                board=str(row["board"]),
                next_offset=int(row["next_offset"]),
                total=int(row["total"]) if row["total"] is not None else None,
                facet_parameter=str(row["facet_parameter"]),
                facet_ids=tuple(json.loads(str(row["facet_ids"]))),
            )
            for row in rows
        }

    def liveness_queue(
        self,
        since: datetime,
        checked_before: datetime,
        feeds: Collection[str],
        boards: Collection[str],
        url_marks: Sequence[str],
    ) -> list[Job]:
        marked = " OR ".join("j.apply_url_raw LIKE '%' || ? || '%'" for _ in url_marks) or "0"
        rows = self._conn.execute(
            "SELECT j.* FROM jobs j LEFT JOIN detail_fetches d ON d.id = ? || j.id "
            "LEFT JOIN source_visits v ON v.source = j.source "
            "AND v.board = stage_board_of(j.id, j.source) "
            "WHERE j.status = ? AND j.duplicate_of IS NULL AND j.first_seen >= ? "
            "AND (d.fetched_at IS NULL OR d.fetched_at < ?) "
            f"AND ((j.source IN {EACH} AND ({marked})) "
            f"OR (j.source IN {EACH} AND j.last_seen < v.last_success_at)) "
            "ORDER BY d.fetched_at IS NOT NULL, d.fetched_at, j.last_seen, j.id",
            (
                CHECK_PREFIX,
                OPEN,
                to_text(since),
                to_text(checked_before),
                listed(feeds),
                *url_marks,
                listed(boards),
            ),
        )
        return [job_from(row) for row in rows]

    def record_checks(
        self,
        checks: Sequence[LivenessCheck],
        pushback: Sequence[RateState] = (),
        *,
        wait_ms: int | None = None,
    ) -> tuple[str, ...]:
        previous = int(self._conn.execute("PRAGMA busy_timeout").fetchone()[0])
        if wait_ms is not None:
            self._conn.execute(f"PRAGMA busy_timeout = {int(wait_ms)}")
        closed: list[str] = []
        try:
            with self._transaction() as conn:
                for check in checks:
                    keys = [check.id]
                    if check.verdict is Liveness.GONE:
                        shut = [
                            str(row["id"])
                            for row in conn.execute(
                                "SELECT id FROM jobs WHERE (id = ? OR (duplicate_of = ? "
                                f"AND source NOT IN {EACH} AND NOT stage_other_opening("
                                f"(SELECT {link_of()} FROM jobs WHERE id = ?), {link_of()}))) "
                                "AND status = ? ORDER BY id",
                                (
                                    check.id,
                                    check.id,
                                    listed(sorted(BOARD_SOURCES)),
                                    check.id,
                                    OPEN,
                                ),
                            )
                        ]
                        conn.execute(
                            f"UPDATE jobs SET status = ? WHERE id IN {EACH}", (CLOSED, listed(shut))
                        )
                        closed.extend(shut)
                        keys = sorted({check.id, *shut})
                    conn.executemany(
                        UPSERT_CHECK,
                        [
                            (
                                CHECK_PREFIX + key,
                                to_text(check.checked_at),
                                int(check.verdict is Liveness.OPEN),
                                int(check.verdict is Liveness.UNKNOWN),
                                key,
                            )
                            for key in keys
                        ],
                    )
                if closed:
                    stamp = to_text(max(check.checked_at for check in checks))
                    remember_heirs(conn, sorted(promote_open_copies(conn, closed).items()), stamp)
                conn.executemany(UPSERT_PUSHBACK, [rate_state_params(state) for state in pushback])
        finally:
            self._conn.execute(f"PRAGMA busy_timeout = {previous}")
        return tuple(closed)
