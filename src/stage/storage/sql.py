import json
import sqlite3
from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache

from stage.domain.visits import (
    FEED_DECISION_PREFIX,
    FEED_DETAIL_PREFIX,
    FEED_PENDING_PREFIX,
    FEED_REJECTION_PREFIX,
)

EACH = "(SELECT value FROM json_each(?))"


def verified_feed_hold(alias: str) -> str:
    return (
        f"(EXISTS (SELECT 1 FROM detail_fetches d WHERE d.id = ? || {alias}.id "
        "AND d.resolved = 1 AND d.failed = 0) OR "
        f"({alias}.matched_phrase NOT GLOB 'copy of *' AND NOT EXISTS "
        f"(SELECT 1 FROM detail_fetches v WHERE v.id = ? || {alias}.id) AND EXISTS "
        f"(SELECT 1 FROM detail_fetches d WHERE d.id = ? || {alias}.id "
        "AND d.resolved = 1 AND d.failed = 0)))"
    )


def link_of(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"coalesce(nullif({prefix}apply_url_canonical, ''), {prefix}apply_url_raw)"


def listed[T: (str, int)](values: Iterable[T]) -> str:
    return json.dumps(sorted(set(values)))


def feed_marker_snapshot(
    conn: sqlite3.Connection, ids: Collection[str]
) -> tuple[tuple[str, str, str, int, int, int], ...]:
    keys = (
        prefix + ident
        for ident in ids
        for prefix in (
            FEED_DETAIL_PREFIX,
            FEED_DECISION_PREFIX,
            FEED_REJECTION_PREFIX,
            FEED_PENDING_PREFIX,
        )
    )
    rows = conn.execute(
        f"SELECT id, source, fetched_at, resolved, attempts, failed "
        f"FROM detail_fetches WHERE id IN {EACH} ORDER BY id",
        (listed(keys),),
    )
    return tuple(
        (str(row[0]), str(row[1]), str(row[2]), int(row[3]), int(row[4]), int(row[5]))
        for row in rows
    )


def marks(count: int) -> str:
    return ", ".join("?" * count)


def starting(prefix: str) -> str:
    return f"{prefix}*"


def board_rows(board: str) -> str:
    return starting(f"{board}:")


@lru_cache(maxsize=4096)
def to_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Stage stores only times with a time zone.")
    return value.astimezone(UTC).isoformat()


def from_text(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def required_time(value: str | None, column: str) -> datetime:
    parsed = from_text(value)
    if parsed is None:
        raise ValueError(f"The database has a row with no {column}. Run stage doctor to check it.")
    return parsed


def limit_clause(limit: int | None, offset: int = 0) -> tuple[str, tuple[int, ...]]:
    if not offset:
        return ("", ()) if limit is None else (" LIMIT ?", (limit,))
    return " LIMIT ? OFFSET ?", (-1 if limit is None else limit, offset)


def fold_company(value: str) -> str:
    from stage.domain.text import plain_company
    from stage.lexicon import fold

    return fold(plain_company(value))


def count(conn: sqlite3.Connection, sql: str, params: Iterable[object] = ()) -> int:
    return int(conn.execute(sql, tuple(params)).fetchone()[0])


class Store:
    _conn: sqlite3.Connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
