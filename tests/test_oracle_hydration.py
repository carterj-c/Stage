from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from stage.domain import Company, DetailFetch, Platform
from stage.http import HttpClient, RequestRefusedError, profile
from stage.services.screening import Screener, normalize_batch
from stage.services.sync import _persist_source_batch
from stage.sources.oracle_cloud import DETAIL_PATH, MAX_PAGES, OracleCloudAdapter, _detail_body
from stage.storage import SourceBatch, open_repository

NOW = datetime(2026, 11, 18, tzinfo=UTC)
COMPANY = Company(
    name="Macy's",
    platform=Platform.ORACLE_CLOUD,
    slug="macys",
    oracle_host="ebwh.fa.us2.oraclecloud.com",
    oracle_site="CX_1001",
)
ADAPTER = OracleCloudAdapter()
DETAIL_URL = f"https://{COMPANY.oracle_host}{DETAIL_PATH}"
BODY = "<p>Develop software. Must be a US citizen.</p><p>GPA &gt; 3, x&lt;y and y&gt;z.</p>"


def _posting(identifier: str = "91731") -> dict[str, object]:
    return {
        "Id": identifier,
        "Title": "Software Engineer Intern Summer 2027",
        "PrimaryLocation": "New York, NY",
        "ShortDescriptionStr": "Build systems",
    }


def _listing(*rows: dict[str, object], total: int | None = None) -> dict[str, object]:
    return {
        "items": [
            {"TotalJobsCount": len(rows) if total is None else total, "requisitionList": list(rows)}
        ]
    }


def _detail(identifier: str = "91731", body: str = BODY) -> dict[str, object]:
    return {
        "items": [
            {
                "Id": identifier,
                "ExternalDescriptionStr": body,
                "ExternalResponsibilitiesStr": "Responsibilities: build services",
                "ExternalQualificationsStr": "Qualifications: Python",
                "CorporateDescriptionStr": "Corporate information",
                "OrganizationDescriptionStr": "Organization information",
            }
        ]
    }


def _id(identifier: str = "91731") -> str:
    return ADAPTER._to_job(
        COMPANY, ADAPTER._validate(COMPANY, _listing(_posting(identifier)))[0][0], NOW
    ).id


def _client(**kwargs: Any) -> HttpClient:
    return HttpClient(
        allowed_hosts=ADAPTER.hosts_for((COMPANY,)),
        posture=replace(profile(ADAPTER.rate_profile), min_interval_s=0),
        jitter=False,
        **kwargs,
    )


@respx.mock
async def test_summary_full_detail_and_next_sync_preserve_and_rescreen(db_path: Path) -> None:
    listing = respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client, open_repository(db_path) as repo:
        summary = await ADAPTER.fetch(COMPANY, client, NOW)
        assert summary.jobs[0].description == ""
        await _persist_source_batch(
            repo, Screener(), SourceBatch(source=ADAPTER.name, run_started_at=NOW), summary.jobs
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [_id()]
        full = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert full.authoritative and full.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        assert detail.calls[0].request.url.params["finder"] == "ById;Id=91731,siteNumber=CX_1001"
        body = full.jobs[0].description
        assert "x<y and y>z" in body
        for section in ("Responsibilities", "Qualifications", "Corporate", "Organization"):
            assert section in body
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(hours=1),
                detail_fetches=full.detail_fetches,
            ),
            full.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == []
        later = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=2))
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(source=ADAPTER.name, run_started_at=NOW + timedelta(hours=2)),
            later.jobs,
        )
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == body
        assert stored.work_auth_flag is True
        # Even a direct stale empty-body write cannot erase the body.
        await repo.apply_source_batch(
            SourceBatch(source=ADAPTER.name, run_started_at=NOW, jobs=later.jobs)
        )
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == body
    assert listing.call_count == 3 and detail.call_count == 1


@respx.mock
async def test_classified_legacy_short_summary_is_eligible_and_failures_retry(
    db_path: Path,
) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(404)
    async with _client() as client, open_repository(db_path) as repo:
        raw = (await ADAPTER.fetch(COMPANY, client, NOW)).jobs[0]
        jobs, rejected = normalize_batch(
            [replace(raw, description="Build systems. No sponsorship needed.")]
        )
        assert not rejected and jobs[0].work_auth_flag is False and jobs[0].term != "unknown"
        await repo.apply_source_batch(
            SourceBatch(source=ADAPTER.name, run_started_at=NOW, jobs=jobs)
        )
        for attempt in range(3):
            assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [
                _id()
            ]
            result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
            assert result.authoritative and result.detail_fetches == (
                DetailFetch(id=_id(), failed=True),
            )
            await repo.apply_source_batch(
                SourceBatch(
                    source=ADAPTER.name,
                    run_started_at=NOW + timedelta(hours=attempt),
                    detail_fetches=result.detail_fetches,
                )
            )
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == []
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == ""
        assert stored.work_auth_flag is None
    assert detail.call_count == 3


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"items": []},
        {"items": [{}, {}]},
        {"items": [{}]},
        {"items": [{"Id": "91731", "ExternalDescriptionStr": 42}]},
        _detail("other"),
        _detail(body=" "),
        {"items": [{"Id": "91731", "ExternalDescriptionStr": "<b>Build</b> systems"}]},
        {
            "items": [
                {
                    "Id": "91731",
                    "ExternalDescriptionStr": "build &amp; test",
                    "ShortDescriptionStr": "<p>build &amp; test</p>",
                }
            ]
        },
    ],
)
@respx.mock
async def test_invalid_or_summary_only_detail_never_resolves(payload: object) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    respx.get(DETAIL_URL).respond(200, json=payload)
    async with _client() as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert result.authoritative
    assert result.jobs[0].description == ""
    assert result.detail_fetches == (DetailFetch(id=_id(), failed=True),)


@respx.mock
async def test_budget_deduplicates_and_ignores_foreign_or_absent_ids() -> None:
    rows = [_posting(str(n)) for n in range(ADAPTER.detail_budget + 2)]
    rows.append(rows[0])
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(*rows))
    detail = respx.get(DETAIL_URL).mock(
        side_effect=lambda req: httpx.Response(
            200, json=_detail(req.url.params["finder"].split("Id=")[1].split(",")[0])
        )
    )
    ids = [_id(str(n)) for n in range(ADAPTER.detail_budget + 2)]
    async with _client() as client:
        result = await ADAPTER.fetch(
            COMPANY, client, NOW, details=["oracle_cloud:foreign:1", _id("999"), *ids, *ids]
        )
    assert detail.call_count == ADAPTER.detail_budget
    assert len(result.detail_fetches) == ADAPTER.detail_budget
    assert result.jobs[0].description == result.jobs[-1].description
    assert ADAPTER.max_requests_per_company == MAX_PAGES + ADAPTER.detail_budget


@respx.mock
async def test_refused_request_does_not_spend_attempt_or_lose_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    async with _client() as client:
        original = client.get_json

        async def refused(url: str, **kwargs: Any) -> Any:
            if url == DETAIL_URL:
                raise RequestRefusedError("request cap")
            return await original(url, **kwargs)

        monkeypatch.setattr(client, "get_json", refused)
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert result.authoritative and len(result.jobs) == 1
    assert not result.detail_fetches


@respx.mock
async def test_hydration_does_not_make_partial_listing_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stage.sources.oracle_cloud.MAX_PAGES", 1)
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting(), total=2))
    respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
    assert not result.authoritative and result.jobs[0].description
    assert result.detail_fetches == (DetailFetch(id=_id(), resolved=True),)


def test_public_sections_are_complete_without_arbitrary_length_threshold() -> None:
    assert _detail_body(_detail(body="A short complete posting."), "91731", "summary")
    body = _detail_body(
        _detail(body="<p>first</p>" + "<p>required qualification</p>" * 2000 + "<p>last</p>"),
        "91731",
        "summary",
    )
    assert body.count("required qualification") == 2000 and "last" in body


@respx.mock
async def test_real_http_cap_defers_detail_without_spending_attempt(db_path: Path) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    posture = replace(profile(ADAPTER.rate_profile), min_interval_s=0, max_requests_per_run=1)
    async with HttpClient(
        allowed_hosts=ADAPTER.hosts_for((COMPANY,)), posture=posture, jitter=False
    ) as client:
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert client.request_count == 1
    assert result.authoritative and result.detail_fetches == () and not detail.called
    async with open_repository(db_path) as repo:
        await repo.apply_source_batch(
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW,
                jobs=result.jobs,
                detail_fetches=result.detail_fetches,
            )
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [_id()]


@respx.mock
async def test_queued_details_bypass_listing_and_detail_validators() -> None:
    listing = respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client:
        listing_url = str(
            httpx.URL(ADAPTER.url_for(COMPANY), params=ADAPTER.params_for(COMPANY, 0))
        )
        detail_url = str(
            httpx.URL(
                DETAIL_URL,
                params={"onlyData": "true", "finder": "ById;Id=91731,siteNumber=CX_1001"},
            )
        )
        for url in (listing_url, detail_url):
            client.cache.record(url, httpx.Headers({"etag": '"old"'}), NOW)
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert result.jobs[0].description
        assert "if-none-match" not in listing.calls[-1].request.headers
        assert "if-none-match" not in detail.calls[-1].request.headers


def test_matching_summary_with_extra_detail_sections_is_full() -> None:
    payload = _detail(body="<p>Build systems</p>")
    body = _detail_body(payload, "91731", "Build systems")
    assert "Build systems" in body
    assert "Responsibilities: build services" in body
    assert "Qualifications: Python" in body


@respx.mock
async def test_quarantined_known_role_needs_fresh_detail_to_reopen(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage.classify.scope import Rejection
    from stage.domain import QuarantineFilters, RejectionReason
    from stage.services import screening

    def reject_body(job: Any) -> Rejection | None:
        if "RESTRICTED" in job.description:
            return Rejection(RejectionReason.OUT_OF_SCOPE_DEGREE, "restricted requirement")
        return None

    monkeypatch.setattr(screening, "screen_is_cs_role", reject_body)
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail(body="RESTRICTED requirement"))
    async with _client() as client, open_repository(db_path) as repo:
        full = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert full.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name, run_started_at=NOW, detail_fetches=full.detail_fetches
            ),
            full.jobs,
        )
        assert await repo.get_job(_id()) is None
        (held,) = await repo.list_quarantined(QuarantineFilters(limit=None))
        assert held.reason is RejectionReason.OUT_OF_SCOPE_DEGREE
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == []
        from stage.domain.visits import ORACLE_DETAIL_REFRESH_DAYS

        assert await repo.detail_queue(
            ADAPTER.name, 20, now=NOW + timedelta(days=ORACLE_DETAIL_REFRESH_DAYS)
        ) == [_id()]
        listing = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=1))
        kept, rejected = normalize_batch(listing.jobs)
        assert kept and not rejected  # Known-role exemption used to release this row.
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(source=ADAPTER.name, run_started_at=NOW + timedelta(hours=1)),
            listing.jobs,
        )
        assert await repo.get_job(_id()) is None
        (still_held,) = await repo.list_quarantined(QuarantineFilters(limit=None))
        assert still_held.reason is held.reason
        # Even a stale carried full body cannot reopen it without a fresh outcome.
        stale, _ = normalize_batch([replace(listing.jobs[0], description="previous complete body")])
        await repo.apply_source_batch(
            SourceBatch(source=ADAPTER.name, run_started_at=NOW + timedelta(hours=2), jobs=stale)
        )
        assert await repo.get_job(_id()) is None
        detail.respond(200, json=_detail(body="New complete posting; no restricted requirement."))
        fresh = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=3), details=[_id()])
        assert fresh.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(hours=3),
                detail_fetches=fresh.detail_fetches,
            ),
            fresh.jobs,
        )
        saved = await repo.get_job(_id())
        assert saved is not None and "New complete posting" in saved.description
        assert not await repo.list_quarantined(QuarantineFilters(limit=None))


@respx.mock
async def test_rules_change_requeues_full_detail_and_failure_preserves_body(db_path: Path) -> None:
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    detail = respx.get(DETAIL_URL).respond(200, json=_detail())
    async with _client() as client, open_repository(db_path) as repo:
        full = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert full.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name, run_started_at=NOW, detail_fetches=full.detail_fetches
            ),
            full.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == []
        await repo.mark_screened(0)
        await repo.mark_screened(1, at=NOW + timedelta(hours=1))
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [_id()]
        detail.respond(404)
        failure = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=2), details=[_id()])
        assert failure.detail_fetches == (DetailFetch(id=_id(), failed=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(hours=2),
                detail_fetches=failure.detail_fetches,
            ),
            failure.jobs,
        )
        stored = await repo.get_job(_id())
        assert stored is not None and stored.description == full.jobs[0].description
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [_id()]
        detail.respond(200, json=_detail(body="Updated complete content"))
        refreshed = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(hours=3), details=[_id()])
        assert refreshed.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(hours=3),
                detail_fetches=refreshed.detail_fetches,
            ),
            refreshed.jobs,
        )
        saved = await repo.get_job(_id())
        assert saved is not None and "Updated complete content" in saved.description
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == []


def test_migration_hides_unverified_summaries_and_preserves_proven_bodies(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import closing

    from stage.storage import migrations
    from stage.storage.sqlite_repo import SqliteRepository

    original = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: original[:4])
        with closing(SqliteRepository.connect(db_path)) as repo:
            for ident, source, body in (
                (_id(), ADAPTER.name, "Legacy short summary"),
                (_id("91732"), ADAPTER.name, "Verified full body"),
                (_id("91733"), ADAPTER.name, "Unknown body after failed attempt"),
                ("other:board:1", "other", "Other source body"),
            ):
                repo._conn.execute(
                    "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                    "apply_url_raw, description, first_seen, last_seen, work_auth_known) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (
                        ident,
                        source,
                        "Company",
                        "Software Intern",
                        "software intern",
                        "https://example.com",
                        body,
                        NOW.isoformat(),
                        NOW.isoformat(),
                    ),
                )
            for ident, resolved, failed in ((_id("91732"), 1, 0), (_id("91733"), 0, 1)):
                repo._conn.execute(
                    "INSERT INTO detail_fetches (id, source, fetched_at, resolved, failed) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ident, ADAPTER.name, NOW.isoformat(), resolved, failed),
                )
    with closing(SqliteRepository.connect(db_path)) as repo:
        rows = {
            row["id"]: row
            for row in repo._conn.execute("SELECT id, description, work_auth_known FROM jobs")
        }
        assert rows[_id()]["description"] == "" and rows[_id()]["work_auth_known"] == 0
        assert rows[_id("91733")]["description"] == ""
        assert rows[_id("91732")]["description"] == "Verified full body"
        assert rows["other:board:1"]["description"] == "Other source body"
        archived = dict(
            repo._conn.execute("SELECT id, description FROM oracle_unverified_descriptions")
        )
        assert archived == {
            _id(): "Legacy short summary",
            _id("91733"): "Unknown body after failed attempt",
        }
        assert repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(hours=4)) == [
            _id(),
            _id("91733"),
        ]
        assert not repo._conn.execute(
            "SELECT rowid FROM jobs_fts WHERE jobs_fts MATCH ?", ("legacy",)
        ).fetchall()
        # A stale writer cannot cause the carry-forward path to trust an unverified body.
        repo._conn.execute("UPDATE jobs SET description = ? WHERE id = ?", ("Stale summary", _id()))
        assert repo.stored_details([_id()])[_id()][0] == ""
        assert repo._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrations.migrate(repo._conn, db_path) == ()


@pytest.mark.parametrize("field", ["CorporateDescriptionStr", "OrganizationDescriptionStr"])
def test_boilerplate_alone_does_not_turn_summary_into_complete_body(field: str) -> None:
    payload = {
        "items": [
            {
                "Id": "91731",
                "ExternalDescriptionStr": "Build systems",
                field: "We are an equal opportunity employer.",
            }
        ]
    }
    assert _detail_body(payload, "91731", "<p>Build systems</p>") == ""
    payload["items"][0]["ExternalQualificationsStr"] = "Must know Python"
    body = _detail_body(payload, "91731", "Build systems")
    assert "Must know Python" in body and "equal opportunity" in body


@respx.mock
async def test_periodic_refresh_is_bounded_rescreens_and_retries_after_exhaustion(
    db_path: Path,
) -> None:
    from stage.domain.visits import ORACLE_DETAIL_REFRESH_DAYS

    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting(), _posting("91732")))
    detail = respx.get(DETAIL_URL).mock(
        side_effect=lambda req: httpx.Response(
            200,
            json=_detail(
                req.url.params["finder"].split("Id=")[1].split(",")[0], "Original full description"
            ),
        )
    )
    expiry = NOW + timedelta(days=ORACLE_DETAIL_REFRESH_DAYS)
    async with _client() as client, open_repository(db_path) as repo:
        first = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id(), _id("91732")])
        outcomes = tuple(x for x in first.detail_fetches if isinstance(x, DetailFetch))
        assert len(outcomes) == 2
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(source=ADAPTER.name, run_started_at=NOW, detail_fetches=outcomes),
            first.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=expiry - timedelta(seconds=1)) == []
        assert len(await repo.detail_queue(ADAPTER.name, 1, now=expiry)) == 1
        assert set(await repo.detail_queue(ADAPTER.name, 20, now=expiry)) == {_id(), _id("91732")}
        # Only the chosen queued ID consumes a detail request, and the updated body is screened.
        detail.mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=_detail(
                    req.url.params["finder"].split("Id=")[1].split(",")[0],
                    "Updated job. Must be a US citizen.",
                ),
            )
        )
        updated = await ADAPTER.fetch(COMPANY, client, expiry, details=[_id()])
        assert updated.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name, run_started_at=expiry, detail_fetches=updated.detail_fetches
            ),
            updated.jobs,
        )
        saved = await repo.get_job(_id())
        assert (
            saved is not None
            and saved.work_auth_flag is True
            and "Updated job" in saved.description
        )
        assert await repo.detail_queue(ADAPTER.name, 20, now=expiry) == [_id("91732")]
        # Three failures stop immediate retries while retaining the old verified body.
        detail.respond(404)
        for attempt in range(3):
            at = expiry + timedelta(hours=attempt + 1)
            failed = await ADAPTER.fetch(COMPANY, client, at, details=[_id("91732")])
            assert failed.detail_fetches == (DetailFetch(id=_id("91732"), failed=True),)
            await _persist_source_batch(
                repo,
                Screener(),
                SourceBatch(
                    source=ADAPTER.name, run_started_at=at, detail_fetches=failed.detail_fetches
                ),
                failed.jobs,
            )
        end = expiry + timedelta(hours=3)
        assert await repo.detail_queue(ADAPTER.name, 20, now=end) == []
        retained = await repo.get_job(_id("91732"))
        assert retained is not None and "Original full description" in retained.description
        later = end + timedelta(days=ORACLE_DETAIL_REFRESH_DAYS)
        assert _id("91732") in await repo.detail_queue(ADAPTER.name, 20, now=later)
        # Another failed periodic attempt starts a fresh bounded retry cycle.
        failed = await ADAPTER.fetch(COMPANY, client, later, details=[_id("91732")])
        assert failed.detail_fetches == (DetailFetch(id=_id("91732"), failed=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name, run_started_at=later, detail_fetches=failed.detail_fetches
            ),
            failed.jobs,
        )
        assert _id("91732") in await repo.detail_queue(ADAPTER.name, 20, now=later)


@pytest.mark.parametrize("baseline", [4, 5])
@respx.mock
async def test_migration_rearms_exhausted_unverified_reread_without_exposing_text(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: int
) -> None:
    from contextlib import closing

    from stage.storage import migrations
    from stage.storage.sqlite_repo import SqliteRepository

    shipped = migrations.discover()
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "discover", lambda: shipped[:4])
        with closing(SqliteRepository.connect(db_path)) as repo:
            repo._conn.execute(
                "INSERT INTO jobs (id, source, company, title_raw, title_normalized, "
                "apply_url_raw, description, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _id(),
                    ADAPTER.name,
                    "Company",
                    "Software Intern",
                    "software intern",
                    "https://example.com",
                    "Possibly full former body",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
            repo._conn.execute(
                "INSERT INTO detail_fetches (id, source, fetched_at, resolved, failed, attempts) "
                "VALUES (?, ?, ?, 0, 1, 3)",
                (_id(), ADAPTER.name, NOW.isoformat()),
            )
        if baseline == 5:
            patch.setattr(migrations, "discover", lambda: shipped[:5])
            with closing(SqliteRepository.connect(db_path)) as repo:
                prior = repo.get_job(_id())
                assert prior is not None and prior.description == ""
                assert repo.detail_queue(ADAPTER.name, 20, now=NOW) == []
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(_posting()))
    respx.get(DETAIL_URL).respond(200, json=_detail(body="Recovered verified body"))
    async with _client() as client, open_repository(db_path) as repo:
        saved = await repo.get_job(_id())
        assert saved is not None and saved.description == ""
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW) == [_id()]
        result = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert result.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(minutes=1),
                detail_fetches=result.detail_fetches,
            ),
            result.jobs,
        )
        restored = await repo.get_job(_id())
        assert restored is not None and "Recovered verified body" in restored.description
        assert await repo.detail_queue(ADAPTER.name, 20, now=NOW + timedelta(minutes=2)) == []
    with closing(SqliteRepository.connect(db_path)) as repo:
        assert (
            repo._conn.execute(
                "SELECT description FROM oracle_unverified_descriptions WHERE id = ?", (_id(),)
            ).fetchone()[0]
            == "Possibly full former body"
        )
        assert repo._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@respx.mock
async def test_repeated_listing_sections_do_not_count_as_new_detail(db_path: Path) -> None:
    listing = _posting()
    listing["ShortDescriptionStr"] = "Build <b>systems</b>"
    listing["ExternalQualificationsStr"] = "Python &amp; SQL"
    listing["ExternalResponsibilitiesStr"] = "Develop services"
    respx.get(ADAPTER.url_for(COMPANY)).respond(200, json=_listing(listing))
    detail = respx.get(DETAIL_URL).respond(
        200,
        json={
            "items": [
                {
                    "Id": "91731",
                    "ExternalDescriptionStr": "<p>Build systems</p>",
                    "ExternalQualificationsStr": "Python &amp; SQL",
                    "ExternalResponsibilitiesStr": "Develop services",
                    "CorporateDescriptionStr": "We are an equal opportunity employer.",
                }
            ]
        },
    )
    async with _client() as client, open_repository(db_path) as repo:
        summary = await ADAPTER.fetch(COMPANY, client, NOW)
        await _persist_source_batch(
            repo, Screener(), SourceBatch(source=ADAPTER.name, run_started_at=NOW), summary.jobs
        )
        rejected = await ADAPTER.fetch(COMPANY, client, NOW, details=[_id()])
        assert rejected.authoritative
        assert rejected.jobs[0].description == ""
        assert rejected.detail_fetches == (DetailFetch(id=_id(), failed=True),)
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(minutes=1),
                detail_fetches=rejected.detail_fetches,
            ),
            rejected.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 1, now=NOW + timedelta(minutes=1)) == [_id()]
        saved = await repo.get_job(_id())
        assert saved is not None and saved.description == ""
        detail.respond(
            200,
            json={
                "items": [
                    {
                        "Id": "91731",
                        "ExternalDescriptionStr": "<p>Build systems</p>",
                        "ExternalQualificationsStr": (
                            "Python &amp; SQL. Must have 2 years of experience."
                        ),
                        "ExternalResponsibilitiesStr": "Develop services",
                        "CorporateDescriptionStr": "We are an equal opportunity employer.",
                    }
                ]
            },
        )
        full = await ADAPTER.fetch(COMPANY, client, NOW + timedelta(minutes=2), details=[_id()])
        assert full.detail_fetches == (DetailFetch(id=_id(), resolved=True),)
        assert "Must have 2 years of experience" in full.jobs[0].description
        await _persist_source_batch(
            repo,
            Screener(),
            SourceBatch(
                source=ADAPTER.name,
                run_started_at=NOW + timedelta(minutes=2),
                detail_fetches=full.detail_fetches,
            ),
            full.jobs,
        )
        assert await repo.detail_queue(ADAPTER.name, 1, now=NOW + timedelta(minutes=2)) == []
        saved = await repo.get_job(_id())
        assert saved is not None and "Must have 2 years of experience" in saved.description


def test_reordered_assembled_listing_sections_are_still_only_summary() -> None:
    payload = {
        "items": [
            {
                "Id": "91731",
                "ExternalDescriptionStr": "Develop services. Python &amp; SQL. Build systems.",
            }
        ]
    }
    assert (
        _detail_body(payload, "91731", "Build systems", "Python &amp; SQL", "Develop services")
        == ""
    )
