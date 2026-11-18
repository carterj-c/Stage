from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from bs4 import BeautifulSoup

from stage.domain import Company, DetailFetch, Platform, job_id
from stage.http import HttpClient, RatePosture
from stage.services.sync import _with_stored_details
from stage.sources.base import FetchResult
from stage.sources.jobvite import HOST, MAX_SHOW_MORE, JobviteAdapter
from stage.storage import SourceBatch, open_repository

FIXTURES = Path(__file__).parent / "fixtures"
BOARD = (FIXTURES / "jobvite_ziffdavis_board.html").read_text(encoding="utf-8")
DETAIL = (FIXTURES / "jobvite_ziffdavis_detail.html").read_text(encoding="utf-8")
COMPANY = Company(name="Ziff Davis", platform=Platform.JOBVITE, slug="ziffdavis")
NOW = datetime(2026, 11, 18, tzinfo=UTC)
POSTING = "oib4AfwR"
IDENT = job_id("jobvite", COMPANY.slug, POSTING)
URL = f"https://{HOST}/{COMPANY.slug}/job/{POSTING}"
LIST_URL = f"https://{HOST}/{COMPANY.slug}/jobs"


def _client() -> HttpClient:
    return HttpClient(
        allowed_hosts=JobviteAdapter.hosts,
        posture=RatePosture(min_interval_s=0, max_requests_per_run=100),
        jitter=False,
    )


def _board(*urls: str) -> str:
    rows = "".join(
        f'<tr><td class="jv-job-list-name"><a href="{url}">Intern</a></td></tr>' for url in urls
    )
    return f'<table class="jv-job-list"><tbody>{rows}</tbody></table>'


async def _fetch(
    details: tuple[str, ...] = (IDENT,), board: str = BOARD, adapter: JobviteAdapter | None = None
) -> FetchResult:
    respx.get(LIST_URL).respond(200, text=board)
    async with _client() as client:
        return await (adapter or JobviteAdapter()).fetch(COMPANY, client, NOW, details=details)


@respx.mock
async def test_saved_full_posting_keeps_all_duties_requirements_and_tail() -> None:
    detail = respx.get(URL).respond(200, text=DETAIL)
    result = await _fetch()
    job = next(job for job in result.jobs if job.id == IDENT)
    assert detail.call_count == 1
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)
    assert "Key Responsibilities" in job.description
    assert "Month-End Close Support" in job.description
    assert "3+ years of accounts payable" in job.description
    assert "Associate or Bachelor’s degree" in job.description
    assert "enabling our thousands of employees worldwide to thrive" in job.description
    page = BeautifulSoup(DETAIL, "html.parser")
    body = page.select_one("div.jv-job-detail-description")
    assert body is not None
    for meta in body.select("h3, .jv-meta"):
        meta.decompose()
    assert "".join(job.description.split()) == "".join(body.get_text().split())
    assert len(job.description) > 3800
    assert all(not job.description for job in result.jobs if job.id != IDENT)


@respx.mock
async def test_detail_preserves_encoded_comparisons() -> None:
    detail = DETAIL.replace("Key Responsibilities", "GPA &gt; 3.0 &amp; experience &lt; 5 years", 1)
    respx.get(URL).respond(200, text=detail)
    result = await _fetch()
    job = next(job for job in result.jobs if job.id == IDENT)
    assert "GPA > 3.0 & experience < 5 years" in job.description


@respx.mock
@pytest.mark.parametrize("status", [403, 404, 410, 304])
async def test_detail_http_failures_leave_listing_authoritative_and_retryable(status: int) -> None:
    respx.get(URL).respond(status)
    result = await _fetch()
    assert result.authoritative
    assert not result.degraded
    assert not any(job.description for job in result.jobs)
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)


@respx.mock
async def test_transport_failure_is_a_detail_failure() -> None:
    respx.get(URL).mock(side_effect=httpx.ReadError("connection lost"))
    result = await _fetch()
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)


@respx.mock
@pytest.mark.parametrize(
    "body",
    [
        "<h1>Jobvite Page Unavailable</h1>",
        '<div class="jv-job-detail-description">Service unavailable</div>',
        DETAIL.split("Qualifications", 1)[0],
        DETAIL.split('<div class="jv-job-detail-bottom-actions">', 1)[0],
        DETAIL.replace(f"/{COMPANY.slug}/job/{POSTING}/apply", "/other/job/wrong/apply"),
        DETAIL.replace("jv-job-detail-description", "unknown-description"),
        '<div class="jv-job-detail-description"><h3>Description</h3>'
        '<div class="jv-meta">Position at Ziff Davis</div></div>'
        f'<div class="jv-job-detail-bottom-actions"><a class="jv-button-apply" '
        f'href="{URL}/apply">Apply</a></div>',
        f'<div class="jv-job-detail-description">Truncated body'
        f'<div class="jv-job-detail-bottom-actions"><a class="jv-button-apply" '
        f'href="{URL}/apply">Apply</a></div>',
    ],
    ids=[
        "outage",
        "no-actions",
        "cut-requirements",
        "cut-actions",
        "wrong-posting",
        "missing-body",
        "empty-wrapper",
        "unclosed-body",
    ],
)
async def test_error_empty_and_partial_html_never_resolve(body: str) -> None:
    respx.get(URL).respond(200, text=body)
    result = await _fetch()
    assert result.authoritative
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)
    assert not any(job.description for job in result.jobs)


@respx.mock
async def test_queue_is_board_scoped_and_bypasses_only_its_own_validators() -> None:
    listing = respx.get(LIST_URL).respond(200, text=BOARD)
    detail = respx.get(URL).respond(200, text=DETAIL)
    async with _client() as client:
        for url in (LIST_URL, URL):
            client.cache.record(url, httpx.Headers({"etag": '"old"'}), NOW)
        adapter = JobviteAdapter()
        await adapter.fetch(COMPANY, client, NOW, details=[IDENT])
        assert "if-none-match" not in listing.calls[-1].request.headers
        assert "if-none-match" not in detail.calls[-1].request.headers
        listing.respond(304)
        result = await adapter.fetch(
            COMPANY, client, NOW, details=[job_id("jobvite", "other", POSTING)]
        )
        assert listing.calls[-1].request.headers["if-none-match"] == '"old"'
        assert result.not_modified
        assert not result.detail_fetches
        assert detail.call_count == 1


@respx.mock
async def test_no_queue_means_no_speculative_details() -> None:
    result = await _fetch(details=())
    assert result.jobs
    assert not result.detail_fetches
    assert len(respx.calls) == 1


@respx.mock
async def test_stale_queue_ids_do_not_generate_requests_or_outcomes() -> None:
    result = await _fetch(details=(job_id("jobvite", COMPANY.slug, "gone"),))
    assert not result.detail_fetches
    assert len(respx.calls) == 1


@respx.mock
async def test_detail_budget_caps_requests_even_with_duplicates_and_foreign_boards() -> None:
    adapter = JobviteAdapter()
    budget = adapter.detail_budget
    assert 0 < budget < 150
    assert adapter.max_requests_per_company == 1 + MAX_SHOW_MORE + budget
    urls = [f"https://{HOST}/{COMPANY.slug}/job/p{n}" for n in range(budget + 3)]
    ids = [job_id("jobvite", COMPANY.slug, f"p{n}") for n in range(budget + 3)]
    route = respx.get(url__startswith=f"https://{HOST}/{COMPANY.slug}/job/").respond(404)
    result = await _fetch(
        details=(job_id("jobvite", "other", "p0"), *ids, *ids), board=_board(*urls)
    )
    assert route.call_count == budget
    assert len(result.jobs) == budget + 3
    assert {
        outcome.id for outcome in result.detail_fetches if isinstance(outcome, DetailFetch)
    } == set(ids[:budget])


@respx.mock
@pytest.mark.parametrize(
    "url",
    [
        f"https://evil.example/{COMPANY.slug}/job/{POSTING}",
        f"https://{HOST}/other/job/{POSTING}",
        f"http://{HOST}/{COMPANY.slug}/job/{POSTING}",
        f"https://{HOST}:443/{COMPANY.slug}/job/{POSTING}",
        f"https://user@{HOST}/{COMPANY.slug}/job/{POSTING}",
    ],
)
async def test_listing_links_cannot_escape_the_detail_host_or_board(url: str) -> None:
    result = await _fetch(board=_board(url))
    assert result.detail_fetches == (DetailFetch(id=IDENT, failed=True),)
    assert len(respx.calls) == 1


@respx.mock
async def test_tracking_urls_are_normalized_and_duplicate_ids_fetch_once() -> None:
    detail = respx.get(URL).respond(200, text=DETAIL)
    result = await _fetch(board=_board(URL + "?src=rss", URL + "?src=other"))
    assert detail.call_count == 1
    assert all(job.description for job in result.jobs)
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)


@respx.mock
async def test_partial_listing_stays_partial_after_successful_hydration() -> None:
    respx.get(URL).respond(200, text=DETAIL)
    board = _board(URL).replace(
        "</tbody>",
        '<tr><td><a href="/ziffdavis/search?c=IT&p=0">Show More</a></td></tr></tbody>',
    )
    respx.get(f"https://{HOST}/ziffdavis/search?c=IT&p=0").respond(404)
    result = await _fetch(board=board)
    assert not result.authoritative
    assert "Show More" in result.degraded
    assert result.jobs[0].description
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)


@respx.mock
async def test_queue_failure_then_success_and_later_listing_preserves_body(db_path: Path) -> None:
    from stage.domain import UNKNOWN_TERM, RoleCategory

    adapter = JobviteAdapter()
    first = await _fetch(details=())
    job = next(job for job in first.jobs if job.id == IDENT)
    job = replace(job, term=UNKNOWN_TERM, role=RoleCategory.UNKNOWN)
    detail = respx.get(URL).respond(200, text="<h1>Temporarily unavailable</h1>")
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(
                source="jobvite",
                run_started_at=NOW,
                jobs=(job,),
            )
        )
        queued = await repository.detail_queue("jobvite", adapter.detail_budget)
        assert queued == [IDENT]
        failed = await _fetch(details=tuple(queued))
        await repository.apply_source_batch(
            SourceBatch(
                source="jobvite",
                run_started_at=NOW,
                detail_fetches=tuple(
                    o for o in failed.detail_fetches if isinstance(o, DetailFetch)
                ),
            )
        )
        assert await repository.detail_queue("jobvite", adapter.detail_budget) == [IDENT]
        detail.respond(200, text=DETAIL)
        hydrated = await _fetch(details=tuple(queued))
        complete = next(job for job in hydrated.jobs if job.id == IDENT)
        await repository.apply_source_batch(
            SourceBatch(
                source="jobvite",
                run_started_at=NOW,
                jobs=(complete,),
                detail_fetches=tuple(
                    o for o in hydrated.detail_fetches if isinstance(o, DetailFetch)
                ),
            )
        )
        assert await repository.detail_queue("jobvite", adapter.detail_budget) == []
        later = await _fetch(details=())
        carried = await _with_stored_details(repository, later.jobs)
        assert next(job for job in carried if job.id == IDENT).description == complete.description
        assert detail.call_count == 2


@respx.mock
async def test_section_headings_and_nested_content_are_preserved() -> None:
    body = (
        '<div class="jv-job-detail-description"><h3>Description</h3>'
        '<div class="jv-meta">Position at Ziff Davis</div>'
        "<h3>Responsibilities</h3><div><p>Build &amp; ship.</p></div>"
        "<h3>Requirements</h3><ul><li>Master’s degree.</li><li>2+ years.</li></ul>"
        "<script>tracking()</script><style>.tracking {color:red}</style></div>"
        f'<div class="jv-job-detail-bottom-actions"><a class="jv-button-apply" '
        f'href="{URL}/apply">Apply</a></div>'
    )
    respx.get(URL).respond(200, text=body)
    result = await _fetch(board=_board(URL))
    assert result.jobs[0].description == (
        "Responsibilities Build & ship. Requirements Master’s degree. 2+ years."
    )
    assert result.detail_fetches == (DetailFetch(id=IDENT, resolved=True),)
