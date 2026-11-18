"""Public-shaped single-posting fixtures; no network requests to providers."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from stage.http import HostBudget, HostBudgetExceededError, HttpClient, RatePosture
from stage.services.hydration_targets import (
    HOSTS,
    HydrationTarget,
    fetch_target,
    target_for_url,
)

UUID = "337314ee-4de5-4b96-a419-1b9287a7df68"
URLS = {
    "greenhouse": "https://job-boards.greenhouse.io/Acme/jobs/12345",
    "lever": f"https://jobs.lever.co/Acme/{UUID}",
    "ashby": f"https://jobs.ashbyhq.com/Acme/{UUID}",
    "workable": "https://apply.workable.com/acme/j/ABC1234567/",
}


def target(provider: str) -> HydrationTarget:
    result = target_for_url(URLS[provider])
    assert result is not None
    return result


def posting(provider: str) -> dict[str, object]:
    if provider == "greenhouse":
        return {
            "id": 12345,
            "title": "Engineer",
            "content": (
                "&lt;h2&gt;Responsibilities&lt;/h2&gt;&lt;p&gt;Build systems&lt;/p&gt;"
                "&lt;h2&gt;Requirements&lt;/h2&gt;&lt;p&gt;Master's degree&lt;/p&gt;"
            ),
        }
    if provider == "workable":
        return {
            "shortcode": "ABC1234567",
            "title": "Engineer",
            "state": "published",
            "description": "<p>Build systems</p>",
            "requirements": "<ul><li>Master's degree</li></ul>",
            "benefits": "<p>Health insurance</p>",
        }
    return {
        "id": UUID,
        "text": "Engineer",
        "hostedUrl": URLS["lever"],
        "applyUrl": URLS["lever"] + "/apply",
        "descriptionPlain": "Introduction",
        "descriptionBodyPlain": "Build systems",
        "lists": [{"text": "Requirements", "content": "<ul><li>Master's degree</li></ul>"}],
        "salaryDescriptionPlain": "Salary band",
        "additionalPlain": "Health insurance",
    }


def client_for(response: httpx.Response) -> HttpClient:
    return HttpClient(
        allowed_hosts=HOSTS,
        jitter=False,
        posture=RatePosture(min_interval_s=0, max_requests_per_run=10),
        transport=httpx.MockTransport(lambda _request: response),
    )


@pytest.mark.parametrize(
    ("url", "provider", "endpoint"),
    [
        (
            URLS["greenhouse"],
            "greenhouse",
            "https://boards-api.greenhouse.io/v1/boards/Acme/jobs/12345",
        ),
        (
            "https://boards.greenhouse.io/Acme/jobs/12345/?gh_src=abc#app",
            "greenhouse",
            "https://boards-api.greenhouse.io/v1/boards/Acme/jobs/12345",
        ),
        (
            URLS["lever"] + "/apply?lever-source=feed",
            "lever",
            f"https://api.lever.co/v0/postings/Acme/{UUID}?mode=json",
        ),
        (URLS["ashby"] + "/application", "ashby", URLS["ashby"]),
        (
            URLS["workable"],
            "workable",
            "https://apply.workable.com/api/v2/accounts/acme/jobs/ABC1234567",
        ),
    ],
)
def test_exact_target(url: str, provider: str, endpoint: str) -> None:
    resolved = target_for_url(url)
    assert resolved is not None
    assert resolved.provider == provider
    assert resolved.url == endpoint
    assert resolved.hosts <= HOSTS


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://boards.greenhouse.io/acme/jobs/123",
        "//jobs.lever.co/acme/" + UUID,
        "https://user:password@boards.greenhouse.io/acme/jobs/123",
        "https://boards.greenhouse.io:443/acme/jobs/123",
        "https://boards.greenhouse.io:/acme/jobs/123",
        "https://boards.greenhouse.io.evil.test/acme/jobs/123",
        "https://boards.greenhouse.io./acme/jobs/123",
        "https://boards.greenhouse.io@evil.test/acme/jobs/123",
        "https://evil.test/?url=" + URLS["lever"],
        "https://boards.greenhouse.io/acme/../jobs/123",
        "https://boards.greenhouse.io/%2e%2e/jobs/123",
        "https://boards.greenhouse.io/acme%2fother/jobs/123",
        "https://boards.greenhouse.io/acme/jobs/0",
        "https://boards.greenhouse.io/acme/jobs/00123",
        "https://boards.greenhouse.io/acme/jobs/-123",
        "https://boards.greenhouse.io/acme/jobs/123/extra",
        "https://boards.greenhouse.io/acme/jobs/123\n",
        "https://boards.greenhouse.io\\evil.test/acme/jobs/123",
        "https://boards.greenhouse.io/acme/jobs",
        "https://jobs.lever.co/acme/not-a-uuid",
        URLS["lever"] + "/extra",
        URLS["lever"] + "%2fapply",
        "https://jobs.eu.lever.co/acme/" + UUID,
        "https://jobs.ashbyhq.com/acme/not-a-uuid",
        "https://apply.workable.com/acme/j/abc1234567/",
        "https://apply.workable.com/acme/j/ABC1234567/apply/",
        "https://apply.workable.com/acme/j/ABC/",
        "https://[invalid/acme/jobs/1",
    ],
)
def test_unsafe_or_unsupported_url_has_no_target(url: str) -> None:
    assert target_for_url(url) is None


@pytest.mark.parametrize("provider", ["greenhouse", "lever", "workable"])
async def test_full_multisection_detail(provider: str) -> None:
    async with client_for(httpx.Response(200, json=posting(provider))) as client:
        result = await fetch_target(target(provider), client)
        assert client.request_count == 1
    assert result is not None and result.complete
    assert "Build systems" in result.description
    assert "Master's degree" in result.description
    if provider == "lever":
        assert "Introduction" in result.description
        assert "### Requirements" in result.description
        assert "Salary band" in result.description
    if provider in {"lever", "workable"}:
        assert "Health insurance" in result.description


@pytest.mark.parametrize(
    ("provider", "patch"),
    [
        ("greenhouse", {"id": 54321}),
        ("greenhouse", {"id": True}),
        ("greenhouse", {"absolute_url": "https://boards.greenhouse.io/other/jobs/12345"}),
        ("greenhouse", {"content": "<p></p>"}),
        ("greenhouse", {"content": {"text": "x"}}),
        ("greenhouse", {"error": "not found"}),
        ("greenhouse", {"title": ""}),
        ("lever", {"id": "other"}),
        ("lever", {"lists": None}),
        ("lever", {"lists": [{}]}),
        ("lever", {"lists": [{"text": "Requirements"}]}),
        ("lever", {"hostedUrl": URLS["lever"].replace("Acme", "Other")}),
        ("lever", {"applyUrl": "https://evil.test/"}),
        ("lever", {"descriptionPlain": None}),
        ("workable", {"shortcode": "OTHER12345"}),
        ("workable", {"state": "draft"}),
        ("workable", {"state": None}),
        ("workable", {"state": "closed"}),
        ("workable", {"description": ""}),
        ("workable", {"requirements": None}),
        ("workable", {"benefits": []}),
    ],
)
async def test_bad_detail_is_rejected(provider: str, patch: dict[str, object]) -> None:
    data = posting(provider) | patch
    async with client_for(httpx.Response(200, json=data)) as client:
        assert await fetch_target(target(provider), client) is None


@pytest.mark.parametrize("provider", ["greenhouse", "lever", "workable"])
@pytest.mark.parametrize("payload", [[], {}, {"message": "error"}, None])
async def test_wrong_payload_shapes(provider: str, payload: object) -> None:
    async with client_for(httpx.Response(200, content=json.dumps(payload))) as client:
        assert await fetch_target(target(provider), client) is None


@pytest.mark.parametrize(
    "field",
    [
        "lists",
        "descriptionPlain",
        "descriptionBodyPlain",
        "salaryDescriptionPlain",
        "additionalPlain",
    ],
)
async def test_missing_lever_section_never_implies_full(field: str) -> None:
    data = posting("lever")
    del data[field]
    async with client_for(httpx.Response(200, json=data)) as client:
        result = await fetch_target(target("lever"), client)
    assert result is not None and not result.complete
    assert result.limitation


@pytest.mark.parametrize("field", ["state", "description", "requirements", "benefits"])
async def test_workable_missing_public_or_body_fields_fail_closed(field: str) -> None:
    data = posting("workable")
    del data[field]
    async with client_for(httpx.Response(200, json=data)) as client:
        assert await fetch_target(target("workable"), client) is None


async def test_empty_optional_workable_sections_are_explicitly_valid() -> None:
    data = posting("workable") | {"requirements": "", "benefits": ""}
    async with client_for(httpx.Response(200, json=data)) as client:
        result = await fetch_target(target("workable"), client)
    assert result is not None and result.complete


def ashby_page(url: str = URLS["ashby"]) -> str:
    node = {
        "@type": "JobPosting",
        "url": url,
        "title": "Engineer",
        "description": "<p>Build systems</p><h2>Requirements</h2><p>Master's degree</p>",
    }
    return (
        '<meta name="description" content="SUMMARY">'
        '<script type="application/ld+json">'
        + json.dumps({"@graph": [node]})
        + "</script><main>Unrelated body</main>"
    )


async def test_ashby_uses_only_matching_jsonld_and_does_not_claim_full() -> None:
    async with client_for(httpx.Response(200, text=ashby_page())) as client:
        result = await fetch_target(target("ashby"), client)
    assert result is not None and not result.complete
    assert "Master's degree" in result.description
    assert "SUMMARY" not in result.description
    assert "Unrelated" not in result.description


@pytest.mark.parametrize(
    "page",
    [
        '<meta name="description" content="Summary"><main>Body</main>',
        '<script type="application/ld+json">invalid</script>',
        ashby_page(URLS["ashby"].replace(UUID, "11111111-1111-1111-1111-111111111111")),
        ashby_page(URLS["ashby"].replace("Acme", "Other")),
        ashby_page() + ashby_page(),
        ashby_page().replace("JobPosting", "WebPage"),
    ],
)
async def test_ashby_rejects_missing_mismatched_or_ambiguous_body(page: str) -> None:
    async with client_for(httpx.Response(200, text=page)) as client:
        assert await fetch_target(target("ashby"), client) is None


async def test_forged_target_never_reaches_network() -> None:
    client = AsyncMock(spec=HttpClient)
    for forged in [
        replace(target("lever"), url="https://evil.test/"),
        replace(target("lever"), provider="unknown"),
        replace(target("lever"), account="../other"),
    ]:
        assert await fetch_target(forged, client) is None
    client.get_json.assert_not_called()
    client.get_text.assert_not_called()


@pytest.mark.parametrize("provider", list(URLS))
async def test_revalidation_is_explicit(provider: str) -> None:
    client = AsyncMock(spec=HttpClient)
    method = client.get_text if provider == "ashby" else client.get_json
    method.return_value.status = 304
    method.return_value.not_modified = True
    resolved = target(provider)
    assert await fetch_target(resolved, client) is None
    method.assert_awaited_once_with(resolved.url, revalidate=True)


async def test_caller_shared_budget_and_host_policy_are_preserved() -> None:
    posture = RatePosture(min_interval_s=0, max_requests_per_run=1)
    budgets: dict[str, HostBudget] = {}
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=posting("greenhouse"))
    )
    for attempt in range(2):
        async with HttpClient(
            allowed_hosts=target("greenhouse").hosts,
            budgets=budgets,
            posture=posture,
            transport=transport,
            jitter=False,
        ) as client:
            if attempt == 0:
                assert await fetch_target(target("greenhouse"), client) is not None
            else:
                with pytest.raises(HostBudgetExceededError):
                    await fetch_target(target("greenhouse"), client)


async def test_redirect_failure_is_not_followed_or_swallowed() -> None:
    from stage.http import RedirectNotAllowedError

    async with client_for(
        httpx.Response(302, headers={"location": "https://evil.test/job"})
    ) as client:
        with pytest.raises(RedirectNotAllowedError):
            await fetch_target(target("greenhouse"), client)


@pytest.mark.parametrize("value", [None, 123, [], {}])
def test_non_string_url_has_no_target(value: object) -> None:
    assert target_for_url(value) is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"error": "not found"}),
        httpx.Response(200, text="<html>Error page</html>"),
    ],
)
async def test_http_and_unreadable_errors_propagate(response: httpx.Response) -> None:
    from stage.http import HttpError

    async with client_for(response) as client:
        with pytest.raises(HttpError):
            await fetch_target(target("greenhouse"), client)


async def test_client_host_allowlist_cannot_be_bypassed() -> None:
    from stage.http import HostNotAllowedError

    async with HttpClient(allowed_hosts=frozenset()) as client:
        with pytest.raises(HostNotAllowedError):
            await fetch_target(target("greenhouse"), client)
        assert client.request_count == 0
