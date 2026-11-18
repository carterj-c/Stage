"""Single-posting discovery-link hydration; no crawling or sync integration.

Pass a caller-owned HttpClient configured with ``target.hosts`` (or HOSTS) and
shared budgets. HTTP/budget failures propagate so callers can schedule retries.
``complete`` means the endpoint's body contract was verified, not that the
employer supplied every requirement. Never promote unverified text to full.
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from pydantic import ValidationError

from stage.domain import Company, Platform
from stage.http import HttpClient
from stage.sources._text import strip_html
from stage.sources.lever import LeverAdapter, LeverPosting

HOSTS = frozenset(
    {
        "boards-api.greenhouse.io",
        "api.lever.co",
        "jobs.ashbyhq.com",
        "apply.workable.com",
    }
)
_SLUG = r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}"
_UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"


@dataclass(frozen=True, slots=True)
class HydrationTarget:
    provider: str
    account: str
    posting_id: str
    url: str

    @property
    def hosts(self) -> frozenset[str]:
        return frozenset({urlsplit(self.url).hostname or ""})


@dataclass(frozen=True, slots=True)
class HydrationDetail:
    description: str
    complete: bool
    limitation: str = ""


def target_for_url(url: object) -> HydrationTarget | None:
    """Resolve only exact supported public job paths to fixed HTTPS targets.

    Tracking query strings/fragments are discarded. Percent-encoded path tokens,
    credentials, explicit ports (including 443), controls and backslashes fail
    closed. Application paths are accepted only for Lever and Ashby.
    """
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        return None
    if "\\" in url:
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != parsed.hostname:
            return None
    except ValueError:
        return None
    host, path = parsed.hostname, parsed.path
    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"}:
        match = re.fullmatch(rf"/({_SLUG})/jobs/([1-9][0-9]{{0,19}})/?", path)
        if match:
            account, posting_id = match.groups()
            return HydrationTarget(
                "greenhouse",
                account,
                posting_id,
                f"https://boards-api.greenhouse.io/v1/boards/{account}/jobs/{posting_id}",
            )
    elif host == "jobs.lever.co":
        match = re.fullmatch(rf"/({_SLUG})/({_UUID})(?:/apply)?/?", path)
        if match:
            account, posting_id = match.groups()
            return HydrationTarget(
                "lever",
                account,
                posting_id,
                f"https://api.lever.co/v0/postings/{account}/{posting_id}?mode=json",
            )
    elif host == "jobs.ashbyhq.com":
        match = re.fullmatch(rf"/({_SLUG})/({_UUID})(?:/application)?/?", path)
        if match:
            account, posting_id = match.groups()
            return HydrationTarget(
                "ashby",
                account,
                posting_id,
                f"https://jobs.ashbyhq.com/{account}/{posting_id}",
            )
    elif host == "apply.workable.com":
        match = re.fullmatch(rf"/({_SLUG})/j/([A-Z0-9]{{10}})/?", path)
        if match:
            account, posting_id = match.groups()
            return HydrationTarget(
                "workable",
                account,
                posting_id,
                f"https://apply.workable.com/api/v2/accounts/{account}/jobs/{posting_id}",
            )
    return None


def _canonical(target: HydrationTarget) -> bool:
    """Do not trust manually constructed targets as request destinations."""
    paths = {
        "greenhouse": f"https://boards.greenhouse.io/{target.account}/jobs/{target.posting_id}",
        "lever": f"https://jobs.lever.co/{target.account}/{target.posting_id}",
        "ashby": f"https://jobs.ashbyhq.com/{target.account}/{target.posting_id}",
        "workable": f"https://apply.workable.com/{target.account}/j/{target.posting_id}",
    }
    return target_for_url(paths.get(target.provider, "")) == target


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _body(value: object) -> str:
    return strip_html(value) if isinstance(value, str) else ""


def _json_nodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [node for entry in value for node in _json_nodes(entry)]
    if isinstance(value, dict):
        return [value, *_json_nodes(value.get("@graph"))]
    return []


def _ashby_detail(target: HydrationTarget, page: str) -> HydrationDetail | None:
    matches: list[str] = []
    soup = BeautifulSoup(page, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            nodes = _json_nodes(json.loads(script.get_text()))
        except (ValueError, RecursionError):
            continue
        for node in nodes:
            kind = node.get("@type")
            if kind != "JobPosting" and not (isinstance(kind, list) and "JobPosting" in kind):
                continue
            linked = node.get("url")
            if not isinstance(linked, str) or target_for_url(linked) != target:
                continue
            if not _has_text(node.get("title")):
                continue
            body = _body(node.get("description"))
            if body:
                matches.append(body)
    # Ambiguous structured data is not safe evidence of the requested body.
    if len(matches) != 1:
        return None
    return HydrationDetail(
        matches[0],
        False,
        "Ashby JSON-LD may be abridged; full page-body equivalence is unverified",
    )


def _json_detail(target: HydrationTarget, data: object) -> HydrationDetail | None:
    if not isinstance(data, dict) or any(key in data for key in ("error", "errors")):
        return None
    if target.provider == "greenhouse":
        identity = data.get("id")
        if type(identity) is not int or str(identity) != target.posting_id:
            return None
        if not _has_text(data.get("title")):
            return None
        if data.get("absolute_url") and target_for_url(data["absolute_url"]) != target:
            return None
        body = _body(data.get("content"))
        return HydrationDetail(body, True) if body else None
    if target.provider == "workable":
        if data.get("shortcode") != target.posting_id or data.get("state") != "published":
            return None
        if not _has_text(data.get("title")) or not _body(data.get("description")):
            return None
        fields = ("description", "requirements", "benefits")
        if not all(isinstance(data.get(field), str) for field in fields):
            return None
        body = "\n\n".join(text for field in fields if (text := _body(data[field])))
        return HydrationDetail(body, True) if body else None
    if target.provider != "lever" or data.get("id") != target.posting_id:
        return None
    if not _has_text(data.get("text")):
        return None
    # Validate raw lists before Pydantic can silently default a malformed section.
    if "lists" in data and (
        not isinstance(data["lists"], list)
        or any(
            not isinstance(section, dict)
            or not all(isinstance(section.get(key), str) for key in ("text", "content"))
            or not _body(section.get("content"))
            for section in data["lists"]
        )
    ):
        return None
    # A different linked job is an error even when the top-level id matches.
    for key in ("hostedUrl", "applyUrl"):
        if data.get(key) and target_for_url(data[key]) != target:
            return None
    try:
        row = LeverPosting.model_validate(data)
        company = Company(name=target.account, platform=Platform.LEVER, slug=target.account)
        body = LeverAdapter().to_job(company, row, datetime.now(UTC)).description.strip()
    except (ValidationError, ValueError, OverflowError, OSError):
        return None
    if not body:
        return None
    lever_fields = (
        "descriptionPlain",
        "descriptionBodyPlain",
        "salaryDescriptionPlain",
        "additionalPlain",
    )
    complete = "lists" in data and all(isinstance(data.get(field), str) for field in lever_fields)
    return HydrationDetail(
        body,
        complete,
        "" if complete else "Lever section fields absent; completeness unverified",
    )


async def fetch_target(
    target: HydrationTarget,
    client: HttpClient,
) -> HydrationDetail | None:
    """Fetch one target using caller-owned security, pacing and shared budgets.

    Invalid/error bodies return None; transport and HTTP failures propagate.
    The caller must inspect ``complete`` before storing text as full detail.
    """
    if not _canonical(target):
        return None
    if target.provider == "ashby":
        response = await client.get_text(target.url, revalidate=True)
        if response.status != 200 or response.not_modified:
            return None
        return _ashby_detail(target, response.text)
    response_json = await client.get_json(target.url, revalidate=True)
    if response_json.status != 200 or response_json.not_modified:
        return None
    return _json_detail(target, response_json.payload)
