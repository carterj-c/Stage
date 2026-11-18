import re
from dataclasses import dataclass
from urllib.parse import urlsplit

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


def target_for_url(url: object) -> HydrationTarget | None:
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
                f"https://api.ashbyhq.com/posting-api/job-board/{account}",
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


def posting_identity(url: object) -> str | None:
    target = target_for_url(url)
    if target is None:
        return None
    posting = (
        target.posting_id.lower() if target.provider in {"lever", "ashby"} else target.posting_id
    )
    return f"{target.provider}:{target.account}:{posting}"
