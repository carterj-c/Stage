import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any, ClassVar

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from stage.domain import Company, DetailFetch, Job, Platform, board_key, job_id
from stage.http import HttpClient, HttpError, RequestRefusedError
from stage.sources import register
from stage.sources._text import collapse_whitespace
from stage.sources.base import (
    FetchResult,
    NullableStr,
    malformed_note,
    payload_error,
    sentences,
    validate_rows,
)
from stage.sources.platforms import oracle_target

PATH = "/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
DETAIL_PATH = "/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
DETAIL_BUDGET = 20
PAGE_SIZE = 100
MAX_PAGES = 16
KEYWORD = "internship"
_LABEL = re.compile(r"\b[\w][\w &/'’\-\s]{0,80}[:：]")
_LABEL_WORDS = frozenset(
    {
        "about",
        "and",
        "candidate",
        "core",
        "criteria",
        "description",
        "duties",
        "eligibility",
        "essential",
        "experience",
        "job",
        "key",
        "minimum",
        "overview",
        "preferred",
        "qualification",
        "qualifications",
        "required",
        "requirement",
        "requirements",
        "responsibilities",
        "responsibility",
        "role",
        "skill",
        "skills",
        "the",
        "your",
    }
)


class OraclePosting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Id: str
    Title: str
    PostedDate: date | None = None
    PrimaryLocation: NullableStr = ""
    PrimaryLocationCountry: NullableStr = ""
    ShortDescriptionStr: NullableStr = ""
    ExternalQualificationsStr: NullableStr = ""
    ExternalResponsibilitiesStr: NullableStr = ""


class OracleDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Id: str
    ExternalDescriptionStr: str
    ExternalQualificationsStr: NullableStr = ""
    ExternalResponsibilitiesStr: NullableStr = ""
    CorporateDescriptionStr: NullableStr = ""
    OrganizationDescriptionStr: NullableStr = ""
    ShortDescriptionStr: NullableStr = ""


def _text(raw: str, *, omit_headings: bool = False) -> str:
    # Parse markup before decoding entities so encoded comparisons remain text.
    soup = BeautifulSoup(raw, "html.parser")
    omitted = ["script", "style"]
    if omit_headings:
        omitted.extend(("h1", "h2", "h3", "h4", "h5", "h6"))
    for element in soup(omitted):
        element.decompose()
    return collapse_whitespace(soup.get_text(" ", strip=True))


def _detail_body(
    payload: Any,
    identifier: str,
    summary: str,
    listing_qualifications: str = "",
    listing_responsibilities: str = "",
) -> str:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != 1:
        return ""
    row = OracleDetail.model_validate(items[0])
    description = _text(row.ExternalDescriptionStr)
    if row.Id != identifier or not description:
        return ""
    core_fields = (
        row.ExternalDescriptionStr,
        row.ExternalResponsibilitiesStr,
        row.ExternalQualificationsStr,
    )
    core = [text for raw in core_fields if (text := _text(raw))]
    listing = [
        text.casefold()
        for raw in (summary, listing_qualifications, listing_responsibilities)
        if (text := _text(raw))
    ]
    detail_summary = _text(row.ShortDescriptionStr).casefold()
    known = [text for text in (" ".join(listing), detail_summary) if text]

    def adds_content(section: str) -> bool:
        section = section.casefold()
        if any(section in text for text in known):
            return False
        # A detail paragraph can assemble the same listing sections in a different order.
        removed_known = False
        for text in (*listing, detail_summary):
            if text and text in section:
                section = section.replace(text, " ")
                removed_known = True
        # A label alone does not establish that the job's actual body is present.
        section = _LABEL.sub(" ", section)
        words = re.findall(r"[^\W_]+", section)
        if not words or all(word in _LABEL_WORDS for word in words):
            return False
        # A lone word left after an echo could itself be an unlabelled heading.
        return not (removed_known and len(words) == 1 and words[0].isalpha())

    if known and not any(
        adds_content(text) for raw in core_fields if (text := _text(raw, omit_headings=True))
    ):
        return ""
    return "\n\n".join(
        [
            *core,
            *(
                _text(raw)
                for raw in (row.CorporateDescriptionStr, row.OrganizationDescriptionStr)
                if raw
            ),
        ]
    )


class OraclePage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    TotalJobsCount: int = Field(ge=0)
    requisitionList: list[Any] = Field(default_factory=list)


@register
class OracleCloudAdapter:
    name: ClassVar[str] = "oracle_cloud"
    platform: ClassVar[Platform] = Platform.ORACLE_CLOUD
    rate_profile: ClassVar[str] = "paginated"
    hosts: ClassVar[frozenset[str]] = frozenset()
    bucket_key: ClassVar[str] = "oracle_cloud"
    detail_budget: ClassVar[int] = DETAIL_BUDGET
    rotation_slice: ClassVar[int] = 0
    max_requests_per_company: ClassVar[int] = MAX_PAGES + DETAIL_BUDGET

    def host_for(self, company: Company) -> str:
        host, _ = oracle_target(company.oracle_host or "", company.oracle_site or "")
        return host

    def hosts_for(self, companies: Sequence[Company]) -> frozenset[str]:
        return frozenset(self.host_for(company) for company in companies)

    def board_key(self, company: Company) -> str:
        return board_key(self.name, self._board(company))

    def url_for(self, company: Company) -> str:
        return f"https://{self.host_for(company)}{PATH}"

    def params_for(self, company: Company, offset: int) -> dict[str, str]:
        _, site = oracle_target(company.oracle_host or "", company.oracle_site or "")
        keyword = KEYWORD if company.oracle_keyword is None else company.oracle_keyword.strip()
        narrowed = f",keyword={keyword}" if keyword else ""
        return {
            "onlyData": "true",
            "expand": "requisitionList",
            "finder": (f"findReqs;siteNumber={site}{narrowed},offset={offset},limit={PAGE_SIZE}"),
        }

    def plan(self, company: Company) -> tuple[str, ...]:
        return (self.url_for(company),)

    async def fetch(
        self,
        company: Company,
        client: HttpClient,
        now: datetime,
        facets: object = None,
        details: Sequence[str] = (),
    ) -> FetchResult:
        wanted = set(
            dict.fromkeys(
                ident for ident in details if ident.startswith(f"{self.board_key(company)}:")
            )
        )
        postings: list[OraclePosting] = []
        malformed = 0
        truncated = False
        premature_end = False

        for page in range(MAX_PAGES):
            response = await client.get_json(
                self.url_for(company),
                params=self.params_for(company, page * PAGE_SIZE),
                revalidate=bool(wanted),
            )
            if response.not_modified:
                if page == 0:
                    return FetchResult(not_modified=True)
                return await self._attach_descriptions(
                    company,
                    client,
                    postings,
                    wanted,
                    FetchResult(
                        jobs=tuple(self._to_job(company, row, now) for row in postings),
                        authoritative=False,
                        degraded="A later page answered 304, so the walk ended early.",
                    ),
                )
            rows, dropped, total = self._validate(company, response.payload)
            malformed += dropped
            if not rows and not dropped:
                premature_end = len(postings) + malformed < total
                break
            postings.extend(rows)
            if len(postings) + malformed >= total:
                break
        else:
            truncated = True

        return await self._attach_descriptions(
            company,
            client,
            postings,
            wanted,
            FetchResult(
                jobs=tuple(self._to_job(company, row, now) for row in postings),
                authoritative=not (truncated or premature_end or malformed),
                degraded=sentences(
                    f"Stage stopped at the {MAX_PAGES}-page cap" if truncated else "",
                    "The board sent an empty page before its stated total" if premature_end else "",
                    malformed_note(malformed),
                ),
            ),
        )

    async def _attach_descriptions(
        self,
        company: Company,
        client: HttpClient,
        postings: Sequence[OraclePosting],
        wanted: set[str],
        result: FetchResult,
    ) -> FetchResult:
        host, site = oracle_target(company.oracle_host or "", company.oracle_site or "")
        bodies: dict[str, str] = {}
        outcomes: list[DetailFetch] = []
        for row, job in zip(postings, result.jobs, strict=True):
            if job.id not in wanted or job.id in bodies:
                continue
            if len(outcomes) >= self.detail_budget:
                break
            body = ""
            if re.fullmatch(r"[0-9]+", row.Id):
                try:
                    response = await client.get_json(
                        f"https://{host}{DETAIL_PATH}",
                        params={
                            "onlyData": "true",
                            "finder": f"ById;Id={row.Id},siteNumber={site}",
                        },
                        revalidate=True,
                    )
                    if not response.not_modified:
                        body = _detail_body(
                            response.payload,
                            row.Id,
                            row.ShortDescriptionStr,
                            row.ExternalQualificationsStr,
                            row.ExternalResponsibilitiesStr,
                        )
                except RequestRefusedError:
                    # No request was admitted: do not consume a persisted retry attempt.
                    break
                except (HttpError, httpx.HTTPError, ValidationError):
                    pass
            bodies[job.id] = body
            outcomes.append(DetailFetch(id=job.id, resolved=bool(body), failed=not bool(body)))
        return replace(
            result,
            jobs=tuple(
                replace(job, description=bodies[job.id]) if bodies.get(job.id) else job
                for job in result.jobs
            ),
            detail_fetches=tuple(outcomes),
        )

    def _validate(self, company: Company, payload: Any) -> tuple[list[OraclePosting], int, int]:
        try:
            parents = payload["items"] if isinstance(payload, dict) else None
            if not isinstance(parents, list) or len(parents) != 1:
                raise ValueError("items must contain exactly one search result")
            page = OraclePage.model_validate(parents[0])
        except (KeyError, TypeError, ValidationError, ValueError) as exc:
            raise payload_error(
                self.name,
                company.slug,
                payload,
                f"{self.name}/{company.slug} sent an invalid recruiting search response",
            ) from exc
        rows, dropped = validate_rows(
            OraclePosting, page.requisitionList, source=self.name, slug=company.slug
        )
        return rows, dropped, page.TotalJobsCount

    def _to_job(self, company: Company, row: OraclePosting, now: datetime) -> Job:
        host, site = oracle_target(company.oracle_host or "", company.oracle_site or "")
        title = collapse_whitespace(row.Title)
        # Search fields are summaries. Empty bodies let sync carry stored full text
        # before screening; only validated ById responses may supply a new body.
        description = ""
        location = collapse_whitespace(
            ", ".join(part for part in (row.PrimaryLocation, row.PrimaryLocationCountry) if part)
        )
        posted = (
            datetime.combine(row.PostedDate, datetime.min.time(), tzinfo=UTC)
            if row.PostedDate is not None
            else None
        )
        return Job(
            id=job_id(self.name, self._board(company), row.Id),
            source=self.name,
            company=company.name,
            title_raw=title,
            title_normalized=title.lower(),
            apply_url_raw=f"https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{row.Id}",
            description=description,
            location_raw=location,
            first_seen=now,
            last_seen=now,
            source_posted_at=posted,
        )

    def _board(self, company: Company) -> str:
        return f"{company.oracle_host or company.slug}-{company.oracle_site or ''}"
