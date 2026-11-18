import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from stage.domain import Company, DetailFetch, Job, Platform, job_id
from stage.http import HttpClient, HttpError, RequestRefusedError
from stage.sources import register
from stage.sources._text import collapse_whitespace, strip_html
from stage.sources.base import BoardAdapter, FetchResult, NullableBool, NullableStr

HOST_TEMPLATE = "{slug}.bamboohr.com"
PATH = "/careers/list"


class BambooLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: NullableStr = ""
    state: NullableStr = ""
    country: NullableStr = ""

    def label(self) -> str:
        return ", ".join(part for part in (self.city, self.state, self.country) if part)


class BambooPosting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    jobOpeningName: str
    departmentLabel: NullableStr = ""
    employmentStatusLabel: NullableStr = ""
    location: BambooLocation | None = None
    atsLocation: BambooLocation | None = None
    isRemote: NullableBool = False

    @field_validator("id", mode="before")
    @classmethod
    def valid_id(cls, value: Any) -> Any:
        if (type(value) is not int and not isinstance(value, str)) or not re.fullmatch(
            r"[1-9][0-9]*", str(value)
        ):
            raise ValueError("posting id must be a positive decimal integer")
        return value

    def where(self) -> str:
        for candidate in (self.location, self.atsLocation):
            if candidate is not None:
                label = candidate.label()
                if label:
                    return label
        return "Remote" if self.isRemote else ""


class BambooBoard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: list[Any]


@register
class BambooHrAdapter(BoardAdapter):
    name: ClassVar[str] = "bamboohr"
    platform: ClassVar[Platform] = Platform.BAMBOOHR
    rate_profile: ClassVar[str] = "moderate"
    bucket_key: ClassVar[str] = "bamboohr"
    detail_budget: ClassVar[int] = 20
    max_requests_per_company: ClassVar[int] = 1 + detail_budget

    host_template: ClassVar[str] = HOST_TEMPLATE
    path: ClassVar[str] = PATH
    root_model: ClassVar[type[BaseModel] | None] = BambooBoard
    rows_field: ClassVar[str] = "result"
    row_model: ClassVar[type[BaseModel]] = BambooPosting

    async def fetch(
        self,
        company: Company,
        client: HttpClient,
        now: datetime,
        facets: object = None,
        details: Sequence[str] = (),
    ) -> FetchResult:
        prefix = f"{self.board_key(company)}:"
        wanted = set(
            list(dict.fromkeys(ident for ident in details if ident.startswith(prefix)))[
                : self.detail_budget
            ]
        )
        response = await client.get_json(self.url_for(company), revalidate=bool(wanted))
        if response.not_modified:
            return FetchResult(not_modified=True)
        result = await self.built(company, response, now)
        bodies: dict[str, str] = {}
        outcomes: list[DetailFetch] = []
        for job in result.jobs:
            if job.id not in wanted or job.id in bodies:
                continue
            posting = job.id.removeprefix(prefix)
            if not re.fullmatch(r"[1-9][0-9]*", posting):
                continue
            url = f"https://{self.host_for(company)}/careers/{posting}"
            try:
                detail = await client.get_json(f"{url}/detail", revalidate=True)
                body = "" if detail.not_modified else _description(detail.payload, job, url)
            except RequestRefusedError:
                continue
            except (HttpError, httpx.HTTPError):
                body = ""
            bodies[job.id] = body
            outcomes.append(DetailFetch(id=job.id, resolved=bool(body), failed=not bool(body)))
        return replace(
            result,
            jobs=tuple(replace(job, description=bodies.get(job.id, "")) for job in result.jobs),
            detail_fetches=tuple(outcomes),
        )

    def to_job(self, company: Company, row: Any, now: datetime) -> Job:
        title = collapse_whitespace(row.jobOpeningName)
        return Job(
            id=job_id(self.name, company.slug, str(row.id)),
            source=self.name,
            company=company.name,
            title_raw=title,
            title_normalized=title.lower(),
            apply_url_raw=f"https://{self.host_for(company)}/careers/{row.id}",
            description="",
            location_raw=collapse_whitespace(row.where()),
            first_seen=now,
            last_seen=now,
        )


def _description(payload: Any, job: Job, url: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        return ""
    opening = payload["result"].get("jobOpening")
    if not isinstance(opening, dict):
        return ""
    title = opening.get("jobOpeningName")
    body = opening.get("description")
    if not isinstance(title, str) or collapse_whitespace(title) != job.title_raw:
        return ""
    if opening.get("jobOpeningShareUrl") != url or not isinstance(body, str):
        return ""
    return strip_html(body)
