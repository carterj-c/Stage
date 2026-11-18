import json
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from mcp import Client
from mcp.types import TextContent
from typer.testing import CliRunner

from stage.ai.server import build_server
from stage.cli.app import app
from stage.domain import Job, JobFilters, JobLevel, RoleCategory
from stage.domain.industry import Industry
from stage.domain.preferences import Preferences
from stage.http import HttpClient, profile
from stage.services.preferences import load_preferences, revise_preferences, save_preferences
from stage.services.screening import normalize_batch
from stage.sources.simplify import SimplifyFeed
from stage.storage.industry import employer_industries, validate_catalogue
from stage.storage.repository import SourceBatch
from stage.storage.sql import fold_company
from stage.storage.sqlite_repo import SqliteRepository
from stage.tui.state import FilterState, filter_rows

NOW = datetime.now(UTC)


def job(name: str, company: str, source: str = "greenhouse") -> Job:
    return Job(
        id=f"{source}:employer:{name}",
        source=source,
        description="",
        company=company,
        title_raw="Software Engineer Intern",
        title_normalized="software engineer intern",
        apply_url_raw=f"https://example.com/{name}",
        first_seen=NOW,
        last_seen=NOW,
        role=RoleCategory.SWE,
        level=JobLevel.INTERNSHIP,
    )


@pytest.fixture
def stored(tmp_path: Path) -> Path:
    path = tmp_path / "jobs.db"
    with closing(SqliteRepository.connect(path)) as repo:
        repo.apply_source_batch(
            SourceBatch(
                source="greenhouse",
                run_started_at=NOW,
                jobs=(
                    job("riot", "Riot Games"),
                    job("stripe", "Stripe"),
                    job("ssense", "SSENSE"),
                    job("bank", "Royal Bank of Canada"),
                    job("ambiguous", "Riot"),
                    replace(job("title", "Unlisted"), title_raw="Fintech Gaming Fashion Intern"),
                ),
            )
        )
        repo.apply_source_batch(
            SourceBatch(
                source="simplify", run_started_at=NOW, jobs=(job("feed", "RIOT GAMES", "simplify"),)
            )
        )
    return path


def test_catalogue_normalizes_and_rejects_collisions() -> None:
    catalogue = employer_industries()
    for company, industry in (
        ("Riot Games", Industry.GAMING),
        ("Ubisoft", Industry.GAMING),
        ("Electronic Arts", Industry.GAMING),
        ("SSENSE", Industry.FASHION),
        ("Shopify", Industry.FINTECH),
        ("Stripe", Industry.FINTECH),
        ("Wealthsimple", Industry.FINTECH),
        ("Behaviour Interactive", Industry.GAMING),
        ("Rockstar Games", Industry.GAMING),
        ("Square Enix", Industry.GAMING),
        ("Nike", Industry.FASHION),
        ("Levi Strauss & Co.", Industry.FASHION),
        ("Adyen", Industry.FINTECH),
        ("Microsoft", Industry.TECH),
        ("Royal Bank of Canada", Industry.FINANCE),
    ):
        assert industry in catalogue[fold_company(company)]
    assert "riot" not in catalogue
    assert "ea" not in catalogue
    assert "square" not in catalogue
    assert catalogue[fold_company("Block")] == {Industry.FINTECH, Industry.TECH, Industry.FINANCE}
    assert catalogue[fold_company("Stripe")] == {Industry.FINTECH, Industry.TECH, Industry.FINANCE}
    assert catalogue[fold_company("Royal Bank of Canada")] == {Industry.FINANCE}
    both = validate_catalogue([{"aliases": ["Dual"], "industries": ["gaming", "fintech"]}])
    assert both["dual"] == {Industry.GAMING, Industry.FINTECH}
    with pytest.raises(ValueError, match="Colliding"):
        validate_catalogue([{"aliases": ["Acme", "ACME"], "industries": ["gaming"]}])
    with pytest.raises(ValueError):
        validate_catalogue([{"aliases": ["Acme"], "industries": ["unknown"]}])


def test_stored_rows_or_unknown_intersection_limits_and_counts(stored: Path) -> None:
    with closing(SqliteRepository.connect(stored)) as repo:
        base = JobFilters(industries=frozenset({Industry.GAMING}), limit=1)
        assert len(repo.list_jobs(base)) == 1
        assert repo.count_jobs(base) == 2
        assert repo.count_search("software", base) == 2
        assert len(repo.search_jobs("software", replace(base, limit=None))) == 2
        assert repo.count_jobs(replace(base, source="simplify")) == 1
        assert repo.count_jobs(replace(base, role=RoleCategory.DATA)) == 0
        assert (
            repo.count_jobs(
                replace(base, industries=frozenset({Industry.GAMING, Industry.FASHION}))
            )
            == 3
        )
        assert repo.count_jobs(replace(base, industries=frozenset({Industry.UNKNOWN}))) == 2
        assert (
            repo.count_jobs(
                replace(base, industries=frozenset({Industry.UNKNOWN, Industry.GAMING}))
            )
            == 4
        )
        assert repo.count_jobs(JobFilters()) == 7


def test_preferences_roundtrip_and_explicit_override(tmp_path: Path) -> None:
    prefs = revise_preferences(Preferences(), {}, industries=["gaming,unknown"])
    path = tmp_path / "preferences.json"
    save_preferences(prefs, path)
    assert load_preferences(path).preferences == prefs
    assert prefs.narrows
    assert prefs.apply(JobFilters()).industries == {Industry.GAMING, Industry.UNKNOWN}
    explicit = JobFilters(industries=frozenset({Industry.FASHION}))
    assert prefs.apply(explicit).industries == explicit.industries
    state = FilterState()
    state.choose("industry", "unknown")
    assert state.as_filters().industries == {Industry.UNKNOWN}
    assert any(row[:2] == ("industry", "unknown") for row in filter_rows(state))


@pytest.mark.parametrize("command", ["list", "search"])
def test_cli_filter_and_mine(stored: Path, command: str) -> None:
    runner = CliRunner()
    saved = runner.invoke(app, ["prefs", "--industry", "gaming", "--json"])
    assert saved.exit_code == 0, saved.output
    args = [command, *(["software"] if command == "search" else []), "--db", str(stored), "--json"]
    result = runner.invoke(app, [*args, "--industry", "fashion", "--industry", "fintech"])
    assert result.exit_code == 0, result.output
    assert "SSENSE" in result.output and "Stripe" in result.output
    assert "Riot Games" not in result.output
    mine = runner.invoke(app, [*args, "--mine"])
    assert mine.exit_code == 0, mine.output
    assert "Riot Games" in mine.output and "Stripe" not in mine.output
    bad = runner.invoke(app, [*args, "--industry", "banking"])
    assert bad.exit_code == 2


async def test_mcp_schema_filter_pagination_and_preferences(stored: Path) -> None:
    async with Client(build_server(stored, actions=True)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        for name in ("search_postings", "export_postings", "set_preferences"):
            assert "industries" in tools[name].input_schema["properties"]
        result = await client.call_tool("search_postings", {"industries": ["gaming"], "limit": 1})
        assert isinstance(result.content[0], TextContent)
        first = json.loads(result.content[0].text)
        assert first["total"] == 2
        assert first["next_cursor"]
        second = await client.call_tool(
            "search_postings",
            {
                "industries": ["gaming"],
                "limit": 1,
                "cursor": first["next_cursor"],
            },
        )
        assert isinstance(second.content[0], TextContent)
        page = json.loads(second.content[0].text)
        assert page["next_cursor"] is None
        assert first["postings"] != page["postings"]
        await client.call_tool("set_preferences", {"industries": ["fashion"]})
        mine = await client.call_tool("search_postings", {"mine": True})
        assert isinstance(mine.content[0], TextContent)
        assert "SSENSE" in mine.content[0].text and "Stripe" not in mine.content[0].text


@respx.mock
async def test_ingested_feed_uses_same_employer_industry(tmp_path: Path) -> None:
    feed = SimplifyFeed()
    respx.get(feed.plan(NOW)[0]).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "riot-feed",
                    "company_name": "Riot Games",
                    "title": "Software Engineer Intern",
                    "url": "https://simplify.jobs/p/riot-feed",
                    "locations": ["Montreal, QC"],
                    "active": True,
                    "is_visible": True,
                }
            ],
        )
    )
    async with HttpClient(
        allowed_hosts=feed.hosts, posture=profile(feed.rate_profile), jitter=False
    ) as client:
        result = await feed.fetch(client, NOW)
    normalized = normalize_batch(result.jobs)
    with closing(SqliteRepository.connect(tmp_path / "feed.db")) as repo:
        repo.apply_source_batch(
            SourceBatch(source="simplify", run_started_at=NOW, jobs=normalized[0])
        )
        assert repo.count_jobs(JobFilters(industries=frozenset({Industry.GAMING}))) == 1


def test_cli_export_filters_before_limit(stored: Path, tmp_path: Path) -> None:
    target = tmp_path / "gaming.json"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--db",
            str(stored),
            "--industry",
            "gaming",
            "--format",
            "json",
            "--out",
            str(target),
            "--all",
        ],
    )
    assert result.exit_code == 0, result.output
    exported = target.read_text()
    assert "Riot Games" in exported and "Stripe" not in exported


def test_multiple_labels_for_one_employer(stored: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalogue = {
        **employer_industries(),
        "riot games": frozenset({Industry.GAMING, Industry.FINTECH}),
    }
    monkeypatch.setattr("stage.storage.industry.employer_industries", lambda: catalogue)
    with closing(SqliteRepository.connect(stored)) as repo:
        assert repo.count_jobs(JobFilters(industries=frozenset({Industry.FINTECH}))) == 3
        assert (
            repo.count_jobs(JobFilters(industries=frozenset({Industry.GAMING, Industry.FINTECH})))
            == 3
        )


def test_finance_tech_and_unknown_stored_postings(stored: Path) -> None:
    with closing(SqliteRepository.connect(stored)) as repo:
        finance = JobFilters(industries=frozenset({Industry.FINANCE}), limit=None)
        assert {job.company for job in repo.list_jobs(finance)} == {
            "Stripe",
            "Royal Bank of Canada",
        }
        fintech = replace(finance, industries=frozenset({Industry.FINTECH}))
        assert {job.company for job in repo.list_jobs(fintech)} == {"Stripe"}
        tech_finance = replace(finance, industries=frozenset({Industry.TECH, Industry.FINANCE}))
        assert repo.count_jobs(tech_finance) == 2
        unknown = replace(finance, industries=frozenset({Industry.UNKNOWN}))
        assert {job.company for job in repo.list_jobs(unknown)} == {"Riot", "Unlisted"}


def test_registered_block_name_matches_industry_without_matching_blockhouse(tmp_path: Path) -> None:
    with closing(SqliteRepository.connect(tmp_path / "block.db")) as repo:
        repo.apply_source_batch(
            SourceBatch(
                source="greenhouse",
                run_started_at=NOW,
                jobs=(
                    replace(job("one", "Block"), id="greenhouse:block:one"),
                    replace(job("two", "Blockhouse"), id="greenhouse:blockhouse:two"),
                ),
            )
        )
        for industry in (Industry.FINTECH, Industry.TECH, Industry.FINANCE):
            filters = JobFilters(industries=frozenset({industry}))
            assert [posting.company for posting in repo.list_jobs(filters)] == ["Block"]
        unknown = JobFilters(industries=frozenset({Industry.UNKNOWN}))
        assert [posting.company for posting in repo.list_jobs(unknown)] == ["Blockhouse"]


def test_cli_new_industries_and_saved_preferences(stored: Path) -> None:
    runner = CliRunner()
    saved = runner.invoke(app, ["prefs", "--industry", "tech,finance", "--json"])
    assert saved.exit_code == 0, saved.output
    found = runner.invoke(app, ["list", "--db", str(stored), "--mine", "--json"])
    assert found.exit_code == 0, found.output
    assert "Stripe" in found.output and "Royal Bank of Canada" in found.output
    assert "Riot Games" not in found.output
    state = FilterState()
    for choice in ("tech", "finance"):
        state.choose("industry", choice)
        assert state.as_filters().industries == {Industry(choice)}
        assert any(row[:2] == ("industry", choice) for row in filter_rows(state))


async def test_mcp_new_industry_choices(stored: Path) -> None:
    async with Client(build_server(stored, actions=True)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        for name in ("search_postings", "export_postings", "set_preferences"):
            schema = tools[name].input_schema["properties"]["industries"]
            assert {"tech", "finance"} <= set(schema["anyOf"][0]["items"]["enum"])
        result = await client.call_tool("search_postings", {"industries": ["finance", "tech"]})
        assert isinstance(result.content[0], TextContent)
        found = json.loads(result.content[0].text)
        assert found["total"] == 2
        assert {posting["company"] for posting in found["postings"]} == {
            "Stripe",
            "Royal Bank of Canada",
        }
        saved = await client.call_tool("set_preferences", {"industries": ["tech", "finance"]})
        assert not saved.is_error
        mine = await client.call_tool("search_postings", {"mine": True})
        assert isinstance(mine.content[0], TextContent)
        assert json.loads(mine.content[0].text)["total"] == 2
