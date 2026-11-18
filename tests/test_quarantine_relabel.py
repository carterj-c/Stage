from datetime import UTC, datetime
from pathlib import Path

from stage.domain import QuarantinedJob, QuarantineFilters, RejectionReason
from stage.services.maintenance import rescreen
from stage.storage import open_repository
from stage.storage.repository import SourceBatch

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def _entry(identifier: str, title: str, reason: RejectionReason, phrase: str) -> QuarantinedJob:
    return QuarantinedJob(
        id=identifier,
        source="greenhouse",
        company="Acme",
        title_raw=title,
        reason=reason,
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw=f"https://jobs.example.test/{identifier}",
        location_raw="Montreal, QC",
        matched_phrase=phrase,
    )


async def _store(repository: object, entries: list[QuarantinedJob]) -> None:
    await repository.apply_source_batch(  # type: ignore[attr-defined]
        SourceBatch(source="greenhouse", run_started_at=NOW, quarantined=tuple(entries))
    )


async def _reasons(repository: object) -> dict[str, tuple[str, str]]:
    rows = await repository.list_quarantined(QuarantineFilters(limit=50))  # type: ignore[attr-defined]
    return {row.id: (row.reason.value, row.matched_phrase) for row in rows}


async def test_a_rejection_gains_the_reason_the_lexicon_now_names(db_path: Path) -> None:
    stale = _entry(
        "a",
        "Marketeer Intern",
        RejectionReason.UNKNOWN_CS_ROLE,
        "no matching CS role or source category",
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [stale])
        result = await rescreen(repository, now=NOW)

        assert result.relabelled == 1, "the sharper reason was not written back"
        reason, phrase = (await _reasons(repository))["a"]
        assert reason == RejectionReason.NOT_A_CS_ROLE.value
        assert phrase == "marketeer", "the evidence must name the rule that fired"


async def test_a_rejection_the_lexicon_still_cannot_explain_is_left_alone(db_path: Path) -> None:
    unchanged = _entry(
        "b",
        "Zzz Widget Wrangler Intern",
        RejectionReason.UNKNOWN_CS_ROLE,
        "no matching CS role or source category",
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [unchanged])
        result = await rescreen(repository, now=NOW)

        assert result.relabelled == 0, "an unchanged verdict must not be rewritten"
        assert (await _reasons(repository))["b"][0] == RejectionReason.UNKNOWN_CS_ROLE.value


async def test_a_posting_that_now_passes_waits_for_a_fresh_listing(
    db_path: Path,
) -> None:
    passing = _entry(
        "c",
        "Quantitative Research Intern",
        RejectionReason.UNKNOWN_CS_ROLE,
        "no matching CS role or source category",
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [passing])
        result = await rescreen(repository, now=NOW)

        assert result.released == 0 and result.quarantine_waiting == 1
        assert result.relabelled == 0
        assert await repository.get_job("c") is None
        assert "c" in await _reasons(repository)

        from dataclasses import replace

        from stage.services.screening import normalize_batch
        from stage.storage.copies import as_job

        fresh = replace(as_job(passing), last_seen=NOW)
        kept, rejected = normalize_batch([fresh])
        assert len(kept) == 1 and not rejected
        await repository.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=kept)
        )
        restored = await repository.get_job("c")
        assert restored is not None
        assert restored.work_auth_flag is None
        assert restored.degree_requirement.value == "unknown"
        assert restored.first_seen == passing.first_seen
        assert "c" not in await _reasons(repository)


async def test_legacy_cs_intern_waits_for_degree_evidence_before_release(db_path: Path) -> None:
    from dataclasses import replace

    from stage.services.screening import normalize_batch
    from stage.storage.copies import as_job

    entry = _entry(
        "legacy",
        "Software Engineer Intern",
        RejectionReason.UNKNOWN_CS_ROLE,
        "no matching CS role or source category",
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [entry])
        result = await rescreen(repository, now=NOW)
        assert result.released == 0 and result.quarantine_waiting == 1
        assert await repository.get_job(entry.id) is None
        fresh = replace(as_job(entry), description="Bachelor's degree required.")
        kept, rejected = normalize_batch([fresh])
        assert len(kept) == 1 and not rejected
        await repository.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=kept)
        )
        restored = await repository.get_job(entry.id)
        assert restored is not None and restored.first_seen == entry.first_seen
        assert restored.degree_requirement.value == "bachelors"
        assert restored.work_auth_flag is False


async def test_rescreen_preserves_explicit_degree_rejection_and_evidence(db_path: Path) -> None:
    entry = _entry(
        "degree",
        "Quantitative Research Intern",
        RejectionReason.OUT_OF_SCOPE_DEGREE,
        "PhD required",
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [entry])
        result = await rescreen(repository, now=NOW)
        assert result.released == 0 and result.relabelled == 0
        assert await repository.get_job(entry.id) is None
        assert (await _reasons(repository))[entry.id] == ("out-of-scope-degree", "PhD required")


async def test_quarantine_rescreen_pages_relabel_without_skipping(
    db_path: Path,
    monkeypatch: object,
) -> None:
    from stage.services import maintenance

    monkeypatch.setattr(maintenance, "RESCREEN_LIMIT", 2)  # type: ignore[attr-defined]
    entries = [
        _entry("a", "Marketeer Intern", RejectionReason.UNKNOWN_CS_ROLE, "old"),
        _entry("b", "Software Engineer Intern", RejectionReason.UNKNOWN_CS_ROLE, "old"),
        _entry("c", "Quantitative Researcher", RejectionReason.NOT_EARLY_CAREER, "old"),
        _entry(
            "d", "Quantitative Research Intern", RejectionReason.OUT_OF_SCOPE_DEGREE, "PhD required"
        ),
    ]
    async with open_repository(db_path) as repository:
        await _store(repository, entries)
        result = await rescreen(repository, now=NOW)
        assert result.released == 0 and result.relabelled == 1
        assert result.quarantine_waiting == 4 and not result.recovery_complete
        assert await repository.get_job("c") is None
        reasons = await _reasons(repository)
        assert reasons["a"] == ("not-a-cs-role", "marketeer")
        assert reasons["d"] == ("out-of-scope-degree", "PhD required")


async def test_removed_or_unavailable_postings_need_a_current_listing(db_path: Path) -> None:
    from dataclasses import replace
    from datetime import timedelta

    from stage.domain import CompanyVisit, JobStatus
    from stage.services.screening import normalize_batch
    from stage.storage.copies import as_job

    old = NOW - timedelta(days=30)
    stale = replace(
        _entry(
            "greenhouse:acme:stale",
            "Quantitative Research Intern",
            RejectionReason.UNKNOWN_CS_ROLE,
            "old role",
        ),
        first_seen=old,
        last_seen=NOW - timedelta(days=20),
    )
    recent = replace(
        _entry(
            "greenhouse:acme:recent",
            "Quantitative Research Intern",
            RejectionReason.UNKNOWN_CS_ROLE,
            "old role",
        ),
        first_seen=old,
        last_seen=NOW,
    )
    async with open_repository(db_path) as repository:
        await _store(repository, [stale, recent])
        await repository.apply_source_batch(
            SourceBatch(
                source="greenhouse",
                run_started_at=NOW,
                visits=(
                    CompanyVisit(board="greenhouse:acme", succeeded=False, error="unavailable"),
                ),
            )
        )
        unavailable = await rescreen(repository, now=NOW)
        assert unavailable.released == 0 and unavailable.quarantine_waiting == 2
        assert await repository.get_job(stale.id) is None
        assert await repository.get_job(recent.id) is None
        await repository.apply_source_batch(
            SourceBatch(
                source="greenhouse",
                run_started_at=NOW + timedelta(hours=1),
                closable_boards=("greenhouse:acme",),
                visits=(CompanyVisit(board="greenhouse:acme", succeeded=True),),
            )
        )
        removed = await rescreen(repository, now=NOW + timedelta(hours=1))
        assert removed.released == 0 and removed.quarantine_waiting == 2
        assert await repository.get_job(stale.id) is None
        assert await repository.get_job(recent.id) is None
        observed = replace(
            as_job(stale), first_seen=NOW + timedelta(hours=2), last_seen=NOW + timedelta(hours=2)
        )
        kept, rejected = normalize_batch([observed])
        assert len(kept) == 1 and not rejected
        await repository.apply_source_batch(
            SourceBatch(
                source="greenhouse",
                run_started_at=NOW + timedelta(hours=2),
                jobs=kept,
            )
        )
        restored = await repository.get_job(stale.id)
        assert restored is not None and restored.status is JobStatus.OPEN
        assert restored.first_seen == old and restored.last_seen == observed.last_seen
        assert restored.work_auth_flag is None and restored.degree_requirement.value == "unknown"
        assert await repository.get_job(recent.id) is None
        assert set(await _reasons(repository)) == {recent.id}
