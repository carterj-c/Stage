from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stage.dedup import resolve_duplicates
from stage.domain import Job, LocationBucket, QuarantinedJob, RejectionReason
from stage.services.screening import normalize_batch
from stage.storage.repository import SourceBatch
from stage.storage.sqlite_repo import SqliteRepository

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
BOARD_URL = "https://job-boards.greenhouse.io/acme/jobs/8046795003"
DOCTORAL = "Pursuing Masters or PhD in Computer Science. Strong C++ skills."
OPEN_TO_ALL = "Pursuing a BS, MS, or PhD in Computer Science. Strong C++ skills."


def _board(description: str = DOCTORAL, when: datetime = NOW) -> Job:
    return Job(
        id="greenhouse:acme:8046795003",
        source="greenhouse",
        company="Acme Semiconductor",
        title_raw="Intern, Power-Thermal Simulation Software Engineer",
        title_normalized="intern power thermal simulation software engineer",
        apply_url_raw=BOARD_URL,
        description=description,
        location_raw="San Jose, CA",
        first_seen=when,
        last_seen=when,
    )


def _listed(identifier: str, title: str, url: str = BOARD_URL, when: datetime = NOW) -> Job:
    return Job(
        id=identifier,
        source="simplify",
        company="Acme",
        title_raw=title,
        title_normalized=title.lower(),
        apply_url_raw=url,
        description="",
        location_raw="San Jose, CA",
        first_seen=when,
        last_seen=when,
    )


COPY = "simplify:acme:copy"
OTHER = "simplify:acme:other"


def _copy(when: datetime = NOW) -> Job:
    return _listed(COPY, "Power-Thermal Simulation Software Engineer Intern", when=when)


def _other(when: datetime = NOW) -> Job:
    return _listed(
        OTHER, "Data Engineer Intern", "https://job-boards.greenhouse.io/acme/jobs/1", when
    )


def _screened(source: str, jobs: list[Job], when: datetime = NOW) -> SourceBatch:
    kept, rejected = normalize_batch(jobs)
    return SourceBatch(
        source=source,
        run_started_at=when,
        jobs=kept,
        quarantined=rejected,
        resolve_duplicates=resolve_duplicates,
    )


def _state(repository: SqliteRepository) -> tuple[dict[str, str | None], dict[str, str]]:
    jobs = {
        str(row["id"]): row["duplicate_of"]
        for row in repository._conn.execute("SELECT id, duplicate_of FROM jobs")
    }
    held = {
        str(row["id"]): str(row["matched_phrase"])
        for row in repository._conn.execute("SELECT id, matched_phrase FROM quarantine")
    }
    return jobs, held


def _linked(db_path: Path) -> SqliteRepository:
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("greenhouse", [_board(OPEN_TO_ALL)]))
    repository.apply_source_batch(_screened("simplify", [_copy(), _other()]))
    jobs, _ = _state(repository)
    assert jobs[COPY] == _board().id, "the fixture needs the list copy linked to its board copy"
    return repository


def test_a_list_copy_inherits_the_degree_rejection_of_its_board_copy(db_path: Path) -> None:
    repository = _linked(db_path)
    later = NOW + timedelta(hours=1)

    result = repository.apply_source_batch(
        _screened("greenhouse", [_board(DOCTORAL, later)], later)
    )
    jobs, held = _state(repository)

    assert COPY not in jobs, "a copy without a description must not be promoted and kept"
    assert held[COPY] == ("copy of greenhouse:acme:8046795003 rejected for pursuing masters or phd")
    assert result.quarantined == 2, "the copy counts as a rejection of this batch"
    repository.close()


def test_an_unrelated_list_posting_is_untouched(db_path: Path) -> None:
    repository = _linked(db_path)
    later = NOW + timedelta(hours=1)

    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, later)], later))
    jobs, held = _state(repository)

    assert jobs[OTHER] is None
    assert OTHER not in held
    repository.close()


def test_ambiguous_canonical_body_does_not_release_fresh_descriptionless_copy(
    db_path: Path,
) -> None:
    repository = _linked(db_path)
    later, latest = NOW + timedelta(hours=1), NOW + timedelta(hours=2)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, later)], later))
    repository.apply_source_batch(_screened("greenhouse", [_board(OPEN_TO_ALL, latest)], latest))
    jobs, held = _state(repository)
    assert COPY not in jobs and COPY in held
    assert _board().id not in held

    fresh = latest + timedelta(hours=1)
    listed = _copy(fresh)
    repository.apply_source_batch(_screened("simplify", [listed], fresh))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    described = fresh + timedelta(hours=1)
    repository.apply_source_batch(
        _screened("simplify", [replace(_copy(described), description=OPEN_TO_ALL)], described)
    )
    restored = repository.get_job(COPY)
    assert restored is not None and restored.duplicate_of == _board().id
    assert restored.first_seen == NOW and restored.last_seen == described
    repository.close()


def test_a_list_copy_arriving_after_its_board_copy_was_rejected_is_rejected_too(
    db_path: Path,
) -> None:
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("greenhouse", [_board()]))

    repository.apply_source_batch(_screened("simplify", [_copy(), _other()]))
    jobs, held = _state(repository)

    assert COPY not in jobs and held[COPY].startswith("copy of greenhouse:acme:8046795003 ")
    assert OTHER in jobs

    again = NOW + timedelta(hours=1)
    repository.apply_source_batch(_screened("simplify", [_copy(again)], again))
    jobs, held = _state(repository)
    assert COPY not in jobs and COPY in held, "the next list read keeps the copy rejected"
    repository.close()


def test_a_copy_with_its_own_description_is_judged_on_its_own(db_path: Path) -> None:
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("greenhouse", [_board()]))
    described = replace(_copy(), description=OPEN_TO_ALL)

    repository.apply_source_batch(_screened("simplify", [described]))
    jobs, held = _state(repository)

    assert COPY in jobs and COPY not in held
    repository.close()


def test_a_better_ranked_copy_never_inherits_from_a_list_rejection(db_path: Path) -> None:
    repository = SqliteRepository.connect(db_path)
    rejected = QuarantinedJob(
        id=COPY,
        source="simplify",
        company="Acme",
        title_raw="Power-Thermal Simulation Software Engineer Intern (MS)",
        reason=RejectionReason.OUT_OF_SCOPE_DEGREE,
        first_seen=NOW,
        last_seen=NOW,
        apply_url_raw=BOARD_URL,
        location=LocationBucket.USA,
        matched_phrase="ms",
    )
    repository.apply_source_batch(
        SourceBatch(source="simplify", run_started_at=NOW, quarantined=(rejected,))
    )

    repository.apply_source_batch(_screened("greenhouse", [_board(description="")]))
    jobs, _ = _state(repository)

    assert _board().id in jobs, "a board copy is never rejected because a list copy was"
    repository.close()


def test_a_list_copy_stored_before_its_board_copy_is_rejected_with_it(db_path: Path) -> None:
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("simplify", [_copy(), _other()]))

    result = repository.apply_source_batch(_screened("greenhouse", [_board()]))
    jobs, held = _state(repository)
    repository.close()

    assert COPY not in jobs and held[COPY].startswith("copy of greenhouse:acme:8046795003 ")
    assert OTHER in jobs and result.quarantined == 2


def test_a_second_posting_of_one_title_on_the_board_never_inherits_a_rejection(
    db_path: Path,
) -> None:
    twin = replace(
        _board(description=""),
        id="greenhouse:acme:8046795004",
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/8046795004",
    )
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("greenhouse", [_board(OPEN_TO_ALL), twin]))
    jobs, _ = _state(repository)
    assert jobs[twin.id] == _board().id, "the fixture needs the twin shown under its first posting"

    later = NOW + timedelta(hours=1)
    repository.apply_source_batch(
        _screened("greenhouse", [_board(DOCTORAL, later), replace(twin, last_seen=later)], later)
    )
    jobs, held = _state(repository)
    repository.close()

    assert jobs[twin.id] is None and twin.id not in held, "the twin may be the undergrad posting"


def test_a_board_label_is_not_part_of_the_company_name() -> None:
    labelled = replace(_board(OPEN_TO_ALL), company="Acme Semiconductor (Workday)")

    kept, _ = normalize_batch([labelled])

    assert kept[0].company == "Acme Semiconductor"


def test_a_list_copy_with_a_different_link_and_extra_words_is_still_found(
    db_path: Path,
) -> None:
    repository = SqliteRepository.connect(db_path)
    repository.apply_source_batch(_screened("greenhouse", [_board()]))
    renamed = _listed(
        COPY,
        "Power-Thermal Simulation Software Engineer Intern - Summer 2027",
        "https://careers.acme.example/jobs/power-thermal",
    )

    repository.apply_source_batch(_screened("simplify", [renamed]))
    jobs, held = _state(repository)

    assert COPY not in jobs and held[COPY].startswith("copy of greenhouse:acme:8046795003 ")
    repository.close()


def test_fresh_descriptionless_quant_copy_keeps_unknown_auth_after_promotion(
    db_path: Path,
) -> None:
    from stage.classify.scope import Rejection, to_quarantined

    repository = SqliteRepository.connect(db_path)
    title = "Quantitative Research Intern"
    main = replace(_board(OPEN_TO_ALL), title_raw=title, title_normalized=title.lower())
    copy = _listed(COPY, title)
    held = to_quarantined(
        copy,
        Rejection(
            RejectionReason.OUT_OF_SCOPE_DEGREE, f"copy of {main.id} rejected for phd required"
        ),
    )
    repository.apply_source_batch(_screened("greenhouse", [main]))
    repository.apply_source_batch(
        SourceBatch(source="simplify", run_started_at=NOW, quarantined=(held,))
    )
    observed = NOW + timedelta(hours=1)
    repository.apply_source_batch(
        _screened("simplify", [replace(copy, last_seen=observed)], observed)
    )
    restored = repository.get_job(COPY)
    assert restored is not None and restored.duplicate_of == main.id
    assert restored.work_auth_flag is None
    after = observed + timedelta(hours=1)
    repository.apply_source_batch(
        SourceBatch(source="greenhouse", run_started_at=after, closes_whole_source=True)
    )
    promoted = repository.get_job(COPY)
    assert promoted is not None and promoted.duplicate_of is None
    assert promoted.work_auth_flag is None
    repository.close()


def test_stale_quarantined_copy_stays_hidden_when_canonical_closes_until_fresh_listing(
    db_path: Path,
) -> None:
    from stage.domain import JobStatus

    repository = _linked(db_path)
    rejected = NOW + timedelta(hours=1)
    accepted = NOW + timedelta(days=20)
    closed = NOW + timedelta(days=21)
    observed = NOW + timedelta(days=22)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, rejected)], rejected))
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, accepted)], accepted)
    )
    assert repository.get_job(COPY) is None
    repository.apply_source_batch(
        SourceBatch(source="greenhouse", run_started_at=closed, closes_whole_source=True)
    )
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    assert repository.get_job(_board().id).status is JobStatus.CLOSED  # type: ignore[union-attr]

    repository.apply_source_batch(_screened("simplify", [_copy(observed)], observed))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    with_description = observed + timedelta(hours=1)
    fresh = replace(_copy(with_description), description=OPEN_TO_ALL)
    repository.apply_source_batch(_screened("simplify", [fresh], with_description))
    restored = repository.get_job(COPY)
    assert restored is not None and restored.status is JobStatus.OPEN
    assert restored.duplicate_of is None
    assert restored.first_seen == NOW and restored.last_seen == with_description
    assert COPY not in _state(repository)[1]
    repository.close()


async def test_rescreen_does_not_restore_quarantined_copy_without_source_read(
    db_path: Path,
) -> None:
    from stage.classify.scope import Rejection, to_quarantined
    from stage.services.maintenance import rescreen
    from stage.storage import open_repository

    main = _board(OPEN_TO_ALL)
    follower = to_quarantined(
        _copy(),
        Rejection(
            RejectionReason.OUT_OF_SCOPE_DEGREE,
            f"copy of {main.id} rejected for pursuing masters or phd",
        ),
    )
    async with open_repository(db_path) as repository:
        await repository.apply_source_batch(
            SourceBatch(source="greenhouse", run_started_at=NOW, jobs=(main,))
        )
        await repository.apply_source_batch(
            SourceBatch(source="simplify", run_started_at=NOW, quarantined=(follower,))
        )
        result = await rescreen(repository, now=NOW)
        assert result.updated >= 1
        assert result.released == 0 and result.quarantine_waiting == 1
        assert await repository.get_job(COPY) is None


def test_fresh_descriptionless_copy_requires_verified_identity(db_path: Path) -> None:
    repository = _linked(db_path)
    rejected = NOW + timedelta(hours=1)
    accepted = NOW + timedelta(hours=2)
    observed = NOW + timedelta(hours=3)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, rejected)], rejected))
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, accepted)], accepted)
    )
    changed = replace(
        _copy(observed),
        title_raw="Data Engineer Intern",
        title_normalized="data engineer intern",
        apply_url_raw="https://job-boards.greenhouse.io/acme/jobs/9999999",
    )
    repository.apply_source_batch(_screened("simplify", [changed], observed))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    repository.close()


def test_descriptionless_copy_needs_a_fresh_observation_and_resolver(db_path: Path) -> None:
    repository = _linked(db_path)
    rejected = NOW + timedelta(hours=1)
    accepted = NOW + timedelta(hours=2)
    observed = NOW + timedelta(hours=3)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, rejected)], rejected))
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, accepted)], accepted)
    )
    kept, rejected_rows = normalize_batch([_copy(observed)])
    assert not rejected_rows
    repository.apply_source_batch(
        SourceBatch(source="simplify", run_started_at=observed, jobs=kept)
    )
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]

    later = observed + timedelta(hours=1)
    repository.apply_source_batch(_screened("simplify", [_copy(later)], later))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    refreshed = later + timedelta(hours=1)
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, refreshed)], refreshed)
    )
    latest = refreshed + timedelta(hours=1)
    repository.apply_source_batch(_screened("simplify", [_copy(latest)], latest))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    repository.close()


def test_old_copy_object_is_not_current_source_evidence(db_path: Path) -> None:
    repository = _linked(db_path)
    rejected = NOW + timedelta(hours=1)
    accepted = NOW + timedelta(hours=2)
    attempted = NOW + timedelta(hours=3)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, rejected)], rejected))
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, accepted)], accepted)
    )
    repository.apply_source_batch(_screened("simplify", [_copy(NOW)], attempted))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    repository.close()


def test_fresh_copy_cannot_use_canonical_description_that_excludes_it(db_path: Path) -> None:
    repository = _linked(db_path)
    rejected = NOW + timedelta(hours=1)
    accepted = NOW + timedelta(hours=2)
    observed = NOW + timedelta(hours=3)
    repository.apply_source_batch(_screened("greenhouse", [_board(DOCTORAL, rejected)], rejected))
    repository.apply_source_batch(
        _screened("greenhouse", [_board(OPEN_TO_ALL, accepted)], accepted)
    )
    repository._conn.execute(
        "UPDATE jobs SET description = ? WHERE id = ?", (DOCTORAL, _board().id)
    )
    repository._conn.commit()
    repository.apply_source_batch(_screened("simplify", [_copy(observed)], observed))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    repository.close()


def test_workday_carried_description_is_not_fresh_degree_evidence(db_path: Path) -> None:
    from stage.domain import DetailFetch
    from stage.services.sync import _carried

    repository = SqliteRepository.connect(db_path)
    url = "https://acme.wd3.myworkdayjobs.com/en-US/careers/job/Seattle/x_R00001"

    def board(description: str, when: datetime) -> Job:
        return replace(
            _board(description, when),
            id="workday:acme-careers:r00001",
            source="workday",
            apply_url_raw=url,
        )

    def listed(when: datetime) -> Job:
        return _listed(COPY, "Power-Thermal Simulation Software Engineer Intern", url, when)

    def board_batch(description: str, when: datetime, *, detail: bool) -> SourceBatch:
        batch = _screened("workday", [board(description, when)], when)
        return replace(
            batch,
            detail_fetches=(DetailFetch(id=board("", when).id, resolved=True),) if detail else (),
        )

    initial = board(OPEN_TO_ALL, NOW)
    repository.apply_source_batch(board_batch(OPEN_TO_ALL, NOW, detail=True))
    repository.apply_source_batch(_screened("simplify", [listed(NOW)]))
    assert _state(repository)[0][COPY] == initial.id
    rejected = NOW + timedelta(hours=1)
    repository.apply_source_batch(board_batch(DOCTORAL, rejected, detail=True))
    assert COPY in _state(repository)[1]
    accepted = NOW + timedelta(hours=2)
    repository.apply_source_batch(board_batch(OPEN_TO_ALL, accepted, detail=True))
    assert COPY in _state(repository)[1]

    repository._conn.execute("UPDATE jobs SET description = ? WHERE id = ?", (DOCTORAL, initial.id))
    repository._conn.commit()
    repository.apply_source_batch(_screened("simplify", [listed(accepted)], accepted))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]
    repository._conn.execute(
        "UPDATE jobs SET description = ? WHERE id = ?", (OPEN_TO_ALL, initial.id)
    )
    repository._conn.commit()

    listing_time = NOW + timedelta(hours=3)
    listing = board("", listing_time)
    carried = _carried(listing, (OPEN_TO_ALL, listing.location_raw))
    assert carried.description == OPEN_TO_ALL
    repository.apply_source_batch(_screened("workday", [carried], listing_time))
    copy_time = NOW + timedelta(hours=4)
    repository.apply_source_batch(_screened("simplify", [listed(copy_time)], copy_time))
    assert repository.get_job(COPY) is None
    assert COPY in _state(repository)[1]

    detail_time = NOW + timedelta(hours=5)
    repository.apply_source_batch(board_batch(DOCTORAL, detail_time, detail=True))
    assert repository.get_job(initial.id) is None
    assert COPY in _state(repository)[1]
    eligible_time = NOW + timedelta(hours=6)
    repository.apply_source_batch(board_batch(OPEN_TO_ALL, eligible_time, detail=True))
    observed = NOW + timedelta(hours=7)
    repository.apply_source_batch(_screened("simplify", [listed(observed)], observed))
    restored = repository.get_job(COPY)
    assert restored is not None and restored.duplicate_of == initial.id
    assert restored.first_seen == NOW and restored.last_seen == observed
    assert restored.work_auth_flag is None and restored.degree_requirement.value == "unknown"
    repository.close()
