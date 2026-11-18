import json
from functools import cache
from importlib.resources import files

from stage.domain.industry import Industry


def validate_catalogue(entries: list[dict[str, object]]) -> dict[str, frozenset[Industry]]:
    from stage.storage.sql import fold_company

    aliases: dict[str, frozenset[Industry]] = {}
    for entry in entries:
        labels = entry["industries"]
        names = entry["aliases"]
        if not isinstance(labels, list) or not labels or not isinstance(names, list) or not names:
            raise ValueError("Each employer needs industries and explicit aliases")
        industries = frozenset(Industry(value) for value in labels)
        if Industry.UNKNOWN in industries:
            raise ValueError("Unknown is reserved for absent employers")
        for name in names:
            if not isinstance(name, str) or not (folded := fold_company(name)):
                raise ValueError("Employer aliases must normalize to a nonempty name")
            if folded in aliases:
                raise ValueError(f"Colliding employer alias: {name}")
            aliases[folded] = industries
    return aliases


@cache
def employer_industries() -> dict[str, frozenset[Industry]]:
    data = files("stage.data").joinpath("industries.json").read_text(encoding="utf-8")
    return validate_catalogue(json.loads(data))
