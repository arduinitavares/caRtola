import json
from pathlib import Path

import pytest

from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    CAPTAIN_SCORING_ENABLED,
    FORMATION_SEARCH,
    SCORING_CONTRACT_VERSION,
    contract_fields,
    validate_report_contract,
)


def test_scoring_contract_constants_are_standard_cartola_2026() -> None:
    assert SCORING_CONTRACT_VERSION == "cartola_standard_2026_v1"
    assert CAPTAIN_SCORING_ENABLED is True
    assert CAPTAIN_MULTIPLIER == 1.5
    assert FORMATION_SEARCH == "all_official_formations"


def test_contract_fields_are_flat_report_columns() -> None:
    assert contract_fields() == {
        "scoring_contract_version": "cartola_standard_2026_v1",
        "captain_scoring_enabled": True,
        "captain_multiplier": 1.5,
        "formation_search": "all_official_formations",
    }


def test_validate_report_contract_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_rejects_old_report_without_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps({"season": 2025}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scoring_contract_version"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_accepts_standard_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps(contract_fields()) + "\n", encoding="utf-8")

    assert validate_report_contract(tmp_path) == contract_fields()
