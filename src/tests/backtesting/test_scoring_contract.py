import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.scoring_contract import (
    CAPTAIN_MULTIPLIER,
    CAPTAIN_SCORING_ENABLED,
    FORMATION_SEARCH,
    SCORING_CONTRACT_VERSION,
    actual_scores_with_captain,
    captain_policy_diagnostics,
    contract_fields,
    validate_contract_mapping,
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
    with pytest.raises(FileNotFoundError, match=r"run_metadata\.json"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_rejects_old_report_without_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps({"season": 2025}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scoring_contract_version"):
        validate_report_contract(tmp_path)


def test_validate_contract_mapping_rejects_mismatched_values() -> None:
    payload = contract_fields()
    payload["captain_multiplier"] = 2.0

    with pytest.raises(ValueError, match="captain_multiplier"):
        validate_contract_mapping(payload)


def test_validate_report_contract_rejects_non_object_metadata(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps(["cartola_standard_2026_v1"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"run_metadata\.json must contain an object"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_rejects_malformed_metadata_with_path(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"Invalid JSON in run_metadata\.json: {metadata_path}"):
        validate_report_contract(tmp_path)


def test_validate_report_contract_accepts_standard_contract(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps(contract_fields()) + "\n", encoding="utf-8")

    assert validate_report_contract(tmp_path) == contract_fields()


def _selected_for_captain_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "apelido": "A",
                "posicao": "ata",
                "nome_clube": "Club",
                "predicted_points": 8.0,
                "prior_points_std": 1.0,
                "pontuacao": 4.0,
                "is_captain": True,
            },
            {
                "id_atleta": 2,
                "apelido": "B",
                "posicao": "mei",
                "nome_clube": "Club",
                "predicted_points": 7.5,
                "prior_points_std": 0.1,
                "pontuacao": 10.0,
                "is_captain": False,
            },
            {
                "id_atleta": 3,
                "apelido": "C",
                "posicao": "tec",
                "nome_clube": "Club",
                "predicted_points": 12.0,
                "prior_points_std": 0.0,
                "pontuacao": 3.0,
                "is_captain": False,
            },
        ]
    )


def test_actual_scores_with_captain_scores_selected_squad_actuals() -> None:
    scores = actual_scores_with_captain(_selected_for_captain_policy(), actual_column="pontuacao")

    assert scores == {
        "actual_points_base": 17.0,
        "captain_bonus_actual": (CAPTAIN_MULTIPLIER - 1.0) * 4.0,
        "actual_points_with_captain": 19.0,
    }


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), "not-a-score"])
def test_actual_scores_with_captain_rejects_missing_or_non_finite_actual_scores(bad_value: object) -> None:
    selected = _selected_for_captain_policy()
    selected["pontuacao"] = selected["pontuacao"].astype(object)
    selected.loc[1, "pontuacao"] = bad_value

    with pytest.raises(ValueError, match="pontuacao"):
        actual_scores_with_captain(selected, actual_column="pontuacao")


def test_actual_scores_with_captain_rejects_missing_actual_column() -> None:
    selected = _selected_for_captain_policy().drop(columns=["pontuacao"])

    with pytest.raises(ValueError, match="pontuacao"):
        actual_scores_with_captain(selected, actual_column="pontuacao")


def test_actual_scores_with_captain_rejects_missing_is_captain_column() -> None:
    selected = _selected_for_captain_policy().drop(columns=["is_captain"])

    with pytest.raises(ValueError, match="exactly one captain, got 0"):
        actual_scores_with_captain(selected, actual_column="pontuacao")


@pytest.mark.parametrize("captain_flags", [[False, False, False], [True, True, False]])
def test_actual_scores_with_captain_rejects_zero_or_multiple_captains(captain_flags: list[bool]) -> None:
    selected = _selected_for_captain_policy()
    selected["is_captain"] = captain_flags

    with pytest.raises(ValueError, match=str(sum(captain_flags))):
        actual_scores_with_captain(selected, actual_column="pontuacao")


def test_captain_policy_diagnostics_emits_ev_safe_and_upside_without_resolving_squad() -> None:
    diagnostics = captain_policy_diagnostics(
        _selected_for_captain_policy(),
        predicted_column="predicted_points",
        actual_column="pontuacao",
    )

    assert [record["policy"] for record in diagnostics] == ["ev", "safe", "upside"]
    assert {record["captain_id"] for record in diagnostics} == {1, 2}

    ev, safe, upside = diagnostics
    assert ev["captain_id"] == 1
    assert safe["captain_id"] == 2
    assert upside["captain_id"] == 1
    assert safe["actual_delta_vs_ev"] == 3.0
    assert safe["actual_points_with_policy"] == 22.0
    assert all(record["captain_position"] != "tec" for record in diagnostics)


def test_captain_policy_diagnostics_omits_actual_fields_when_actual_column_is_none() -> None:
    diagnostics = captain_policy_diagnostics(
        _selected_for_captain_policy(),
        predicted_column="predicted_points",
        actual_column=None,
    )

    for record in diagnostics:
        assert record["actual_captain_points"] is None
        assert record["actual_captain_bonus"] is None
        assert record["actual_points_with_policy"] is None
        assert record["actual_delta_vs_ev"] is None


def test_captain_policy_diagnostics_defaults_missing_prior_points_std_to_zero() -> None:
    selected = _selected_for_captain_policy().drop(columns=["prior_points_std"])

    diagnostics = captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column=None)

    assert {record["policy"]: record["captain_id"] for record in diagnostics} == {
        "ev": 1,
        "safe": 1,
        "upside": 1,
    }


def test_captain_policy_diagnostics_fills_and_coerces_missing_prior_points_std_to_zero() -> None:
    selected = _selected_for_captain_policy()
    selected["prior_points_std"] = [None, "0.1", float("inf")]

    diagnostics = captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column=None)

    assert {record["policy"]: record["captain_id"] for record in diagnostics} == {
        "ev": 1,
        "safe": 1,
        "upside": 1,
    }


def test_captain_policy_diagnostics_records_are_json_serializable() -> None:
    diagnostics = captain_policy_diagnostics(
        _selected_for_captain_policy(),
        predicted_column="predicted_points",
        actual_column="pontuacao",
    )

    json.dumps(diagnostics)


def test_captain_policy_diagnostics_rejects_no_non_tecnico_candidates() -> None:
    selected = _selected_for_captain_policy()
    selected["posicao"] = "tec"

    with pytest.raises(ValueError, match="no non-tecnico captain candidates"):
        captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column=None)


def test_captain_policy_diagnostics_rejects_non_finite_actual_values_when_actual_column_is_provided() -> None:
    selected = _selected_for_captain_policy()
    selected.loc[1, "pontuacao"] = float("inf")

    with pytest.raises(ValueError, match="pontuacao"):
        captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column="pontuacao")


def test_captain_policy_diagnostics_breaks_ties_by_policy_score_predicted_and_id() -> None:
    selected = pd.DataFrame(
        [
            {
                "id_atleta": 20,
                "apelido": "LowerPred",
                "posicao": "ata",
                "nome_clube": "Club",
                "predicted_points": 7.0,
                "prior_points_std": 1.0,
                "is_captain": True,
            },
            {
                "id_atleta": 10,
                "apelido": "LowerId",
                "posicao": "mei",
                "nome_clube": "Club",
                "predicted_points": 8.0,
                "prior_points_std": 2.0,
                "is_captain": False,
            },
            {
                "id_atleta": 30,
                "apelido": "HigherId",
                "posicao": "lat",
                "nome_clube": "Club",
                "predicted_points": 8.0,
                "prior_points_std": 2.0,
                "is_captain": False,
            },
            {
                "id_atleta": 40,
                "apelido": "HigherPolicyScore",
                "posicao": "zag",
                "nome_clube": "Club",
                "predicted_points": 7.0,
                "prior_points_std": 5.0,
                "is_captain": False,
            },
        ]
    )

    diagnostics = captain_policy_diagnostics(selected, predicted_column="predicted_points", actual_column=None)

    assert {record["policy"]: record["captain_id"] for record in diagnostics} == {
        "ev": 10,
        "safe": 10,
        "upside": 40,
    }
