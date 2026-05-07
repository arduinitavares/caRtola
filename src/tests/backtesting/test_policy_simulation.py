from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY
from cartola.backtesting.policy_simulation import (
    PolicySimulationError,
    load_policy_source_context,
    reproduce_no_policy_round,
)
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION, actual_scores_with_captain


def test_policy_source_rejects_fixed_budget(tmp_path: Path) -> None:
    child = _write_policy_child(tmp_path, budget_policy="fixed")

    with pytest.raises(PolicySimulationError, match="budget_policy=moving"):
        load_policy_source_context(child)


def test_policy_source_rejects_missing_required_player_prediction_columns(tmp_path: Path) -> None:
    child = _write_policy_child(tmp_path)
    predictions = pd.read_csv(child / "player_predictions.csv").drop(columns=["entrou_em_campo"])
    predictions.to_csv(child / "player_predictions.csv", index=False)

    with pytest.raises(PolicySimulationError, match="player_predictions.csv.*entrou_em_campo"):
        load_policy_source_context(child)


def test_policy_source_wraps_empty_csv_header_errors(tmp_path: Path) -> None:
    child = _write_policy_child(tmp_path)
    (child / "player_predictions.csv").write_text("", encoding="utf-8")

    with pytest.raises(PolicySimulationError, match="player_predictions.csv"):
        load_policy_source_context(child)


def test_policy_source_maps_actual_path_metadata_shape(tmp_path: Path) -> None:
    child = _write_policy_child(
        tmp_path,
        child_path=tmp_path
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup",
        metadata_overrides={
            "season": 2021,
            "model_id": None,
            "primary_model_id": None,
            "feature_pack": None,
            "strategy_roles": None,
        },
        score_column="xgboost_depth2_slow_score",
    )

    context = load_policy_source_context(child)

    assert context.season == 2021
    assert context.model_id == "xgboost_depth2_slow"
    assert context.feature_pack == "ppg_xg_matchup"
    assert context.fixture_mode == "exploratory"
    assert context.matchup_context_mode == "cartola_matchup_v1"
    assert context.score_column == "xgboost_depth2_slow_score"


def test_policy_source_rejects_duplicate_model_path_segments(tmp_path: Path) -> None:
    child = _write_policy_child(
        tmp_path,
        child_path=tmp_path
        / "model=outer_model"
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup",
        metadata_overrides={
            "season": 2021,
            "model_id": None,
            "primary_model_id": None,
            "feature_pack": None,
            "strategy_roles": None,
        },
        score_column="xgboost_depth2_slow_score",
    )

    with pytest.raises(PolicySimulationError, match="ambiguous"):
        load_policy_source_context(child)


def test_policy_source_rejects_non_adjacent_model_and_feature_pack_path_segments(tmp_path: Path) -> None:
    child = _write_policy_child(
        tmp_path,
        child_path=tmp_path
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "extra"
        / "feature_pack=ppg_xg_matchup",
        metadata_overrides={
            "season": 2021,
            "model_id": None,
            "primary_model_id": None,
            "feature_pack": None,
            "strategy_roles": None,
        },
        score_column="xgboost_depth2_slow_score",
    )

    with pytest.raises(PolicySimulationError, match="canonical child path"):
        load_policy_source_context(child)


def test_policy_source_maps_strategy_roles_primary_model(tmp_path: Path) -> None:
    child = _write_policy_child(
        tmp_path,
        metadata_overrides={
            "model_id": "xgboost_depth2_slow",
            "strategy_roles": {
                "baseline": "baseline",
                "xgboost_depth2_slow": "primary_model",
                "price": "price",
            },
        },
        score_column="xgboost_depth2_slow_score",
    )

    context = load_policy_source_context(child)

    assert context.model_id == "xgboost_depth2_slow"
    assert context.score_column == "xgboost_depth2_slow_score"


def test_no_policy_reproduces_selected_ids_and_captain(synthetic_policy_child: Path) -> None:
    result = reproduce_no_policy_round(synthetic_policy_child, round_number=5)

    assert result.status == "ok"
    assert result.selected_ids_match
    assert result.captain_id_match
    assert result.formation_match
    assert result.budget_used_delta == pytest.approx(0.0)
    assert result.predicted_points_delta == pytest.approx(0.0)
    assert result.actual_points_delta == pytest.approx(0.0)
    assert result.failure_reason is None


def test_no_policy_reproduction_detects_solver_status_mismatch(synthetic_policy_child: Path) -> None:
    round_results_path = synthetic_policy_child / "round_results.csv"
    round_results = pd.read_csv(round_results_path)
    round_results.loc[0, "solver_status"] = "Infeasible"
    round_results.to_csv(round_results_path, index=False)

    result = reproduce_no_policy_round(synthetic_policy_child, round_number=5)

    assert result.status == "mismatch"
    assert result.failure_reason is not None
    assert "solver_status" in result.failure_reason


def test_no_policy_reproduction_detects_selected_id_mismatch(synthetic_policy_child: Path) -> None:
    selected_players_path = synthetic_policy_child / "selected_players.csv"
    selected_players = pd.read_csv(selected_players_path)
    selected_players.loc[selected_players.index[0], "id_atleta"] = 999999
    selected_players.to_csv(selected_players_path, index=False)

    result = reproduce_no_policy_round(synthetic_policy_child, round_number=5)

    assert result.status == "mismatch"
    assert not result.selected_ids_match
    assert result.failure_reason is not None
    assert "selected_ids" in result.failure_reason


def test_no_policy_reproduction_detects_captain_mismatch(synthetic_policy_child: Path) -> None:
    round_results_path = synthetic_policy_child / "round_results.csv"
    round_results = pd.read_csv(round_results_path)
    selected_players = pd.read_csv(synthetic_policy_child / "selected_players.csv")
    non_captain = selected_players.loc[~selected_players["is_captain"].astype(bool)].iloc[0]
    round_results.loc[0, "captain_id"] = int(non_captain["id_atleta"])
    round_results.to_csv(round_results_path, index=False)

    result = reproduce_no_policy_round(synthetic_policy_child, round_number=5)

    assert result.status == "mismatch"
    assert not result.captain_id_match
    assert result.failure_reason is not None
    assert "captain_id" in result.failure_reason


def test_no_policy_reproduction_rejects_duplicate_source_selected_ids(synthetic_policy_child: Path) -> None:
    selected_players_path = synthetic_policy_child / "selected_players.csv"
    selected_players = pd.read_csv(selected_players_path)
    duplicate = selected_players.iloc[[0]]
    selected_players = pd.concat([selected_players, duplicate], ignore_index=True)
    selected_players.to_csv(selected_players_path, index=False)

    with pytest.raises(PolicySimulationError, match="duplicate.*id_atleta"):
        reproduce_no_policy_round(synthetic_policy_child, round_number=5)


def test_no_policy_reproduction_rejects_missing_source_captain(synthetic_policy_child: Path) -> None:
    selected_players_path = synthetic_policy_child / "selected_players.csv"
    selected_players = pd.read_csv(selected_players_path)
    selected_players["is_captain"] = False
    selected_players.to_csv(selected_players_path, index=False)

    with pytest.raises(PolicySimulationError, match="exactly one source captain"):
        reproduce_no_policy_round(synthetic_policy_child, round_number=5)


def test_no_policy_reproduction_rejects_multiple_source_captains(synthetic_policy_child: Path) -> None:
    selected_players_path = synthetic_policy_child / "selected_players.csv"
    selected_players = pd.read_csv(selected_players_path)
    selected_players.loc[selected_players.index[:2], "is_captain"] = True
    selected_players.to_csv(selected_players_path, index=False)

    with pytest.raises(PolicySimulationError, match="exactly one source captain"):
        reproduce_no_policy_round(synthetic_policy_child, round_number=5)


def test_no_policy_reproduction_rejects_non_integral_source_round(synthetic_policy_child: Path) -> None:
    round_results_path = synthetic_policy_child / "round_results.csv"
    round_results = pd.read_csv(round_results_path)
    round_results["rodada"] = round_results["rodada"].astype(float)
    round_results.loc[0, "rodada"] = 5.5
    round_results.to_csv(round_results_path, index=False)

    with pytest.raises(PolicySimulationError, match="round_results.csv.*rodada.*whole-number"):
        reproduce_no_policy_round(synthetic_policy_child, round_number=5)


@pytest.fixture
def synthetic_policy_child(tmp_path: Path) -> Path:
    return _write_policy_child(tmp_path)


def _write_policy_child(
    tmp_path: Path,
    *,
    child_path: Path | None = None,
    budget_policy: str = "moving",
    metadata_overrides: dict[str, object] | None = None,
    score_column: str = "test_model_score",
) -> Path:
    child = child_path if child_path is not None else tmp_path / "child"
    child.mkdir(parents=True)
    model_id = _model_id_from_score_column(score_column)
    metadata: dict[str, object] = {
        "season": 2025,
        "model_id": model_id,
        "feature_pack": "synthetic_pack",
        "budget_policy": budget_policy,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "fixture_mode": "exploratory",
        "matchup_context_mode": "cartola_matchup_v1",
        "start_round": 5,
        "initial_budget": 100.0,
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    candidates = _synthetic_candidates(score_column=score_column)
    candidates.to_csv(child / "player_predictions.csv", index=False)
    _write_matching_source_outputs(child, candidates=candidates, score_column=score_column, model_id=model_id)
    return child


def _write_matching_source_outputs(
    child: Path,
    *,
    candidates: pd.DataFrame,
    score_column: str,
    model_id: str,
) -> None:
    result = optimize_squad(
        candidates,
        score_column=score_column,
        config=BacktestConfig(season=2025, start_round=5, budget=100.0),
        budget=100.0,
        policy=NO_POLICY,
    )
    actual_scores = actual_scores_with_captain(result.selected, actual_column="pontuacao")
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": model_id,
                "solver_status": result.status,
                "formation": result.formation_name,
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": result.budget_used,
                "actual_points_with_captain": actual_scores["actual_points_with_captain"],
                "predicted_points_with_captain": result.predicted_points_with_captain,
                "captain_id": result.captain_id,
            }
        ]
    )
    selected_players = result.selected.copy()
    selected_players["rodada"] = 5
    selected_players["strategy"] = model_id
    round_results.to_csv(child / "round_results.csv", index=False)
    selected_players.to_csv(child / "selected_players.csv", index=False)


def _synthetic_candidates(*, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scores_by_position = {
        "gol": [11.0, 1.0],
        "lat": [10.0, 9.0, 1.0],
        "zag": [8.0, 7.0, 6.0, 1.0],
        "mei": [15.0, 14.0, 13.0, 12.0, 11.0, 1.0],
        "ata": [18.0, 17.0, 16.0, 1.0],
        "tec": [5.0, 1.0],
    }
    player_id = 1
    for position, scores in scores_by_position.items():
        for score in scores:
            rows.append(_candidate_row(player_id, position=position, score=score, score_column=score_column))
            player_id += 1
    return pd.DataFrame(rows)


def _candidate_row(player_id: int, *, position: str, score: float, score_column: str) -> dict[str, object]:
    return {
        "rodada": 5,
        "id_atleta": player_id,
        "apelido": f"{position}-{player_id}",
        "posicao": position,
        "id_clube": 100 + player_id,
        "nome_clube": f"Club {player_id}",
        "preco_pre_rodada": 1.0,
        "pontuacao": score / 2.0,
        "entrou_em_campo": True,
        "variacao": 0.0,
        "baseline_score": score / 3.0,
        "price_score": 1.0,
        score_column: score,
    }


def _model_id_from_score_column(score_column: str) -> str:
    suffix = "_score"
    if score_column.endswith(suffix):
        return score_column[: -len(suffix)]
    return score_column
