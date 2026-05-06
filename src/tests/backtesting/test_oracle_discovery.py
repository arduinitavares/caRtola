from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

import pandas as pd
import pytest

import cartola.backtesting.oracle_discovery as oracle_discovery
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import SquadOptimizationResult
from cartola.backtesting.oracle_discovery import (
    ArtifactValidationError,
    OracleObjectiveError,
    SourceRunContext,
    adapt_oracle_result,
    add_oracle_actual_points,
    load_source_run_contexts,
    run_model_candidate_oracle,
    validate_child_artifacts,
)


def _run_selected_squad_captain_oracle(selected: pd.DataFrame) -> dict[str, object]:
    oracle = getattr(oracle_discovery, "selected_squad_captain_oracle", None)
    assert callable(oracle), "selected_squad_captain_oracle should be implemented"
    return cast("Callable[[pd.DataFrame], dict[str, object]]", oracle)(selected)


def _write_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame([{column: 1 for column in columns}]).to_csv(path, index=False)


def _valid_child_dir(tmp_path: Path) -> Path:
    child = tmp_path / "runs" / "season=2025" / "model=xgboost_depth2_l2_heavy" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    _write_csv(
        child / "round_results.csv",
        [
            "rodada",
            "strategy",
            "solver_status",
            "budget_before_round",
            "budget_after_round",
            "budget_delta",
            "budget_used",
            "actual_points_with_captain",
            "captain_id",
        ],
    )
    _write_csv(
        child / "selected_players.csv",
        [
            "rodada",
            "strategy",
            "id_atleta",
            "apelido",
            "posicao",
            "id_clube",
            "nome_clube",
            "entrou_em_campo",
            "preco_pre_rodada",
            "pontuacao",
            "variacao",
            "is_captain",
        ],
    )
    _write_csv(
        child / "player_predictions.csv",
        [
            "rodada",
            "id_atleta",
            "apelido",
            "posicao",
            "id_clube",
            "nome_clube",
            "status",
            "entrou_em_campo",
            "preco_pre_rodada",
            "pontuacao",
            "variacao",
            "baseline_score",
            "price_score",
            "xgboost_depth2_l2_heavy_score",
        ],
    )
    _write_csv(child / "summary.csv", ["strategy", "rounds"])
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "start_round": 5,
                "initial_budget": 100.0,
                "budget_policy": "moving",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "scoring_contract_version": "cartola_standard_2026_v1",
                "fixture_source_directory": "data/01_raw/fixtures/2025",
                "fixture_manifest_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    return child


def _write_valid_experiment_metadata(experiment: Path, child: Path, output_path: str | None = None) -> None:
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "current_year": 2026,
                "child_runs": [
                    {
                        "child_id": "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
                        "output_path": output_path if output_path is not None else str(child),
                        "season": 2025,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "metadata": {
                            "budget_policy": "moving",
                            "matchup_context_mode": "cartola_matchup_v1",
                        },
                        "strategy_roles": {
                            "baseline": "baseline",
                            "price": "price",
                            "xgboost_depth2_l2_heavy": "primary_model",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_source_run_contexts_derives_score_columns_from_parent_metadata(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    experiment = tmp_path
    _write_valid_experiment_metadata(experiment, child)

    contexts = load_source_run_contexts(experiment)

    assert contexts == [
        SourceRunContext(
            source_experiment_id="exp-1",
            source_child_id="season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
            source_child_path=child,
            season=2025,
            model_id="xgboost_depth2_l2_heavy",
            feature_pack="ppg_xg_matchup",
            fixture_mode="exploratory",
            matchup_context_mode="cartola_matchup_v1",
            budget_policy="moving",
            primary_strategy="xgboost_depth2_l2_heavy",
            strategy_score_columns={
                "baseline": "baseline_score",
                "price": "price_score",
                "xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score",
            },
            analyzed_strategies=("baseline", "price", "xgboost_depth2_l2_heavy"),
            current_year=2026,
        )
    ]


def test_load_source_run_contexts_resolves_project_relative_child_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    child = _valid_child_dir(project)
    experiment = project / "data" / "08_reporting" / "experiments" / "exp-1"
    _write_valid_experiment_metadata(experiment, child, output_path=str(child.relative_to(project)))
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    context = load_source_run_contexts(experiment)[0]
    artifacts = validate_child_artifacts(context)

    assert context.source_child_path == child
    assert not artifacts.round_results.empty


def test_load_source_run_contexts_rejects_conflicting_top_level_and_nested_metadata(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    (tmp_path / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "child_runs": [
                    {
                        "child_id": "child-1",
                        "output_path": str(child),
                        "season": 2025,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "budget_policy": "fixed",
                        "metadata": {
                            "budget_policy": "moving",
                            "matchup_context_mode": "cartola_matchup_v1",
                        },
                        "strategy_roles": {
                            "xgboost_depth2_l2_heavy": "primary_model",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="experiment_metadata.json.*budget_policy"):
        load_source_run_contexts(tmp_path)


def test_load_source_run_contexts_rejects_missing_parent_metadata_field(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    (tmp_path / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "child_runs": [
                    {
                        "child_id": "child-1",
                        "output_path": str(child),
                        "season": 2025,
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "metadata": {
                            "budget_policy": "moving",
                            "matchup_context_mode": "cartola_matchup_v1",
                        },
                        "strategy_roles": {
                            "xgboost_depth2_l2_heavy": "primary_model",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="experiment_metadata.json.*child_runs\\[0\\].model_id"):
        load_source_run_contexts(tmp_path)


def test_load_source_run_contexts_rejects_non_list_child_runs(tmp_path: Path) -> None:
    (tmp_path / "experiment_metadata.json").write_text(
        json.dumps({"experiment_id": "exp-1", "child_runs": {"child_id": "child-1"}}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="experiment_metadata.json.*child_runs.*list"):
        load_source_run_contexts(tmp_path)


def test_load_source_run_contexts_wraps_malformed_parent_json(tmp_path: Path) -> None:
    (tmp_path / "experiment_metadata.json").write_text("{", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="experiment_metadata.json"):
        load_source_run_contexts(tmp_path)


def test_validate_child_artifacts_rejects_missing_score_column(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    predictions = pd.read_csv(child / "player_predictions.csv").drop(columns=["xgboost_depth2_l2_heavy_score"])
    predictions.to_csv(child / "player_predictions.csv", index=False)
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        budget_policy="moving",
        primary_strategy="xgboost_depth2_l2_heavy",
        strategy_score_columns={"xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"},
        analyzed_strategies=("xgboost_depth2_l2_heavy",),
    )

    with pytest.raises(ArtifactValidationError, match="xgboost_depth2_l2_heavy_score"):
        validate_child_artifacts(context)


def test_validate_child_artifacts_rejects_old_fixed_budget_artifacts(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="ridge",
        feature_pack="ppg_xg",
        fixture_mode="none",
        matchup_context_mode="none",
        budget_policy="fixed",
        primary_strategy="ridge",
        strategy_score_columns={"ridge": "ridge_score"},
        analyzed_strategies=("ridge",),
    )

    with pytest.raises(ArtifactValidationError, match="not moving-budget compatible"):
        validate_child_artifacts(context)


def test_validate_child_artifacts_rejects_missing_round_results_column_for_moving_budget(
    tmp_path: Path,
) -> None:
    child = _valid_child_dir(tmp_path)
    round_results = pd.read_csv(child / "round_results.csv").drop(columns=["budget_before_round"])
    round_results.to_csv(child / "round_results.csv", index=False)
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        budget_policy="moving",
        primary_strategy="xgboost_depth2_l2_heavy",
        strategy_score_columns={"xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"},
        analyzed_strategies=("xgboost_depth2_l2_heavy",),
    )

    with pytest.raises(ArtifactValidationError, match="round_results.csv.*budget_before_round"):
        validate_child_artifacts(context)


def test_validate_child_artifacts_wraps_empty_csv_read_error(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    (child / "round_results.csv").write_text("", encoding="utf-8")
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        budget_policy="moving",
        primary_strategy="xgboost_depth2_l2_heavy",
        strategy_score_columns={"xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"},
        analyzed_strategies=("xgboost_depth2_l2_heavy",),
    )

    with pytest.raises(ArtifactValidationError, match="round_results.csv"):
        validate_child_artifacts(context)


def test_validate_child_artifacts_rejects_run_metadata_context_mismatch(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    metadata = json.loads((child / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["season"] = 2024
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    context = SourceRunContext(
        source_experiment_id="exp-1",
        source_child_id="child-1",
        source_child_path=child,
        season=2025,
        model_id="xgboost_depth2_l2_heavy",
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        matchup_context_mode="cartola_matchup_v1",
        budget_policy="moving",
        primary_strategy="xgboost_depth2_l2_heavy",
        strategy_score_columns={"xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"},
        analyzed_strategies=("xgboost_depth2_l2_heavy",),
    )

    with pytest.raises(ArtifactValidationError, match="run_metadata.json.*season"):
        validate_child_artifacts(context)


def test_add_oracle_actual_points_maps_explicit_dnp_null_to_zero() -> None:
    frame = pd.DataFrame(
        [
            {"id_atleta": 1, "pontuacao": None, "entrou_em_campo": False},
            {"id_atleta": 2, "pontuacao": 7.5, "entrou_em_campo": True},
        ]
    )

    result = add_oracle_actual_points(frame)

    assert result["oracle_actual_points"].tolist() == [0.0, 7.5]


def test_add_oracle_actual_points_allows_finite_points_without_entered_column() -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": 7.5}])

    result = add_oracle_actual_points(frame)

    assert result["oracle_actual_points"].tolist() == [7.5]


def test_add_oracle_actual_points_does_not_mutate_input_frame() -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": 3.5, "entrou_em_campo": True}])

    result = add_oracle_actual_points(frame)

    assert "oracle_actual_points" not in frame.columns
    assert result is not frame


@pytest.mark.parametrize(
    "entered",
    [None, "maybe"],
)
def test_add_oracle_actual_points_rejects_ambiguous_null(entered: object) -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": None, "entrou_em_campo": entered}])

    with pytest.raises(OracleObjectiveError, match="Ambiguous missing pontuacao"):
        add_oracle_actual_points(frame)


def test_add_oracle_actual_points_rejects_missing_entered_column_for_null() -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": None}])

    with pytest.raises(OracleObjectiveError, match="Missing required oracle objective columns"):
        add_oracle_actual_points(frame)


@pytest.mark.parametrize(
    "pontuacao",
    ["not-a-number", float("inf")],
)
def test_add_oracle_actual_points_rejects_invalid_pontuacao(pontuacao: object) -> None:
    frame = pd.DataFrame([{"id_atleta": 1, "pontuacao": pontuacao, "entrou_em_campo": False}])

    with pytest.raises(OracleObjectiveError, match="Invalid pontuacao"):
        add_oracle_actual_points(frame)


def test_adapt_oracle_result_renames_prediction_fields() -> None:
    result = SquadOptimizationResult(
        selected=pd.DataFrame(),
        status="Optimal",
        budget_used=98.0,
        predicted_points=75.0,
        predicted_points_base=70.0,
        captain_bonus_predicted=5.0,
        predicted_points_with_captain=75.0,
        formation_name="4-3-3",
        selected_count=12,
        captain_id=10,
        captain_name="A",
        captain_position="ata",
        captain_club="FLA",
        captain_predicted_points=10.0,
        captain_multiplier=1.5,
        scoring_contract_version="cartola_standard_2026_v1",
        formation_scores=[],
        captain_policy_diagnostics=[],
    )

    adapted = adapt_oracle_result(result)

    assert adapted == {
        "optimizer_status": "Optimal",
        "optimizer_formation": "4-3-3",
        "optimizer_budget_used": 98.0,
        "optimizer_selected_count": 12,
        "optimizer_captain_id": 10,
        "oracle_actual_points_base": 70.0,
        "oracle_captain_bonus_actual": 5.0,
        "oracle_actual_points_with_captain": 75.0,
        "oracle_objective_points": 75.0,
    }
    assert all("predicted" not in key for key in adapted)


def test_selected_squad_captain_oracle_picks_best_actual_selected_non_tecnico() -> None:
    selected = pd.DataFrame(
        [
            {
                "id_atleta": 10,
                "apelido": "Model Captain",
                "posicao": "mei",
                "id_clube": 1,
                "nome_clube": "FLA",
                "status": "Provavel",
                "pontuacao": 6.0,
                "entrou_em_campo": True,
                "is_captain": True,
            },
            {
                "id_atleta": 20,
                "apelido": "Oracle Captain",
                "posicao": "ata",
                "id_clube": 2,
                "nome_clube": "PAL",
                "status": "Provavel",
                "pontuacao": 10.0,
                "entrou_em_campo": True,
                "is_captain": False,
            },
            {
                "id_atleta": 30,
                "apelido": "Tecnico",
                "posicao": "tec",
                "id_clube": 3,
                "nome_clube": "SAO",
                "status": "Provavel",
                "pontuacao": 12.0,
                "entrou_em_campo": True,
                "is_captain": False,
            },
        ]
    )

    profile = _run_selected_squad_captain_oracle(selected)

    assert profile["captain_id"] == 20
    assert profile["captain_name"] == "Oracle Captain"
    assert profile["captain_position"] == "ata"
    assert profile["captain_club"] == "PAL"
    assert profile["captain_status"] == "Provavel"
    assert profile["captain_oracle_actual_points"] == 10.0
    assert profile["model_captain_id"] == 10
    assert profile["model_captain_actual_points"] == 6.0
    assert profile["selected_squad_captain_regret"] == pytest.approx(2.0)
    assert profile["full_market_status"] == "not_available"


def test_selected_squad_captain_oracle_sets_missing_optional_profile_fields_to_none() -> None:
    selected = pd.DataFrame(
        [
            {"pontuacao": 5.0, "entrou_em_campo": True, "posicao": "lat", "is_captain": True},
            {"pontuacao": 4.0, "entrou_em_campo": True, "posicao": "mei", "is_captain": False},
        ]
    )

    profile = _run_selected_squad_captain_oracle(selected)

    assert profile["captain_id"] is None
    assert profile["captain_name"] is None
    assert profile["captain_club"] is None
    assert profile["captain_status"] is None
    assert profile["captain_model_score"] is None
    assert profile["model_captain_id"] is None
    assert profile["selected_squad_captain_regret"] == 0.0


def test_selected_squad_captain_oracle_does_not_allow_tecnico_as_oracle_captain() -> None:
    selected = pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "apelido": "Model Captain",
                "posicao": "mei",
                "nome_clube": "FLA",
                "pontuacao": 5.0,
                "entrou_em_campo": True,
                "is_captain": True,
            },
            {
                "id_atleta": 2,
                "apelido": "Tecnico",
                "posicao": "tec",
                "nome_clube": "PAL",
                "pontuacao": 20.0,
                "entrou_em_campo": True,
                "is_captain": False,
            },
        ]
    )

    profile = _run_selected_squad_captain_oracle(selected)

    assert profile["captain_id"] == 1
    assert profile["captain_position"] == "mei"
    assert profile["selected_squad_captain_regret"] == 0.0


def test_selected_squad_captain_oracle_rejects_missing_model_captain() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "posicao": "mei", "pontuacao": 5.0, "entrou_em_campo": True, "is_captain": False},
        ]
    )

    with pytest.raises(OracleObjectiveError, match="exactly one model captain.*got 0"):
        _run_selected_squad_captain_oracle(selected)


def test_selected_squad_captain_oracle_rejects_multiple_model_captains() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "posicao": "mei", "pontuacao": 5.0, "entrou_em_campo": True, "is_captain": True},
            {"id_atleta": 2, "posicao": "ata", "pontuacao": 6.0, "entrou_em_campo": True, "is_captain": True},
        ]
    )

    with pytest.raises(OracleObjectiveError, match="exactly one model captain.*got 2"):
        _run_selected_squad_captain_oracle(selected)


def test_selected_squad_captain_oracle_rejects_selected_squad_without_non_tecnico() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "posicao": "tec", "pontuacao": 5.0, "entrou_em_campo": True, "is_captain": True},
        ]
    )

    with pytest.raises(OracleObjectiveError, match="no non-tecnico"):
        _run_selected_squad_captain_oracle(selected)


def _model_candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    positions = ["gol", "lat", "lat", "zag", "zag", "mei", "mei", "mei", "ata", "ata", "ata", "tec"]
    for index, position in enumerate(positions, start=1):
        rows.append(
            {
                "rodada": 5,
                "id_atleta": index,
                "apelido": f"P{index}",
                "posicao": position,
                "id_clube": 100 + index,
                "nome_clube": f"C{index}",
                "status": "Provavel",
                "entrou_em_campo": True,
                "preco_pre_rodada": 1.0,
                "pontuacao": float(index),
                "variacao": 0.0,
                "model_score": float(100 - index),
            }
        )
    return pd.DataFrame(rows)


def _block_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_optimize(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("optimize_squad should not be called")

    monkeypatch.setattr(oracle_discovery, "optimize_squad", fail_optimize)


def test_run_model_candidate_oracle_uses_actual_points_objective_and_model_ranks() -> None:
    candidates = _model_candidate_rows()
    config = BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=Path("."))

    row, selected = run_model_candidate_oracle(
        candidates,
        config=config,
        budget_before_round=100.0,
        score_column="model_score",
    )

    assert row["optimizer_status"] == "Optimal"
    assert row["oracle_actual_points_base"] == 78.0
    assert row["oracle_captain_bonus_actual"] == 5.5
    assert row["oracle_actual_points_with_captain"] == 83.5
    assert all("predicted" not in key for key in row)
    assert selected["is_oracle_captain"].sum() == 1
    assert int(selected.loc[selected["is_oracle_captain"], "id_atleta"].iloc[0]) == 11
    assert {
        "oracle_actual_points",
        "is_oracle_captain",
        "model_score_column",
        "model_score",
        "model_predicted_rank_overall",
        "model_predicted_rank_position",
    }.issubset(selected.columns)
    player_11 = selected.loc[selected["id_atleta"] == 11].iloc[0]
    assert player_11["oracle_actual_points"] == 11.0
    assert player_11["model_score_column"] == "model_score"
    assert player_11["model_score"] == 89.0
    assert player_11["model_predicted_rank_overall"] == 11.0
    assert player_11["model_predicted_rank_position"] == 3.0


def test_run_model_candidate_oracle_returns_empty_selection_when_optimizer_is_not_optimal() -> None:
    config = BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=Path("."))

    row, selected = run_model_candidate_oracle(
        _model_candidate_rows(),
        config=config,
        budget_before_round=0.0,
        score_column="model_score",
    )

    assert row["optimizer_status"] == "Infeasible"
    assert all("predicted" not in key for key in row)
    assert selected.empty


def test_run_model_candidate_oracle_rejects_missing_round_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_optimizer(monkeypatch)
    candidates = _model_candidate_rows().drop(columns=["rodada"])
    config = BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=Path("."))

    with pytest.raises(OracleObjectiveError, match="exactly one rodada"):
        run_model_candidate_oracle(
            candidates,
            config=config,
            budget_before_round=100.0,
            score_column="model_score",
        )


def test_run_model_candidate_oracle_rejects_multiple_rounds_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_optimizer(monkeypatch)
    candidates = _model_candidate_rows()
    candidates.loc[0, "rodada"] = 6
    config = BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=Path("."))

    with pytest.raises(OracleObjectiveError, match="exactly one rodada"):
        run_model_candidate_oracle(
            candidates,
            config=config,
            budget_before_round=100.0,
            score_column="model_score",
        )


def test_run_model_candidate_oracle_rejects_duplicate_round_athletes_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_optimizer(monkeypatch)
    candidates = pd.concat([_model_candidate_rows(), _model_candidate_rows().iloc[[0]]], ignore_index=True)
    config = BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=Path("."))

    with pytest.raises(OracleObjectiveError, match="Duplicate candidate rows.*rodada.*id_atleta"):
        run_model_candidate_oracle(
            candidates,
            config=config,
            budget_before_round=100.0,
            score_column="model_score",
        )


def _run_build_oracle_discovery_report(*, experiment_path: Path, output_path: Path) -> None:
    builder = getattr(oracle_discovery, "build_oracle_discovery_report", None)
    assert callable(builder), "build_oracle_discovery_report should be implemented"
    cast("Callable[..., None]", builder)(experiment_path=experiment_path, output_path=output_path)


def _write_report_builder_experiment(
    tmp_path: Path,
    *,
    predictions: pd.DataFrame,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
) -> Path:
    child = _valid_child_dir(tmp_path)
    predictions.to_csv(child / "player_predictions.csv", index=False)
    selected_players.to_csv(child / "selected_players.csv", index=False)
    round_results.to_csv(child / "round_results.csv", index=False)
    _write_valid_experiment_metadata(tmp_path, child)
    return tmp_path


def _report_builder_predictions() -> pd.DataFrame:
    predictions = _model_candidate_rows()
    predictions["baseline_score"] = predictions["model_score"]
    predictions["price_score"] = predictions["preco_pre_rodada"]
    predictions["xgboost_depth2_l2_heavy_score"] = predictions["model_score"]
    return predictions


def test_build_oracle_discovery_report_writes_expected_artifacts(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    expected_files = {
        "oracle_round_results.csv",
        "oracle_selected_players.csv",
        "oracle_captain_profiles.csv",
        "oracle_player_profiles.csv",
        "model_vs_oracle_recall.csv",
        "profile_gap_summary.csv",
        "invalid_oracle_rows.csv",
        "oracle_discovery_metadata.json",
        "oracle_knowledge_discovery.html",
    }
    assert {path.name for path in output.iterdir()} == expected_files

    round_output = pd.read_csv(output / "oracle_round_results.csv")
    assert "oracle_actual_points_with_captain" in round_output.columns
    assert not any(column.startswith("predicted_") for column in round_output.columns)
    assert round_output.loc[0, "source_mode"] == "artifact"
    assert round_output.loc[0, "source_experiment_id"] == "exp-1"
    assert round_output.loc[0, "source_child_id"] == (
        "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup"
    )
    assert round_output.loc[0, "season"] == 2025
    assert round_output.loc[0, "rodada"] == 5
    assert round_output.loc[0, "model_id"] == "xgboost_depth2_l2_heavy"
    assert round_output.loc[0, "feature_pack"] == "ppg_xg_matchup"
    assert round_output.loc[0, "fixture_mode"] == "exploratory"
    assert round_output.loc[0, "matchup_context_mode"] == "cartola_matchup_v1"
    assert round_output.loc[0, "budget_policy"] == "moving"
    assert round_output.loc[0, "oracle_type"] == "budget_constrained"
    assert round_output.loc[0, "candidate_universe"] == "model_candidate"
    assert round_output.loc[0, "budget_path"] == "model_budget_path"
    assert round_output.loc[0, "budget_before_round"] == 100.0
    assert round_output.loc[0, "full_market_status"] == "not_available"

    oracle_selected = pd.read_csv(output / "oracle_selected_players.csv")
    assert set(oracle_selected["rodada"]) == {5}
    assert set(oracle_selected["oracle_type"]) == {"budget_constrained"}
    assert set(oracle_selected["candidate_universe"]) == {"model_candidate"}
    assert set(oracle_selected["budget_path"]) == {"model_budget_path"}
    assert set(oracle_selected["model_score_column"]) == {"xgboost_depth2_l2_heavy_score"}

    captain_profiles = pd.read_csv(output / "oracle_captain_profiles.csv")
    assert captain_profiles.loc[0, "captain_id"] == 11
    assert captain_profiles.loc[0, "model_captain_id"] == 10
    assert captain_profiles.loc[0, "full_market_status"] == "not_available"

    recall = pd.read_csv(output / "model_vs_oracle_recall.csv")
    assert "absent_from_model_candidate_artifact" in recall.columns
    assert "not_visible" not in ",".join(recall.columns)
    assert set(recall["full_market_status"]) == {"not_available"}
    assert set(recall["in_model_candidate_artifact"]) == {True}
    assert set(recall["absent_from_model_candidate_artifact"]) == {False}

    profile_headers = pd.read_csv(output / "oracle_player_profiles.csv").columns.tolist()
    gap_headers = pd.read_csv(output / "profile_gap_summary.csv").columns.tolist()
    invalid_headers = pd.read_csv(output / "invalid_oracle_rows.csv").columns.tolist()
    assert profile_headers
    assert gap_headers
    assert "invalid_reason" in invalid_headers

    metadata = json.loads((output / "oracle_discovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_mode"] == "artifact"
    assert metadata["source_experiment_path"] == str(experiment)


def test_build_oracle_discovery_report_includes_strategy_in_shared_identity(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    selected_frames: list[pd.DataFrame] = []
    round_rows: list[dict[str, object]] = []
    strategies = ("baseline", "price")
    for strategy in strategies:
        selected = predictions.head(12).copy()
        selected["strategy"] = strategy
        selected["is_captain"] = selected["id_atleta"].eq(10)
        selected_frames.append(selected)
        round_rows.append(
            {
                "rodada": 5,
                "strategy": strategy,
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=pd.DataFrame(round_rows),
        selected_players=pd.concat(selected_frames, ignore_index=True),
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    identity_columns = [
        "source_mode",
        "source_experiment_id",
        "source_child_id",
        "season",
        "rodada",
        "strategy",
        "model_id",
        "feature_pack",
        "fixture_mode",
        "matchup_context_mode",
        "budget_policy",
        "oracle_type",
        "candidate_universe",
        "budget_path",
    ]
    round_output = pd.read_csv(output / "oracle_round_results.csv")
    assert set(round_output["strategy"]) == set(strategies)
    assert round_output.duplicated(
        subset=[column for column in identity_columns if column != "strategy"],
        keep=False,
    ).all()
    assert not round_output.duplicated(subset=identity_columns).any()

    for artifact_name in (
        "oracle_selected_players.csv",
        "oracle_captain_profiles.csv",
        "model_vs_oracle_recall.csv",
    ):
        frame = pd.read_csv(output / artifact_name)
        assert "strategy" in frame.columns
        assert set(frame["strategy"]) == set(strategies)

    profile_gap_summary = pd.read_csv(output / "profile_gap_summary.csv")
    invalid_rows = pd.read_csv(output / "invalid_oracle_rows.csv")
    assert "strategy" in profile_gap_summary.columns
    assert "strategy" in invalid_rows.columns


def test_build_oracle_discovery_metadata_contains_source_provenance(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    child_path = (
        tmp_path
        / "runs"
        / "season=2025"
        / "model=xgboost_depth2_l2_heavy"
        / "feature_pack=ppg_xg_matchup"
    )
    metadata = json.loads((output / "oracle_discovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_mode"] == "artifact"
    assert metadata["source_experiment_path"] == str(experiment)
    assert metadata["source_experiment_ids"] == ["exp-1"]
    assert metadata["source_child_ids"] == [
        "season=2025/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup"
    ]
    assert metadata["source_child_paths"] == [str(child_path)]
    assert metadata["seasons"] == [2025]
    assert metadata["start_round_values"] == [5]
    assert metadata["current_year_values"] == [2026]
    assert metadata["initial_budget_values"] == [100.0]
    assert metadata["budget_policies"] == ["moving"]
    assert metadata["model_ids"] == ["xgboost_depth2_l2_heavy"]
    assert metadata["feature_packs"] == ["ppg_xg_matchup"]
    assert metadata["fixture_modes"] == ["exploratory"]
    assert metadata["matchup_context_modes"] == ["cartola_matchup_v1"]
    assert metadata["footystats_modes"] == ["ppg_xg"]
    assert metadata["scoring_contract_versions"] == ["cartola_standard_2026_v1"]
    assert metadata["oracle_variants"] == ["budget_constrained", "selected_squad_captain"]
    assert metadata["candidate_universes"] == ["model_candidate", "selected_squad"]
    assert metadata["budget_paths"] == ["model_budget_path"]
    assert metadata["full_market_status"] == "not_available"
    assert metadata["disclaimer"] == "Discovery-only hindsight analysis. Not promotion evidence."

    child_provenance = metadata["source_children"][0]
    assert child_provenance["source_experiment_id"] == "exp-1"
    assert child_provenance["source_child_path"] == str(child_path)
    assert child_provenance["analyzed_strategies"] == ["baseline", "price", "xgboost_depth2_l2_heavy"]
    assert child_provenance["strategy_score_columns"]["baseline"] == "baseline_score"
    assert child_provenance["fixture_source_directory"] == "data/01_raw/fixtures/2025"


def test_build_oracle_discovery_report_writes_html_disclaimer(tmp_path: Path) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    html = (output / "oracle_knowledge_discovery.html").read_text(encoding="utf-8")
    assert "Discovery-only hindsight analysis" in html
    assert "Not promotion evidence" in html
    assert "source_mode=artifact" in html
    assert "candidate_universe=model_candidate" in html
    assert "full_market_status=not_available" in html
    assert "Oracle rounds: 1" in html
    assert "Selected-squad captain regret total: 0.50" in html
    assert "Model-candidate recall: 12" in html
    assert "Model-candidate missed: 0" in html
    lower_html = html.lower()
    assert "production promotion" not in lower_html
    assert "validated policy" not in lower_html
    assert "not-visible" not in lower_html
    assert "eligible" not in lower_html
    assert "candidate-generation failure" not in lower_html


def test_build_oracle_discovery_report_records_invalid_rows_and_continues(tmp_path: Path) -> None:
    round_5 = _report_builder_predictions()
    round_6 = _report_builder_predictions()
    round_6["rodada"] = 6
    round_6["pontuacao"] = round_6["pontuacao"].astype(object)
    round_6.loc[0, "pontuacao"] = "bad-points"
    predictions = pd.concat([round_6, round_5], ignore_index=True)

    selected_players = round_5.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 6,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 55.0,
                "budget_after_round": 55.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 0.0,
                "captain_id": 10,
            },
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            },
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    round_output = pd.read_csv(output / "oracle_round_results.csv")
    assert round_output["rodada"].tolist() == [5]

    invalid_rows = pd.read_csv(output / "invalid_oracle_rows.csv")
    assert invalid_rows.loc[0, "id_atleta"] == 1
    assert invalid_rows.loc[0, "rodada"] == 6
    assert invalid_rows.loc[0, "oracle_type"] == "budget_constrained"
    assert invalid_rows.loc[0, "candidate_universe"] == "model_candidate"
    assert invalid_rows.loc[0, "budget_path"] == "model_budget_path"
    assert invalid_rows.loc[0, "invalid_reason"] == "invalid_pontuacao"


def test_build_oracle_discovery_report_labels_selected_squad_captain_profiles(
    tmp_path: Path,
) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players.loc[selected_players["id_atleta"].eq(9), "pontuacao"] = 99.0
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    round_output = pd.read_csv(output / "oracle_round_results.csv")
    assert round_output.loc[0, "optimizer_captain_id"] == 11
    assert round_output.loc[0, "candidate_universe"] == "model_candidate"
    assert round_output.loc[0, "oracle_type"] == "budget_constrained"

    captain_profiles = pd.read_csv(output / "oracle_captain_profiles.csv")
    assert captain_profiles.loc[0, "captain_id"] == 9
    assert captain_profiles.loc[0, "candidate_universe"] == "selected_squad"
    assert captain_profiles.loc[0, "oracle_type"] == "selected_squad_captain"
    assert captain_profiles.loc[0, "budget_path"] == "model_budget_path"
    assert captain_profiles.loc[0, "candidate_universe"] != "model_candidate"


def test_build_oracle_discovery_report_skips_selected_dependent_outputs_when_selected_slice_missing(
    tmp_path: Path,
) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players["rodada"] = 4
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    recall = pd.read_csv(output / "model_vs_oracle_recall.csv")
    assert recall.empty

    captain_profiles = pd.read_csv(output / "oracle_captain_profiles.csv")
    assert captain_profiles.empty

    invalid_rows = pd.read_csv(output / "invalid_oracle_rows.csv")
    assert invalid_rows.loc[0, "rodada"] == 5
    assert invalid_rows.loc[0, "strategy"] == "xgboost_depth2_l2_heavy"
    assert invalid_rows.loc[0, "oracle_type"] == "selected_squad_captain"
    assert invalid_rows.loc[0, "candidate_universe"] == "selected_squad"
    assert invalid_rows.loc[0, "budget_path"] == "model_budget_path"
    assert invalid_rows.loc[0, "invalid_reason"] == "missing_selected_squad"


def test_build_oracle_discovery_report_labels_invalid_selected_squad_captain_rows(
    tmp_path: Path,
) -> None:
    predictions = _report_builder_predictions()
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = False
    round_results = pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    )
    experiment = _write_report_builder_experiment(
        tmp_path,
        predictions=predictions,
        round_results=round_results,
        selected_players=selected_players,
    )
    output = tmp_path / "oracle_out"

    _run_build_oracle_discovery_report(experiment_path=experiment, output_path=output)

    invalid_rows = pd.read_csv(output / "invalid_oracle_rows.csv")
    assert invalid_rows.loc[0, "rodada"] == 5
    assert invalid_rows.loc[0, "strategy"] == "xgboost_depth2_l2_heavy"
    assert invalid_rows.loc[0, "oracle_type"] == "selected_squad_captain"
    assert invalid_rows.loc[0, "candidate_universe"] == "selected_squad"
    assert invalid_rows.loc[0, "budget_path"] == "model_budget_path"
    assert invalid_rows.loc[0, "invalid_reason"] == "Selected squad must contain exactly one model captain, got 0"


def test_build_oracle_discovery_report_uses_parent_current_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _valid_child_dir(tmp_path)
    metadata = json.loads((child / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["season"] = 2021
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    predictions = _report_builder_predictions()
    predictions.to_csv(child / "player_predictions.csv", index=False)
    selected_players = predictions.head(12).copy()
    selected_players["strategy"] = "xgboost_depth2_l2_heavy"
    selected_players["is_captain"] = selected_players["id_atleta"].eq(10)
    selected_players.to_csv(child / "selected_players.csv", index=False)
    pd.DataFrame(
        [
            {
                "rodada": 5,
                "strategy": "xgboost_depth2_l2_heavy",
                "solver_status": "Optimal",
                "budget_before_round": 100.0,
                "budget_after_round": 100.0,
                "budget_delta": 0.0,
                "budget_used": 12.0,
                "actual_points_with_captain": 83.5,
                "captain_id": 10,
            }
        ]
    ).to_csv(child / "round_results.csv", index=False)
    (tmp_path / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "current_year": 2026,
                "child_runs": [
                    {
                        "child_id": "season=2021/model=xgboost_depth2_l2_heavy/feature_pack=ppg_xg_matchup",
                        "output_path": str(child),
                        "season": 2021,
                        "model_id": "xgboost_depth2_l2_heavy",
                        "feature_pack": "ppg_xg_matchup",
                        "fixture_mode": "exploratory",
                        "metadata": {
                            "budget_policy": "moving",
                            "matchup_context_mode": "cartola_matchup_v1",
                        },
                        "strategy_roles": {
                            "xgboost_depth2_l2_heavy": "primary_model",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    seen_current_years: list[int | None] = []

    def fake_optimize_squad(
        candidates: pd.DataFrame,
        *,
        score_column: str,
        config: BacktestConfig,
        budget: float,
    ) -> SquadOptimizationResult:
        del score_column, budget
        seen_current_years.append(config.current_year)
        selected = candidates.head(12).copy()
        selected["is_captain"] = selected["id_atleta"].eq(11)
        return SquadOptimizationResult(
            selected=selected,
            status="Optimal",
            budget_used=12.0,
            predicted_points=83.5,
            predicted_points_base=78.0,
            captain_bonus_predicted=5.5,
            predicted_points_with_captain=83.5,
            formation_name="4-3-3",
            selected_count=12,
            captain_id=11,
            captain_name="P11",
            captain_position="ata",
            captain_club="C11",
            captain_predicted_points=11.0,
            captain_multiplier=1.5,
            scoring_contract_version="cartola_standard_2026_v1",
            formation_scores=[],
            captain_policy_diagnostics=[],
        )

    monkeypatch.setattr(oracle_discovery, "optimize_squad", fake_optimize_squad)

    _run_build_oracle_discovery_report(experiment_path=tmp_path, output_path=tmp_path / "oracle_out")

    assert seen_current_years == [2026]


def test_load_source_run_contexts_rejects_missing_current_year(tmp_path: Path) -> None:
    child = _valid_child_dir(tmp_path)
    _write_valid_experiment_metadata(tmp_path, child)
    metadata = json.loads((tmp_path / "experiment_metadata.json").read_text(encoding="utf-8"))
    metadata.pop("current_year")
    (tmp_path / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="current_year"):
        load_source_run_contexts(tmp_path)
