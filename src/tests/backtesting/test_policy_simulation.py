from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY, get_policy_set
from cartola.backtesting.policy_simulation import (
    POLICY_PER_SEASON_SUMMARY_COLUMNS,
    POLICY_PROFILE_SUMMARY_COLUMNS,
    POLICY_RANKED_SUMMARY_COLUMNS,
    POLICY_ROUND_RESULT_COLUMNS,
    POLICY_SELECTED_PLAYER_COLUMNS,
    PolicyReplayResult,
    PolicySimulationError,
    build_policy_per_season_summary,
    build_policy_profile_summary,
    build_policy_ranked_summary,
    decide_policy_variant,
    load_policy_source_context,
    reproduce_no_policy_round,
    run_policy_replay_for_child,
    write_policy_simulation_report,
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


def test_policy_replay_tracks_independent_moving_budget_paths(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    no_policy, soft_overlap_penalty_low = get_policy_set("opponent-overlap-v1").policies[:2]

    result = run_policy_replay_for_child(
        child_path=child,
        policies=(no_policy, soft_overlap_penalty_low),
    )

    _assert_no_policy_replay_matches_source(child, result, expected_rounds=(5, 6))

    round_rows = {
        (str(row["policy_variant"]), cast(int, row["rodada"])): row
        for row in result.round_rows
    }
    assert round_rows[("no_policy", 5)]["budget_after_round"] == pytest.approx(105.0)
    assert round_rows[("soft_overlap_penalty_low", 5)]["budget_after_round"] == pytest.approx(97.0)
    assert round_rows[("no_policy", 6)]["budget_before_round"] == pytest.approx(105.0)
    assert round_rows[("soft_overlap_penalty_low", 6)]["budget_before_round"] == pytest.approx(97.0)

    selected = pd.DataFrame(result.selected_player_rows)
    round_one_selected_ids = {
        policy_variant: set(
            selected.loc[
                selected["policy_variant"].eq(policy_variant) & selected["rodada"].eq(5),
                "id_atleta",
            ].astype(int)
        )
        for policy_variant in ("no_policy", "soft_overlap_penalty_low")
    }
    assert 20 in round_one_selected_ids["no_policy"]
    assert 21 in round_one_selected_ids["soft_overlap_penalty_low"]
    assert round_one_selected_ids["no_policy"] != round_one_selected_ids["soft_overlap_penalty_low"]
    assert result.invalid_rows == []


def test_policy_replay_allows_no_policy_without_fixture_artifacts(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    (child / "fixtures_for_round.csv").unlink()

    result = run_policy_replay_for_child(child_path=child, policies=(NO_POLICY,))

    _assert_no_policy_replay_matches_source(child, result, expected_rounds=(5, 6))


def test_policy_replay_loads_fixture_source_directory_when_child_artifact_missing(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    policy = get_policy_set("opponent-overlap-v1").policies[1]
    fixtures = pd.read_csv(child / "fixtures_for_round.csv")
    (child / "fixtures_for_round.csv").unlink()
    fixture_source_directory = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_source_directory.mkdir(parents=True)
    for round_number, round_fixtures in fixtures.groupby("rodada", sort=True):
        round_fixtures.to_csv(
            fixture_source_directory / f"partidas-{int(cast(int, round_number))}.csv",
            index=False,
        )

    metadata = json.loads((child / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["fixture_source_directory"] = str(fixture_source_directory)
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = run_policy_replay_for_child(child_path=child, policies=(policy,))

    assert {row["policy_variant"] for row in result.round_rows} == {policy.policy_variant}
    assert {cast(int, row["rodada"]) for row in result.round_rows} == {5, 6}
    assert result.invalid_rows == []


def test_policy_replay_output_schemas_match_policy_contract(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)

    result = run_policy_replay_for_child(child_path=child, policies=(NO_POLICY,))

    assert list(pd.DataFrame(result.round_rows).columns) == list(POLICY_ROUND_RESULT_COLUMNS)
    assert list(pd.DataFrame(result.selected_player_rows).columns) == list(POLICY_SELECTED_PLAYER_COLUMNS)


@pytest.mark.parametrize("policy_index", [1, 3])
def test_policy_replay_requires_fixture_file_for_fixture_dependent_policy(
    tmp_path: Path,
    policy_index: int,
) -> None:
    child = _write_two_round_policy_child(tmp_path)
    policy = get_policy_set("opponent-overlap-v1").policies[policy_index]
    (child / "fixtures_for_round.csv").unlink()

    with pytest.raises(
        PolicySimulationError,
        match=rf"fixture coverage.*policy_variant={policy.policy_variant!r}.*round=5",
    ):
        run_policy_replay_for_child(child_path=child, policies=(policy,))


@pytest.mark.parametrize("policy_index", [1, 3])
def test_policy_replay_requires_fixture_rows_for_target_round(
    tmp_path: Path,
    policy_index: int,
) -> None:
    child = _write_two_round_policy_child(tmp_path)
    policy = get_policy_set("opponent-overlap-v1").policies[policy_index]
    fixtures = pd.read_csv(child / "fixtures_for_round.csv")
    fixtures["rodada"] = 4
    fixtures.to_csv(child / "fixtures_for_round.csv", index=False)

    with pytest.raises(
        PolicySimulationError,
        match=rf"fixture coverage.*policy_variant={policy.policy_variant!r}.*round=5",
    ):
        run_policy_replay_for_child(child_path=child, policies=(policy,))


def test_policy_replay_requires_fixture_coverage_for_all_candidate_clubs(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    policy = get_policy_set("opponent-overlap-v1").policies[1]
    fixtures = pd.read_csv(child / "fixtures_for_round.csv")
    fixtures = fixtures.loc[fixtures["id_clube_home"].ne(1001) & fixtures["id_clube_away"].ne(1001)]
    fixtures.to_csv(child / "fixtures_for_round.csv", index=False)

    with pytest.raises(
        PolicySimulationError,
        match=rf"fixture coverage.*policy_variant={policy.policy_variant!r}.*round=5",
    ):
        run_policy_replay_for_child(child_path=child, policies=(policy,))


def test_policy_replay_requires_selected_finite_variacao(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    predictions_path = child / "player_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[predictions["id_atleta"].eq(20) & predictions["rodada"].eq(5), "variacao"] = float("nan")
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(PolicySimulationError, match="variacao"):
        run_policy_replay_for_child(child_path=child, policies=(NO_POLICY,))


def test_policy_replay_scores_explicit_dnp_null_pontuacao_as_zero(tmp_path: Path) -> None:
    child = _write_two_round_policy_child(tmp_path)
    predictions_path = child / "player_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[predictions["id_atleta"].eq(20) & predictions["rodada"].eq(5), "entrou_em_campo"] = False
    predictions.loc[predictions["id_atleta"].eq(20) & predictions["rodada"].eq(5), "pontuacao"] = float("nan")
    predictions.to_csv(predictions_path, index=False)

    result = run_policy_replay_for_child(child_path=child, policies=(NO_POLICY,))

    round_one = next(row for row in result.round_rows if cast(int, row["rodada"]) == 5)
    selected = pd.DataFrame(result.selected_player_rows)
    selected_round_one = selected.loc[selected["rodada"].eq(5)].copy()
    captain_score = float(selected_round_one.loc[selected_round_one["is_captain"].eq(True), "pontuacao"].iloc[0])
    expected_actual = float(selected_round_one["pontuacao"].sum()) + 0.5 * captain_score
    dnp_row = selected_round_one.loc[selected_round_one["id_atleta"].eq(20)].iloc[0]

    assert dnp_row["pontuacao"] == pytest.approx(0.0)
    assert round_one["actual_points_with_captain"] == pytest.approx(expected_actual)


def test_policy_decision_marks_non_h001_generation_diagnostic_only() -> None:
    decision = decide_policy_variant(
        selected_seasons=(2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=50.0,
        improved_seasons=3,
        season_2025_delta=5.0,
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
    )

    assert decision.status == "diagnostic_only"
    assert "2021-2025" in decision.reason


def test_policy_decision_rejects_2025_regression() -> None:
    decision = decide_policy_variant(
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
        total_delta=50.0,
        improved_seasons=4,
        season_2025_delta=-25.1,
        non_optimal_delta=0,
        final_budget_delta=0.0,
        min_budget_delta=0.0,
        max_drawdown_delta=0.0,
        top_two_concentration=0.25,
    )

    assert decision.status == "rejected"
    assert "2025" in decision.reason


def test_policy_ranked_summary_marks_policy_ineligible_when_benchmark_season_missing() -> None:
    round_results = _policy_summary_round_results()
    round_results = round_results.loc[
        ~(
            round_results["season"].eq(2024)
            & round_results["policy_variant"].eq("no_policy")
        )
    ].reset_index(drop=True)

    ranked_summary = build_policy_ranked_summary(
        round_results,
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
    )
    policy_row = ranked_summary.loc[
        ranked_summary["policy_variant"].eq("soft_overlap_penalty_low")
    ].iloc[0]

    assert policy_row["decision_status"] == "ineligible"
    assert "no_policy" in str(policy_row["decision_reason"])
    assert "benchmark" in str(policy_row["decision_reason"])
    assert pd.isna(policy_row["benchmark_total_actual_points"])
    assert pd.isna(policy_row["total_delta"])


def test_policy_ranked_summary_marks_policy_ineligible_when_selected_season_missing() -> None:
    round_results = _policy_summary_round_results()
    round_results = round_results.loc[~round_results["season"].eq(2024)].reset_index(drop=True)

    ranked_summary = build_policy_ranked_summary(
        round_results,
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
    )
    policy_row = ranked_summary.loc[
        ranked_summary["policy_variant"].eq("soft_overlap_penalty_low")
    ].iloc[0]

    assert policy_row["decision_status"] == "ineligible"
    assert "missing selected season" in str(policy_row["decision_reason"])
    assert "evidence" in str(policy_row["decision_reason"])
    assert ranked_summary["decision_status"].ne("candidate_policy").all()


def test_policy_ranked_summary_uses_worst_case_final_budget_delta_for_budget_guardrail() -> None:
    round_results = _policy_summary_round_results()
    for season in (2021, 2022, 2023, 2024, 2025):
        if season == 2023:
            _set_summary_budget_path(
                round_results,
                season=season,
                policy_variant="no_policy",
                budgets=(100.0, 110.0, 120.0),
            )
            _set_summary_budget_path(
                round_results,
                season=season,
                policy_variant="soft_overlap_penalty_low",
                budgets=(100.0, 105.0, 110.0),
            )
        else:
            _set_summary_budget_path(
                round_results,
                season=season,
                policy_variant="no_policy",
                budgets=(100.0, 101.0, 102.0),
            )
            _set_summary_budget_path(
                round_results,
                season=season,
                policy_variant="soft_overlap_penalty_low",
                budgets=(100.0, 106.0, 112.0),
            )

    ranked_summary = build_policy_ranked_summary(
        round_results,
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
    )
    policy_row = ranked_summary.loc[
        ranked_summary["policy_variant"].eq("soft_overlap_penalty_low")
    ].iloc[0]

    assert policy_row["final_budget"] == pytest.approx(110.0)
    assert policy_row["benchmark_final_budget"] == pytest.approx(120.0)
    assert policy_row["final_budget_delta"] == pytest.approx(-10.0)
    assert policy_row["decision_status"] == "rejected"
    assert "budget path" in str(policy_row["decision_reason"])


def test_policy_summary_output_schemas_are_stable() -> None:
    round_results = _policy_summary_round_results()
    selected_players = _policy_summary_selected_players()

    ranked_summary = build_policy_ranked_summary(
        round_results,
        selected_seasons=(2021, 2022, 2023, 2024, 2025),
        fixture_identity_status="verified",
    )
    per_season_summary = build_policy_per_season_summary(round_results)
    profile_summary = build_policy_profile_summary(round_results, selected_players)

    assert POLICY_RANKED_SUMMARY_COLUMNS == (
        "rank",
        "model_id",
        "feature_pack",
        "strategy",
        "policy_variant",
        "selected_seasons",
        "fixture_identity_status",
        "rounds",
        "total_actual_points",
        "benchmark_total_actual_points",
        "total_delta",
        "improved_seasons",
        "season_2025_delta",
        "top_two_positive_delta_concentration",
        "final_budget",
        "benchmark_final_budget",
        "final_budget_delta",
        "min_budget",
        "benchmark_min_budget",
        "min_budget_delta",
        "max_budget_drawdown",
        "benchmark_max_budget_drawdown",
        "max_drawdown_delta",
        "non_optimal_rounds",
        "benchmark_non_optimal_rounds",
        "non_optimal_delta",
        "decision_status",
        "decision_reason",
    )
    assert POLICY_PER_SEASON_SUMMARY_COLUMNS == (
        "season",
        "model_id",
        "feature_pack",
        "strategy",
        "policy_variant",
        "rounds",
        "total_actual_points",
        "benchmark_total_actual_points",
        "total_delta",
        "final_budget",
        "benchmark_final_budget",
        "final_budget_delta",
        "min_budget",
        "benchmark_min_budget",
        "min_budget_delta",
        "max_budget_drawdown",
        "benchmark_max_budget_drawdown",
        "max_drawdown_delta",
        "non_optimal_rounds",
        "benchmark_non_optimal_rounds",
        "non_optimal_delta",
    )
    assert POLICY_ROUND_RESULT_COLUMNS == (
        "season",
        "model_id",
        "feature_pack",
        "strategy",
        "policy_variant",
        "rodada",
        "solver_status",
        "formation",
        "captain_id",
        "budget_before_round",
        "budget_used",
        "budget_remaining",
        "budget_delta",
        "budget_after_round",
        "predicted_points_with_captain",
        "actual_points_with_captain",
    )
    assert POLICY_SELECTED_PLAYER_COLUMNS == (
        "season",
        "model_id",
        "feature_pack",
        "strategy",
        "policy_variant",
        "rodada",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "preco_pre_rodada",
        "pontuacao",
        "entrou_em_campo",
        "variacao",
        "is_captain",
    )
    assert POLICY_PROFILE_SUMMARY_COLUMNS == (
        "season",
        "model_id",
        "feature_pack",
        "strategy",
        "policy_variant",
        "id_atleta",
        "apelido",
        "posicao",
        "id_clube",
        "nome_clube",
        "selected_rounds",
        "captain_rounds",
        "total_pontuacao",
        "average_pontuacao",
        "total_variacao",
        "average_preco_pre_rodada",
        "first_round",
        "last_round",
    )
    assert list(ranked_summary.columns) == list(POLICY_RANKED_SUMMARY_COLUMNS)
    assert list(per_season_summary.columns) == list(POLICY_PER_SEASON_SUMMARY_COLUMNS)
    assert list(round_results.columns) == list(POLICY_ROUND_RESULT_COLUMNS)
    assert list(selected_players.columns) == list(POLICY_SELECTED_PLAYER_COLUMNS)
    assert list(profile_summary.columns) == list(POLICY_PROFILE_SUMMARY_COLUMNS)


def test_policy_report_contains_required_literals_and_table_rows(tmp_path: Path) -> None:
    round_results = _policy_summary_round_results()
    selected_players = _policy_summary_selected_players()
    ranked_summary = build_policy_ranked_summary(round_results)
    per_season_summary = build_policy_per_season_summary(round_results)
    profile_summary = build_policy_profile_summary(round_results, selected_players)
    output_dir = tmp_path / "policy_report"

    write_policy_simulation_report(
        output_dir,
        manifest={"experiment_id": "H001-test"},
        ranked_summary=ranked_summary,
        per_season_summary=per_season_summary,
        profile_summary=profile_summary,
        comparability_report={"fixture_identity_status": "unverified"},
    )

    html = (output_dir / "policy_simulation_report.html").read_text(encoding="utf-8")
    assert "Policy Simulation V1" in html
    assert "H001" in html
    assert "research evidence only" in html
    assert "diagnostic_only" in html
    assert "<table" in html
    assert "soft_overlap_penalty_low" in html


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


def _write_two_round_policy_child(tmp_path: Path, *, score_column: str = "test_model_score") -> Path:
    child = tmp_path / "two_round_child"
    child.mkdir()
    model_id = _model_id_from_score_column(score_column)
    metadata: dict[str, object] = {
        "season": 2025,
        "model_id": model_id,
        "feature_pack": "synthetic_pack",
        "budget_policy": "moving",
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "fixture_mode": "exploratory",
        "matchup_context_mode": "cartola_matchup_v1",
        "start_round": 5,
        "initial_budget": 100.0,
    }
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    round_five = _synthetic_candidates(score_column=score_column)
    round_five["rodada"] = 5
    round_five.loc[round_five["id_atleta"].eq(16), "id_clube"] = 1002
    round_five.loc[round_five["id_atleta"].eq(20), ["id_clube", score_column, "pontuacao", "variacao"]] = [
        1001,
        5.0,
        2.5,
        5.0,
    ]
    round_five.loc[round_five["id_atleta"].eq(21), ["id_clube", score_column, "pontuacao", "variacao"]] = [
        1003,
        4.8,
        2.4,
        -3.0,
    ]

    round_six = _synthetic_candidates(score_column=score_column)
    round_six["rodada"] = 6
    round_six.loc[:, "variacao"] = 0.0
    round_six.loc[round_six["id_atleta"].eq(20), score_column] = 5.0
    round_six.loc[round_six["id_atleta"].eq(21), score_column] = 4.8

    candidates = pd.concat([round_five, round_six], ignore_index=True)
    candidates.to_csv(child / "player_predictions.csv", index=False)
    _write_matching_source_outputs(child, candidates=candidates, score_column=score_column, model_id=model_id)
    _complete_fixture_rows(candidates).to_csv(child / "fixtures_for_round.csv", index=False)
    return child


def _write_matching_source_outputs(
    child: Path,
    *,
    candidates: pd.DataFrame,
    score_column: str,
    model_id: str,
) -> None:
    round_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    current_budget = 100.0
    for round_number in sorted(candidates["rodada"].astype(int).unique().tolist()):
        round_candidates = candidates.loc[candidates["rodada"].eq(round_number)].copy()
        result = optimize_squad(
            round_candidates,
            score_column=score_column,
            config=BacktestConfig(season=2025, start_round=round_number, budget=current_budget),
            budget=current_budget,
            policy=NO_POLICY,
        )
        actual_scores = actual_scores_with_captain(result.selected, actual_column="pontuacao")
        budget_delta = float(result.selected["variacao"].sum())
        budget_after_round = current_budget + budget_delta
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": model_id,
                "solver_status": result.status,
                "formation": result.formation_name,
                "budget_before_round": current_budget,
                "budget_after_round": budget_after_round,
                "budget_delta": budget_delta,
                "budget_used": result.budget_used,
                "actual_points_with_captain": actual_scores["actual_points_with_captain"],
                "predicted_points_with_captain": result.predicted_points_with_captain,
                "captain_id": result.captain_id,
            }
        )
        selected_players = result.selected.copy()
        selected_players["rodada"] = round_number
        selected_players["strategy"] = model_id
        selected_rows.append(selected_players)
        current_budget = budget_after_round

    pd.DataFrame(round_rows).to_csv(child / "round_results.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(child / "selected_players.csv", index=False)


def _complete_fixture_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for round_number in sorted(candidates["rodada"].astype(int).unique().tolist()):
        round_candidates = candidates.loc[candidates["rodada"].eq(round_number)]
        club_ids = sorted(round_candidates["id_clube"].astype(int).unique().tolist())
        if 1001 in club_ids and 1002 in club_ids:
            rows.append({"rodada": round_number, "id_clube_home": 1001, "id_clube_away": 1002})
            club_ids = [club_id for club_id in club_ids if club_id not in {1001, 1002}]
        for fixture_index in range(0, len(club_ids), 2):
            home_id = club_ids[fixture_index]
            away_id = (
                club_ids[fixture_index + 1]
                if fixture_index + 1 < len(club_ids)
                else 900_000 + round_number
            )
            rows.append({"rodada": round_number, "id_clube_home": home_id, "id_clube_away": away_id})
    return pd.DataFrame(rows)


def _policy_summary_round_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policy_deltas = {
        2021: (4.0, 3.0),
        2022: (3.0, 2.0),
        2023: (2.0, 2.0),
        2024: (1.0, -1.0),
        2025: (5.0, 4.0),
    }
    for season, deltas in policy_deltas.items():
        for policy_variant in ("no_policy", "soft_overlap_penalty_low"):
            current_budget = 100.0
            for round_offset, round_number in enumerate((5, 6)):
                actual_points = 40.0 + round_offset
                budget_delta = 1.0
                if policy_variant != "no_policy":
                    actual_points += deltas[round_offset]
                    budget_delta = 0.5
                budget_after_round = current_budget + budget_delta
                rows.append(
                    {
                        "season": season,
                        "model_id": "test_model",
                        "feature_pack": "synthetic_pack",
                        "strategy": "test_model",
                        "policy_variant": policy_variant,
                        "rodada": round_number,
                        "solver_status": "Optimal",
                        "formation": "4-3-3",
                        "captain_id": 1,
                        "budget_before_round": current_budget,
                        "budget_used": 90.0,
                        "budget_remaining": current_budget - 90.0,
                        "budget_delta": budget_delta,
                        "budget_after_round": budget_after_round,
                        "predicted_points_with_captain": actual_points + 1.0,
                        "actual_points_with_captain": actual_points,
                    }
                )
                current_budget = budget_after_round
    return pd.DataFrame(rows, columns=pd.Index(POLICY_ROUND_RESULT_COLUMNS))


def _set_summary_budget_path(
    round_results: pd.DataFrame,
    *,
    season: int,
    policy_variant: str,
    budgets: tuple[float, float, float],
) -> None:
    mask = round_results["season"].eq(season) & round_results["policy_variant"].eq(policy_variant)
    indexes = round_results.loc[mask].sort_values("rodada", kind="mergesort").index.to_list()
    assert len(indexes) == 2
    budget_used = float(round_results.loc[indexes[0], "budget_used"])
    for offset, index in enumerate(indexes):
        budget_before_round = budgets[offset]
        budget_after_round = budgets[offset + 1]
        round_results.loc[index, "budget_before_round"] = budget_before_round
        round_results.loc[index, "budget_after_round"] = budget_after_round
        round_results.loc[index, "budget_delta"] = budget_after_round - budget_before_round
        round_results.loc[index, "budget_remaining"] = budget_before_round - budget_used


def _policy_summary_selected_players() -> pd.DataFrame:
    rows = [
        {
            "season": 2025,
            "model_id": "test_model",
            "feature_pack": "synthetic_pack",
            "strategy": "test_model",
            "policy_variant": "no_policy",
            "rodada": 5,
            "id_atleta": 1,
            "apelido": "Player 1",
            "posicao": "ata",
            "id_clube": 1001,
            "nome_clube": "Club 1",
            "preco_pre_rodada": 10.0,
            "pontuacao": 8.0,
            "entrou_em_campo": True,
            "variacao": 1.0,
            "is_captain": True,
        },
        {
            "season": 2025,
            "model_id": "test_model",
            "feature_pack": "synthetic_pack",
            "strategy": "test_model",
            "policy_variant": "soft_overlap_penalty_low",
            "rodada": 5,
            "id_atleta": 2,
            "apelido": "Player 2",
            "posicao": "mei",
            "id_clube": 1002,
            "nome_clube": "Club 2",
            "preco_pre_rodada": 9.0,
            "pontuacao": 7.0,
            "entrou_em_campo": True,
            "variacao": 0.5,
            "is_captain": False,
        },
    ]
    return pd.DataFrame(rows, columns=pd.Index(POLICY_SELECTED_PLAYER_COLUMNS))


def _assert_no_policy_replay_matches_source(
    child: Path,
    result: PolicyReplayResult,
    *,
    expected_rounds: tuple[int, ...],
) -> None:
    round_results = pd.read_csv(child / "round_results.csv")
    selected_players = pd.read_csv(child / "selected_players.csv")
    replay_rounds = {
        cast(int, row["rodada"]): row
        for row in result.round_rows
        if str(row["policy_variant"]) == "no_policy"
    }
    replay_selected = pd.DataFrame(result.selected_player_rows)
    for round_number in expected_rounds:
        source_round = round_results.loc[round_results["rodada"].eq(round_number)].iloc[0]
        replay_round = replay_rounds[round_number]
        source_selected = selected_players.loc[selected_players["rodada"].eq(round_number)]
        replay_selected_round = replay_selected.loc[
            replay_selected["policy_variant"].eq("no_policy")
            & replay_selected["rodada"].eq(round_number)
        ]

        assert set(replay_selected_round["id_atleta"].astype(int)) == set(source_selected["id_atleta"].astype(int))
        assert _selected_captain_id(replay_selected_round) == _selected_captain_id(source_selected)
        assert replay_round["budget_before_round"] == pytest.approx(float(source_round["budget_before_round"]))
        assert replay_round["budget_used"] == pytest.approx(float(source_round["budget_used"]))
        assert replay_round["budget_delta"] == pytest.approx(float(source_round["budget_delta"]))
        assert replay_round["budget_after_round"] == pytest.approx(float(source_round["budget_after_round"]))
        assert replay_round["predicted_points_with_captain"] == pytest.approx(
            float(source_round["predicted_points_with_captain"])
        )
        assert replay_round["actual_points_with_captain"] == pytest.approx(
            float(source_round["actual_points_with_captain"])
        )


def _selected_captain_id(selected: pd.DataFrame) -> int:
    captain = selected.loc[selected["is_captain"].astype(bool)]
    assert len(captain) == 1
    return int(captain["id_atleta"].iloc[0])


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
