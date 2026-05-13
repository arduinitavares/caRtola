from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.fixed_blend_diagnostic import (
    FixedBlendDiagnosticError,
    build_blend_complementarity,
    build_blend_ranked_summary,
    decide_blend_candidate,
    load_blend_candidate_frame,
    parse_blend_specs,
    run_blend_replay_for_season,
    run_fixed_blend_diagnostic,
)
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION, actual_scores_with_captain


def test_parse_blend_specs_accepts_named_weighted_components() -> None:
    specs = parse_blend_specs(("blend_a=model_a:0.25,model_b:0.75",))

    assert len(specs) == 1
    assert specs[0].name == "blend_a"
    assert [(component.model_id, component.weight) for component in specs[0].components] == [
        ("model_a", 0.25),
        ("model_b", 0.75),
    ]


@pytest.mark.parametrize(
    "raw_specs",
    [
        (),
        ("blend=model_a:1.0",),
        ("blend=model_a:0.5,model_a:0.5",),
        ("blend=model_a:0.5,model_b:0.4",),
        ("blend=model_a:nan,model_b:1.0",),
        ("blend=model_a:-0.1,model_b:1.1",),
        ("blend=model_a:0.5,model_b:0.5", "blend=model_c:0.5,model_d:0.5"),
    ],
)
def test_parse_blend_specs_rejects_invalid_specs(raw_specs: tuple[str, ...]) -> None:
    with pytest.raises(FixedBlendDiagnosticError):
        parse_blend_specs(raw_specs)


def test_load_blend_candidate_frame_combines_component_scores(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5,), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    _write_predictions(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        frame=base.rename(columns={"model_a_score": "model_b_score"}).assign(model_b_score=base["model_a_score"] + 10.0),
    )
    blend = parse_blend_specs(("blend_a=model_a:0.25,model_b:0.75",))[0]

    candidates = load_blend_candidate_frame(
        experiment_path=experiment_path,
        season=2025,
        feature_pack="ppg",
        blend_spec=blend,
    )

    assert "m006_component_model_a_score" in candidates.columns
    assert "m006_component_model_b_score" in candidates.columns
    assert "m006_blend_blend_a_score" in candidates.columns
    assert candidates["m006_blend_blend_a_score"].iloc[0] == pytest.approx(
        (base["model_a_score"].iloc[0] * 0.25) + ((base["model_a_score"].iloc[0] + 10.0) * 0.75)
    )
    assert candidates[["rodada", "id_atleta", "id_clube", "posicao"]].equals(
        candidates[["rodada", "id_atleta", "id_clube", "posicao"]].sort_values(
            ["rodada", "id_atleta", "id_clube", "posicao"],
            kind="mergesort",
        )
    )


def test_load_blend_candidate_frame_rejects_candidate_identity_mismatch(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5,), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    mismatched = base.rename(columns={"model_a_score": "model_b_score"}).copy()
    mismatched.loc[mismatched.index[0], "id_atleta"] = 999
    _write_predictions(experiment_path, season=2025, model_id="model_b", feature_pack="ppg", frame=mismatched)
    blend = parse_blend_specs(("blend_a=model_a:0.5,model_b:0.5",))[0]

    with pytest.raises(FixedBlendDiagnosticError, match="candidate identity"):
        load_blend_candidate_frame(
            experiment_path=experiment_path,
            season=2025,
            feature_pack="ppg",
            blend_spec=blend,
        )


def test_run_blend_replay_for_season_advances_each_blend_budget_independently(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5, 6), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    _write_predictions(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        frame=base.rename(columns={"model_a_score": "model_b_score"}).assign(model_b_score=base["model_a_score"] + 1.0),
    )
    blends = parse_blend_specs(
        (
            "blend_a=model_a:1.0,model_b:0.0",
            "blend_b=model_a:0.0,model_b:1.0",
        )
    )

    replay = run_blend_replay_for_season(
        experiment_path=experiment_path,
        season=2025,
        feature_pack="ppg",
        blend_specs=blends,
        config=BacktestConfig(season=2025, start_round=5, budget=50.0, project_root=tmp_path),
    )

    assert replay.invalid_rows.empty
    assert set(replay.round_rows["blend_name"]) == {"blend_a", "blend_b"}
    assert replay.round_rows["solver_status"].tolist() == ["Optimal", "Optimal", "Optimal", "Optimal"]
    first_rounds = replay.round_rows.loc[replay.round_rows["rodada"].eq(5)]
    second_rounds = replay.round_rows.loc[replay.round_rows["rodada"].eq(6)]
    assert first_rounds["budget_before_round"].tolist() == [50.0, 50.0]
    assert first_rounds["budget_after_round"].tolist() == [62.0, 62.0]
    assert second_rounds["budget_before_round"].tolist() == [62.0, 62.0]
    assert set(replay.selected_player_rows["blend_name"]) == {"blend_a", "blend_b"}
    assert set(replay.selected_player_rows["predicted_points"]) == set(
        replay.selected_player_rows["m006_blend_blend_a_score"].dropna()
    ) | set(replay.selected_player_rows["m006_blend_blend_b_score"].dropna())


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    [
        ({"source_valid": False}, "invalid"),
        ({"non_optimal_delta": 1}, "rejected"),
        ({"disaster_rounds_under45_delta": 2}, "rejected"),
        ({}, "candidate_blend"),
        ({"aggregate_delta": 50.0, "worst_season_delta": -30.0, "season_2025_delta": -20.0}, "weak_positive_research_lead"),
        ({"aggregate_delta": 10.0}, "inconclusive"),
        ({"aggregate_delta": -25.0}, "rejected"),
    ],
)
def test_decide_blend_candidate_applies_frozen_gates(
    overrides: dict[str, object],
    expected_status: str,
) -> None:
    metrics: dict[str, object] = {
        "blend_name": "blend_a",
        "source_valid": True,
        "aggregate_delta": 90.0,
        "improved_seasons": 3,
        "worst_season_delta": -20.0,
        "season_2025_delta": -10.0,
        "final_budget_delta": -5.0,
        "min_budget_delta": -5.0,
        "max_drawdown_delta": 5.0,
        "selected_calibration_slope": 1.0,
        "top50_spearman_delta": 0.0,
        "disaster_rounds_under45_delta": 0,
        "worst_2_round_delta": -5.0,
        "top_two_concentration": 0.4,
        "non_optimal_delta": 0,
    }
    metrics.update(overrides)

    decision = decide_blend_candidate(**metrics)

    assert decision.status == expected_status
    assert decision.blend_name == "blend_a"
    assert decision.reason


def test_build_blend_complementarity_joins_candidate_rows(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5,), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    _write_predictions(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        frame=base.rename(columns={"model_a_score": "model_b_score"}).assign(
            model_b_score=base["model_a_score"] + 1.0,
        ),
    )

    complementarity = build_blend_complementarity(
        experiment_path=experiment_path,
        seasons=(2025,),
        feature_pack="ppg",
        model_a="model_a",
        model_b="model_b",
    )

    assert complementarity["scope"].tolist() == ["season", "overall"]
    assert complementarity["season"].astype(str).tolist() == ["2025", "all"]
    assert complementarity["prediction_correlation"].notna().all()
    assert complementarity["mean_abs_pred_diff"].iloc[0] == pytest.approx(1.0)


def test_ranked_summary_uses_top_two_positive_round_delta_concentration() -> None:
    ranked = build_blend_ranked_summary(
        _per_season_summary_rows(
            [
                {"season": 2025, "actual_points_delta": 20.0},
            ]
        ),
        _selected_players_for_ranked_summary(),
        source_valid=True,
        round_delta_summary=pd.DataFrame(
            [
                {"blend_name": "blend_a", "season": 2025, "rodada": 5, "actual_points_delta": 10.0},
                {"blend_name": "blend_a", "season": 2025, "rodada": 6, "actual_points_delta": 5.0},
                {"blend_name": "blend_a", "season": 2025, "rodada": 7, "actual_points_delta": 5.0},
                {"blend_name": "blend_a", "season": 2025, "rodada": 8, "actual_points_delta": -20.0},
            ]
        ),
    )

    assert ranked.loc[0, "top_two_concentration"] == pytest.approx(0.75)


def test_ranked_summary_preserves_worst_two_round_downside_by_season() -> None:
    ranked = build_blend_ranked_summary(
        _per_season_summary_rows(
            [
                {"season": 2024, "worst_2_round_delta": -30.0},
                {"season": 2025, "worst_2_round_delta": 25.0},
            ]
        ),
        _selected_players_for_ranked_summary(),
        source_valid=True,
        round_delta_summary=_round_delta_rows(),
    )

    assert ranked.loc[0, "worst_2_round_delta"] == pytest.approx(-30.0)


def test_ranked_summary_uses_downside_preserving_budget_risk_columns() -> None:
    ranked = build_blend_ranked_summary(
        _per_season_summary_rows(
            [
                {
                    "season": 2024,
                    "final_budget_delta": -30.0,
                    "min_budget_delta": -40.0,
                    "max_drawdown_delta": 25.0,
                },
                {
                    "season": 2025,
                    "final_budget_delta": 20.0,
                    "min_budget_delta": 15.0,
                    "max_drawdown_delta": -5.0,
                },
            ]
        ),
        _selected_players_for_ranked_summary(),
        source_valid=True,
        round_delta_summary=_round_delta_rows(),
    )

    assert ranked.loc[0, "final_budget_delta"] == pytest.approx(-30.0)
    assert ranked.loc[0, "min_budget_delta"] == pytest.approx(-40.0)
    assert ranked.loc[0, "max_drawdown_delta"] == pytest.approx(25.0)


def test_run_fixed_blend_diagnostic_writes_artifacts_from_synthetic_source(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    output_root = tmp_path / "blend_diagnostics"
    model_a = _candidate_frame(rounds=(5, 6), score_column="model_a_score", score_offset=0.0)
    model_b = model_a.rename(columns={"model_a_score": "model_b_score"}).assign(
        model_b_score=model_a["model_a_score"] + 0.5,
    )
    _write_experiment_child(
        experiment_path,
        season=2025,
        model_id="model_a",
        feature_pack="ppg",
        predictions=model_a,
        score_column="model_a_score",
    )
    _write_experiment_child(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        predictions=model_b,
        score_column="model_b_score",
    )

    output_path = run_fixed_blend_diagnostic(
        experiment_path=experiment_path,
        seasons=(2025,),
        feature_pack="ppg",
        control_model="model_a",
        blend_specs=parse_blend_specs(("blend_a=model_a:0.5,model_b:0.5",)),
        initial_budget=50.0,
        current_year=2026,
        output_root=output_root,
    )

    assert output_path.parent == output_root
    assert output_path.name.startswith("fixed_blend_started_at=")
    for artifact_name in (
        "fixed_blend_manifest.json",
        "blend_complementarity.csv",
        "blend_round_results.csv",
        "blend_selected_players.csv",
        "blend_per_season_summary.csv",
        "blend_ranked_summary.csv",
        "blend_decision.json",
        "invalid_rows.csv",
        "fixed_blend_report.html",
    ):
        assert (output_path / artifact_name).exists()

    manifest = json.loads((output_path / "fixed_blend_manifest.json").read_text(encoding="utf-8"))
    assert manifest["hypothesis_id"] == "M006"
    assert manifest["design_revision"] == "fixed_blend_v1"
    assert manifest["source_experiment_path"] == str(experiment_path)
    assert manifest["seasons"] == [2025]
    assert manifest["feature_pack"] == "ppg"
    assert manifest["control_model"] == "model_a"
    assert manifest["initial_budget"] == 50.0
    assert manifest["budget_policy"] == "moving"
    assert manifest["current_year"] == 2026
    assert manifest["source_valid"] is True

    ranked_summary = pd.read_csv(output_path / "blend_ranked_summary.csv")
    assert ranked_summary["source_valid"].tolist() == [True]
    assert set(ranked_summary["decision_status"]).issubset(
        {"rejected", "inconclusive", "weak_positive_research_lead", "candidate_blend"}
    )
    assert pd.read_csv(output_path / "invalid_rows.csv").empty


def test_run_fixed_blend_diagnostic_marks_non_moving_source_metadata_invalid(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    output_root = tmp_path / "blend_diagnostics"
    model_a = _candidate_frame(rounds=(5, 6), score_column="model_a_score", score_offset=0.0)
    model_b = model_a.rename(columns={"model_a_score": "model_b_score"}).assign(
        model_b_score=model_a["model_a_score"] + 0.5,
    )
    control_child = _write_experiment_child(
        experiment_path,
        season=2025,
        model_id="model_a",
        feature_pack="ppg",
        predictions=model_a,
        score_column="model_a_score",
    )
    _write_experiment_child(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        predictions=model_b,
        score_column="model_b_score",
    )
    metadata_path = control_child / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["budget_policy"] = "fixed"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    output_path = run_fixed_blend_diagnostic(
        experiment_path=experiment_path,
        seasons=(2025,),
        feature_pack="ppg",
        control_model="model_a",
        blend_specs=parse_blend_specs(("blend_a=model_a:0.5,model_b:0.5",)),
        initial_budget=50.0,
        current_year=2026,
        output_root=output_root,
    )

    manifest = json.loads((output_path / "fixed_blend_manifest.json").read_text(encoding="utf-8"))
    invalid_rows = pd.read_csv(output_path / "invalid_rows.csv")
    ranked_summary = pd.read_csv(output_path / "blend_ranked_summary.csv")

    assert manifest["source_valid"] is False
    assert ranked_summary["source_valid"].tolist() == [False]
    assert ranked_summary["decision_status"].tolist() == ["invalid"]
    assert "budget_policy" in " ".join(invalid_rows["reason"].astype(str))


def _write_predictions(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
    frame: pd.DataFrame,
) -> None:
    child_path = experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"
    child_path.mkdir(parents=True)
    frame.to_csv(child_path / "player_predictions.csv", index=False)


def _write_experiment_child(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
    predictions: pd.DataFrame,
    score_column: str,
) -> Path:
    child_path = experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"
    child_path.mkdir(parents=True)
    metadata = {
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "budget_policy": "moving",
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "fixture_mode": "none",
        "matchup_context_mode": "none",
        "start_round": 5,
        "initial_budget": 50.0,
        "score_column": score_column,
    }
    (child_path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    predictions.to_csv(child_path / "player_predictions.csv", index=False)
    _write_matching_source_outputs(
        child_path,
        season=season,
        predictions=predictions,
        score_column=score_column,
        model_id=model_id,
    )
    return child_path


def _write_matching_source_outputs(
    child_path: Path,
    *,
    season: int,
    predictions: pd.DataFrame,
    score_column: str,
    model_id: str,
) -> None:
    round_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    current_budget = 50.0
    for round_number in sorted(predictions["rodada"].astype(int).unique().tolist()):
        round_candidates = predictions.loc[predictions["rodada"].eq(round_number)].copy()
        result = optimize_squad(
            round_candidates,
            score_column=score_column,
            config=BacktestConfig(season=season, start_round=round_number, budget=current_budget),
            budget=current_budget,
            policy=NO_POLICY,
        )
        actual_scores = actual_scores_with_captain(result.selected, actual_column="pontuacao")
        budget_delta = float(result.selected["variacao"].sum())
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": model_id,
                "solver_status": result.status,
                "formation": result.formation_name,
                "budget_before_round": current_budget,
                "budget_after_round": current_budget + budget_delta,
                "budget_delta": budget_delta,
                "budget_used": result.budget_used,
                "actual_points_with_captain": actual_scores["actual_points_with_captain"],
                "predicted_points_with_captain": result.predicted_points_with_captain,
                "captain_id": result.captain_id,
            }
        )
        selected = result.selected.copy()
        selected["rodada"] = round_number
        selected["strategy"] = model_id
        selected_rows.append(selected)
        current_budget += budget_delta

    pd.DataFrame(round_rows).to_csv(child_path / "round_results.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(child_path / "selected_players.csv", index=False)


def _candidate_frame(*, rounds: tuple[int, ...], score_column: str, score_offset: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    player_id = 1
    position_scores = {
        "gol": [11.0],
        "lat": [10.0, 9.0],
        "zag": [8.0, 7.0, 6.0],
        "mei": [15.0, 14.0, 13.0, 12.0, 11.0],
        "ata": [12.0, 11.0],
        "tec": [5.0],
    }
    for rodada in rounds:
        for posicao, scores in position_scores.items():
            for score in scores:
                rows.append(
                    {
                        "rodada": rodada,
                        "id_atleta": player_id,
                        "id_clube": player_id,
                        "posicao": posicao,
                        "apelido": f"{posicao}-{player_id}",
                        "nome_clube": f"club-{player_id}",
                        "clube": f"club-{player_id}",
                        "preco_pre_rodada": 1.0,
                        "pontuacao": score,
                        "entrou_em_campo": True,
                        "variacao": 1.0,
                        score_column: score + score_offset,
                    }
                )
                player_id += 1
    return pd.DataFrame(rows)


def _selected_players_for_ranked_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "blend_name": ["blend_a", "blend_a", "blend_a"],
            "id_atleta": [1, 1, 1],
            "predicted_points": [8.0, 10.0, 12.0],
            "pontuacao": [7.0, 11.0, 13.0],
        }
    )


def _round_delta_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"blend_name": "blend_a", "season": 2024, "rodada": 5, "actual_points_delta": 5.0},
            {"blend_name": "blend_a", "season": 2025, "rodada": 5, "actual_points_delta": 5.0},
        ]
    )


def _per_season_summary_rows(overrides: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for override in overrides:
        row = {
            "blend_name": "blend_a",
            "season": 2025,
            "control_actual_points": 100.0,
            "blend_actual_points": 120.0,
            "actual_points_delta": 20.0,
            "control_final_budget": 100.0,
            "blend_final_budget": 100.0,
            "final_budget_delta": 0.0,
            "control_min_budget": 90.0,
            "blend_min_budget": 90.0,
            "min_budget_delta": 0.0,
            "control_max_drawdown": 5.0,
            "blend_max_drawdown": 5.0,
            "max_drawdown_delta": 0.0,
            "control_non_optimal_rounds": 0,
            "blend_non_optimal_rounds": 0,
            "non_optimal_delta": 0,
            "control_disaster_rounds_under45": 0,
            "blend_disaster_rounds_under45": 0,
            "disaster_rounds_under45_delta": 0,
            "control_worst_2_round_total": 80.0,
            "blend_worst_2_round_total": 80.0,
            "worst_2_round_delta": 0.0,
        }
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows)
