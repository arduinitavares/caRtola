from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.ridge_tuning_config import (
    PRIMARY_INCUMBENT_CANDIDATE_ID,
    SECONDARY_CONTROL_CANDIDATE_ID,
    candidate_id_for,
)
from cartola.backtesting.ridge_tuning_runner import RidgeTuningProgressEvent, run_ridge_tuning
from cartola.backtesting.runner import BacktestMetadata, BacktestResult


def test_run_ridge_tuning_passes_alpha_params_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[dict[str, object]] = []

    def fake_run_backtest_for_experiment(
        config: object,
        *,
        primary_model_id: str,
        model_params: dict[str, object],
    ) -> BacktestResult:
        observed.append(
            {
                "primary_model_id": primary_model_id,
                "model_params": model_params,
            }
        )
        return _fake_result(config=config, total_actual_points=100.0 + float(model_params["alpha"]))

    monkeypatch.setattr(
        "cartola.backtesting.ridge_tuning_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )

    result = run_ridge_tuning(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=1,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning"),
        started_at_utc="20260501T120000000000Z",
        skip_final_rerun=True,
    )

    assert result.experiment_id.startswith("group=ridge-alpha-tuning__")
    assert (result.output_path / "tuning_generation_manifest.json").exists()
    assert (result.output_path / "ranked_summary.csv").exists()
    assert (result.output_path / "promotion_report.json").exists()
    assert observed
    assert {call["primary_model_id"] for call in observed} == {"ridge"}
    assert {call["model_params"]["alpha"] for call in observed} >= {0.01, 1.0, 300.0}


def test_run_ridge_tuning_skip_final_rerun_does_not_promote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run_backtest_for_experiment(
        config: object,
        *,
        primary_model_id: str,
        model_params: dict[str, object],
    ) -> BacktestResult:
        return _fake_result(config=config, total_actual_points=150.0 + float(model_params["alpha"]))

    monkeypatch.setattr(
        "cartola.backtesting.ridge_tuning_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )

    result = run_ridge_tuning(
        seasons=(2024, 2025),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=1,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning"),
        started_at_utc="20260501T120000000001Z",
        skip_final_rerun=True,
    )

    promotion_report = json.loads((result.output_path / "promotion_report.json").read_text(encoding="utf-8"))

    assert promotion_report["recommendation"] == "keep_incumbent"
    assert promotion_report["reason"] == "final_rerun_skipped"
    assert promotion_report["promoted_candidate_id"] is None


def test_run_ridge_tuning_rejects_current_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Tuning seasons must be before current_year"):
        run_ridge_tuning(
            seasons=(2026,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=1,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/experiments/model_tuning"),
            started_at_utc="20260501T120000000002Z",
        )


def test_run_ridge_tuning_final_stage_runs_incumbents_and_top_challengers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final_candidate_ids: list[str] = []

    def fake_run_backtest_for_experiment(
        config: object,
        *,
        primary_model_id: str,
        model_params: dict[str, object],
    ) -> BacktestResult:
        candidate_id = _candidate_id_from_config(config, model_params=model_params)
        if "/stage=final/" in str(getattr(config, "_output_path_override")):
            final_candidate_ids.append(candidate_id)
        total = _total_for_candidate(candidate_id, alpha=float(model_params["alpha"]))
        return _fake_result(config=config, total_actual_points=total)

    monkeypatch.setattr(
        "cartola.backtesting.ridge_tuning_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )

    run_ridge_tuning(
        seasons=(2024, 2025),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=1,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning"),
        started_at_utc="20260501T120000000003Z",
    )

    expected_challengers = [
        candidate_id_for(alpha=300.0, feature_pack="ppg_xg"),
        candidate_id_for(alpha=300.0, feature_pack="ppg"),
    ]
    assert sorted(set(final_candidate_ids)) == sorted(
        {
            PRIMARY_INCUMBENT_CANDIDATE_ID,
            SECONDARY_CONTROL_CANDIDATE_ID,
            *expected_challengers,
        }
    )


def test_run_ridge_tuning_emits_child_progress_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[RidgeTuningProgressEvent] = []

    def fake_run_backtest_for_experiment(
        config: object,
        *,
        primary_model_id: str,
        model_params: dict[str, object],
    ) -> BacktestResult:
        return _fake_result(config=config, total_actual_points=100.0 + float(model_params["alpha"]))

    monkeypatch.setattr(
        "cartola.backtesting.ridge_tuning_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )

    run_ridge_tuning(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=1,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_tuning"),
        started_at_utc="20260501T120000000004Z",
        progress_callback=events.append,
        skip_final_rerun=True,
    )

    event_types = [event.event_type for event in events]
    assert "child_started" in event_types
    assert "child_finished" in event_types
    assert all(event.child_id for event in events if event.event_type in {"child_started", "child_finished"})


def _fake_result(*, config: object, total_actual_points: float) -> BacktestResult:
    season = int(getattr(config, "season"))
    footystats_mode = str(getattr(config, "footystats_mode"))
    round_results = pd.DataFrame(
        [
            _round_result_row(strategy="ridge", total_actual_points=total_actual_points),
            _round_result_row(strategy="baseline", total_actual_points=80.0),
            _round_result_row(strategy="price", total_actual_points=70.0),
        ]
    )
    selected_players = pd.DataFrame(
        [
            _selected_player_row(1, predicted_points=10.0, actual_points=11.0),
            _selected_player_row(2, predicted_points=8.0, actual_points=9.0),
        ]
    )
    player_predictions = pd.DataFrame(
        [
            _prediction_row(1, predicted_points=10.0, actual_points=11.0),
            _prediction_row(2, predicted_points=8.0, actual_points=9.0),
            _prediction_row(3, predicted_points=6.0, actual_points=7.0),
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": "ridge",
                "rounds": 51,
                "total_actual_points": total_actual_points,
                "average_actual_points": total_actual_points / 51.0,
                "total_predicted_points": total_actual_points,
                "average_predicted_points": total_actual_points / 51.0,
            }
        ]
    )
    metadata = BacktestMetadata(
        season=season,
        start_round=5,
        max_round=5,
        cache_enabled=True,
        prediction_frames_built=1,
        wall_clock_seconds=1.0,
        backtest_jobs=1,
        backtest_workers_effective=1,
        model_n_jobs_effective=-1,
        parallel_backend="sequential",
        thread_env={},
        scoring_contract_version="cartola_standard_2026_v1",
        captain_scoring_enabled=True,
        captain_multiplier=1.5,
        formation_search="all_official_formations",
        fixture_mode="none",
        strict_alignment_policy="fail",
        matchup_context_mode="none",
        matchup_context_feature_columns=[],
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[],
        footystats_mode=footystats_mode,
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug="brazil-serie-a",
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )
    return BacktestResult(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=pd.DataFrame(),
        metadata=metadata,
    )


def _round_result_row(*, strategy: str, total_actual_points: float) -> dict[str, object]:
    return {
        "rodada": 5,
        "strategy": strategy,
        "solver_status": "Optimal",
        "formation": "4-3-3",
        "selected_count": 12,
        "budget_used": 100.0,
        "predicted_points": total_actual_points,
        "predicted_points_base": total_actual_points,
        "captain_bonus_predicted": 0.0,
        "predicted_points_with_captain": total_actual_points,
        "actual_points": total_actual_points,
        "actual_points_base": total_actual_points,
        "captain_bonus_actual": 0.0,
        "actual_points_with_captain": total_actual_points,
        "captain_id": 1,
        "captain_name": "A",
        "captain_policy_ev_id": 1,
        "captain_policy_safe_id": 1,
        "captain_policy_upside_id": 1,
        "actual_points_with_ev_captain": total_actual_points,
        "actual_points_with_safe_captain": total_actual_points,
        "actual_points_with_upside_captain": total_actual_points,
    }


def _selected_player_row(player_id: int, *, predicted_points: float, actual_points: float) -> dict[str, object]:
    return {
        "rodada": 5,
        "strategy": "ridge",
        "id_atleta": player_id,
        "apelido": f"Player {player_id}",
        "posicao": "ata",
        "id_clube": 1,
        "status": "Provavel",
        "preco_pre_rodada": 10.0 + player_id,
        "predicted_points": predicted_points,
        "pontuacao": actual_points,
        "is_captain": player_id == 1,
    }


def _prediction_row(player_id: int, *, predicted_points: float, actual_points: float) -> dict[str, object]:
    return {
        "rodada": 5,
        "id_atleta": player_id,
        "apelido": f"Player {player_id}",
        "posicao": "ata",
        "id_clube": 1,
        "status": "Provavel",
        "preco_pre_rodada": 10.0 + player_id,
        "ridge_score": predicted_points,
        "pontuacao": actual_points,
    }


def _candidate_id_from_config(config: object, *, model_params: dict[str, object]) -> str:
    output_path = getattr(config, "_output_path_override")
    candidate_part = Path(str(output_path)).name
    if candidate_part.startswith("candidate="):
        return candidate_part.removeprefix("candidate=")
    return candidate_id_for(alpha=float(model_params["alpha"]), feature_pack=_feature_pack_from_config(config))


def _feature_pack_from_config(config: object) -> str:
    return "ppg_xg" if getattr(config, "footystats_mode") == "ppg_xg" else "ppg"


def _total_for_candidate(candidate_id: str, *, alpha: float) -> float:
    feature_bonus = 100.0 if candidate_id.endswith("__ppg_xg") else 0.0
    return 1_000.0 + feature_bonus + alpha
