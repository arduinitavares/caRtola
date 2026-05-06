import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from cartola.backtesting.budgeting import BUDGET_POLICY_FIXED, BUDGET_POLICY_MOVING
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_config import ExperimentGroup, build_child_run_specs, config_hash, experiment_id
from cartola.backtesting.experiment_index import ExperimentIndex
from cartola.backtesting.experiment_runner import ExperimentProgressEvent, _rank_summary, run_model_experiment
from cartola.backtesting.experiment_signatures import ComparabilityError
from cartola.backtesting.experiment_tracking import InMemoryExperimentTracker, TrackerStatus, TrackerWarning
from cartola.backtesting.runner import BacktestMetadata, BacktestResult
from cartola.backtesting.scoring_contract import contract_fields


def _metadata(config: BacktestConfig, *, model_n_jobs_effective: int = 7) -> BacktestMetadata:
    contract = contract_fields()
    return BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=5,
        cache_enabled=True,
        prediction_frames_built=1,
        wall_clock_seconds=0.5,
        backtest_jobs=config.jobs,
        backtest_workers_effective=1,
        model_n_jobs_effective=model_n_jobs_effective,
        parallel_backend="sequential",
        budget_policy=BUDGET_POLICY_MOVING,
        initial_budget=config.budget,
        thread_env={
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "BLIS_NUM_THREADS": None,
        },
        scoring_contract_version=str(contract["scoring_contract_version"]),
        captain_scoring_enabled=bool(contract["captain_scoring_enabled"]),
        captain_multiplier=cast(float, contract["captain_multiplier"]),
        formation_search=str(contract["formation_search"]),
        fixture_mode=config.fixture_mode,
        strict_alignment_policy=config.strict_alignment_policy,
        matchup_context_mode=config.matchup_context_mode,
        matchup_context_feature_columns=[],
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[],
        footystats_mode=config.footystats_mode,
        footystats_evaluation_scope=config.footystats_evaluation_scope,
        footystats_league_slug=config.footystats_league_slug,
        footystats_matches_source_path=None,
        footystats_matches_source_sha256=None,
        footystats_feature_columns=[],
        footystats_missing_join_keys_by_round={},
        footystats_duplicate_join_keys_by_round={},
        footystats_extra_club_rows_by_round={},
    )


def _result(
    config: BacktestConfig,
    *,
    model_id: str,
    candidate_id: int = 101,
    candidate_price: float = 10.0,
    candidate_count: int = 1,
    total_actual_points: float = 2.0,
    total_predicted_points: float | None = None,
) -> BacktestResult:
    if total_predicted_points is None:
        total_predicted_points = total_actual_points
    candidate_rows = [
        {
            "rodada": 5,
            "id_atleta": candidate_id + index,
            "posicao": "ata",
            "id_clube": 1,
            "status": "Provavel",
            "preco_pre_rodada": candidate_price + (index / 100),
            "pontuacao": float(index + 1),
            f"{model_id}_score": float(index + 1),
        }
        for index in range(candidate_count)
    ]
    selected_rows = [
        {
            **row,
            "strategy": model_id,
            "predicted_points": row[f"{model_id}_score"],
        }
        for row in candidate_rows[: min(candidate_count, 12)]
    ]
    round_results = pd.DataFrame(
        [
            {"rodada": 5, "strategy": "baseline", "solver_status": "Optimal", "actual_points": 1.0, "predicted_points": 1.0},
            {
                "rodada": 5,
                "strategy": model_id,
                "solver_status": "Optimal",
                "actual_points": total_actual_points,
                "predicted_points": total_predicted_points,
            },
            {"rodada": 5, "strategy": "price", "solver_status": "Optimal", "actual_points": 1.5, "predicted_points": 1.5},
        ]
    )
    player_predictions = pd.DataFrame(candidate_rows)
    summary = pd.DataFrame(
        [
            {
                "strategy": "baseline",
                "rounds": 1,
                "total_actual_points": 1.0,
                "average_actual_points": 1.0,
            "total_predicted_points": 1.0,
            "initial_budget": config.budget,
            "final_budget": config.budget,
            "total_budget_delta": 0.0,
            "min_budget": config.budget,
            "max_budget_drawdown": 0.0,
            "budget_constrained_rounds": 0,
            "actual_points_delta_vs_price": -0.5,
        },
            {
                "strategy": model_id,
                "rounds": 1,
                "total_actual_points": total_actual_points,
                "average_actual_points": total_actual_points,
            "total_predicted_points": total_predicted_points,
            "initial_budget": config.budget,
            "final_budget": config.budget + 1.0,
            "total_budget_delta": 1.0,
            "min_budget": config.budget,
            "max_budget_drawdown": 0.0,
            "budget_constrained_rounds": 0,
            "actual_points_delta_vs_price": 0.5,
        },
            {
                "strategy": "price",
                "rounds": 1,
                "total_actual_points": 1.5,
                "average_actual_points": 1.5,
            "total_predicted_points": 1.5,
            "initial_budget": config.budget,
            "final_budget": config.budget,
            "total_budget_delta": 0.0,
            "min_budget": config.budget,
            "max_budget_drawdown": 0.0,
            "budget_constrained_rounds": 0,
            "actual_points_delta_vs_price": 0.0,
        },
        ]
    )
    return BacktestResult(
        round_results=round_results,
        selected_players=pd.DataFrame(selected_rows),
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=pd.DataFrame(),
        metadata=_metadata(config),
    )


class _FinalizeWarningTracker(InMemoryExperimentTracker):
    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        super().log_parent_artifacts(artifact_paths)
        self.warnings.append(TrackerWarning(phase="log_parent_artifacts", message="late artifact warning"))

    def end_experiment(self, *, status: TrackerStatus) -> None:
        super().end_experiment(status=status)
        self.warnings.append(TrackerWarning(phase="end_experiment", message=f"late close warning: {status}"))


class _RaisingFinalizeTracker(InMemoryExperimentTracker):
    def end_experiment(self, *, status: TrackerStatus) -> None:
        super().end_experiment(status=status)
        raise RuntimeError(f"tracker close failed: {status}")


def test_experiment_runner_executes_child_runs_sequentially(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_model_ids: list[str] = []

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        observed_model_ids.append(primary_model_id)
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    assert observed_model_ids == [
        "random_forest",
        "random_forest",
        "extra_trees",
        "extra_trees",
        "hist_gradient_boosting",
        "hist_gradient_boosting",
        "ridge",
        "ridge",
    ]
    assert result.output_path == tmp_path / "data/08_reporting/experiments/model_feature" / result.experiment_id
    for artifact in (
        "experiment_metadata.json",
        "ranked_summary.csv",
        "per_season_summary.csv",
        "prediction_metrics.csv",
        "calibration_deciles.csv",
        "comparability_report.json",
        "comparison_report.md",
        "calibration_plots.html",
        "squad_performance_comparison.html",
    ):
        assert (result.output_path / artifact).exists()
    squad_html = (result.output_path / "squad_performance_comparison.html").read_text(encoding="utf-8")
    calibration_html = (result.output_path / "calibration_plots.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in squad_html
    metadata = json.loads((result.output_path / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["budget_policy"] == BUDGET_POLICY_MOVING
    assert metadata["initial_budget"] == 100.0
    assert "Plotly.newPlot" in calibration_html
    assert "random_forest / ppg / none" in squad_html
    assert "promotion_eligible" in squad_html


def test_experiment_runner_emits_progress_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[ExperimentProgressEvent] = []

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
        progress_callback=events.append,
    )

    assert [event.event_type for event in events].count("child_started") == 8
    assert [event.event_type for event in events].count("child_finished") == 8
    assert events[0].event_type == "experiment_started"
    assert events[-1].event_type == "experiment_finished"
    assert events[0].total_children == 8
    assert events[1].child_index == 1
    assert events[1].child_id == "season=2025/model=random_forest/feature_pack=ppg"
    assert events[-1].completed_children == 8


def test_experiment_runner_aborts_on_child_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        raise RuntimeError("child failed")

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(RuntimeError, match="child failed"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments"),
            started_at_utc="20260430T200000000000Z",
        )

    output_path = _expected_output_path(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )
    assert (output_path / "experiment_metadata.json").exists()
    assert (output_path / "comparability_report.json").exists()
    assert not (output_path / "ranked_summary.csv").exists()


def test_experiment_runner_failure_metadata_preserves_completed_child_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_child_ids: list[str] = []

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        child_id = f"season={config.season}/model={primary_model_id}/{config.output_path.parts[-1]}"
        observed_child_ids.append(child_id)
        if len(observed_child_ids) == 2:
            raise RuntimeError("second child failed")
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(RuntimeError, match="second child failed"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments"),
            started_at_utc="20260430T200000000000Z",
        )

    output_path = _expected_output_path(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )
    metadata = json.loads((output_path / "experiment_metadata.json").read_text(encoding="utf-8"))

    assert len(observed_child_ids) == 2
    assert metadata["status"] == "failed"
    assert metadata["child_runs"][0]["child_id"] == observed_child_ids[0]
    assert len(metadata["child_runs"]) == 1
    assert metadata["failure"]["phase"] == "child_run"
    assert metadata["failure"]["child_id"] == observed_child_ids[1]
    assert observed_child_ids[0] in metadata["candidate_pool_signatures"]
    assert observed_child_ids[0] in metadata["solver_status_signatures"]
    assert not (output_path / "ranked_summary.csv").exists()


def test_experiment_runner_rejects_output_collision(tmp_path: Path) -> None:
    output_path = _expected_output_path(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )
    output_path.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments"),
            started_at_utc="20260430T200000000000Z",
        )


def test_experiment_runner_fails_on_candidate_mismatch_before_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        feature_pack = config.output_path.parts[-1]
        candidate_price = 11.0 if feature_pack == "feature_pack=ppg_xg" else 10.0
        return _result(config, model_id=primary_model_id, candidate_price=candidate_price)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(ComparabilityError):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments"),
            started_at_utc="20260430T200000000000Z",
        )

    output_path = _expected_output_path(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )
    assert (output_path / "comparability_report.json").exists()
    assert not (output_path / "ranked_summary.csv").exists()


def test_ranked_summary_aggregates_configs_across_seasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_points = {
        (2024, "random_forest", "feature_pack=ppg"): 10.0,
        (2025, "random_forest", "feature_pack=ppg"): 20.0,
        (2024, "extra_trees", "feature_pack=ppg"): 13.0,
        (2025, "extra_trees", "feature_pack=ppg"): 22.0,
    }

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        feature_pack = config.output_path.parts[-1]
        total_actual_points = actual_points.get((config.season, primary_model_id, feature_pack), 5.0)
        return _result(
            config,
            model_id=primary_model_id,
            candidate_count=60,
            total_actual_points=total_actual_points,
            total_predicted_points=total_actual_points + 1.0,
        )

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2024, 2025),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )

    ranked = pd.read_csv(result.output_path / "ranked_summary.csv")
    per_season = pd.read_csv(result.output_path / "per_season_summary.csv")

    assert len(per_season) == 16
    assert len(ranked) == 8
    assert ranked[["model_id", "feature_pack", "fixture_mode"]].duplicated().sum() == 0

    row = ranked[(ranked["model_id"] == "extra_trees") & (ranked["feature_pack"] == "ppg")].iloc[0]
    assert row["seasons_evaluated"] == 2
    assert row["total_rounds"] == 2
    assert row["total_actual_points"] == 35.0
    assert row["baseline_total_actual_points"] == 30.0
    assert row["aggregate_delta"] == 5.0
    assert row["average_actual_delta_per_round"] == 2.5
    assert row["improved_seasons"] == 2
    assert row["worst_season_avg_delta"] == 2.0
    assert row["worst_min_budget"] == 100.0
    assert row["worst_max_budget_drawdown"] == 0.0
    assert row["total_budget_constrained_rounds"] == 0


def test_ranked_summary_keeps_budget_policies_distinct() -> None:
    per_season = pd.DataFrame(
        [
            {
                "season": 2025,
                "model_id": "random_forest",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "budget_policy": BUDGET_POLICY_FIXED,
                "rounds": 1,
                "total_actual_points": 10.0,
                "total_predicted_points": 10.0,
            },
            {
                "season": 2025,
                "model_id": "random_forest",
                "feature_pack": "ppg",
                "fixture_mode": "none",
                "budget_policy": BUDGET_POLICY_MOVING,
                "rounds": 1,
                "total_actual_points": 20.0,
                "total_predicted_points": 20.0,
            },
        ]
    )

    ranked = _rank_summary(per_season, pd.DataFrame())

    assert len(ranked) == 2
    assert set(ranked["budget_policy"]) == {BUDGET_POLICY_FIXED, BUDGET_POLICY_MOVING}
    assert set(ranked["total_actual_points"]) == {10.0, 20.0}


def test_prediction_metrics_and_calibration_deciles_are_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )

    prediction_metrics = pd.read_csv(result.output_path / "prediction_metrics.csv")
    calibration_deciles = pd.read_csv(result.output_path / "calibration_deciles.csv")

    assert not prediction_metrics.empty
    assert not calibration_deciles.empty
    assert {"candidate_pool", "top50_candidates", "selected_players"}.issubset(
        set(prediction_metrics["metric_scope"])
    )
    assert prediction_metrics[prediction_metrics["metric_scope"] == "top50_candidates"]["observed_count"].min() == 50
    assert prediction_metrics[prediction_metrics["metric_scope"] == "selected_players"]["observed_count"].min() == 12
    assert calibration_deciles["decile"].between(1, 10).all()


def test_experiment_runner_writes_failure_artifacts_when_candidate_signature_build_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_child_ids: list[str] = []

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        child_id = f"season={config.season}/model={primary_model_id}/{config.output_path.parts[-1]}"
        observed_child_ids.append(child_id)
        result = _result(config, model_id=primary_model_id)
        if len(observed_child_ids) == 2:
            return replace(result, player_predictions=result.player_predictions.drop(columns=["preco_pre_rodada"]))
        return result

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(ComparabilityError, match="Missing required candidate columns: preco_pre_rodada"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments"),
            started_at_utc="20260430T200000000000Z",
        )

    output_path = _expected_output_path(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )
    metadata_path = output_path / "experiment_metadata.json"
    report_path = output_path / "comparability_report.json"

    assert metadata_path.exists()
    assert report_path.exists()
    assert not (output_path / "ranked_summary.csv").exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(observed_child_ids) == 2
    assert metadata["status"] == "failed"
    assert metadata["failure"]["phase"] == "comparability"
    assert metadata["failure"]["message"] == "Missing required candidate columns: preco_pre_rodada"
    assert metadata["failure"]["child_id"] == observed_child_ids[1]
    assert len(metadata["child_runs"]) == 2
    assert observed_child_ids[0] in metadata["candidate_pool_signatures"]
    assert observed_child_ids[1] not in metadata["candidate_pool_signatures"]
    assert report["status"] == "failed"
    assert report["failure"] == metadata["failure"]


def test_experiment_runner_allows_candidate_pools_to_differ_across_seasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_ids_by_season = {2024: 101, 2025: 202}

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(
            config,
            model_id=primary_model_id,
            candidate_id=candidate_ids_by_season[config.season],
        )

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2024, 2025),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments"),
        started_at_utc="20260430T200000000000Z",
    )

    assert (result.output_path / "ranked_summary.csv").exists()


def test_experiment_runner_writes_index_rows_for_successful_fake_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    index_path = tmp_path / "data/08_reporting/experiments/experiment_index.sqlite"
    with sqlite3.connect(index_path) as connection:
        experiment_rows = connection.execute(
            "SELECT experiment_id, status, budget_policy, completed_child_run_count, failed_child_run_count FROM experiments"
        ).fetchall()
        child_count = connection.execute("SELECT COUNT(*) FROM child_runs").fetchone()[0]
        first_child = connection.execute(
            "SELECT child_run_id, status, budget_policy, comparable_within_partition FROM child_runs ORDER BY child_run_id LIMIT 1"
        ).fetchone()

    assert experiment_rows == [(result.experiment_id, "ok", BUDGET_POLICY_MOVING, 8, 0)]
    assert child_count == 8
    assert first_child[1:] == ("ok", BUDGET_POLICY_MOVING, 1)


def test_experiment_runner_sends_tracker_events_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = InMemoryExperimentTracker()

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
        tracker=tracker,
    )

    assert tracker.events[0]["event"] == "start_experiment"
    assert tracker.events[0]["experiment_name"] == "cartola-production-parity"
    assert tracker.events[0]["run_name"] == result.experiment_id
    assert tracker.events[0]["params"]["budget_policy"] == BUDGET_POLICY_MOVING
    assert tracker.events[0]["params"]["initial_budget"] == 100.0
    assert [event["event"] for event in tracker.events].count("start_child") == 8
    assert tracker.events[1]["params"]["budget_policy"] == BUDGET_POLICY_MOVING
    assert tracker.events[1]["params"]["initial_budget"] == 100.0
    assert [event["event"] for event in tracker.events].count("end_child") == 8
    assert tracker.events[-1] == {"event": "end_experiment", "status": "ok"}


def test_experiment_runner_finalizes_tracker_and_index_on_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _FinalizeWarningTracker()

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        if primary_model_id == "extra_trees":
            raise RuntimeError("boom")
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments/model_feature"),
            started_at_utc="20260430T200000000000Z",
            tracker=tracker,
        )

    index_path = tmp_path / "data/08_reporting/experiments/experiment_index.sqlite"
    with sqlite3.connect(index_path) as connection:
        status, completed, failed = connection.execute(
            "SELECT status, completed_child_run_count, failed_child_run_count FROM experiments"
        ).fetchone()
        completed_children = connection.execute("SELECT COUNT(*) FROM child_runs WHERE status = 'ok'").fetchone()[0]

    assert (status, completed, failed) == ("failed", 2, 1)
    assert completed_children == 2
    assert tracker.events[-2:] == [{"event": "end_child", "status": "failed"}, {"event": "end_experiment", "status": "failed"}]
    metadata = json.loads(
        (
            _expected_output_path(
                group="production-parity",
                seasons=(2025,),
                start_round=5,
                budget=100.0,
                current_year=2026,
                jobs=4,
                project_root=tmp_path,
                output_root=Path("experiments/model_feature"),
                started_at_utc="20260430T200000000000Z",
            )
            / "experiment_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["tracking_warnings"] == [
        {"message": "late close warning: failed", "phase": "end_experiment"}
    ]


def test_tracker_close_failure_does_not_mask_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _RaisingFinalizeTracker()

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        if primary_model_id == "extra_trees":
            raise ValueError("original child failure")
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(ValueError, match="original child failure"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments/model_feature"),
            started_at_utc="20260430T200000000000Z",
            tracker=tracker,
        )

    metadata = json.loads(
        (
            _expected_output_path(
                group="production-parity",
                seasons=(2025,),
                start_round=5,
                budget=100.0,
                current_year=2026,
                jobs=4,
                project_root=tmp_path,
                output_root=Path("experiments/model_feature"),
                started_at_utc="20260430T200000000000Z",
            )
            / "experiment_metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert metadata["failure"]["message"] == "original child failure"
    assert metadata["tracking_warnings"] == [
        {"message": "RuntimeError: tracker close failed: failed", "phase": "end_experiment"}
    ]
    assert [event for event in tracker.events if event["event"] == "end_experiment"] == [
        {"event": "end_experiment", "status": "failed"}
    ]


def test_experiment_runner_writes_artifact_pointers_for_large_child_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        result = _result(config, model_id=primary_model_id, candidate_count=60)
        config.output_path.mkdir(parents=True, exist_ok=True)
        result.player_predictions.to_csv(config.output_path / "player_predictions.csv", index=False)
        result.selected_players.to_csv(config.output_path / "selected_players.csv", index=False)
        return result

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    pointer_path = (
        _expected_output_path(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments/model_feature"),
            started_at_utc="20260430T200000000000Z",
        )
        / "runs"
        / "season=2025"
        / "model=random_forest"
        / "feature_pack=ppg"
        / "artifact_pointers.json"
    )
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))

    assert sorted(payload["artifacts"]) == ["player_predictions.csv", "selected_players.csv"]
    assert payload["artifacts"]["player_predictions.csv"]["size_bytes"] > 0
    assert payload["artifacts"]["selected_players.csv"]["size_bytes"] > 0


def test_tracker_none_does_not_change_scientific_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    metadata = json.loads((result.output_path / "experiment_metadata.json").read_text(encoding="utf-8"))
    ranked = pd.read_csv(result.output_path / "ranked_summary.csv")

    assert metadata["status"] == "ok"
    assert metadata["tracking_warnings"] == []
    assert metadata["index_warnings"] == []
    assert len(ranked) == 8
    assert (tmp_path / "data/08_reporting/experiments/experiment_index.sqlite").exists()


def test_success_metadata_records_late_observability_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _FinalizeWarningTracker()
    original_upsert_experiment = ExperimentIndex.upsert_experiment

    def flaky_upsert_experiment(self: ExperimentIndex, row: Mapping[str, object]) -> None:
        if row["status"] == "ok":
            raise RuntimeError("final index failed")
        original_upsert_experiment(self, row)

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(ExperimentIndex, "upsert_experiment", flaky_upsert_experiment)
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
        tracker=tracker,
    )

    metadata = json.loads((result.output_path / "experiment_metadata.json").read_text(encoding="utf-8"))

    assert result.ranked_summary.shape[0] == 8
    assert "upsert_experiment: RuntimeError: final index failed" in metadata["index_warnings"]
    assert metadata["tracking_warnings"] == [
        {"message": "late artifact warning", "phase": "log_parent_artifacts"},
        {"message": "late close warning: ok", "phase": "end_experiment"},
    ]


def _expected_output_path(
    *,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    project_root: Path,
    output_root: Path,
    started_at_utc: str,
) -> Path:
    specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root,
        current_year=current_year,
        jobs=jobs,
    )
    matrix_hash = config_hash({"child_runs": [spec.config_identity for spec in specs]})
    run_id = experiment_id(group=group, started_at_utc=started_at_utc, matrix_hash=matrix_hash)
    return project_root / output_root / run_id
