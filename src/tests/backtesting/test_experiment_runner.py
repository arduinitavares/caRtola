import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_config import build_child_run_specs, config_hash, experiment_id
from cartola.backtesting.experiment_runner import run_model_experiment
from cartola.backtesting.experiment_signatures import ComparabilityError
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
        thread_env={
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "BLIS_NUM_THREADS": None,
        },
        scoring_contract_version=str(contract["scoring_contract_version"]),
        captain_scoring_enabled=bool(contract["captain_scoring_enabled"]),
        captain_multiplier=float(contract["captain_multiplier"]),
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
) -> BacktestResult:
    round_results = pd.DataFrame(
        [
            {"rodada": 5, "strategy": "baseline", "solver_status": "Optimal", "actual_points": 1.0, "predicted_points": 1.0},
            {"rodada": 5, "strategy": model_id, "solver_status": "Optimal", "actual_points": 2.0, "predicted_points": 2.0},
            {"rodada": 5, "strategy": "price", "solver_status": "Optimal", "actual_points": 1.5, "predicted_points": 1.5},
        ]
    )
    player_predictions = pd.DataFrame(
        [
            {
                "rodada": 5,
                "id_atleta": candidate_id,
                "posicao": "ata",
                "id_clube": 1,
                "status": "Provavel",
                "preco_pre_rodada": candidate_price,
                "pontuacao": 2.0,
                f"{model_id}_score": 2.0,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": "baseline",
                "rounds": 1,
                "total_actual_points": 1.0,
                "average_actual_points": 1.0,
                "total_predicted_points": 1.0,
                "actual_points_delta_vs_price": -0.5,
            },
            {
                "strategy": model_id,
                "rounds": 1,
                "total_actual_points": 2.0,
                "average_actual_points": 2.0,
                "total_predicted_points": 2.0,
                "actual_points_delta_vs_price": 0.5,
            },
            {
                "strategy": "price",
                "rounds": 1,
                "total_actual_points": 1.5,
                "average_actual_points": 1.5,
                "total_predicted_points": 1.5,
                "actual_points_delta_vs_price": 0.0,
            },
        ]
    )
    return BacktestResult(
        round_results=round_results,
        selected_players=pd.DataFrame(),
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=pd.DataFrame(),
        metadata=_metadata(config),
    )


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
    child_index = 0

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        nonlocal child_index
        child_index += 1
        candidate_price = 11.0 if child_index == 3 else 10.0
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


def _expected_output_path(
    *,
    group: str,
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
