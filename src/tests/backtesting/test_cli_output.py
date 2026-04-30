from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

import cartola.backtesting.cli_output as cli_output
from cartola.backtesting.cli_output import (
    ChartOutput,
    _build_performance_figure,
    _prepare_chart_data,
    render_backtest_success,
    write_performance_chart,
)
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import BacktestMetadata, BacktestResult
from cartola.backtesting.scoring_contract import contract_fields


def _metadata_for_config(
    config: BacktestConfig,
    *,
    warnings: list[str] | None = None,
    prediction_frames_built: int = 12,
    wall_clock_seconds: float = 3.456,
    scoring_contract_version: str | None = None,
) -> BacktestMetadata:
    contract = contract_fields()
    return BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=10,
        cache_enabled=True,
        prediction_frames_built=prediction_frames_built,
        wall_clock_seconds=wall_clock_seconds,
        backtest_jobs=config.jobs,
        backtest_workers_effective=2,
        model_n_jobs_effective=-1,
        parallel_backend="loky",
        thread_env={
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "BLIS_NUM_THREADS": None,
        },
        scoring_contract_version=scoring_contract_version or str(contract["scoring_contract_version"]),
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
        warnings=[] if warnings is None else warnings,
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


def _backtest_result(
    config: BacktestConfig,
    *,
    summary: pd.DataFrame | None = None,
    metadata: BacktestMetadata | None = None,
) -> BacktestResult:
    return BacktestResult(
        round_results=pd.DataFrame(),
        selected_players=pd.DataFrame(),
        player_predictions=pd.DataFrame(),
        summary=pd.DataFrame() if summary is None else summary,
        diagnostics=pd.DataFrame(),
        metadata=_metadata_for_config(config) if metadata is None else metadata,
    )


def _render_text(config: BacktestConfig, result: BacktestResult, chart_output: ChartOutput) -> str:
    console = Console(record=True, width=140)
    render_backtest_success(console, config=config, result=result, chart_output=chart_output)
    return console.export_text(styles=False)


def _valid_round_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 12.0,
                "formation": "4-4-2",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "baseline",
                "solver_status": "Infeasible",
                "actual_points": 0.0,
                "formation": "",
            },
        ]
    )


def test_render_backtest_success_prints_summary_and_run_details(tmp_path: Path) -> None:
    config = BacktestConfig(
        season=2026,
        project_root=tmp_path,
        output_root=Path("data/08_reporting/backtests"),
        footystats_mode="ppg",
        matchup_context_mode="cartola_matchup_v1",
        jobs=4,
    )
    summary = pd.DataFrame(
        [
            {
                "strategy": "random_forest",
                "rounds": 8,
                "total_actual_points": 70.25,
                "average_actual_points": 8.78125,
                "total_predicted_points": 75.0,
                "actual_points_delta_vs_price": 5.5,
            }
        ]
    )
    result = _backtest_result(config, summary=summary)
    chart_path = config.output_path / "charts" / "strategy_performance_by_round.html"

    text = _render_text(config, result, ChartOutput(path=chart_path, warnings=[]))

    assert "Backtest complete" in text
    assert "season=2026" in text
    assert "start_round=5" in text
    assert "output=data/08_reporting/backtests/2026" in text
    assert "Strategy" in text
    assert "random_forest" in text
    assert "70.25" in text
    assert "+5.50" in text
    assert "Run details" in text
    assert "Performance chart" in text
    assert "data/08_reporting/backtests/2026/charts/strategy_performance_by_round.html" in text
    assert "cartola_standard_2026_v1" in text


def test_render_backtest_success_prints_warning_panel_without_plain_warning_prefix(tmp_path: Path) -> None:
    config = BacktestConfig(project_root=tmp_path)
    metadata = _metadata_for_config(config, warnings=["metadata warning"])
    result = _backtest_result(config, metadata=metadata)

    text = _render_text(config, result, ChartOutput(path=None, warnings=["chart warning"]))

    assert "Backtest warnings" in text
    assert "metadata warning" in text
    assert "chart warning" in text
    assert "WARNING:" not in text


def test_render_backtest_success_handles_missing_summary_values(tmp_path: Path) -> None:
    config = BacktestConfig(project_root=tmp_path)
    summary = pd.DataFrame([{"strategy": "empty_strategy"}])
    result = _backtest_result(config, summary=summary)

    text = _render_text(config, result, ChartOutput(path=None, warnings=[]))

    assert "empty_strategy" in text
    assert "n/a" in text


def test_prepare_chart_data_filters_skipped_rows_from_score_traces() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "Empty",
                "actual_points": 0.0,
                "formation": "",
            },
            {
                "rodada": 3,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 7.5,
                "formation": "4-4-2",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
        ]
    )

    chart_data = _prepare_chart_data(round_results)

    score_rows = chart_data.score_rows.sort_values(["strategy", "rodada"]).reset_index(drop=True)
    assert score_rows[["strategy", "rodada", "actual_points", "cumulative_points"]].to_dict("records") == [
        {"strategy": "baseline", "rodada": 1, "actual_points": 8.0, "cumulative_points": 8.0},
        {"strategy": "random_forest", "rodada": 1, "actual_points": 10.0, "cumulative_points": 10.0},
        {"strategy": "random_forest", "rodada": 3, "actual_points": 7.5, "cumulative_points": 17.5},
    ]
    assert chart_data.status_rows[["strategy", "rodada", "solver_status"]].to_dict("records") == [
        {"strategy": "random_forest", "rodada": 2, "solver_status": "Empty"}
    ]
    assert chart_data.formation_rows[["rodada", "formation"]].to_dict("records") == [
        {"rodada": 1, "formation": "4-3-3"},
        {"rodada": 3, "formation": "4-4-2"},
    ]


def test_prepare_chart_data_accepts_all_non_optimal_rows_without_fake_zero_lines() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Empty",
                "actual_points": "not a score",
                "formation": "",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": None,
                "formation": "",
            },
            {
                "rodada": 3,
                "strategy": "price",
                "solver_status": "Infeasible",
                "actual_points": "",
                "formation": "",
            },
        ]
    )

    chart_data = _prepare_chart_data(round_results)

    assert chart_data.score_rows.empty
    assert chart_data.status_rows["solver_status"].tolist() == ["Empty", "TrainingEmpty", "Infeasible"]
    assert chart_data.formation_rows.empty


def test_prepare_chart_data_rejects_missing_columns() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "solver_status": "Optimal",
                "actual_points": 10.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing chart columns: formation, strategy"):
        _prepare_chart_data(round_results)


def test_prepare_chart_data_rejects_non_numeric_round() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": "first",
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            }
        ]
    )

    with pytest.raises(ValueError, match="Chart column 'rodada' must be numeric"):
        _prepare_chart_data(round_results)


def test_prepare_chart_data_rejects_non_numeric_actual_points_on_optimal_rows() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": "ten",
                "formation": "4-3-3",
            },
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Empty",
                "actual_points": "not checked",
                "formation": "",
            },
        ]
    )

    with pytest.raises(ValueError, match="Chart column 'actual_points' must be numeric for Optimal rows"):
        _prepare_chart_data(round_results)


def test_write_performance_chart_skips_empty_round_results(tmp_path: Path) -> None:
    chart_output = write_performance_chart(pd.DataFrame(), tmp_path)

    assert chart_output == ChartOutput(path=None, warnings=[])
    assert (tmp_path / "charts").is_dir()


def test_performance_figure_has_three_panels_without_zero_status_trace() -> None:
    round_results = pd.DataFrame(
        [
            {
                "rodada": 1,
                "strategy": "random_forest",
                "solver_status": "Optimal",
                "actual_points": 10.0,
                "formation": "4-3-3",
            },
            {
                "rodada": 2,
                "strategy": "random_forest",
                "solver_status": "TrainingEmpty",
                "actual_points": 0.0,
                "formation": "",
            },
            {
                "rodada": 1,
                "strategy": "baseline",
                "solver_status": "Optimal",
                "actual_points": 8.0,
                "formation": "4-3-3",
            },
        ]
    )

    figure = _build_performance_figure(_prepare_chart_data(round_results))

    subplot_titles = [annotation.text for annotation in figure.layout.annotations]
    assert subplot_titles[:3] == [
        "Cumulative actual points",
        "Per-round actual points",
        "Random forest formation by round",
    ]
    assert "TrainingEmpty (1)" in subplot_titles
    assert any(trace.name == "random_forest cumulative" for trace in figure.data)
    assert any(trace.name == "random_forest per round" for trace in figure.data)
    assert any(trace.name == "random_forest formation" for trace in figure.data)
    assert not any(trace.name == "non-optimal status" and list(trace.y) == [0] for trace in figure.data)


def test_write_performance_chart_writes_standalone_html(tmp_path: Path) -> None:
    chart_output = write_performance_chart(_valid_round_results(), tmp_path)

    expected_path = tmp_path / "charts" / "strategy_performance_by_round.html"
    assert chart_output == ChartOutput(path=expected_path, warnings=[])
    assert expected_path.is_file()
    html = expected_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "random_forest" in html
    assert "baseline" in html
    assert "https://cdn.plot.ly" not in html


def test_write_performance_chart_reports_non_fatal_export_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cli_output.go.Figure, "write_html", raise_disk_full)

    chart_output = write_performance_chart(_valid_round_results(), tmp_path)

    assert chart_output.path is None
    assert len(chart_output.warnings) == 1
    assert chart_output.warnings[0].startswith("Performance chart: n/a (")
    assert "disk full" in chart_output.warnings[0]


def test_write_performance_chart_keeps_schema_failures_loud(tmp_path: Path) -> None:
    round_results = _valid_round_results().drop(columns=["formation"])

    with pytest.raises(ValueError, match="Missing chart columns: formation"):
        write_performance_chart(round_results, tmp_path)


def test_write_performance_chart_reports_non_fatal_charts_dir_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_mkdir = Path.mkdir

    def raise_for_charts_dir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self.name == "charts":
            raise OSError("permission denied")
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", raise_for_charts_dir)

    chart_output = write_performance_chart(pd.DataFrame(), tmp_path)

    assert chart_output.path is None
    assert len(chart_output.warnings) == 1
    assert chart_output.warnings[0].startswith("Performance chart: n/a (")
    assert "permission denied" in chart_output.warnings[0]
