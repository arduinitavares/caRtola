from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_config import feature_pack_to_modes
from cartola.backtesting.experiment_metrics import calibration_slope_intercept
from cartola.backtesting.runner import BacktestResult, run_backtest_for_experiment

CSV_FLOAT_FORMAT = "%.10f"


class OptunaTrial(Protocol):
    number: int

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int: ...

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float: ...

    def set_user_attr(self, key: str, value: object) -> None: ...


class OptunaFrozenTrialState(Protocol):
    name: str


class OptunaFrozenTrial(Protocol):
    number: int
    value: float | None
    params: dict[str, object]
    state: OptunaFrozenTrialState
    user_attrs: dict[str, object]


class OptunaStudy(Protocol):
    trials: list[OptunaFrozenTrial]
    best_trial: OptunaFrozenTrial

    def optimize(
        self,
        func: Callable[[OptunaTrial], float],
        *,
        n_trials: int,
        timeout: int | None = None,
    ) -> None: ...


class OptunaSamplers(Protocol):
    def TPESampler(self, *, seed: int) -> object: ...


class OptunaModule(Protocol):
    samplers: OptunaSamplers

    def create_study(
        self,
        *,
        direction: str,
        sampler: object,
        storage: str,
        study_name: str,
        load_if_exists: bool,
    ) -> OptunaStudy: ...


@dataclass(frozen=True)
class TrialAggregateMetrics:
    total_actual_points: float
    total_rounds: int
    worst_min_budget: float
    worst_max_budget_drawdown: float
    total_budget_constrained_rounds: int
    selected_calibration_slope: float | None
    season_actual_points: dict[int, float]
    total_predicted_points: float = 0.0
    model_params: Mapping[str, object] = field(default_factory=dict)
    output_path: str | None = None


@dataclass(frozen=True)
class BalancedObjectiveWeights:
    min_budget_floor: float = 75.0
    allowed_drawdown_delta: float = 15.0
    allowed_budget_constrained_delta: int = 2
    recent_season: int = 2025
    recent_season_min_delta: float = 50.0
    min_budget_penalty: float = 15.0
    drawdown_penalty: float = 10.0
    budget_constrained_penalty: float = 75.0
    recent_regression_penalty: float = 2.0
    calibration_penalty: float = 300.0


def suggest_xgboost_parameters(trial: OptunaTrial) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 200.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }


def balanced_objective_score(
    candidate: TrialAggregateMetrics,
    control: TrialAggregateMetrics,
    *,
    weights: BalancedObjectiveWeights = BalancedObjectiveWeights(),
) -> float:
    points_delta = candidate.total_actual_points - control.total_actual_points
    min_budget_gap = max(0.0, weights.min_budget_floor - candidate.worst_min_budget)
    drawdown_gap = max(
        0.0,
        candidate.worst_max_budget_drawdown
        - control.worst_max_budget_drawdown
        - weights.allowed_drawdown_delta,
    )
    budget_constrained_gap = max(
        0,
        candidate.total_budget_constrained_rounds
        - control.total_budget_constrained_rounds
        - weights.allowed_budget_constrained_delta,
    )
    recent_delta = (
        candidate.season_actual_points.get(weights.recent_season, 0.0)
        - control.season_actual_points.get(weights.recent_season, 0.0)
    )
    recent_gap = max(0.0, weights.recent_season_min_delta - recent_delta)
    calibration_gap = _calibration_gap(candidate.selected_calibration_slope)
    penalty = (
        min_budget_gap * weights.min_budget_penalty
        + drawdown_gap * weights.drawdown_penalty
        + budget_constrained_gap * weights.budget_constrained_penalty
        + recent_gap * weights.recent_regression_penalty
        + calibration_gap * weights.calibration_penalty
    )
    return round(points_delta - penalty, 10)


def summarize_trial_results(
    results_by_season: Mapping[int, BacktestResult],
    *,
    model_id: str,
    model_params: Mapping[str, object],
    output_path: Path,
) -> TrialAggregateMetrics:
    total_actual_points = 0.0
    total_predicted_points = 0.0
    total_rounds = 0
    worst_min_budget: float | None = None
    worst_max_budget_drawdown = 0.0
    total_budget_constrained_rounds = 0
    season_actual_points: dict[int, float] = {}
    selected_frames: list[pd.DataFrame] = []

    for season, result in sorted(results_by_season.items()):
        summary_row = _strategy_summary_row(result.summary, strategy=model_id)
        season_total = float(summary_row["total_actual_points"])
        season_actual_points[int(season)] = season_total
        total_actual_points += season_total
        total_predicted_points += float(summary_row["total_predicted_points"])
        total_rounds += int(summary_row["rounds"])
        min_budget = float(summary_row["min_budget"])
        worst_min_budget = min_budget if worst_min_budget is None else min(worst_min_budget, min_budget)
        worst_max_budget_drawdown = max(worst_max_budget_drawdown, float(summary_row["max_budget_drawdown"]))
        total_budget_constrained_rounds += int(summary_row["budget_constrained_rounds"])
        selected_frames.append(_selected_players_for_strategy(result.selected_players, strategy=model_id))

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    calibration = calibration_slope_intercept(selected["predicted_points"], selected["pontuacao"]) if not selected.empty else {}
    slope = calibration.get("calibration_slope")
    return TrialAggregateMetrics(
        total_actual_points=round(total_actual_points, 10),
        total_rounds=total_rounds,
        worst_min_budget=round(float(worst_min_budget or 0.0), 10),
        worst_max_budget_drawdown=round(worst_max_budget_drawdown, 10),
        total_budget_constrained_rounds=total_budget_constrained_rounds,
        selected_calibration_slope=None if slope is None else float(slope),
        season_actual_points=season_actual_points,
        total_predicted_points=round(total_predicted_points, 10),
        model_params=dict(model_params),
        output_path=str(output_path),
    )


def load_control_metrics(
    *,
    experiment_path: Path,
    control_model: str,
    control_feature_pack: str,
    seasons: tuple[int, ...],
) -> TrialAggregateMetrics:
    per_season = pd.read_csv(experiment_path / "per_season_summary.csv")
    ranked = pd.read_csv(experiment_path / "ranked_summary.csv")
    rows = per_season[
        (per_season["model_id"] == control_model)
        & (per_season["feature_pack"] == control_feature_pack)
        & (per_season["season"].isin(seasons))
    ].copy()
    if len(rows) != len(seasons):
        missing = sorted(set(seasons) - set(rows["season"].astype(int)))
        raise ValueError(f"Missing control rows for seasons: {missing}")
    rank_row = ranked[(ranked["model_id"] == control_model) & (ranked["feature_pack"] == control_feature_pack)]
    selected_calibration_slope = None
    if not rank_row.empty and "selected_calibration_slope" in rank_row:
        value = rank_row.iloc[0]["selected_calibration_slope"]
        selected_calibration_slope = None if pd.isna(value) else float(value)
    return TrialAggregateMetrics(
        total_actual_points=round(float(rows["total_actual_points"].sum()), 10),
        total_rounds=int(rows["rounds"].sum()),
        worst_min_budget=round(float(rows["min_budget"].min()), 10),
        worst_max_budget_drawdown=round(float(rows["max_budget_drawdown"].max()), 10),
        total_budget_constrained_rounds=int(rows["budget_constrained_rounds"].sum()),
        selected_calibration_slope=selected_calibration_slope,
        season_actual_points={
            int(row["season"]): float(row["total_actual_points"]) for row in rows.to_dict(orient="records")
        },
        total_predicted_points=round(float(rows["total_predicted_points"].sum()), 10),
        model_params={},
        output_path=str(experiment_path),
    )


def run_xgboost_optuna_tuning(
    *,
    source_experiment_path: Path,
    seasons: tuple[int, ...],
    current_year: int,
    n_trials: int = 40,
    start_round: int = 5,
    budget: float = 100.0,
    project_root: Path = Path("."),
    output_root: Path | None = None,
    control_model: str = "xgboost_depth2_l2_heavy",
    control_feature_pack: str = "ppg_xg",
    feature_pack: str = "ppg_xg",
    jobs: int = 1,
    study_seed: int = 123,
    study_name: str = "xgboost_optuna_tuning",
    timeout_seconds: int | None = None,
    profile_runtime: bool = False,
) -> Path:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not source_experiment_path.exists():
        raise FileNotFoundError(source_experiment_path)

    optuna = _load_optuna()
    output_path = _resolve_output_path(output_root)
    output_path.mkdir(parents=True, exist_ok=output_root is not None)
    control_metrics = load_control_metrics(
        experiment_path=source_experiment_path,
        control_model=control_model,
        control_feature_pack=control_feature_pack,
        seasons=seasons,
    )
    feature_modes = feature_pack_to_modes(feature_pack)

    def objective(trial: OptunaTrial) -> float:
        model_params = suggest_xgboost_parameters(trial)
        trial_path = output_path / "trials" / f"trial={trial.number:03d}"
        trial_results: dict[int, BacktestResult] = {}
        for season in seasons:
            season_path = trial_path / f"season={season}"
            config = BacktestConfig(
                season=season,
                start_round=start_round,
                budget=budget,
                project_root=project_root,
                output_root=season_path,
                fixture_mode="none",
                matchup_context_mode=feature_modes.matchup_context_mode,
                footystats_mode=feature_modes.footystats_mode,
                feature_augmentation_mode=feature_modes.feature_augmentation_mode,
                current_year=current_year,
                jobs=jobs,
                profile_runtime=profile_runtime,
                _output_path_override=season_path,
            )
            trial_results[season] = run_backtest_for_experiment(
                config,
                primary_model_id=control_model,
                model_params=model_params,
            )
        metrics = summarize_trial_results(
            trial_results,
            model_id=control_model,
            model_params=model_params,
            output_path=trial_path,
        )
        score = balanced_objective_score(metrics, control_metrics)
        trial.set_user_attr("metrics", asdict(metrics))
        trial.set_user_attr("objective_score", score)
        _write_json(trial_path / "trial_summary.json", {"objective_score": score, "metrics": asdict(metrics)})
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=study_seed),
        storage=_sqlite_storage_url(output_path / "optuna_study.sqlite"),
        study_name=study_name,
        load_if_exists=True,
    )
    completed_trials_before_resume = _completed_trial_count(study.trials)
    remaining_trials = max(0, n_trials - completed_trials_before_resume)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout_seconds)

    trials = _trial_rows(study.trials)
    trials.to_csv(output_path / "optuna_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    best_trial = study.best_trial
    best_value = _required_trial_value(best_trial)
    best_metrics = best_trial.user_attrs.get("metrics", {})
    _write_json(output_path / "best_candidate_config.json", dict(best_trial.params))
    _write_json(
        output_path / "xgboost_optuna_tuning.json",
        {
            "milestone_id": "M009",
            "status": "completed",
            "source_experiment_path": str(source_experiment_path),
            "control_model": control_model,
            "control_feature_pack": control_feature_pack,
            "feature_pack": feature_pack,
            "seasons": list(seasons),
            "requested_n_trials": n_trials,
            "completed_trials_before_resume": completed_trials_before_resume,
            "completed_trial_count": _completed_trial_count(study.trials),
            "study_seed": study_seed,
            "study_name": study_name,
            "optuna_storage_path": str(output_path / "optuna_study.sqlite"),
            "objective": "balanced_points_minus_budget_recent_calibration_penalties",
            "best_trial_number": int(best_trial.number),
            "best_objective_score": best_value,
            "best_params": dict(best_trial.params),
            "best_metrics": best_metrics,
            "control_metrics": asdict(control_metrics),
            "recommendation": "rerun_top_candidates_through_production_parity_and_m008_gates",
        },
    )
    _write_markdown_report(output_path / "xgboost_optuna_tuning.md", best_trial_number=best_trial.number)
    return output_path


def _strategy_summary_row(summary: pd.DataFrame, *, strategy: str) -> pd.Series:
    rows = summary[summary["strategy"] == strategy]
    if rows.empty:
        raise ValueError(f"Missing summary row for strategy={strategy}")
    return cast(pd.Series, rows.iloc[0])


def _selected_players_for_strategy(selected_players: pd.DataFrame, *, strategy: str) -> pd.DataFrame:
    if "strategy" not in selected_players.columns:
        return selected_players
    return selected_players[selected_players["strategy"] == strategy].copy()


def _calibration_gap(slope: float | None) -> float:
    if slope is None:
        return 0.75
    if slope < 0.75:
        return 0.75 - slope
    if slope > 1.25:
        return slope - 1.25
    return 0.0


def _resolve_output_path(output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    started_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("data/08_reporting/experiments/model_tuning") / f"xgboost_optuna_tuning_started_at={started_at}"


def _sqlite_storage_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def _completed_trial_count(trials: list[OptunaFrozenTrial]) -> int:
    return sum(1 for trial in trials if trial.value is not None)


def _load_optuna() -> OptunaModule:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency is installed in project env
        raise RuntimeError("Optuna is required for M009 tuning. Install project dependencies with `uv sync`.") from exc
    return cast(OptunaModule, optuna)


def _trial_rows(trials: list[OptunaFrozenTrial]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial in trials:
        metrics = trial.user_attrs.get("metrics", {})
        rows.append(
            {
                "trial_number": int(trial.number),
                "state": str(trial.state.name),
                "objective_score": trial.value,
                **{f"param_{key}": value for key, value in trial.params.items()},
                **_flatten_metrics(cast(Mapping[str, object], metrics)),
            }
        )
    return pd.DataFrame(rows)


def _required_trial_value(trial: OptunaFrozenTrial) -> float:
    if trial.value is None:
        raise ValueError(f"Trial {trial.number} has no objective value")
    return float(trial.value)


def _flatten_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "total_actual_points",
        "total_rounds",
        "worst_min_budget",
        "worst_max_budget_drawdown",
        "total_budget_constrained_rounds",
        "selected_calibration_slope",
        "total_predicted_points",
    )
    return {key: metrics.get(key) for key in keys}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _write_markdown_report(path: Path, *, best_trial_number: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# M009 XGBoost Optuna Tuning",
                "",
                f"Best trial: `{best_trial_number}`.",
                "",
                "This artifact is research evidence only. Promote a candidate only after rerunning it through the "
                "standard production-parity and balanced M008-style decision gates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
