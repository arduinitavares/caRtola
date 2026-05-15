#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from cartola.backtesting.xgboost_optuna_tuning import run_xgboost_optuna_tuning


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded Optuna tuning for the no-fixture XGBoost profile.")
    parser.add_argument("--source-experiment-path", type=Path, required=True)
    parser.add_argument("--seasons", default="2020,2021,2022,2023,2024,2025")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--control-model", default="xgboost_depth2_l2_heavy")
    parser.add_argument("--control-feature-pack", default="ppg_xg")
    parser.add_argument("--feature-pack", default="ppg_xg")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--study-seed", type=int, default=123)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--profile-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    project_root = args.project_root.expanduser()
    load_dotenv(dotenv_path=project_root / ".env", override=False)
    if not args.source_experiment_path.exists():
        console.print(Panel(str(args.source_experiment_path), title="Missing source experiment", border_style="red"))
        return 1
    try:
        output_path = run_xgboost_optuna_tuning(
            source_experiment_path=args.source_experiment_path,
            seasons=_parse_seasons(args.seasons),
            n_trials=args.n_trials,
            start_round=args.start_round,
            budget=args.budget,
            current_year=args.current_year,
            project_root=project_root,
            output_root=args.output_root,
            control_model=args.control_model,
            control_feature_pack=args.control_feature_pack,
            feature_pack=args.feature_pack,
            jobs=args.jobs,
            study_seed=args.study_seed,
            timeout_seconds=args.timeout_seconds,
            profile_runtime=args.profile_runtime,
        )
    except Exception as exc:
        console.print(Panel(str(exc), title="XGBoost Optuna tuning failed", border_style="red"))
        return 1

    console.print(
        Panel(
            f"output_path={output_path}",
            title="XGBoost Optuna tuning complete",
            border_style="green",
        )
    )
    return 0


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seasons:
        raise ValueError("At least one season is required")
    return seasons


if __name__ == "__main__":
    raise SystemExit(main())
