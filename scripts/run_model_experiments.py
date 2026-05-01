#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel

from cartola.backtesting.experiment_runner import run_model_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cartola model feature experiment groups.")
    parser.add_argument("--group", choices=("production-parity", "matchup-research"), required=True)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/experiments/model_feature"))
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args(argv)


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seasons:
        raise ValueError("At least one season is required")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _print_error(console: Console, error: Exception) -> None:
    console.print(Panel(str(error), title="Model experiment failed", border_style="red"))


def _print_success(console: Console, *, experiment_id: str, output_path: Path) -> None:
    console.print(
        Panel(
            f"experiment_id={experiment_id}\noutput_path={output_path}",
            title="Model experiment complete",
            border_style="green",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stdout = Console()
    stderr = Console(stderr=True)
    try:
        result = run_model_experiment(
            group=args.group,
            seasons=_parse_seasons(args.seasons),
            start_round=args.start_round,
            budget=args.budget,
            current_year=args.current_year,
            jobs=args.jobs,
            project_root=args.project_root,
            output_root=args.output_root,
            started_at_utc=_timestamp(),
        )
    except Exception as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, experiment_id=result.experiment_id, output_path=result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
