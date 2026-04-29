#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.live_workflow import LiveWorkflowConfig, run_live_round


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the open Cartola market and generate a live squad recommendation.")
    parser.add_argument("--season", type=_positive_int, required=True)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/recommendations"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--capture-policy", choices=("fresh", "missing", "skip"), default="fresh")
    parser.add_argument("--allow-finalized-live-data", action="store_true")
    return parser.parse_args(argv)


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _print_success(console: Console, metadata: dict[str, object]) -> None:
    console.print(
        Panel(
            f"season={metadata.get('season')}  round={metadata.get('target_round')}  status={metadata.get('status')}",
            title="Live round complete",
            border_style="green",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Capture policy", str(metadata.get("capture_policy")))
    table.add_row("Target round", str(metadata.get("target_round")))
    table.add_row("Capture timestamp", str(metadata.get("capture_captured_at_utc")))
    table.add_row("Capture age seconds", str(metadata.get("capture_age_seconds")))
    table.add_row("FootyStats mode", str(metadata.get("footystats_mode")))
    table.add_row("Selected players", str(metadata.get("selected_count")))
    table.add_row("Predicted total", _format_float(metadata.get("predicted_points")))
    table.add_row("Captain", str(metadata.get("captain_name", "n/a")))
    table.add_row("Captain bonus", _format_float(metadata.get("captain_bonus_predicted")))
    table.add_row("Budget used", _format_float(metadata.get("budget_used")))
    table.add_row("Recommendation output", str(metadata.get("recommendation_output_path")))
    table.add_row("Capture metadata", str(metadata.get("capture_metadata_path")))
    console.print(table)


def _print_error(console: Console, error: ValueError) -> None:
    console.print(Panel(str(error), title="Live round failed", border_style="red"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = LiveWorkflowConfig(
        season=args.season,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
        capture_policy=args.capture_policy,
        allow_finalized_live_data=args.allow_finalized_live_data,
    )
    stdout = Console()
    stderr = Console(stderr=True)
    try:
        result = run_live_round(config)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, result.workflow_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
