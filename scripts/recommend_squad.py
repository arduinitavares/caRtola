from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.recommendation import RecommendationConfig, run_recommendation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single-round Cartola squad recommendation.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--target-round", type=_positive_int, required=True)
    parser.add_argument("--mode", choices=("live", "replay"), required=True)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/recommendations"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--allow-finalized-live-data", action="store_true")
    return parser.parse_args(argv)


def _float_summary(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _format_points(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_actual_points(summary: dict[str, object]) -> str:
    actual_points = _float_summary(summary, "actual_points")
    if actual_points is None and summary.get("mode") == "live":
        return "n/a (live mode)"
    return _format_points(actual_points)


def _format_delta(summary: dict[str, object]) -> str:
    predicted_points = _float_summary(summary, "predicted_points_with_captain")
    if predicted_points is None:
        predicted_points = _float_summary(summary, "predicted_points")
    actual_points = _float_summary(summary, "actual_points")
    if predicted_points is None or actual_points is None:
        return "n/a"
    return f"{actual_points - predicted_points:+.2f}"


def _format_capture_rate(summary: dict[str, object]) -> str:
    capture_rate = _float_summary(summary, "oracle_capture_rate")
    if capture_rate is None:
        return "n/a"
    return f"{capture_rate * 100:.2f}%"


def _print_success(console: Console, result_summary: dict[str, object]) -> None:
    season = result_summary.get("season", "")
    target_round = result_summary.get("target_round", "")
    mode = result_summary.get("mode", "")
    console.print(
        Panel(
            f"season={season}  round={target_round}  mode={mode}",
            title="Recommendation complete",
            border_style="green",
        )
    )

    budget_used = _float_summary(result_summary, "budget_used")
    budget = _float_summary(result_summary, "budget")
    budget_display = "n/a" if budget_used is None or budget is None else f"{budget_used:.2f} / {budget:.2f}"
    predicted_total = _float_summary(result_summary, "predicted_points_with_captain")
    if predicted_total is None:
        predicted_total = _float_summary(result_summary, "predicted_points")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Predicted total", _format_points(predicted_total))
    table.add_row("Predicted base", _format_points(_float_summary(result_summary, "predicted_points_base")))
    table.add_row("Captain bonus", _format_points(_float_summary(result_summary, "captain_bonus_predicted")))
    table.add_row("Captain", str(result_summary.get("captain_name", "n/a")))
    table.add_row("Formation", str(result_summary.get("formation", "n/a")))
    table.add_row("Actual points", _format_actual_points(result_summary))
    table.add_row("Delta", _format_delta(result_summary))
    table.add_row("Best in candidate pool", _format_points(_float_summary(result_summary, "oracle_actual_points")))
    table.add_row("Gap to best", _format_points(_float_summary(result_summary, "oracle_gap")))
    table.add_row("Capture rate", _format_capture_rate(result_summary))
    table.add_row("Budget used", budget_display)
    table.add_row("Selected players", str(result_summary.get("selected_count", "n/a")))
    table.add_row("Output", str(result_summary.get("output_directory", "n/a")))
    console.print(table)


def _error_hint(error_message: str) -> str | None:
    if error_message.startswith("Target round ") and " not found in season " in error_message:
        try:
            prefix, season_part = error_message.split(" not found in season ", maxsplit=1)
            target_round = prefix.removeprefix("Target round ")
            season = season_part.split(" ", maxsplit=1)[0]
        except ValueError:
            return None
        return f"Save data/01_raw/{season}/rodada-{target_round}.csv before running live mode."
    return None


def _print_expected_error(console: Console, error: ValueError) -> None:
    error_message = str(error)
    hint = _error_hint(error_message)
    body = error_message if hint is None else f"{error_message}\n\n{hint}"
    console.print(Panel(body, title="Recommendation failed", border_style="red"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RecommendationConfig(
        season=args.season,
        target_round=args.target_round,
        mode=args.mode,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
        allow_finalized_live_data=args.allow_finalized_live_data,
    )

    stdout = Console()
    stderr = Console(stderr=True)
    try:
        result = run_recommendation(config)
    except ValueError as error:
        _print_expected_error(stderr, error)
        return 1

    _print_success(stdout, result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
