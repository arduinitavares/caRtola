#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureConfig,
    StrictRoundFixtureCaptureResult,
    format_project_path,
    format_utc_z,
    run_strict_round_fixture_capture,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a strict round fixture snapshot and generate strict fixtures.")
    parser.add_argument("--season", type=_positive_int, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--auto", action="store_true")
    selector.add_argument("--round", dest="round_number", type=_positive_int, default=None)
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--source", choices=["cartola_api"], default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = StrictRoundFixtureCaptureConfig(
        season=args.season,
        round_number=args.round_number,
        auto=args.auto,
        current_year=args.current_year,
        source=args.source,
        project_root=args.project_root,
        force_generate=args.force_generate,
    )

    stdout = Console(width=140)
    stderr = Console(stderr=True, width=140)
    try:
        result = run_strict_round_fixture_capture(config)
    except StrictFixtureGenerationError as exc:
        _print_generation_error(stderr, args.project_root, exc)
        return 1
    except StrictFixtureCaptureError as exc:
        _print_capture_error(stderr, exc)
        return 1
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError, requests.RequestException) as exc:
        _print_error(stderr, str(exc))
        return 1

    _print_success(stdout, args.project_root, result)
    return 0


def _print_success(console: Console, project_root: Path, result: StrictRoundFixtureCaptureResult) -> None:
    console.print(
        Panel(
            f"season={result.season}  round={result.round_number}",
            title="Strict fixture capture complete",
            border_style="green",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_row("Season", str(result.season))
    table.add_row("Round", str(result.round_number))
    table.add_row("Snapshot directory", format_project_path(project_root, result.capture_dir))
    table.add_row("Strict fixture", format_project_path(project_root, result.fixture_path))
    table.add_row("Manifest", format_project_path(project_root, result.manifest_path))
    table.add_row("Captured at UTC", format_utc_z(result.captured_at_utc))
    table.add_row("Deadline at UTC", format_utc_z(result.deadline_at_utc))
    console.print(table)


def _print_capture_error(console: Console, error: StrictFixtureCaptureError) -> None:
    table = Table(show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_row("Round", str(error.round_number))
    table.add_row("Original error", f"{type(error.original).__name__}: {error.original}")
    console.print(Panel(table, title="Strict fixture capture failed", border_style="red"))


def _print_generation_error(console: Console, project_root: Path, error: StrictFixtureGenerationError) -> None:
    capture_result = error.capture_result
    table = Table(show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_row("Retained snapshot directory", format_project_path(project_root, capture_result.capture_dir))
    table.add_row("Captured at UTC", format_utc_z(capture_result.captured_at_utc))
    table.add_row("Deadline at UTC", format_utc_z(capture_result.deadline_at_utc))
    table.add_row("Original error", f"{type(error.original).__name__}: {error.original}")
    console.print(Panel(table, title="Strict fixture generation failed after snapshot capture", border_style="red"))


def _print_error(console: Console, message: str) -> None:
    console.print(Panel(message, title="Strict fixture capture failed", border_style="red"))


if __name__ == "__main__":
    raise SystemExit(main())
