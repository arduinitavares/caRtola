#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

build_h004_residual_diagnostic: Callable[..., Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H004 residual diagnostic from model experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/hypotheses"))
    parser.add_argument("--seasons", type=_parse_seasons, default=(2021, 2022, 2023, 2024, 2025))
    parser.add_argument("--model-id", default="xgboost_depth2_slow")
    parser.add_argument("--feature-pack", default="ppg_xg_matchup")
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_dotenv(project_root: Path | None = None) -> bool:
    resolved_project_root = _project_root() if project_root is None else project_root
    dotenv_path = resolved_project_root.expanduser() / ".env"
    if not dotenv_path.is_file():
        return False
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return True


def _load_runtime_dependencies() -> None:
    global build_h004_residual_diagnostic
    if build_h004_residual_diagnostic is None:
        from cartola.backtesting.h004_residual_diagnostic import (
            build_h004_residual_diagnostic as imported_build_h004_residual_diagnostic,
        )

        build_h004_residual_diagnostic = imported_build_h004_residual_diagnostic


def _parse_seasons(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seasons value: {value}") from exc
    duplicates = sorted({season for season in seasons if seasons.count(season) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"Duplicate seasons are not allowed: {duplicates}")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()

    console = Console()
    seasons = args.seasons
    output_path = args.output_root / f"h004_residual_diagnostic_started_at={_timestamp()}"
    console.print(
        f"H004 residual diagnostic started: seasons={','.join(str(season) for season in seasons)} "
        f"model_id={args.model_id} feature_pack={args.feature_pack} output={output_path}"
    )

    with console.status("Loading H004 residual diagnostic runtime..."):
        _load_runtime_dependencies()
    if build_h004_residual_diagnostic is None:
        raise RuntimeError("H004 diagnostic runtime dependencies were not loaded.")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task_id = progress.add_task("Loading artifacts and computing residual diagnostics", total=None)
        result = build_h004_residual_diagnostic(
            experiment_path=args.experiment_path,
            output_path=output_path,
            seasons=seasons,
            model_id=str(args.model_id),
            feature_pack=str(args.feature_pack),
        )
        progress.update(task_id, description="Writing H004 residual diagnostic artifacts")

    console.print(
        Panel(
            f"diagnostic_status={result.decision.get('diagnostic_status')}\noutput_path={result.output_path}",
            title="H004 residual diagnostic complete",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
