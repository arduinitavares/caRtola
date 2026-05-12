#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

build_h005_mechanism_audit: Callable[..., Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H005 count-aware matchup reliability mechanism audit.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/hypotheses"))
    parser.add_argument("--seasons", type=_parse_seasons, default=(2021, 2022, 2023, 2024, 2025))
    parser.add_argument("--model-id", default="xgboost_depth2_slow")
    parser.add_argument("--feature-pack", default="ppg_xg_matchup")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_runtime_dependencies() -> None:
    global build_h005_mechanism_audit
    if build_h005_mechanism_audit is None:
        from cartola.backtesting.h005_mechanism_audit import (
            build_h005_mechanism_audit as imported_build_h005_mechanism_audit,
        )

        build_h005_mechanism_audit = imported_build_h005_mechanism_audit


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
    console = Console()
    seasons = args.seasons
    output_path = args.output_root / f"h005_mechanism_audit_started_at={_timestamp()}"
    console.print(
        f"H005 mechanism audit started: seasons={','.join(str(season) for season in seasons)} "
        f"model_id={args.model_id} feature_pack={args.feature_pack} output={output_path}"
    )

    with console.status("Loading H005 mechanism audit runtime..."):
        _load_runtime_dependencies()
    if build_h005_mechanism_audit is None:
        raise RuntimeError("H005 mechanism audit runtime dependencies were not loaded.")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task_id = progress.add_task("Loading source artifacts and recomputing H005 features", total=None)
        result = build_h005_mechanism_audit(
            experiment_path=args.experiment_path,
            output_path=output_path,
            seasons=seasons,
            model_id=str(args.model_id),
            feature_pack=str(args.feature_pack),
            project_root=args.project_root,
        )
        progress.update(task_id, description="Writing H005 mechanism audit artifacts")

    console.print(
        Panel(
            f"audit_status={result.decision.get('audit_status')}\noutput_path={result.output_path}",
            title="H005 mechanism audit complete",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
