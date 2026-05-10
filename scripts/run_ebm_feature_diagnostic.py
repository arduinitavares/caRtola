#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel


class _DiagnosticResult(Protocol):
    output_path: Path
    decision: Mapping[str, object]


build_ebm_feature_diagnostic: Callable[..., _DiagnosticResult] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EBM feature diagnostic from model experiment artifacts.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/ebm_diagnostics"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-pack", required=True)
    parser.add_argument("--seasons", type=_parse_seasons, required=True)
    parser.add_argument("--fixture-mode", default="exploratory")
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--max-interactions", type=int, default=10)
    parser.add_argument("--min-validation-rows", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--profile-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap_dotenv()

    console = Console()
    output_path = args.output_root / f"ebm_diagnostic_started_at={_timestamp()}"
    console.print(
        f"EBM diagnostic started: seasons={','.join(str(value) for value in args.seasons)} "
        f"model_id={args.model_id} feature_pack={args.feature_pack} fixture_mode={args.fixture_mode} "
        f"output_path={output_path}"
    )
    try:
        _load_runtime_dependencies()
        if build_ebm_feature_diagnostic is None:
            raise RuntimeError("EBM diagnostic runtime was not loaded.")
        result = build_ebm_feature_diagnostic(
            experiment_path=args.experiment_path,
            output_path=output_path,
            seasons=args.seasons,
            model_id=args.model_id,
            feature_pack=args.feature_pack,
            fixture_mode=args.fixture_mode,
            current_year=args.current_year,
            max_interactions=args.max_interactions,
            min_validation_rows=args.min_validation_rows,
            random_seed=args.random_seed,
            profile_runtime=args.profile_runtime,
            progress_callback=lambda message: console.print(message),
        )
    except Exception as exc:
        console.print(
            Panel(
                f"diagnostic_status=failed\noutput_path={output_path}\nerror={exc}",
                title="EBM diagnostic failed",
                border_style="red",
            )
        )
        return 1

    console.print(
        Panel(
            _completion_summary(result),
            title="EBM diagnostic complete",
            border_style="green",
        )
    )
    return 0


def _completion_summary(result: _DiagnosticResult) -> str:
    decision = result.decision
    lines = [f"diagnostic_status={decision.get('diagnostic_status')}"]
    diagnostic_phase = decision.get("diagnostic_phase") or decision.get("diagnostic_scope")
    if diagnostic_phase is not None:
        lines.append(f"diagnostic_phase={diagnostic_phase}")
    lines.append(f"output_path={result.output_path}")
    return "\n".join(lines)


def _parse_seasons(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seasons value: {value}") from exc
    if not seasons:
        raise argparse.ArgumentTypeError("At least one season is required")
    duplicate_seasons = tuple(season for season in sorted(set(seasons)) if seasons.count(season) > 1)
    if duplicate_seasons:
        duplicate_list = ", ".join(str(season) for season in duplicate_seasons)
        raise argparse.ArgumentTypeError(f"Duplicate seasons are not allowed: {duplicate_list}")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _bootstrap_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _load_runtime_dependencies() -> None:
    global build_ebm_feature_diagnostic
    if build_ebm_feature_diagnostic is None:
        from cartola.backtesting.ebm_feature_diagnostic import (
            build_ebm_feature_diagnostic as imported_build_ebm_feature_diagnostic,
        )

        build_ebm_feature_diagnostic = imported_build_ebm_feature_diagnostic


if __name__ == "__main__":
    raise SystemExit(main())
