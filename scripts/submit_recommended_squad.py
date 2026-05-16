#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.squad_submission import (
    ContractUnverifiedError,
    SquadSubmissionError,
    SquadSubmissionResult,
    SubmissionConfig,
    run_submission,
)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Phase 1 Cartola squad submission plan.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--recommendation-path", type=Path)
    source_group.add_argument("--submission-plan", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--confirm-payload-sha256")
    parser.add_argument("--allow-non-approved-model", action="store_true")
    parser.add_argument("--override-reason")
    parser.add_argument("--safety-margin-seconds", type=_positive_int, default=120)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> SubmissionConfig:
    return SubmissionConfig(
        project_root=args.project_root,
        recommendation_path=args.recommendation_path,
        submission_plan=args.submission_plan,
        timeout_seconds=args.timeout_seconds,
        confirm_submit=args.confirm_submit,
        confirm_payload_sha256=args.confirm_payload_sha256,
        allow_non_approved_model=args.allow_non_approved_model,
        override_reason=args.override_reason,
        safety_margin_seconds=args.safety_margin_seconds,
    )


def _path_text(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path)


def _summary_table(result: SquadSubmissionResult) -> Table:
    table = Table(show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Status", result.status)
    table.add_row("Payload SHA-256", result.payload_sha256 or "")
    table.add_row("Attempt directory", _path_text(result.attempt_directory))
    table.add_row("Plan path", _path_text(result.submission_plan_path))
    table.add_row("Result path", _path_text(result.submission_result_path))
    return table


def _print_success(console: Console, result: SquadSubmissionResult) -> None:
    console.print(Panel(_summary_table(result), title="Submission plan ready"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    console = Console()
    error_console = Console(stderr=True)

    try:
        result = run_submission(config)
    except (ContractUnverifiedError, SquadSubmissionError) as exc:
        error_console.print(str(exc))
        return 1

    _print_success(console, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
