#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cartola.backtesting.experiment_runner import ExperimentProgressEvent, run_model_experiment


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


class _ExperimentProgressDisplay:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.progress: Progress | None = None
        self.task_id: int | None = None
        self._line_mode = not console.is_terminal

    def __enter__(self):
        if not self._line_mode:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Model experiments"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[current]}"),
                console=self.console,
                transient=False,
            )
            self.progress.start()
        return self.handle

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.progress is not None:
            self.progress.stop()

    def handle(self, event: ExperimentProgressEvent) -> None:
        if self._line_mode:
            self._handle_line_mode(event)
            return
        self._handle_progress_mode(event)

    def _handle_progress_mode(self, event: ExperimentProgressEvent) -> None:
        if self.progress is None:
            return
        if event.event_type == "experiment_started":
            self.task_id = self.progress.add_task(
                "model_experiments",
                total=event.total_children,
                completed=event.completed_children,
                current=f"output={event.output_path}",
            )
            self.progress.log(f"Experiment started: total_child_runs={event.total_children} output={event.output_path}")
            return
        if self.task_id is None:
            return
        current = _event_label(event)
        if event.event_type == "child_started":
            self.progress.update(self.task_id, completed=event.completed_children, current=f"current: {current}")
            self.progress.log(f"START {event.child_index}/{event.total_children} {current}")
            return
        if event.event_type == "child_finished":
            duration = _format_duration(event.child_duration_seconds)
            self.progress.update(
                self.task_id, completed=event.completed_children, current=f"last: {current} {duration}"
            )
            self.progress.log(f"DONE  {event.child_index}/{event.total_children} {current} duration={duration}")
            return
        if event.event_type == "child_failed":
            duration = _format_duration(event.child_duration_seconds)
            self.progress.update(self.task_id, completed=event.completed_children, current=f"failed: {current}")
            self.progress.log(
                f"FAIL  {event.child_index}/{event.total_children} {current} phase={event.phase} "
                f"duration={duration} message={event.message}"
            )
            return
        if event.event_type == "experiment_finished":
            elapsed = _format_duration(event.elapsed_seconds)
            self.progress.update(
                self.task_id,
                completed=event.completed_children,
                current=f"complete: elapsed={elapsed} output={event.output_path}",
            )

    def _handle_line_mode(self, event: ExperimentProgressEvent) -> None:
        if event.event_type == "experiment_started":
            self.console.print(f"START experiment total_child_runs={event.total_children} output={event.output_path}")
            return
        if event.event_type == "child_started":
            self.console.print(f"START child {event.child_index}/{event.total_children} {_event_label(event)}")
            return
        if event.event_type == "child_finished":
            self.console.print(
                f"DONE child {event.child_index}/{event.total_children} {_event_label(event)} "
                f"duration={_format_duration(event.child_duration_seconds)}"
            )
            return
        if event.event_type == "child_failed":
            self.console.print(
                f"FAILED child {event.child_index}/{event.total_children} {_event_label(event)} "
                f"phase={event.phase} duration={_format_duration(event.child_duration_seconds)} "
                f"message={event.message}"
            )
            return
        if event.event_type == "experiment_finished":
            self.console.print(
                f"DONE experiment completed={event.completed_children}/{event.total_children} "
                f"elapsed={_format_duration(event.elapsed_seconds)} output={event.output_path}"
            )


def _event_label(event: ExperimentProgressEvent) -> str:
    return f"season={event.season} model={event.model_id} feature_pack={event.feature_pack}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total_seconds = max(0, int(round(seconds)))
    minutes, second = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minute:02d}:{second:02d}"
    return f"{minute:d}:{second:02d}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stdout = Console()
    stderr = Console(stderr=True)
    try:
        with _ExperimentProgressDisplay(stderr) as progress_callback:
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
                progress_callback=progress_callback,
            )
    except Exception as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, experiment_id=result.experiment_id, output_path=result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
