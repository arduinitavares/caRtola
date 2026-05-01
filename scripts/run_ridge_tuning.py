#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cartola.backtesting.ridge_tuning_runner import RidgeTuningProgressEvent, run_ridge_tuning


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cartola constrained Ridge tuning.")
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--current-year", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/experiments/model_tuning"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-final-rerun", action="store_true")
    return parser.parse_args(argv)


def _parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seasons:
        raise ValueError("At least one season is required")
    return seasons


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _print_error(console: Console, error: Exception) -> None:
    console.print(Panel(str(error), title="Ridge tuning failed", border_style="red"))


def _print_success(console: Console, *, experiment_id: str, output_path: Path) -> None:
    console.print(
        Panel(
            f"experiment_id={experiment_id}\noutput_path={output_path}",
            title="Ridge tuning complete",
            border_style="green",
        )
    )


class _RidgeTuningProgressDisplay:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None
        self._line_mode = not console.is_terminal

    def __enter__(self) -> Callable[[RidgeTuningProgressEvent], None]:
        if not self._line_mode:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Ridge tuning"),
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.progress is not None:
            self.progress.stop()

    def handle(self, event: RidgeTuningProgressEvent) -> None:
        if self._line_mode:
            self._handle_line_mode(event)
            return
        self._handle_progress_mode(event)

    def _handle_progress_mode(self, event: RidgeTuningProgressEvent) -> None:
        if self.progress is None:
            return
        if event.event_type == "experiment_started":
            self.task_id = self.progress.add_task(
                "ridge_tuning",
                total=event.total_children,
                completed=event.completed_children,
                current=f"stage={event.phase} output={event.output_path}",
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

    def _handle_line_mode(self, event: RidgeTuningProgressEvent) -> None:
        if event.event_type == "experiment_started":
            self.console.print(
                f"START experiment stage={event.phase} total_child_runs={event.total_children} "
                f"output={event.output_path}"
            )
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


def _event_label(event: RidgeTuningProgressEvent) -> str:
    return (
        f"stage={event.stage} season={event.season} candidate_id={event.candidate_id} "
        f"feature_pack={event.feature_pack} alpha={event.alpha}"
    )


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
        with _RidgeTuningProgressDisplay(stderr) as progress_callback:
            result = run_ridge_tuning(
                seasons=_parse_seasons(args.seasons),
                start_round=args.start_round,
                budget=args.budget,
                current_year=args.current_year,
                jobs=args.jobs,
                project_root=args.project_root,
                output_root=args.output_root,
                started_at_utc=_timestamp(),
                progress_callback=progress_callback,
                skip_final_rerun=args.skip_final_rerun,
            )
    except Exception as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, experiment_id=result.experiment_id, output_path=result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
