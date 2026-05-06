#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from cartola.backtesting.oracle_discovery import OracleDiscoveryProgressEvent


build_oracle_discovery_report: Callable[..., Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cartola oracle knowledge discovery report.")
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/oracle_discovery"))
    parser.add_argument("--current-year", type=int, default=None)
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _load_runtime_dependencies() -> None:
    global build_oracle_discovery_report

    if build_oracle_discovery_report is None:
        from cartola.backtesting.oracle_discovery import (
            build_oracle_discovery_report as imported_build_oracle_discovery_report,
        )

        build_oracle_discovery_report = imported_build_oracle_discovery_report


def _print_error(console: Console, error: Exception) -> None:
    console.print(Panel(str(error), title="Oracle discovery failed", border_style="red"))


def _print_success(console: Console, *, output_path: Path) -> None:
    console.print(
        Panel(
            f"output_path={output_path}",
            title="Oracle discovery complete",
            border_style="green",
        )
    )


class _OracleDiscoveryProgressDisplay:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None
        self._line_mode = not console.is_terminal

    def __enter__(self) -> Callable[[OracleDiscoveryProgressEvent], None]:
        if not self._line_mode:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]Oracle discovery"),
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

    def handle(self, event: OracleDiscoveryProgressEvent) -> None:
        if self._line_mode:
            self._handle_line_mode(event)
            return
        self._handle_progress_mode(event)

    def _handle_progress_mode(self, event: OracleDiscoveryProgressEvent) -> None:
        event_type = _event_attr(event, "event_type")
        if self.progress is None:
            return
        if event_type == "report_started":
            self.task_id = self.progress.add_task(
                "oracle_discovery",
                total=None,
                completed=0,
                current=f"output={_event_attr(event, 'output_path')}",
            )
            self.progress.log(f"Oracle discovery started: output={_event_attr(event, 'output_path')}")
            return
        if self.task_id is None:
            return
        if event_type == "work_planned":
            self.progress.update(
                self.task_id,
                total=_event_int(event, "total_rounds"),
                completed=_event_int(event, "completed_rounds"),
                current=f"planned: oracle_rounds={_event_int(event, 'total_rounds')}",
            )
            self.progress.log(f"PLAN oracle rounds={_event_int(event, 'total_rounds')}")
            return
        if event_type == "child_started":
            self.progress.update(self.task_id, current=f"current child: {_event_label(event)}")
            self.progress.log(
                f"START child {_event_int(event, 'child_index')}/{_event_int(event, 'total_children')} "
                f"{_event_label(event)}"
            )
            return
        if event_type == "strategy_started":
            self.progress.update(self.task_id, current=f"current strategy: {_event_label(event)}")
            self.progress.log(f"START strategy {_event_label(event)}")
            return
        if event_type == "round_finished":
            self.progress.update(
                self.task_id,
                completed=_event_int(event, "completed_rounds"),
                current=f"last round: {_event_label(event)}",
            )
            message = _event_attr(event, "message")
            suffix = f" message={message}" if message else ""
            self.progress.log(
                f"DONE round {_event_int(event, 'completed_rounds')}/{_event_int(event, 'total_rounds')} "
                f"{_event_label(event)}{suffix}"
            )
            return
        if event_type == "strategy_finished":
            self.progress.update(self.task_id, current=f"finished strategy: {_event_label(event)}")
            self.progress.log(f"DONE strategy {_event_label(event)}")
            return
        if event_type == "child_finished":
            self.progress.update(self.task_id, current=f"finished child: {_event_label(event)}")
            self.progress.log(
                f"DONE child {_event_int(event, 'child_index')}/{_event_int(event, 'total_children')} "
                f"{_event_label(event)}"
            )
            return
        if event_type == "report_finished":
            self.progress.update(
                self.task_id,
                total=_event_int(event, "total_rounds"),
                completed=_event_int(event, "completed_rounds"),
                current=(
                    f"complete: elapsed={_format_duration(_event_float(event, 'elapsed_seconds'))} "
                    f"output={_event_attr(event, 'output_path')}"
                ),
            )

    def _handle_line_mode(self, event: OracleDiscoveryProgressEvent) -> None:
        event_type = _event_attr(event, "event_type")
        if event_type == "report_started":
            self.console.print(f"START oracle discovery output={_event_attr(event, 'output_path')}")
            return
        if event_type == "work_planned":
            self.console.print(f"PLAN oracle rounds={_event_int(event, 'total_rounds')}")
            return
        if event_type == "child_started":
            self.console.print(
                f"START child {_event_int(event, 'child_index')}/{_event_int(event, 'total_children')} "
                f"{_event_label(event)}"
            )
            return
        if event_type == "strategy_started":
            self.console.print(f"START strategy {_event_label(event)}")
            return
        if event_type == "round_finished":
            message = _event_attr(event, "message")
            suffix = f" message={message}" if message else ""
            self.console.print(
                f"DONE round {_event_int(event, 'completed_rounds')}/{_event_int(event, 'total_rounds')} "
                f"{_event_label(event)}{suffix}"
            )
            return
        if event_type == "strategy_finished":
            self.console.print(f"DONE strategy {_event_label(event)}")
            return
        if event_type == "child_finished":
            self.console.print(
                f"DONE child {_event_int(event, 'child_index')}/{_event_int(event, 'total_children')} "
                f"{_event_label(event)}"
            )
            return
        if event_type == "report_finished":
            self.console.print(
                f"DONE oracle discovery completed={_event_int(event, 'completed_rounds')}/"
                f"{_event_int(event, 'total_rounds')} "
                f"elapsed={_format_duration(_event_float(event, 'elapsed_seconds'))} "
                f"output={_event_attr(event, 'output_path')}"
            )


def _event_attr(event: object, name: str) -> object | None:
    return getattr(event, name, None)


def _event_int(event: object, name: str) -> int:
    value = _event_attr(event, name)
    if value is None:
        return 0
    return int(value)


def _event_float(event: object, name: str) -> float | None:
    value = _event_attr(event, name)
    if value is None:
        return None
    return float(value)


def _event_label(event: object) -> str:
    parts: list[str] = []
    for event_field, label in (
        ("season", "season"),
        ("strategy", "strategy"),
        ("model_id", "model"),
        ("feature_pack", "feature_pack"),
        ("round_number", "round"),
    ):
        value = _event_attr(event, event_field)
        if value is not None:
            parts.append(f"{label}={value}")
    return " ".join(parts) if parts else "n/a"


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
    output_path = args.output_root / f"oracle_discovery_started_at={_timestamp()}"

    try:
        _load_runtime_dependencies()
        if build_oracle_discovery_report is None:
            raise RuntimeError("Oracle discovery runtime dependencies were not loaded.")
        with _OracleDiscoveryProgressDisplay(stderr) as progress_callback:
            build_oracle_discovery_report(
                experiment_path=args.experiment_path,
                output_path=output_path,
                progress_callback=progress_callback,
            )
    except Exception as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
