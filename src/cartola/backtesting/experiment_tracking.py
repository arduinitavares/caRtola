from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

MetricValue = float | int | None
TrackerStatus = Literal["ok", "failed"]

HEAVY_CHILD_ARTIFACT_NAMES = {"player_predictions.csv", "selected_players.csv"}


@dataclass(frozen=True)
class TrackerWarning:
    phase: str
    message: str


class ExperimentTracker(Protocol):
    warnings: list[TrackerWarning]
    parent_run_id: str | None
    child_run_id: str | None

    def start_experiment(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None: ...

    def start_child(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None: ...

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None: ...

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None: ...

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None: ...

    def end_child(self, *, status: TrackerStatus) -> None: ...

    def end_experiment(self, *, status: TrackerStatus) -> None: ...


class _MLflowRunInfo(Protocol):
    run_id: str


class _MLflowRun(Protocol):
    info: _MLflowRunInfo


class _MLflowModule(Protocol):
    def set_tracking_uri(self, uri: str) -> None: ...

    def set_experiment(self, name: str) -> None: ...

    def start_run(self, *, run_name: str, nested: bool = False) -> _MLflowRun: ...

    def log_params(self, params: dict[str, str]) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, path: str) -> None: ...

    def end_run(self, status: str) -> None: ...


def _default_import_module(name: str) -> object:
    return importlib.import_module(name)


class NoOpExperimentTracker:
    def __init__(self) -> None:
        self.warnings: list[TrackerWarning] = []
        self.parent_run_id: str | None = None
        self.child_run_id: str | None = None

    def start_experiment(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        return None

    def start_child(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        return None

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None:
        return None

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        return None

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        return None

    def end_child(self, *, status: TrackerStatus) -> None:
        return None

    def end_experiment(self, *, status: TrackerStatus) -> None:
        return None


class InMemoryExperimentTracker(NoOpExperimentTracker):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def start_experiment(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        self.events.append(
            {
                "event": "start_experiment",
                "experiment_name": experiment_name,
                "run_name": run_name,
                "params": dict(params),
                "tags": dict(tags),
            }
        )

    def start_child(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        self.events.append(
            {
                "event": "start_child",
                "run_name": run_name,
                "params": dict(params),
                "tags": dict(tags),
            }
        )

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None:
        self.events.append({"event": "log_child_metrics", "metrics": _numeric_metrics(metrics)})

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self.events.append(
            {
                "event": "log_child_artifacts",
                "artifact_paths": [str(path) for path in artifact_paths],
            }
        )

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self.events.append(
            {
                "event": "log_parent_artifacts",
                "artifact_paths": [str(path) for path in artifact_paths],
            }
        )

    def end_child(self, *, status: TrackerStatus) -> None:
        self.events.append({"event": "end_child", "status": status})

    def end_experiment(self, *, status: TrackerStatus) -> None:
        self.events.append({"event": "end_experiment", "status": status})


class MLflowExperimentTracker(NoOpExperimentTracker):
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        import_module: Callable[[str], object] = _default_import_module,
    ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri
        self._import_module = import_module
        self._mlflow: _MLflowModule | None = None
        self._mlflow_unavailable = False
        self._parent_active = False
        self._child_active = False

    def start_experiment(
        self,
        *,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None:
            return
        try:
            if self.tracking_uri is not None:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)
            self.parent_run_id = str(run.info.run_id)
            self._parent_active = True
            mlflow.log_params(_string_values(params))
            mlflow.set_tags(_string_values(tags))
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn("start_experiment", exc)

    def start_child(
        self,
        *,
        run_name: str,
        params: Mapping[str, object],
        tags: Mapping[str, object],
    ) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._parent_active:
            return
        try:
            run = mlflow.start_run(run_name=run_name, nested=True)
            self.child_run_id = str(run.info.run_id)
            self._child_active = True
            mlflow.log_params(_string_values(params))
            mlflow.set_tags(_string_values(tags))
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn("start_child", exc)

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._child_active:
            return
        try:
            numeric_metrics = _numeric_metrics(metrics)
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn("log_child_metrics", exc)

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        if not self._child_active:
            self._warn(
                "log_child_artifacts",
                RuntimeError("child run is not active; skipping artifact logging"),
            )
            return
        self._log_artifacts(artifact_paths, skip_heavy=True, phase="log_child_artifacts")

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        if not self._parent_active:
            self._warn(
                "log_parent_artifacts",
                RuntimeError("parent run is not active; skipping artifact logging"),
            )
            return
        if self._child_active:
            self._warn(
                "log_parent_artifacts",
                RuntimeError("child run is active; skipping parent artifact logging"),
            )
            return
        self._log_artifacts(artifact_paths, skip_heavy=False, phase="log_parent_artifacts")

    def end_child(self, *, status: TrackerStatus) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._child_active:
            return
        try:
            mlflow.end_run(status=_mlflow_status(status))
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn("end_child", exc)
            return
        self._child_active = False

    def end_experiment(self, *, status: TrackerStatus) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._parent_active:
            return
        try:
            if self._child_active:
                self.end_child(status=status)
                if self._child_active:
                    self._warn(
                        "end_experiment",
                        RuntimeError("child run is still active; skipping parent close"),
                    )
                    return
            mlflow.end_run(status=_mlflow_status(status))
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn("end_experiment", exc)
            return
        self._parent_active = False

    def _log_artifacts(self, artifact_paths: Sequence[Path], *, skip_heavy: bool, phase: str) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None:
            return
        try:
            for path in artifact_paths:
                if skip_heavy and path.name in HEAVY_CHILD_ARTIFACT_NAMES:
                    continue
                if path.exists():
                    mlflow.log_artifact(str(path))
        except Exception as exc:  # pragma: no cover - behavior exercised through warnings
            self._warn(phase, exc)

    def _load_mlflow(self) -> _MLflowModule | None:
        if self._mlflow is not None:
            return self._mlflow
        if self._mlflow_unavailable:
            return None
        try:
            self._mlflow = cast("_MLflowModule", self._import_module("mlflow"))
        except Exception as exc:
            self._mlflow_unavailable = True
            self._warn("import_mlflow", exc)
            return None
        return self._mlflow

    def _warn(self, phase: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        warning = TrackerWarning(phase=phase, message=message)
        if not self.warnings or self.warnings[-1] != warning:
            self.warnings.append(warning)


def _numeric_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if value is not None}


def _string_values(values: Mapping[str, object]) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in values.items()}


def _mlflow_status(status: TrackerStatus) -> str:
    return "FINISHED" if status == "ok" else "FAILED"
