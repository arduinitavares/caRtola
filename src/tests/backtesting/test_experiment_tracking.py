from __future__ import annotations

from pathlib import Path

from cartola.backtesting.experiment_tracking import (
    InMemoryExperimentTracker,
    MLflowExperimentTracker,
    NoOpExperimentTracker,
    TrackerWarning,
)


def test_noop_tracker_accepts_all_calls_and_has_empty_warnings(tmp_path: Path) -> None:
    tracker = NoOpExperimentTracker()

    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={},
        tags={},
    )
    tracker.start_child(
        run_name="child",
        params={"model_id": "random_forest"},
        tags={"partition": "season=2025"},
    )
    tracker.log_child_metrics(
        {
            "squad/actual_points_total": 10.0,
            "prediction/candidate_pool/mae": None,
        }
    )
    tracker.log_child_artifacts([tmp_path / "missing.csv"])
    tracker.log_parent_artifacts([tmp_path / "missing-summary.csv"])
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert tracker.warnings == []


def test_in_memory_tracker_records_sequence_and_filters_none_metrics() -> None:
    tracker = InMemoryExperimentTracker()

    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={"group": "production-parity"},
        tags={},
    )
    tracker.start_child(
        run_name="child",
        params={"season": 2025},
        tags={"comparability_partition": "season=2025"},
    )
    tracker.log_child_metrics(
        {
            "squad/actual_points_total": 10.0,
            "prediction/candidate_pool/mae": None,
        }
    )
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert [event["event"] for event in tracker.events] == [
        "start_experiment",
        "start_child",
        "log_child_metrics",
        "end_child",
        "end_experiment",
    ]
    assert tracker.events[2]["metrics"] == {"squad/actual_points_total": 10.0}


def test_mlflow_tracker_degrades_when_import_fails() -> None:
    tracker = MLflowExperimentTracker(
        tracking_uri=None,
        import_module=lambda _name: (_ for _ in ()).throw(ImportError("no mlflow")),
    )

    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={},
        tags={},
    )
    tracker.start_child(run_name="child", params={}, tags={})
    tracker.log_child_metrics({"squad/actual_points_total": 10.0})
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert tracker.parent_run_id is None
    assert tracker.warnings
    assert isinstance(tracker.warnings[0], TrackerWarning)
    assert "no mlflow" in tracker.warnings[0].message


class _FakeRunInfo:
    run_id: str

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class _FakeRun:
    info: _FakeRunInfo

    def __init__(self, run_id: str) -> None:
        self.info = _FakeRunInfo(run_id)


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self._next_run_id = 0
        self.end_run_error: Exception | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    def start_run(self, *, run_name: str, nested: bool = False) -> _FakeRun:
        self._next_run_id += 1
        run = _FakeRun(f"run-{self._next_run_id}")
        self.calls.append(
            (
                "start_run",
                {
                    "run_name": run_name,
                    "nested": nested,
                    "run_id": run.info.run_id,
                },
            )
        )
        return run

    def log_params(self, params: dict[str, str]) -> None:
        self.calls.append(("log_params", params))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", tags))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.calls.append(("log_metrics", metrics))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("log_artifact", path))

    def end_run(self, status: str) -> None:
        if self.end_run_error is not None:
            raise self.end_run_error
        self.calls.append(("end_run", status))


def test_mlflow_tracker_logs_parent_child_metrics_and_small_artifacts(
    tmp_path: Path,
) -> None:
    fake_mlflow = _FakeMlflow()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("strategy,total\n", encoding="utf-8")
    parent_artifact = tmp_path / "experiment_summary.md"
    parent_artifact.write_text("# Summary\n", encoding="utf-8")
    heavy_artifact = tmp_path / "player_predictions.csv"
    heavy_artifact.write_text("large\n", encoding="utf-8")
    selected_players = tmp_path / "selected_players.csv"
    selected_players.write_text("also-large\n", encoding="utf-8")

    tracker = MLflowExperimentTracker(
        tracking_uri="file:///tmp/mlruns",
        import_module=lambda _name: fake_mlflow,
    )
    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={"group": "production-parity"},
        tags={"experiment_id": "exp"},
    )
    tracker.start_child(
        run_name="season=2025 model=random_forest feature_pack=ppg",
        params={"season": 2025},
        tags={"comparability_partition": "season=2025"},
    )
    tracker.log_child_metrics(
        {
            "squad/actual_points_total": 10.0,
            "prediction/candidate_pool/mae": None,
        }
    )
    tracker.log_child_artifacts([artifact, heavy_artifact, selected_players])
    tracker.end_child(status="ok")
    tracker.log_parent_artifacts([parent_artifact, heavy_artifact])
    tracker.end_experiment(status="failed")

    assert tracker.parent_run_id == "run-1"
    assert tracker.child_run_id == "run-2"
    assert ("set_tracking_uri", "file:///tmp/mlruns") in fake_mlflow.calls
    assert (
        "start_run",
        {"run_name": "exp", "nested": False, "run_id": "run-1"},
    ) in fake_mlflow.calls
    assert (
        "start_run",
        {
            "run_name": "season=2025 model=random_forest feature_pack=ppg",
            "nested": True,
            "run_id": "run-2",
        },
    ) in fake_mlflow.calls
    assert ("log_params", {"group": "production-parity"}) in fake_mlflow.calls
    assert ("set_tags", {"experiment_id": "exp"}) in fake_mlflow.calls
    assert ("log_metrics", {"squad/actual_points_total": 10.0}) in fake_mlflow.calls
    assert ("log_artifact", str(artifact)) in fake_mlflow.calls
    assert ("log_artifact", str(parent_artifact)) in fake_mlflow.calls
    assert ("log_artifact", str(heavy_artifact)) in fake_mlflow.calls
    assert fake_mlflow.calls.count(("log_artifact", str(heavy_artifact))) == 1
    assert ("log_artifact", str(selected_players)) not in fake_mlflow.calls
    assert ("end_run", "FINISHED") in fake_mlflow.calls
    assert ("end_run", "FAILED") in fake_mlflow.calls


def test_mlflow_tracker_does_not_log_child_artifacts_after_child_ends(
    tmp_path: Path,
) -> None:
    fake_mlflow = _FakeMlflow()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("strategy,total\n", encoding="utf-8")
    tracker = MLflowExperimentTracker(
        tracking_uri=None,
        import_module=lambda _name: fake_mlflow,
    )

    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={},
        tags={},
    )
    tracker.start_child(run_name="child", params={}, tags={})
    tracker.end_child(status="ok")
    tracker.log_child_artifacts([artifact])

    assert ("log_artifact", str(artifact)) not in fake_mlflow.calls
    assert tracker.warnings[-1].phase == "log_child_artifacts"


def test_mlflow_tracker_does_not_log_parent_artifacts_while_child_is_active(
    tmp_path: Path,
) -> None:
    fake_mlflow = _FakeMlflow()
    artifact = tmp_path / "experiment_summary.md"
    artifact.write_text("# Summary\n", encoding="utf-8")
    tracker = MLflowExperimentTracker(
        tracking_uri=None,
        import_module=lambda _name: fake_mlflow,
    )

    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={},
        tags={},
    )
    tracker.start_child(run_name="child", params={}, tags={})
    tracker.log_parent_artifacts([artifact])

    assert ("log_artifact", str(artifact)) not in fake_mlflow.calls
    assert tracker.warnings[-1].phase == "log_parent_artifacts"


def test_mlflow_tracker_preserves_child_state_when_child_close_fails() -> None:
    fake_mlflow = _FakeMlflow()
    tracker = MLflowExperimentTracker(
        tracking_uri=None,
        import_module=lambda _name: fake_mlflow,
    )
    tracker.start_experiment(
        experiment_name="cartola-production-parity",
        run_name="exp",
        params={},
        tags={},
    )
    tracker.start_child(run_name="child", params={}, tags={})

    fake_mlflow.end_run_error = RuntimeError("cannot close child")
    tracker.end_child(status="failed")

    assert tracker._child_active is True
    assert tracker._parent_active is True
    assert tracker.warnings[-1].phase == "end_child"
    tracker.end_experiment(status="failed")

    assert ("end_run", "FAILED") not in fake_mlflow.calls
    assert tracker._child_active is True
    assert tracker._parent_active is True
    assert tracker.warnings[-1].phase == "end_experiment"

    fake_mlflow.end_run_error = None
    tracker.end_experiment(status="failed")

    assert fake_mlflow.calls.count(("end_run", "FAILED")) == 2
    assert tracker._child_active is False
    assert tracker._parent_active is False
