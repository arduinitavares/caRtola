# Experiment Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable SQLite experiment index and optional best-effort MLflow tracking to `scripts/run_model_experiments.py` without changing backtest science, optimizer behavior, model behavior, or live workflows.

**Architecture:** Treat the current CSV/JSON/Markdown/HTML reports as authoritative. Add `experiment_index.py` as a local SQLite ledger, `experiment_tracking.py` as a no-throw observer interface, and wire both into `experiment_runner.py` after child artifacts are written. Keep MLflow optional and lazily imported.

**Tech Stack:** Python 3.13, stdlib `sqlite3`, pandas, current experiment runner/report objects, optional lazy `mlflow`, pytest, Ruff/ty/Bandit through `scripts/pyrepo-check`.

---

## Preflight

Run implementation in a dedicated worktree so the active long experiment in the main checkout is not disturbed:

```bash
cd /Users/aaat/projects/caRtola
git worktree add ../caRtola-experiment-observability -b dev/experiment-observability
cd ../caRtola-experiment-observability
```

Expected: a clean worktree on branch `dev/experiment-observability`.

Do not edit `data/08_reporting/experiments/model_feature/...` outputs from the running experiment. New tests should use `tmp_path`.

## File Structure

- Create `src/cartola/backtesting/experiment_index.py`
  - Owns SQLite schema, WAL/busy-timeout initialization, schema version, upserts, source/lock hashing helpers, and artifact pointer JSON generation.
- Create `src/cartola/backtesting/experiment_tracking.py`
  - Owns `ExperimentTracker`, `NoOpExperimentTracker`, `InMemoryExperimentTracker`, and optional `MLflowExperimentTracker`.
- Modify `src/cartola/backtesting/experiment_runner.py`
  - Accepts an optional tracker and index writer.
  - Writes experiment rows and child rows at the right lifecycle points.
  - Calls tracker methods in no-throw mode and finalizes tracker in `finally`.
- Modify `scripts/run_model_experiments.py`
  - Adds `--tracker none|mlflow` and `--mlflow-tracking-uri`.
  - Creates the tracker at the CLI boundary.
- Modify `.gitignore`
  - Ignore root `mlruns/`.
- Add tests:
  - `src/tests/backtesting/test_experiment_index.py`
  - `src/tests/backtesting/test_experiment_tracking.py`
  - Extend `src/tests/backtesting/test_experiment_runner.py`
  - Extend `src/tests/backtesting/test_run_model_experiments_cli.py`

## Task 1: SQLite Experiment Index

**Files:**
- Create: `src/cartola/backtesting/experiment_index.py`
- Test: `src/tests/backtesting/test_experiment_index.py`

- [ ] **Step 1: Write failing tests for schema initialization, WAL, user version, and upserts**

Create `src/tests/backtesting/test_experiment_index.py`:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cartola.backtesting.experiment_index import (
    SCHEMA_VERSION,
    ExperimentIndex,
    artifact_pointer_payload,
    sha256_json,
    sha256_optional_file,
    source_hash_summary,
)


def test_index_initializes_schema_wal_timeout_and_version(tmp_path: Path) -> None:
    db_path = tmp_path / "experiment_index.sqlite"

    index = ExperimentIndex(db_path)
    index.initialize()

    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert journal_mode == "wal"
    assert busy_timeout == 5000
    assert user_version == SCHEMA_VERSION
    assert {"experiments", "child_runs"}.issubset(tables)


def test_experiment_and_child_upserts_are_keyed(tmp_path: Path) -> None:
    index = ExperimentIndex(tmp_path / "experiment_index.sqlite")
    index.initialize()

    index.upsert_experiment(
        {
            "experiment_id": "exp-1",
            "group": "production-parity",
            "started_at_utc": "20260501T120000000000Z",
            "finished_at_utc": None,
            "status": "running",
            "output_path": "data/08_reporting/experiments/model_feature/exp-1",
            "matrix_hash": "matrix",
            "seasons": json.dumps([2025]),
            "start_round": 5,
            "budget": 100.0,
            "current_year": 2026,
            "jobs": 12,
            "scoring_contract_version": "cartola_standard_2026_v1",
            "git_commit": "abc123",
            "git_branch": "dev/test",
            "git_dirty": 0,
            "python_version": "3.13.12",
            "uv_lock_hash": None,
            "mlflow_enabled": 0,
            "mlflow_status": "disabled",
            "mlflow_parent_run_id": None,
            "warning_count": 0,
            "child_run_count": 8,
            "completed_child_run_count": 0,
            "failed_child_run_count": 0,
        }
    )
    index.upsert_experiment(
        {
            "experiment_id": "exp-1",
            "group": "production-parity",
            "started_at_utc": "20260501T120000000000Z",
            "finished_at_utc": "20260501T121000000000Z",
            "status": "ok",
            "output_path": "data/08_reporting/experiments/model_feature/exp-1",
            "matrix_hash": "matrix",
            "seasons": json.dumps([2025]),
            "start_round": 5,
            "budget": 100.0,
            "current_year": 2026,
            "jobs": 12,
            "scoring_contract_version": "cartola_standard_2026_v1",
            "git_commit": "abc123",
            "git_branch": "dev/test",
            "git_dirty": 0,
            "python_version": "3.13.12",
            "uv_lock_hash": None,
            "mlflow_enabled": 0,
            "mlflow_status": "disabled",
            "mlflow_parent_run_id": None,
            "warning_count": 0,
            "child_run_count": 8,
            "completed_child_run_count": 8,
            "failed_child_run_count": 0,
        }
    )
    index.upsert_child_run(
        {
            "experiment_id": "exp-1",
            "child_run_id": "season=2025/model=random_forest/feature_pack=ppg",
            "season": 2025,
            "model_id": "random_forest",
            "feature_pack": "ppg",
            "fixture_mode": "none",
            "footystats_mode": "ppg",
            "matchup_context_mode": "none",
            "output_path": "data/08_reporting/experiments/model_feature/exp-1/runs/season=2025/model=random_forest/feature_pack=ppg",
            "status": "ok",
            "wall_clock_seconds": 10.5,
            "backtest_jobs": 12,
            "backtest_workers_effective": 12,
            "model_n_jobs_effective": 1,
            "total_actual_points": 100.0,
            "avg_actual_points": 50.0,
            "total_predicted_points": 98.0,
            "prediction_mae": 1.5,
            "prediction_rmse": 2.5,
            "prediction_r2": 0.1,
            "prediction_pearson": 0.2,
            "prediction_spearman": 0.3,
            "selected_calibration_slope": 1.0,
            "top50_spearman": 0.4,
            "optimal_round_count": 2,
            "skipped_round_count": 0,
            "candidate_pool_signature_hash": "cand",
            "solver_status_signature_hash": "solver",
            "comparability_partition": "season=2025",
            "comparable_within_partition": 1,
            "ineligibility_reason": None,
            "source_hash_summary": "source",
            "mlflow_child_run_id": None,
        }
    )

    with sqlite3.connect(index.path) as connection:
        experiments = connection.execute("SELECT status, completed_child_run_count FROM experiments").fetchall()
        children = connection.execute("SELECT child_run_id, total_actual_points FROM child_runs").fetchall()

    assert experiments == [("ok", 8)]
    assert children == [("season=2025/model=random_forest/feature_pack=ppg", 100.0)]


def test_hash_helpers_and_artifact_pointer_payload(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    player_predictions = tmp_path / "player_predictions.csv"
    selected_players = tmp_path / "selected_players.csv"
    player_predictions.write_text("a,b\n1,2\n", encoding="utf-8")
    selected_players.write_text("a,b\n3,4\n", encoding="utf-8")

    raw_identity = {"season": 2025, "files": [{"path": "data/01_raw/2025/rodada-1.csv", "sha256": "abc"}]}
    payload = artifact_pointer_payload(
        project_root=tmp_path,
        child_run_id="season=2025/model=random_forest/feature_pack=ppg",
        output_path=tmp_path,
        artifact_paths=[player_predictions, selected_players, tmp_path / "missing.csv"],
    )

    assert sha256_optional_file(lock) is not None
    assert sha256_optional_file(tmp_path / "missing.lock") is None
    assert source_hash_summary(raw_identity) == sha256_json(raw_identity)
    assert payload["child_run_id"] == "season=2025/model=random_forest/feature_pack=ppg"
    assert payload["output_path"] == "."
    assert sorted(payload["artifacts"]) == ["player_predictions.csv", "selected_players.csv"]
    assert payload["artifacts"]["player_predictions.csv"]["size_bytes"] == player_predictions.stat().st_size
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_index.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'cartola.backtesting.experiment_index'`.

- [ ] **Step 3: Implement `experiment_index.py`**

Create `src/cartola/backtesting/experiment_index.py`:

```python
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000

EXPERIMENT_COLUMNS = (
    "experiment_id",
    "group",
    "started_at_utc",
    "finished_at_utc",
    "status",
    "output_path",
    "matrix_hash",
    "seasons",
    "start_round",
    "budget",
    "current_year",
    "jobs",
    "scoring_contract_version",
    "git_commit",
    "git_branch",
    "git_dirty",
    "python_version",
    "uv_lock_hash",
    "mlflow_enabled",
    "mlflow_status",
    "mlflow_parent_run_id",
    "warning_count",
    "child_run_count",
    "completed_child_run_count",
    "failed_child_run_count",
)

CHILD_RUN_COLUMNS = (
    "experiment_id",
    "child_run_id",
    "season",
    "model_id",
    "feature_pack",
    "fixture_mode",
    "footystats_mode",
    "matchup_context_mode",
    "output_path",
    "status",
    "wall_clock_seconds",
    "backtest_jobs",
    "backtest_workers_effective",
    "model_n_jobs_effective",
    "total_actual_points",
    "avg_actual_points",
    "total_predicted_points",
    "prediction_mae",
    "prediction_rmse",
    "prediction_r2",
    "prediction_pearson",
    "prediction_spearman",
    "selected_calibration_slope",
    "top50_spearman",
    "optimal_round_count",
    "skipped_round_count",
    "candidate_pool_signature_hash",
    "solver_status_signature_hash",
    "comparability_partition",
    "comparable_within_partition",
    "ineligibility_reason",
    "source_hash_summary",
    "mlflow_child_run_id",
)


class ExperimentIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported experiment index schema version: {user_version}")
            _create_schema(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def upsert_experiment(self, row: Mapping[str, object]) -> None:
        self._upsert("experiments", EXPERIMENT_COLUMNS, row)

    def upsert_child_run(self, row: Mapping[str, object]) -> None:
        self._upsert("child_runs", CHILD_RUN_COLUMNS, row)

    def _upsert(self, table: str, columns: Sequence[str], row: Mapping[str, object]) -> None:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"Missing {table} columns: {', '.join(missing)}")
        placeholders = ", ".join("?" for _column in columns)
        column_sql = ", ".join(columns)
        if table == "experiments":
            conflict_target = "experiment_id"
        elif table == "child_runs":
            conflict_target = "experiment_id, child_run_id"
        else:
            raise ValueError(f"Unsupported table: {table}")
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in conflict_target.split(", "))
        values = [_sqlite_value(row[column]) for column in columns]
        sql = (
            f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict_target}) DO UPDATE SET {updates}"
        )
        with self._connect() as connection:
            connection.execute(sql, values)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            "group" TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            status TEXT NOT NULL,
            output_path TEXT NOT NULL,
            matrix_hash TEXT NOT NULL,
            seasons TEXT NOT NULL,
            start_round INTEGER NOT NULL,
            budget REAL NOT NULL,
            current_year INTEGER NOT NULL,
            jobs INTEGER NOT NULL,
            scoring_contract_version TEXT NOT NULL,
            git_commit TEXT,
            git_branch TEXT,
            git_dirty INTEGER,
            python_version TEXT,
            uv_lock_hash TEXT,
            mlflow_enabled INTEGER NOT NULL,
            mlflow_status TEXT NOT NULL,
            mlflow_parent_run_id TEXT,
            warning_count INTEGER NOT NULL,
            child_run_count INTEGER NOT NULL,
            completed_child_run_count INTEGER NOT NULL,
            failed_child_run_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS child_runs (
            experiment_id TEXT NOT NULL,
            child_run_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            feature_pack TEXT NOT NULL,
            fixture_mode TEXT NOT NULL,
            footystats_mode TEXT NOT NULL,
            matchup_context_mode TEXT NOT NULL,
            output_path TEXT NOT NULL,
            status TEXT NOT NULL,
            wall_clock_seconds REAL,
            backtest_jobs INTEGER NOT NULL,
            backtest_workers_effective INTEGER,
            model_n_jobs_effective INTEGER,
            total_actual_points REAL,
            avg_actual_points REAL,
            total_predicted_points REAL,
            prediction_mae REAL,
            prediction_rmse REAL,
            prediction_r2 REAL,
            prediction_pearson REAL,
            prediction_spearman REAL,
            selected_calibration_slope REAL,
            top50_spearman REAL,
            optimal_round_count INTEGER,
            skipped_round_count INTEGER,
            candidate_pool_signature_hash TEXT,
            solver_status_signature_hash TEXT,
            comparability_partition TEXT NOT NULL,
            comparable_within_partition INTEGER NOT NULL,
            ineligibility_reason TEXT,
            source_hash_summary TEXT,
            mlflow_child_run_id TEXT,
            PRIMARY KEY (experiment_id, child_run_id)
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_child_runs_model_feature ON child_runs(model_id, feature_pack)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_child_runs_partition ON child_runs(comparability_partition)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_optional_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hash_summary(raw_source_identity: Mapping[str, object]) -> str:
    return sha256_json(raw_source_identity)


def artifact_pointer_payload(
    *,
    project_root: Path,
    child_run_id: str,
    output_path: Path,
    artifact_paths: Sequence[Path],
) -> dict[str, object]:
    project_root = project_root.resolve()
    return {
        "child_run_id": child_run_id,
        "output_path": _display_path(output_path, project_root=project_root),
        "artifacts": {
            path.name: {
                "path": _display_path(path, project_root=project_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
            if path.exists()
        },
    }


def _display_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
```

- [ ] **Step 4: Run the index tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_index.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/cartola/backtesting/experiment_index.py src/tests/backtesting/test_experiment_index.py
git commit -m "feat: add experiment index store"
```

## Task 2: Tracker Interface And Optional MLflow Adapter

**Files:**
- Create: `src/cartola/backtesting/experiment_tracking.py`
- Test: `src/tests/backtesting/test_experiment_tracking.py`

- [ ] **Step 1: Write failing tracker tests**

Create `src/tests/backtesting/test_experiment_tracking.py`:

```python
from __future__ import annotations

from pathlib import Path

from cartola.backtesting.experiment_tracking import (
    InMemoryExperimentTracker,
    MLflowExperimentTracker,
    NoOpExperimentTracker,
    TrackerWarning,
)


def test_noop_tracker_accepts_all_calls(tmp_path: Path) -> None:
    tracker = NoOpExperimentTracker()

    tracker.start_experiment(experiment_name="cartola-production-parity", run_name="exp", params={}, tags={})
    tracker.start_child(run_name="child", params={"model_id": "random_forest"}, tags={"partition": "season=2025"})
    tracker.log_child_metrics({"squad/actual_points_total": 10.0, "prediction/candidate_pool/mae": None})
    tracker.log_child_artifacts([tmp_path / "missing.csv"])
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert tracker.warnings == []


def test_in_memory_tracker_records_sequence() -> None:
    tracker = InMemoryExperimentTracker()

    tracker.start_experiment(experiment_name="cartola-production-parity", run_name="exp", params={"group": "production-parity"}, tags={})
    tracker.start_child(run_name="child", params={"season": 2025}, tags={"comparability_partition": "season=2025"})
    tracker.log_child_metrics({"squad/actual_points_total": 10.0, "prediction/candidate_pool/mae": None})
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
    tracker = MLflowExperimentTracker(tracking_uri=None, import_module=lambda _name: (_ for _ in ()).throw(ImportError("no mlflow")))

    tracker.start_experiment(experiment_name="cartola-production-parity", run_name="exp", params={}, tags={})
    tracker.start_child(run_name="child", params={}, tags={})
    tracker.log_child_metrics({"squad/actual_points_total": 10.0})
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert tracker.parent_run_id is None
    assert tracker.warnings
    assert isinstance(tracker.warnings[0], TrackerWarning)
    assert "no mlflow" in tracker.warnings[0].message


class _FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = _FakeRunInfo(run_id)


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self._next_run_id = 0

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    def start_run(self, *, run_name: str, nested: bool = False) -> _FakeRun:
        self._next_run_id += 1
        run = _FakeRun(f"run-{self._next_run_id}")
        self.calls.append(("start_run", {"run_name": run_name, "nested": nested, "run_id": run.info.run_id}))
        return run

    def log_params(self, params: dict[str, object]) -> None:
        self.calls.append(("log_params", params))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", tags))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.calls.append(("log_metrics", metrics))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("log_artifact", path))

    def end_run(self, status: str) -> None:
        self.calls.append(("end_run", status))


def test_mlflow_tracker_logs_parent_child_metrics_and_small_artifacts(tmp_path: Path) -> None:
    fake_mlflow = _FakeMlflow()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("strategy,total\n", encoding="utf-8")
    heavy_artifact = tmp_path / "player_predictions.csv"
    heavy_artifact.write_text("large\n", encoding="utf-8")

    tracker = MLflowExperimentTracker(tracking_uri="file:///tmp/mlruns", import_module=lambda _name: fake_mlflow)
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
    tracker.log_child_metrics({"squad/actual_points_total": 10.0, "prediction/candidate_pool/mae": None})
    tracker.log_child_artifacts([artifact, heavy_artifact])
    tracker.end_child(status="ok")
    tracker.end_experiment(status="ok")

    assert tracker.parent_run_id == "run-1"
    assert tracker.child_run_id == "run-2"
    assert ("set_tracking_uri", "file:///tmp/mlruns") in fake_mlflow.calls
    assert ("log_metrics", {"squad/actual_points_total": 10.0}) in fake_mlflow.calls
    assert ("log_artifact", str(artifact)) in fake_mlflow.calls
    assert ("log_artifact", str(heavy_artifact)) not in fake_mlflow.calls
    assert ("end_run", "FINISHED") in fake_mlflow.calls
```

- [ ] **Step 2: Run the tracker tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_tracking.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'cartola.backtesting.experiment_tracking'`.

- [ ] **Step 3: Implement `experiment_tracking.py`**

Create `src/cartola/backtesting/experiment_tracking.py`:

```python
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol

MetricValue = float | int | None
TrackerStatus = Literal["ok", "failed"]

HEAVY_ARTIFACT_NAMES = {"player_predictions.csv", "selected_players.csv"}


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

    def start_child(self, *, run_name: str, params: Mapping[str, object], tags: Mapping[str, object]) -> None: ...

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None: ...

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None: ...

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None: ...

    def end_child(self, *, status: TrackerStatus) -> None: ...

    def end_experiment(self, *, status: TrackerStatus) -> None: ...


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

    def start_child(self, *, run_name: str, params: Mapping[str, object], tags: Mapping[str, object]) -> None:
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

    def start_child(self, *, run_name: str, params: Mapping[str, object], tags: Mapping[str, object]) -> None:
        self.events.append({"event": "start_child", "run_name": run_name, "params": dict(params), "tags": dict(tags)})

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None:
        self.events.append({"event": "log_child_metrics", "metrics": _numeric_metrics(metrics)})

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self.events.append({"event": "log_child_artifacts", "artifact_paths": [str(path) for path in artifact_paths]})

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self.events.append({"event": "log_parent_artifacts", "artifact_paths": [str(path) for path in artifact_paths]})

    def end_child(self, *, status: TrackerStatus) -> None:
        self.events.append({"event": "end_child", "status": status})

    def end_experiment(self, *, status: TrackerStatus) -> None:
        self.events.append({"event": "end_experiment", "status": status})


class MLflowExperimentTracker(NoOpExperimentTracker):
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        import_module: Callable[[str], ModuleType | object] = importlib.import_module,
    ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri
        self._import_module = import_module
        self._mlflow: object | None = None
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
            mlflow.log_params(_string_params(params))
            mlflow.set_tags(_string_tags(tags))
        except Exception as exc:
            self._warn("start_experiment", exc)

    def start_child(self, *, run_name: str, params: Mapping[str, object], tags: Mapping[str, object]) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._parent_active:
            return
        try:
            run = mlflow.start_run(run_name=run_name, nested=True)
            self.child_run_id = str(run.info.run_id)
            self._child_active = True
            mlflow.log_params(_string_params(params))
            mlflow.set_tags(_string_tags(tags))
        except Exception as exc:
            self._warn("start_child", exc)

    def log_child_metrics(self, metrics: Mapping[str, MetricValue]) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._child_active:
            return
        try:
            numeric_metrics = _numeric_metrics(metrics)
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
        except Exception as exc:
            self._warn("log_child_metrics", exc)

    def log_child_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self._log_artifacts(artifact_paths, skip_heavy=True, phase="log_child_artifacts")

    def log_parent_artifacts(self, artifact_paths: Sequence[Path]) -> None:
        self._log_artifacts(artifact_paths, skip_heavy=False, phase="log_parent_artifacts")

    def end_child(self, *, status: TrackerStatus) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._child_active:
            return
        try:
            mlflow.end_run(status=_mlflow_status(status))
        except Exception as exc:
            self._warn("end_child", exc)
        finally:
            self._child_active = False

    def end_experiment(self, *, status: TrackerStatus) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None or not self._parent_active:
            return
        try:
            if self._child_active:
                self.end_child(status=status)
            mlflow.end_run(status=_mlflow_status(status))
        except Exception as exc:
            self._warn("end_experiment", exc)
        finally:
            self._parent_active = False

    def _log_artifacts(self, artifact_paths: Sequence[Path], *, skip_heavy: bool, phase: str) -> None:
        mlflow = self._load_mlflow()
        if mlflow is None:
            return
        try:
            for path in artifact_paths:
                if skip_heavy and path.name in HEAVY_ARTIFACT_NAMES:
                    continue
                if path.exists():
                    mlflow.log_artifact(str(path))
        except Exception as exc:
            self._warn(phase, exc)

    def _load_mlflow(self) -> object | None:
        if self._mlflow is not None:
            return self._mlflow
        try:
            self._mlflow = self._import_module("mlflow")
        except Exception as exc:
            self._warn("import_mlflow", exc)
            return None
        return self._mlflow

    def _warn(self, phase: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if not self.warnings or self.warnings[-1] != TrackerWarning(phase=phase, message=message):
            self.warnings.append(TrackerWarning(phase=phase, message=message))


def _numeric_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if value is not None}


def _string_params(params: Mapping[str, object]) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in params.items()}


def _string_tags(tags: Mapping[str, object]) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in tags.items()}


def _mlflow_status(status: TrackerStatus) -> str:
    return "FINISHED" if status == "ok" else "FAILED"
```

- [ ] **Step 4: Run tracker tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_tracking.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/cartola/backtesting/experiment_tracking.py src/tests/backtesting/test_experiment_tracking.py
git commit -m "feat: add experiment tracker adapters"
```

## Task 3: Wire Index And Tracker Into Experiment Runner

**Files:**
- Modify: `src/cartola/backtesting/experiment_runner.py`
- Test: `src/tests/backtesting/test_experiment_runner.py`

- [ ] **Step 1: Write failing integration tests for index rows and tracker event sequence**

Append to `src/tests/backtesting/test_experiment_runner.py`:

```python
import sqlite3

from cartola.backtesting.experiment_tracking import InMemoryExperimentTracker


def test_experiment_runner_writes_index_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    index_path = tmp_path / "data/08_reporting/experiments/experiment_index.sqlite"
    with sqlite3.connect(index_path) as connection:
        experiment_rows = connection.execute(
            "SELECT experiment_id, status, completed_child_run_count FROM experiments"
        ).fetchall()
        child_count = connection.execute("SELECT COUNT(*) FROM child_runs").fetchone()[0]
        first_child = connection.execute(
            "SELECT child_run_id, status, comparable_within_partition FROM child_runs ORDER BY child_run_id LIMIT 1"
        ).fetchone()

    assert experiment_rows == [(result.experiment_id, "ok", 8)]
    assert child_count == 8
    assert first_child[1:] == ("ok", 1)


def test_experiment_runner_sends_tracker_events_and_finalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = InMemoryExperimentTracker()

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
        tracker=tracker,
    )

    assert tracker.events[0]["event"] == "start_experiment"
    assert [event["event"] for event in tracker.events].count("start_child") == 8
    assert [event["event"] for event in tracker.events].count("end_child") == 8
    assert tracker.events[-1] == {"event": "end_experiment", "status": "ok"}


def test_experiment_runner_finalizes_tracker_and_index_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = InMemoryExperimentTracker()

    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        if primary_model_id == "extra_trees":
            raise RuntimeError("boom")
        return _result(config, model_id=primary_model_id)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_model_experiment(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            current_year=2026,
            jobs=4,
            project_root=tmp_path,
            output_root=Path("experiments/model_feature"),
            started_at_utc="20260430T200000000000Z",
            tracker=tracker,
        )

    index_path = tmp_path / "data/08_reporting/experiments/experiment_index.sqlite"
    with sqlite3.connect(index_path) as connection:
        status = connection.execute("SELECT status FROM experiments").fetchone()[0]
        completed_children = connection.execute("SELECT COUNT(*) FROM child_runs WHERE status = 'ok'").fetchone()[0]

    assert status == "failed"
    assert completed_children == 2
    assert tracker.events[-1] == {"event": "end_experiment", "status": "failed"}
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_runner.py -q
```

Expected: new tests fail because `run_model_experiment()` has no `tracker` argument and does not write the index.

- [ ] **Step 3: Add imports and function signature changes**

Modify `src/cartola/backtesting/experiment_runner.py` imports:

```python
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
```

Add imports from new modules:

```python
from cartola.backtesting.experiment_index import (
    ExperimentIndex,
    artifact_pointer_payload,
    sha256_json,
    sha256_optional_file,
    source_hash_summary,
)
from cartola.backtesting.experiment_tracking import ExperimentTracker, NoOpExperimentTracker
```

Change `run_model_experiment()` signature:

```python
def run_model_experiment(
    *,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    project_root: Path,
    output_root: Path,
    started_at_utc: str,
    progress_callback: ExperimentProgressCallback | None = None,
    tracker: ExperimentTracker | None = None,
) -> ExperimentRunResult:
```

- [ ] **Step 4: Add metadata/index helper functions**

Append helper functions near `_child_record()` in `experiment_runner.py`:

```python
def _experiment_index(project_root: Path) -> ExperimentIndex:
    return ExperimentIndex(project_root / "data" / "08_reporting" / "experiments" / "experiment_index.sqlite")


def _utc_now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty(project_root: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _experiment_index_row(
    *,
    experiment_id_value: str,
    group: ExperimentGroup,
    started_at_utc: str,
    finished_at_utc: str | None,
    status: str,
    output_path: Path,
    matrix_hash: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    child_run_count: int,
    completed_child_run_count: int,
    failed_child_run_count: int,
    project_root: Path,
    tracker: ExperimentTracker,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id_value,
        "group": group,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": status,
        "output_path": _relative_path(output_path, project_root=project_root),
        "matrix_hash": matrix_hash,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "scoring_contract_version": "cartola_standard_2026_v1",
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(project_root, "branch", "--show-current"),
        "git_dirty": _git_dirty(project_root),
        "python_version": sys.version,
        "uv_lock_hash": sha256_optional_file(project_root / "uv.lock"),
        "mlflow_enabled": tracker.__class__.__name__ == "MLflowExperimentTracker",
        "mlflow_status": _mlflow_status_from_tracker(tracker),
        "mlflow_parent_run_id": tracker.parent_run_id,
        "warning_count": len(tracker.warnings),
        "child_run_count": child_run_count,
        "completed_child_run_count": completed_child_run_count,
        "failed_child_run_count": failed_child_run_count,
    }


def _relative_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _mlflow_status_from_tracker(tracker: ExperimentTracker) -> str:
    if tracker.__class__.__name__ != "MLflowExperimentTracker":
        return "disabled"
    if not tracker.warnings:
        return "ok"
    return "partial"
```

- [ ] **Step 5: Add child metric/index helpers**

Append these helpers:

```python
def _child_index_row(
    *,
    spec: ChildRunSpec,
    result: BacktestResult,
    child_id: str,
    experiment_id_value: str,
    project_root: Path,
    raw_source_identity: Mapping[str, object],
    candidate_pool_signature_hash: str | None,
    solver_status_signature_hash: str | None,
    comparable: bool,
    tracker: ExperimentTracker,
) -> dict[str, object]:
    primary_summary = result.summary[result.summary["strategy"] == spec.model_id]
    summary_row = primary_summary.iloc[0].to_dict() if not primary_summary.empty else {}
    prediction_rows = _prediction_metric_rows(spec, result, child_id=child_id)
    prediction_by_scope = {str(row["metric_scope"]): row for row in prediction_rows}
    candidate_metrics = prediction_by_scope.get("candidate_pool", {})
    selected_metrics = prediction_by_scope.get("selected_players", {})
    top50_metrics = prediction_by_scope.get("top50_candidates", {})
    return {
        "experiment_id": experiment_id_value,
        "child_run_id": child_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "footystats_mode": spec.backtest_config.footystats_mode,
        "matchup_context_mode": spec.backtest_config.matchup_context_mode,
        "output_path": _relative_path(spec.output_path, project_root=project_root),
        "status": "ok",
        "wall_clock_seconds": result.metadata.wall_clock_seconds,
        "backtest_jobs": spec.jobs,
        "backtest_workers_effective": result.metadata.backtest_workers_effective,
        "model_n_jobs_effective": _model_n_jobs_for_child(spec, result),
        "total_actual_points": _float_or_none(summary_row.get("total_actual_points")),
        "avg_actual_points": _float_or_none(summary_row.get("average_actual_points")),
        "total_predicted_points": _float_or_none(summary_row.get("total_predicted_points")),
        "prediction_mae": _float_or_none(candidate_metrics.get("mae")),
        "prediction_rmse": _float_or_none(candidate_metrics.get("rmse")),
        "prediction_r2": _float_or_none(candidate_metrics.get("r2")),
        "prediction_pearson": _float_or_none(candidate_metrics.get("pearson")),
        "prediction_spearman": _float_or_none(candidate_metrics.get("spearman")),
        "selected_calibration_slope": _float_or_none(selected_metrics.get("calibration_slope")),
        "top50_spearman": _float_or_none(top50_metrics.get("spearman")),
        "optimal_round_count": int(result.round_results["solver_status"].eq("Optimal").sum()),
        "skipped_round_count": int(result.round_results["solver_status"].ne("Optimal").sum()),
        "candidate_pool_signature_hash": candidate_pool_signature_hash,
        "solver_status_signature_hash": solver_status_signature_hash,
        "comparability_partition": _comparability_partition(spec),
        "comparable_within_partition": comparable,
        "ineligibility_reason": None,
        "source_hash_summary": source_hash_summary(raw_source_identity),
        "mlflow_child_run_id": tracker.child_run_id,
    }


def _model_n_jobs_for_child(spec: ChildRunSpec, result: BacktestResult) -> int | None:
    return (
        result.metadata.model_n_jobs_effective
        if model_n_jobs_for_metadata(spec.model_id, requested_n_jobs=spec.jobs) is not None
        else None
    )


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _child_metrics(row: Mapping[str, object]) -> dict[str, float | int | None]:
    return {
        "squad/actual_points_total": _float_or_none(row.get("total_actual_points")),
        "squad/actual_points_mean": _float_or_none(row.get("avg_actual_points")),
        "squad/predicted_points_total": _float_or_none(row.get("total_predicted_points")),
        "prediction/candidate_pool/mae": _float_or_none(row.get("prediction_mae")),
        "prediction/candidate_pool/rmse": _float_or_none(row.get("prediction_rmse")),
        "prediction/candidate_pool/r2": _float_or_none(row.get("prediction_r2")),
        "prediction/candidate_pool/pearson": _float_or_none(row.get("prediction_pearson")),
        "prediction/candidate_pool/spearman": _float_or_none(row.get("prediction_spearman")),
        "prediction/selected_players/calibration_slope": _float_or_none(row.get("selected_calibration_slope")),
        "prediction/top50/spearman": _float_or_none(row.get("top50_spearman")),
        "runtime/wall_clock_seconds": _float_or_none(row.get("wall_clock_seconds")),
        "rounds/optimal_count": row.get("optimal_round_count"),
        "rounds/skipped_count": row.get("skipped_round_count"),
    }


def _child_params(spec: ChildRunSpec) -> dict[str, object]:
    return {
        "group": spec.group,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "footystats_mode": spec.backtest_config.footystats_mode,
        "matchup_context_mode": spec.backtest_config.matchup_context_mode,
        "start_round": spec.backtest_config.start_round,
        "budget": spec.backtest_config.budget,
        "current_year": spec.backtest_config.current_year,
        "jobs": spec.jobs,
        "scoring_contract_version": "cartola_standard_2026_v1",
        **{f"model/{key}": value for key, value in spec.model_parameters.items()},
    }
```

- [ ] **Step 6: Wire lifecycle in `run_model_experiment()`**

Inside `run_model_experiment()` after `output_path.mkdir(parents=True)`, add:

```python
    tracker = tracker or NoOpExperimentTracker()
    index = _experiment_index(project_root)
    index_warnings: list[str] = []
    try:
        index.initialize()
    except Exception as exc:
        index_warnings.append(f"{type(exc).__name__}: {exc}")
```

After `raw_sources = ...`, add:

```python
    start_row = _experiment_index_row(
        experiment_id_value=run_id,
        group=group,
        started_at_utc=started_at_utc,
        finished_at_utc=None,
        status="running",
        output_path=output_path,
        matrix_hash=matrix_hash,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        current_year=current_year,
        jobs=jobs,
        child_run_count=total_children,
        completed_child_run_count=0,
        failed_child_run_count=0,
        project_root=project_root,
        tracker=tracker,
    )
    _safe_index_write(index, "upsert_experiment", start_row, index_warnings)
    tracker.start_experiment(
        experiment_name=f"cartola-{group}",
        run_name=run_id,
        params={
            "group": group,
            "start_round": start_round,
            "budget": budget,
            "current_year": current_year,
            "jobs": jobs,
            "scoring_contract_version": "cartola_standard_2026_v1",
        },
        tags={
            "experiment_id": run_id,
            "matrix_hash": matrix_hash,
            "git.commit": _git_value(project_root, "rev-parse", "HEAD"),
            "git.branch": _git_value(project_root, "branch", "--show-current"),
            "git.dirty": _git_dirty(project_root),
            "python.version": sys.version,
            "uv.lock.hash": sha256_optional_file(project_root / "uv.lock"),
            "cartola.version": _package_version("cartola"),
            "pandas.version": _package_version("pandas"),
            "numpy.version": _package_version("numpy"),
            "scikit-learn.version": _package_version("scikit-learn"),
            "plotly.version": _package_version("plotly"),
            "mlflow.version": _package_version("mlflow"),
            "platform": platform.platform(),
        },
    )
```

Add `_safe_index_write()` helper:

```python
def _safe_index_write(index: ExperimentIndex, method_name: str, row: Mapping[str, object], warnings: list[str]) -> None:
    try:
        getattr(index, method_name)(row)
    except Exception as exc:
        warnings.append(f"{method_name}: {type(exc).__name__}: {exc}")
```

Wrap the main child loop and final write path with a `try/finally` so `tracker.end_experiment()` always runs. The implementation can use a local `experiment_status = "failed"` and set it to `"ok"` after `_write_success_artifacts()`. In the `finally`, call:

```python
    tracker.end_experiment(status="ok" if experiment_status == "ok" else "failed")
```

Do not swallow the original exception.

- [ ] **Step 7: Start each child tracker run before executing the child backtest**

Inside the child loop, immediately after emitting the existing `"child_started"` progress event and before calling `run_backtest_for_experiment(...)`, add:

```python
        tracker.start_child(
            run_name=f"season={spec.season} model={spec.model_id} feature_pack={spec.feature_pack}",
            params=_child_params(spec),
            tags={
                "experiment_id": run_id,
                "child_run_id": child_id,
                "output_path": _relative_path(spec.output_path, project_root=project_root),
                "comparability_partition": _comparability_partition(spec),
            },
        )
```

In each existing child failure block, before writing failure metadata and raising, add:

```python
            tracker.end_child(status="failed")
```

This ensures the MLflow child run ID exists before the child index row is built and ensures failed child MLflow runs are not left open.

- [ ] **Step 8: Log each child after artifacts are written**

After `child_runs.append(_child_record(...))` and after signatures are computed for the child, build and write child state:

```python
            child_candidate_signatures = _candidate_signatures_by_round(result.player_predictions)
            child_solver_signature = solver_status_signature(result.round_results, primary_model_id=spec.model_id)
            candidate_pool_signatures[child_id] = child_candidate_signatures
            solver_status_signatures[child_id] = child_solver_signature
```

Replace duplicate direct signature assignment with those local variables.

After per-season/prediction/calibration rows are extended, add:

```python
            child_row = _child_index_row(
                spec=spec,
                result=result,
                child_id=child_id,
                experiment_id_value=run_id,
                project_root=project_root,
                raw_source_identity=raw_sources[str(spec.season)],
                candidate_pool_signature_hash=sha256_json(child_candidate_signatures),
                solver_status_signature_hash=sha256_json(child_solver_signature),
                comparable=True,
                tracker=tracker,
            )
            _safe_index_write(index, "upsert_child_run", child_row, index_warnings)
            pointer_payload = artifact_pointer_payload(
                project_root=project_root,
                child_run_id=child_id,
                output_path=spec.output_path,
                artifact_paths=[
                    spec.output_path / "player_predictions.csv",
                    spec.output_path / "selected_players.csv",
                ],
            )
            pointer_path = spec.output_path / "artifact_pointers.json"
            pointer_path.write_text(json.dumps(pointer_payload, indent=2, sort_keys=True), encoding="utf-8")
            tracker.log_child_metrics(_child_metrics(child_row))
            tracker.log_child_artifacts(
                [
                    spec.output_path / "summary.csv",
                    spec.output_path / "diagnostics.csv",
                    spec.output_path / "run_metadata.json",
                    pointer_path,
                    spec.output_path / "player_predictions.csv",
                    spec.output_path / "selected_players.csv",
                ]
            )
            tracker.end_child(status="ok")
```

If a child fails before a child row can be written, keep existing failure metadata behavior and finalize experiment as failed.

- [ ] **Step 9: Finalize index and tracker after top-level reports**

After `_write_success_artifacts(...)`, add:

```python
    finish_row = _experiment_index_row(
        experiment_id_value=run_id,
        group=group,
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now_id(),
        status="ok",
        output_path=output_path,
        matrix_hash=matrix_hash,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        current_year=current_year,
        jobs=jobs,
        child_run_count=total_children,
        completed_child_run_count=len(child_runs),
        failed_child_run_count=0,
        project_root=project_root,
        tracker=tracker,
    )
    _safe_index_write(index, "upsert_experiment", finish_row, index_warnings)
    tracker.log_parent_artifacts(
        [
            output_path / "ranked_summary.csv",
            output_path / "per_season_summary.csv",
            output_path / "prediction_metrics.csv",
            output_path / "calibration_deciles.csv",
            output_path / "comparability_report.json",
            output_path / "experiment_metadata.json",
            output_path / "comparison_report.md",
            output_path / "calibration_plots.html",
            output_path / "squad_performance_comparison.html",
        ]
    )
```

In each existing failure block, before raising, upsert a failed experiment row with `failed_child_run_count=1` and `completed_child_run_count=len(child_runs)`.

- [ ] **Step 10: Run focused experiment runner tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_runner.py src/tests/backtesting/test_experiment_index.py src/tests/backtesting/test_experiment_tracking.py -q
```

Expected: all tests pass.

- [ ] **Step 11: Commit Task 3**

```bash
git add src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_experiment_runner.py
git commit -m "feat: index model experiment runs"
```

## Task 4: CLI Flags And Gitignore

**Files:**
- Modify: `scripts/run_model_experiments.py`
- Modify: `.gitignore`
- Test: `src/tests/backtesting/test_run_model_experiments_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Append to `src/tests/backtesting/test_run_model_experiments_cli.py`:

```python
def test_parse_args_tracker_defaults() -> None:
    args = parse_args(["--group", "production-parity", "--current-year", "2026"])

    assert args.tracker == "none"
    assert args.mlflow_tracking_uri is None


def test_main_passes_mlflow_tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_model_experiment(**kwargs: object) -> object:
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(
        [
            "--group",
            "production-parity",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--tracker",
            "mlflow",
            "--mlflow-tracking-uri",
            "file:///tmp/cartola-mlruns",
        ]
    )

    assert exit_code == 0
    assert observed["tracker"].__class__.__name__ == "MLflowExperimentTracker"
    assert observed["tracker"].tracking_uri == "file:///tmp/cartola-mlruns"
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py -q
```

Expected: fails because parser has no tracker flags.

- [ ] **Step 3: Add CLI flags and tracker construction**

Modify imports in `scripts/run_model_experiments.py`:

```python
from cartola.backtesting.experiment_tracking import MLflowExperimentTracker, NoOpExperimentTracker
```

Add parser args:

```python
    parser.add_argument("--tracker", choices=("none", "mlflow"), default="none")
    parser.add_argument("--mlflow-tracking-uri", default=None)
```

Before `run_model_experiment(...)`, add:

```python
        tracker = (
            MLflowExperimentTracker(tracking_uri=args.mlflow_tracking_uri)
            if args.tracker == "mlflow"
            else NoOpExperimentTracker()
        )
```

Pass `tracker=tracker` into `run_model_experiment(...)`.

After successful run, if `tracker.warnings`, print a warning panel to stderr:

```python
    if tracker.warnings:
        warning_lines = "\n".join(f"{warning.phase}: {warning.message}" for warning in tracker.warnings[:5])
        stderr.print(Panel(warning_lines, title="Experiment tracking warnings", border_style="yellow"))
```

- [ ] **Step 4: Ignore root MLflow store**

Append to `.gitignore` near custom ignores:

```gitignore
# MLflow local tracking store
mlruns/
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/run_model_experiments.py src/tests/backtesting/test_run_model_experiments_cli.py .gitignore
git commit -m "feat: expose optional experiment tracker"
```

## Task 5: Metadata Warnings And Report Stability

**Files:**
- Modify: `src/cartola/backtesting/experiment_runner.py`
- Test: `src/tests/backtesting/test_experiment_runner.py`

- [ ] **Step 1: Write failing tests for unchanged scientific reports with tracking off**

Append to `src/tests/backtesting/test_experiment_runner.py`:

```python
def test_tracker_none_does_not_change_scientific_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest_for_experiment(config: BacktestConfig, *, primary_model_id: str) -> BacktestResult:
        return _result(config, model_id=primary_model_id, candidate_count=60)

    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.run_backtest_for_experiment",
        fake_run_backtest_for_experiment,
    )
    monkeypatch.setattr(
        "cartola.backtesting.experiment_runner.raw_cartola_source_identity",
        lambda *, project_root, season: {"season": season, "sha256": "raw"},
    )

    result = run_model_experiment(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        current_year=2026,
        jobs=4,
        project_root=tmp_path,
        output_root=Path("experiments/model_feature"),
        started_at_utc="20260430T200000000000Z",
    )

    metadata = json.loads((result.output_path / "experiment_metadata.json").read_text(encoding="utf-8"))
    ranked = pd.read_csv(result.output_path / "ranked_summary.csv")

    assert metadata["status"] == "ok"
    assert "tracking_warnings" in metadata
    assert metadata["tracking_warnings"] == []
    assert len(ranked) == 8
    assert (tmp_path / "data/08_reporting/experiments/experiment_index.sqlite").exists()
```

- [ ] **Step 2: Update metadata helper to include tracking/index warnings**

Modify `_metadata(...)` signature to accept `tracking_warnings: Sequence[Mapping[str, object]] | None = None` and `index_warnings: Sequence[str] | None = None`.

Add to the returned dict:

```python
        "tracking_warnings": list(tracking_warnings or []),
        "index_warnings": list(index_warnings or []),
```

Pass `tracking_warnings=[asdict(warning) for warning in tracker.warnings]` and `index_warnings=index_warnings` in success and failure metadata calls.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_experiment_runner.py src/tests/backtesting/test_run_model_experiments_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit Task 5**

```bash
git add src/cartola/backtesting/experiment_runner.py src/tests/backtesting/test_experiment_runner.py
git commit -m "feat: record experiment tracking warnings"
```

## Task 6: Full Verification And Cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run all focused observability tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_experiment_index.py \
  src/tests/backtesting/test_experiment_tracking.py \
  src/tests/backtesting/test_experiment_runner.py \
  src/tests/backtesting/test_run_model_experiments_cli.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest pass.

- [ ] **Step 3: Smoke-test CLI with tracker disabled**

Run a tiny mocked test path through pytest rather than a real full experiment:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_model_experiments_cli.py::test_main_calls_runner -q
```

Expected: passes, proving default CLI still uses normal reports and no MLflow dependency.

- [ ] **Step 4: Check no heavy artifacts are staged**

Run:

```bash
git status --short
find mlruns -maxdepth 2 -type f 2>/dev/null | head
```

Expected: source/test/docs changes only; `mlruns/` is ignored or absent.

- [ ] **Step 5: Commit final cleanup if needed**

Only if Step 2 or Step 4 required edits:

```bash
git add <changed-files>
git commit -m "test: verify experiment observability"
```

## Self-Review Checklist

- Spec coverage:
  - SQLite index: Tasks 1 and 3.
  - WAL/busy timeout/schema version: Task 1.
  - Primary-key upserts/state machine: Tasks 1 and 3.
  - Optional MLflow adapter: Tasks 2 and 4.
  - Lazy/no-fatal MLflow import: Task 2.
  - No heavy artifact duplication: Tasks 2 and 3.
  - Tracking URI precedence: Task 4.
  - Artifact pointer JSON: Tasks 1 and 3.
  - Metadata warning visibility: Task 5.
  - No semantic model/scoring changes: all tasks keep `run_backtest_for_experiment()` behavior unchanged.
- Placeholder scan: no `TBD`, `TODO`, or unresolved design choices in this plan.
- Type consistency:
  - `ExperimentTracker` methods match CLI and runner usage.
  - `ExperimentIndex` upsert rows use schema column names.
  - `child_run_id` matches `season=<season>/model=<model_id>/feature_pack=<feature_pack>`.
