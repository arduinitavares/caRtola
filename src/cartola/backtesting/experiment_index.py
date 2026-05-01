from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Mapping, Sequence, TypedDict

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


class ArtifactPointerEntry(TypedDict):
    path: str
    size_bytes: int
    sha256: str


class ArtifactPointerPayload(TypedDict):
    child_run_id: str
    output_path: str
    artifacts: dict[str, ArtifactPointerEntry]


class ExperimentIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported experiment index schema version: {user_version}")
            _create_schema(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def upsert_experiment(self, row: Mapping[str, object]) -> None:
        self._upsert(
            table="experiments",
            columns=EXPERIMENT_COLUMNS,
            conflict_columns=("experiment_id",),
            row=row,
        )

    def upsert_child_run(self, row: Mapping[str, object]) -> None:
        self._upsert(
            table="child_runs",
            columns=CHILD_RUN_COLUMNS,
            conflict_columns=("experiment_id", "child_run_id"),
            row=row,
        )

    def _upsert(
        self,
        *,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
        row: Mapping[str, object],
    ) -> None:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"Missing {table} columns: {', '.join(missing)}")

        column_sql = ", ".join(_quote_identifier(column) for column in columns)
        placeholders = ", ".join("?" for _column in columns)
        conflict_sql = ", ".join(_quote_identifier(column) for column in conflict_columns)
        update_sql = ", ".join(
            f"{_quote_identifier(column)}=excluded.{_quote_identifier(column)}"
            for column in columns
            if column not in conflict_columns
        )
        values = [_sqlite_value(row[column]) for column in columns]
        sql = (
            f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict_sql}) DO UPDATE SET {update_sql}"
        )
        with self._connect() as connection:
            connection.execute(sql, values)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return connection


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
) -> ArtifactPointerPayload:
    resolved_project_root = project_root.resolve()
    artifacts: dict[str, ArtifactPointerEntry] = {}
    for path in artifact_paths:
        if not path.exists():
            continue
        artifacts[path.name] = {
            "path": _project_relative_path(path, project_root=resolved_project_root),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "child_run_id": child_run_id,
        "output_path": _project_relative_path(
            output_path,
            project_root=resolved_project_root,
        ),
        "artifacts": artifacts,
    }


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
            git_dirty INTEGER NOT NULL,
            python_version TEXT NOT NULL,
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
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_runs_model_feature
        ON child_runs(model_id, feature_pack)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_child_runs_partition
        ON child_runs(comparability_partition)
        """
    )


def _project_relative_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
