from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cartola.backtesting.experiment_index import (
    SCHEMA_VERSION,
    ExperimentIndex,
    artifact_pointer_payload,
    sha256_file,
    sha256_json,
    sha256_optional_file,
    source_hash_summary,
)


def test_index_initializes_schema_wal_timeout_and_version(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "experiment_index.sqlite"

    index = ExperimentIndex(db_path)
    index.initialize()

    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert journal_mode == "wal"
    assert busy_timeout == 5000
    assert user_version == SCHEMA_VERSION
    assert {"experiments", "child_runs"}.issubset(tables)


def test_experiment_upsert_is_keyed_by_experiment_id(tmp_path: Path) -> None:
    index = ExperimentIndex(tmp_path / "experiment_index.sqlite")
    index.initialize()

    first = _experiment_row(status="running", completed_child_run_count=0)
    second = _experiment_row(
        status="ok",
        finished_at_utc="2026-05-01T12:10:00Z",
        completed_child_run_count=8,
        git_dirty=True,
        seasons=[2024, 2025],
    )
    index.upsert_experiment(first)
    index.upsert_experiment(second)

    with sqlite3.connect(index.path) as connection:
        rows = connection.execute(
            """
            SELECT status, completed_child_run_count, git_dirty, seasons
            FROM experiments
            WHERE experiment_id = ?
            """,
            ("exp-1",),
        ).fetchall()

    assert rows == [("ok", 8, 1, "[2024,2025]")]


def test_child_upsert_is_keyed_by_experiment_id_and_child_run_id(
    tmp_path: Path,
) -> None:
    index = ExperimentIndex(tmp_path / "experiment_index.sqlite")
    index.initialize()

    first = _child_run_row(status="running", total_actual_points=None)
    second = _child_run_row(
        status="ok",
        total_actual_points=100.0,
        source_hash_summary={"raw": "identity"},
    )
    index.upsert_child_run(first)
    index.upsert_child_run(second)

    with sqlite3.connect(index.path) as connection:
        rows = connection.execute(
            """
            SELECT child_run_id, status, total_actual_points, source_hash_summary
            FROM child_runs
            WHERE experiment_id = ?
            """,
            ("exp-1",),
        ).fetchall()

    assert rows == [
        (
            "season=2025/model=random_forest/feature_pack=ppg",
            "ok",
            100.0,
            '{"raw":"identity"}',
        )
    ]


def test_sha256_optional_file_hashes_existing_file_and_returns_none_for_missing(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")

    assert sha256_optional_file(lock) == sha256_file(lock)
    assert sha256_optional_file(tmp_path / "missing.lock") is None


def test_source_hash_summary_matches_canonical_json_hash() -> None:
    raw_identity = {
        "season": 2025,
        "files": [{"path": "data/01_raw/2025/rodada-1.csv", "sha256": "abc"}],
    }

    assert source_hash_summary(raw_identity) == sha256_json(raw_identity)


def test_artifact_pointer_payload_uses_relative_paths_and_existing_artifacts(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "data" / "08_reporting" / "run-1"
    output_path.mkdir(parents=True)
    player_predictions = output_path / "player_predictions.csv"
    selected_players = output_path / "selected_players.csv"
    player_predictions.write_text("a,b\n1,2\n", encoding="utf-8")
    selected_players.write_text("a,b\n3,4\n", encoding="utf-8")

    payload = artifact_pointer_payload(
        project_root=tmp_path,
        child_run_id="season=2025/model=random_forest/feature_pack=ppg",
        output_path=output_path,
        artifact_paths=[
            player_predictions,
            selected_players,
            output_path / "missing.csv",
        ],
    )

    assert payload["child_run_id"] == "season=2025/model=random_forest/feature_pack=ppg"
    assert payload["output_path"] == "data/08_reporting/run-1"
    assert sorted(payload["artifacts"]) == [
        "player_predictions.csv",
        "selected_players.csv",
    ]
    assert payload["artifacts"]["player_predictions.csv"] == {
        "path": "data/08_reporting/run-1/player_predictions.csv",
        "size_bytes": player_predictions.stat().st_size,
        "sha256": sha256_file(player_predictions),
    }


def _experiment_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "exp-1",
        "group": "production-parity",
        "started_at_utc": "2026-05-01T12:00:00Z",
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
        "git_dirty": False,
        "python_version": "3.13.12",
        "uv_lock_hash": None,
        "mlflow_enabled": False,
        "mlflow_status": "disabled",
        "mlflow_parent_run_id": None,
        "warning_count": 0,
        "child_run_count": 8,
        "completed_child_run_count": 0,
        "failed_child_run_count": 0,
    }
    row.update(overrides)
    return row


def _child_run_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "exp-1",
        "child_run_id": "season=2025/model=random_forest/feature_pack=ppg",
        "season": 2025,
        "model_id": "random_forest",
        "feature_pack": "ppg",
        "fixture_mode": "none",
        "footystats_mode": "ppg",
        "matchup_context_mode": "none",
        "output_path": (
            "data/08_reporting/experiments/model_feature/exp-1/runs/season=2025/model=random_forest/feature_pack=ppg"
        ),
        "status": "running",
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
        "comparable_within_partition": True,
        "ineligibility_reason": None,
        "source_hash_summary": "source",
        "mlflow_child_run_id": None,
    }
    row.update(overrides)
    return row
