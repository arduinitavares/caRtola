# Experiment Observability Design

## Goal

Improve visibility into long-running model/feature experiment matrices without changing the scientific contract of the backtest runner.

The immediate pain is that `scripts/run_model_experiments.py` can run for hours. The current reports are useful after completion, but they are hard to compare across experiment runs and do not provide a durable local index of historical experiment outcomes. MLflow can help as a run browser, but it must not become the source of truth or duplicate large report files.

This feature adds a lightweight experiment index and an optional tracking adapter layer. Existing CSV, JSON, Markdown, and HTML reports remain authoritative.

## Current State

The project already has:

- `scripts/run_model_experiments.py`;
- fixed model ids: `random_forest`, `extra_trees`, `hist_gradient_boosting`, `ridge`;
- fixed feature packs: `ppg`, `ppg_xg`, `ppg_matchup`, `ppg_xg_matchup`;
- sequential child backtests;
- `--jobs` for target-round parallelism inside each child backtest;
- per-child backtest artifacts under `data/08_reporting/experiments/model_feature/<experiment_id>/runs/...`;
- top-level ranked summaries, prediction metrics, calibration deciles, comparability reports, Markdown reports, and HTML reports;
- strong comparability checks around candidate pools, skipped rounds, solver statuses, scoring contract, and source metadata.

The current long-running production-parity matrix is still valuable. The problem is not that experiments are long. The problem is that long experiments need better observability and cross-run indexing.

## Problem

Manual inspection of output folders does not scale once we run multiple full matrices.

The project needs:

- a local index of experiment and child-run outcomes;
- stable child-run identities;
- clear links from summary rows back to full report folders;
- searchable/loggable model, feature, source, scoring, and comparability metadata;
- optional MLflow integration for scalar metric browsing;
- no duplication of heavy CSV artifacts;
- no change to optimizer, scoring, feature generation, or model behavior.

Without this boundary, MLflow can accidentally become a second report store, add disk bloat, and encourage comparisons that the experiment runner would otherwise reject.

## Non-Goals

Do not add:

- model changes;
- feature-pack changes;
- optimizer changes;
- scoring-contract changes;
- live recommendation default changes;
- child-run parallelism;
- hyperparameter search;
- automatic model promotion;
- required MLflow usage for normal backtests or live recommendations;
- full resume/filter execution semantics.

Resume support is a separate workflow feature. This design may log enough child-run identity data to make resume easier later, but it does not change execution semantics.

## Design Summary

Implement **Experiment Observability v1** as two layers:

1. **Local experiment index**
   - Always written.
   - Stores one row per experiment run and one row per child run.
   - Points to authoritative report folders.
   - Does not copy heavy artifacts.

2. **Optional tracker adapter**
   - `NoOpExperimentTracker` is the default behavior.
   - `MLflowExperimentTracker` can be enabled for experiment runs.
   - MLflow logs params, tags, scalar metrics, selected small artifacts, and pointers/hashes for large artifacts.
   - MLflow failures are best-effort warnings and do not invalidate successful backtest reports.

The existing report tree remains the source of truth. The index and MLflow are views over that tree.

## Experiment Index

Write a SQLite index at:

```text
data/08_reporting/experiments/experiment_index.sqlite
```

Use SQLite for v1 because it is in the Python standard library, needs no new dependency, supports incremental writes, and is easy to query locally. Do not use Parquet for v1.

The index lives at the `experiments/` level, not under `model_feature/`, so it can support future experiment types beyond model/feature matrices.

The runner attempts to write the index for every experiment run. The index is a report/index sidecar, not the scientific source of truth. If index initialization or writes fail, the experiment continues, records a warning in experiment metadata, and keeps the normal CSV/JSON/Markdown/HTML reports authoritative.

### SQLite Contract

Initialize the database with:

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA user_version = 1;
```

The v1 schema version is `1`. Future incompatible schema changes must bump `PRAGMA user_version` and include an explicit migration or fail with a clear schema-version error.

Concurrent experiment commands may write to the same index. SQLite WAL mode allows concurrent readers and one writer; `busy_timeout=5000` gives a competing writer up to five seconds before surfacing a warning. The implementation should keep write transactions short.

Schema constraints:

- `experiments.experiment_id` is the primary key.
- `child_runs(experiment_id, child_run_id)` is the primary key.
- Child rows use upsert semantics.
- Experiment rows use upsert semantics.
- Each child row is committed only after the child report artifacts are successfully written.
- Experiment finalization is committed after top-level reports are written.

Allowed experiment statuses:

- `running`;
- `ok`;
- `failed`.

Allowed child statuses:

- `ok`;
- `failed`;
- `skipped`.

Allowed MLflow statuses:

- `disabled`;
- `ok`;
- `partial`;
- `failed`.

Index state machine:

1. At experiment start, insert or update the experiment row with `status="running"`.
2. After each child completes, insert or update its child row with `status="ok"` or `status="failed"`.
3. At experiment completion, update the experiment row with `status="ok"`, `finished_at_utc`, and final counts.
4. If the experiment raises, attempt to update the experiment row with `status="failed"`, `finished_at_utc`, and counts available so far.
5. If the process is killed, a `running` experiment row with partial child rows is valid crash-diagnosis state.

The index has two logical tables.

SQLite column type rules:

- identifiers, modes, statuses, hashes, paths, versions, and timestamps are `TEXT`;
- counts and seasons are `INTEGER`;
- point totals, metric values, and runtimes are `REAL`;
- booleans are stored as `INTEGER` values `0` or `1`;
- list or dict payloads, such as `seasons` and compact source summaries, are stored as canonical JSON `TEXT`.

### Experiment Rows

One row per matrix execution:

- `experiment_id`;
- `group`;
- `started_at_utc`;
- `finished_at_utc`;
- `status`;
- `output_path`;
- `matrix_hash`;
- `seasons`;
- `start_round`;
- `budget`;
- `current_year`;
- `jobs`;
- `scoring_contract_version`;
- `git_commit`;
- `git_branch`;
- `git_dirty`;
- `python_version`;
- `uv_lock_hash`;
- `mlflow_enabled`;
- `mlflow_status`;
- `mlflow_parent_run_id`;
- `warning_count`;
- `child_run_count`;
- `completed_child_run_count`;
- `failed_child_run_count`.

### Child Run Rows

One row per `(season, model_id, feature_pack)` child:

- `experiment_id`;
- `child_run_id`;
- `season`;
- `model_id`;
- `feature_pack`;
- `fixture_mode`;
- `footystats_mode`;
- `matchup_context_mode`;
- `output_path`;
- `status`;
- `wall_clock_seconds`;
- `backtest_jobs`;
- `backtest_workers_effective`;
- `model_n_jobs_effective`;
- `total_actual_points`;
- `avg_actual_points`;
- `total_predicted_points`;
- `prediction_mae`;
- `prediction_rmse`;
- `prediction_r2`;
- `prediction_pearson`;
- `prediction_spearman`;
- `selected_calibration_slope`;
- `top50_spearman`;
- `optimal_round_count`;
- `skipped_round_count`;
- `candidate_pool_signature_hash`;
- `solver_status_signature_hash`;
- `comparability_partition`;
- `comparable_within_partition`;
- `ineligibility_reason`;
- `source_hash_summary`;
- `mlflow_child_run_id`.

Null metric values stay null. Do not coerce missing metrics to zero.

`oracle_gap` and `capture_rate` are not v1 index columns because the current experiment runner does not compute recommendation-oracle metrics. They belong in a later metrics expansion if the experiment runner gains an oracle comparison.

### Hash Definitions

`uv_lock_hash` is:

```text
sha256(project_root / "uv.lock")
```

when that file exists, otherwise `null`.

`source_hash_summary` is:

```text
sha256(json.dumps(raw_cartola_source_identity(project_root, season), sort_keys=True, separators=(",", ":")))
```

This is a compact single hash representing the raw Cartola source provenance for that child's season. The full raw source identity remains in experiment metadata rather than being expanded into MLflow tags.

Duplicate matrix executions are allowed. Repeated runs with the same `matrix_hash` produce distinct `experiment_id` rows. The shared `matrix_hash` is intentionally queryable so users can find previous executions of the same matrix.

## Child Run Identity

Every child run has a deterministic identity:

```text
season=<season>/model=<model_id>/feature_pack=<feature_pack>
```

This identity appears in:

- output directory path;
- experiment index rows;
- MLflow run name or tags;
- warning/error records;
- future resume metadata.

## Tracker Interface

Add a small adapter boundary. The experiment runner should not call `mlflow.*` directly.

Target shape:

```python
class ExperimentTracker(Protocol):
    def start_experiment(self, ...) -> None: ...
    def start_child(self, ...) -> None: ...
    def log_child_metrics(self, ...) -> None: ...
    def log_child_artifacts(self, ...) -> None: ...
    def end_child(self, ...) -> None: ...
    def end_experiment(self, ...) -> None: ...
```

Implementations:

- `NoOpExperimentTracker`: default, no external behavior.
- `InMemoryExperimentTracker`: tests only, records event order and payloads.
- `MLflowExperimentTracker`: optional observer.

Production tracker implementations must be no-throw from the caller's perspective. The MLflow tracker catches its own failures and records them as warnings. Test-only trackers may raise only in tests that explicitly assert failure behavior.

## MLflow Integration

MLflow is optional and additive.

### Tracking Store

Use a local file-based tracking store by default:

```text
mlruns/
```

The path must be gitignored.

Precedence is:

1. CLI `--mlflow-tracking-uri`;
2. environment `MLFLOW_TRACKING_URI`;
3. local `mlruns/`.

Do not place MLflow's tracking store inside a specific experiment output directory. The report tree and the tracking store have different lifecycles.

### Experiment And Run Shape

Use one stable MLflow experiment per experiment group:

- `cartola-production-parity`;
- `cartola-matchup-research`.

Use one parent MLflow run per matrix execution.

Use child MLflow runs for each child backtest. The child run name is:

```text
season=<season> model=<model_id> feature_pack=<feature_pack>
```

Do not add season-level MLflow runs in v1. Instead, every child run must include `comparability_partition=season=<season>` as a tag. This keeps the run graph simple while preserving the comparability boundary.

### Params

Log stable configuration values as params:

- `group`;
- `season`;
- `model_id`;
- `feature_pack`;
- `fixture_mode`;
- `footystats_mode`;
- `matchup_context_mode`;
- `start_round`;
- `budget`;
- `current_year`;
- `jobs`;
- `scoring_contract_version`;
- flattened model params with `model/` prefixes.

### Tags

Log search/reproducibility metadata as tags:

- `experiment_id`;
- `child_run_id`;
- `matrix_hash`;
- `git.commit`;
- `git.branch`;
- `git.dirty`;
- `python.version`;
- `uv.lock.hash`;
- `output_path`;
- `comparability_partition`;
- `comparable_within_partition`;
- `candidate_pool_signature_hash`;
- `solver_status_signature_hash`;
- `source_hash_summary`;
- `mlflow_logging_status`;
- `cartola.version`.

Dependency versions should be logged when available:

- cartola;
- pandas;
- numpy;
- scikit-learn;
- PuLP;
- plotly;
- mlflow;
- Python.

Use `importlib.metadata.version()` for package versions and record `null` when a package version cannot be discovered.

Solver version should be logged if it can be discovered reliably.

### Metrics

Use namespaced metric keys. Do not use ambiguous names like `mae` or `points`.

Examples:

- `squad/actual_points_total`;
- `squad/actual_points_mean`;
- `squad/predicted_points_total`;
- `prediction/candidate_pool/mae`;
- `prediction/candidate_pool/rmse`;
- `prediction/candidate_pool/r2`;
- `prediction/candidate_pool/pearson`;
- `prediction/candidate_pool/spearman`;
- `prediction/selected_players/calibration_slope`;
- `prediction/top50/spearman`;
- `runtime/wall_clock_seconds`;
- `rounds/optimal_count`;
- `rounds/skipped_count`;
- `candidates/mean_per_round`.

Do not log round-level time-series metrics in v1. They are useful, but they expand the metric taxonomy and should be added only after child-level tracking is stable. If added later, use `step=rodada` and namespaced keys such as `round/actual_points` and `round/predicted_points`.

### Artifact Policy

Do not log heavy child artifacts by default.

Default child artifacts:

- `summary.csv`;
- `diagnostics.csv`;
- `run_metadata.json`;
- an artifact pointer JSON containing paths, sizes, and hashes for large files.

The child artifact pointer JSON schema is:

```json
{
  "child_run_id": "season=2023/model=random_forest/feature_pack=ppg",
  "output_path": "data/08_reporting/experiments/model_feature/<experiment_id>/runs/season=2023/model=random_forest/feature_pack=ppg",
  "artifacts": {
    "player_predictions.csv": {
      "path": "data/08_reporting/experiments/model_feature/<experiment_id>/runs/season=2023/model=random_forest/feature_pack=ppg/player_predictions.csv",
      "size_bytes": 7412345,
      "sha256": "abc123..."
    },
    "selected_players.csv": {
      "path": "data/08_reporting/experiments/model_feature/<experiment_id>/runs/season=2023/model=random_forest/feature_pack=ppg/selected_players.csv",
      "size_bytes": 1324567,
      "sha256": "def456..."
    }
  }
}
```

Paths are project-root-relative when the artifact is under the project root; otherwise they are absolute. Missing optional large artifacts are omitted from `artifacts`.

Default parent artifacts:

- `ranked_summary.csv`;
- `per_season_summary.csv`;
- `prediction_metrics.csv`;
- `calibration_deciles.csv`;
- `comparability_report.json`;
- `experiment_metadata.json`;
- Markdown report;
- HTML report.

Do not copy these child files by default:

- `player_predictions.csv`;
- `selected_players.csv`;
- large per-round or per-player artifacts.

Those files remain accessible through `output_path` and artifact pointer metadata.

An explicit future option may add full artifact logging, but it is not part of v1.

## Failure Semantics

The experiment reports are authoritative. MLflow is not.

If MLflow logging fails:

- continue the experiment;
- record a warning;
- store warning details in experiment metadata;
- set `mlflow_status` to `partial` or `failed`;
- keep the experiment exit code tied to experiment success, not MLflow logging success.

If the experiment itself fails, the experiment should still try to write whatever metadata and index rows can be written safely.

Do not let an MLflow filesystem, import, or tracking-store problem discard hours of valid backtest output.

The MLflow tracker must handle `ImportError` gracefully. If `mlflow` cannot be imported because it is not installed or does not support the active Python version, the tracker records one warning and behaves as a no-op. MLflow must be lazily imported inside the MLflow tracker, not at module import time.

MLflow is not a required runtime dependency for v1. Normal CI and non-MLflow experiment runs must work without importing the `mlflow` package.

## Comparability Guardrails

The existing comparability checks remain authoritative.

The index and MLflow must surface comparability state clearly:

- `comparability_partition`;
- `comparable_within_partition`;
- candidate-pool signature hash;
- solver-status signature hash;
- skipped-round counts;
- optimal-round counts;
- ineligibility reason.

MLflow cannot prevent a user from comparing invalid runs. Therefore the metadata must make invalid comparisons visible and filterable.

## CLI Behavior

Add optional tracking flags to `scripts/run_model_experiments.py`.

Suggested v1 flags:

```text
--tracker none|mlflow
--mlflow-tracking-uri <uri>
```

Defaults:

- `--tracker none`;
- local reports and local index are always written;
- MLflow is opt-in.

Normal backtest CLI and live recommendation scripts do not initialize MLflow in v1.

## Data Flow

1. Experiment runner builds the child matrix.
2. Experiment index starts or updates an experiment row with `status="running"`.
3. Tracker starts the parent run if enabled.
4. For each child:
   - run the child backtest unchanged;
   - write normal child artifacts unchanged;
   - compute child summary metrics from existing result/report objects;
   - append/update child index row;
   - send params, metrics, tags, and selected artifacts to tracker.
5. After all children:
   - existing comparability and ranking reports are written unchanged;
   - parent index row is finalized;
   - tracker logs parent artifacts and closes the parent run.

`tracker.end_experiment(status)` must be called from a `finally` block so an enabled MLflow parent run is always terminated as `FINISHED` or `FAILED`.

## Testing

Required tests:

- normal single backtest CLI does not initialize MLflow;
- live recommendation workflow does not initialize MLflow;
- experiment runner with `--tracker none` produces the same reports as before, excluding index files;
- experiment index writes one experiment row and one child row per child run;
- child-run identity is deterministic;
- `InMemoryExperimentTracker` records the expected event sequence;
- MLflow tracker logs required params, tags, and metrics through a mocked MLflow module;
- MLflow tracker does not log heavy artifacts by default;
- MLflow logging failures are swallowed and recorded as warnings;
- null metrics are not coerced to zero;
- comparability tags and signature hashes are written for every child run;
- output paths point to the authoritative report folders;
- git dirty state is recorded;
- `uv.lock` hash is recorded when the file exists;
- missing MLflow dependency or unreachable tracking URI does not fail an otherwise successful experiment in best-effort mode.
- SQLite initializes with WAL mode, busy timeout, and schema version `1`;
- experiment and child index rows use primary-key upserts;
- interrupted experiments leave `running` rows or explicitly finalize as `failed`;
- artifact pointer JSON includes project-relative paths, sizes, and hashes for large child artifacts.

## Acceptance Criteria

The feature is complete when:

- long model experiments still produce the same scientific reports as before;
- a local index can answer "what child runs have I run and how did they perform?";
- MLflow can be enabled without changing experiment results;
- heavy artifacts are not duplicated by default;
- failed MLflow logging is visible but non-fatal;
- comparability status is searchable in the index and visible in MLflow tags.

## Deferred Work

Separate specs should cover:

- resume/skip completed child runs;
- model-specific profiling for the slow HistGradientBoosting path;
- experiment oracle-gap and capture-rate metrics;
- child-run parallelism;
- model hyperparameter search;
- promotion of a new production model/feature pack;
- live recommendation integration for any winning pipeline.
