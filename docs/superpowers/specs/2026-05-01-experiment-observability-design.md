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

The index has two logical tables.

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
- `oracle_gap`;
- `capture_rate`;
- `optimal_round_count`;
- `skipped_round_count`;
- `candidate_pool_signature_hash`;
- `solver_status_signature_hash`;
- `comparability_partition`;
- `comparable_within_partition`;
- `ineligibility_reason`;
- `source_hash_summary`.

Null metric values stay null. Do not coerce missing metrics to zero.

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

All tracker implementations must be no-throw from the caller's perspective. The MLflow tracker catches its own failures and records them as warnings.

## MLflow Integration

MLflow is optional and additive.

### Tracking Store

Use a local file-based tracking store by default:

```text
mlruns/
```

The path must be gitignored. `MLFLOW_TRACKING_URI` may override it.

Do not place MLflow's tracking store inside a specific experiment output directory. The report tree and the tracking store have different lifecycles.

### Experiment And Run Shape

Use one stable MLflow experiment per experiment group:

- `cartola/production-parity`;
- `cartola/matchup-research`.

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
- `mlflow_logging_status`.

Dependency versions should be logged when available:

- pandas;
- numpy;
- scikit-learn;
- PuLP;
- plotly;
- mlflow;
- Python.

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
- `optimizer/oracle_gap`;
- `optimizer/capture_rate`;
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
2. Experiment index starts or updates an experiment row.
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
- child-run parallelism;
- model hyperparameter search;
- promotion of a new production model/feature pack;
- live recommendation integration for any winning pipeline.
