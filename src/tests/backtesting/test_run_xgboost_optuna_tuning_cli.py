from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_xgboost_optuna_tuning.py"
SPEC = importlib.util.spec_from_file_location("run_xgboost_optuna_tuning", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_accepts_bounded_tuning_arguments() -> None:
    args = cli.parse_args(
        [
            "--source-experiment-path",
            "source",
            "--seasons",
            "2020,2021,2022,2023,2024,2025",
            "--n-trials",
            "40",
            "--study-seed",
            "17",
            "--study-name",
            "m009-resume",
            "--control-model",
            "xgboost_depth2_l2_heavy",
            "--control-feature-pack",
            "ppg_xg",
            "--feature-pack",
            "ppg_xg",
            "--current-year",
            "2026",
        ]
    )

    assert args.source_experiment_path == Path("source")
    assert args.seasons == "2020,2021,2022,2023,2024,2025"
    assert args.n_trials == 40
    assert args.study_seed == 17
    assert args.study_name == "m009-resume"
    assert args.control_model == "xgboost_depth2_l2_heavy"
    assert args.control_feature_pack == "ppg_xg"
    assert args.feature_pack == "ppg_xg"
    assert args.current_year == 2026


def test_main_writes_tuning_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "tuning"

    def fake_run(**kwargs: object) -> Path:
        assert kwargs["source_experiment_path"] == source
        assert kwargs["output_root"] == output
        assert kwargs["study_name"] == "xgboost_optuna_tuning"
        output.mkdir()
        (output / "xgboost_optuna_tuning.json").write_text(
            json.dumps({"best_trial_number": 0, "best_objective_score": 12.3}),
            encoding="utf-8",
        )
        (output / "optuna_trials.csv").write_text("trial_number,objective_score\n0,12.3\n", encoding="utf-8")
        return output

    monkeypatch.setattr(cli, "run_xgboost_optuna_tuning", fake_run)

    exit_code = cli.main(
        [
            "--source-experiment-path",
            str(source),
            "--seasons",
            "2020",
            "--n-trials",
            "1",
            "--output-root",
            str(output),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert (output / "xgboost_optuna_tuning.json").is_file()
    assert (output / "optuna_trials.csv").is_file()


def test_main_exits_nonzero_on_missing_source_experiment(tmp_path: Path) -> None:
    exit_code = cli.main(
        [
            "--source-experiment-path",
            str(tmp_path / "missing"),
            "--seasons",
            "2020",
            "--n-trials",
            "1",
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 1
