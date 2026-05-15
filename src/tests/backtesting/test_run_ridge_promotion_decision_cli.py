from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_ridge_promotion_decision.py"
SPEC = importlib.util.spec_from_file_location("run_ridge_promotion_decision", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_accepts_candidate_control_and_baseline() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "experiment",
            "--candidate-model",
            "ridge",
            "--candidate-feature-pack",
            "ppg_xg",
            "--control-model",
            "xgboost_depth2_l2_heavy",
            "--control-feature-pack",
            "ppg_xg",
            "--baseline-model",
            "random_forest",
            "--baseline-feature-pack",
            "ppg",
            "--promotion-seasons",
            "2020,2021,2022,2023,2024,2025",
        ]
    )

    assert args.experiment_path == Path("experiment")
    assert args.candidate_model == "ridge"
    assert args.candidate_feature_pack == "ppg_xg"
    assert args.control_model == "xgboost_depth2_l2_heavy"
    assert args.control_feature_pack == "ppg_xg"
    assert args.baseline_model == "random_forest"
    assert args.baseline_feature_pack == "ppg"
    assert args.promotion_seasons == "2020,2021,2022,2023,2024,2025"


def test_main_writes_json_and_markdown_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()

    def fake_write(**kwargs: object) -> Path:
        assert kwargs["experiment_path"] == experiment
        json_path = experiment / "ridge_promotion_decision.json"
        json_path.write_text(json.dumps({"decision_status": "promote_candidate"}), encoding="utf-8")
        (experiment / "ridge_promotion_decision.md").write_text("# Ridge Promotion Decision\n", encoding="utf-8")
        return json_path

    monkeypatch.setattr(cli, "write_ridge_promotion_decision", fake_write)

    exit_code = cli.main(
        [
            "--experiment-path",
            str(experiment),
            "--candidate-model",
            "ridge",
            "--candidate-feature-pack",
            "ppg_xg",
            "--control-model",
            "xgboost_depth2_l2_heavy",
            "--control-feature-pack",
            "ppg_xg",
            "--baseline-model",
            "random_forest",
            "--baseline-feature-pack",
            "ppg",
            "--promotion-seasons",
            "2020,2021,2022,2023,2024,2025",
        ]
    )

    assert exit_code == 0
    assert (experiment / "ridge_promotion_decision.json").is_file()
    assert (experiment / "ridge_promotion_decision.md").is_file()


def test_main_exits_nonzero_on_missing_experiment_path(tmp_path: Path) -> None:
    exit_code = cli.main(
        [
            "--experiment-path",
            str(tmp_path / "missing"),
            "--candidate-model",
            "ridge",
            "--candidate-feature-pack",
            "ppg_xg",
            "--control-model",
            "xgboost_depth2_l2_heavy",
            "--control-feature-pack",
            "ppg_xg",
            "--baseline-model",
            "random_forest",
            "--baseline-feature-pack",
            "ppg",
            "--promotion-seasons",
            "2020,2021,2022,2023,2024,2025",
        ]
    )

    assert exit_code == 1
