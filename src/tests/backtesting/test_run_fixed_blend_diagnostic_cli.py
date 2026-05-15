from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_fixed_blend_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("run_fixed_blend_diagnostic", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_parse_args_builds_fixed_blend_defaults() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "data/experiment",
            "--blend",
            "xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("data/experiment")
    assert args.seasons == "2021,2022,2023,2024,2025"
    assert args.promotion_seasons is None
    assert args.feature_pack == "ppg_xg"
    assert args.control_model == "xgboost_depth2_l2_heavy"
    assert args.blend == ["xgb90_ridge10=xgboost_depth2_l2_heavy:0.9,ridge:0.1"]
    assert args.initial_budget == 100.0
    assert args.current_year == 2026
    assert args.output_root == Path("data/08_reporting/blend_diagnostics")


def test_parse_args_accepts_repeated_blends_and_overrides() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "experiment",
            "--seasons",
            "2024,2025",
            "--promotion-seasons",
            "2025",
            "--feature-pack",
            "ppg",
            "--control-model",
            "model_a",
            "--blend",
            "blend_a=model_a:0.8,model_b:0.2",
            "--blend",
            "blend_b=model_a:0.7,model_b:0.3",
            "--initial-budget",
            "80",
            "--current-year",
            "2026",
            "--output-root",
            "out",
        ]
    )

    assert args.seasons == "2024,2025"
    assert args.promotion_seasons == "2025"
    assert args.feature_pack == "ppg"
    assert args.control_model == "model_a"
    assert args.blend == ["blend_a=model_a:0.8,model_b:0.2", "blend_b=model_a:0.7,model_b:0.3"]
    assert args.initial_budget == 80.0
    assert args.output_root == Path("out")
