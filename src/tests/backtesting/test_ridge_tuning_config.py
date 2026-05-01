from pathlib import Path

import pytest

from cartola.backtesting.ridge_tuning_config import (
    PRIMARY_INCUMBENT_CANDIDATE_ID,
    RIDGE_ALPHA_VALUES,
    RIDGE_TUNING_FEATURE_PACKS,
    build_ridge_tuning_specs,
    candidate_id_for,
)


def test_fixed_ridge_alpha_values() -> None:
    assert RIDGE_ALPHA_VALUES == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def test_fixed_ridge_tuning_feature_packs() -> None:
    assert RIDGE_TUNING_FEATURE_PACKS == ("ppg", "ppg_xg")


@pytest.mark.parametrize(
    ("alpha", "feature_pack", "expected"),
    [
        (0.01, "ppg", "ridge_alpha_0_01__ppg"),
        (1.0, "ppg_xg", "ridge_alpha_1_0__ppg_xg"),
        (300.0, "ppg_xg", "ridge_alpha_300_0__ppg_xg"),
    ],
)
def test_candidate_id_for_encodes_alpha(alpha: float, feature_pack: str, expected: str) -> None:
    assert candidate_id_for(alpha=alpha, feature_pack=feature_pack) == expected


def test_build_ridge_tuning_specs_builds_full_matrix() -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
        current_year=2026,
        jobs=12,
        stage="screen",
    )

    assert len(specs) == 3 * len(RIDGE_ALPHA_VALUES) * len(RIDGE_TUNING_FEATURE_PACKS)
    assert {spec.season for spec in specs} == {2023, 2024, 2025}
    assert {spec.alpha for spec in specs} == set(RIDGE_ALPHA_VALUES)
    assert {spec.feature_pack for spec in specs} == set(RIDGE_TUNING_FEATURE_PACKS)


def test_build_ridge_tuning_specs_sets_ridge_stage_and_incumbent() -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
        current_year=2026,
        jobs=12,
        stage="final",
    )

    assert {spec.model_id for spec in specs} == {"ridge"}
    assert {spec.stage for spec in specs} == {"final"}
    assert {spec.candidate_id for spec in specs} >= {PRIMARY_INCUMBENT_CANDIDATE_ID}

    incumbent = next(spec for spec in specs if spec.candidate_id == PRIMARY_INCUMBENT_CANDIDATE_ID)
    assert incumbent.backtest_config.fixture_mode == "none"
    assert incumbent.backtest_config.matchup_context_mode == "none"
    assert incumbent.backtest_config.footystats_mode == "ppg_xg"
    assert incumbent.backtest_config.output_path == Path(
        "/repo/data/08_reporting/experiments/ridge_tuning/test/runs/"
        "stage=final/season=2025/candidate=ridge_alpha_1_0__ppg_xg"
    )


def test_build_ridge_tuning_specs_rejects_tuning_current_year() -> None:
    with pytest.raises(ValueError, match="Tuning seasons must be before current_year"):
        build_ridge_tuning_specs(
            seasons=(2025, 2026),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
            current_year=2026,
            jobs=12,
            stage="screen",
        )


def test_distinct_alphas_have_distinct_model_params_hashes_for_feature_pack() -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
        current_year=2026,
        jobs=12,
        stage="screen",
        candidate_ids={
            "ridge_alpha_0_01__ppg",
            "ridge_alpha_0_03__ppg",
            "ridge_alpha_0_1__ppg",
        },
    )

    assert len({spec.model_params_hash for spec in specs}) == 3


def test_build_ridge_tuning_specs_rejects_unknown_candidate_id() -> None:
    with pytest.raises(ValueError, match="Unknown ridge tuning candidate_id: missing_candidate"):
        build_ridge_tuning_specs(
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
            current_year=2026,
            jobs=12,
            stage="final",
            candidate_ids={"ridge_alpha_1_0__ppg_xg", "missing_candidate"},
        )


def test_specs_share_one_tuning_generation_hash() -> None:
    specs = build_ridge_tuning_specs(
        seasons=(2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/ridge_tuning/test"),
        current_year=2026,
        jobs=12,
        stage="screen",
    )

    assert len({spec.tuning_generation_hash for spec in specs}) == 1
