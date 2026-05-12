from pathlib import Path

import pytest

from cartola.backtesting.experiment_config import (
    FeaturePack,
    build_child_run_specs,
    config_hash,
    experiment_id,
    feature_pack_to_modes,
)


def test_production_parity_matrix() -> None:
    specs = build_child_run_specs(
        group="production-parity",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 24
    assert {spec.fixture_mode for spec in specs} == {"none"}
    assert {spec.feature_pack for spec in specs} == {"ppg", "ppg_xg"}


def test_matchup_research_matrix() -> None:
    specs = build_child_run_specs(
        group="matchup-research",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 48
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.feature_pack for spec in specs} == {
        "ppg",
        "ppg_xg",
        "ppg_matchup",
        "ppg_xg_matchup",
    }


def test_xgboost_research_matrix_uses_fixed_candidates_only() -> None:
    specs = build_child_run_specs(
        group="xgboost-research",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 18
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.feature_pack for spec in specs} == {"ppg_xg", "ppg_xg_matchup"}
    assert {spec.model_id for spec in specs} == {
        "xgboost_conservative",
        "xgboost_balanced",
        "xgboost_capacity",
    }


def test_xgboost_sensitivity_v2_matrix_uses_controls_and_local_candidates() -> None:
    specs = build_child_run_specs(
        group="xgboost-sensitivity-v2",
        seasons=(2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 33
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.feature_pack for spec in specs} == {"ppg_xg_matchup"}
    assert {spec.model_id for spec in specs} == {
        "ridge",
        "xgboost_conservative",
        "xgboost_depth1_stumps",
        "xgboost_depth2_slow",
        "xgboost_depth2_fast",
        "xgboost_depth2_more_trees",
        "xgboost_depth2_heavy_child",
        "xgboost_depth2_subsample",
        "xgboost_depth2_l2_heavy",
        "xgboost_depth2_l1_gamma",
        "xgboost_depth3_slow",
    }


def test_h004_feature_pack_to_modes() -> None:
    feature_pack = feature_pack_to_modes("ppg_xg_matchup_h004")

    assert feature_pack == FeaturePack(
        feature_pack="ppg_xg_matchup_h004",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
        feature_augmentation_mode="h004_attack_defense_v1",
    )


def test_h004_attack_defense_mismatch_matrix_is_control_vs_challenger_only() -> None:
    specs = build_child_run_specs(
        group="h004-attack-defense-mismatch",
        seasons=(2021, 2022, 2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )

    assert len(specs) == 10
    assert {spec.fixture_mode for spec in specs} == {"exploratory"}
    assert {spec.model_id for spec in specs} == {"xgboost_depth2_slow"}
    assert {spec.feature_pack for spec in specs} == {
        "ppg_xg_matchup",
        "ppg_xg_matchup_h004",
    }
    h004_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup_h004"]
    control_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup"]
    assert {spec.backtest_config.feature_augmentation_mode for spec in h004_specs} == {
        "h004_attack_defense_v1"
    }
    assert {spec.backtest_config.feature_augmentation_mode for spec in control_specs} == {"none"}


def test_h005_feature_pack_to_modes() -> None:
    feature_pack = feature_pack_to_modes("ppg_xg_matchup_h005")

    assert feature_pack.feature_pack == "ppg_xg_matchup_h005"
    assert feature_pack.footystats_mode == "ppg_xg"
    assert feature_pack.matchup_context_mode == "cartola_matchup_v1"
    assert feature_pack.feature_augmentation_mode == "h005_matchup_reliability_v1"


def test_h005_count_aware_matchup_reliability_matrix_is_control_vs_challenger_only() -> None:
    specs = build_child_run_specs(
        group="h005-count-aware-matchup-shrinkage",
        seasons=(2021, 2022, 2023, 2024, 2025),
        start_round=5,
        budget=100.0,
        project_root=Path("."),
        current_year=2026,
        jobs=12,
        output_root=Path("out"),
    )

    assert len(specs) == 10
    assert {spec.model_id for spec in specs} == {"xgboost_depth2_slow"}
    assert {spec.feature_pack for spec in specs} == {"ppg_xg_matchup", "ppg_xg_matchup_h005"}
    h005_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup_h005"]
    control_specs = [spec for spec in specs if spec.feature_pack == "ppg_xg_matchup"]
    assert {spec.backtest_config.feature_augmentation_mode for spec in h005_specs} == {
        "h005_matchup_reliability_v1"
    }
    assert {spec.backtest_config.feature_augmentation_mode for spec in control_specs} == {"none"}


def test_build_child_run_specs_can_include_only_selected_models() -> None:
    specs = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
        models=("ridge",),
    )

    assert len(specs) == 2
    assert {spec.model_id for spec in specs} == {"ridge"}


def test_build_child_run_specs_can_exclude_models() -> None:
    specs = build_child_run_specs(
        group="matchup-research",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
        exclude_models=("hist_gradient_boosting",),
    )

    assert len(specs) == 12
    assert "hist_gradient_boosting" not in {spec.model_id for spec in specs}


def test_build_child_run_specs_rejects_empty_model_filter() -> None:
    with pytest.raises(ValueError, match="At least one model must remain"):
        build_child_run_specs(
            group="production-parity",
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/model_feature/test"),
            current_year=2026,
            jobs=12,
            models=("ridge",),
            exclude_models=("ridge",),
        )


def test_feature_pack_to_modes() -> None:
    assert feature_pack_to_modes("ppg") == FeaturePack(
        feature_pack="ppg",
        footystats_mode="ppg",
        matchup_context_mode="none",
    )
    assert feature_pack_to_modes("ppg_xg_matchup") == FeaturePack(
        feature_pack="ppg_xg_matchup",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
    )


def test_feature_pack_to_modes_rejects_unknown_pack() -> None:
    with pytest.raises(ValueError, match="Unsupported feature_pack"):
        feature_pack_to_modes("bad_pack")  # type: ignore[arg-type]


def test_build_child_run_specs_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment group"):
        build_child_run_specs(
            group="bad_group",  # type: ignore[arg-type]
            seasons=(2025,),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/model_feature/test"),
            current_year=2026,
            jobs=12,
        )


def test_experiment_rejects_live_year() -> None:
    with pytest.raises(ValueError, match="Experiment seasons must be before current_year"):
        build_child_run_specs(
            group="production-parity",
            seasons=(2025, 2026),
            start_round=5,
            budget=100.0,
            project_root=Path("/repo"),
            output_root=Path("data/08_reporting/experiments/model_feature/test"),
            current_year=2026,
            jobs=12,
        )


def test_child_paths_are_deterministic() -> None:
    spec = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]

    assert spec.output_path == Path(
        "/repo/data/08_reporting/experiments/model_feature/test/runs/"
        "season=2025/model=random_forest/feature_pack=ppg"
    )
    assert spec.backtest_config.output_path == spec.output_path


def test_config_hash_changes_for_material_fields() -> None:
    base = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=5,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]
    changed = build_child_run_specs(
        group="production-parity",
        seasons=(2025,),
        start_round=6,
        budget=100.0,
        project_root=Path("/repo"),
        output_root=Path("data/08_reporting/experiments/model_feature/test"),
        current_year=2026,
        jobs=12,
    )[0]

    assert config_hash(base.config_identity) != config_hash(changed.config_identity)


def test_experiment_id_includes_group_and_hash() -> None:
    value = experiment_id(
        group="production-parity",
        started_at_utc="20260430T200000000000Z",
        matrix_hash="abcdef1234567890",
    )

    assert value == "group=production-parity__started_at=20260430T200000000000Z__matrix=abcdef123456"
