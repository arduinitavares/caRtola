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
