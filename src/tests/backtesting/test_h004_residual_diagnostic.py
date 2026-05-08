from __future__ import annotations

import json
from pathlib import Path

import pytest

from cartola.backtesting.h004_residual_diagnostic import (
    H004_CONTROL_FEATURE_PACK,
    H004_CONTROL_MODEL_ID,
    H004_PRIMARY_SCORE_COLUMN,
    H004SourceChild,
    discover_h004_source_children,
)


def _write_child(tmp_path: Path, *, season: int = 2025) -> Path:
    child = (
        tmp_path
        / "experiment"
        / "runs"
        / f"season={season}"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
    )
    child.mkdir(parents=True)
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": season,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "scoring_contract_version": "cartola_standard_2026_v1",
                "budget_policy": "moving",
                "fixture_identity_status": "verified",
                "footystats_matches_source_sha256": "footy-sha",
            }
        ),
        encoding="utf-8",
    )
    return child


def test_discover_h004_source_children_derives_season_from_context_not_prediction_csv(tmp_path: Path) -> None:
    child = _write_child(tmp_path, season=2025)

    children = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )

    assert children == (
        H004SourceChild(
            season=2025,
            model_id="xgboost_depth2_slow",
            feature_pack="ppg_xg_matchup",
            child_path=child,
            score_column=H004_PRIMARY_SCORE_COLUMN,
            fixture_mode="exploratory",
            matchup_context_mode="cartola_matchup_v1",
            footystats_mode="ppg_xg",
            fixture_identity_status="verified",
            footystats_source_identity={"footystats_matches_source_sha256": "footy-sha"},
        ),
    )


def test_discover_h004_source_children_fails_when_child_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="season=2025.*xgboost_depth2_slow.*ppg_xg_matchup"):
        discover_h004_source_children(
            experiment_path=tmp_path / "experiment",
            seasons=(2025,),
            model_id=H004_CONTROL_MODEL_ID,
            feature_pack=H004_CONTROL_FEATURE_PACK,
        )
