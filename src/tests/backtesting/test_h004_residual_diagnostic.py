from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.h004_residual_diagnostic import (
    H004_CONTROL_FEATURE_PACK,
    H004_CONTROL_MODEL_ID,
    H004_PRIMARY_SCORE_COLUMN,
    H004PredictionBundle,
    H004SourceChild,
    build_h004_residual_correlations,
    build_h004_residual_quintiles,
    discover_h004_source_children,
    load_h004_prediction_bundle,
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


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "id_atleta": [1, 2, 3],
            "posicao": ["ata", "mei", "tec"],
            "id_clube": [10, 20, 30],
            "pontuacao": [8.0, 2.0, None],
            "entrou_em_campo": [True, True, False],
            "xgboost_depth2_slow_score": [5.0, 4.0, 3.0],
            "matchup_is_home": [1, 0, 1],
            "footystats_xg_diff": [0.6, -0.2, 0.1],
            "footystats_ppg_diff": [0.8, -0.4, 0.2],
            "matchup_opponent_allowed_points_roll5": [4.0, 5.0, 3.0],
            "matchup_opponent_allowed_position_points_roll5": [6.0, 4.0, 3.0],
            "matchup_club_position_points_roll5": [7.0, 3.5, 2.0],
            "matchup_opponent_allowed_position_count": [5, 5, 0],
            "matchup_club_position_count": [5, 5, 0],
            "position_points_prior": [4.0, 3.0, 2.0],
        }
    )


def _selected_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": [5],
            "id_atleta": [1],
            "posicao": ["ata"],
            "pontuacao": [8.0],
            "entrou_em_campo": [True],
        }
    )


def _played_signal_rows() -> pd.DataFrame:
    rows = []
    for index in range(120):
        rows.append(
            {
                "season": 2025,
                "posicao": "ata",
                "prediction_residual": float(index) / 20.0,
                "footystats_xg_diff": float(index) / 100.0,
                "matchup_opponent_allowed_position_points_roll5": float(index) / 30.0,
                "matchup_is_home": 1,
            }
        )
    return pd.DataFrame(rows)


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


def test_load_h004_prediction_bundle_adds_context_season_and_residuals(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    _prediction_rows().to_csv(child_path / "player_predictions.csv", index=False)
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)
    child = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )[0]

    bundle = load_h004_prediction_bundle(child)

    assert isinstance(bundle, H004PredictionBundle)
    assert bundle.played["season"].tolist() == [2025, 2025]
    assert bundle.played["predicted_points"].tolist() == [5.0, 4.0]
    assert bundle.played["prediction_residual"].tolist() == [3.0, -2.0]
    assert bundle.dnp["id_atleta"].tolist() == [3]
    assert bundle.selected_players["season"].tolist() == [2025]
    assert bundle.selected_players["id_atleta"].tolist() == [1]


def test_load_h004_prediction_bundle_fails_for_missing_score_column(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    _prediction_rows().drop(columns=["xgboost_depth2_slow_score"]).to_csv(
        child_path / "player_predictions.csv",
        index=False,
    )
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)
    child = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )[0]

    with pytest.raises(ValueError, match="xgboost_depth2_slow_score"):
        load_h004_prediction_bundle(child)


def test_load_h004_prediction_bundle_keeps_string_false_rows_as_dnp(tmp_path: Path) -> None:
    child_path = _write_child(tmp_path, season=2025)
    predictions = _prediction_rows()
    predictions["entrou_em_campo"] = predictions["entrou_em_campo"].astype(object)
    predictions.loc[0, "entrou_em_campo"] = "true"
    predictions.loc[1, "entrou_em_campo"] = "1"
    predictions.loc[2, "entrou_em_campo"] = "False"
    predictions.to_csv(child_path / "player_predictions.csv", index=False)
    _selected_rows().to_csv(child_path / "selected_players.csv", index=False)
    child = discover_h004_source_children(
        experiment_path=tmp_path / "experiment",
        seasons=(2025,),
        model_id=H004_CONTROL_MODEL_ID,
        feature_pack=H004_CONTROL_FEATURE_PACK,
    )[0]

    bundle = load_h004_prediction_bundle(child)

    assert bundle.dnp["id_atleta"].tolist() == [3]
    assert bundle.played["id_atleta"].tolist() == [1, 2]


def test_build_h004_residual_correlations_requires_minimum_rows_and_flags_signal() -> None:
    correlations = build_h004_residual_correlations(_played_signal_rows())

    row = correlations[
        correlations["context_column"].eq("footystats_xg_diff")
        & correlations["position"].eq("ata")
        & correlations["season"].eq(2025)
    ].iloc[0]
    assert row["row_count"] == 120
    assert row["spearman"] > 0.99
    assert bool(row["passes_signal"])


def test_build_h004_residual_quintiles_outputs_deterministic_quintile_rows() -> None:
    quintiles = build_h004_residual_quintiles(_played_signal_rows())

    subset = quintiles[
        quintiles["context_column"].eq("footystats_xg_diff")
        & quintiles["position"].eq("ata")
        & quintiles["season"].eq(2025)
    ]
    assert subset["quintile"].tolist() == [1, 2, 3, 4, 5]
    assert subset["row_count"].sum() == 120
