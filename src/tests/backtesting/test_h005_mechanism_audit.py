from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from cartola.backtesting.h005_mechanism_audit import (
    H005MechanismAuditError,
    build_h005_mechanism_audit,
    discover_h005_source_children,
)


def test_discover_h005_source_children_requires_one_child_per_season(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2021,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
            }
        ),
        encoding="utf-8",
    )

    children = discover_h005_source_children(
        experiment_path=experiment,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
    )

    assert len(children) == 1
    assert children[0].season == 2021


def test_discover_h005_source_children_rejects_metadata_mismatch(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    (child / "run_metadata.json").write_text(
        json.dumps({"season": 2021, "model_id": "ridge", "feature_pack": "ppg_xg_matchup"}),
        encoding="utf-8",
    )

    with pytest.raises(H005MechanismAuditError, match="metadata mismatch"):
        discover_h005_source_children(
            experiment_path=experiment,
            seasons=(2021,),
            model_id="xgboost_depth2_slow",
            feature_pack="ppg_xg_matchup",
        )


def test_h005_mechanism_audit_invalidates_recomputed_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"

    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season(),
    )
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_fixtures",
        lambda season, project_root: _fixtures(),
    )

    predictions = pd.read_csv(
        experiment
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
        / "player_predictions.csv"
    )
    predictions["matchup_opponent_allowed_position_count"] = 99
    predictions.to_csv(
        experiment
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
        / "player_predictions.csv",
        index=False,
    )

    result = build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )

    assert result.decision["audit_status"] == "invalid"
    failed_checks = cast("list[str]", result.decision["failed_checks"])
    assert "recomputed_count_mismatch" in failed_checks


def test_h005_mechanism_audit_writes_required_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season(),
    )
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_fixtures",
        lambda season, project_root: _fixtures(),
    )

    result = build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )

    assert result.output_path == output_path
    assert (output_path / "h005_mechanism_audit.csv").is_file()
    assert (output_path / "h005_raw_count_audit.csv").is_file()
    assert (output_path / "h005_mechanism_audit_decision.json").is_file()


def test_h005_mechanism_audit_artifacts_include_decision_contract_and_summary_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season(),
    )
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_fixtures",
        lambda season, project_root: _fixtures(),
    )

    build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )

    decision = json.loads((output_path / "h005_mechanism_audit_decision.json").read_text(encoding="utf-8"))
    ratio_audit = pd.read_csv(output_path / "h005_mechanism_audit.csv")
    raw_count_audit = pd.read_csv(output_path / "h005_raw_count_audit.csv")

    assert decision["hypothesis_id"] == "H005"
    assert decision["audit_status"] == "mixed_or_weak"
    assert decision["failed_checks"] == []
    assert decision["manual_points_shrinkage"] is False
    assert decision["h005_design_revision"] == "reliability_v1"
    assert {
        "row_count",
        "round_count",
        "source_residual_mean",
        "source_overprediction_rate",
        "source_base_count_mean",
        "h005_available_match_count_mean",
        "h005_expected_count_mean",
        "h005_count_ratio_mean",
        "source_position_points_mean",
        "source_all_points_mean",
        "position_allowed_delta_mean",
    }.issubset(ratio_audit.columns)
    assert "raw_count_bin" in raw_count_audit.columns


def _write_source_experiment(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "rodada": 3,
                "id_atleta": 301,
                "apelido": "LAT",
                "id_clube": 10,
                "posicao": "lat",
                "status": "Provavel",
                "preco_pre_rodada": 10.0,
                "pontuacao": 4.0,
                "entrou_em_campo": True,
                "xgboost_depth2_slow_score": 5.0,
                "matchup_opponent_allowed_position_count": 1,
                "matchup_opponent_allowed_position_points_roll5": 5.0,
                "matchup_opponent_allowed_points_roll5": 6.0,
            }
        ]
    ).to_csv(child / "player_predictions.csv", index=False)
    (child / "selected_players.csv").write_text(
        "rodada,id_atleta,entrou_em_campo\n3,301,true\n",
        encoding="utf-8",
    )
    (child / "run_metadata.json").write_text(
        json.dumps(
            {
                "season": 2021,
                "model_id": "xgboost_depth2_slow",
                "feature_pack": "ppg_xg_matchup",
                "fixture_mode": "exploratory",
                "matchup_context_mode": "cartola_matchup_v1",
                "footystats_mode": "ppg_xg",
                "fixture_source_sha256": {"fixture.csv": "fixture-sha"},
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _raw_season() -> pd.DataFrame:
    rows = [
        _raw_player(1, 101, 10, "lat", 5.0),
        _raw_player(1, 201, 20, "gol", 3.0),
        _raw_player(2, 102, 10, "ata", 7.0),
        _raw_player(2, 202, 20, "gol", 4.0),
        _raw_player(3, 301, 10, "lat", 4.0),
    ]
    frame = pd.DataFrame(rows)
    for scout in ("G", "A", "DS", "V"):
        if scout not in frame.columns:
            frame[scout] = 0
    return frame


def _raw_player(round_number: int, athlete_id: int, club_id: int, position: str, points: float) -> dict[str, object]:
    return {
        "rodada": round_number,
        "id_atleta": athlete_id,
        "apelido": str(athlete_id),
        "slug": str(athlete_id),
        "posicao": position,
        "status": "Provavel",
        "preco": 10.0,
        "preco_pre_rodada": 10.0,
        "pontuacao": points,
        "media": points,
        "num_jogos": 1,
        "variacao": 0.0,
        "id_clube": club_id,
        "nome_clube": str(club_id),
        "entrou_em_campo": True,
    }


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 1, "id_clube_home": 10, "id_clube_away": 20, "data": "2021-01-01"},
            {"rodada": 2, "id_clube_home": 20, "id_clube_away": 10, "data": "2021-01-08"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 20, "data": "2021-01-15"},
        ]
    )
