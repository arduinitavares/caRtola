from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from cartola.backtesting.h005_mechanism_audit import (
    H005MechanismAuditError,
    _support_gate,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixture_mode", "none"),
        ("matchup_context_mode", "none"),
        ("footystats_mode", "ppg"),
    ],
)
def test_discover_h005_source_children_rejects_source_context_metadata_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    metadata = {
        "season": 2021,
        "model_id": "xgboost_depth2_slow",
        "feature_pack": "ppg_xg_matchup",
        "fixture_mode": "exploratory",
        "matchup_context_mode": "cartola_matchup_v1",
        "footystats_mode": "ppg_xg",
    }
    metadata[field] = value
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(H005MechanismAuditError, match="metadata mismatch"):
        discover_h005_source_children(
            experiment_path=experiment,
            seasons=(2021,),
            model_id="xgboost_depth2_slow",
            feature_pack="ppg_xg_matchup",
        )


def test_discover_h005_source_children_rejects_feature_augmentation_metadata_mismatch(tmp_path: Path) -> None:
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
                "feature_augmentation_mode": "h005_matchup_reliability_v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(H005MechanismAuditError, match="metadata mismatch"):
        discover_h005_source_children(
            experiment_path=experiment,
            seasons=(2021,),
            model_id="xgboost_depth2_slow",
            feature_pack="ppg_xg_matchup",
        )


def test_h005_support_gate_supports_reliability_when_all_phase0_criteria_pass() -> None:
    mechanism_audit = _support_gate_mechanism_audit()
    raw_count_audit = _support_gate_raw_count_audit()

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "supports_reliability_hypothesis"
    )


def test_h005_support_gate_rejects_weak_low_vs_normal_residual_spread() -> None:
    mechanism_audit = _support_gate_mechanism_audit(low_residual=0.12, normal_residual=0.05)
    raw_count_audit = _support_gate_raw_count_audit()

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "mixed_or_weak"
    )


def test_h005_support_gate_rejects_single_season_low_reliability_concentration() -> None:
    mechanism_audit = _support_gate_mechanism_audit(low_rows_by_season={2021: 901, 2022: 600, 2023: 600})
    raw_count_audit = _support_gate_raw_count_audit()

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "mixed_or_weak"
    )


def test_h005_support_gate_rejects_when_raw_count_spread_exceeds_ratio_spread() -> None:
    mechanism_audit = _support_gate_mechanism_audit(low_residual=0.30, normal_residual=0.05)
    raw_count_audit = _support_gate_raw_count_audit(low_residual=0.50, normal_residual=0.05)

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "mixed_or_weak"
    )


def test_h005_support_gate_requires_four_positions_with_total_rows_at_least_500() -> None:
    mechanism_audit = _support_gate_mechanism_audit(total_rows_by_position={"mei": 499})
    raw_count_audit = _support_gate_raw_count_audit()

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "mixed_or_weak"
    )


def test_h005_support_gate_rejects_position_imbalanced_raw_normal_support() -> None:
    mechanism_audit = _support_gate_mechanism_audit()
    raw_count_audit = _support_gate_raw_count_audit(normal_positions=("gol",), normal_row_count=600)

    assert (
        _support_gate(
            failed_checks=set(),
            mechanism_audit=mechanism_audit,
            raw_count_audit=raw_count_audit,
        )
        == "mixed_or_weak"
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


def test_h005_mechanism_audit_duplicate_source_keys_invalidates_without_crashing(
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
    predictions_path = (
        experiment
        / "runs"
        / "season=2021"
        / "model=xgboost_depth2_slow"
        / "feature_pack=ppg_xg_matchup"
        / "player_predictions.csv"
    )
    predictions = pd.read_csv(predictions_path)
    pd.concat([predictions, predictions], ignore_index=True).to_csv(predictions_path, index=False)

    result = build_h005_mechanism_audit(
        experiment_path=experiment,
        output_path=output_path,
        seasons=(2021,),
        model_id="xgboost_depth2_slow",
        feature_pack="ppg_xg_matchup",
        project_root=tmp_path,
    )
    decision = json.loads((output_path / "h005_mechanism_audit_decision.json").read_text(encoding="utf-8"))

    assert result.decision["audit_status"] == "invalid"
    failed_checks = cast("list[str]", result.decision["failed_checks"])
    assert "row_identity_mismatch" in failed_checks
    assert decision["audit_status"] == "invalid"
    assert "row_identity_mismatch" in decision["failed_checks"]


def test_h005_mechanism_audit_source_row_order_does_not_invalidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_two_player_source_experiment(tmp_path, reverse_order=True)
    output_path = tmp_path / "audit"
    monkeypatch.setattr(
        "cartola.backtesting.h005_mechanism_audit.load_season_data",
        lambda season, project_root: _raw_season_with_two_round3_players(),
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

    failed_checks = cast("list[str]", result.decision["failed_checks"])
    assert "row_identity_mismatch" not in failed_checks
    assert result.decision["audit_status"] == "mixed_or_weak"


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
    assert decision["source_prediction_artifacts"][0]["path"].endswith("player_predictions.csv")
    assert decision["source_prediction_artifacts"][0]["sha256"]
    assert decision["raw_season_artifacts"][0]["season"] == 2021
    assert decision["raw_season_artifacts"][0]["sha256"]
    assert decision["raw_season_artifacts"][0]["status"] == "dataframe_hash"
    assert decision["raw_season_artifacts"][0]["paths"] == []
    assert decision["fixture_artifacts"][0]["season"] == 2021
    assert decision["fixture_artifacts"][0]["sha256"]
    assert decision["fixture_artifacts"][0]["status"] == "dataframe_hash"
    assert decision["fixture_artifacts"][0]["paths"] == []
    assert decision["recomputed_count_match_status"] == "ok"
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


def test_h005_mechanism_audit_provenance_uses_file_hash_when_raw_and_fixture_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _write_source_experiment(tmp_path)
    output_path = tmp_path / "audit"
    raw_file = tmp_path / "data" / "01_raw" / "2021" / "rodada-1.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("rodada,id_atleta\n1,101\n", encoding="utf-8")
    fixture_file = tmp_path / "data" / "01_raw" / "fixtures" / "2021" / "partidas-1.csv"
    fixture_file.parent.mkdir(parents=True)
    fixture_file.write_text("rodada,id_clube_home,id_clube_away,data\n1,10,20,2021-01-01\n", encoding="utf-8")
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

    assert decision["raw_season_artifacts"][0]["status"] == "file_hash"
    assert decision["raw_season_artifacts"][0]["paths"] == [str(raw_file)]
    assert decision["fixture_artifacts"][0]["status"] == "file_hash"
    assert decision["fixture_artifacts"][0]["paths"] == [str(fixture_file)]
    assert decision["recomputed_count_match_status"] == "ok"


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


def _write_two_player_source_experiment(tmp_path: Path, *, reverse_order: bool) -> Path:
    experiment = tmp_path / "experiment"
    child = experiment / "runs" / "season=2021" / "model=xgboost_depth2_slow" / "feature_pack=ppg_xg_matchup"
    child.mkdir(parents=True)
    rows = [
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
        },
        {
            "rodada": 3,
            "id_atleta": 302,
            "apelido": "GOL",
            "id_clube": 20,
            "posicao": "gol",
            "status": "Provavel",
            "preco_pre_rodada": 10.0,
            "pontuacao": 2.0,
            "entrou_em_campo": True,
            "xgboost_depth2_slow_score": 3.0,
            "matchup_opponent_allowed_position_count": 2,
            "matchup_opponent_allowed_position_points_roll5": 3.5,
            "matchup_opponent_allowed_points_roll5": 3.5,
        },
    ]
    if reverse_order:
        rows = list(reversed(rows))
    pd.DataFrame(rows).to_csv(child / "player_predictions.csv", index=False)
    (child / "selected_players.csv").write_text(
        "rodada,id_atleta,entrou_em_campo\n3,301,true\n3,302,true\n",
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


def _raw_season_with_two_round3_players() -> pd.DataFrame:
    return pd.concat(
        [
            _raw_season(),
            pd.DataFrame([_raw_player(3, 302, 20, "gol", 2.0)]),
        ],
        ignore_index=True,
    )


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


def _support_gate_mechanism_audit(
    *,
    low_residual: float = 0.30,
    normal_residual: float = 0.05,
    low_rows_by_season: dict[int, int] | None = None,
    total_rows_by_position: dict[str, int] | None = None,
) -> pd.DataFrame:
    seasons = (2021, 2022, 2023)
    positions = ("gol", "lat", "zag", "mei")
    low_bins = ("0", "(0, 0.5]", "(0.5, 0.8]")
    normal_bins = ("(0.8, 1.0]", "(1.0, 1.5]")
    rows: list[dict[str, object]] = []
    for season in seasons:
        low_rows_per_position = (low_rows_by_season or {}).get(season, 600) // len(positions)
        for position in positions:
            if total_rows_by_position and position in total_rows_by_position:
                low_rows_per_position = 100
                normal_rows_per_bin = max((total_rows_by_position[position] - 300) // 6, 0)
            else:
                normal_rows_per_bin = 75
            rows.append(_support_gate_row(season, position, "0", low_rows_per_position, 20, low_residual))
            for ratio_bin in low_bins[1:]:
                rows.append(_support_gate_row(season, position, ratio_bin, 0, 0, low_residual))
            for ratio_bin in normal_bins:
                rows.append(_support_gate_row(season, position, ratio_bin, normal_rows_per_bin, 20, normal_residual))
    return pd.DataFrame(rows)


def _support_gate_raw_count_audit(
    *,
    low_residual: float = 0.20,
    normal_residual: float = 0.05,
    normal_positions: tuple[str, ...] = ("gol", "lat", "zag", "mei"),
    normal_row_count: int = 150,
) -> pd.DataFrame:
    seasons = (2021, 2022, 2023)
    positions = ("gol", "lat", "zag", "mei")
    rows: list[dict[str, object]] = []
    for season in seasons:
        for position in positions:
            rows.append(_support_gate_row(season, position, "0", 150, 20, low_residual, bin_column="raw_count_bin"))
        for position in normal_positions:
            rows.append(
                _support_gate_row(
                    season,
                    position,
                    "(10, 20]",
                    normal_row_count,
                    20,
                    normal_residual,
                    bin_column="raw_count_bin",
                )
            )
    return pd.DataFrame(rows)


def _support_gate_row(
    season: int,
    position: str,
    bin_value: str,
    row_count: int,
    round_count: int,
    residual_mean: float,
    *,
    bin_column: str = "ratio_bin",
) -> dict[str, object]:
    return {
        "season": season,
        "posicao": position,
        bin_column: bin_value,
        "row_count": row_count,
        "round_count": round_count,
        "source_residual_mean": residual_mean,
        "source_overprediction_rate": 0.5,
        "source_base_count_mean": 1.0,
        "h005_available_match_count_mean": 1.0,
        "h005_expected_count_mean": 1.0,
        "h005_count_ratio_mean": 1.0,
        "source_position_points_mean": 1.0,
        "source_all_points_mean": 1.0,
        "position_allowed_delta_mean": 0.0,
    }
