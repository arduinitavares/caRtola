from __future__ import annotations

import pandas as pd
import pytest

from cartola.backtesting.oracle_profiles import (
    build_oracle_player_profile_rows,
    build_profile_gap_summary_rows,
)

IDENTITY = {
    "source_mode": "artifact",
    "source_experiment_id": "exp-1",
    "source_child_id": "child-1",
    "season": 2025,
    "rodada": 5,
    "strategy": "model_a",
    "model_id": "model_a",
    "feature_pack": "ppg_xg_matchup",
    "fixture_mode": "exploratory",
    "matchup_context_mode": "cartola_matchup_v1",
    "budget_policy": "moving",
    "oracle_type": "budget_constrained",
    "candidate_universe": "model_candidate",
    "budget_path": "model_budget_path",
}


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-05-01"},
            {"rodada": 5, "id_clube_home": 30, "id_clube_away": 40, "data": "2025-05-01"},
        ],
    )


def _selected(source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_atleta": 1,
                "rodada": 5,
                "id_clube": 10,
                "posicao": "ata",
                "preco_pre_rodada": 20.0,
                "model_predicted_rank_overall": 3,
                "model_predicted_rank_position": 1,
                "matchup_is_home": True,
                "footystats_ppg_diff": 0.5,
                "source": source,
            },
            {
                "id_atleta": 2,
                "rodada": 5,
                "id_clube": 20,
                "posicao": "zag",
                "preco_pre_rodada": 5.0,
                "model_predicted_rank_overall": 44,
                "model_predicted_rank_position": 7,
                "matchup_is_home": False,
                "footystats_ppg_diff": -0.5,
                "source": source,
            },
            {
                "id_atleta": 3,
                "rodada": 5,
                "id_clube": 30,
                "posicao": "tec",
                "preco_pre_rodada": 8.0,
                "model_predicted_rank_overall": 10,
                "model_predicted_rank_position": 2,
                "matchup_is_home": True,
                "footystats_ppg_diff": 0.2,
                "source": source,
            },
        ],
    )


def test_build_oracle_player_profile_rows_emits_home_rank_price_and_opponent_overlap() -> None:
    rows = build_oracle_player_profile_rows(
        identity=IDENTITY,
        oracle_selected=_selected("oracle"),
        model_selected=_selected("model"),
        fixtures=_fixtures(),
    )

    metric_values = {(row["id_atleta"], row["profile_metric"]): row["profile_value"] for row in rows}
    assert rows[0]["source_experiment_id"] == "exp-1"
    assert metric_values[(1, "is_home")] is True
    assert metric_values[(2, "is_home")] is False
    assert metric_values[(1, "opponent_overlap_in_lineup")] is True
    assert metric_values[(2, "opponent_overlap_in_lineup")] is True
    assert metric_values[(1, "same_club_selected_count")] == 1.0
    assert metric_values[(1, "model_predicted_rank_overall")] == 3.0
    assert metric_values[(1, "model_predicted_rank_position")] == 1.0
    assert metric_values[(2, "model_predicted_rank_position")] == 7.0
    assert metric_values[(1, "preco_pre_rodada")] == 20.0
    assert metric_values[(1, "favorite_proxy_ppg_diff_positive")] is True
    assert metric_values[(2, "favorite_proxy_ppg_diff_positive")] is False
    assert metric_values[(2, "footystats_ppg_diff")] == -0.5


def test_build_oracle_player_profile_rows_rejects_ambiguous_boolean_strings() -> None:
    selected = pd.DataFrame(
        [
            {"id_atleta": 1, "rodada": 5, "id_clube": 10, "posicao": "ata", "matchup_is_home": "true"},
            {"id_atleta": 2, "rodada": 5, "id_clube": 20, "posicao": "zag", "matchup_is_home": "false"},
            {"id_atleta": 3, "rodada": 5, "id_clube": 30, "posicao": "mei", "matchup_is_home": "0"},
            {"id_atleta": 4, "rodada": 5, "id_clube": 40, "posicao": "lat", "matchup_is_home": "unknown"},
            {"id_atleta": 5, "rodada": 5, "id_clube": 50, "posicao": "gol", "matchup_is_home": " "},
            {"id_atleta": 6, "rodada": 5, "id_clube": 60, "posicao": "ata", "matchup_is_home": "0.0"},
            {"id_atleta": 7, "rodada": 5, "id_clube": 70, "posicao": "mei", "matchup_is_home": 1},
            {"id_atleta": 8, "rodada": 5, "id_clube": 80, "posicao": "zag", "matchup_is_home": 0},
            {"id_atleta": 9, "rodada": 5, "id_clube": 90, "posicao": "lat", "matchup_is_home": 2},
        ],
    )

    rows = build_oracle_player_profile_rows(
        identity=IDENTITY,
        oracle_selected=selected,
        model_selected=selected,
        fixtures=_fixtures(),
    )

    is_home = {
        row["id_atleta"]: row["profile_value"] for row in rows if row["profile_metric"] == "is_home"
    }
    assert is_home[1] is True
    assert is_home[2] is False
    assert is_home[3] is False
    assert is_home[4] is None
    assert is_home[5] is None
    assert is_home[6] is None
    assert is_home[7] is True
    assert is_home[8] is False
    assert is_home[9] is None


def test_build_profile_gap_summary_rows_compares_oracle_to_model_selected_baseline() -> None:
    oracle = _selected("oracle")
    model = _selected("model").loc[lambda frame: frame["id_clube"].isin([10, 30])].copy()

    rows = build_profile_gap_summary_rows(
        identity=IDENTITY,
        oracle_selected=oracle,
        model_selected=model,
        fixtures=_fixtures(),
    )

    metrics = {row["profile_metric"]: row for row in rows}
    assert metrics["opponent_overlap_round_rate"]["oracle_value"] == 1.0
    assert metrics["opponent_overlap_round_rate"]["baseline_value"] == 0.0
    assert metrics["avg_players_in_opponent_overlap"]["oracle_value"] == 2.0
    assert metrics["home_player_share"]["oracle_value"] == pytest.approx(0.5)
    assert metrics["home_player_share"]["baseline_value"] == pytest.approx(1.0)
    assert metrics["favorite_proxy_ppg_diff_positive_share"]["oracle_value"] == pytest.approx(0.5)
    assert metrics["favorite_proxy_ppg_diff_positive_share"]["baseline_value"] == pytest.approx(1.0)
    assert metrics["median_model_predicted_rank_position"]["oracle_value"] == pytest.approx(4.0)
    assert metrics["top5_position_rank_share"]["oracle_value"] == pytest.approx(0.5)
    assert metrics["avg_same_club_selected_count"]["oracle_value"] == pytest.approx(1.0)
