from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.recommendation import (
    RecommendationConfig,
    _finalized_live_data_evidence,
    _validate_mode_scope,
    _visible_season_frame,
)


def test_recommendation_config_output_path() -> None:
    config = RecommendationConfig(
        season=2025,
        target_round=14,
        mode="live",
        project_root=Path("/tmp/cartola"),
    )

    assert config.output_path == Path("/tmp/cartola/data/08_reporting/recommendations/2025/round-14/live")


def test_recommendation_config_selected_formation() -> None:
    config = RecommendationConfig(
        season=2025,
        target_round=14,
        mode="replay",
        formation_name="3-4-3",
        formations={"3-4-3": {"gol": 1, "zag": 3, "mei": 4, "ata": 3, "tec": 1}},
    )

    assert config.selected_formation == {"gol": 1, "zag": 3, "mei": 4, "ata": 3, "tec": 1}


def _round_frame(
    round_number: int,
    *,
    finalized: bool = True,
    zero_filled_scouts: bool = False,
    points_offset: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    player_id = 1
    for posicao, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row: dict[str, object] = {
                "id_atleta": player_id,
                "apelido": f"{posicao}-{offset}",
                "slug": f"{posicao}-{offset}",
                "id_clube": player_id,
                "nome_clube": f"Club {player_id}",
                "posicao": posicao,
                "status": "Provavel",
                "rodada": round_number,
                "preco": 5.0,
                "preco_pre_rodada": 5.0,
                "pontuacao": float(round_number + offset + points_offset) if finalized else 0.0,
                "media": float(round_number + offset),
                "num_jogos": round_number - 1,
                "variacao": 0.0,
                "entrou_em_campo": finalized,
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 0 if zero_filled_scouts else (1 if finalized and scout == "DS" else 0)
            rows.append(row)
            player_id += 1
    return pd.DataFrame(rows)


def _season_frame(rounds: range, *, target_round: int | None = None, live_target: bool = False) -> pd.DataFrame:
    frames = []
    for round_number in rounds:
        frames.append(
            _round_frame(
                round_number,
                finalized=not (live_target and target_round == round_number),
                zero_filled_scouts=live_target and target_round == round_number,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _footystats_rows(rounds: range, clubs: range = range(1, 19)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_number in rounds:
        for club_id in clubs:
            opponent_id = club_id + 1 if club_id % 2 == 1 else club_id - 1
            team_ppg = float(round_number) + club_id / 100.0
            opponent_ppg = float(round_number) + opponent_id / 100.0
            rows.append(
                {
                    "rodada": round_number,
                    "id_clube": club_id,
                    "opponent_id_clube": opponent_id,
                    "is_home_footystats": int(club_id % 2 == 1),
                    "footystats_team_pre_match_ppg": team_ppg,
                    "footystats_opponent_pre_match_ppg": opponent_ppg,
                    "footystats_ppg_diff": team_ppg - opponent_ppg,
                }
            )
    return pd.DataFrame(rows)


def test_visible_season_frame_excludes_future_rounds() -> None:
    season_df = _season_frame(range(1, 6), target_round=3, live_target=True)

    visible = _visible_season_frame(season_df, target_round=3)

    assert sorted(visible["rodada"].unique().tolist()) == [1, 2, 3]
    assert 4 not in visible["rodada"].unique()
    assert 5 not in visible["rodada"].unique()


def test_live_mode_requires_current_year() -> None:
    config = RecommendationConfig(season=2025, target_round=10, mode="live", current_year=2026)

    with pytest.raises(ValueError, match="live mode requires season 2025 to equal current_year 2026"):
        _validate_mode_scope(config)


def test_replay_mode_allows_historical_season() -> None:
    config = RecommendationConfig(season=2025, target_round=10, mode="replay", current_year=2026)

    _validate_mode_scope(config)


def test_finalized_evidence_ignores_zero_filled_live_rows() -> None:
    target = _round_frame(14, finalized=False, zero_filled_scouts=True)

    evidence = _finalized_live_data_evidence(target)

    assert evidence == {
        "pontuacao_non_zero_count": 0,
        "entrou_em_campo_true_count": 0,
        "non_zero_scout_count": 0,
    }


def test_finalized_evidence_detects_played_rows_and_non_zero_scouts() -> None:
    target = _round_frame(14, finalized=True)

    evidence = _finalized_live_data_evidence(target)

    assert evidence["pontuacao_non_zero_count"] > 0
    assert evidence["entrou_em_campo_true_count"] > 0
    assert evidence["non_zero_scout_count"] > 0


def test_finalized_evidence_parses_false_entry_strings() -> None:
    target = pd.DataFrame(
        {
            "pontuacao": [0.0, 0.0, 0.0, 0.0],
            "entrou_em_campo": ["False", "0", "", None],
        }
    )

    evidence = _finalized_live_data_evidence(target)

    assert evidence["entrou_em_campo_true_count"] == 0


def test_finalized_evidence_parses_true_entry_strings() -> None:
    target = pd.DataFrame(
        {
            "pontuacao": [0.0, 0.0],
            "entrou_em_campo": ["True", "1"],
        }
    )

    evidence = _finalized_live_data_evidence(target)

    assert evidence["entrou_em_campo_true_count"] == 2


def test_finalized_evidence_respects_custom_scout_columns() -> None:
    target = pd.DataFrame(
        {
            "pontuacao": [0.0, 0.0],
            "entrou_em_campo": [False, False],
            "CUSTOM_SCOUT": [1, 2],
        }
    )

    default_evidence = _finalized_live_data_evidence(target)
    custom_evidence = _finalized_live_data_evidence(target, scout_columns=("CUSTOM_SCOUT",))

    assert default_evidence["non_zero_scout_count"] == 0
    assert custom_evidence["non_zero_scout_count"] == 2
