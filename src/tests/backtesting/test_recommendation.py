from __future__ import annotations

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS


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
