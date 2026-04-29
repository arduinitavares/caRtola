from __future__ import annotations

from pathlib import Path

from cartola.backtesting.market_capture import build_live_market_frame

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.data import load_round_file


def _status_payload(*, rodada_atual: int = 14, status_mercado: int = 1) -> dict[str, object]:
    return {
        "rodada_atual": rodada_atual,
        "status_mercado": status_mercado,
        "fechamento": {"timestamp": 1777748340},
    }


def _market_payload(*, stale_round: int = 13) -> dict[str, object]:
    return {
        "clubes": {
            "264": {"id": 264, "nome": "Corinthians"},
            "276": {"id": 276, "nome": "Sao Paulo"},
        },
        "atletas": [
            {
                "atleta_id": 100652,
                "apelido": "Yuri Alberto",
                "slug": "yuri-alberto",
                "nome": "Yuri Alberto",
                "foto": "https://example.invalid/yuri.png",
                "apelido_abreviado": "Yuri",
                "clube_id": 264,
                "posicao_id": 5,
                "status_id": 7,
                "preco_num": 10.45,
                "pontos_num": 9.9,
                "media_num": 2.49,
                "jogos_num": 9,
                "variacao_num": -0.39,
                "rodada_id": stale_round,
                "entrou_em_campo": True,
                "minimo_para_valorizar": 6.12,
                "scout": {"G": 2, "A": 1, "DS": 4},
            },
            {
                "atleta_id": 80287,
                "apelido": "Luciano",
                "slug": "luciano",
                "nome": "Luciano",
                "foto": "https://example.invalid/luciano.png",
                "apelido_abreviado": "Luc",
                "clube_id": 276,
                "posicao_id": 5,
                "status_id": 7,
                "preco_num": 15.0,
                "pontos_num": 11.8,
                "media_num": 5.35,
                "jogos_num": 11,
                "variacao_num": 0.67,
                "rodada_id": stale_round,
                "entrou_em_campo": True,
                "minimo_para_valorizar": 7.2,
                "scout": {"G": 1, "FS": 2},
            },
        ],
    }


def test_build_live_market_frame_replaces_stale_round_and_sanitizes_outcomes() -> None:
    frame = build_live_market_frame(_market_payload(stale_round=13), target_round=14)

    assert frame["atletas.rodada_id"].tolist() == [14, 14]
    assert frame["atletas.pontos_num"].tolist() == [0.0, 0.0]
    assert frame["atletas.variacao_num"].tolist() == [0.0, 0.0]
    assert frame["atletas.entrou_em_campo"].tolist() == [False, False]
    assert frame["atletas.clube.id.full.name"].tolist() == ["Corinthians", "Sao Paulo"]
    assert frame["atletas.preco_num"].tolist() == [10.45, 15.0]
    for scout in DEFAULT_SCOUT_COLUMNS:
        assert scout in frame.columns
        assert frame[scout].tolist() == [0, 0]


def test_build_live_market_frame_loads_through_round_loader(tmp_path: Path) -> None:
    frame = build_live_market_frame(_market_payload(stale_round=13), target_round=14)
    csv_path = tmp_path / "rodada-14.csv"
    frame.to_csv(csv_path, index=False)

    loaded = load_round_file(csv_path)

    assert loaded["rodada"].tolist() == [14, 14]
    assert loaded["status"].tolist() == ["Provavel", "Provavel"]
    assert loaded["posicao"].tolist() == ["ata", "ata"]
    assert loaded["preco_pre_rodada"].tolist() == [10.45, 15.0]
    assert loaded["pontuacao"].tolist() == [0.0, 0.0]
    assert loaded["entrou_em_campo"].tolist() == [False, False]
