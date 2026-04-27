from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.data import (
    build_round_alignment_report,
    load_fixtures,
    load_round_file,
    load_season_data,
    normalize_round_frame,
)


def _base_raw_round(**overrides):
    data = {
        "atletas.rodada_id": [1],
        "atletas.status_id": [7],
        "atletas.posicao_id": [5],
        "atletas.atleta_id": [10],
        "atletas.apelido": ["Player"],
        "atletas.clube_id": [100],
        "atletas.preco_num": [12.3],
        "atletas.pontos_num": [4.5],
        "atletas.media_num": [4.5],
        "atletas.jogos_num": [1],
        "atletas.variacao_num": [0.2],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_normalize_round_frame_drops_index_maps_columns_and_fills_scouts():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [0],
            "atletas.rodada_id": [1],
            "atletas.status_id": [7],
            "atletas.posicao_id": [5],
            "atletas.atleta_id": [10],
            "atletas.apelido": ["Player"],
            "atletas.slug": ["player"],
            "atletas.clube_id": [100],
            "atletas.clube.id.full.name": ["Club"],
            "atletas.preco_num": [12.3],
            "atletas.pontos_num": [4.5],
            "atletas.media_num": [4.5],
            "atletas.jogos_num": [1],
            "atletas.variacao_num": [0.2],
            "atletas.entrou_em_campo": [True],
        }
    )

    normalized = normalize_round_frame(raw, source=Path("rodada-1.csv"))

    assert "Unnamed: 0" not in normalized.columns
    assert normalized.loc[0, "status"] == "Provavel"
    assert normalized.loc[0, "posicao"] == "ata"
    assert normalized.loc[0, "id_atleta"] == 10
    assert normalized.loc[0, "pontuacao"] == 4.5
    assert normalized.loc[0, "num_jogos"] == 1
    assert normalized.loc[0, "preco_pre_rodada"] == 12.1
    assert "pontos" not in normalized.columns
    assert "jogos" not in normalized.columns
    assert normalized.loc[0, "V"] == 0
    assert normalized.loc[0, "G"] == 0
    assert bool(normalized.loc[0, "entrou_em_campo"]) is True


def test_normalize_round_frame_accepts_already_normalized_status_and_position_values():
    raw = _base_raw_round(**{"atletas.status_id": ["Provavel"], "atletas.posicao_id": ["ata"]})

    normalized = normalize_round_frame(raw, source=Path("rodada-1.csv"))

    assert normalized.loc[0, "status"] == "Provavel"
    assert normalized.loc[0, "posicao"] == "ata"


def test_normalize_round_frame_fills_blank_scout_values_with_zero():
    raw = _base_raw_round(G=[None], V=[""])

    normalized = normalize_round_frame(raw, source=Path("rodada-1.csv"))

    assert normalized.loc[0, "G"] == 0
    assert normalized.loc[0, "V"] == 0


def test_normalize_round_frame_reconstructs_market_open_price_from_price_variation():
    raw = _base_raw_round(**{"atletas.preco_num": [15.75], "atletas.variacao_num": [1.25]})

    normalized = normalize_round_frame(raw, source=Path("rodada-5.csv"))

    assert normalized.loc[0, "preco"] == 15.75
    assert normalized.loc[0, "preco_pre_rodada"] == 14.5


def test_normalize_round_frame_rejects_unknown_status():
    raw = pd.DataFrame(
        {
            "atletas.rodada_id": [1],
            "atletas.status_id": [99],
            "atletas.posicao_id": [5],
            "atletas.atleta_id": [10],
            "atletas.apelido": ["Player"],
            "atletas.clube_id": [100],
            "atletas.preco_num": [12.3],
            "atletas.pontos_num": [4.5],
            "atletas.media_num": [4.5],
            "atletas.jogos_num": [1],
            "atletas.variacao_num": [0.2],
        }
    )

    with pytest.raises(ValueError, match="Unknown status_id values"):
        normalize_round_frame(raw, source=Path("rodada-1.csv"))


def test_normalize_round_frame_reports_missing_required_columns():
    raw = _base_raw_round().drop(columns=["atletas.atleta_id"])

    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_round_frame(raw, source=Path("rodada-1.csv"))


def test_load_round_file_reads_and_normalizes_csv(tmp_path):
    round_file = tmp_path / "rodada-1.csv"
    _base_raw_round(**{"": ["saved-index"]}).to_csv(round_file, index=False)

    loaded = load_round_file(round_file)

    assert loaded.loc[0, "rodada"] == 1
    assert loaded.loc[0, "status"] == "Provavel"
    assert loaded.loc[0, "pontuacao"] == 4.5
    assert loaded.loc[0, "num_jogos"] == 1
    assert "pontos" not in loaded.columns
    assert "jogos" not in loaded.columns
    assert "" not in loaded.columns


def test_load_season_data_orders_rounds_numerically(tmp_path):
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    base = {
        "atletas.status_id": [7],
        "atletas.posicao_id": [5],
        "atletas.atleta_id": [10],
        "atletas.apelido": ["Player"],
        "atletas.clube_id": [100],
        "atletas.preco_num": [12.3],
        "atletas.pontos_num": [4.5],
        "atletas.media_num": [4.5],
        "atletas.jogos_num": [1],
        "atletas.variacao_num": [0.2],
    }
    pd.DataFrame({**base, "atletas.rodada_id": [10]}).to_csv(season_dir / "rodada-10.csv", index=False)
    pd.DataFrame({**base, "atletas.rodada_id": [2]}).to_csv(season_dir / "rodada-2.csv", index=False)

    loaded = load_season_data(2025, project_root=tmp_path)

    assert loaded["rodada"].tolist() == [2, 10]


def test_load_season_data_reports_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Season directory not found"):
        load_season_data(2025, project_root=tmp_path)


def test_load_season_data_reports_empty_directory(tmp_path):
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No round CSV files found"):
        load_season_data(2025, project_root=tmp_path)


def _write_fixture_round(root: Path, round_number: int, rows: list[dict[str, object]]) -> None:
    fixture_dir = root / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["rodada", "id_clube_home", "id_clube_away", "data"]).to_csv(
        fixture_dir / f"partidas-{round_number}.csv",
        index=False,
    )


def test_load_fixtures_reads_and_normalizes_round_files(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40, "data": "2025-04-06"},
        ],
    )

    loaded = load_fixtures(2025, project_root=tmp_path)

    assert loaded["rodada"].tolist() == [2, 2]
    assert loaded["id_clube_home"].tolist() == [10, 30]
    assert loaded["id_clube_away"].tolist() == [20, 40]
    assert str(loaded.loc[0, "data"]) == "2025-04-05"


def test_load_fixtures_drops_extra_columns(tmp_path):
    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "rodada": [2],
            "id_clube_home": [10],
            "id_clube_away": [20],
            "data": ["2025-04-05"],
            "source_id": ["official-api"],
        }
    ).to_csv(fixture_dir / "partidas-2.csv", index=False)

    loaded = load_fixtures(2025, project_root=tmp_path)

    assert loaded.columns.tolist() == ["rodada", "id_clube_home", "id_clube_away", "data"]


def test_load_fixtures_rejects_missing_required_columns(tmp_path):
    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    pd.DataFrame({"rodada": [2], "id_clube_home": [10], "data": ["2025-04-05"]}).to_csv(
        fixture_dir / "partidas-2.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="Missing required fixture columns"):
        load_fixtures(2025, project_root=tmp_path)


def test_load_fixtures_rejects_duplicate_club_appearance_in_round(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-06"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate fixture club entries"):
        load_fixtures(2025, project_root=tmp_path)


def test_load_fixtures_rejects_self_matches(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [{"rodada": 2, "id_clube_home": 10, "id_clube_away": 10, "data": "2025-04-05"}],
    )

    with pytest.raises(ValueError, match="Fixture rows cannot have the same home and away club"):
        load_fixtures(2025, project_root=tmp_path)


def test_build_round_alignment_report_compares_fixture_and_played_clubs():
    fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-12"},
        ]
    )
    season_df = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 10, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 20, "entrou_em_campo": True},
            {"rodada": 3, "id_clube": 10, "entrou_em_campo": True},
            {"rodada": 3, "id_clube": 40, "entrou_em_campo": True},
        ]
    )
    official_fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-12"},
            {"rodada": 3, "id_clube_home": 50, "id_clube_away": 60, "data": "2025-04-12"},
        ]
    )

    report = build_round_alignment_report(fixtures, season_df, official_fixtures=official_fixtures)
    round_2 = report.loc[report["rodada"] == 2].iloc[0]
    round_3 = report.loc[report["rodada"] == 3].iloc[0]

    assert bool(round_2["is_valid"]) is True
    assert bool(round_3["is_valid"]) is False
    assert round_3["missing_from_fixtures"] == "40"
    assert round_3["extra_in_fixtures"] == "30"
    assert round_3["discarded_official_match_count"] == 1
    assert round_3["discarded_official_clubs"] == "50,60"


def test_build_round_alignment_report_rejects_extra_canonical_fixture_clubs_without_missing_played_clubs():
    fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40, "data": "2025-04-05"},
        ]
    )
    season_df = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 10, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 20, "entrou_em_campo": True},
        ]
    )

    report = build_round_alignment_report(fixtures, season_df)
    round_2 = report.loc[report["rodada"] == 2].iloc[0]

    assert bool(round_2["is_valid"]) is False
    assert round_2["missing_from_fixtures"] == ""
    assert round_2["extra_in_fixtures"] == "30,40"
