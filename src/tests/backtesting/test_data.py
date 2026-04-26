from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.data import load_round_file, load_season_data, normalize_round_frame


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
    assert normalized.loc[0, "V"] == 0
    assert normalized.loc[0, "G"] == 0
    assert bool(normalized.loc[0, "entrou_em_campo"]) is True


def test_normalize_round_frame_accepts_already_normalized_status_and_position_values():
    raw = _base_raw_round(**{"atletas.status_id": ["Provavel"], "atletas.posicao_id": ["ata"]})

    normalized = normalize_round_frame(raw, source=Path("rodada-1.csv"))

    assert normalized.loc[0, "status"] == "Provavel"
    assert normalized.loc[0, "posicao"] == "ata"


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
