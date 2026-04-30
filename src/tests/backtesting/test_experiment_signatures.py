import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.experiment_signatures import (
    ComparabilityError,
    candidate_pool_signature,
    compare_signature_sets,
    raw_cartola_source_identity,
    solver_status_signature,
)


def test_candidate_pool_signature_uses_canonical_candidate_fields() -> None:
    frame = pd.DataFrame(
        {
            "id_atleta": [2, 1],
            "posicao": ["mei", "ata"],
            "id_clube": [20, 10],
            "status": ["Provavel", "Provavel"],
            "preco_pre_rodada": [8.12345678911, 10.0],
            "rodada": [5, 5],
            "random_forest_score": [9.0, 1.0],
        }
    )

    first = candidate_pool_signature(frame)
    second = candidate_pool_signature(frame.sort_values("id_atleta", ascending=False))

    assert first == second


def test_candidate_pool_signature_changes_when_price_changes() -> None:
    frame = pd.DataFrame(
        {
            "id_atleta": [1],
            "posicao": ["ata"],
            "id_clube": [10],
            "status": ["Provavel"],
            "preco_pre_rodada": [10.0],
            "rodada": [5],
        }
    )
    changed = frame.copy()
    changed["preco_pre_rodada"] = [10.1]

    assert candidate_pool_signature(frame) != candidate_pool_signature(changed)


def test_candidate_pool_signature_missing_required_column_raises() -> None:
    frame = pd.DataFrame(
        {
            "id_atleta": [1],
            "posicao": ["ata"],
            "id_clube": [10],
            "status": ["Provavel"],
            "preco_pre_rodada": [10.0],
        }
    )

    with pytest.raises(ComparabilityError, match="Missing required candidate columns"):
        candidate_pool_signature(frame)


def test_solver_status_signature_maps_primary_role() -> None:
    rows = pd.DataFrame(
        {
            "rodada": [5, 5, 5],
            "strategy": ["baseline", "extra_trees", "price"],
            "solver_status": ["Optimal", "Optimal", "Infeasible"],
        }
    )

    assert solver_status_signature(rows, primary_model_id="extra_trees") == {
        "5:baseline": "Optimal",
        "5:primary_model": "Optimal",
        "5:price": "Infeasible",
    }


def test_solver_status_signature_rejects_unexpected_strategy() -> None:
    rows = pd.DataFrame(
        {
            "rodada": [5],
            "strategy": ["random_forest"],
            "solver_status": ["Optimal"],
        }
    )

    with pytest.raises(ComparabilityError, match="Unexpected strategy"):
        solver_status_signature(rows, primary_model_id="extra_trees")


def test_compare_signature_sets_raises_on_mismatch() -> None:
    with pytest.raises(ComparabilityError, match="candidate pools differ"):
        compare_signature_sets(
            label="candidate pools",
            signatures={
                "run-a": {"2025:5": "abc"},
                "run-b": {"2025:5": "def"},
            },
        )


def test_compare_signature_sets_passes_for_equal_signatures() -> None:
    compare_signature_sets(
        label="candidate pools",
        signatures={
            "run-a": {"2025:5": "abc"},
            "run-b": {"2025:5": "abc"},
        },
    )


def test_raw_cartola_source_identity_hashes_sorted_files(tmp_path: Path) -> None:
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    (season_dir / "rodada-2.csv").write_text("b\n", encoding="utf-8")
    (season_dir / "rodada-1.csv").write_text("a\n", encoding="utf-8")
    (season_dir / "rodada-2.capture.json").write_text("ignored\n", encoding="utf-8")

    identity = raw_cartola_source_identity(project_root=tmp_path, season=2025)

    assert identity["season"] == 2025
    assert len(identity["files"]) == 2
    assert [file_record["path"] for file_record in identity["files"]] == [
        "data/01_raw/2025/rodada-1.csv",
        "data/01_raw/2025/rodada-2.csv",
    ]
    json.dumps(identity, sort_keys=True)


def test_raw_cartola_source_identity_changes_when_file_content_changes(tmp_path: Path) -> None:
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    source_file = season_dir / "rodada-1.csv"
    source_file.write_text("a\n", encoding="utf-8")

    first = raw_cartola_source_identity(project_root=tmp_path, season=2025)
    source_file.write_text("changed\n", encoding="utf-8")
    second = raw_cartola_source_identity(project_root=tmp_path, season=2025)

    assert first["sha256"] != second["sha256"]
