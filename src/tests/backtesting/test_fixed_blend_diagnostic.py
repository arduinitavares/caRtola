from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.fixed_blend_diagnostic import (
    FixedBlendDiagnosticError,
    load_blend_candidate_frame,
    parse_blend_specs,
    run_blend_replay_for_season,
)


def test_parse_blend_specs_accepts_named_weighted_components() -> None:
    specs = parse_blend_specs(("blend_a=model_a:0.25,model_b:0.75",))

    assert len(specs) == 1
    assert specs[0].name == "blend_a"
    assert [(component.model_id, component.weight) for component in specs[0].components] == [
        ("model_a", 0.25),
        ("model_b", 0.75),
    ]


@pytest.mark.parametrize(
    "raw_specs",
    [
        (),
        ("blend=model_a:1.0",),
        ("blend=model_a:0.5,model_a:0.5",),
        ("blend=model_a:0.5,model_b:0.4",),
        ("blend=model_a:nan,model_b:1.0",),
        ("blend=model_a:-0.1,model_b:1.1",),
        ("blend=model_a:0.5,model_b:0.5", "blend=model_c:0.5,model_d:0.5"),
    ],
)
def test_parse_blend_specs_rejects_invalid_specs(raw_specs: tuple[str, ...]) -> None:
    with pytest.raises(FixedBlendDiagnosticError):
        parse_blend_specs(raw_specs)


def test_load_blend_candidate_frame_combines_component_scores(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5,), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    _write_predictions(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        frame=base.rename(columns={"model_a_score": "model_b_score"}).assign(model_b_score=base["model_a_score"] + 10.0),
    )
    blend = parse_blend_specs(("blend_a=model_a:0.25,model_b:0.75",))[0]

    candidates = load_blend_candidate_frame(
        experiment_path=experiment_path,
        season=2025,
        feature_pack="ppg",
        blend_spec=blend,
    )

    assert "m006_component_model_a_score" in candidates.columns
    assert "m006_component_model_b_score" in candidates.columns
    assert "m006_blend_blend_a_score" in candidates.columns
    assert candidates["m006_blend_blend_a_score"].iloc[0] == pytest.approx(
        (base["model_a_score"].iloc[0] * 0.25) + ((base["model_a_score"].iloc[0] + 10.0) * 0.75)
    )
    assert candidates[["rodada", "id_atleta", "id_clube", "posicao"]].equals(
        candidates[["rodada", "id_atleta", "id_clube", "posicao"]].sort_values(
            ["rodada", "id_atleta", "id_clube", "posicao"],
            kind="mergesort",
        )
    )


def test_load_blend_candidate_frame_rejects_candidate_identity_mismatch(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5,), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    mismatched = base.rename(columns={"model_a_score": "model_b_score"}).copy()
    mismatched.loc[mismatched.index[0], "id_atleta"] = 999
    _write_predictions(experiment_path, season=2025, model_id="model_b", feature_pack="ppg", frame=mismatched)
    blend = parse_blend_specs(("blend_a=model_a:0.5,model_b:0.5",))[0]

    with pytest.raises(FixedBlendDiagnosticError, match="candidate identity"):
        load_blend_candidate_frame(
            experiment_path=experiment_path,
            season=2025,
            feature_pack="ppg",
            blend_spec=blend,
        )


def test_run_blend_replay_for_season_advances_each_blend_budget_independently(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    base = _candidate_frame(rounds=(5, 6), score_column="model_a_score", score_offset=0.0)
    _write_predictions(experiment_path, season=2025, model_id="model_a", feature_pack="ppg", frame=base)
    _write_predictions(
        experiment_path,
        season=2025,
        model_id="model_b",
        feature_pack="ppg",
        frame=base.rename(columns={"model_a_score": "model_b_score"}).assign(model_b_score=base["model_a_score"] + 1.0),
    )
    blends = parse_blend_specs(
        (
            "blend_a=model_a:1.0,model_b:0.0",
            "blend_b=model_a:0.0,model_b:1.0",
        )
    )

    replay = run_blend_replay_for_season(
        experiment_path=experiment_path,
        season=2025,
        feature_pack="ppg",
        blend_specs=blends,
        config=BacktestConfig(season=2025, start_round=5, budget=50.0, project_root=tmp_path),
    )

    assert replay.invalid_rows.empty
    assert set(replay.round_rows["blend_name"]) == {"blend_a", "blend_b"}
    assert replay.round_rows["solver_status"].tolist() == ["Optimal", "Optimal", "Optimal", "Optimal"]
    first_rounds = replay.round_rows.loc[replay.round_rows["rodada"].eq(5)]
    second_rounds = replay.round_rows.loc[replay.round_rows["rodada"].eq(6)]
    assert first_rounds["budget_before_round"].tolist() == [50.0, 50.0]
    assert first_rounds["budget_after_round"].tolist() == [62.0, 62.0]
    assert second_rounds["budget_before_round"].tolist() == [62.0, 62.0]
    assert set(replay.selected_player_rows["blend_name"]) == {"blend_a", "blend_b"}
    assert set(replay.selected_player_rows["predicted_points"]) == set(
        replay.selected_player_rows["m006_blend_blend_a_score"].dropna()
    ) | set(replay.selected_player_rows["m006_blend_blend_b_score"].dropna())


def _write_predictions(
    experiment_path: Path,
    *,
    season: int,
    model_id: str,
    feature_pack: str,
    frame: pd.DataFrame,
) -> None:
    child_path = experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"
    child_path.mkdir(parents=True)
    frame.to_csv(child_path / "player_predictions.csv", index=False)


def _candidate_frame(*, rounds: tuple[int, ...], score_column: str, score_offset: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    player_id = 1
    position_scores = {
        "gol": [11.0],
        "lat": [10.0, 9.0],
        "zag": [8.0, 7.0, 6.0],
        "mei": [15.0, 14.0, 13.0, 12.0, 11.0],
        "ata": [12.0, 11.0],
        "tec": [5.0],
    }
    for rodada in rounds:
        for posicao, scores in position_scores.items():
            for score in scores:
                rows.append(
                    {
                        "rodada": rodada,
                        "id_atleta": player_id,
                        "id_clube": player_id,
                        "posicao": posicao,
                        "apelido": f"{posicao}-{player_id}",
                        "nome_clube": f"club-{player_id}",
                        "clube": f"club-{player_id}",
                        "preco_pre_rodada": 1.0,
                        "pontuacao": score,
                        "entrou_em_campo": True,
                        "variacao": 1.0,
                        score_column: score + score_offset,
                    }
                )
                player_id += 1
    return pd.DataFrame(rows)
