from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.recommendation import (
    RecommendationConfig,
    _finalized_live_data_evidence,
    _validate_mode_scope,
    _visible_season_frame,
    run_recommendation,
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


def test_run_recommendation_ignores_future_cartola_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 6), target_round=3, live_target=True)
    load_calls: list[dict[str, object]] = []

    def fake_load_footystats(**kwargs: object):
        load_calls.append(kwargs)
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    result = run_recommendation(config)

    assert result.metadata["visible_max_round"] == 3
    assert result.metadata["training_rounds"] == [1, 2]
    required_keys = load_calls[0]["required_keys"]
    assert int(required_keys["rodada"].max()) == 3
    assert result.candidate_predictions["rodada"].eq(3).all()


def test_run_recommendation_replay_reports_actual_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="replay",
        project_root=tmp_path,
        current_year=2026,
    )

    result = run_recommendation(config)

    assert result.summary["actual_points"] is not None
    assert "pontuacao" in result.recommended_squad.columns
    assert result.summary["optimizer_status"] == "Optimal"


def test_run_recommendation_live_suppresses_actual_columns_when_finalized_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        allow_finalized_live_data=True,
    )

    result = run_recommendation(config)

    assert result.summary["actual_points"] is None
    assert "pontuacao" not in result.recommended_squad.columns
    assert "entrou_em_campo" not in result.candidate_predictions.columns
    assert result.metadata["finalized_live_data_detected"] is True


def test_live_mode_rejects_finalized_target_round_without_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    with pytest.raises(ValueError, match="appears finalized"):
        run_recommendation(config)


def test_run_recommendation_writes_expected_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)

    def fake_load_footystats(**kwargs: object):
        from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult

        return FootyStatsPPGLoadResult(
            rows=_footystats_rows(range(1, 4)),
            source_path=tmp_path / "data/footystats/source.csv",
            source_sha256="sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr(
        "cartola.backtesting.recommendation.load_footystats_feature_rows_for_recommendation",
        fake_load_footystats,
    )
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
    )

    run_recommendation(config)

    output_path = tmp_path / "data/08_reporting/recommendations/2026/round-3/live"
    assert (output_path / "recommended_squad.csv").exists()
    assert (output_path / "candidate_predictions.csv").exists()
    assert (output_path / "recommendation_summary.json").exists()
    assert (output_path / "run_metadata.json").exists()
    summary = json.loads((output_path / "recommendation_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert summary["actual_points"] is None
    assert metadata["training_rounds"] == [1, 2]
    assert metadata["footystats_matches_source_sha256"] == "sha"
