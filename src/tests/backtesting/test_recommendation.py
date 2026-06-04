from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.footystats_features import FootyStatsPPGLoadResult
from cartola.backtesting.optimizer import SquadOptimizationResult
from cartola.backtesting.recommendation import (
    RecommendationConfig,
    _backtest_config,
    _finalized_live_data_evidence,
    _validate_mode_scope,
    _validate_selected_budget,
    _visible_season_frame,
    run_recommendation,
)
from cartola.backtesting.scoring_contract import (
    SCORING_CONTRACT_VERSION,
    actual_scores_with_captain,
    contract_fields,
)


def test_recommendation_config_output_path() -> None:
    config = RecommendationConfig(
        season=2025,
        target_round=14,
        mode="live",
        project_root=Path("/tmp/cartola"),
    )

    assert config.output_path == Path("/tmp/cartola/data/08_reporting/recommendations/2025/round-14/live")


def test_recommendation_config_output_path_includes_output_run_id() -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        project_root=Path("/tmp/cartola"),
        output_run_id="run_started_at=20260429T123456000000Z",
    )

    assert config.output_path == Path(
        "/tmp/cartola/data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456000000Z"
    )


@pytest.mark.parametrize("output_run_id", ["", ".", "..", "../escape", "/tmp/escape", "foo/bar", "foo\\bar"])
def test_recommendation_config_rejects_output_run_id_with_path_separator(output_run_id: str) -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        output_run_id=output_run_id,
    )

    with pytest.raises(ValueError, match="output_run_id"):
        _validate_mode_scope(config)


def test_recommendation_config_has_no_public_fixed_formation_api() -> None:
    config_fields = {field.name for field in fields(RecommendationConfig)}

    assert "formation_name" not in config_fields
    assert "formations" not in config_fields
    assert not isinstance(RecommendationConfig.__dict__.get("selected_formation"), property)


def test_validate_selected_budget_rejects_infeasible_optimizer_status() -> None:
    config = RecommendationConfig(season=2026, target_round=14, mode="live", budget=92.5, current_year=2026)
    optimized = _optimization_result(status="Infeasible", budget_used=0.0)

    with pytest.raises(ValueError, match="No valid squad was found within the operator-provided budget 92.50"):
        _validate_selected_budget(config, optimized)


def test_validate_selected_budget_rejects_post_selection_budget_violation() -> None:
    config = RecommendationConfig(season=2026, target_round=14, mode="live", budget=92.5, current_year=2026)
    optimized = _optimization_result(status="Optimal", budget_used=92.5001)

    with pytest.raises(ValueError, match="Selected squad exceeds the operator-provided budget"):
        _validate_selected_budget(config, optimized)


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


def _optimization_result(*, status: str, budget_used: float) -> SquadOptimizationResult:
    return SquadOptimizationResult(
        selected=pd.DataFrame(),
        status=status,
        budget_used=budget_used,
        predicted_points=0.0,
        predicted_points_base=0.0,
        captain_bonus_predicted=0.0,
        predicted_points_with_captain=0.0,
        formation_name="4-3-3",
        selected_count=0,
        captain_id=None,
        captain_name=None,
        captain_position=None,
        captain_club=None,
        captain_predicted_points=None,
        captain_multiplier=1.5,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        formation_scores=[],
        captain_policy_diagnostics=[],
    )


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


def _strict_fixture_rows(rounds: range, clubs: range = range(1, 19)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    club_ids = list(clubs)
    for round_number in rounds:
        for index in range(0, len(club_ids), 2):
            rows.append(
                {
                    "rodada": round_number,
                    "id_clube_home": club_ids[index],
                    "id_clube_away": club_ids[index + 1],
                    "data": f"2026-05-{round_number:02d}T19:00:00Z",
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


def test_matchup_context_requires_strict_fixture_mode() -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        current_year=2026,
        matchup_context_mode="cartola_matchup_v1",
    )

    with pytest.raises(ValueError, match="matchup_context_mode='cartola_matchup_v1' requires fixture_mode='strict'"):
        _validate_mode_scope(config)


def test_backtest_config_preserves_strict_matchup_modes(tmp_path: Path) -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        fixture_mode="strict",
        matchup_context_mode="cartola_matchup_v1",
    )

    backtest_config = _backtest_config(config)

    assert backtest_config.fixture_mode == "strict"
    assert backtest_config.matchup_context_mode == "cartola_matchup_v1"


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

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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


def test_run_recommendation_rejects_backtest_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        output_root=Path("data/08_reporting/backtests"),
        current_year=2026,
        footystats_mode="none",
    )

    with pytest.raises(ValueError, match="Recommendation output_root cannot be inside backtest reports"):
        run_recommendation(config)

    assert not (tmp_path / "data/08_reporting/backtests/2026/round-3/live").exists()


def test_run_recommendation_replay_reports_actual_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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
    actual_scores = actual_scores_with_captain(result.recommended_squad, actual_column="pontuacao")

    assert result.summary["actual_points"] == pytest.approx(actual_scores["actual_points_with_captain"])
    assert result.summary["actual_points_base"] == pytest.approx(actual_scores["actual_points_base"])
    assert result.summary["captain_bonus_actual"] == pytest.approx(actual_scores["captain_bonus_actual"])
    assert result.summary["actual_points_with_captain"] == pytest.approx(actual_scores["actual_points_with_captain"])
    assert result.summary["oracle_actual_points"] == pytest.approx(59.0)
    assert result.summary["oracle_gap"] == pytest.approx(59.0 - actual_scores["actual_points_with_captain"])
    assert result.summary["oracle_capture_rate"] == pytest.approx(actual_scores["actual_points_with_captain"] / 59.0)
    assert result.summary["oracle_optimizer_status"] == "Optimal"
    assert "pontuacao" in result.recommended_squad.columns
    assert "is_captain" in result.recommended_squad.columns
    assert result.summary["optimizer_status"] == "Optimal"


def test_run_recommendation_replay_nulls_missing_actual_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))
    season_df.loc[season_df["rodada"].eq(3), "pontuacao"] = pd.NA

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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

    assert result.summary["actual_points"] is None
    assert result.summary["actual_points_base"] is None
    assert result.summary["captain_bonus_actual"] is None
    assert result.summary["actual_points_with_captain"] is None
    assert result.summary["oracle_actual_points"] is None
    assert result.summary["oracle_gap"] is None
    assert result.summary["oracle_capture_rate"] is None
    assert result.summary["oracle_optimizer_status"] is None
    assert "Replay actual_points is null" in result.metadata["warnings"][0]


def test_run_recommendation_replay_nulls_oracle_when_any_candidate_actual_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))
    target_low_value = season_df["rodada"].eq(3) & season_df["posicao"].eq("lat") & season_df["apelido"].eq("lat-0")
    season_df.loc[target_low_value, "pontuacao"] = pd.NA
    season_df.loc[target_low_value, "preco_pre_rodada"] = 999.0

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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
    assert result.summary["oracle_actual_points"] is None
    assert result.summary["oracle_gap"] is None
    assert result.summary["oracle_capture_rate"] is None
    assert result.summary["oracle_optimizer_status"] is None
    assert (
        "Oracle actual_points is null because 1 candidate rows have missing or non-finite pontuacao."
        in (result.metadata["warnings"])
    )


def test_run_recommendation_replay_nulls_capture_rate_when_oracle_is_not_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))
    season_df.loc[season_df["rodada"].eq(3), "pontuacao"] = 0.0

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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

    assert result.summary["actual_points"] == 0.0
    assert result.summary["actual_points_base"] == 0.0
    assert result.summary["captain_bonus_actual"] == 0.0
    assert result.summary["actual_points_with_captain"] == 0.0
    assert result.summary["oracle_actual_points"] == 0.0
    assert result.summary["oracle_gap"] == 0.0
    assert result.summary["oracle_capture_rate"] is None
    assert "Oracle capture_rate is null because oracle_actual_points is not positive." in (result.metadata["warnings"])


def test_run_recommendation_live_suppresses_actual_columns_when_finalized_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4))

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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
    assert result.summary["actual_points_base"] is None
    assert result.summary["captain_bonus_actual"] is None
    assert result.summary["actual_points_with_captain"] is None
    assert result.summary["oracle_actual_points"] is None
    assert result.summary["oracle_gap"] is None
    assert result.summary["oracle_capture_rate"] is None
    assert result.summary["oracle_optimizer_status"] is None
    assert "pontuacao" not in result.recommended_squad.columns
    assert "entrou_em_campo" not in result.candidate_predictions.columns
    assert "is_captain" in result.recommended_squad.columns
    assert "captain_policy_ev" in result.recommended_squad.columns
    assert "captain_policy_safe" in result.recommended_squad.columns
    assert "captain_policy_upside" in result.recommended_squad.columns
    assert int(result.recommended_squad["captain_policy_ev"].sum()) == 1
    assert int(result.recommended_squad["captain_policy_safe"].sum()) == 1
    assert int(result.recommended_squad["captain_policy_upside"].sum()) == 1
    assert "is_captain" not in result.candidate_predictions.columns
    assert "captain_policy_ev" not in result.candidate_predictions.columns
    assert result.metadata["finalized_live_data_detected"] is True
    assert result.metadata["finalized_live_data_evidence"]["pontuacao_non_zero_count"] > 0
    assert result.metadata["finalized_live_data_evidence"]["entrou_em_campo_true_count"] > 0
    assert result.metadata["finalized_live_data_evidence"]["non_zero_scout_count"] > 0
    run_metadata = json.loads((config.output_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["finalized_live_data_detected"] is True
    assert run_metadata["finalized_live_data_evidence"]["pontuacao_non_zero_count"] > 0
    assert run_metadata["allow_finalized_live_data"] is True


def test_run_recommendation_outputs_captain_contract_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
    )

    result = run_recommendation(config)

    assert result.summary["scoring_contract_version"] == SCORING_CONTRACT_VERSION
    assert result.summary["captain_scoring_enabled"] is True
    assert result.summary["formation_search"] == "all_official_formations"
    assert result.summary["selected_count"] == 12
    assert result.summary["captain_id"] in set(result.recommended_squad["id_atleta"])
    assert len(result.recommended_squad) == 12
    assert int(result.recommended_squad["is_captain"].sum()) == 1
    assert {
        "rodada",
        "id_atleta",
        "apelido",
        "id_clube",
        "nome_clube",
        "posicao",
        "status",
        "preco_pre_rodada",
        "baseline_score",
        "price_score",
        "random_forest_score",
        "predicted_points",
        "is_captain",
        "captain_policy_ev",
        "captain_policy_safe",
        "captain_policy_upside",
    }.issubset(result.recommended_squad.columns)
    assert "predicted_points_base" in result.summary
    assert "captain_bonus_predicted" in result.summary
    assert "predicted_points_with_captain" in result.summary
    assert result.summary["predicted_points"] == result.summary["predicted_points_with_captain"]
    assert result.summary["budget_used"] <= result.summary["budget"]
    assert result.summary["predicted_points_base"] > 0
    assert result.summary["captain_bonus_predicted"] > 0
    assert "predicted_points" in result.recommended_squad.columns
    assert "is_captain" in result.recommended_squad.columns
    assert "captain_policy_ev" in result.recommended_squad.columns
    assert "captain_policy_safe" in result.recommended_squad.columns
    assert "captain_policy_upside" in result.recommended_squad.columns
    assert "is_captain" not in result.candidate_predictions.columns
    assert "captain_policy_ev" not in result.candidate_predictions.columns
    assert result.summary["captain_policy_diagnostics"]

    for key, value in contract_fields().items():
        assert result.metadata[key] == value
        assert result.summary[key] == value
    assert result.metadata["formation"] == result.summary["formation"]
    assert result.metadata["formation_search"] == "all_official_formations"


def test_run_recommendation_strict_matchup_loads_fixture_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cartola.backtesting.strict_fixtures import StrictFixturesLoadResult

    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    load_calls: list[dict[str, object]] = []

    def fake_load_strict_fixtures(**kwargs: object) -> StrictFixturesLoadResult:
        load_calls.append(kwargs)
        return StrictFixturesLoadResult(
            fixtures=_strict_fixture_rows(range(1, 4)),
            manifest_paths=[
                "data/01_raw/fixtures_strict/2026/partidas-1.manifest.json",
                "data/01_raw/fixtures_strict/2026/partidas-2.manifest.json",
                "data/01_raw/fixtures_strict/2026/partidas-3.manifest.json",
            ],
            manifest_sha256={
                "data/01_raw/fixtures_strict/2026/partidas-1.manifest.json": "a" * 64,
                "data/01_raw/fixtures_strict/2026/partidas-2.manifest.json": "b" * 64,
                "data/01_raw/fixtures_strict/2026/partidas-3.manifest.json": "c" * 64,
            },
            generator_versions=["fixture_snapshot_v1"],
        )

    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_strict_fixtures", fake_load_strict_fixtures)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
        fixture_mode="strict",
        matchup_context_mode="cartola_matchup_v1",
    )

    result = run_recommendation(config)

    assert load_calls == [
        {
            "season": 2026,
            "project_root": tmp_path,
            "required_rounds": [1, 2, 3],
        }
    ]
    assert result.metadata["fixture_mode"] == "strict"
    assert result.metadata["matchup_context_mode"] == "cartola_matchup_v1"
    assert result.metadata["fixture_source_directory"] == "data/01_raw/fixtures_strict/2026"
    assert result.metadata["fixture_manifest_paths"] == [
        "data/01_raw/fixtures_strict/2026/partidas-1.manifest.json",
        "data/01_raw/fixtures_strict/2026/partidas-2.manifest.json",
        "data/01_raw/fixtures_strict/2026/partidas-3.manifest.json",
    ]
    assert (
        result.metadata["fixture_manifest_sha256"]["data/01_raw/fixtures_strict/2026/partidas-3.manifest.json"]
        == "c" * 64
    )
    assert result.metadata["fixture_generator_versions"] == ["fixture_snapshot_v1"]
    assert "matchup_is_home" in result.metadata["feature_columns"]


def test_run_recommendation_strict_fixture_missing_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)

    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
        fixture_mode="strict",
        matchup_context_mode="cartola_matchup_v1",
    )

    with pytest.raises(FileNotFoundError, match="Required strict fixture"):
        run_recommendation(config)


def test_run_recommendation_supports_ridge_primary_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
        model_id="ridge",
    )

    result = run_recommendation(config)

    assert result.summary["strategy"] == "ridge"
    assert result.metadata["model_id"] == "ridge"
    assert "ridge_score" in result.recommended_squad.columns
    assert "ridge_score" in result.candidate_predictions.columns
    assert "random_forest_score" not in result.recommended_squad.columns
    assert "random_forest_score" not in result.candidate_predictions.columns


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

    with pytest.raises(ValueError, match="appears finalized") as exc_info:
        run_recommendation(config)

    error_message = str(exc_info.value)
    assert "pontuacao_non_zero_count" in error_message
    assert "entrou_em_campo_true_count" in error_message
    assert "non_zero_scout_count" in error_message
    assert not config.output_path.exists()


def test_run_recommendation_writes_expected_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)

    def fake_load_footystats(**kwargs: object) -> FootyStatsPPGLoadResult:
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
    assert (output_path / "risk_audit.json").exists()
    summary = json.loads((output_path / "recommendation_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_path / "run_metadata.json").read_text(encoding="utf-8"))
    risk_audit = json.loads((output_path / "risk_audit.json").read_text(encoding="utf-8"))
    assert {
        "season",
        "target_round",
        "mode",
        "strategy",
        "formation",
        "budget",
        "budget_used",
        "optimizer_status",
        "selected_count",
        "predicted_points",
        "predicted_points_base",
        "captain_bonus_predicted",
        "predicted_points_with_captain",
        "captain_id",
        "captain_name",
        "captain_position",
        "captain_club",
        "captain_policy_diagnostics",
        "output_directory",
        *contract_fields().keys(),
    }.issubset(summary)
    assert {
        "season",
        "target_round",
        "mode",
        "current_year",
        "training_rounds",
        "candidate_round",
        "visible_max_round",
        "fixture_mode",
        "matchup_context_mode",
        "model_id",
        "footystats_mode",
        "feature_columns",
        "playable_statuses",
        "formation",
        "allowed_formations",
        "captain_policy_definitions",
        "captain_policy_diagnostics",
        "budget",
        "random_seed",
        "finalized_live_data_detected",
        "finalized_live_data_evidence",
        "allow_finalized_live_data",
        "live_workflow",
        "optimizer_status",
        "warnings",
        "generated_at_utc",
        *contract_fields().keys(),
    }.issubset(metadata)
    assert summary["actual_points"] is None
    assert metadata["training_rounds"] == [1, 2]
    assert metadata["footystats_matches_source_sha256"] == "sha"
    assert risk_audit["schema_version"] == "cartola.risk_audit.v1"
    assert risk_audit["advisory_only"] is True
    assert risk_audit["budget_utilization_pct"] == pytest.approx(
        summary["budget_used"] / summary["budget"] * 100.0
    )
    assert risk_audit["captain_risk_policy"] in {"ev", "safe", "upside"}
    assert risk_audit["overall_risk_level"] in {"low", "medium", "high"}
    assert len(risk_audit["dnp_risk"]) == summary["selected_count"]


def test_run_recommendation_writes_live_workflow_metadata_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    live_workflow = {
        "capture_policy": "fresh",
        "target_round": 3,
        "capture_csv_path": str(tmp_path / "data/01_raw/2026/rodada-3.csv"),
        "capture_metadata_path": str(tmp_path / "data/01_raw/2026/rodada-3.capture.json"),
        "capture_csv_sha256": "a" * 64,
        "recommendation_output_path": str(
            tmp_path / "data/08_reporting/recommendations/2026/round-3/live/runs/run_started_at=20260429T123456000000Z"
        ),
    }
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
        output_run_id="run_started_at=20260429T123456000000Z",
        live_workflow=live_workflow,
    )

    result = run_recommendation(config)

    metadata_path = config.output_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["live_workflow"] == live_workflow
    assert result.metadata["live_workflow"] == live_workflow


def test_run_recommendation_rejects_absolute_output_root_outside_project(tmp_path: Path) -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        output_root=Path("/tmp/outside-cartola-recommendations"),
        current_year=2026,
        footystats_mode="none",
    )

    with pytest.raises(ValueError, match="inside project_root"):
        run_recommendation(config)
