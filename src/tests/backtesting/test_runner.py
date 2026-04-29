from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS, BacktestConfig
from cartola.backtesting.features import FOOTYSTATS_PPG_FEATURE_COLUMNS, FOOTYSTATS_XG_FEATURE_COLUMNS
from cartola.backtesting.footystats_features import FootyStatsJoinDiagnostics, FootyStatsPPGLoadResult
from cartola.backtesting.runner import run_backtest
from cartola.backtesting.scoring_contract import contract_fields
from cartola.backtesting.strict_fixtures import StrictFixturesLoadResult


def _tiny_round(round_number: int) -> pd.DataFrame:
    rows = []
    player_id = 1
    for pos, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row = {
                "id_atleta": player_id,
                "apelido": f"{pos}-{offset}",
                "slug": f"{pos}-{offset}",
                "id_clube": player_id,
                "nome_clube": "Club",
                "posicao": pos,
                "status": "Provavel",
                "rodada": round_number,
                "preco": 5.0 + round_number,
                "preco_pre_rodada": 5.0,
                "pontuacao": float(10 - offset + round_number),
                "media": float(5 + offset),
                "num_jogos": round_number,
                "variacao": 0.0,
                "entrou_em_campo": True,
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 0
            rows.append(row)
            player_id += 1
    return pd.DataFrame(rows)


def _tiny_fixtures(rounds: range) -> pd.DataFrame:
    rows = []
    for round_number in rounds:
        for home_id in range(1, 18, 2):
            rows.append(
                {
                    "rodada": round_number,
                    "id_clube_home": home_id,
                    "id_clube_away": home_id + 1,
                    "data": "2025-04-26",
                }
            )
    return pd.DataFrame(rows, columns=["rodada", "id_clube_home", "id_clube_away", "data"])


def _write_tiny_fixture_files(root: Path, rounds: range) -> None:
    fixture_dir = root / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    fixtures = _tiny_fixtures(rounds)
    for round_number, round_fixtures in fixtures.groupby("rodada", sort=True):
        round_fixtures.to_csv(fixture_dir / f"partidas-{round_number}.csv", index=False)


def _tiny_footystats_rows(rounds: range, clubs: range = range(1, 19)) -> pd.DataFrame:
    rows = []
    for round_number in rounds:
        for club_id in clubs:
            opponent_id = club_id + 1 if club_id % 2 == 1 else club_id - 1
            team_ppg = round_number + (club_id / 100)
            opponent_ppg = round_number + (opponent_id / 100)
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


def test_run_backtest_writes_round_players_predictions_and_summary(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert set(result.round_results["strategy"]) == {"baseline", "random_forest", "price"}
    assert (tmp_path / "data/08_reporting/backtests/2025/round_results.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/selected_players.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/player_predictions.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/summary.csv").exists()
    assert (tmp_path / "data/08_reporting/backtests/2025/diagnostics.csv").exists()


def test_run_backtest_writes_metadata_for_no_fixture_mode(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    run_backtest(config, season_df=season_df)

    metadata = pd.read_json(tmp_path / "data/08_reporting/backtests/2025/run_metadata.json", typ="series").to_dict()
    assert metadata["fixture_mode"] == "none"
    assert metadata["fixture_source_directory"] is None
    assert metadata["fixture_manifest_paths"] == []
    assert metadata["fixture_manifest_sha256"] == {}
    assert metadata["warnings"] == []
    for field, expected_value in contract_fields().items():
        assert metadata[field] == expected_value


def test_run_backtest_metadata_records_default_footystats_mode(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    run_backtest(config, season_df=season_df)

    metadata = pd.read_json(tmp_path / "data/08_reporting/backtests/2025/run_metadata.json", typ="series").to_dict()
    assert metadata["footystats_mode"] == "none"
    assert metadata["footystats_evaluation_scope"] == "historical_candidate"
    assert metadata["footystats_league_slug"] == "brazil-serie-a"
    assert metadata["footystats_matches_source_path"] is None
    assert metadata["footystats_matches_source_sha256"] is None
    assert metadata["footystats_feature_columns"] == []
    assert metadata["footystats_missing_join_keys_by_round"] == {}
    assert metadata["footystats_duplicate_join_keys_by_round"] == {}
    assert metadata["footystats_extra_club_rows_by_round"] == {}


def test_run_backtest_rejects_live_current_scope(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        footystats_mode="ppg",
        footystats_evaluation_scope="live_current",
    )

    with pytest.raises(ValueError, match="live_current is not supported"):
        run_backtest(config, season_df=season_df)


def test_run_backtest_rejects_live_current_scope_with_footystats_none(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        footystats_mode="none",
        footystats_evaluation_scope="live_current",
    )

    with pytest.raises(ValueError, match="live_current is not supported"):
        run_backtest(config, season_df=season_df)


def test_run_backtest_ppg_passes_footystats_rows_and_metadata(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    observed_calls: list[dict[str, object]] = []
    source_path = tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        observed_calls.append(kwargs)
        return FootyStatsPPGLoadResult(
            rows=_tiny_footystats_rows(range(1, 6)),
            source_path=source_path,
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        footystats_mode="ppg",
        footystats_dir=Path("custom/footystats"),
        footystats_league_slug="custom-league",
        current_year=2025,
    )

    result = run_backtest(config, season_df=season_df)

    assert observed_calls == [
        {
            "season": 2025,
            "project_root": tmp_path,
            "footystats_dir": Path("custom/footystats"),
            "league_slug": "custom-league",
            "evaluation_scope": "historical_candidate",
            "current_year": 2025,
            "footystats_mode": "ppg",
        }
    ]
    assert result.metadata.footystats_mode == "ppg"
    assert result.metadata.footystats_matches_source_path == str(source_path)
    assert result.metadata.footystats_matches_source_sha256 == "fake-sha"
    assert result.metadata.footystats_feature_columns == FOOTYSTATS_PPG_FEATURE_COLUMNS
    assert result.metadata.footystats_missing_join_keys_by_round == {}
    assert result.metadata.footystats_duplicate_join_keys_by_round == {}
    assert result.metadata.footystats_extra_club_rows_by_round == {}
    assert "footystats_ppg_diff" in result.player_predictions.columns

    metadata = pd.read_json(tmp_path / "data/08_reporting/backtests/2025/run_metadata.json", typ="series").to_dict()
    assert metadata["footystats_mode"] == "ppg"
    assert metadata["footystats_matches_source_path"] == str(source_path)
    assert metadata["footystats_matches_source_sha256"] == "fake-sha"
    assert metadata["footystats_feature_columns"] == FOOTYSTATS_PPG_FEATURE_COLUMNS


def test_run_backtest_ppg_xg_passes_mode_and_records_feature_columns(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    source_path = tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv"
    rows = _tiny_footystats_rows(range(1, 6))
    rows["footystats_team_pre_match_xg"] = 1.2
    rows["footystats_opponent_pre_match_xg"] = 0.8
    rows["footystats_xg_diff"] = 0.4
    calls: list[dict[str, object]] = []
    feature_columns = (*FOOTYSTATS_PPG_FEATURE_COLUMNS, *FOOTYSTATS_XG_FEATURE_COLUMNS)

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        calls.append(kwargs)
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=source_path,
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
            footystats_mode="ppg_xg",
            feature_columns=feature_columns,
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )

    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, footystats_mode="ppg_xg")
    result = run_backtest(config, season_df=season_df)

    assert calls[0]["footystats_mode"] == "ppg_xg"
    assert result.metadata.footystats_feature_columns == list(feature_columns)
    assert "footystats_xg_diff" in result.player_predictions.columns


def test_run_backtest_ppg_rejects_missing_join_keys(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    rows = _tiny_footystats_rows(range(1, 6))
    rows = rows[~((rows["rodada"] == 5) & (rows["id_clube"] == 18))].copy()

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv",
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, footystats_mode="ppg")

    with pytest.raises(ValueError, match="missing join keys"):
        run_backtest(config, season_df=season_df)


def test_run_backtest_ppg_records_extra_footystats_rows(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    rows = pd.concat(
        [
            _tiny_footystats_rows(range(1, 6)),
            pd.DataFrame(
                [
                    {
                        "rodada": 5,
                        "id_clube": 99,
                        "opponent_id_clube": 1,
                        "is_home_footystats": 1,
                        "footystats_team_pre_match_ppg": 2.0,
                        "footystats_opponent_pre_match_ppg": 1.0,
                        "footystats_ppg_diff": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    def fake_load_footystats_feature_rows(**kwargs: object) -> FootyStatsPPGLoadResult:
        return FootyStatsPPGLoadResult(
            rows=rows,
            source_path=tmp_path / "data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv",
            source_sha256="fake-sha",
            diagnostics=FootyStatsJoinDiagnostics(),
        )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_footystats_feature_rows",
        fake_load_footystats_feature_rows,
        raising=False,
    )
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, footystats_mode="ppg")

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.footystats_extra_club_rows_by_round == {"5": [{"rodada": 5, "id_clube": 99}]}


def test_run_backtest_default_none_ignores_exploratory_fixture_files(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    _write_tiny_fixture_files(tmp_path, range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.fixture_mode == "none"
    assert result.metadata.fixture_source_directory is None
    assert result.metadata.warnings == []
    assert result.player_predictions["is_home"].eq(0).all()


def test_run_backtest_default_none_ignores_explicit_fixtures(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df, fixtures=fixtures)

    assert result.metadata.fixture_mode == "none"
    assert result.player_predictions["is_home"].eq(0).all()


def test_run_backtest_uses_fixture_files_when_available(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    _write_tiny_fixture_files(tmp_path, range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="exploratory")

    result = run_backtest(config, season_df=season_df)
    round_5 = result.player_predictions[result.player_predictions["rodada"] == 5]
    club_1 = round_5[round_5["id_clube"] == 1].iloc[0]
    club_2 = round_5[round_5["id_clube"] == 2].iloc[0]

    assert result.metadata.fixture_mode == "exploratory"
    assert result.metadata.fixture_source_directory == "data/01_raw/fixtures/2025"
    assert result.metadata.warnings
    assert club_1["is_home"] == 1
    assert club_2["is_home"] == 0
    assert "opponent_club_points_roll3" in result.player_predictions.columns


def test_run_backtest_exploratory_without_files_records_no_source(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="exploratory")

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.fixture_mode == "exploratory"
    assert result.metadata.fixture_source_directory is None
    assert "not found" in " ".join(result.metadata.warnings)
    assert result.player_predictions["is_home"].eq(0).all()


def test_run_backtest_uses_explicit_fixtures_without_fixture_files(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="exploratory")

    result = run_backtest(config, season_df=season_df, fixtures=fixtures)
    round_5 = result.player_predictions[result.player_predictions["rodada"] == 5]
    club_1 = round_5[round_5["id_clube"] == 1].iloc[0]
    club_2 = round_5[round_5["id_clube"] == 2].iloc[0]

    assert club_1["is_home"] == 1
    assert club_2["is_home"] == 0


def test_run_backtest_strict_mode_missing_files_raises_for_first_required_round(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="strict")

    with pytest.raises(FileNotFoundError, match="partidas-1"):
        run_backtest(config, season_df=season_df)


def test_run_backtest_strict_mode_loads_required_rounds_and_records_manifest_metadata(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    observed_calls: list[dict[str, object]] = []

    def fake_load_strict_fixtures(**kwargs: object) -> StrictFixturesLoadResult:
        observed_calls.append(kwargs)
        return StrictFixturesLoadResult(
            fixtures=_tiny_fixtures(range(1, 6)),
            manifest_paths=[
                "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json",
                "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json",
            ],
            manifest_sha256={
                "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "sha-1",
                "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json": "sha-2",
            },
            generator_versions=["fixture_snapshot_v1"],
        )

    monkeypatch.setattr("cartola.backtesting.runner.load_strict_fixtures", fake_load_strict_fixtures)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="strict")

    result = run_backtest(config, season_df=season_df)

    assert observed_calls == [
        {
            "season": 2025,
            "project_root": tmp_path,
            "required_rounds": [1, 2, 3, 4, 5],
        }
    ]
    assert result.metadata.fixture_mode == "strict"
    assert result.metadata.fixture_source_directory == "data/01_raw/fixtures_strict/2025"
    assert result.metadata.fixture_manifest_paths == [
        "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json",
        "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json",
    ]
    assert result.metadata.fixture_manifest_sha256 == {
        "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "sha-1",
        "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json": "sha-2",
    }
    assert result.metadata.generator_versions == ["fixture_snapshot_v1"]


def test_run_backtest_rejects_fixture_alignment_gaps(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 2, "data": "2025-04-26"},
        ],
        columns=["rodada", "id_clube_home", "id_clube_away", "data"],
    )
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="exploratory")

    with pytest.raises(ValueError, match="Fixture alignment failed"):
        run_backtest(config, season_df=season_df, fixtures=fixtures)


def test_strict_alignment_policy_exclude_round_removes_invalid_round_before_training(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    fixtures = fixtures[fixtures["rodada"] != 3].copy()
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=3,
        budget=100,
        fixture_mode="strict",
        strict_alignment_policy="exclude_round",
    )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_strict_fixtures",
        lambda **kwargs: StrictFixturesLoadResult(
            fixtures=fixtures,
            manifest_paths=["data/01_raw/fixtures_strict/2025/partidas-1.manifest.json"],
            manifest_sha256={"data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "abc"},
            generator_versions=["fixture_snapshot_v1"],
        ),
    )

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.excluded_rounds == [3]
    assert result.metadata.fixture_manifest_paths == ["data/01_raw/fixtures_strict/2025/partidas-1.manifest.json"]
    assert result.metadata.fixture_manifest_sha256 == {
        "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "abc"
    }
    assert 3 not in set(result.player_predictions["rodada"].dropna().astype(int).tolist())
    assert 3 not in set(result.round_results["rodada"].dropna().astype(int).tolist())


def test_strict_alignment_policy_exclude_round_removes_missing_strict_fixture_round(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=3,
        budget=100,
        fixture_mode="strict",
        strict_alignment_policy="exclude_round",
    )

    def fake_load_strict_fixtures(**kwargs: object) -> StrictFixturesLoadResult:
        required_rounds = kwargs["required_rounds"]
        if 3 in required_rounds:
            raise FileNotFoundError("partidas-3")

        round_number = required_rounds[0]
        manifest_path = f"data/01_raw/fixtures_strict/2025/partidas-{round_number}.manifest.json"
        return StrictFixturesLoadResult(
            fixtures=_tiny_fixtures(range(round_number, round_number + 1)),
            manifest_paths=[manifest_path],
            manifest_sha256={manifest_path: f"sha-{round_number}"},
            generator_versions=["fixture_snapshot_v1"],
        )

    monkeypatch.setattr("cartola.backtesting.runner.load_strict_fixtures", fake_load_strict_fixtures)

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.excluded_rounds == [3]
    assert 3 not in set(result.player_predictions["rodada"].dropna().astype(int).tolist())
    assert 3 not in set(result.round_results["rodada"].dropna().astype(int).tolist())
    assert result.metadata.fixture_manifest_paths == [
        "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json",
        "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json",
        "data/01_raw/fixtures_strict/2025/partidas-4.manifest.json",
        "data/01_raw/fixtures_strict/2025/partidas-5.manifest.json",
    ]
    missing_manifest_path = "data/01_raw/fixtures_strict/2025/partidas-3.manifest.json"
    assert missing_manifest_path not in result.metadata.fixture_manifest_paths
    assert result.metadata.fixture_manifest_sha256 == {
        "data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "sha-1",
        "data/01_raw/fixtures_strict/2025/partidas-2.manifest.json": "sha-2",
        "data/01_raw/fixtures_strict/2025/partidas-4.manifest.json": "sha-4",
        "data/01_raw/fixtures_strict/2025/partidas-5.manifest.json": "sha-5",
    }
    assert result.metadata.generator_versions == ["fixture_snapshot_v1"]


def test_run_backtest_records_selected_players_and_prediction_diagnostics(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.round_results["solver_status"].eq("Optimal").all()
    assert result.round_results["selected_count"].eq(12).all()
    assert {
        "predicted_points_base",
        "captain_bonus_predicted",
        "predicted_points_with_captain",
        "actual_points_base",
        "captain_bonus_actual",
        "actual_points_with_captain",
        "captain_id",
        "captain_name",
        "captain_policy_ev_id",
        "captain_policy_safe_id",
        "captain_policy_upside_id",
        "actual_points_with_ev_captain",
        "actual_points_with_safe_captain",
        "actual_points_with_upside_captain",
    }.issubset(result.round_results.columns)
    optimal = result.round_results[result.round_results["solver_status"].eq("Optimal")]
    assert optimal["predicted_points"].equals(optimal["predicted_points_with_captain"])
    assert optimal["actual_points"].equals(optimal["actual_points_with_captain"])
    assert set(result.selected_players["strategy"]) == {"baseline", "random_forest", "price"}
    assert result.selected_players["rodada"].eq(5).all()
    assert result.selected_players["predicted_points"].notna().all()
    assert "is_captain" in result.selected_players.columns
    captain_counts = result.selected_players.groupby(["rodada", "strategy"])["is_captain"].sum()
    assert captain_counts.eq(1).all()
    assert len(result.selected_players) == 36
    assert {
        "rodada",
        "baseline_score",
        "random_forest_score",
        "price_score",
    }.issubset(result.player_predictions.columns)
    assert len(result.player_predictions) == 18
    assert set(result.summary["strategy"]) == {"baseline", "random_forest", "price"}
    assert {
        "section",
        "strategy",
        "position",
        "metric",
        "value",
    }.issubset(result.diagnostics.columns)
    assert not result.diagnostics.empty


def test_run_backtest_normalizes_tiny_float_drift_in_returned_outputs(tmp_path, monkeypatch):
    class NoisyRandomForestPointPredictor:
        calls = 0

        def __init__(self, random_seed: int = 123, feature_columns: list[str] | None = None) -> None:
            self.random_seed = random_seed
            self.feature_columns = feature_columns

        def fit(self, frame: pd.DataFrame) -> "NoisyRandomForestPointPredictor":
            return self

        def predict(self, frame: pd.DataFrame) -> pd.Series:
            NoisyRandomForestPointPredictor.calls += 1
            return pd.Series(
                frame["prior_points_mean"].astype(float) + NoisyRandomForestPointPredictor.calls * 0.000000000001,
                index=frame.index,
            )

    monkeypatch.setattr(
        "cartola.backtesting.runner.RandomForestPointPredictor",
        NoisyRandomForestPointPredictor,
    )
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)

    first = run_backtest(
        BacktestConfig(project_root=tmp_path / "first", start_round=5, budget=100), season_df=season_df
    )
    second = run_backtest(
        BacktestConfig(project_root=tmp_path / "second", start_round=5, budget=100), season_df=season_df
    )

    assert_frame_equal(first.round_results, second.round_results, check_exact=True)
    assert_frame_equal(first.selected_players, second.selected_players, check_exact=True)
    assert_frame_equal(first.player_predictions, second.player_predictions, check_exact=True)
    assert_frame_equal(first.summary, second.summary, check_exact=True)


def test_selected_players_predicted_points_match_strategy_score_column(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    for strategy, score_column in {
        "baseline": "baseline_score",
        "random_forest": "random_forest_score",
        "price": "price_score",
    }.items():
        selected = result.selected_players[result.selected_players["strategy"] == strategy]
        assert selected["predicted_points"].equals(selected[score_column])
        captain = selected.loc[selected["is_captain"]].iloc[0]
        assert captain["predicted_points"] == captain[score_column]
        assert captain["predicted_points"] * 1.5 != captain["predicted_points"]


def test_run_backtest_skipped_round_uses_empty_formation_without_config_formation_name(tmp_path):
    season_df = _tiny_round(1)
    config = BacktestConfig(project_root=tmp_path, start_round=1, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert set(result.round_results["solver_status"]) == {"TrainingEmpty"}
    assert result.round_results["formation"].eq("").all()
    assert result.round_results["selected_count"].eq(0).all()
    assert result.round_results["predicted_points"].eq(0.0).all()
    assert result.round_results["actual_points"].eq(0.0).all()


def test_price_strategy_scores_market_open_price_not_post_round_price(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.player_predictions["price_score"].eq(result.player_predictions["preco_pre_rodada"]).all()
    assert not result.player_predictions["price_score"].eq(result.player_predictions["preco"]).all()
