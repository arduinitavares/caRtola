from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.live_workflow import LiveWorkflowConfig, run_live_round
from cartola.backtesting.market_capture import LiveCaptureMetadata, MarketCaptureConfig, MarketCaptureResult
from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult
from cartola.backtesting.scoring_contract import contract_fields


def _capture_metadata(tmp_path: Path, *, round_number: int = 14) -> LiveCaptureMetadata:
    return LiveCaptureMetadata(
        csv_path=tmp_path / f"data/01_raw/2026/rodada-{round_number}.csv",
        metadata_path=tmp_path / f"data/01_raw/2026/rodada-{round_number}.capture.json",
        season=2026,
        target_round=round_number,
        csv_sha256="a" * 64,
        captured_at_utc="2026-04-29T12:00:00Z",
        status_mercado=1,
        deadline_timestamp=1777748340,
        deadline_parse_status="ok",
    )


def _round_frame(round_number: int, *, finalized: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    player_id = 1
    for position, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row: dict[str, object] = {
                "id_atleta": player_id,
                "apelido": f"{position}-{offset}",
                "slug": f"{position}-{offset}",
                "id_clube": player_id,
                "nome_clube": f"Club {player_id}",
                "posicao": position,
                "status": "Provavel",
                "rodada": round_number,
                "preco": 5.0,
                "preco_pre_rodada": 5.0,
                "pontuacao": float(round_number + offset) if finalized else 0.0,
                "media": float(round_number + offset),
                "num_jogos": round_number - 1,
                "variacao": 0.0,
                "entrou_em_campo": finalized,
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 1 if finalized and scout == "DS" else 0
            rows.append(row)
            player_id += 1
    return pd.DataFrame(rows)


def _season_frame(rounds: range, *, target_round: int) -> pd.DataFrame:
    return pd.concat(
        [_round_frame(round_number, finalized=round_number != target_round) for round_number in rounds],
        ignore_index=True,
    )


def _recommendation_result(config: RecommendationConfig) -> RecommendationResult:
    summary = {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "budget": config.budget,
        "budget_used": 99.5,
        "predicted_points": 73.25,
        "predicted_points_base": 70.0,
        "captain_bonus_predicted": 3.25,
        "captain_id": 123,
        "captain_name": "Captain A",
        "formation": "4-3-3",
        "selected_count": 12,
        "output_directory": str(config.output_path),
        **contract_fields(),
    }
    config.output_path.mkdir(parents=True, exist_ok=True)
    (config.output_path / "run_metadata.json").write_text(
        json.dumps({"live_workflow": config.live_workflow}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RecommendationResult(
        recommended_squad=pd.DataFrame(),
        candidate_predictions=pd.DataFrame(),
        summary=summary,
        metadata={"live_workflow": config.live_workflow},
    )


def test_run_live_round_fresh_captures_and_uses_capture_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls = []
    recommendation_calls = []
    metadata = _capture_metadata(tmp_path, round_number=14)

    def fake_capture(config: MarketCaptureConfig, **kwargs: object) -> MarketCaptureResult:
        capture_calls.append(config)
        return MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    def fake_load_capture(**kwargs: object) -> LiveCaptureMetadata:
        assert kwargs == {"project_root": tmp_path, "season": 2026, "target_round": 14}
        return metadata

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert capture_calls[0].auto is True
    assert capture_calls[0].force is True
    assert recommendation_calls[0].target_round == 14
    assert recommendation_calls[0].mode == "live"
    assert recommendation_calls[0].output_run_id == "run_started_at=20260429T123456123456Z"
    assert recommendation_calls[0].live_workflow["capture_policy"] == "fresh"
    assert recommendation_calls[0].live_workflow["capture_csv_sha256"] == "a" * 64
    assert result.workflow_metadata["predicted_points"] == 73.25
    assert result.workflow_metadata["predicted_points_base"] == 70.0
    assert result.workflow_metadata["captain_bonus_predicted"] == 3.25
    assert result.workflow_metadata["captain_id"] == 123
    assert result.workflow_metadata["captain_name"] == "Captain A"
    assert result.workflow_metadata["formation"] == "4-3-3"
    assert result.workflow_metadata["formation_search"] == "all_official_formations"
    assert result.workflow_metadata["captain_scoring_enabled"] is True
    assert result.workflow_metadata["status"] == "ok"


def test_run_live_round_passes_strict_matchup_modes_to_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    recommendation_calls: list[RecommendationConfig] = []

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(
            season=2026,
            project_root=tmp_path,
            current_year=2026,
            model_id="xgboost_depth2_slow",
            footystats_mode="ppg_xg",
            fixture_mode="strict",
            matchup_context_mode="cartola_matchup_v1",
        ),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert recommendation_calls[0].fixture_mode == "strict"
    assert recommendation_calls[0].matchup_context_mode == "cartola_matchup_v1"
    assert recommendation_calls[0].model_id == "xgboost_depth2_slow"
    assert recommendation_calls[0].footystats_mode == "ppg_xg"
    assert recommendation_calls[0].live_workflow["fixture_mode"] == "strict"
    assert recommendation_calls[0].live_workflow["matchup_context_mode"] == "cartola_matchup_v1"
    assert result.workflow_metadata["fixture_mode"] == "strict"
    assert result.workflow_metadata["matchup_context_mode"] == "cartola_matchup_v1"


def test_run_live_round_missing_reuses_valid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    capture_calls = []
    status_calls = []
    recommendation_calls = []

    def fake_fetch_status(config: LiveWorkflowConfig) -> int:
        status_calls.append(config)
        return 14

    def fake_load_capture(**kwargs: object) -> LiveCaptureMetadata:
        return metadata

    def fake_capture(config: MarketCaptureConfig, **kwargs: object) -> NoReturn:
        capture_calls.append(config)
        raise AssertionError("missing policy should not capture when valid capture exists")

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
        now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
    )

    assert status_calls
    assert capture_calls == []
    assert recommendation_calls[0].target_round == 14
    assert result.workflow_metadata["capture_age_seconds"] == 300.0


def test_run_live_round_missing_captures_when_capture_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    capture_calls = []
    load_calls = 0

    def fake_fetch_status(config: LiveWorkflowConfig) -> int:
        return 14

    def fake_load_capture(**kwargs: object) -> LiveCaptureMetadata:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise FileNotFoundError("live capture files missing for season=2026 target_round=14")
        return metadata

    def fake_capture(config: MarketCaptureConfig, **kwargs: object) -> MarketCaptureResult:
        capture_calls.append(config)
        return MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", _recommendation_result)

    run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
        now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
    )

    assert capture_calls[0].auto is True
    assert capture_calls[0].force is False


def test_run_live_round_skip_requires_valid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_status(config: LiveWorkflowConfig) -> int:
        return 14

    def fake_load_capture(**kwargs: object) -> NoReturn:
        raise ValueError("destination is not a previous valid live capture")

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)

    with pytest.raises(ValueError, match="previous valid live capture"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="skip"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()


def test_run_live_round_skip_rejects_closed_capture_before_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recommendation_calls = []

    def fake_fetch_status(config: LiveWorkflowConfig) -> int:
        return 14

    def fake_load_capture(**kwargs: object) -> NoReturn:
        raise ValueError("destination is not a previous valid live capture: status_mercado=2 is not open")

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    with pytest.raises(ValueError, match="status_mercado=2"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="skip"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert recommendation_calls == []
    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()


def test_run_live_round_missing_fails_on_invalid_existing_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls = []

    def fake_fetch_status(config: LiveWorkflowConfig) -> int:
        return 14

    def fake_load_capture(**kwargs: object) -> NoReturn:
        raise ValueError("destination is not a previous valid live capture")

    def fake_capture(config: MarketCaptureConfig, **kwargs: object) -> NoReturn:
        capture_calls.append(config)
        raise AssertionError("invalid existing capture must not be overwritten by missing policy")

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)

    with pytest.raises(ValueError, match="previous valid live capture"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert capture_calls == []
    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()


def test_run_live_round_recommendation_failure_writes_failed_workflow_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)

    def fail_recommendation(config: RecommendationConfig) -> NoReturn:
        raise ValueError("FootyStats recommendation missing join keys: {14: [264]}")

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fail_recommendation)

    with pytest.raises(ValueError, match="missing join keys"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    output_path = (
        tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
    )
    workflow = json.loads((output_path / "live_workflow_metadata.json").read_text(encoding="utf-8"))
    assert workflow["status"] == "failed"
    assert workflow["error_stage"] == "recommendation"
    assert workflow["capture_csv_sha256"] == "a" * 64
    assert workflow["error_type"] == "ValueError"


def test_run_live_round_archive_collision_fails_before_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    output_path = (
        tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
    )
    output_path.mkdir(parents=True)
    recommend_calls = []

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.run_recommendation", lambda config: recommend_calls.append(config)
    )

    with pytest.raises(FileExistsError, match="recommendation archive already exists"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    assert recommend_calls == []


def test_run_live_round_creates_archive_before_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    archive_exists_during_recommendation = False

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        nonlocal archive_exists_during_recommendation
        archive_exists_during_recommendation = config.output_path.is_dir()
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert archive_exists_during_recommendation is True
    assert result.output_path == (
        tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
    )


def test_run_live_round_rejects_unsafe_output_root_before_failure_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    outside_output_root = tmp_path / "outside-recommendations"
    metadata = _capture_metadata(project_root, round_number=14)
    recommend_calls = []

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.run_recommendation", lambda config: recommend_calls.append(config)
    )

    with pytest.raises(ValueError, match="Recommendation output_root must resolve inside project_root"):
        run_live_round(
            LiveWorkflowConfig(
                season=2026,
                project_root=project_root,
                output_root=outside_output_root,
                current_year=2026,
            ),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    assert recommend_calls == []
    assert not outside_output_root.exists()


def test_workflow_metadata_matches_recommendation_live_workflow_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", _recommendation_result)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert result.output_path is not None
    workflow = json.loads((result.output_path / "live_workflow_metadata.json").read_text(encoding="utf-8"))
    recommendation_metadata = json.loads((result.output_path / "run_metadata.json").read_text(encoding="utf-8"))
    link = recommendation_metadata["live_workflow"]

    assert {
        "workflow_version",
        "run_started_at_utc",
        "capture_policy",
        "season",
        "current_year",
        "target_round",
        "budget",
        "model_id",
        "fixture_mode",
        "matchup_context_mode",
        "footystats_mode",
        "footystats_league_slug",
        "capture_csv_path",
        "capture_metadata_path",
        "capture_csv_sha256",
        "capture_captured_at_utc",
        "capture_age_seconds",
        "capture_status_mercado",
        "capture_deadline_timestamp",
        "capture_deadline_parse_status",
        "recommendation_output_path",
        "recommendation_summary_path",
        "recommendation_metadata_path",
        "recommended_squad_path",
        "candidate_predictions_path",
        "selected_count",
        "predicted_points",
        "predicted_points_base",
        "captain_bonus_predicted",
        "captain_id",
        "captain_name",
        "formation",
        "budget_used",
        "finalized_live_data_detected",
        "finalized_live_data_evidence",
        "allow_finalized_live_data",
        "status",
        "error_stage",
        "error_type",
        "error_message",
    }.issubset(workflow)

    for key in (
        "capture_policy",
        "target_round",
        "capture_csv_path",
        "capture_metadata_path",
        "capture_csv_sha256",
        "recommendation_output_path",
    ):
        assert workflow[key] == link[key]


def test_workflow_metadata_includes_finalized_override_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    evidence = {
        "pontuacao_non_zero_count": 1,
        "entrou_em_campo_true_count": 2,
        "non_zero_scout_count": 3,
    }

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)

    def fake_recommend(config: RecommendationConfig) -> RecommendationResult:
        result = _recommendation_result(config)
        return RecommendationResult(
            recommended_squad=result.recommended_squad,
            candidate_predictions=result.candidate_predictions,
            summary=result.summary,
            metadata={
                **result.metadata,
                "finalized_live_data_detected": True,
                "finalized_live_data_evidence": evidence,
                "allow_finalized_live_data": True,
            },
        )

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(
            season=2026,
            project_root=tmp_path,
            current_year=2026,
            allow_finalized_live_data=True,
        ),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert result.workflow_metadata["finalized_live_data_detected"] is True
    assert result.workflow_metadata["finalized_live_data_evidence"] == evidence
    assert result.workflow_metadata["allow_finalized_live_data"] is True


def test_run_live_round_writes_canonical_artifacts_and_unique_run_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=3)
    season_df = _season_frame(range(1, 4), target_round=3)

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=3,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *args, **kwargs: season_df)

    config = LiveWorkflowConfig(
        season=2026,
        budget=100.0,
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
    )

    first = run_live_round(config, now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC))
    second = run_live_round(config, now=lambda: datetime(2026, 4, 29, 12, 34, 57, 123456, tzinfo=UTC))

    assert first.output_path is not None
    assert second.output_path is not None
    assert first.output_path != second.output_path
    assert first.output_path.name == "run_started_at=20260429T123456123456Z"
    assert second.output_path.name == "run_started_at=20260429T123457123456Z"

    required_summary_fields = {
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
    }
    required_metadata_fields = {
        "season",
        "target_round",
        "mode",
        "current_year",
        "training_rounds",
        "candidate_round",
        "visible_max_round",
        "fixture_mode",
        "matchup_context_mode",
        "fixture_source_directory",
        "fixture_manifest_paths",
        "fixture_manifest_sha256",
        "fixture_generator_versions",
        "model_id",
        "footystats_mode",
        "footystats_evaluation_scope",
        "footystats_league_slug",
        "footystats_matches_source_path",
        "footystats_matches_source_sha256",
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
    }
    required_workflow_fields = {
        "workflow_version",
        "run_started_at_utc",
        "capture_policy",
        "season",
        "current_year",
        "target_round",
        "budget",
        "model_id",
        "fixture_mode",
        "matchup_context_mode",
        "footystats_mode",
        "footystats_league_slug",
        "capture_csv_path",
        "capture_metadata_path",
        "capture_csv_sha256",
        "capture_captured_at_utc",
        "capture_age_seconds",
        "capture_status_mercado",
        "capture_deadline_timestamp",
        "capture_deadline_parse_status",
        "recommendation_output_path",
        "recommendation_summary_path",
        "recommendation_metadata_path",
        "recommended_squad_path",
        "candidate_predictions_path",
        "selected_count",
        "predicted_points",
        "predicted_points_base",
        "captain_bonus_predicted",
        "captain_id",
        "captain_name",
        "formation",
        "budget_used",
        "finalized_live_data_detected",
        "finalized_live_data_evidence",
        "allow_finalized_live_data",
        "status",
        "error_stage",
        "error_type",
        "error_message",
    }
    required_squad_columns = {
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
    }
    forbidden_live_columns = {"pontuacao", "entrou_em_campo", *DEFAULT_SCOUT_COLUMNS}
    forbidden_candidate_columns = {
        "is_captain",
        "captain_policy_ev",
        "captain_policy_safe",
        "captain_policy_upside",
        *forbidden_live_columns,
    }

    for result in (first, second):
        output_path = result.output_path
        assert output_path is not None
        assert (output_path / "recommended_squad.csv").exists()
        assert (output_path / "candidate_predictions.csv").exists()
        assert (output_path / "recommendation_summary.json").exists()
        assert (output_path / "run_metadata.json").exists()
        assert (output_path / "live_workflow_metadata.json").exists()
        assert (output_path / "risk_audit.json").exists()
        assert (output_path / "run_manifest.json").exists()

        squad = pd.read_csv(output_path / "recommended_squad.csv")
        candidates = pd.read_csv(output_path / "candidate_predictions.csv")
        summary = json.loads((output_path / "recommendation_summary.json").read_text(encoding="utf-8"))
        metadata_json = json.loads((output_path / "run_metadata.json").read_text(encoding="utf-8"))
        workflow = json.loads((output_path / "live_workflow_metadata.json").read_text(encoding="utf-8"))
        risk_audit = json.loads((output_path / "risk_audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_path / "run_manifest.json").read_text(encoding="utf-8"))

        assert required_squad_columns.issubset(squad.columns)
        assert forbidden_live_columns.isdisjoint(squad.columns)
        assert forbidden_candidate_columns.isdisjoint(candidates.columns)
        assert len(squad) == 12
        assert summary["selected_count"] == 12
        assert squad["is_captain"].astype(str).str.lower().eq("true").sum() == 1

        assert required_summary_fields.issubset(summary)
        assert required_metadata_fields.issubset(metadata_json)
        assert required_workflow_fields.issubset(workflow)
        assert {"pontuacao_non_zero_count", "entrou_em_campo_true_count", "non_zero_scout_count"}.issubset(
            metadata_json["finalized_live_data_evidence"]
        )
        assert workflow["recommendation_output_path"] == str(output_path)
        assert workflow["recommended_squad_path"] == str(output_path / "recommended_squad.csv")
        assert workflow["candidate_predictions_path"] == str(output_path / "candidate_predictions.csv")
        assert workflow["recommendation_summary_path"] == str(output_path / "recommendation_summary.json")
        assert workflow["recommendation_metadata_path"] == str(output_path / "run_metadata.json")
        assert metadata_json["live_workflow"]["capture_csv_sha256"] == "a" * 64
        assert workflow["capture_csv_sha256"] == "a" * 64
        assert workflow["status"] == "ok"
        assert workflow["error_stage"] is None
        assert workflow["error_type"] is None
        assert workflow["error_message"] is None
        assert summary["budget_used"] <= 100.0
        assert summary["actual_points"] is None
        assert risk_audit["schema_version"] == "cartola.risk_audit.v1"
        assert risk_audit["advisory_only"] is True
        assert risk_audit["budget_utilization_pct"] == pytest.approx(summary["budget_used"])
        assert risk_audit["overall_risk_level"] in {"low", "medium", "high"}
        assert len(risk_audit["dnp_risk"]) == 12
        assert manifest["schema_version"] == "cartola.run_manifest.v1"
        assert manifest["generated_at_utc"] == result.workflow_metadata["run_started_at_utc"]
        manifest_by_path = {entry["relative_path"]: entry for entry in manifest["artifacts"]}
        assert set(manifest_by_path) == {
            "candidate_predictions.csv",
            "live_workflow_metadata.json",
            "recommendation_summary.json",
            "recommended_squad.csv",
            "risk_audit.json",
            "run_metadata.json",
        }
        assert manifest["artifact_count"] == len(manifest_by_path)
        for relative_path, entry in manifest_by_path.items():
            artifact_path = output_path / relative_path
            assert entry["sha256"] == sha256(artifact_path.read_bytes()).hexdigest()
            assert entry["size_bytes"] == artifact_path.stat().st_size
            assert str(entry["created_at_utc"]).endswith("Z")
        assert list(output_path.glob(".run_manifest.*")) == []
