from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.live_workflow import LiveWorkflowConfig, run_live_round
from cartola.backtesting.market_capture import LiveCaptureMetadata, MarketCaptureResult
from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult


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


def _recommendation_result(config: RecommendationConfig) -> RecommendationResult:
    summary = {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "budget": config.budget,
        "budget_used": 99.5,
        "predicted_points": 73.25,
        "selected_count": 12,
        "output_directory": str(config.output_path),
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

    def fake_capture(config, **kwargs):
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

    def fake_load_capture(**kwargs):
        assert kwargs == {"project_root": tmp_path, "season": 2026, "target_round": 14}
        return metadata

    def fake_recommend(config):
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
    assert result.workflow_metadata["status"] == "ok"


def test_run_live_round_missing_reuses_valid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    capture_calls = []
    status_calls = []
    recommendation_calls = []

    def fake_fetch_status(config):
        status_calls.append(config)
        return 14

    def fake_load_capture(**kwargs):
        return metadata

    def fake_capture(config, **kwargs):
        capture_calls.append(config)
        raise AssertionError("missing policy should not capture when valid capture exists")

    def fake_recommend(config):
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

    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise FileNotFoundError("live capture files missing for season=2026 target_round=14")
        return metadata

    def fake_capture(config, **kwargs):
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
    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        raise ValueError("destination is not a previous valid live capture")

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)

    with pytest.raises(ValueError, match="previous valid live capture"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="skip"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()


def test_run_live_round_missing_fails_on_invalid_existing_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls = []

    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        raise ValueError("destination is not a previous valid live capture")

    def fake_capture(config, **kwargs):
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

    def fail_recommendation(config):
        raise ValueError("FootyStats recommendation missing join keys: {14: [264]}")

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fail_recommendation)

    with pytest.raises(ValueError, match="missing join keys"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    output_path = (
        tmp_path
        / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
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
        tmp_path
        / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
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
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", lambda config: recommend_calls.append(config))

    with pytest.raises(FileExistsError, match="recommendation archive already exists"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    assert recommend_calls == []


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
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", lambda config: recommend_calls.append(config))

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

    for key in (
        "capture_policy",
        "target_round",
        "capture_csv_path",
        "capture_metadata_path",
        "capture_csv_sha256",
        "recommendation_output_path",
    ):
        assert workflow[key] == link[key]
