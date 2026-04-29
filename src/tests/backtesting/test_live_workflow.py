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
