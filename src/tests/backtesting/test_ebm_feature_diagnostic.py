from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDependencyError,
    EbmDiagnosticConfig,
    EbmDiagnosticInvalid,
    inspect_ebm_runtime,
    resolve_source_children,
)


class _FakeEbm:
    def __init__(
        self,
        *,
        interactions: int = 0,
        validation_size: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.interactions = interactions
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, x_values: object, y_values: object) -> "_FakeEbm":
        return self


class _FakeEbmWithValidation:
    def fit(
        self,
        x_values: object,
        y_values: object,
        X_val: object,
        y_val: object,
    ) -> "_FakeEbmWithValidation":
        return self


class _FakeEbmWithValidationNameSubstrings:
    def fit(
        self,
        x_values: object,
        y_values: object,
        not_X_val: object,
        not_y_val: object,
    ) -> "_FakeEbmWithValidationNameSubstrings":
        return self


def _write_source_child(
    tmp_path: Path,
    *,
    child_id: str = "child-1",
    season: int = 2025,
    model_id: str = "ridge",
    feature_pack: str = "ppg_xg",
    fixture_mode: str = "none",
    prediction_score_column: str | None = None,
) -> dict[str, object]:
    child_path = tmp_path / "children" / child_id
    child_path.mkdir(parents=True)
    parent_metadata: dict[str, object] = {
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "matchup_context_mode": "none",
        "footystats_mode": "ppg_xg",
        "budget_policy": "moving",
        "scoring_contract_version": "cartola_standard_2026_v1",
    }
    (child_path / "run_metadata.json").write_text(json.dumps(parent_metadata), encoding="utf-8")
    score_column = prediction_score_column or f"{model_id}_score"
    pd.DataFrame({"rodada": [5], "id_atleta": [10], score_column: [6.5]}).to_csv(
        child_path / "player_predictions.csv",
        index=False,
    )
    return {
        "child_id": child_id,
        "output_path": str(child_path),
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "metadata": parent_metadata,
    }


def _write_parent(experiment_path: Path, child_runs: list[dict[str, object]]) -> None:
    experiment_path.mkdir(parents=True)
    (experiment_path / "experiment_metadata.json").write_text(
        json.dumps({"experiment_id": "exp-1", "child_runs": child_runs}),
        encoding="utf-8",
    )


def test_inspect_ebm_runtime_records_constructor_and_fit_signatures() -> None:
    info = inspect_ebm_runtime(ebm_class=_FakeEbm, package_version="9.9.9")

    assert info.available is True
    assert info.version == "9.9.9"
    assert "validation_size" in info.constructor_signature
    assert "x_values" in info.fit_signature
    assert info.supports_explicit_validation is False


def test_inspect_ebm_runtime_detects_explicit_validation_parameters() -> None:
    info = inspect_ebm_runtime(
        ebm_class=_FakeEbmWithValidation,
        package_version="9.9.9",
    )

    assert info.supports_explicit_validation is True


def test_inspect_ebm_runtime_ignores_validation_name_substrings() -> None:
    info = inspect_ebm_runtime(
        ebm_class=_FakeEbmWithValidationNameSubstrings,
        package_version="9.9.9",
    )

    assert info.supports_explicit_validation is False


def test_inspect_ebm_runtime_raises_clear_error_when_missing() -> None:
    with pytest.raises(EbmDependencyError, match="InterpretML is required"):
        inspect_ebm_runtime(ebm_class=None, package_version=None)


def test_resolve_source_children_requires_one_match_per_season(tmp_path: Path) -> None:
    child = _write_source_child(tmp_path)
    experiment_path = tmp_path / "experiment"
    _write_parent(experiment_path, [child])

    contexts, report = resolve_source_children(
        EbmDiagnosticConfig(
            experiment_path=experiment_path,
            seasons=(2025,),
            model_id="ridge",
            feature_pack="ppg_xg",
            fixture_mode="none",
        )
    )

    assert report.empty
    assert len(contexts) == 1
    context = contexts[0]
    assert context.season == 2025
    assert context.score_column == "ridge_score"
    assert context.source_prediction_provenance_status == "verified"


def test_resolve_source_children_reports_duplicate_matches(tmp_path: Path) -> None:
    first = _write_source_child(tmp_path, child_id="child-1")
    second = _write_source_child(tmp_path, child_id="child-2")
    experiment_path = tmp_path / "experiment"
    _write_parent(experiment_path, [first, second])

    with pytest.raises(EbmDiagnosticInvalid, match="Duplicate source child matches"):
        resolve_source_children(
            EbmDiagnosticConfig(
                experiment_path=experiment_path,
                seasons=(2025,),
                model_id="ridge",
                feature_pack="ppg_xg",
                fixture_mode="none",
            )
        )


def test_resolve_source_children_rejects_missing_score_column(tmp_path: Path) -> None:
    child = _write_source_child(tmp_path, prediction_score_column="random_forest_score")
    experiment_path = tmp_path / "experiment"
    _write_parent(experiment_path, [child])

    with pytest.raises(EbmDiagnosticInvalid, match="Missing score column.*ridge_score"):
        resolve_source_children(
            EbmDiagnosticConfig(
                experiment_path=experiment_path,
                seasons=(2025,),
                model_id="ridge",
                feature_pack="ppg_xg",
                fixture_mode="none",
            )
        )
