from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting import ebm_feature_diagnostic
from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDependencyError,
    EbmDiagnosticConfig,
    EbmDiagnosticInvalid,
    SourceChildContext,
    inspect_ebm_runtime,
    prepare_diagnostic_dataset,
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


class _RecordingEbm:
    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_rows = 0
        self.feature_names_in_: list[str] = []
        self.term_names_: list[str] = []
        self.term_scores_: list[list[float]] = []
        self.bins_: list[list[list[float]]] = []

    def fit(self, x_values: pd.DataFrame, y_values: pd.Series) -> "_RecordingEbm":
        self.fit_rows = len(x_values)
        self.feature_names_in_ = list(x_values.columns)
        self.term_names_ = list(x_values.columns)
        self.term_scores_ = [[-0.5, 0.5] for _ in self.term_names_]
        self.bins_ = [[[0.0]] for _ in self.term_names_]
        return self

    def predict(self, x_values: pd.DataFrame) -> list[float]:
        return [0.25 for _ in range(len(x_values))]


class _SeriesPredictingEbm(_RecordingEbm):
    def predict(self, x_values: pd.DataFrame) -> pd.Series:
        return pd.Series([0.75 for _ in range(len(x_values))], index=range(len(x_values)))


class _RecordingEbmWithoutValidationSize:
    def __init__(self, *, interactions: int = 0, random_state: int | None = None) -> None:
        self.params: dict[str, object] = {
            "interactions": interactions,
            "random_state": random_state,
        }
        self.fit_rows = 0

    def fit(self, x_values: pd.DataFrame, y_values: pd.Series) -> "_RecordingEbmWithoutValidationSize":
        self.fit_rows = len(x_values)
        return self

    def predict(self, x_values: pd.DataFrame) -> list[float]:
        return [0.5 for _ in range(len(x_values))]


def _write_source_child(
    tmp_path: Path,
    *,
    child_id: str = "child-1",
    child_path: Path | None = None,
    output_path: str | None = None,
    season: int = 2025,
    model_id: str = "ridge",
    feature_pack: str = "ppg_xg",
    fixture_mode: str = "none",
    prediction_score_column: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_child_path = child_path or tmp_path / "children" / child_id
    resolved_child_path.mkdir(parents=True)
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
    child_metadata = {**parent_metadata, **(metadata_overrides or {})}
    (resolved_child_path / "run_metadata.json").write_text(json.dumps(child_metadata), encoding="utf-8")
    score_column = prediction_score_column or f"{model_id}_score"
    pd.DataFrame({"rodada": [5], "id_atleta": [10], score_column: [6.5]}).to_csv(
        resolved_child_path / "player_predictions.csv",
        index=False,
    )
    return {
        "child_id": child_id,
        "output_path": output_path if output_path is not None else str(resolved_child_path),
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "metadata": parent_metadata,
    }


def _write_parent(experiment_path: Path, child_runs: list[dict[str, object]]) -> None:
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "experiment_metadata.json").write_text(
        json.dumps({"experiment_id": "exp-1", "child_runs": child_runs}),
        encoding="utf-8",
    )


def _source_context(tmp_path: Path) -> SourceChildContext:
    return SourceChildContext(
        source_experiment_id="exp-1",
        season=2025,
        model_id="ridge",
        feature_pack="ppg_xg",
        fixture_mode="none",
        matchup_context_mode="none",
        footystats_mode="ppg_xg",
        budget_policy="moving",
        scoring_contract_version="cartola_standard_2026_v1",
        score_column="ridge_score",
        child_path=tmp_path / "child-1",
        source_prediction_provenance_status="verified",
    )


def test_build_season_folds_uses_whole_validation_seasons() -> None:
    folds = ebm_feature_diagnostic.build_season_folds((2025, 2021, 2023, 2024, 2022))

    assert folds == (
        ebm_feature_diagnostic.SeasonFold(
            fold_id="A",
            train_seasons=(2021, 2022),
            validation_season=2023,
        ),
        ebm_feature_diagnostic.SeasonFold(
            fold_id="B",
            train_seasons=(2021, 2022, 2023),
            validation_season=2024,
        ),
        ebm_feature_diagnostic.SeasonFold(
            fold_id="C",
            train_seasons=(2021, 2022, 2023, 2024),
            validation_season=2025,
        ),
    )
    assert {fold.inner_validation_mode for fold in folds} == {"disabled_full_outer_train"}


def test_build_season_folds_rejects_duplicate_seasons() -> None:
    with pytest.raises(EbmDiagnosticInvalid, match="Duplicate seasons"):
        ebm_feature_diagnostic.build_season_folds((2021, 2022, 2022))


def test_compute_predictive_metrics_uses_residual_corrected_predictions() -> None:
    metrics = ebm_feature_diagnostic.compute_predictive_metrics(
        pd.DataFrame(
            {
                "rodada": [5, 5, 5],
                "target_actual_points": [10.0, 20.0, 30.0],
                "source_model_score": [9.0, 19.0, 29.0],
                "predicted_actual_points": [8.0, 22.0, 34.0],
                "predicted_source_residual": [1.0, 1.0, 2.0],
            }
        ),
        fold_id="A",
        validation_season=2023,
    )

    by_prediction = metrics.set_index("prediction_type")
    assert by_prediction.loc["residual_corrected", "mae"] == pytest.approx(1 / 3)
    assert by_prediction.loc["actual_points", "mae"] == pytest.approx(8 / 3)
    assert metrics["discovery_only"].tolist() == [True, True, True]


def test_compute_predictive_metrics_uses_same_rows_for_all_predictions() -> None:
    metrics = ebm_feature_diagnostic.compute_predictive_metrics(
        pd.DataFrame(
            {
                "rodada": [5, 5, 5],
                "target_actual_points": [10.0, 20.0, 100.0],
                "source_model_score": [10.0, 20.0, 0.0],
                "predicted_actual_points": [11.0, 21.0, float("nan")],
                "predicted_source_residual": [0.0, 0.0, 0.0],
            }
        ),
        fold_id="A",
        validation_season=2023,
    )

    by_prediction = metrics.set_index("prediction_type")
    assert by_prediction["shared_evaluation_row_count"].tolist() == [2, 2, 2]
    assert by_prediction.loc["source_model", "mae"] == pytest.approx(0.0)
    assert by_prediction.loc["actual_points", "mae"] == pytest.approx(1.0)


def test_compute_predictive_metrics_computes_top50_spearman_by_round() -> None:
    metrics = ebm_feature_diagnostic.compute_predictive_metrics(
        pd.DataFrame(
            {
                "rodada": [7] * 50,
                "target_actual_points": list(range(50)),
                "source_model_score": list(range(50)),
                "predicted_actual_points": list(range(50)),
                "predicted_source_residual": [0.0] * 50,
            }
        ),
        fold_id="A",
        validation_season=2023,
    )

    source_row = metrics.set_index("prediction_type").loc["source_model"]
    assert math.isfinite(float(source_row["top50_spearman"]))
    assert source_row["top50_spearman"] == pytest.approx(1.0)


def test_compute_predictive_metrics_breaks_top50_ties_by_player_id() -> None:
    high_predicted_rows = [
        {
            "rodada": 7,
            "id_atleta": 100 + index,
            "target_actual_points": float(index),
            "source_model_score": 2.0,
            "predicted_actual_points": 2.0,
            "predicted_source_residual": 0.0,
        }
        for index in range(49)
    ]
    tied_cutoff_rows = [
        {
            "rodada": 7,
            "id_atleta": 2,
            "target_actual_points": 1000.0,
            "source_model_score": 1.0,
            "predicted_actual_points": 1.0,
            "predicted_source_residual": 0.0,
        },
        {
            "rodada": 7,
            "id_atleta": 1,
            "target_actual_points": -1000.0,
            "source_model_score": 1.0,
            "predicted_actual_points": 1.0,
            "predicted_source_residual": 0.0,
        },
    ]
    metrics = ebm_feature_diagnostic.compute_predictive_metrics(
        pd.DataFrame([*high_predicted_rows, *tied_cutoff_rows]),
        fold_id="A",
        validation_season=2023,
    )

    source_row = metrics.set_index("prediction_type").loc["source_model"]
    assert source_row["top50_spearman"] == pytest.approx(0.24253562503633297)


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


def test_fit_ebm_fold_target_disables_internal_validation() -> None:
    train = pd.DataFrame({"feature_a": [0.0, 1.0], "target_actual_points": [1.0, 2.0]})
    validation = pd.DataFrame({"feature_a": [2.0], "target_actual_points": [3.0]}, index=[99])

    result = ebm_feature_diagnostic.fit_ebm_fold_target(
        ebm_class=_RecordingEbm,
        train_rows=train,
        validation_rows=validation,
        feature_columns=("feature_a",),
        target_column="target_actual_points",
        target_type="actual_points",
        fold_id="A",
        validation_season=2023,
        random_seed=123,
    )

    assert result.predictions.tolist() == [0.25]
    assert result.predictions.index.tolist() == [99]
    assert result.model.params["interactions"] == 0
    assert result.model.params["validation_size"] == 0.0
    assert result.model.params["early_stopping_rounds"] == 100
    assert result.fit_row_count == 2


def test_fit_ebm_fold_target_preserves_prediction_positions() -> None:
    train = pd.DataFrame({"feature_a": [0.0, 1.0], "target_actual_points": [1.0, 2.0]})
    validation = pd.DataFrame({"feature_a": [2.0], "target_actual_points": [3.0]}, index=[99])

    result = ebm_feature_diagnostic.fit_ebm_fold_target(
        ebm_class=_SeriesPredictingEbm,
        train_rows=train,
        validation_rows=validation,
        feature_columns=("feature_a",),
        target_column="target_actual_points",
        target_type="actual_points",
        fold_id="A",
        validation_season=2023,
        random_seed=123,
    )

    assert result.predictions.tolist() == [0.75]
    assert result.predictions.index.tolist() == [99]


def test_fit_ebm_fold_target_omits_unsupported_validation_size() -> None:
    train = pd.DataFrame({"feature_a": [0.0, 1.0], "target_actual_points": [1.0, 2.0]})
    validation = pd.DataFrame({"feature_a": [2.0], "target_actual_points": [3.0]})

    result = ebm_feature_diagnostic.fit_ebm_fold_target(
        ebm_class=_RecordingEbmWithoutValidationSize,
        train_rows=train,
        validation_rows=validation,
        feature_columns=("feature_a",),
        target_column="target_actual_points",
        target_type="actual_points",
        fold_id="A",
        validation_season=2023,
        random_seed=123,
    )

    assert result.predictions.tolist() == [0.5]
    assert result.model.params["interactions"] == 0
    assert result.model.params["random_state"] == 123
    assert "validation_size" not in result.model.params


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


def test_resolve_source_children_resolves_project_relative_output_path_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    child_path = project_root / "data" / "08_reporting" / "backtests" / "2025" / "child-1"
    output_path = str(child_path.relative_to(project_root))
    child = _write_source_child(project_root, child_path=child_path, output_path=output_path)
    experiment_path = project_root / "data" / "08_reporting" / "experiments" / "exp-1"
    _write_parent(experiment_path, [child])
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

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
    assert contexts[0].child_path == child_path


def test_resolve_source_children_resolves_experiment_relative_output_path(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    child_path = experiment_path / "runs" / "child-1"
    child = _write_source_child(tmp_path, child_path=child_path, output_path="runs/child-1")
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
    assert contexts[0].child_path == child_path


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


def test_resolve_source_children_rejects_run_metadata_disagreement(tmp_path: Path) -> None:
    child = _write_source_child(tmp_path, metadata_overrides={"model_id": "random_forest"})
    experiment_path = tmp_path / "experiment"
    _write_parent(experiment_path, [child])

    with pytest.raises(EbmDiagnosticInvalid, match="run_metadata.json field model_id=.*disagrees"):
        resolve_source_children(
            EbmDiagnosticConfig(
                experiment_path=experiment_path,
                seasons=(2025,),
                model_id="ridge",
                feature_pack="ppg_xg",
                fixture_mode="none",
            )
        )


def test_resolve_source_children_rejects_non_object_matching_metadata(tmp_path: Path) -> None:
    child = _write_source_child(tmp_path)
    child["metadata"] = ["budget_policy", "moving"]
    experiment_path = tmp_path / "experiment"
    _write_parent(experiment_path, [child])

    with pytest.raises(EbmDiagnosticInvalid, match=r"child_runs\[0\]\.metadata.*object"):
        resolve_source_children(
            EbmDiagnosticConfig(
                experiment_path=experiment_path,
                seasons=(2025,),
                model_id="ridge",
                feature_pack="ppg_xg",
                fixture_mode="none",
            )
        )


def test_prepare_diagnostic_dataset_maps_dnp_nulls_to_zero_and_excludes_coaches(tmp_path: Path) -> None:
    dataset = prepare_diagnostic_dataset(
        _source_context(tmp_path),
        pd.DataFrame(
            {
                "season": [2025, 2025, 2025],
                "rodada": [5, 5, 5],
                "id_atleta": [10, 11, 12],
                "apelido": ["Played", "DNP", "Coach"],
                "id_clube": [1, 2, 3],
                "posicao": ["ata", "lat", "tec"],
                "status": ["Provavel", "Provavel", "Provavel"],
                "pontuacao": [7.0, None, 5.0],
                "entrou_em_campo": [True, False, True],
                "preco_pre_rodada": [10.0, 8.0, 12.0],
                "ridge_score": [6.0, 2.0, 4.0],
                "numeric_feature": [1.5, 2.5, 3.5],
            }
        ),
        feature_columns=("season", "rodada", "id_atleta", "id_clube", "apelido", "posicao", "numeric_feature"),
    )

    assert dataset.valid_rows["target_actual_points"].tolist() == [7.0, 0.0]
    assert dataset.valid_rows["target_source_residual"].tolist() == [1.0, -2.0]
    assert "posicao_ata" in dataset.feature_columns
    assert "posicao_lat" in dataset.feature_columns
    assert "posicao" not in dataset.feature_columns
    assert "numeric_feature" in dataset.feature_columns
    assert dataset.coach_row_count == 1
    assert dataset.invalid_rows.empty


def test_prepare_diagnostic_dataset_keeps_null_played_points_as_invalid(tmp_path: Path) -> None:
    dataset = prepare_diagnostic_dataset(
        _source_context(tmp_path),
        pd.DataFrame(
            {
                "rodada": [5],
                "id_atleta": [10],
                "apelido": ["Null Played"],
                "id_clube": [1],
                "posicao": ["ata"],
                "status": ["Provavel"],
                "pontuacao": [None],
                "entrou_em_campo": [True],
                "preco_pre_rodada": [10.0],
                "ridge_score": [6.0],
                "numeric_feature": [1.5],
            }
        ),
        feature_columns=("posicao", "numeric_feature"),
    )

    assert dataset.valid_rows.empty
    assert dataset.invalid_rows["id_atleta"].tolist() == [10]
    assert dataset.invalid_rows["invalid_reason"].tolist() == ["missing_actual_points_for_entered_player"]


def test_prepare_diagnostic_dataset_rejects_nonnumeric_retained_feature(tmp_path: Path) -> None:
    with pytest.raises(EbmDiagnosticInvalid, match="Feature column text_feature must be numeric and finite"):
        prepare_diagnostic_dataset(
            _source_context(tmp_path),
            pd.DataFrame(
                {
                    "rodada": [5],
                    "id_atleta": [10],
                    "apelido": ["Played"],
                    "id_clube": [1],
                    "posicao": ["ata"],
                    "status": ["Provavel"],
                    "pontuacao": [7.0],
                    "entrou_em_campo": [True],
                    "preco_pre_rodada": [10.0],
                    "ridge_score": [6.0],
                    "text_feature": ["bad"],
                }
            ),
            feature_columns=("posicao", "text_feature"),
        )
