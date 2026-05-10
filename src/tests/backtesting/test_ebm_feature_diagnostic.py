from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from cartola.backtesting import ebm_feature_diagnostic
from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDependencyError,
    EbmDiagnosticConfig,
    EbmDiagnosticInvalid,
    SourceChildContext,
    aggregate_candidate_hypotheses,
    assign_continuous_bins,
    build_ebm_feature_diagnostic,
    compute_interaction_cell_support,
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


class _PipelineFakeEbm:
    fit_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **params: object) -> None:
        self.params = params
        self.prediction = 0.0

    def fit(self, x_values: pd.DataFrame, y_values: pd.Series) -> "_PipelineFakeEbm":
        numeric_target = pd.to_numeric(y_values, errors="raise")
        self.prediction = float(numeric_target.mean())
        type(self).fit_calls.append(
            {
                "row_count": len(x_values),
                "target_name": str(y_values.name),
                "validation_size": self.params.get("validation_size"),
            }
        )
        return self

    def predict(self, x_values: pd.DataFrame) -> list[float]:
        return [self.prediction for _ in range(len(x_values))]


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


def _write_synthetic_player_predictions(
    child: dict[str, object],
    *,
    season: int,
    model_id: str,
) -> None:
    child_path = Path(str(child["output_path"]))
    metadata_path = child_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_columns"] = ["feature_a", "feature_b", "posicao"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    score_column = f"{model_id}_score"
    rows = []
    for index in range(60):
        source_score = 2.0 + float(index % 12) * 0.25 + float(season - 2021) * 0.1
        rows.append(
            {
                "season": season,
                "rodada": 5,
                "id_atleta": season * 1000 + index,
                "apelido": f"A{season}-{index}",
                "id_clube": 1 + index % 4,
                "posicao": "ata" if index < 30 else "lat",
                "status": "Provavel",
                "preco_pre_rodada": 8.0 + float(index % 5),
                "pontuacao": source_score + float((index % 7) - 3) * 0.2,
                "entrou_em_campo": True,
                score_column: source_score,
                "feature_a": float(index),
                "feature_b": float(index % 6),
            }
        )
    pd.DataFrame(rows).to_csv(child_path / "player_predictions.csv", index=False)


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


def test_assign_continuous_bins_matches_learned_edges_and_missing_bin() -> None:
    values = pd.Series([None, -1.0, 0.0, 0.5, 2.0])

    bins = assign_continuous_bins(values, learned_edges=(0.0, 1.0))

    assert bins.tolist() == [-1, 0, 1, 1, 2]


@pytest.mark.parametrize("learned_edges", [(1.0, 0.0), (0.0, float("nan")), (0.0, float("inf"))])
def test_assign_continuous_bins_rejects_invalid_learned_edges(learned_edges: tuple[float, ...]) -> None:
    with pytest.raises(EbmDiagnosticInvalid, match="learned_edges"):
        assign_continuous_bins(pd.Series([0.5]), learned_edges=learned_edges)


def test_compute_interaction_cell_support_counts_rows_and_rounds() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [5, 5, 6, 7],
            "feature_a_bin": [0, 0, 1, 1],
            "feature_b_bin": [1, 1, 1, 2],
        }
    )

    support = compute_interaction_cell_support(
        frame,
        feature_a_bin="feature_a_bin",
        feature_b_bin="feature_b_bin",
    )

    assert support[(0, 1)] == {"row_support": 2, "round_support": 1}
    assert support[(1, 1)] == {"row_support": 1, "round_support": 1}
    assert support[(1, 2)] == {"row_support": 1, "round_support": 1}


@pytest.mark.parametrize("missing_column", ["rodada", "feature_a_bin", "feature_b_bin"])
def test_compute_interaction_cell_support_rejects_missing_required_columns(missing_column: str) -> None:
    frame = pd.DataFrame(
        {
            "rodada": [5],
            "feature_a_bin": [0],
            "feature_b_bin": [1],
        }
    ).drop(columns=missing_column)

    with pytest.raises(EbmDiagnosticInvalid, match=missing_column):
        compute_interaction_cell_support(
            frame,
            feature_a_bin="feature_a_bin",
            feature_b_bin="feature_b_bin",
        )


def test_compute_interaction_cell_support_rejects_nan_bin_values() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [5],
            "feature_a_bin": [float("nan")],
            "feature_b_bin": [1],
        }
    )

    with pytest.raises(EbmDiagnosticInvalid, match="NaN.*feature_a_bin"):
        compute_interaction_cell_support(
            frame,
            feature_a_bin="feature_a_bin",
            feature_b_bin="feature_b_bin",
        )


def test_compute_interaction_cell_support_rejects_non_integral_bin_values() -> None:
    frame = pd.DataFrame(
        {
            "rodada": [5],
            "feature_a_bin": [1.5],
            "feature_b_bin": [1],
        }
    )

    with pytest.raises(EbmDiagnosticInvalid, match="integral.*feature_a_bin"):
        compute_interaction_cell_support(
            frame,
            feature_a_bin="feature_a_bin",
            feature_b_bin="feature_b_bin",
        )


def _main_effect_summary(
    *,
    target_type: str = "source_residual",
    feature_name: str = "feature_a",
    fold_ids: tuple[str, ...] = ("A", "B"),
    validation_seasons: tuple[int, ...] = (2023, 2024),
    monotonicity_hints: tuple[str, ...] = ("increasing", "increasing"),
    row_support: tuple[object, ...] = (600, 650),
    positive_bin_row_support: tuple[object, ...] = (60, 70),
    positive_bin_round_support: tuple[object, ...] = (5, 6),
    negative_bin_row_support: tuple[object, ...] = (55, 65),
    negative_bin_round_support: tuple[object, ...] = (5, 6),
    fold_candidate_signals: tuple[bool, ...] = (True, True),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discovery_only": [True] * len(fold_ids),
            "target_type": [target_type] * len(fold_ids),
            "feature_name": [feature_name] * len(fold_ids),
            "fold_id": list(fold_ids),
            "validation_season": list(validation_seasons),
            "effect_range": [0.6, 0.7][: len(fold_ids)],
            "largest_positive_bin_row_support": list(positive_bin_row_support),
            "largest_positive_bin_round_support": list(positive_bin_round_support),
            "largest_negative_bin_row_support": list(negative_bin_row_support),
            "largest_negative_bin_round_support": list(negative_bin_round_support),
            "monotonicity_hint": list(monotonicity_hints),
            "row_support": list(row_support),
            "fold_candidate_signal": list(fold_candidate_signals),
        }
    )


def _pairwise_interactions(
    *,
    target_type: str = "source_residual",
    fold_ids: tuple[str, ...] = ("A", "B"),
    validation_seasons: tuple[int, ...] = (2023, 2024),
    row_support: tuple[object, ...] = (600, 650),
    max_cell_row_support: tuple[object, ...] = (60, 70),
    max_cell_round_support: tuple[object, ...] = (5, 6),
    min_cell_row_support: tuple[object, ...] = (55, 65),
    min_cell_round_support: tuple[object, ...] = (5, 6),
    fold_candidate_signals: tuple[bool, ...] = (True, True),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discovery_only": [True] * len(fold_ids),
            "target_type": [target_type] * len(fold_ids),
            "interaction_name": ["feature_a x feature_b"] * len(fold_ids),
            "feature_a": ["feature_a"] * len(fold_ids),
            "feature_b": ["feature_b"] * len(fold_ids),
            "fold_id": list(fold_ids),
            "validation_season": list(validation_seasons),
            "effect_range": [0.9, 1.1][: len(fold_ids)],
            "term_support_extraction_status": ["ok"] * len(fold_ids),
            "max_effect_cell_row_support": list(max_cell_row_support),
            "max_effect_cell_round_support": list(max_cell_round_support),
            "min_effect_cell_row_support": list(min_cell_row_support),
            "min_effect_cell_round_support": list(min_cell_round_support),
            "row_support": list(row_support),
            "season_support": [2] * len(fold_ids),
            "fold_candidate_signal": list(fold_candidate_signals),
        }
    )


def test_aggregate_candidate_hypotheses_flags_residual_two_fold_signal() -> None:
    summaries = _main_effect_summary()

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["target_type"] == "source_residual"
    assert row["candidate_type"] == "main_effect"
    assert row["term_name"] == "feature_a"
    assert row["feature_a"] == "feature_a"
    assert row["feature_b"] == ""
    assert row["fold_signal_count"] == 2
    assert row["validation_seasons_with_signal"] == "2023,2024"
    assert row["total_row_support"] == 1250
    assert bool(row["candidate_hypothesis_flag"])
    assert row["candidate_scope"] == "human_review_only"


def test_aggregate_candidate_hypotheses_does_not_flag_actual_points_only_signal() -> None:
    summaries = _main_effect_summary(target_type="actual_points")

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["target_type"] == "actual_points"
    assert not bool(row["candidate_hypothesis_flag"])
    assert row["candidate_scope"] == "human_review_only"


def test_aggregate_candidate_hypotheses_does_not_flag_contradictory_directions() -> None:
    summaries = _main_effect_summary(monotonicity_hints=("increasing", "decreasing"))

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["direction_summary"] == "decreasing,increasing"
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_does_not_flag_low_main_effect_bin_support() -> None:
    summaries = _main_effect_summary(
        positive_bin_row_support=(49, 70),
        negative_bin_round_support=(5, 4),
    )

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    row = candidates.iloc[0]
    assert row["min_bin_or_cell_row_support"] == 49
    assert row["min_bin_or_cell_round_support"] == 4
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_rejects_nan_main_effect_signal_support() -> None:
    summaries = _main_effect_summary(positive_bin_row_support=(float("nan"), 70))

    with pytest.raises(EbmDiagnosticInvalid, match=r"feature_a.*largest_positive_bin_row_support"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=summaries,
            pairwise_interactions=pd.DataFrame(),
        )


def test_aggregate_candidate_hypotheses_does_not_flag_monotone_and_shaped_directions() -> None:
    summaries = _main_effect_summary(monotonicity_hints=("increasing", "u_shaped"))

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    row = candidates.iloc[0]
    assert row["direction_summary"] == "increasing,u_shaped"
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_rejects_duplicate_main_effect_signal_rows() -> None:
    summaries = _main_effect_summary(
        fold_ids=("A", "A"),
        validation_seasons=(2023, 2023),
    )

    with pytest.raises(EbmDiagnosticInvalid, match=r"Duplicate signal rows.*feature_a.*A.*2023"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=summaries,
            pairwise_interactions=pd.DataFrame(),
        )


def test_aggregate_candidate_hypotheses_rejects_mixed_signal_duplicate_main_effect_rows() -> None:
    summaries = _main_effect_summary(
        fold_ids=("A", "A"),
        validation_seasons=(2023, 2023),
        fold_candidate_signals=(True, False),
    )

    with pytest.raises(EbmDiagnosticInvalid, match=r"Duplicate fold rows.*feature_a.*A.*2023"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=summaries,
            pairwise_interactions=pd.DataFrame(),
        )


def test_aggregate_candidate_hypotheses_requires_distinct_main_effect_validation_seasons() -> None:
    summaries = _main_effect_summary(
        fold_ids=("A", "B"),
        validation_seasons=(2023, 2023),
    )

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=summaries,
        pairwise_interactions=pd.DataFrame(),
    )

    row = candidates.iloc[0]
    assert row["fold_signal_count"] == 2
    assert row["validation_seasons_with_signal"] == "2023"
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_flags_residual_pairwise_signal() -> None:
    interactions = _pairwise_interactions()

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=interactions,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["target_type"] == "source_residual"
    assert row["candidate_type"] == "interaction"
    assert row["term_name"] == "feature_a x feature_b"
    assert row["feature_a"] == "feature_a"
    assert row["feature_b"] == "feature_b"
    assert row["fold_signal_count"] == 2
    assert row["total_row_support"] == 1250
    assert row["min_bin_or_cell_row_support"] == 55
    assert row["min_bin_or_cell_round_support"] == 5
    assert row["direction_summary"] == "interaction_mixed"
    assert bool(row["candidate_hypothesis_flag"])
    assert row["candidate_scope"] == "human_review_only"


def test_aggregate_candidate_hypotheses_does_not_flag_actual_points_pairwise_signal() -> None:
    interactions = _pairwise_interactions(target_type="actual_points")

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=interactions,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["target_type"] == "actual_points"
    assert row["candidate_type"] == "interaction"
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_does_not_flag_low_pairwise_cell_support() -> None:
    interactions = _pairwise_interactions(
        max_cell_row_support=(49, 70),
        min_cell_round_support=(5, 4),
    )

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=interactions,
    )

    row = candidates.iloc[0]
    assert row["min_bin_or_cell_row_support"] == 49
    assert row["min_bin_or_cell_round_support"] == 4
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_rejects_nonnumeric_pairwise_signal_support() -> None:
    interactions = _pairwise_interactions(max_cell_row_support=("bad", 70))

    with pytest.raises(EbmDiagnosticInvalid, match=r"feature_a x feature_b.*max_effect_cell_row_support"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=interactions,
        )


def test_aggregate_candidate_hypotheses_rejects_duplicate_pairwise_signal_rows() -> None:
    interactions = _pairwise_interactions(
        fold_ids=("A", "A"),
        validation_seasons=(2023, 2023),
    )

    with pytest.raises(EbmDiagnosticInvalid, match=r"Duplicate signal rows.*feature_a x feature_b.*A.*2023"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=interactions,
        )


def test_aggregate_candidate_hypotheses_rejects_mixed_signal_duplicate_pairwise_rows() -> None:
    interactions = _pairwise_interactions(
        fold_ids=("A", "A"),
        validation_seasons=(2023, 2023),
        fold_candidate_signals=(True, False),
    )

    with pytest.raises(EbmDiagnosticInvalid, match=r"Duplicate fold rows.*feature_a x feature_b.*A.*2023"):
        aggregate_candidate_hypotheses(
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=interactions,
        )


def test_aggregate_candidate_hypotheses_requires_distinct_pairwise_validation_seasons() -> None:
    interactions = _pairwise_interactions(
        fold_ids=("A", "B"),
        validation_seasons=(2023, 2023),
    )

    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=interactions,
    )

    row = candidates.iloc[0]
    assert row["fold_signal_count"] == 2
    assert row["validation_seasons_with_signal"] == "2023"
    assert not bool(row["candidate_hypothesis_flag"])


def test_aggregate_candidate_hypotheses_empty_result_has_expected_columns() -> None:
    candidates = aggregate_candidate_hypotheses(
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=pd.DataFrame(),
    )

    assert candidates.empty
    assert list(candidates.columns) == [
        "discovery_only",
        "target_type",
        "candidate_type",
        "term_name",
        "feature_a",
        "feature_b",
        "fold_signal_count",
        "validation_seasons_with_signal",
        "total_row_support",
        "min_bin_or_cell_row_support",
        "min_bin_or_cell_round_support",
        "effect_range_median",
        "direction_summary",
        "failed_validation_seasons",
        "candidate_hypothesis_flag",
        "candidate_scope",
    ]


def test_write_ebm_diagnostic_artifacts_adds_discovery_metadata(tmp_path: Path) -> None:
    writer = getattr(ebm_feature_diagnostic, "write_ebm_diagnostic_artifacts", None)
    assert writer is not None

    output = tmp_path / "out"
    writer(
        output_path=output,
        manifest={"diagnostic_status": "invalid"},
        source_context=pd.DataFrame([{"discovery_only": False, "match_status": "missing"}]),
        fold_assignments=pd.DataFrame(),
        predictive_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        feature_shape_summary=pd.DataFrame(),
        pairwise_interactions=pd.DataFrame(),
        candidate_hypotheses=pd.DataFrame(),
        invalid_rows=pd.DataFrame(),
        invalid_report=pd.DataFrame([{"discovery_only": False, "reason_type": "schema", "message": "missing"}]),
        decision={"diagnostic_status": "invalid"},
    )

    expected_files = {
        "ebm_diagnostic_manifest.json",
        "ebm_diagnostic_decision.json",
        "source_context.csv",
        "fold_assignments.csv",
        "predictive_metrics.csv",
        "feature_importance_by_fold.csv",
        "feature_shape_summary.csv",
        "pairwise_interactions.csv",
        "candidate_hypotheses.csv",
        "invalid_ebm_rows.csv",
        "invalid_diagnostic_report.csv",
        "ebm_feature_diagnostic.html",
    }
    assert {path.name for path in output.iterdir()} == expected_files

    manifest = json.loads((output / "ebm_diagnostic_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "ebm_diagnostic_decision.json").read_text(encoding="utf-8"))
    assert manifest == {"diagnostic_status": "invalid", "discovery_only": True}
    assert decision == {"diagnostic_status": "invalid", "discovery_only": True}

    csv_artifacts = expected_files - {
        "ebm_diagnostic_manifest.json",
        "ebm_diagnostic_decision.json",
        "ebm_feature_diagnostic.html",
    }
    for artifact_name in sorted(csv_artifacts):
        frame = pd.read_csv(output / artifact_name)
        assert frame.columns[0] == "discovery_only"

    source_context = pd.read_csv(output / "source_context.csv")
    invalid_report = pd.read_csv(output / "invalid_diagnostic_report.csv")
    assert source_context["discovery_only"].tolist() == [True]
    assert invalid_report["discovery_only"].tolist() == [True]
    assert list(pd.read_csv(output / "predictive_metrics.csv").columns) == [
        "discovery_only",
        "target_type",
        "prediction_type",
        "fold_id",
        "validation_season",
        "shared_evaluation_row_count",
        "mae",
        "rmse",
        "spearman",
        "top50_spearman",
        "calibration_slope",
        "mean_prediction_bias",
    ]
    assert list(pd.read_csv(output / "invalid_ebm_rows.csv").columns) == [
        "discovery_only",
        "season",
        "rodada",
        "id_atleta",
        "apelido",
        "posicao",
        "invalid_reason",
        "pontuacao",
        "entrou_em_campo",
    ]

    html_report = (output / "ebm_feature_diagnostic.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html_report
    assert "discovery_only=true" in html_report
    assert "diagnostic_status=invalid" in html_report


def test_build_ebm_feature_diagnostic_runs_full_pipeline_with_injected_ebm(tmp_path: Path) -> None:
    model_id = "xgboost_depth2_slow"
    child_runs = [
        _write_source_child(
            tmp_path,
            child_id=f"child-{season}",
            season=season,
            model_id=model_id,
            feature_pack="ppg_xg_matchup",
            fixture_mode="exploratory",
        )
        for season in (2021, 2022, 2023)
    ]
    for child, season in zip(child_runs, (2021, 2022, 2023), strict=True):
        _write_synthetic_player_predictions(child, season=season, model_id=model_id)
    experiment_path = tmp_path / "experiment"
    output_path = tmp_path / "ebm-output"
    _write_parent(experiment_path, child_runs)
    events: list[str] = []
    _PipelineFakeEbm.fit_calls = []

    result = build_ebm_feature_diagnostic(
        experiment_path=experiment_path,
        output_path=output_path,
        seasons=(2021, 2022, 2023),
        model_id=model_id,
        feature_pack="ppg_xg_matchup",
        fixture_mode="exploratory",
        current_year=2026,
        max_interactions=10,
        min_validation_rows=50,
        random_seed=123,
        profile_runtime=True,
        progress_callback=events.append,
        ebm_class=_PipelineFakeEbm,
    )

    manifest = json.loads((output_path / "ebm_diagnostic_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output_path / "ebm_diagnostic_decision.json").read_text(encoding="utf-8"))
    fold_assignments = pd.read_csv(output_path / "fold_assignments.csv")
    predictive_metrics = pd.read_csv(output_path / "predictive_metrics.csv")
    assert result.decision["diagnostic_status"] == "diagnostic_complete"
    assert manifest["diagnostic_phase"] == "full_pipeline"
    assert decision["diagnostic_phase"] == "full_pipeline"
    assert manifest["source_child_count"] == 3
    assert manifest["profile_runtime"] is True
    assert manifest["max_interactions"] == 10
    assert manifest["min_validation_rows"] == 50
    assert not fold_assignments.empty
    assert not predictive_metrics.empty
    assert (output_path / "invalid_ebm_rows.csv").is_file()
    assert (output_path / "invalid_diagnostic_report.csv").is_file()
    assert {call["target_name"] for call in _PipelineFakeEbm.fit_calls} == {
        "target_actual_points",
        "target_source_residual",
    }
    assert any("source validation" in event for event in events)
    assert any("dataset load" in event for event in events)
    assert any("fold=A target=actual_points pass=main_effect" in event for event in events)
    assert any("fold=A target=source_residual pass=main_effect" in event for event in events)
    assert any("artifact write" in event for event in events)
    assert any("complete" in event.lower() for event in events)


def test_build_ebm_feature_diagnostic_writes_invalid_dependency_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "ridge"
    child_runs = [
        _write_source_child(
            tmp_path,
            child_id=f"child-{season}",
            season=season,
            model_id=model_id,
            feature_pack="ppg_xg",
            fixture_mode="none",
        )
        for season in (2021, 2022, 2023)
    ]
    for child, season in zip(child_runs, (2021, 2022, 2023), strict=True):
        _write_synthetic_player_predictions(child, season=season, model_id=model_id)
    experiment_path = tmp_path / "experiment"
    output_path = tmp_path / "ebm-output"
    _write_parent(experiment_path, child_runs)
    events: list[str] = []
    monkeypatch.setattr(ebm_feature_diagnostic, "_load_default_ebm_class", lambda: (None, None))

    result = build_ebm_feature_diagnostic(
        experiment_path=experiment_path,
        output_path=output_path,
        seasons=(2021, 2022, 2023),
        model_id=model_id,
        feature_pack="ppg_xg",
        fixture_mode="none",
        current_year=2026,
        max_interactions=10,
        min_validation_rows=50,
        random_seed=123,
        profile_runtime=False,
        progress_callback=events.append,
    )

    manifest = json.loads((output_path / "ebm_diagnostic_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output_path / "ebm_diagnostic_decision.json").read_text(encoding="utf-8"))
    invalid_report = pd.read_csv(output_path / "invalid_diagnostic_report.csv")
    assert result.decision["diagnostic_status"] == "invalid"
    assert decision["diagnostic_status"] == "invalid"
    assert manifest["diagnostic_status"] == "invalid"
    assert manifest["diagnostic_phase"] in {"dependency_unavailable", "full_pipeline"}
    assert invalid_report["reason_type"].tolist() == ["dependency"]
    assert any("dependency" in event for event in events)
    assert any("artifact write" in event for event in events)


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
