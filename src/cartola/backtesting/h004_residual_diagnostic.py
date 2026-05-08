from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd

H004_CONTROL_MODEL_ID = "xgboost_depth2_slow"
H004_CONTROL_FEATURE_PACK = "ppg_xg_matchup"
H004_PRIMARY_SCORE_COLUMN = "xgboost_depth2_slow_score"
H004_REQUIRED_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
H004_MIN_CORRELATION_ROWS = 100
H004_MIN_ABS_SPEARMAN = 0.05
H004_MIN_QUINTILE_SPREAD = 0.25
H004_SIGNAL_COLUMNS: tuple[str, ...] = (
    "footystats_xg_diff",
    "matchup_opponent_allowed_position_points_roll5",
    "diagnostic_home_xg_edge",
)
H004_REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "posicao",
    "id_clube",
    "pontuacao",
    "entrou_em_campo",
    "matchup_is_home",
    "footystats_xg_diff",
    "footystats_ppg_diff",
    "matchup_opponent_allowed_points_roll5",
    "matchup_opponent_allowed_position_points_roll5",
    "matchup_club_position_points_roll5",
    "matchup_opponent_allowed_position_count",
    "matchup_club_position_count",
    "position_points_prior",
)
H004_REQUIRED_SELECTED_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "posicao",
    "pontuacao",
    "entrou_em_campo",
)
H004_TRUE_VALUES = frozenset(("1", "1.0", "true", "t", "yes", "y"))
H004_FALSE_VALUES = frozenset(("0", "0.0", "false", "f", "no", "n"))


@dataclass(frozen=True)
class H004SourceChild:
    season: int
    model_id: str
    feature_pack: str
    child_path: Path
    score_column: str
    fixture_mode: str
    matchup_context_mode: str
    footystats_mode: str
    fixture_identity_status: str
    footystats_source_identity: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["child_path"] = str(self.child_path)
        return payload


@dataclass(frozen=True)
class H004PredictionBundle:
    child: H004SourceChild
    all_candidates: pd.DataFrame
    played: pd.DataFrame
    dnp: pd.DataFrame
    selected_players: pd.DataFrame


def discover_h004_source_children(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
) -> tuple[H004SourceChild, ...]:
    children: list[H004SourceChild] = []
    for season in seasons:
        child_path = (
            experiment_path
            / "runs"
            / f"season={season}"
            / f"model={model_id}"
            / f"feature_pack={feature_pack}"
        )
        if not child_path.is_dir():
            raise FileNotFoundError(
                f"Missing H004 source child for season={season} model={model_id} "
                f"feature_pack={feature_pack}: {child_path}"
            )
        metadata = _read_json(child_path / "run_metadata.json")
        children.append(
            H004SourceChild(
                season=_metadata_int(metadata, "season", fallback=season),
                model_id=_metadata_str(metadata, "model_id", fallback=model_id),
                feature_pack=_metadata_str(metadata, "feature_pack", fallback=feature_pack),
                child_path=child_path,
                score_column=f"{model_id}_score",
                fixture_mode=_metadata_str(metadata, "fixture_mode"),
                matchup_context_mode=_metadata_str(metadata, "matchup_context_mode"),
                footystats_mode=_metadata_str(metadata, "footystats_mode"),
                fixture_identity_status=_metadata_str(metadata, "fixture_identity_status", fallback="unverified"),
                footystats_source_identity=_footystats_source_identity(metadata),
            )
        )
    return tuple(children)


def load_h004_prediction_bundle(child: H004SourceChild) -> H004PredictionBundle:
    predictions_path = child.child_path / "player_predictions.csv"
    selected_path = child.child_path / "selected_players.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)

    predictions = pd.read_csv(predictions_path)
    selected_players = pd.read_csv(selected_path)
    required_prediction_columns = (*H004_REQUIRED_PREDICTION_COLUMNS, child.score_column)
    _validate_columns("player_predictions.csv", predictions, required_prediction_columns)
    _validate_columns("selected_players.csv", selected_players, H004_REQUIRED_SELECTED_COLUMNS)

    frame = predictions.copy()
    frame["season"] = child.season
    frame["model_id"] = child.model_id
    frame["feature_pack"] = child.feature_pack
    frame["predicted_points"] = pd.to_numeric(frame[child.score_column], errors="coerce")
    frame["actual_points"] = pd.to_numeric(frame["pontuacao"], errors="coerce")
    frame["entered_field"] = _parse_bool_like_series(
        "player_predictions.csv",
        frame["entrou_em_campo"],
    )

    numeric_columns = (
        "rodada",
        "id_atleta",
        "id_clube",
        "predicted_points",
        "actual_points",
        "matchup_is_home",
        "footystats_xg_diff",
        "footystats_ppg_diff",
        "matchup_opponent_allowed_points_roll5",
        "matchup_opponent_allowed_position_points_roll5",
        "matchup_club_position_points_roll5",
        "matchup_opponent_allowed_position_count",
        "matchup_club_position_count",
        "position_points_prior",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_prediction = frame["predicted_points"].isna() | frame["predicted_points"].isin(
        [float("inf"), float("-inf")]
    )
    if bool(invalid_prediction.any()):
        raise ValueError(f"Non-finite H004 predicted points in {predictions_path}")

    played = frame.loc[frame["entered_field"] & frame["actual_points"].notna()].copy()
    played["prediction_residual"] = played["actual_points"] - played["predicted_points"]
    dnp = frame.loc[~frame["entered_field"]].copy()

    selected_frame = selected_players.copy()
    selected_frame["season"] = child.season
    selected_frame["model_id"] = child.model_id
    selected_frame["feature_pack"] = child.feature_pack
    selected_frame["rodada"] = pd.to_numeric(selected_frame["rodada"], errors="coerce")
    selected_frame["id_atleta"] = pd.to_numeric(selected_frame["id_atleta"], errors="coerce")
    selected_frame["entrou_em_campo"] = _parse_bool_like_series(
        "selected_players.csv",
        selected_frame["entrou_em_campo"],
    )

    return H004PredictionBundle(
        child=child,
        all_candidates=frame,
        played=played,
        dnp=dnp,
        selected_players=selected_frame,
    )


def build_h004_residual_correlations(played: pd.DataFrame) -> pd.DataFrame:
    frame = _with_diagnostic_signal_columns(played)
    rows: list[dict[str, object]] = []
    for season, position, column in _season_position_column_keys(frame, H004_SIGNAL_COLUMNS):
        group = _valid_metric_group(frame, season=season, position=position, column=column)
        row_count = int(len(group))
        spearman = float("nan")
        spread = float("nan")
        passes_signal = False
        if row_count >= H004_MIN_CORRELATION_ROWS and group[column].nunique(dropna=True) > 1:
            spearman = float(group["prediction_residual"].corr(group[column], method="spearman"))
            spread = _quintile_spread(group, column)
            passes_signal = bool(
                pd.notna(spearman)
                and spearman >= H004_MIN_ABS_SPEARMAN
                and pd.notna(spread)
                and spread >= H004_MIN_QUINTILE_SPREAD
            )
        rows.append(
            {
                "season": int(season),
                "position": str(position),
                "signal_family": _signal_family(position=str(position), column=column),
                "context_column": column,
                "row_count": row_count,
                "spearman": spearman,
                "quintile_residual_spread": spread,
                "passes_signal": passes_signal,
            }
        )
    return (
        pd.DataFrame(rows, columns=_residual_correlation_columns())
        .sort_values(["season", "position", "context_column"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_h004_residual_quintiles(played: pd.DataFrame) -> pd.DataFrame:
    frame = _with_diagnostic_signal_columns(played)
    rows: list[dict[str, object]] = []
    for season, position, column in _season_position_column_keys(frame, H004_SIGNAL_COLUMNS):
        group = _valid_metric_group(frame, season=season, position=position, column=column)
        if group.empty:
            continue
        ranked = group.sort_values([column, "prediction_residual"], kind="mergesort").copy()
        ranked["quintile"] = pd.qcut(
            ranked[column].rank(method="first"),
            q=min(5, len(ranked)),
            labels=False,
        ) + 1
        for quintile, quintile_group in ranked.groupby("quintile", sort=True):
            quintile_value = int(cast("int", quintile))
            rows.append(
                {
                    "season": int(season),
                    "position": str(position),
                    "context_column": column,
                    "quintile": quintile_value,
                    "row_count": int(len(quintile_group)),
                    "context_min": float(quintile_group[column].min()),
                    "context_max": float(quintile_group[column].max()),
                    "mean_residual": float(quintile_group["prediction_residual"].mean()),
                    "median_residual": float(quintile_group["prediction_residual"].median()),
                }
            )
    return (
        pd.DataFrame(rows, columns=_residual_quintile_columns())
        .sort_values(["season", "position", "context_column", "quintile"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_h004_top_actual_recall(played: pd.DataFrame) -> pd.DataFrame:
    frame = played.copy()
    if frame.empty:
        return pd.DataFrame(columns=_top_actual_recall_columns())

    frame["context_edge"] = _context_edge(frame)
    rows: list[pd.DataFrame] = []
    for _, group in frame.groupby(["season", "rodada", "posicao"], sort=True):
        played_count = len(group)
        ranked_prediction = group.copy()
        ranked_prediction["predicted_rank"] = ranked_prediction["predicted_points"].rank(
            method="average",
            ascending=False,
        )
        ranked_prediction["predicted_rank_percentile"] = (
            (ranked_prediction["predicted_rank"] - 1.0) / max(float(played_count - 1), 1.0)
        )
        actual_top = ranked_prediction.sort_values(
            ["actual_points", "id_atleta"],
            ascending=[False, True],
            kind="mergesort",
        ).head(min(5, played_count))
        rows.append(actual_top)

    if not rows:
        return pd.DataFrame(columns=_top_actual_recall_columns())

    actual_top_frame = pd.concat(rows, ignore_index=True)
    summary = (
        actual_top_frame.groupby(["season", "posicao"], as_index=False)
        .agg(
            row_count=("id_atleta", "count"),
            median_predicted_rank_percentile=("predicted_rank_percentile", "median"),
            median_context_edge=("context_edge", "median"),
        )
        .rename(columns={"posicao": "position"})
    )
    summary["passes_signal"] = (
        summary["median_predicted_rank_percentile"].ge(0.35)
        & summary["median_context_edge"].ge(0.25)
    )
    return (
        summary.loc[:, _top_actual_recall_columns()]
        .sort_values(["season", "position"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_h004_diagnostic_decision(
    *,
    correlations: pd.DataFrame,
    top_actual_recall: pd.DataFrame,
    source_experiment_path: Path,
    children: tuple[H004SourceChild, ...],
    missing_or_invalid_columns: tuple[str, ...],
) -> dict[str, object]:
    family_results = {
        "A": _family_result(correlations, family="A"),
        "B": _family_result(correlations, family="B"),
        "C": _family_result(top_actual_recall, family=None),
    }
    passed_families = [
        family for family in ("A", "B", "C") if bool(family_results[family]["passed"])
    ]
    diagnostic_status = (
        "invalid"
        if missing_or_invalid_columns
        else ("passes" if passed_families else "rejected")
    )
    score_column_mapping = {
        child.model_id: child.score_column for child in sorted(children, key=lambda child: child.model_id)
    } or {H004_CONTROL_MODEL_ID: H004_PRIMARY_SCORE_COLUMN}

    return {
        "diagnostic_status": diagnostic_status,
        "passed_families": passed_families,
        "family_results": family_results,
        "source_experiment_path": str(source_experiment_path),
        "source_children": [child.as_dict() for child in sorted(children, key=lambda child: child.season)],
        "score_column_mapping": score_column_mapping,
        "fixture_identity_status": _fixture_identity_status(children),
        "footystats_source_identity": {
            str(child.season): child.footystats_source_identity
            for child in sorted(children, key=lambda child: child.season)
        },
        "missing_or_invalid_columns": sorted(missing_or_invalid_columns),
    }


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _metadata_str(metadata: dict[str, object], key: str, *, fallback: str | None = None) -> str:
    value = metadata.get(key, fallback)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing H004 source metadata field: {key}")
    return str(value)


def _metadata_int(metadata: dict[str, object], key: str, *, fallback: int) -> int:
    value = metadata.get(key, fallback)
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid H004 source metadata field {key}: {value!r}") from exc


def _footystats_source_identity(metadata: dict[str, object]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in metadata.items()
        if str(key).startswith("footystats_") and ("sha" in str(key) or "source" in str(key))
    }


def _validate_columns(frame_name: str, frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {frame_name}: {', '.join(missing)}")


def _parse_bool_like_series(frame_name: str, values: pd.Series) -> pd.Series:
    column_name = str(values.name)
    return values.map(lambda value: _parse_bool_like(value, frame_name=frame_name, column_name=column_name))


def _parse_bool_like(value: object, *, frame_name: str, column_name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip().lower()
    if normalized == "":
        return False
    if normalized in H004_TRUE_VALUES:
        return True
    if normalized in H004_FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognized boolean value in {frame_name}.{column_name}: {value!r}")


def _with_diagnostic_signal_columns(played: pd.DataFrame) -> pd.DataFrame:
    frame = played.copy()
    frame["diagnostic_home_xg_edge"] = (
        pd.to_numeric(frame["matchup_is_home"], errors="coerce")
        * pd.to_numeric(frame["footystats_xg_diff"], errors="coerce")
    )
    return frame


def _season_position_column_keys(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> list[tuple[int, str, str]]:
    seasons = sorted(int(value) for value in frame["season"].dropna().unique())
    positions = sorted(str(value) for value in frame["posicao"].dropna().unique())
    return [(season, position, column) for season in seasons for position in positions for column in columns]


def _valid_metric_group(
    frame: pd.DataFrame,
    *,
    season: int,
    position: str,
    column: str,
) -> pd.DataFrame:
    group = frame.loc[frame["season"].eq(season) & frame["posicao"].eq(position)].copy()
    values = pd.to_numeric(group[column], errors="coerce")
    residuals = pd.to_numeric(group["prediction_residual"], errors="coerce")
    valid = values.notna() & residuals.notna()
    group = group.loc[valid].copy()
    group[column] = values.loc[valid]
    group["prediction_residual"] = residuals.loc[valid]
    return group


def _quintile_spread(group: pd.DataFrame, column: str) -> float:
    if len(group) < 5 or group[column].nunique(dropna=True) < 2:
        return float("nan")
    ranked = group.sort_values([column, "prediction_residual"], kind="mergesort").copy()
    ranked["quintile"] = pd.qcut(ranked[column].rank(method="first"), q=5, labels=False) + 1
    means = ranked.groupby("quintile")["prediction_residual"].mean()
    return float(means.loc[5] - means.loc[1])


def _context_edge(frame: pd.DataFrame) -> pd.Series:
    xg = _zscore_by_season_position(frame, "footystats_xg_diff")
    position_allowed = _zscore_by_season_position(
        frame,
        "matchup_opponent_allowed_position_points_roll5",
    )
    return xg + position_allowed


def _zscore_by_season_position(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.assign(_value=values).groupby(["season", "posicao"])["_value"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    valid_std = std.notna() & std.ne(0.0)
    result = pd.Series(0.0, index=frame.index, dtype="float64")
    result.loc[valid_std] = (values.loc[valid_std] - mean.loc[valid_std]) / std.loc[valid_std]
    return result.fillna(0.0)


def _family_result(correlations: pd.DataFrame, *, family: str | None) -> dict[str, object]:
    if correlations.empty or "passes_signal" not in correlations.columns or "season" not in correlations.columns:
        return {"passed": False, "passed_seasons": []}
    subset = correlations
    if family is not None:
        if "signal_family" not in correlations.columns:
            return {"passed": False, "passed_seasons": []}
        subset = subset.loc[subset["signal_family"].eq(family)]
    subset = subset.loc[subset["passes_signal"].eq(True)]
    seasons = sorted(int(value) for value in subset["season"].dropna().unique())
    return {"passed": len(seasons) >= 3, "passed_seasons": seasons}


def _fixture_identity_status(children: tuple[H004SourceChild, ...]) -> str:
    if not children:
        return "unavailable"
    statuses = {child.fixture_identity_status for child in children}
    return "verified" if statuses == {"verified"} else "unverified"


def _signal_family(*, position: str, column: str) -> str:
    if position in {"ata", "mei"} and column in {
        "footystats_xg_diff",
        "matchup_opponent_allowed_position_points_roll5",
    }:
        return "A"
    if position in {"gol", "lat", "zag"} and column == "diagnostic_home_xg_edge":
        return "B"
    return "descriptive"


def _residual_correlation_columns() -> pd.Index:
    return pd.Index(
        [
            "season",
            "position",
            "signal_family",
            "context_column",
            "row_count",
            "spearman",
            "quintile_residual_spread",
            "passes_signal",
        ]
    )


def _residual_quintile_columns() -> pd.Index:
    return pd.Index(
        [
            "season",
            "position",
            "context_column",
            "quintile",
            "row_count",
            "context_min",
            "context_max",
            "mean_residual",
            "median_residual",
        ]
    )


def _top_actual_recall_columns() -> pd.Index:
    return pd.Index(
        [
            "season",
            "position",
            "row_count",
            "median_predicted_rank_percentile",
            "median_context_edge",
            "passes_signal",
        ]
    )
