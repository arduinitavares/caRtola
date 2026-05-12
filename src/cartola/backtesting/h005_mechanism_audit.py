from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cartola.backtesting.data import load_fixtures, load_season_data
from cartola.backtesting.features import build_prediction_frame

H005_HYPOTHESIS_ID = "H005"
H005_DESIGN_REVISION = "reliability_v1"
H005_SCORE_COLUMN_TEMPLATE = "{model_id}_score"
H005_REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "apelido",
    "id_clube",
    "posicao",
    "status",
    "preco_pre_rodada",
    "pontuacao",
    "entrou_em_campo",
    "matchup_opponent_allowed_position_count",
    "matchup_opponent_allowed_position_points_roll5",
    "matchup_opponent_allowed_points_roll5",
)
H005_RECOMPUTED_COLUMNS: tuple[str, ...] = (
    "rodada",
    "id_atleta",
    "matchup_opponent_allowed_position_count",
    "h005_opponent_position_available_match_count_roll5",
    "h005_opponent_position_expected_count_roll5",
    "h005_opponent_position_count_ratio",
)
H005_RATIO_BINS: tuple[float, ...] = (-np.inf, 0.0, 0.5, 0.8, 1.0, 1.5, np.inf)
H005_RATIO_LABELS: tuple[str, ...] = ("0", "(0, 0.5]", "(0.5, 0.8]", "(0.8, 1.0]", "(1.0, 1.5]", "> 1.5")
H005_RAW_COUNT_BINS: tuple[float, ...] = (-np.inf, 0.0, 5.0, 10.0, 20.0, 30.0, np.inf)
H005_RAW_COUNT_LABELS: tuple[str, ...] = ("0", "(0, 5]", "(5, 10]", "(10, 20]", "(20, 30]", "> 30")
H005_NON_COACH_POSITIONS: frozenset[str] = frozenset(("gol", "lat", "zag", "mei", "ata"))


class H005MechanismAuditError(ValueError):
    """Raised when H005 source artifacts are inconsistent with the requested audit."""


@dataclass(frozen=True)
class H005SourceChild:
    season: int
    model_id: str
    feature_pack: str
    child_path: Path
    score_column: str
    fixture_mode: str
    matchup_context_mode: str
    footystats_mode: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["child_path"] = str(self.child_path)
        return payload


@dataclass(frozen=True)
class H005MechanismAuditResult:
    output_path: Path
    mechanism_audit: pd.DataFrame
    raw_count_audit: pd.DataFrame
    decision: dict[str, object]


def discover_h005_source_children(
    *,
    experiment_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
) -> tuple[H005SourceChild, ...]:
    children: list[H005SourceChild] = []
    for season in seasons:
        child_path = _canonical_child_path(experiment_path, season, model_id, feature_pack)
        if not child_path.is_dir():
            raise FileNotFoundError(
                f"Missing H005 source child for season={season} model={model_id} "
                f"feature_pack={feature_pack}: {child_path}"
            )
        metadata = _read_json(child_path / "run_metadata.json")
        metadata_season = _metadata_int(metadata, "season", fallback=season)
        metadata_model_id = _metadata_str(metadata, "model_id", fallback=model_id)
        metadata_feature_pack = _metadata_str(metadata, "feature_pack", fallback=feature_pack)
        conflicts = []
        if metadata_season != season:
            conflicts.append(f"season={metadata_season!r}")
        if metadata_model_id != model_id:
            conflicts.append(f"model_id={metadata_model_id!r}")
        if metadata_feature_pack != feature_pack:
            conflicts.append(f"feature_pack={metadata_feature_pack!r}")
        if conflicts:
            raise H005MechanismAuditError(
                f"H005 source metadata mismatch for {child_path}: {', '.join(conflicts)}"
            )
        children.append(
            H005SourceChild(
                season=season,
                model_id=model_id,
                feature_pack=feature_pack,
                child_path=child_path,
                score_column=H005_SCORE_COLUMN_TEMPLATE.format(model_id=model_id),
                fixture_mode=_metadata_str(metadata, "fixture_mode"),
                matchup_context_mode=_metadata_str(metadata, "matchup_context_mode"),
                footystats_mode=_metadata_str(metadata, "footystats_mode"),
                metadata=metadata,
            )
        )
    return tuple(children)


def write_h005_mechanism_audit_artifacts(result: H005MechanismAuditResult) -> None:
    result.output_path.mkdir(parents=True, exist_ok=True)
    result.mechanism_audit.to_csv(result.output_path / "h005_mechanism_audit.csv", index=False)
    result.raw_count_audit.to_csv(result.output_path / "h005_raw_count_audit.csv", index=False)
    (result.output_path / "h005_mechanism_audit_decision.json").write_text(
        json.dumps(result.decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_h005_mechanism_audit(
    *,
    experiment_path: Path,
    output_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
    project_root: Path,
) -> H005MechanismAuditResult:
    children = discover_h005_source_children(
        experiment_path=experiment_path,
        seasons=seasons,
        model_id=model_id,
        feature_pack=feature_pack,
    )
    audit_frames: list[pd.DataFrame] = []
    failed_checks: set[str] = set()
    for child in children:
        source = _load_source_predictions(child)
        recomputed = _recompute_h005_features(child, source, project_root)
        merged, child_failed_checks = _merge_source_with_recomputed(source, recomputed)
        failed_checks.update(child_failed_checks)
        audit_frames.append(merged)

    audit_frame = pd.concat(audit_frames, ignore_index=True) if audit_frames else _empty_audit_frame()
    if audit_frame.empty:
        failed_checks.add("empty_audit_frame")
    mechanism_audit = _build_ratio_audit(audit_frame)
    raw_count_audit = _build_raw_count_audit(audit_frame)
    decision = _build_audit_decision(
        failed_checks=failed_checks,
        mechanism_audit=mechanism_audit,
        raw_count_audit=raw_count_audit,
        children=children,
        experiment_path=experiment_path,
    )
    result = H005MechanismAuditResult(
        output_path=output_path,
        mechanism_audit=mechanism_audit,
        raw_count_audit=raw_count_audit,
        decision=decision,
    )
    write_h005_mechanism_audit_artifacts(result)
    return result


def _canonical_child_path(experiment_path: Path, season: int, model_id: str, feature_pack: str) -> Path:
    return experiment_path / "runs" / f"season={season}" / f"model={model_id}" / f"feature_pack={feature_pack}"


def _load_source_predictions(child: H005SourceChild) -> pd.DataFrame:
    predictions_path = child.child_path / "player_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_csv(predictions_path)
    _validate_columns("player_predictions.csv", predictions, (*H005_REQUIRED_PREDICTION_COLUMNS, child.score_column))

    frame = predictions.copy()
    frame["season"] = child.season
    frame["model_id"] = child.model_id
    frame["feature_pack"] = child.feature_pack
    frame["predicted_points"] = pd.to_numeric(frame[child.score_column], errors="coerce")
    frame["actual_points"] = pd.to_numeric(frame["pontuacao"], errors="coerce")
    frame["entered_field"] = _parse_bool_like_series(frame["entrou_em_campo"])
    frame["source_residual"] = frame["actual_points"] - frame["predicted_points"]
    numeric_columns = (
        "season",
        "rodada",
        "id_atleta",
        "id_clube",
        "predicted_points",
        "actual_points",
        "matchup_opponent_allowed_position_count",
        "matchup_opponent_allowed_position_points_roll5",
        "matchup_opponent_allowed_points_roll5",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _recompute_h005_features(child: H005SourceChild, source: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    season_df = load_season_data(child.season, project_root)
    fixtures = load_fixtures(child.season, project_root)
    frames: list[pd.DataFrame] = []
    for round_number in sorted(source["rodada"].dropna().astype(int).unique()):
        round_frame = build_prediction_frame(
            season_df,
            int(round_number),
            fixtures=fixtures,
            matchup_context_mode="cartola_matchup_v1",
            feature_augmentation_mode="h005_matchup_reliability_v1",
        )
        _validate_columns("recomputed H005 prediction frame", round_frame, H005_RECOMPUTED_COLUMNS)
        selected = round_frame.loc[:, H005_RECOMPUTED_COLUMNS].copy()
        selected["season"] = child.season
        frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=pd.Index([*H005_RECOMPUTED_COLUMNS, "season"]))
    return pd.concat(frames, ignore_index=True)


def _merge_source_with_recomputed(source: pd.DataFrame, recomputed: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    source_keys = source[["season", "rodada", "id_atleta"]].copy()
    recomputed_keys = recomputed[["season", "rodada", "id_atleta"]].copy()
    source_key_index = pd.MultiIndex.from_frame(source_keys)
    recomputed_key_index = pd.MultiIndex.from_frame(recomputed_keys)
    failed_checks: set[str] = set()
    if not source_key_index.equals(recomputed_key_index) or set(source_key_index) != set(recomputed_key_index):
        failed_checks.add("row_identity_mismatch")

    merged = source.merge(
        recomputed,
        on=["season", "rodada", "id_atleta"],
        how="left",
        suffixes=("_source", "_recomputed"),
        validate="one_to_one",
        indicator=True,
    )
    if bool(merged["_merge"].ne("both").any()):
        failed_checks.add("row_identity_mismatch")

    valid_rows = merged["entered_field"] & merged["actual_points"].notna() & merged["predicted_points"].notna()
    count_source = pd.to_numeric(
        merged["matchup_opponent_allowed_position_count_source"],
        errors="coerce",
    )
    count_recomputed = pd.to_numeric(
        merged["matchup_opponent_allowed_position_count_recomputed"],
        errors="coerce",
    )
    count_mismatch = valid_rows & (count_source != count_recomputed)
    if bool(count_mismatch.fillna(True).any()):
        failed_checks.add("recomputed_count_mismatch")

    merged["source_base_count"] = count_source
    merged["h005_available_match_count"] = pd.to_numeric(
        merged["h005_opponent_position_available_match_count_roll5"],
        errors="coerce",
    )
    merged["h005_expected_count"] = pd.to_numeric(
        merged["h005_opponent_position_expected_count_roll5"],
        errors="coerce",
    )
    merged["h005_count_ratio"] = pd.to_numeric(
        merged["h005_opponent_position_count_ratio"],
        errors="coerce",
    )
    merged["source_position_points"] = pd.to_numeric(
        merged["matchup_opponent_allowed_position_points_roll5"],
        errors="coerce",
    )
    merged["source_all_points"] = pd.to_numeric(
        merged["matchup_opponent_allowed_points_roll5"],
        errors="coerce",
    )
    merged["position_allowed_delta"] = merged["source_position_points"] - merged["source_all_points"]
    merged["source_overprediction"] = merged["predicted_points"] > merged["actual_points"]
    return merged.drop(columns=["_merge"], errors="ignore"), failed_checks


def _build_ratio_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_summary_frame("ratio_bin")
    audit_frame = frame.copy()
    audit_frame["ratio_bin"] = _closed_right_bins(
        audit_frame["h005_count_ratio"],
        bins=H005_RATIO_BINS,
        labels=H005_RATIO_LABELS,
    )
    return _audit_group(audit_frame, "ratio_bin")


def _build_raw_count_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_summary_frame("raw_count_bin")
    audit_frame = frame.copy()
    audit_frame["raw_count_bin"] = _closed_right_bins(
        audit_frame["source_base_count"],
        bins=H005_RAW_COUNT_BINS,
        labels=H005_RAW_COUNT_LABELS,
    )
    return _audit_group(audit_frame, "raw_count_bin")


def _audit_group(frame: pd.DataFrame, bin_column: str) -> pd.DataFrame:
    played = frame.loc[frame["entered_field"] & frame["actual_points"].notna() & frame["predicted_points"].notna()].copy()
    if played.empty:
        return _empty_summary_frame(bin_column)
    grouped = played.groupby(["season", "posicao", bin_column], observed=False).agg(
        row_count=("id_atleta", "size"),
        round_count=("rodada", "nunique"),
        source_residual_mean=("source_residual", "mean"),
        source_overprediction_rate=("source_overprediction", "mean"),
        source_base_count_mean=("source_base_count", "mean"),
        h005_available_match_count_mean=("h005_available_match_count", "mean"),
        h005_expected_count_mean=("h005_expected_count", "mean"),
        h005_count_ratio_mean=("h005_count_ratio", "mean"),
        source_position_points_mean=("source_position_points", "mean"),
        source_all_points_mean=("source_all_points", "mean"),
        position_allowed_delta_mean=("position_allowed_delta", "mean"),
    ).reset_index()
    return grouped.sort_values(["season", "posicao", bin_column], kind="mergesort").reset_index(drop=True)


def _build_audit_decision(
    *,
    failed_checks: set[str],
    mechanism_audit: pd.DataFrame,
    raw_count_audit: pd.DataFrame,
    children: tuple[H005SourceChild, ...],
    experiment_path: Path,
) -> dict[str, object]:
    audit_status = _support_gate(
        failed_checks=failed_checks,
        mechanism_audit=mechanism_audit,
        raw_count_audit=raw_count_audit,
    )
    return {
        "hypothesis_id": H005_HYPOTHESIS_ID,
        "audit_status": audit_status,
        "failed_checks": sorted(failed_checks),
        "manual_points_shrinkage": False,
        "h005_design_revision": H005_DESIGN_REVISION,
        "experiment_path": str(experiment_path),
        "source_children": [child.as_dict() for child in children],
    }


def _support_gate(
    *,
    failed_checks: set[str],
    mechanism_audit: pd.DataFrame,
    raw_count_audit: pd.DataFrame,
) -> str:
    if failed_checks:
        return "invalid"
    if raw_count_audit.empty or mechanism_audit.empty:
        return "mixed_or_weak"
    low_ratio = mechanism_audit[
        mechanism_audit["ratio_bin"].astype(str).isin(("0", "(0, 0.5]"))
        & mechanism_audit["posicao"].astype(str).isin(H005_NON_COACH_POSITIONS)
        & (pd.to_numeric(mechanism_audit["row_count"], errors="coerce") >= 100)
    ]
    if low_ratio["posicao"].nunique() >= 4:
        return "supports_reliability_hypothesis"
    return "mixed_or_weak"


def _closed_right_bins(
    values: pd.Series,
    *,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    clipped = numeric.mask(numeric < 0.0, 0.0)
    return pd.cut(clipped, bins=bins, labels=labels, right=True)


def _empty_summary_frame(bin_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=pd.Index(
            [
                "season",
                "posicao",
                bin_column,
                "row_count",
                "round_count",
                "source_residual_mean",
                "source_overprediction_rate",
                "source_base_count_mean",
                "h005_available_match_count_mean",
                "h005_expected_count_mean",
                "h005_count_ratio_mean",
                "source_position_points_mean",
                "source_all_points_mean",
                "position_allowed_delta_mean",
            ]
        )
    )


def _empty_audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=pd.Index(
            [
                "season",
                "rodada",
                "id_atleta",
                "posicao",
                "entered_field",
                "actual_points",
                "predicted_points",
                "source_residual",
                "source_base_count",
                "h005_available_match_count",
                "h005_expected_count",
                "h005_count_ratio",
                "source_position_points",
                "source_all_points",
                "position_allowed_delta",
                "source_overprediction",
            ]
        )
    )


def _validate_columns(source_name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise H005MechanismAuditError(f"{source_name} missing required columns: {', '.join(missing_columns)}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise H005MechanismAuditError(f"Expected JSON object in {path}")
    return payload


def _metadata_str(metadata: dict[str, Any], key: str, fallback: str = "") -> str:
    value = metadata.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def _metadata_int(metadata: dict[str, Any], key: str, fallback: int) -> int:
    value = metadata.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise H005MechanismAuditError(f"Invalid integer metadata field {key!r}: {value!r}") from exc


def _parse_bool_like_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    true_values = frozenset(("1", "1.0", "true", "t", "yes", "y"))
    false_values = frozenset(("0", "0.0", "false", "f", "no", "n", "", "nan", "none"))
    invalid = ~normalized.isin(true_values | false_values)
    if bool(invalid.any()):
        bad_values = sorted(normalized.loc[invalid].unique().tolist())
        raise H005MechanismAuditError(f"Invalid boolean values in entrou_em_campo: {bad_values}")
    return normalized.isin(true_values)
