from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

H004_CONTROL_MODEL_ID = "xgboost_depth2_slow"
H004_CONTROL_FEATURE_PACK = "ppg_xg_matchup"
H004_PRIMARY_SCORE_COLUMN = "xgboost_depth2_slow_score"
H004_REQUIRED_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
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
    frame["entered_field"] = frame["entrou_em_campo"].fillna(False).astype(bool)

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

    return H004PredictionBundle(
        child=child,
        all_candidates=frame,
        played=played,
        dnp=dnp,
        selected_players=selected_frame,
    )


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
