from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import pandas as pd

IDENTITY_COLUMNS: list[str] = [
    "source_mode",
    "source_experiment_id",
    "source_child_id",
    "season",
    "rodada",
    "strategy",
    "model_id",
    "feature_pack",
    "fixture_mode",
    "matchup_context_mode",
    "budget_policy",
    "oracle_type",
    "candidate_universe",
    "budget_path",
]

_NON_PLAYER_POSITION = "tec"
_OPPONENT_OVERLAP_COLUMN = "opponent_overlap_in_lineup"


def build_oracle_player_profile_rows(
    *,
    identity: Mapping[str, object],
    oracle_selected: pd.DataFrame,
    model_selected: pd.DataFrame,
    fixtures: pd.DataFrame | None,
) -> list[dict[str, object]]:
    selected = _with_lineup_context(oracle_selected, fixtures=fixtures)
    rows: list[dict[str, object]] = []
    for _, player in selected.iterrows():
        for metric, value in _player_metric_values(player).items():
            rows.append(
                {
                    **{column: identity.get(column) for column in IDENTITY_COLUMNS},
                    "id_atleta": player.get("id_atleta"),
                    "posicao": player.get("posicao"),
                    "profile_section": "oracle_player",
                    "profile_metric": metric,
                    "profile_value": value,
                    "baseline_name": "model_selected",
                    "baseline_value": None,
                    "sample_size": 1,
                    "full_market_status": "not_available",
                },
            )
    return rows


def build_profile_gap_summary_rows(
    *,
    identity: Mapping[str, object],
    oracle_selected: pd.DataFrame,
    model_selected: pd.DataFrame,
    fixtures: pd.DataFrame | None,
) -> list[dict[str, object]]:
    oracle = _with_lineup_context(oracle_selected, fixtures=fixtures)
    model = _with_lineup_context(model_selected, fixtures=fixtures)
    metrics = {
        "opponent_overlap_round_rate": (
            _opponent_overlap_round_rate(oracle),
            _opponent_overlap_round_rate(model),
        ),
        "avg_players_in_opponent_overlap": (
            _avg_players_in_opponent_overlap(oracle),
            _avg_players_in_opponent_overlap(model),
        ),
        "home_player_share": (
            _share_true(oracle, "matchup_is_home", exclude_tecnico=True),
            _share_true(model, "matchup_is_home", exclude_tecnico=True),
        ),
        "favorite_proxy_ppg_diff_positive_share": (
            _share_positive(oracle, "footystats_ppg_diff", exclude_tecnico=True),
            _share_positive(model, "footystats_ppg_diff", exclude_tecnico=True),
        ),
        "median_model_predicted_rank_position": (
            _median_numeric(oracle, "model_predicted_rank_position", exclude_tecnico=True),
            _median_numeric(model, "model_predicted_rank_position", exclude_tecnico=True),
        ),
        "top5_position_rank_share": (
            _share_at_most(oracle, "model_predicted_rank_position", 5.0, exclude_tecnico=True),
            _share_at_most(model, "model_predicted_rank_position", 5.0, exclude_tecnico=True),
        ),
        "avg_same_club_selected_count": (
            _avg_same_club_count(oracle),
            _avg_same_club_count(model),
        ),
    }
    rows: list[dict[str, object]] = []
    for metric, (oracle_value, baseline_value) in metrics.items():
        rows.append(
            {
                **{column: identity.get(column) for column in IDENTITY_COLUMNS},
                "profile_section": "lineup_profile",
                "profile_metric": metric,
                "oracle_value": oracle_value,
                "baseline_name": "model_selected",
                "baseline_value": baseline_value,
                "absolute_gap": _numeric_gap(oracle_value, baseline_value),
                "relative_gap": None,
                "sample_size": int(len(oracle)),
                "season_stability_count": None,
                "stability_label": "round_level",
                "full_market_status": "not_available",
            },
        )
    return rows


def _player_metric_values(player: pd.Series) -> dict[str, object]:
    return {
        "is_home": _bool_or_none(player.get("matchup_is_home")),
        "opponent_overlap_in_lineup": _bool_or_none(player.get(_OPPONENT_OVERLAP_COLUMN)),
        "same_club_selected_count": _numeric_or_none(player.get("same_club_selected_count")),
        "model_predicted_rank_overall": _numeric_or_none(player.get("model_predicted_rank_overall")),
        "model_predicted_rank_position": _numeric_or_none(player.get("model_predicted_rank_position")),
        "preco_pre_rodada": _numeric_or_none(player.get("preco_pre_rodada")),
        "favorite_proxy_ppg_diff_positive": _positive_bool_or_none(player.get("footystats_ppg_diff")),
        "footystats_ppg_diff": _numeric_or_none(player.get("footystats_ppg_diff")),
    }


def _with_lineup_context(selected: pd.DataFrame, *, fixtures: pd.DataFrame | None) -> pd.DataFrame:
    output = selected.copy()
    if output.empty:
        output["same_club_selected_count"] = pd.Series(dtype="float64")
        output[_OPPONENT_OVERLAP_COLUMN] = pd.Series(dtype="bool")
        return output

    if {"id_clube", "id_atleta"}.issubset(output.columns):
        output["same_club_selected_count"] = output.groupby("id_clube", dropna=False)["id_atleta"].transform("nunique")
    else:
        output["same_club_selected_count"] = None

    overlap_keys = _opponent_overlap_keys(output, fixtures)
    output[_OPPONENT_OVERLAP_COLUMN] = [
        key is not None and key in overlap_keys for key in _lineup_player_keys(output)
    ]
    return output


def _opponent_overlap_keys(selected: pd.DataFrame, fixtures: pd.DataFrame | None) -> set[tuple[int | None, int]]:
    if fixtures is None or selected.empty or "id_clube" not in selected.columns:
        return set()
    if not {"rodada", "id_clube_home", "id_clube_away"}.issubset(fixtures.columns):
        return set()

    selected_clubs_by_round: dict[int | None, set[int]] = {}
    for _, player in selected.iterrows():
        club_id = _int_or_none(player.get("id_clube"))
        if club_id is None:
            continue
        round_key = _int_or_none(player.get("rodada")) if "rodada" in selected.columns else None
        selected_clubs_by_round.setdefault(round_key, set()).add(club_id)

    overlap: set[tuple[int | None, int]] = set()
    for _, fixture in fixtures.iterrows():
        home = _int_or_none(fixture.get("id_clube_home"))
        away = _int_or_none(fixture.get("id_clube_away"))
        fixture_round = _int_or_none(fixture.get("rodada"))
        if home is None or away is None:
            continue
        for selected_round, club_ids in selected_clubs_by_round.items():
            if selected_round is not None and fixture_round != selected_round:
                continue
            if home in club_ids and away in club_ids:
                overlap.update({(selected_round, home), (selected_round, away)})
    return overlap


def _lineup_player_keys(selected: pd.DataFrame) -> list[tuple[int | None, int] | None]:
    keys: list[tuple[int | None, int] | None] = []
    for _, player in selected.iterrows():
        club_id = _int_or_none(player.get("id_clube"))
        if club_id is None:
            keys.append(None)
            continue
        round_key = _int_or_none(player.get("rodada")) if "rodada" in selected.columns else None
        keys.append((round_key, club_id))
    return keys


def _opponent_overlap_round_rate(frame: pd.DataFrame) -> float:
    if frame.empty or _OPPONENT_OVERLAP_COLUMN not in frame.columns:
        return 0.0
    values = frame[_OPPONENT_OVERLAP_COLUMN].map(_bool_or_none).fillna(False)
    if "rodada" not in frame.columns:
        return float(values.any())
    round_flags = values.groupby(frame["rodada"], dropna=False).any()
    if round_flags.empty:
        return 0.0
    return float(round_flags.mean())


def _avg_players_in_opponent_overlap(frame: pd.DataFrame) -> float:
    if frame.empty or not {_OPPONENT_OVERLAP_COLUMN, "id_atleta"}.issubset(frame.columns):
        return 0.0
    values = frame[_OPPONENT_OVERLAP_COLUMN].map(_bool_or_none).fillna(False)
    if "rodada" not in frame.columns:
        return float(frame.loc[values, "id_atleta"].nunique())

    all_rounds = pd.Index(frame["rodada"].drop_duplicates())
    overlap_counts = frame.loc[values].groupby("rodada", dropna=False)["id_atleta"].nunique()
    if all_rounds.empty:
        return 0.0
    return float(overlap_counts.reindex(all_rounds, fill_value=0).mean())


def _share_true(frame: pd.DataFrame, column: str, *, exclude_tecnico: bool) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = _metric_frame(frame, exclude_tecnico=exclude_tecnico)[column].map(_bool_or_none).dropna()
    if values.empty:
        return None
    return float(values.mean())


def _share_positive(frame: pd.DataFrame, column: str, *, exclude_tecnico: bool) -> float | None:
    values = _numeric_values(frame, column, exclude_tecnico=exclude_tecnico)
    if values.empty:
        return None
    return float(values.gt(0).mean())


def _share_at_most(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
    *,
    exclude_tecnico: bool,
) -> float | None:
    values = _numeric_values(frame, column, exclude_tecnico=exclude_tecnico)
    if values.empty:
        return None
    return float(values.le(threshold).mean())


def _median_numeric(frame: pd.DataFrame, column: str, *, exclude_tecnico: bool) -> float | None:
    values = _numeric_values(frame, column, exclude_tecnico=exclude_tecnico)
    if values.empty:
        return None
    return float(values.median())


def _numeric_values(frame: pd.DataFrame, column: str, *, exclude_tecnico: bool) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64")
    metric_frame = _metric_frame(frame, exclude_tecnico=exclude_tecnico)
    return pd.to_numeric(metric_frame[column], errors="coerce").dropna()


def _metric_frame(frame: pd.DataFrame, *, exclude_tecnico: bool) -> pd.DataFrame:
    if not exclude_tecnico or "posicao" not in frame.columns:
        return frame
    position = frame["posicao"].astype(str).str.lower()
    return frame.loc[position.ne(_NON_PLAYER_POSITION)]


def _avg_same_club_count(frame: pd.DataFrame) -> float | None:
    if frame.empty or not {"id_clube", "id_atleta"}.issubset(frame.columns):
        return None
    return float(frame.groupby("id_clube", dropna=False)["id_atleta"].nunique().mean())


def _numeric_gap(left: object, right: object) -> float | None:
    left_value = _numeric_or_none(left)
    right_value = _numeric_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _bool_or_none(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric_value = float(value)
        if numeric_value == 1.0:
            return True
        if numeric_value == 0.0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
        return None
    return None


def _positive_bool_or_none(value: object) -> bool | None:
    numeric = _numeric_or_none(value)
    if numeric is None:
        return None
    return numeric > 0


def _numeric_or_none(value: object) -> float | None:
    if _is_missing(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if _is_missing(numeric):
        return None
    return float(numeric)


def _int_or_none(value: object) -> int | None:
    numeric = _numeric_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return bool(pd.isna(value))
