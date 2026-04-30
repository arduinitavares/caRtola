from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, cast

from cartola.backtesting.config import BacktestConfig, FixtureMode, FootyStatsMode, MatchupContextMode
from cartola.backtesting.model_registry import MODEL_SPECS, ModelId
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION

ExperimentGroup = Literal["production-parity", "matchup-research"]
FeaturePackId = Literal["ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup"]


@dataclass(frozen=True)
class FeaturePack:
    feature_pack: FeaturePackId
    footystats_mode: FootyStatsMode
    matchup_context_mode: MatchupContextMode


@dataclass(frozen=True)
class ChildRunSpec:
    group: ExperimentGroup
    season: int
    model_id: ModelId
    feature_pack: FeaturePackId
    fixture_mode: FixtureMode
    footystats_mode: FootyStatsMode
    matchup_context_mode: MatchupContextMode
    start_round: int
    budget: float
    current_year: int
    jobs: int
    scoring_contract_version: str
    model_parameters: Mapping[str, object]
    output_path: Path
    backtest_config: BacktestConfig
    config_identity: Mapping[str, object]


_GROUP_FIXTURE_MODES: Mapping[ExperimentGroup, FixtureMode] = {
    "production-parity": "none",
    "matchup-research": "exploratory",
}

_GROUP_FEATURE_PACKS: Mapping[ExperimentGroup, tuple[FeaturePackId, ...]] = {
    "production-parity": ("ppg", "ppg_xg"),
    "matchup-research": ("ppg", "ppg_xg", "ppg_matchup", "ppg_xg_matchup"),
}

_FEATURE_PACKS: Mapping[FeaturePackId, FeaturePack] = {
    "ppg": FeaturePack(
        feature_pack="ppg",
        footystats_mode="ppg",
        matchup_context_mode="none",
    ),
    "ppg_xg": FeaturePack(
        feature_pack="ppg_xg",
        footystats_mode="ppg_xg",
        matchup_context_mode="none",
    ),
    "ppg_matchup": FeaturePack(
        feature_pack="ppg_matchup",
        footystats_mode="ppg",
        matchup_context_mode="cartola_matchup_v1",
    ),
    "ppg_xg_matchup": FeaturePack(
        feature_pack="ppg_xg_matchup",
        footystats_mode="ppg_xg",
        matchup_context_mode="cartola_matchup_v1",
    ),
}


def _supported_values(values: Iterable[str]) -> str:
    return ", ".join(sorted(values))


def _validate_feature_pack(feature_pack: str) -> FeaturePackId:
    if feature_pack not in _FEATURE_PACKS:
        raise ValueError(
            f"Unsupported feature_pack: {feature_pack}. Supported values: {_supported_values(_FEATURE_PACKS)}"
        )
    return cast(FeaturePackId, feature_pack)


def _validate_experiment_group(group: str) -> ExperimentGroup:
    if group not in _GROUP_FIXTURE_MODES:
        raise ValueError(
            f"Unsupported experiment group: {group}. Supported values: {_supported_values(_GROUP_FIXTURE_MODES)}"
        )
    return cast(ExperimentGroup, group)


def feature_pack_to_modes(feature_pack: str) -> FeaturePack:
    return _FEATURE_PACKS[_validate_feature_pack(feature_pack)]


def build_child_run_specs(
    *,
    group: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    project_root: Path,
    output_root: Path,
    current_year: int,
    jobs: int,
) -> list[ChildRunSpec]:
    if any(season >= current_year for season in seasons):
        raise ValueError("Experiment seasons must be before current_year")

    experiment_group = _validate_experiment_group(group)
    fixture_mode = _GROUP_FIXTURE_MODES[experiment_group]
    specs: list[ChildRunSpec] = []

    for season in seasons:
        for model_id in MODEL_SPECS:
            model_parameters = MODEL_SPECS[model_id].parameters
            for feature_pack_id in _GROUP_FEATURE_PACKS[experiment_group]:
                feature_pack = feature_pack_to_modes(feature_pack_id)
                child_output_path = (
                    project_root
                    / output_root
                    / "runs"
                    / f"season={season}"
                    / f"model={model_id}"
                    / f"feature_pack={feature_pack_id}"
                )
                backtest_config = BacktestConfig(
                    season=season,
                    start_round=start_round,
                    budget=budget,
                    project_root=project_root,
                    output_root=output_root,
                    fixture_mode=fixture_mode,
                    matchup_context_mode=feature_pack.matchup_context_mode,
                    footystats_mode=feature_pack.footystats_mode,
                    current_year=current_year,
                    jobs=jobs,
                    _output_path_override=child_output_path,
                )
                config_identity = {
                    "group": experiment_group,
                    "season": season,
                    "model_id": model_id,
                    "feature_pack": feature_pack_id,
                    "fixture_mode": fixture_mode,
                    "footystats_mode": feature_pack.footystats_mode,
                    "matchup_context_mode": feature_pack.matchup_context_mode,
                    "start_round": start_round,
                    "budget": budget,
                    "current_year": current_year,
                    "jobs": jobs,
                    "scoring_contract_version": SCORING_CONTRACT_VERSION,
                    "model_parameters": model_parameters,
                }
                specs.append(
                    ChildRunSpec(
                        group=experiment_group,
                        season=season,
                        model_id=model_id,
                        feature_pack=feature_pack_id,
                        fixture_mode=fixture_mode,
                        footystats_mode=feature_pack.footystats_mode,
                        matchup_context_mode=feature_pack.matchup_context_mode,
                        start_round=start_round,
                        budget=budget,
                        current_year=current_year,
                        jobs=jobs,
                        scoring_contract_version=SCORING_CONTRACT_VERSION,
                        model_parameters=model_parameters,
                        output_path=child_output_path,
                        backtest_config=backtest_config,
                        config_identity=config_identity,
                    )
                )

    return specs


def config_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_id(*, group: ExperimentGroup, started_at_utc: str, matrix_hash: str) -> str:
    return f"group={group}__started_at={started_at_utc}__matrix={matrix_hash[:12]}"


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return cast(Any, str(value))
