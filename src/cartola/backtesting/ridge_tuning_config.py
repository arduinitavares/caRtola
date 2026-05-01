from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.experiment_config import config_hash, feature_pack_to_modes
from cartola.backtesting.model_registry import ModelId
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION

RidgeTuningStage = Literal["screen", "final"]
RidgeTuningFeaturePack = Literal["ppg", "ppg_xg"]

RIDGE_ALPHA_VALUES: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
RIDGE_TUNING_FEATURE_PACKS: tuple[RidgeTuningFeaturePack, ...] = ("ppg", "ppg_xg")

PRIMARY_INCUMBENT_CANDIDATE_ID = "ridge_alpha_1_0__ppg_xg"
SECONDARY_CONTROL_CANDIDATE_ID = "ridge_alpha_1_0__ppg"


@dataclass(frozen=True)
class RidgeTuningSpec:
    stage: RidgeTuningStage
    season: int
    candidate_id: str
    model_id: ModelId
    feature_pack: RidgeTuningFeaturePack
    alpha: float
    start_round: int
    budget: float
    current_year: int
    jobs: int
    model_parameters: Mapping[str, object]
    model_params_hash: str
    tuning_generation_hash: str
    output_path: Path
    backtest_config: BacktestConfig
    config_identity: Mapping[str, object]


def candidate_id_for(*, alpha: float, feature_pack: str) -> str:
    encoded_alpha = str(alpha).replace(".", "_")
    return f"ridge_alpha_{encoded_alpha}__{feature_pack}"


def build_ridge_tuning_specs(
    *,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    project_root: Path,
    output_root: Path,
    current_year: int,
    jobs: int,
    stage: RidgeTuningStage,
    candidate_ids: set[str] | None = None,
) -> list[RidgeTuningSpec]:
    if any(season >= current_year for season in seasons):
        raise ValueError("Tuning seasons must be before current_year")

    valid_candidate_ids = {
        candidate_id_for(alpha=alpha, feature_pack=feature_pack_id)
        for alpha in RIDGE_ALPHA_VALUES
        for feature_pack_id in RIDGE_TUNING_FEATURE_PACKS
    }
    if candidate_ids is not None:
        unknown_candidate_ids = sorted(candidate_ids - valid_candidate_ids)
        if unknown_candidate_ids:
            raise ValueError(f"Unknown ridge tuning candidate_id: {unknown_candidate_ids[0]}")

    fixture_mode = "none"
    matchup_context_mode = "none"
    tuning_generation_hash = config_hash(
        {
            "alphas": RIDGE_ALPHA_VALUES,
            "feature_packs": RIDGE_TUNING_FEATURE_PACKS,
            "seasons": seasons,
            "start_round": start_round,
            "budget": budget,
            "current_year": current_year,
            "jobs": jobs,
            "fixture_mode": fixture_mode,
            "matchup_context_mode": matchup_context_mode,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
        }
    )

    specs: list[RidgeTuningSpec] = []
    model_id: ModelId = "ridge"

    for season in seasons:
        for alpha in RIDGE_ALPHA_VALUES:
            model_parameters: Mapping[str, object] = {
                "estimator": "sklearn.linear_model.Ridge",
                "alpha": alpha,
            }
            model_params_hash = config_hash(model_parameters)

            for feature_pack_id in RIDGE_TUNING_FEATURE_PACKS:
                candidate_id = candidate_id_for(alpha=alpha, feature_pack=feature_pack_id)
                if candidate_ids is not None and candidate_id not in candidate_ids:
                    continue

                feature_pack = feature_pack_to_modes(feature_pack_id)
                child_output_path = (
                    project_root
                    / output_root
                    / "runs"
                    / f"stage={stage}"
                    / f"season={season}"
                    / f"candidate={candidate_id}"
                )
                backtest_config = BacktestConfig(
                    season=season,
                    start_round=start_round,
                    budget=budget,
                    project_root=project_root,
                    output_root=output_root,
                    fixture_mode=fixture_mode,
                    matchup_context_mode=matchup_context_mode,
                    footystats_mode=feature_pack.footystats_mode,
                    current_year=current_year,
                    jobs=jobs,
                    _output_path_override=child_output_path,
                )
                config_identity = {
                    "stage": stage,
                    "season": season,
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "feature_pack": feature_pack_id,
                    "alpha": alpha,
                    "start_round": start_round,
                    "budget": budget,
                    "current_year": current_year,
                    "jobs": jobs,
                    "fixture_mode": fixture_mode,
                    "footystats_mode": feature_pack.footystats_mode,
                    "matchup_context_mode": matchup_context_mode,
                    "scoring_contract_version": SCORING_CONTRACT_VERSION,
                    "model_parameters": model_parameters,
                    "model_params_hash": model_params_hash,
                    "tuning_generation_hash": tuning_generation_hash,
                }
                specs.append(
                    RidgeTuningSpec(
                        stage=stage,
                        season=season,
                        candidate_id=candidate_id,
                        model_id=model_id,
                        feature_pack=feature_pack_id,
                        alpha=alpha,
                        start_round=start_round,
                        budget=budget,
                        current_year=current_year,
                        jobs=jobs,
                        model_parameters=model_parameters,
                        model_params_hash=model_params_hash,
                        tuning_generation_hash=tuning_generation_hash,
                        output_path=child_output_path,
                        backtest_config=backtest_config,
                        config_identity=config_identity,
                    )
                )

    return specs
