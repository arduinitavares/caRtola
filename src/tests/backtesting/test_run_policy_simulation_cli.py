from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.optimizer import optimize_squad
from cartola.backtesting.optimizer_policies import NO_POLICY
from cartola.backtesting.policy_simulation import PolicySimulationError
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION, actual_scores_with_captain

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_policy_simulation.py"
SPEC = importlib.util.spec_from_file_location("run_policy_simulation", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_policy_simulation_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_policy_simulation_cli)
main = run_policy_simulation_cli.main
parse_args = run_policy_simulation_cli.parse_args


def test_parse_args_builds_policy_simulation_defaults() -> None:
    args = parse_args(
        [
            "--experiment-path",
            "experiment",
            "--hypothesis-id",
            "H001",
            "--policy-set",
            "opponent-overlap-v1",
            "--models",
            "ridge,xgboost_depth2_slow",
            "--feature-packs",
            "ppg,ppg_xg",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("experiment")
    assert args.hypothesis_id == "H001"
    assert args.policy_set == "opponent-overlap-v1"
    assert args.models == "ridge,xgboost_depth2_slow"
    assert args.feature_packs == "ppg,ppg_xg"
    assert args.seasons == "2021,2022,2023,2024,2025"
    assert args.current_year == 2026
    assert args.output_root == Path("data/08_reporting/policy_simulations")
    assert args.allow_incomplete_report is False


def test_main_writes_artifacts_and_progress_for_synthetic_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_path = tmp_path / "experiment"
    output_root = tmp_path / "policy_simulations"
    _write_policy_child(
        experiment_path
        / "runs"
        / "season=2025"
        / "model=test_model"
        / "feature_pack=synthetic_pack"
    )

    exit_code = main(
        [
            "--experiment-path",
            str(experiment_path),
            "--hypothesis-id",
            "H001-smoke",
            "--policy-set",
            "opponent-overlap-v1",
            "--models",
            "test_model",
            "--feature-packs",
            "synthetic_pack",
            "--seasons",
            "2025",
            "--current-year",
            "2026",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    progress_text = captured.out + captured.err
    assert "Policy simulation started" in progress_text
    assert "START child" in progress_text
    assert "season=2025 model=test_model feature_pack=synthetic_pack" in progress_text
    assert "DONE child" in progress_text
    assert "Policy simulation complete" in progress_text

    run_dirs = sorted(output_root.glob("policy_simulation_started_at=*"))
    assert len(run_dirs) == 1
    output_path = run_dirs[0]
    for artifact_name in (
        "policy_simulation_manifest.json",
        "policy_ranked_summary.csv",
        "policy_per_season_summary.csv",
        "policy_round_results.csv",
        "policy_selected_players.csv",
        "policy_invalid_rows.csv",
        "policy_profile_summary.csv",
        "policy_comparability_report.json",
        "policy_simulation_report.html",
    ):
        assert (output_path / artifact_name).exists()

    manifest = json.loads((output_path / "policy_simulation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["hypothesis_id"] == "H001-smoke"
    assert manifest["policy_set_id"] == "opponent-overlap-v1"
    assert manifest["experiment_path"] == str(experiment_path)
    assert manifest["selected_seasons"] == [2025]
    assert manifest["selected_models"] == ["test_model"]
    assert manifest["selected_feature_packs"] == ["synthetic_pack"]
    assert manifest["child_count"] == 1
    assert manifest["fixture_identity_status"] == "unverified"
    assert manifest["budget_policy"] == "moving"
    assert manifest["source_candidate_signature_status"] == "artifact_backed_unverified"

    comparability_report = json.loads((output_path / "policy_comparability_report.json").read_text(encoding="utf-8"))
    assert comparability_report["status"] == "diagnostic_only"
    assert comparability_report["fixture_identity_status"] == "unverified"
    assert comparability_report["budget_policy"] == "moving"

    ranked_summary = pd.read_csv(output_path / "policy_ranked_summary.csv")
    assert "diagnostic_only" in set(ranked_summary["decision_status"].astype(str))
    assert "no_policy" in set(pd.read_csv(output_path / "policy_round_results.csv")["policy_variant"].astype(str))


def test_main_allows_incomplete_report_and_writes_invalid_rows(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    output_root = tmp_path / "policy_simulations"
    child = (
        experiment_path
        / "runs"
        / "season=2025"
        / "model=test_model"
        / "feature_pack=synthetic_pack"
    )
    _write_policy_child(child)
    (child / "fixtures_for_round.csv").unlink()

    exit_code = main(
        [
            "--experiment-path",
            str(experiment_path),
            "--hypothesis-id",
            "H001-smoke",
            "--policy-set",
            "opponent-overlap-v1",
            "--models",
            "test_model",
            "--feature-packs",
            "synthetic_pack",
            "--seasons",
            "2025",
            "--current-year",
            "2026",
            "--output-root",
            str(output_root),
            "--allow-incomplete-report",
        ]
    )

    assert exit_code == 0
    output_path = next(output_root.glob("policy_simulation_started_at=*"))
    for artifact_name in (
        "policy_simulation_manifest.json",
        "policy_ranked_summary.csv",
        "policy_per_season_summary.csv",
        "policy_round_results.csv",
        "policy_selected_players.csv",
        "policy_invalid_rows.csv",
        "policy_profile_summary.csv",
        "policy_comparability_report.json",
        "policy_simulation_report.html",
    ):
        assert (output_path / artifact_name).exists()

    manifest = json.loads((output_path / "policy_simulation_manifest.json").read_text(encoding="utf-8"))
    invalid_rows = pd.read_csv(output_path / "policy_invalid_rows.csv")
    assert manifest["invalid_row_count"] > 0
    assert not invalid_rows.empty
    assert invalid_rows.loc[0, "season"] == 2025
    assert invalid_rows.loc[0, "model_id"] == "test_model"
    assert invalid_rows.loc[0, "feature_pack"] == "synthetic_pack"
    assert "fixture coverage" in str(invalid_rows.loc[0, "error_message"])


def test_main_allows_incomplete_report_for_duplicate_candidate_rows(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment"
    output_root = tmp_path / "policy_simulations"
    child = (
        experiment_path
        / "runs"
        / "season=2025"
        / "model=test_model"
        / "feature_pack=synthetic_pack"
    )
    _write_policy_child(child)
    player_predictions_path = child / "player_predictions.csv"
    player_predictions = pd.read_csv(player_predictions_path)
    duplicate = player_predictions.iloc[[0]].copy()
    duplicate.loc[:, "test_model_score"] = 999.0
    pd.concat([player_predictions, duplicate], ignore_index=True).to_csv(player_predictions_path, index=False)

    exit_code = main(
        [
            "--experiment-path",
            str(experiment_path),
            "--hypothesis-id",
            "H001-smoke",
            "--policy-set",
            "opponent-overlap-v1",
            "--models",
            "test_model",
            "--feature-packs",
            "synthetic_pack",
            "--seasons",
            "2025",
            "--current-year",
            "2026",
            "--output-root",
            str(output_root),
            "--allow-incomplete-report",
        ]
    )

    assert exit_code == 0
    output_path = next(output_root.glob("policy_simulation_started_at=*"))
    assert (output_path / "policy_invalid_rows.csv").exists()
    manifest = json.loads((output_path / "policy_simulation_manifest.json").read_text(encoding="utf-8"))
    invalid_rows = pd.read_csv(output_path / "policy_invalid_rows.csv")
    assert manifest["invalid_row_count"] > 0
    assert not invalid_rows.empty
    assert invalid_rows.loc[0, "season"] == 2025
    assert invalid_rows.loc[0, "model_id"] == "test_model"
    assert invalid_rows.loc[0, "feature_pack"] == "synthetic_pack"
    assert invalid_rows.loc[0, "error_type"] == "DuplicateCandidateError"
    assert "Conflicting duplicate candidate rows" in str(invalid_rows.loc[0, "error_message"])


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--models", "test_model,test_model", "Duplicate models"),
        ("--seasons", "2025,2025", "Duplicate seasons"),
        ("--feature-packs", "synthetic_pack,synthetic_pack", "Duplicate feature_packs"),
    ],
)
def test_main_rejects_duplicate_selectors(tmp_path: Path, flag: str, value: str, message: str) -> None:
    args = [
        "--experiment-path",
        str(tmp_path / "experiment"),
        "--hypothesis-id",
        "H001-smoke",
        "--policy-set",
        "opponent-overlap-v1",
        "--models",
        "test_model",
        "--feature-packs",
        "synthetic_pack",
        "--seasons",
        "2025",
        "--current-year",
        "2026",
    ]
    flag_index = args.index(flag) + 1
    args[flag_index] = value

    with pytest.raises(PolicySimulationError, match=message):
        main(args)


def _write_policy_child(child: Path) -> None:
    child.mkdir(parents=True)
    score_column = "test_model_score"
    metadata: dict[str, object] = {
        "season": 2025,
        "model_id": "test_model",
        "feature_pack": "synthetic_pack",
        "budget_policy": "moving",
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "fixture_mode": "exploratory",
        "matchup_context_mode": "cartola_matchup_v1",
        "start_round": 5,
        "initial_budget": 100.0,
    }
    (child / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    candidates = _synthetic_candidates(score_column=score_column)
    candidates.to_csv(child / "player_predictions.csv", index=False)
    _write_matching_source_outputs(child, candidates=candidates, score_column=score_column)
    _complete_fixture_rows(candidates).to_csv(child / "fixtures_for_round.csv", index=False)


def _write_matching_source_outputs(
    child: Path,
    *,
    candidates: pd.DataFrame,
    score_column: str,
) -> None:
    round_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    current_budget = 100.0
    for round_number in sorted(candidates["rodada"].astype(int).unique().tolist()):
        round_candidates = candidates.loc[candidates["rodada"].eq(round_number)].copy()
        result = optimize_squad(
            round_candidates,
            score_column=score_column,
            config=BacktestConfig(season=2025, start_round=round_number, budget=current_budget),
            budget=current_budget,
            policy=NO_POLICY,
        )
        actual_scores = actual_scores_with_captain(result.selected, actual_column="pontuacao")
        budget_delta = float(result.selected["variacao"].sum())
        budget_after_round = current_budget + budget_delta
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": "test_model",
                "solver_status": result.status,
                "formation": result.formation_name,
                "budget_before_round": current_budget,
                "budget_after_round": budget_after_round,
                "budget_delta": budget_delta,
                "budget_used": result.budget_used,
                "actual_points_with_captain": actual_scores["actual_points_with_captain"],
                "predicted_points_with_captain": result.predicted_points_with_captain,
                "captain_id": result.captain_id,
            }
        )
        selected_players = result.selected.copy()
        selected_players["rodada"] = round_number
        selected_players["strategy"] = "test_model"
        selected_rows.append(selected_players)
        current_budget = budget_after_round

    pd.DataFrame(round_rows).to_csv(child / "round_results.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(child / "selected_players.csv", index=False)


def _synthetic_candidates(*, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scores_by_position = {
        "gol": [11.0, 1.0],
        "lat": [10.0, 9.0, 1.0],
        "zag": [8.0, 7.0, 6.0, 1.0],
        "mei": [15.0, 14.0, 13.0, 12.0, 11.0, 1.0],
        "ata": [18.0, 17.0, 16.0, 1.0],
        "tec": [5.0, 1.0],
    }
    player_id = 1
    for position, scores in scores_by_position.items():
        for score in scores:
            rows.append(_candidate_row(player_id, position=position, score=score, score_column=score_column))
            player_id += 1
    return pd.DataFrame(rows)


def _candidate_row(player_id: int, *, position: str, score: float, score_column: str) -> dict[str, object]:
    return {
        "rodada": 5,
        "id_atleta": player_id,
        "apelido": f"{position}-{player_id}",
        "posicao": position,
        "id_clube": 100 + player_id,
        "nome_clube": f"Club {player_id}",
        "preco_pre_rodada": 1.0,
        "pontuacao": score / 2.0,
        "entrou_em_campo": True,
        "variacao": 0.0,
        "baseline_score": score / 3.0,
        "price_score": 1.0,
        score_column: score,
    }


def _complete_fixture_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for round_number in sorted(candidates["rodada"].astype(int).unique().tolist()):
        club_ids = sorted(candidates.loc[candidates["rodada"].eq(round_number), "id_clube"].astype(int).unique())
        for fixture_index in range(0, len(club_ids), 2):
            home_id = club_ids[fixture_index]
            away_id = club_ids[fixture_index + 1] if fixture_index + 1 < len(club_ids) else 900_000 + round_number
            rows.append({"rodada": round_number, "id_clube_home": home_id, "id_clube_away": away_id})
    return pd.DataFrame(rows)
