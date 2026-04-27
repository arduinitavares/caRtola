from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from cartola.backtesting.data import build_round_alignment_report, load_fixtures, load_season_data
from cartola.backtesting.fixture_import import DEFAULT_THESPORTSDB_LEAGUE_ID, import_thesportsdb_fixtures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Brasileirão fixture schedules from TheSportsDB.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--api-key", default=os.environ.get("THESPORTSDB_API_KEY", "3"))
    parser.add_argument("--league-id", type=int, default=DEFAULT_THESPORTSDB_LEAGUE_ID)
    parser.add_argument("--first-round", type=int, default=1)
    parser.add_argument("--last-round", type=int, default=38)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rounds = range(args.first_round, args.last_round + 1)
    season_df = load_season_data(args.season, project_root=args.project_root)
    result = import_thesportsdb_fixtures(
        season=args.season,
        season_df=season_df,
        rounds=rounds,
        project_root=args.project_root,
        api_key=args.api_key,
        league_id=args.league_id,
    )

    fixtures = load_fixtures(args.season, project_root=args.project_root)
    report = build_round_alignment_report(fixtures, season_df, official_fixtures=result.official_fixtures)
    report_dir = args.project_root / "data" / "08_reporting" / "fixtures" / str(args.season)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "round_alignment.csv"
    report.to_csv(report_path, index=False)

    invalid_report = report.loc[~report["is_valid"]]
    if not invalid_report.empty:
        print(invalid_report.to_string(index=False))
        return 1

    print(f"Imported fixture schedules for season={args.season} rounds={args.first_round}-{args.last_round}")
    print(f"Wrote round alignment report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
