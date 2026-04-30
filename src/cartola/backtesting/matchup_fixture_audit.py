from __future__ import annotations

import argparse
import json
import traceback as traceback_module
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from cartola.backtesting.data import load_season_data, normalize_fixture_frame, played_club_set
from cartola.backtesting.strict_fixtures import sha256_file, validate_strict_manifest

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/fixtures")
CSV_ERROR_MESSAGE_LIMIT = 300

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

CONTEXT_COLUMNS: tuple[str, ...] = (
    "season",
    "rodada",
    "id_clube",
    "opponent_id_clube",
    "opponent_nome_clube",
    "is_home",
    "fixture_source",
    "source_file",
    "source_manifest",
    "source_sha256",
    "source_manifest_sha256",
)

CSV_COLUMNS: tuple[str, ...] = (
    "season",
    "season_status",
    "metrics_comparable",
    "fixture_status",
    "round_file_count",
    "min_round",
    "max_round",
    "detected_rounds",
    "expected_club_round_count",
    "fixture_context_row_count",
    "missing_context_count",
    "duplicate_context_count",
    "extra_context_count",
    "strict_context_count",
    "exploratory_context_count",
    "fixture_sources",
    "error_stage",
    "error_type",
    "error_message",
    "notes",
)


@dataclass(frozen=True)
class MatchupFixtureAuditConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    expected_complete_rounds: int = 38
    complete_round_threshold: int = 38
    current_year: int | None = None

    def resolved_current_year(self, clock: Callable[[], datetime] | None = None) -> int:
        if self.current_year is not None:
            return self.current_year
        now = clock() if clock is not None else datetime.now(UTC)
        return now.year


@dataclass(frozen=True)
class ErrorDetail:
    stage: str
    exception_type: str
    message: str
    traceback: str | None = None


@dataclass
class SeasonMatchupFixtureRecord:
    season: int
    season_status: str
    metrics_comparable: bool
    fixture_status: str
    round_file_count: int
    min_round: int | None
    max_round: int | None
    detected_rounds: list[int]
    expected_club_round_count: int = 0
    fixture_context_row_count: int = 0
    missing_context_count: int = 0
    duplicate_context_count: int = 0
    extra_context_count: int = 0
    strict_context_count: int = 0
    exploratory_context_count: int = 0
    fixture_sources: list[str] = field(default_factory=list)
    rounds: list[dict[str, object]] = field(default_factory=list)
    missing_context_keys: list[dict[str, int]] = field(default_factory=list)
    duplicate_context_keys: list[dict[str, int]] = field(default_factory=list)
    extra_context_keys: list[dict[str, int]] = field(default_factory=list)
    fixture_context_rows: list[dict[str, object]] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_manifests: list[str] = field(default_factory=list)
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    notes: list[str] = field(default_factory=list)
    error_detail: ErrorDetail | None = None

    def to_csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "season_status": self.season_status,
            "metrics_comparable": self.metrics_comparable,
            "fixture_status": self.fixture_status,
            "round_file_count": self.round_file_count,
            "min_round": self.min_round,
            "max_round": self.max_round,
            "detected_rounds": ",".join(str(round_number) for round_number in self.detected_rounds),
            "expected_club_round_count": self.expected_club_round_count,
            "fixture_context_row_count": self.fixture_context_row_count,
            "missing_context_count": self.missing_context_count,
            "duplicate_context_count": self.duplicate_context_count,
            "extra_context_count": self.extra_context_count,
            "strict_context_count": self.strict_context_count,
            "exploratory_context_count": self.exploratory_context_count,
            "fixture_sources": ",".join(self.fixture_sources),
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "notes": "; ".join(self.notes),
        }

    def to_json_object(self) -> dict[str, object]:
        row = self.to_csv_row()
        row["detected_rounds"] = self.detected_rounds
        row["fixture_sources"] = self.fixture_sources
        row["notes"] = self.notes
        row["rounds"] = self.rounds
        row["missing_context_keys"] = self.missing_context_keys
        row["duplicate_context_keys"] = self.duplicate_context_keys
        row["extra_context_keys"] = self.extra_context_keys
        row["fixture_context_rows"] = self.fixture_context_rows
        row["source_files"] = self.source_files
        row["source_manifests"] = self.source_manifests
        row["error_detail"] = None if self.error_detail is None else asdict(self.error_detail)
        return row


@dataclass(frozen=True)
class MatchupFixtureAuditRunResult:
    config: MatchupFixtureAuditConfig
    records: list[SeasonMatchupFixtureRecord]
    csv_path: Path
    json_path: Path
    decision: dict[str, object]


def parse_seasons(value: str) -> tuple[int, ...]:
    raw_parts = value.split(",")
    if any(part.strip() == "" for part in raw_parts):
        raise ValueError("seasons must not contain empty entries")

    seasons: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        try:
            season = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid season: {part}") from exc
        if season <= 0:
            raise ValueError("seasons must be positive integers")
        if season in seen:
            raise ValueError(f"duplicate season: {season}")
        seasons.append(season)
        seen.add(season)
    return tuple(seasons)


def parse_seasons_arg(value: str) -> tuple[int, ...]:
    try:
        return parse_seasons(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Cartola matchup fixture coverage.")
    parser.add_argument("--seasons", type=parse_seasons_arg, default=DEFAULT_SEASONS)
    parser.add_argument("--current-year", type=_positive_int_arg, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-complete-rounds", type=_positive_int_arg, default=38)
    parser.add_argument("--complete-round-threshold", type=_positive_int_arg, default=38)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MatchupFixtureAuditConfig:
    return MatchupFixtureAuditConfig(
        seasons=args.seasons,
        project_root=args.project_root,
        output_root=args.output_root,
        expected_complete_rounds=args.expected_complete_rounds,
        complete_round_threshold=args.complete_round_threshold,
        current_year=args.current_year,
    )


def classify_season(
    season: int,
    detected_rounds: list[int],
    config: MatchupFixtureAuditConfig,
) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    max_round = max(detected_rounds) if detected_rounds else 0
    if season == config.resolved_current_year() and max_round < config.complete_round_threshold:
        notes.append("partial current season; matchup coverage is not historically comparable")
        return "partial_current", False, notes

    expected = list(range(1, config.expected_complete_rounds + 1))
    if detected_rounds == expected:
        return "complete_historical", True, notes

    notes.append("historical season has unusual round file count or round sequence")
    return "irregular_historical", False, notes


def audit_one_season(season: int, config: MatchupFixtureAuditConfig) -> SeasonMatchupFixtureRecord:
    try:
        season_df = load_season_data(season, project_root=config.project_root)
        detected_rounds = _detected_rounds(season_df)
    except Exception as exc:  # noqa: BLE001 - audit reports per-season failures
        error = _error_detail("load", exc)
        record = SeasonMatchupFixtureRecord(
            season=season,
            season_status="load_failed",
            metrics_comparable=False,
            fixture_status=STATUS_FAILED,
            round_file_count=0,
            min_round=None,
            max_round=None,
            detected_rounds=[],
        )
        _apply_error(record, "load", error)
        return record

    season_status, season_comparable, notes = classify_season(season, detected_rounds, config)
    record = SeasonMatchupFixtureRecord(
        season=season,
        season_status=season_status,
        metrics_comparable=season_comparable,
        fixture_status=STATUS_SKIPPED,
        round_file_count=len(detected_rounds),
        min_round=min(detected_rounds) if detected_rounds else None,
        max_round=max(detected_rounds) if detected_rounds else None,
        detected_rounds=detected_rounds,
        notes=notes,
    )

    expected = _expected_keys(season_df, detected_rounds)
    context_frames: list[pd.DataFrame] = []
    for round_number in detected_rounds:
        try:
            round_context = load_round_fixture_context(
                config.project_root,
                season=season,
                round_number=round_number,
            )
        except Exception as exc:  # noqa: BLE001 - audit reports per-season failures
            _apply_error(record, "fixture", _error_detail("fixture", exc))
            return record
        context_frames.append(round_context)
        round_played_clubs = played_club_set(season_df, round_number)
        record.rounds.append(
            {
                "rodada": int(round_number),
                "expected_club_count": len(round_played_clubs),
                "fixture_context_row_count": int(len(round_context)),
                "fixture_sources": _unique_strings(round_context, "fixture_source"),
            }
        )

    context = pd.concat(context_frames, ignore_index=True) if context_frames else _empty_context_frame()
    context = _with_opponent_names(context, season_df)
    record.fixture_context_rows = _context_json_records(context)
    _populate_context_counts(record, expected, context)

    if record.missing_context_count or record.duplicate_context_count or record.extra_context_count:
        record.fixture_status = STATUS_FAILED
        record.metrics_comparable = False
        record.notes.append("fixture context coverage has missing, duplicate, or extra club-round keys")
        return record

    record.fixture_status = STATUS_OK
    record.metrics_comparable = season_comparable
    return record


def build_decision(records: list[SeasonMatchupFixtureRecord]) -> dict[str, object]:
    comparable = {record.season for record in records if record.metrics_comparable}
    requested = {record.season for record in records}
    core = {2023, 2024, 2025}
    if core.issubset(comparable):
        status = "ready_for_matchup_context"
        next_step = "implement matchup_context_mode=cartola_matchup_v1"
    elif 2025 in comparable:
        status = "exploratory_only"
        next_step = "keep matchup context exploratory until 2023-2024 fixture coverage is fixed"
    else:
        status = "coverage_blocked"
        next_step = "fix or import fixture coverage before feature work"

    return {
        "status": status,
        "comparable_seasons": sorted(comparable),
        "requested_seasons": sorted(requested),
        "recommended_next_step": next_step,
    }


def write_audit_reports(
    records: list[SeasonMatchupFixtureRecord],
    config: MatchupFixtureAuditConfig,
    *,
    generated_at_utc: str,
) -> tuple[Path, Path]:
    output_dir = config.output_root if config.output_root.is_absolute() else config.project_root / config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matchup_fixture_coverage.csv"
    json_path = output_dir / "matchup_fixture_coverage.json"

    pd.DataFrame([record.to_csv_row() for record in records], columns=pd.Index(CSV_COLUMNS)).to_csv(
        csv_path,
        index=False,
    )
    payload = {
        "generated_at_utc": generated_at_utc,
        "project_root": str(config.project_root),
        "config": _config_json(config),
        "decision": build_decision(records),
        "seasons": [record.to_json_object() for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def run_matchup_fixture_audit(
    config: MatchupFixtureAuditConfig,
    *,
    clock: Callable[[], datetime] | None = None,
) -> MatchupFixtureAuditRunResult:
    resolved_config = replace(config, current_year=config.resolved_current_year(clock))
    generated_at = _generated_at_utc(clock)
    records = [audit_one_season(season, resolved_config) for season in resolved_config.seasons]
    csv_path, json_path = write_audit_reports(records, resolved_config, generated_at_utc=generated_at)
    return MatchupFixtureAuditRunResult(
        config=resolved_config,
        records=records,
        csv_path=csv_path,
        json_path=json_path,
        decision=build_decision(records),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    result = run_matchup_fixture_audit(config)
    print("Matchup fixture coverage audit complete")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Decision: {result.decision['status']}")
    return 0


def load_round_fixture_context(
    project_root: str | Path,
    *,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    root = Path(project_root)
    strict_path = root / "data" / "01_raw" / "fixtures_strict" / str(season) / f"partidas-{round_number}.csv"
    if strict_path.exists():
        validate_strict_manifest(
            project_root=root,
            fixture_path=strict_path,
            season=season,
            round_number=round_number,
        )
        strict_frame = normalize_fixture_frame(pd.read_csv(strict_path), source=strict_path)
        return _context_rows_from_fixture(
            strict_frame,
            season=season,
            source="strict",
            source_file=strict_path,
            source_manifest=strict_path.with_suffix(".manifest.json"),
            project_root=root,
        )

    exploratory_path = root / "data" / "01_raw" / "fixtures" / str(season) / f"partidas-{round_number}.csv"
    if exploratory_path.exists():
        exploratory_frame = normalize_fixture_frame(pd.read_csv(exploratory_path), source=exploratory_path)
        return _context_rows_from_fixture(
            exploratory_frame,
            season=season,
            source="exploratory",
            source_file=exploratory_path,
            source_manifest=None,
            project_root=root,
        )

    return _empty_context_frame()


def _config_json(config: MatchupFixtureAuditConfig) -> dict[str, object]:
    return {
        "seasons": list(config.seasons),
        "project_root": str(config.project_root),
        "output_root": str(config.output_root),
        "expected_complete_rounds": config.expected_complete_rounds,
        "complete_round_threshold": config.complete_round_threshold,
        "current_year": config.current_year,
    }


def _generated_at_utc(clock: Callable[[], datetime] | None = None) -> str:
    now = clock() if clock is not None else datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).isoformat()


def _empty_context_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CONTEXT_COLUMNS))


def _with_opponent_names(context: pd.DataFrame, season_df: pd.DataFrame) -> pd.DataFrame:
    if context.empty or "nome_clube" not in season_df.columns:
        return context

    lookup = _club_name_lookup(season_df)
    enriched = context.copy()
    enriched["opponent_nome_clube"] = enriched["opponent_id_clube"].map(lookup)
    return enriched


def _club_name_lookup(season_df: pd.DataFrame) -> dict[int, str]:
    clubs = season_df[["id_clube", "nome_clube"]].copy()
    clubs = clubs.dropna(subset=["id_clube", "nome_clube"])
    if clubs.empty:
        return {}
    clubs["id_clube"] = pd.to_numeric(clubs["id_clube"], errors="coerce")
    clubs = clubs.dropna(subset=["id_clube"])
    clubs["id_clube"] = clubs["id_clube"].astype(int)
    clubs["nome_clube"] = clubs["nome_clube"].astype(str).str.strip()
    clubs = clubs.loc[clubs["nome_clube"].ne("")]
    clubs = clubs.drop_duplicates(subset=["id_clube", "nome_clube"]).sort_values(["id_clube", "nome_clube"])
    return {
        int(row["id_clube"]): str(row["nome_clube"])
        for row in clubs.drop_duplicates(subset=["id_clube"], keep="first").to_dict("records")
    }


def _context_json_records(context: pd.DataFrame) -> list[dict[str, object]]:
    if context.empty:
        return []
    rows: list[dict[str, object]] = []
    for row in context.sort_values(["rodada", "id_clube"]).to_dict("records"):
        opponent_name = row["opponent_nome_clube"]
        rows.append(
            {
                "season": int(row["season"]),
                "rodada": int(row["rodada"]),
                "id_clube": int(row["id_clube"]),
                "opponent_id_clube": int(row["opponent_id_clube"]),
                "opponent_nome_clube": None if pd.isna(opponent_name) else str(opponent_name),
                "is_home": int(row["is_home"]),
                "fixture_source": str(row["fixture_source"]),
                "source_file": str(row["source_file"]),
                "source_manifest": None if pd.isna(row["source_manifest"]) else str(row["source_manifest"]),
                "source_sha256": str(row["source_sha256"]),
                "source_manifest_sha256": (
                    None if pd.isna(row["source_manifest_sha256"]) else str(row["source_manifest_sha256"])
                ),
            }
        )
    return rows


def _detected_rounds(season_df: pd.DataFrame) -> list[int]:
    rounds = pd.to_numeric(season_df["rodada"], errors="raise")
    return sorted(rounds.dropna().astype(int).unique().tolist())


def _expected_keys(season_df: pd.DataFrame, detected_rounds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for round_number in detected_rounds:
        for club_id in sorted(played_club_set(season_df, round_number)):
            rows.append({"rodada": int(round_number), "id_clube": int(club_id)})
    return pd.DataFrame(rows, columns=pd.Index(["rodada", "id_clube"]))


def _populate_context_counts(
    record: SeasonMatchupFixtureRecord,
    expected: pd.DataFrame,
    context: pd.DataFrame,
) -> None:
    record.expected_club_round_count = int(len(expected))
    record.fixture_context_row_count = int(len(context))
    if not context.empty:
        record.strict_context_count = int(context["fixture_source"].eq("strict").sum())
        record.exploratory_context_count = int(context["fixture_source"].eq("exploratory").sum())
        record.fixture_sources = _unique_strings(context, "fixture_source")
        record.source_files = _unique_strings(context, "source_file")
        record.source_manifests = _unique_strings(context, "source_manifest")

    unique_context = (
        context[["rodada", "id_clube"]].drop_duplicates()
        if not context.empty
        else pd.DataFrame(columns=pd.Index(["rodada", "id_clube"]))
    )
    missing = expected.merge(unique_context, on=["rodada", "id_clube"], how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only"), ["rodada", "id_clube"]]
    extra = unique_context.merge(expected, on=["rodada", "id_clube"], how="left", indicator=True)
    extra = extra.loc[extra["_merge"].eq("left_only"), ["rodada", "id_clube"]]

    record.missing_context_keys = _key_records(missing, ["rodada", "id_clube"])
    record.duplicate_context_keys = _duplicate_key_records(context)
    record.extra_context_keys = _key_records(extra, ["rodada", "id_clube"])
    record.missing_context_count = len(record.missing_context_keys)
    record.duplicate_context_count = len(record.duplicate_context_keys)
    record.extra_context_count = len(record.extra_context_keys)


def _key_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, int]]:
    if frame.empty:
        return []
    return [
        {column: int(row[column]) for column in columns}
        for row in frame[columns].drop_duplicates().sort_values(columns).to_dict("records")
    ]


def _duplicate_key_records(context: pd.DataFrame) -> list[dict[str, int]]:
    if context.empty:
        return []
    counts = context.groupby(["rodada", "id_clube"], as_index=False).size()
    counts = counts.loc[counts["size"].gt(1)].rename(columns={"size": "count"})
    return [
        {"rodada": int(row["rodada"]), "id_clube": int(row["id_clube"]), "count": int(row["count"])}
        for row in counts.sort_values(["rodada", "id_clube"]).to_dict("records")
    ]


def _unique_strings(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique())


def _error_detail(stage: str, exc: BaseException) -> ErrorDetail:
    return ErrorDetail(
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__)),
    )


def _short_error_message(message: str) -> str:
    one_line = " ".join(str(message).split())
    if len(one_line) <= CSV_ERROR_MESSAGE_LIMIT:
        return one_line
    return one_line[: CSV_ERROR_MESSAGE_LIMIT - 3] + "..."


def _apply_error(record: SeasonMatchupFixtureRecord, stage: str, error: ErrorDetail) -> None:
    record.fixture_status = STATUS_FAILED
    record.metrics_comparable = False
    record.error_stage = stage
    record.error_type = error.exception_type
    record.error_message = _short_error_message(error.message)
    record.error_detail = error


def _context_rows_from_fixture(
    fixture_frame: pd.DataFrame,
    *,
    season: int,
    source: str,
    source_file: Path,
    source_manifest: Path | None,
    project_root: Path,
) -> pd.DataFrame:
    source_file_label = _relative_path(source_file, project_root)
    source_manifest_label = None if source_manifest is None else _relative_path(source_manifest, project_root)
    source_hash = sha256_file(source_file)
    manifest_hash = None if source_manifest is None else sha256_file(source_manifest)
    rows: list[dict[str, object]] = []

    for fixture in fixture_frame.to_dict("records"):
        round_number = int(fixture["rodada"])
        home_id = int(fixture["id_clube_home"])
        away_id = int(fixture["id_clube_away"])
        rows.append(
            _context_row(
                season=season,
                round_number=round_number,
                club_id=home_id,
                opponent_id=away_id,
                is_home=1,
                source=source,
                source_file=source_file_label,
                source_manifest=source_manifest_label,
                source_hash=source_hash,
                manifest_hash=manifest_hash,
            )
        )
        rows.append(
            _context_row(
                season=season,
                round_number=round_number,
                club_id=away_id,
                opponent_id=home_id,
                is_home=0,
                source=source,
                source_file=source_file_label,
                source_manifest=source_manifest_label,
                source_hash=source_hash,
                manifest_hash=manifest_hash,
            )
        )

    return pd.DataFrame(rows, columns=pd.Index(CONTEXT_COLUMNS))


def _context_row(
    *,
    season: int,
    round_number: int,
    club_id: int,
    opponent_id: int,
    is_home: int,
    source: str,
    source_file: str,
    source_manifest: str | None,
    source_hash: str,
    manifest_hash: str | None,
) -> dict[str, object]:
    return {
        "season": season,
        "rodada": round_number,
        "id_clube": club_id,
        "opponent_id_clube": opponent_id,
        "opponent_nome_clube": None,
        "is_home": is_home,
        "fixture_source": source,
        "source_file": source_file,
        "source_manifest": source_manifest,
        "source_sha256": source_hash,
        "source_manifest_sha256": manifest_hash,
    }


def _relative_path(path: Path, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)
