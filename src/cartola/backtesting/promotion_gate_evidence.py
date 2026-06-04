from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


class PromotionGateEvidenceError(ValueError):
    """Raised when promotion gate evidence cannot be scanned."""


@dataclass(frozen=True)
class EvidenceField:
    name: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class FieldCheck:
    name: str
    present: bool
    source_path: str | None
    value_preview: str | None


@dataclass(frozen=True)
class ArtifactCheck:
    path: Path
    relative_path: str
    sha256: str
    status: str
    fields: tuple[FieldCheck, ...]

    @property
    def artifact_id(self) -> str:
        return f"artifact-{self.sha256[:12]}"

    @property
    def missing_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if not field.present)


MANDATORY_EVIDENCE_FIELDS: tuple[EvidenceField, ...] = (
    EvidenceField("decision_artifact_id", ("decision_artifact_id", "decision_id", "artifact_id")),
    EvidenceField("decision_artifact_sha256", ("decision_artifact_sha256", "artifact_sha256", "sha256")),
    EvidenceField("generated_at_utc", ("generated_at_utc", "created_at_utc", "timestamp_utc", "started_at")),
    EvidenceField("candidate_model_id", ("candidate_model_id", "candidate_strategy.model_id", "challenger_strategy.model_id")),
    EvidenceField(
        "candidate_feature_pack_or_mode",
        (
            "candidate_feature_pack_or_mode",
            "candidate_strategy.feature_pack",
            "challenger_strategy.feature_pack",
            "decisions.0.blend_name",
        ),
    ),
    EvidenceField("control_model_id", ("control_model_id", "control_strategy.model_id")),
    EvidenceField(
        "control_feature_pack_or_mode",
        ("control_feature_pack_or_mode", "control_strategy.feature_pack"),
    ),
    EvidenceField("comparison_seasons", ("comparison_seasons", "promotion_seasons", "requested_seasons")),
    EvidenceField("budget_policy", ("budget_policy", "candidate_budget_policy", "source_budget_policy")),
    EvidenceField(
        "points_delta",
        (
            "points_delta",
            "candidate_vs_control_summary.total_actual_points_delta",
            "aggregate_actual_points_delta",
            "decisions.0.aggregate_delta",
        ),
    ),
    EvidenceField("budget_risk_checks", ("budget_risk_checks", "budget_risk_summary", "budget_deltas")),
    EvidenceField(
        "dnp_or_availability_checks",
        ("dnp_or_availability_checks", "availability_checks", "dnp_checks"),
    ),
    EvidenceField(
        "calibration_checks",
        ("calibration_checks", "gate_results.calibration_pass", "gate_results.selected_calibration_pass"),
    ),
    EvidenceField(
        "comparability_status",
        (
            "comparability_status",
            "source_comparability_status",
            "gate_results.comparability_ok",
            "fixture_identity_status",
        ),
    ),
    EvidenceField("final_decision", ("final_decision", "decision_status", "recommendation")),
    EvidenceField("final_decision_reason", ("final_decision_reason", "decision_reason", "recommendation")),
    EvidenceField("authority_refs", ("authority_refs", "authority_references", "authority_ids")),
    EvidenceField("source_artifact_refs", ("source_artifact_refs", "source_artifacts", "source_experiment_path")),
)


def discover_decision_artifacts(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        raise PromotionGateEvidenceError(f"Evidence root does not exist: {root}")
    if not root.is_dir():
        raise PromotionGateEvidenceError(f"Evidence root must be a directory: {root}")
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        name = path.name.lower()
        if name in {"promotion_gate_evidence_verification.json"}:
            continue
        if "decision" in name or name == "promotion_report.json":
            candidates.append(path)
    return tuple(candidates)


def scan_promotion_gate_evidence(root: Path, *, project_root: Path | None = None) -> tuple[ArtifactCheck, ...]:
    base = project_root or Path.cwd()
    checks = []
    for path in discover_decision_artifacts(root):
        checks.append(_scan_artifact(path, project_root=base))
    return tuple(checks)


def write_markdown_report(checks: tuple[ArtifactCheck, ...], output_path: Path, *, generated_at: datetime | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(checks, generated_at=generated_at), encoding="utf-8")
    return output_path


def _scan_artifact(path: Path, *, project_root: Path) -> ArtifactCheck:
    data = _read_json_object(path)
    fields = tuple(_check_field(data, field) for field in MANDATORY_EVIDENCE_FIELDS)
    relative_path = _relative(path, project_root)
    sha256 = _sha256(path)
    status = "pass" if all(field.present for field in fields) else "fail"
    return ArtifactCheck(path=path, relative_path=relative_path, sha256=sha256, status=status, fields=fields)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionGateEvidenceError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(data, dict):
        raise PromotionGateEvidenceError(f"JSON artifact must be an object: {path}")
    return cast("dict[str, object]", data)


def _check_field(data: dict[str, object], field: EvidenceField) -> FieldCheck:
    for source_path in field.source_paths:
        value = _nested_get(data, source_path)
        if _populated(value):
            return FieldCheck(
                name=field.name,
                present=True,
                source_path=source_path,
                value_preview=_preview(value),
            )
    return FieldCheck(name=field.name, present=False, source_path=None, value_preview=None)


def _nested_get(data: object, source_path: str) -> object:
    value = data
    for part in source_path.split("."):
        if isinstance(value, Mapping):
            mapping = cast("Mapping[str, object]", value)
            if part not in mapping:
                return None
            value = mapping[part]
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            if index >= len(value):
                return None
            value = value[index]
        else:
            return None
    return value


def _populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def _preview(value: object) -> str:
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, bool | int | float):
        return str(value)
    encoded = json.dumps(value, sort_keys=True, default=str)
    return encoded[:120]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _markdown_report(checks: tuple[ArtifactCheck, ...], *, generated_at: datetime | None = None) -> str:
    timestamp = (generated_at or datetime.now(UTC)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    pass_count = sum(1 for check in checks if check.status == "pass")
    fail_count = sum(1 for check in checks if check.status == "fail")
    lines = [
        "# Promotion Gate Evidence Verification",
        "",
        f"Generated at: `{timestamp}`",
        "",
        "This report scans existing frozen promotion decision artifacts against",
        "`data/08_reporting/governance/promotion_gate_evidence_contract.md`.",
        "",
        "The scan is read-only. It does not modify source artifacts and does not",
        "change the caRtola live default.",
        "",
        "## Summary",
        "",
        f"- Report generated at UTC: `{timestamp}`",
        "- Artifact identifiers are deterministic links derived from each scanned artifact SHA-256.",
        f"- Artifacts scanned: `{len(checks)}`",
        f"- Passing artifacts: `{pass_count}`",
        f"- Failing artifacts: `{fail_count}`",
        "",
        "## Artifact Results",
        "",
        "| Artifact ID | Artifact | Status | Missing Fields | SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in checks:
        missing = ", ".join(check.missing_fields) if check.missing_fields else "none"
        lines.append(
            f"| [`{check.artifact_id}`](#{check.artifact_id}) | `{check.relative_path}` | `{check.status}` | "
            f"{missing} | `{check.sha256}` |"
        )
    lines.extend(["", "## Field Details", ""])
    for check in checks:
        lines.extend(
            [
                f'<a id="{check.artifact_id}"></a>',
                f"### `{check.artifact_id}` - `{check.relative_path}`",
                "",
                f"- Artifact ID: `{check.artifact_id}`",
                f"- Status: `{check.status}`",
                f"- SHA-256: `{check.sha256}`",
                f"- Source artifact: `{check.relative_path}`",
                "",
                "| Field | Result | Source Path | Value Preview |",
                "| --- | --- | --- | --- |",
            ]
        )
        for field in check.fields:
            result = "present" if field.present else "missing"
            source_path = field.source_path or ""
            preview = (field.value_preview or "").replace("|", "\\|")
            lines.append(f"| `{field.name}` | `{result}` | `{source_path}` | {preview} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
