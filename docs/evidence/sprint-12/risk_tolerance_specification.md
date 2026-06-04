# Risk Tolerance Specification

Tracked copy metadata:

- Original generated path: `data/08_reporting/governance/risk_tolerance_specification.md`
- Related approved options artifact: `data/08_reporting/governance/risk_tolerance_options.json`
- Approved options artifact SHA256: `452a712f895acef6771975f9fba0c665b777c07cffe693fabfdcc3ecd194ded6`
- Approved option set: `risk_balanced_review_option`
- Approval scope: review/configuration documentation only.
- Runtime enforcement: disabled.
- Recommendation rejection: disabled.

Generated for Product Owner review on `2026-06-04`.

## Purpose

This document defines proposed risk-tolerance fields and value ranges for
caRtola live recommendation governance. It is an approval artifact, not a
runtime enforcement artifact.

The current backlog slice must formalize options for review. It must not turn
captain-floor, budget-drawdown, or DNP-exposure thresholds into automatic
rejection gates until a Product Owner approval artifact accepts those thresholds
and a future implementation slice wires enforcement into the live workflow.

## Authority References

- `OPEN_QUESTION.risk-thresholds`: accepted DNP, budget drawdown, and
  captain-risk thresholds remain unresolved.
- `REQ.default-promotion-gate`: live default changes require frozen promotion
  evidence.
- `DATA.promotion-decision`: promotion decisions must record budget-risk and
  DNP or availability checks when applicable.

## Approval State

Current status: `proposed_options_only`.

Required approval metadata before enforcement:

| Field | Required Content |
| --- | --- |
| `approval_status` | One of `draft`, `po_approved`, `rejected`, or `superseded`. |
| `approved_option_set_id` | Stable identifier for the accepted risk option set. |
| `approved_by` | Product Owner identity or role approving the option set. |
| `approved_at_utc` | UTC approval timestamp. |
| `approval_notes` | Human-readable rationale and known tradeoffs. |
| `authority_refs` | At minimum `OPEN_QUESTION.risk-thresholds`, plus any accepted future enforcement authority. |

## Risk Tolerance Fields

| Field | Type | Proposed Range | Default Before Approval | Meaning |
| --- | --- | --- | --- | --- |
| `max_dnp_allowed` | integer or null | `0` to `3` | `null` | Maximum selected-player DNP count tolerated by a future gate. `null` means no approved DNP gate. |
| `max_budget_drawdown_percent` | number or null | `0.0` to `100.0` | `null` | Maximum budget drawdown tolerated relative to the operator-provided live budget. `null` means no approved drawdown gate. |
| `captain_floor_points` | number or null | `0.0` to `20.0` | `null` | Minimum raw predicted captain points tolerated by a future captain-risk gate. `null` means no approved captain floor. |
| `captain_risk_preference` | string | `safe`, `balanced`, `aggressive`, or `custom` | `custom` | Named PO preference used to select or explain captain floor options. |

## Proposed Option Sets

These values are proposed review options only. They are not live enforcement
thresholds until an approval artifact records `approval_status=po_approved`.

| Option Set | `captain_risk_preference` | `captain_floor_points` | `max_budget_drawdown_percent` | `max_dnp_allowed` | Planning Intent |
| --- | --- | --- | --- | --- | --- |
| `risk_safe_review_option` | `safe` | `8.0` | `5.0` | `0` | Minimize availability and budget volatility, even if projected upside drops. |
| `risk_balanced_review_option` | `balanced` | `6.0` | `10.0` | `1` | Balance upside with moderate availability and budget protection. |
| `risk_aggressive_review_option` | `aggressive` | `0.0` | `20.0` | `2` | Allow higher upside and volatility when the operator accepts the risk. |
| `risk_custom_review_option` | `custom` | PO-defined | PO-defined | PO-defined | Preserve Product Owner discretion when named bands are insufficient. |

## Captain Preference Mapping

`captain_risk_preference` maps to `captain_floor_points` only inside the
reviewed option set:

| Preference | Proposed Floor |
| --- | --- |
| `safe` | `8.0` |
| `balanced` | `6.0` |
| `aggressive` | `0.0` |
| `custom` | Explicit PO-approved value in range `0.0` to `20.0`. |

## Runtime Boundary

Until PO approval exists, the live recommendation workflow may record these
fields as proposed options, display them in audit output, or use them in
discovery analysis. It must not fail a recommendation, reject a squad, or
change the live default solely because one of these proposed values is exceeded.

Future enforcement must cite the approved option set, the approval metadata, and
the accepted authority item that makes enforcement part of runtime scope.
