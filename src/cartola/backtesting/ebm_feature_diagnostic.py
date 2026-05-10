from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


class EbmDependencyError(RuntimeError):
    """Raised when InterpretML is unavailable or incompatible."""


@dataclass(frozen=True)
class EbmRuntimeInfo:
    available: bool
    version: str | None
    constructor_signature: str
    fit_signature: str
    supports_explicit_validation: bool


def inspect_ebm_runtime(
    *, ebm_class: type[Any] | None, package_version: str | None
) -> EbmRuntimeInfo:
    if ebm_class is None:
        raise EbmDependencyError(
            "InterpretML is required for EBM diagnostics. Install the optional diagnostic dependencies."
        )
    constructor_signature = str(inspect.signature(ebm_class))
    fit_inspection = inspect.signature(ebm_class.fit)
    fit_signature = str(fit_inspection)
    fit_parameters = fit_inspection.parameters
    supports_explicit_validation = "X_val" in fit_parameters and "y_val" in fit_parameters
    return EbmRuntimeInfo(
        available=True,
        version=package_version,
        constructor_signature=constructor_signature,
        fit_signature=fit_signature,
        supports_explicit_validation=supports_explicit_validation,
    )
