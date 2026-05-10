from __future__ import annotations

import pytest

from cartola.backtesting.ebm_feature_diagnostic import (
    EbmDependencyError,
    inspect_ebm_runtime,
)


class _FakeEbm:
    def __init__(
        self,
        *,
        interactions: int = 0,
        validation_size: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.interactions = interactions
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, x_values: object, y_values: object) -> "_FakeEbm":
        return self


class _FakeEbmWithValidation:
    def fit(
        self,
        x_values: object,
        y_values: object,
        X_val: object,
        y_val: object,
    ) -> "_FakeEbmWithValidation":
        return self


class _FakeEbmWithValidationNameSubstrings:
    def fit(
        self,
        x_values: object,
        y_values: object,
        not_X_val: object,
        not_y_val: object,
    ) -> "_FakeEbmWithValidationNameSubstrings":
        return self


def test_inspect_ebm_runtime_records_constructor_and_fit_signatures() -> None:
    info = inspect_ebm_runtime(ebm_class=_FakeEbm, package_version="9.9.9")

    assert info.available is True
    assert info.version == "9.9.9"
    assert "validation_size" in info.constructor_signature
    assert "x_values" in info.fit_signature
    assert info.supports_explicit_validation is False


def test_inspect_ebm_runtime_detects_explicit_validation_parameters() -> None:
    info = inspect_ebm_runtime(
        ebm_class=_FakeEbmWithValidation,
        package_version="9.9.9",
    )

    assert info.supports_explicit_validation is True


def test_inspect_ebm_runtime_ignores_validation_name_substrings() -> None:
    info = inspect_ebm_runtime(
        ebm_class=_FakeEbmWithValidationNameSubstrings,
        package_version="9.9.9",
    )

    assert info.supports_explicit_validation is False


def test_inspect_ebm_runtime_raises_clear_error_when_missing() -> None:
    with pytest.raises(EbmDependencyError, match="InterpretML is required"):
        inspect_ebm_runtime(ebm_class=None, package_version=None)
