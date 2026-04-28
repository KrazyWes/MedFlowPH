"""Unit tests for step 02 preprocessing validation helpers (no full dataset required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from philgeps_step02_validation import (
    DATE_COLUMNS,
    DROP_COLUMNS,
    KEEP_COLUMNS,
    Step02ValidationError,
    ValidationResult,
    apply_validation_result,
    validate_dates_parseable,
    validate_finite_numeric,
    validate_kept_columns_exist,
    validate_money_nonnegative,
    validate_no_constant_dummies,
    validate_no_dropped_columns,
    validate_no_nulls,
    validate_one_hot_partition,
    validate_scaled_in_unit_interval,
    validate_written_csv,
)


# ---------------------------------------------------------------------------
# Layer A: input contract
# ---------------------------------------------------------------------------


def _minimal_keep_frame() -> pd.DataFrame:
    """Tiny frame with all KEEP_COLUMNS present and 0 nulls."""
    n = 4
    data = {}
    for c in KEEP_COLUMNS:
        if c in DATE_COLUMNS:
            data[c] = pd.to_datetime(["2024-01-15"] * n)
        elif c in (
            "Year", "Contract Duration", "Quantity",
            "Approved Budget of the Contract", "Item Budget", "Contract Amount",
        ):
            data[c] = [1.0] * n
        else:
            data[c] = ["x"] * n
    return pd.DataFrame(data)


def test_kept_columns_exist_ok() -> None:
    r = validate_kept_columns_exist(_minimal_keep_frame())
    assert r.ok


def test_kept_columns_missing() -> None:
    df = _minimal_keep_frame().drop(columns=["Region"])
    r = validate_kept_columns_exist(df)
    assert not r.ok
    assert any("Region" in e for e in r.errors)


def test_no_dropped_columns_clean() -> None:
    df = _minimal_keep_frame()
    r = validate_no_dropped_columns(df)
    assert r.ok


def test_no_dropped_columns_leak() -> None:
    df = _minimal_keep_frame().assign(**{"Awardee Joint Venture": ["jv"] * 4})
    r = validate_no_dropped_columns(df)
    assert not r.ok
    assert any("Awardee Joint Venture" in e for e in r.errors)


def test_no_nulls_ok_and_fail() -> None:
    df = _minimal_keep_frame()
    r = validate_no_nulls(df, ["Region", "Year"])
    assert r.ok

    df.loc[0, "Region"] = None
    r2 = validate_no_nulls(df, ["Region", "Year"])
    assert not r2.ok
    assert any("Region" in e for e in r2.errors)


def test_dates_parseable_warning() -> None:
    df = pd.DataFrame(
        {
            "Published Date": ["2024-01-15", "garbage"] * 50,
            "Closing Date": ["2024-01-20"] * 100,
        },
    )
    r = validate_dates_parseable(df, columns=("Published Date", "Closing Date"))
    # Warnings, not errors -> still .ok
    assert r.ok
    assert any("Published Date" in w for w in r.warnings)


def test_money_nonnegative() -> None:
    df = pd.DataFrame(
        {
            "Approved Budget of the Contract": [100.0, -1.0, 0.0],
            "Item Budget": [10.0, 20.0, 30.0],
            "Contract Amount": [10.0, 20.0, 30.0],
        },
    )
    r = validate_money_nonnegative(df)
    assert r.ok  # warnings only
    assert any("negative" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Layer B: OHE
# ---------------------------------------------------------------------------


def test_ohe_partition_drop_first_ok() -> None:
    src = pd.Series(["A", "B", "C", "A"], name="X")
    dummies = pd.get_dummies(src, drop_first=True, dtype=np.uint8)
    r = validate_one_hot_partition(src, dummies, drop_first=True)
    assert r.ok


def test_ohe_partition_no_drop_first_ok() -> None:
    src = pd.Series(["A", "B"], name="X")
    dummies = pd.get_dummies(src, drop_first=False, dtype=np.uint8)
    r = validate_one_hot_partition(src, dummies, drop_first=False)
    assert r.ok


def test_ohe_partition_bad_sums() -> None:
    src = pd.Series(["A", "B"], name="X")
    bad = pd.DataFrame({"X_A": [1, 1], "X_B": [1, 1]}, dtype=np.uint8)
    r = validate_one_hot_partition(src, bad, drop_first=True)
    assert not r.ok


def test_constant_dummies_warn() -> None:
    df = pd.DataFrame({"a": [0, 0, 0], "b": [1, 1, 1], "c": [0, 1, 0]}, dtype=np.uint8)
    r = validate_no_constant_dummies(df, min_rate=0.0)
    assert r.ok  # warnings only
    assert any("constant" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Layer C: scaling
# ---------------------------------------------------------------------------


def test_scaled_in_unit_interval_ok() -> None:
    df = pd.DataFrame({"a": [0.0, 0.5, 1.0], "b": [0.1, 0.2, 0.3]})
    r = validate_scaled_in_unit_interval(df, columns=["a", "b"])
    assert r.ok


def test_scaled_out_of_range() -> None:
    df = pd.DataFrame({"a": [0.0, 1.5], "b": [-0.1, 1.0]})
    r = validate_scaled_in_unit_interval(df, columns=["a", "b"])
    assert not r.ok


def test_finite_numeric_catches_nan_inf() -> None:
    df = pd.DataFrame({"a": [0.0, np.nan, 1.0], "b": [1.0, np.inf, 0.0]})
    r = validate_finite_numeric(df, columns=["a", "b"])
    assert not r.ok


# ---------------------------------------------------------------------------
# CSV / dispatch
# ---------------------------------------------------------------------------


def test_written_csv_roundtrip(tmp_path) -> None:
    p = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    r = validate_written_csv(str(p), 3, label="test")
    assert r.ok

    r2 = validate_written_csv(str(p), 2, label="test")
    assert not r2.ok


def test_apply_raises() -> None:
    bad = ValidationResult(errors=["x"])
    with pytest.raises(Step02ValidationError):
        apply_validation_result(bad, None, raise_on_error=True)
