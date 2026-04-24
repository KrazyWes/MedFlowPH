"""Unit tests for step 01 validation helpers (no full raw dataset required)."""

from __future__ import annotations

import pandas as pd
import pytest

from philgeps_step01_validation import (
    Step01ValidationError,
    apply_validation_result,
    run_post_imputation_validations,
    validate_after_concat,
    validate_medical_keyword_coverage,
    validate_no_duplicate_rows,
    validate_numeric_imputation_no_nulls,
    validate_raw_row_accounting,
    validate_written_csv,
    validate_year_partition,
)

# Subset of step 01 keyword pattern for tests
KW = r"medical|health|hospital"


def test_raw_row_accounting_ok() -> None:
    stats = {"input_rows": 100, "output_rows": 10}
    raw_by = {2020: 40, 2021: 60}
    r = validate_raw_row_accounting(stats, 100, raw_by)
    assert r.ok


def test_raw_row_accounting_mismatch() -> None:
    stats = {"input_rows": 99, "output_rows": 10}
    raw_by = {2020: 40, 2021: 60}
    r = validate_raw_row_accounting(stats, 100, raw_by)
    assert not r.ok
    assert any("input_rows" in e for e in r.errors)


def test_after_concat_ok() -> None:
    r = validate_after_concat({"output_rows": 50}, 30, 30)
    assert r.ok


def test_after_concat_length_mismatch() -> None:
    r = validate_after_concat({"output_rows": 50}, 30, 29)
    assert not r.ok


def test_no_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert validate_no_duplicate_rows(df).ok
    df2 = pd.DataFrame({"a": [1, 1]})
    assert not validate_no_duplicate_rows(df2).ok


def test_year_partition() -> None:
    df = pd.DataFrame({"Year": [2020, 2020, 2021]})
    assert validate_year_partition(df).ok
    df2 = pd.DataFrame({"x": [1]})
    assert not validate_year_partition(df2).ok


def test_medical_keyword_coverage() -> None:
    df = pd.DataFrame(
        {
            "UNSPSC Description": ["office supplies", "x"],
            "Item Name": ["a", "b"],
            "Item Description": ["b", "health clinic"],
        },
    )
    r = validate_medical_keyword_coverage(df, KW)
    assert not r.ok
    df2 = pd.DataFrame(
        {
            "UNSPSC Description": ["medical supplies", "medical kit"],
            "Item Name": ["a", "b"],
            "Item Description": ["b", "c"],
        },
    )
    assert validate_medical_keyword_coverage(df2, KW).ok


def test_numeric_no_nulls_after_fill() -> None:
    df = pd.DataFrame({"Contract Amount": [1.0, 2.0], "Item Budget": [3.0, None]})
    assert not validate_numeric_imputation_no_nulls(df).ok
    df["Item Budget"] = df["Item Budget"].fillna(0.0)
    assert validate_numeric_imputation_no_nulls(df).ok


def test_written_csv_roundtrip(tmp_path) -> None:
    p = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    r = validate_written_csv(str(p), 3, label="test")
    assert r.ok
    r2 = validate_written_csv(str(p), 2, label="test")
    assert not r2.ok


def test_apply_raises() -> None:
    from philgeps_step01_validation import ValidationResult

    bad = ValidationResult(errors=["x"])
    with pytest.raises(Step01ValidationError):
        apply_validation_result(bad, None, raise_on_error=True)


def test_run_post_imputation_minimal_ok() -> None:
    df = pd.DataFrame(
        {
            "Year": [2020, 2021],
            "UNSPSC Description": ["medical goods", "health supplies"],
            "Item Name": ["a", "b"],
            "Item Description": ["c", "d"],
            "Contract Amount": [1.0, 2.0],
            "Item Budget": [1.0, 2.0],
            "Quantity": [1, 2],
            "Approved Budget of the Contract": [1.0, 2.0],
            "Line Item No": [1, 2],
        },
    )
    r = run_post_imputation_validations(df, KW)
    assert r.ok
