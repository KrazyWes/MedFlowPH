"""
Step 01 validation / QA checks for PhilGEPS medical cleaning pipeline.

Used by 01_data_cleaning_philgeps.py and pytest. Raises Step01ValidationError on hard failures
when raise_on_error=True (default for pipeline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

# Minimum columns expected in cleaned medical output (subset of canonical export)
REQUIRED_CLEANED_COLUMNS = (
    "Year",
    "UNSPSC Description",
    "Item Name",
    "Item Description",
)

NUMERIC_IMPUTE_COLUMNS = (
    "Contract Amount",
    "Item Budget",
    "Quantity",
    "Approved Budget of the Contract",
    "Line Item No",
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class Step01ValidationError(RuntimeError):
    """Raised when step 01 post-condition checks fail."""


def validate_raw_row_accounting(
    med_stats_total: dict[str, int] | None,
    total_raw_rows: int,
    raw_rows_by_year: dict[int, int],
) -> ValidationResult:
    r = ValidationResult()
    if med_stats_total is None:
        r.errors.append("med_stats_total is None")
        return r
    if med_stats_total.get("input_rows", -1) != total_raw_rows:
        r.errors.append(
            f"Raw row accounting: med_stats_total['input_rows']={med_stats_total.get('input_rows')!r} "
            f"!= total_raw_rows={total_raw_rows}",
        )
    summed = sum(raw_rows_by_year.values())
    if summed != total_raw_rows:
        r.errors.append(
            f"Raw row accounting: sum(raw_rows_by_year)={summed} != total_raw_rows={total_raw_rows}",
        )
    return r


def validate_medical_chunks_nonempty(rows_after_chunk_dedup: int) -> ValidationResult:
    r = ValidationResult()
    if rows_after_chunk_dedup <= 0:
        r.errors.append("No medical rows after within-file dedup (rows_after_chunk_dedup <= 0)")
    return r


def validate_after_concat(
    med_stats_total: dict[str, int],
    rows_after_chunk_dedup: int,
    len_after_concat: int,
) -> ValidationResult:
    r = ValidationResult()
    if len_after_concat != rows_after_chunk_dedup:
        r.errors.append(
            f"Concat length {len_after_concat} != sum of chunk sizes after within-file dedup "
            f"{rows_after_chunk_dedup}",
        )
    if len_after_concat > med_stats_total["output_rows"]:
        r.errors.append(
            f"Concat size {len_after_concat} exceeds sum of medical filter outputs pre chunk-dedup "
            f"{med_stats_total['output_rows']}",
        )
    return r


def validate_no_duplicate_rows(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    dups = int(df.duplicated().sum())
    if dups:
        r.errors.append(f"Found {dups:,} duplicate rows after final drop_duplicates()")
    return r


def validate_year_partition(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    if "Year" not in df.columns:
        r.errors.append("Column 'Year' missing")
        return r
    gsum = int(df.groupby("Year", sort=False).size().sum())
    if gsum != len(df):
        r.errors.append(
            f"Year partition: groupby sizes sum {gsum} != len(df) {len(df)}",
        )
    return r


def validate_key_columns(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    for c in REQUIRED_CLEANED_COLUMNS:
        if c not in df.columns:
            r.errors.append(f"Required column missing: {c!r}")
    return r


def validate_numeric_imputation_no_nulls(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    for col in NUMERIC_IMPUTE_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(series.isna().sum())
        if n_bad:
            r.errors.append(f"Numeric column {col!r} still has {n_bad:,} null/non-numeric after imputation")
    return r


def validate_medical_keyword_coverage(df: pd.DataFrame, pattern: str) -> ValidationResult:
    """Every row must match UNSPSC or Item Name or Item Description keyword rule (same as filter)."""
    r = ValidationResult()
    if df.empty:
        return r
    col_u = "UNSPSC Description"
    col_in = "Item Name"
    col_id = "Item Description"
    unspsc_match = (
        df[col_u].astype(str).str.contains(pattern, case=False, na=False)
        if col_u in df.columns
        else pd.Series(False, index=df.index)
    )
    in_match = (
        df[col_in].astype(str).str.contains(pattern, case=False, na=False)
        if col_in in df.columns
        else pd.Series(False, index=df.index)
    )
    id_match = (
        df[col_id].astype(str).str.contains(pattern, case=False, na=False)
        if col_id in df.columns
        else pd.Series(False, index=df.index)
    )
    covered = unspsc_match | in_match | id_match
    n_bad = int((~covered).sum())
    if n_bad:
        r.errors.append(
            f"Medical keyword coverage: {n_bad:,} rows do not match UNSPSC/Item Name/Item Description pattern",
        )
    return r


def validate_contract_amount_non_negative(df: pd.DataFrame) -> ValidationResult:
    r = ValidationResult()
    if "Contract Amount" not in df.columns or df.empty:
        return r
    s = pd.to_numeric(df["Contract Amount"], errors="coerce")
    neg = int((s < 0).sum())
    if neg:
        r.warnings.append(f"Contract Amount: {neg:,} negative values (review if unexpected)")
    return r


def count_csv_data_rows(path: str) -> int:
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def validate_written_csv(
    path: str,
    expected_rows: int,
    *,
    label: str,
) -> ValidationResult:
    r = ValidationResult()
    if not os.path.isfile(path):
        r.errors.append(f"{label}: file missing: {path}")
        return r
    if os.path.getsize(path) == 0:
        r.errors.append(f"{label}: file empty: {path}")
        return r
    try:
        n = count_csv_data_rows(path)
    except OSError as e:
        r.errors.append(f"{label}: cannot read {path}: {e}")
        return r
    if n != expected_rows:
        r.errors.append(f"{label}: row count mismatch file={n:,} expected={expected_rows:,} ({path})")
    return r


def run_post_imputation_validations(
    medical_df: pd.DataFrame,
    pattern: str,
) -> ValidationResult:
    """All checks that apply to the final in-memory frame before writing CSVs."""
    out = ValidationResult()
    out.merge(validate_key_columns(medical_df))
    out.merge(validate_no_duplicate_rows(medical_df))
    out.merge(validate_year_partition(medical_df))
    out.merge(validate_numeric_imputation_no_nulls(medical_df))
    out.merge(validate_medical_keyword_coverage(medical_df, pattern))
    out.merge(validate_contract_amount_non_negative(medical_df))
    return out


def run_output_file_validations(
    output_combined: str,
    yearly_paths: list[tuple[str, int]],
) -> ValidationResult:
    """Validate combined + per-year CSV row counts."""
    out = ValidationResult()
    total_expected = sum(n for _, n in yearly_paths)
    out.merge(validate_written_csv(output_combined, total_expected, label="Combined CSV"))
    for ypath, nexp in yearly_paths:
        out.merge(validate_written_csv(ypath, nexp, label=f"Yearly CSV {os.path.basename(ypath)}"))
    return out


def apply_validation_result(
    result: ValidationResult,
    log_fn: Callable[[str], None] | None,
    *,
    raise_on_error: bool = True,
) -> None:
    for w in result.warnings:
        msg = f"VALIDATION WARNING: {w}"
        if log_fn:
            log_fn(msg)
        print(msg, flush=True)
    for e in result.errors:
        msg = f"VALIDATION ERROR: {e}"
        if log_fn:
            log_fn(msg)
        print(msg, flush=True)
    if raise_on_error and result.errors:
        raise Step01ValidationError("; ".join(result.errors))
