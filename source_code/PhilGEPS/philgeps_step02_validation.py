"""
Step 02 validation / QA checks for PhilGEPS preprocessing pipeline.

Used by 02_data_preprocessing_philgeps.py and pytest. Mirrors the step 01 contract:
each check returns a ValidationResult, and apply_validation_result raises Step02ValidationError
on hard failures (raise_on_error=True; default for the pipeline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column policy (single source of truth shared with the pipeline + tests)
# ---------------------------------------------------------------------------

KEEP_COLUMNS: tuple[str, ...] = (
    "Client Agency",
    "Region",
    "Province",
    "City/Municipality",
    "Government Branch",
    "PE Organization Type (Grouped)",
    "Year",
    "Notice Type",
    "Classification",
    "Procurement Mode",
    "Business Category",
    "Funding Source",
    "Funding Instrument",
    "Trade Agreement",
    "Approved Budget of the Contract",
    "Published Date",
    "Closing Date",
    "PreBid Date",
    "Area of Delivery",
    "Contract Duration",
    "Calendar Type",
    "Notice Status",
    "Bid Notice Status",
    "Quantity",
    "UOM",
    "Item Budget",
    "Award Type",
    "Award Notice Status",
    "Published Date(Award)",
    "Award Date",
    "Notice to Proceed Date",
    "Contract Efectivity Date",
    "Contract End Date",
    "Contract Effectivity Date",
    "Contract Amount",
    "Award Status",
    "Country of Awardee",
    "Region of Awardee",
    "Province of Awardee",
    "City/Municipality of Awardee",
    "Awardee Size",
)

DROP_COLUMNS: tuple[str, ...] = (
    "Procuring Entity",
    "Procuring Entity (PE)",
    "Bid Reference No.",
    "Solicitation No.",
    "Notice Title",
    "Created By",
    "Line Item No",
    "Line Item No.",
    "Item Name",
    "Item Description",
    "UNSPSC Code",
    "UNSPSC Description",
    "Award No.",
    "Award Title",
    "Award Reference No.",
    "Contract No",
    "Contract No.",
    "Reason for Award",
    "Awardee Organization Name",
    "Awardee Contact Person",
    "PE Organization Type",
    "Awardee Joint Venture",
)

MONEY_COLUMNS: tuple[str, ...] = (
    "Approved Budget of the Contract",
    "Item Budget",
    "Contract Amount",
)

DATE_COLUMNS: tuple[str, ...] = (
    "Published Date",
    "Closing Date",
    "PreBid Date",
    "Published Date(Award)",
    "Award Date",
    "Notice to Proceed Date",
    "Contract Efectivity Date",
    "Contract End Date",
    "Contract Effectivity Date",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class Step02ValidationError(RuntimeError):
    """Raised when step 02 post-condition checks fail."""


# ---------------------------------------------------------------------------
# Sanity checks (Layer A: feature selection / engineering)
# ---------------------------------------------------------------------------


def validate_kept_columns_exist(df: pd.DataFrame, expected: Iterable[str] = KEEP_COLUMNS) -> ValidationResult:
    r = ValidationResult()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        r.errors.append(f"Kept columns missing from input: {missing}")
    return r


def validate_no_dropped_columns(df: pd.DataFrame, drop_set: Iterable[str] = DROP_COLUMNS) -> ValidationResult:
    r = ValidationResult()
    leaked = [c for c in drop_set if c in df.columns]
    if leaked:
        r.errors.append(f"Drop-list columns still present after selection: {leaked}")
    return r


def validate_no_nulls(df: pd.DataFrame, columns: Iterable[str]) -> ValidationResult:
    r = ValidationResult()
    cols = [c for c in columns if c in df.columns]
    nulls = df[cols].isna().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        r.errors.append(
            "Null values still present after selection: "
            + ", ".join(f"{k!r}={int(v):,}" for k, v in bad.items()),
        )
    return r


def validate_dates_parseable(
    df: pd.DataFrame,
    columns: Iterable[str] = DATE_COLUMNS,
    *,
    min_ratio: float = 0.95,
) -> ValidationResult:
    """Each date-like column must parse to >= min_ratio of non-null rows.

    Step 01 wrote some dates as ``dd/mm/yyyy`` and others as ISO timestamps in the same
    column; we mirror the pipeline's ``format='mixed', dayfirst=True`` parse so the
    validator reflects what the pipeline can actually consume.
    """
    import warnings as _warnings  # local import to keep top-level light

    r = ValidationResult()
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        n_non_null_in = int(s.notna().sum())
        if n_non_null_in == 0:
            continue
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
        n_parsed = int(parsed.notna().sum())
        ratio = n_parsed / max(n_non_null_in, 1)
        if ratio < min_ratio:
            r.warnings.append(
                f"Date column {c!r}: only {n_parsed:,}/{n_non_null_in:,} "
                f"({ratio:.1%}) values parsed (< {min_ratio:.0%})",
            )
    return r


def validate_money_nonnegative(df: pd.DataFrame, columns: Iterable[str] = MONEY_COLUMNS) -> ValidationResult:
    r = ValidationResult()
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        neg = int((s < 0).sum())
        if neg:
            r.warnings.append(f"Money column {c!r}: {neg:,} negative values (will be clipped to 0)")
    return r


# ---------------------------------------------------------------------------
# Layer B: One-hot encoding
# ---------------------------------------------------------------------------


def validate_one_hot_partition(
    original_series: pd.Series,
    dummy_frame: pd.DataFrame,
    *,
    drop_first: bool = True,
) -> ValidationResult:
    """
    For a single source categorical column and its dummy block, every row must sum to
    1 (no drop_first) or {0, 1} (drop_first=True; reference category encoded as all-zero).
    """
    r = ValidationResult()
    if dummy_frame.empty:
        r.errors.append("Dummy frame is empty")
        return r

    if dummy_frame.isna().any().any():
        r.errors.append(f"Dummy frame for {original_series.name!r} contains NaN")

    row_sums = dummy_frame.sum(axis=1)
    unique = pd.unique(row_sums)
    if drop_first:
        bad = [u for u in unique if u not in (0, 1)]
        if bad:
            r.errors.append(
                f"OHE row sums for {original_series.name!r} must be in {{0,1}} "
                f"(drop_first=True); found {bad[:5]}",
            )
    else:
        bad = [u for u in unique if u != 1]
        if bad:
            r.errors.append(
                f"OHE row sums for {original_series.name!r} must equal 1 "
                f"(drop_first=False); found {bad[:5]}",
            )
    return r


def validate_no_constant_dummies(dummy_frame: pd.DataFrame, *, min_rate: float = 0.0) -> ValidationResult:
    """Warn when a dummy column is all-zero (or all-one), which adds no signal to k-means."""
    r = ValidationResult()
    if dummy_frame.empty:
        return r
    means = dummy_frame.mean(axis=0)
    flat = means[(means <= min_rate) | (means >= 1.0 - min_rate)]
    if not flat.empty:
        r.warnings.append(
            f"Dummies with constant value (rate <= {min_rate} or >= {1 - min_rate}): "
            f"{list(flat.index)[:10]} (showing first 10 of {len(flat):,})",
        )
    return r


# ---------------------------------------------------------------------------
# Layer C: Min-max scaling
# ---------------------------------------------------------------------------


def validate_scaled_in_unit_interval(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    eps: float = 1e-9,
) -> ValidationResult:
    r = ValidationResult()
    cols = [c for c in columns if c in df.columns]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n_bad_nan = int(s.isna().sum())
        if n_bad_nan:
            r.errors.append(f"Scaled column {c!r}: {n_bad_nan:,} NaN/non-numeric")
            continue
        smin = float(s.min())
        smax = float(s.max())
        if smin < -eps or smax > 1.0 + eps:
            r.errors.append(
                f"Scaled column {c!r} out of [0, 1]: min={smin:.6g}, max={smax:.6g}",
            )
    return r


def validate_finite_numeric(df: pd.DataFrame, columns: Iterable[str]) -> ValidationResult:
    r = ValidationResult()
    cols = [c for c in columns if c in df.columns]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(s.to_numpy()).all():
            n_bad = int((~np.isfinite(s.to_numpy())).sum())
            r.errors.append(f"Numeric column {c!r}: {n_bad:,} non-finite values (NaN/inf)")
    return r


# ---------------------------------------------------------------------------
# CSV existence / row-count
# ---------------------------------------------------------------------------


def count_csv_data_rows(path: str) -> int:
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def validate_written_csv(path: str, expected_rows: int, *, label: str) -> ValidationResult:
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


# ---------------------------------------------------------------------------
# Composite checks
# ---------------------------------------------------------------------------


def run_layer_a_validations(df: pd.DataFrame) -> ValidationResult:
    """Layer A — feature selection / engineering input contract."""
    out = ValidationResult()
    out.merge(validate_kept_columns_exist(df))
    out.merge(validate_no_nulls(df, KEEP_COLUMNS))
    out.merge(validate_dates_parseable(df))
    out.merge(validate_money_nonnegative(df))
    return out


def run_layer_c_validations(df: pd.DataFrame, scaled_columns: Iterable[str]) -> ValidationResult:
    """Layer C — min-max scaling output contract."""
    out = ValidationResult()
    out.merge(validate_finite_numeric(df, scaled_columns))
    out.merge(validate_scaled_in_unit_interval(df, scaled_columns))
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
        raise Step02ValidationError("; ".join(result.errors))
