"""
Step 03 validation / QA checks for PhilGEPS k-means clustering pipeline.

Used by 03_kmeans_implementation_philgeps.py and pytest. Mirrors the step 02 contract:
each check returns a ValidationResult, and apply_validation_result raises Step03ValidationError
on hard failures (raise_on_error=True; default for the pipeline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


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


class Step03ValidationError(RuntimeError):
    """Raised when step 03 post-condition checks fail."""


# ---------------------------------------------------------------------------
# Input matrix checks
# ---------------------------------------------------------------------------


def validate_input_matrix(
    df: pd.DataFrame,
    *,
    min_columns: int = 50,
    eps: float = 1e-9,
) -> ValidationResult:
    """The step 02 preprocessed matrix must be wide, finite, and in [0, 1]."""
    r = ValidationResult()
    if df.empty:
        r.errors.append("Preprocessed matrix is empty")
        return r
    if df.shape[1] < min_columns:
        r.errors.append(
            f"Preprocessed matrix has only {df.shape[1]} columns (expected >= {min_columns})",
        )
    arr = df.to_numpy()
    if arr.dtype.kind not in {"f", "i", "u"}:
        r.errors.append(f"Preprocessed matrix has non-numeric dtype: {arr.dtype}")
        return r
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        r.errors.append(f"Preprocessed matrix has {n_bad:,} non-finite values (NaN/inf)")
    smin = float(np.nanmin(arr))
    smax = float(np.nanmax(arr))
    if smin < -eps or smax > 1.0 + eps:
        r.errors.append(
            f"Preprocessed matrix out of [0, 1]: min={smin:.6g}, max={smax:.6g}",
        )
    return r


# ---------------------------------------------------------------------------
# PCA checks
# ---------------------------------------------------------------------------


def validate_pca_variance_threshold(
    explained_variance_ratio: Sequence[float],
    *,
    target: float = 0.05,
    n_components: int = 3,
) -> ValidationResult:
    """Warn (do not error) if the first ``n_components`` PCs do not reach ``target``."""
    r = ValidationResult()
    ratios = list(explained_variance_ratio)
    if len(ratios) < n_components:
        r.errors.append(
            f"PCA produced only {len(ratios)} components (expected >= {n_components})",
        )
        return r
    cum = float(np.sum(ratios[:n_components]))
    if cum + 1e-9 < target:
        r.warnings.append(
            f"First {n_components} PCs explain only {cum:.4f} cumulative variance "
            f"(below configured floor {target:.0%}). "
            "Common in high-dimensional sparse/one-hot inputs; 3D scatter omits most variance along higher PCs.",
        )
    return r


# ---------------------------------------------------------------------------
# Silhouette / sweep checks
# ---------------------------------------------------------------------------


def validate_silhouette_curve(
    k_to_score: dict[int, float],
) -> ValidationResult:
    r = ValidationResult()
    if not k_to_score:
        r.errors.append("Silhouette sweep produced no results")
        return r
    bad = {k: v for k, v in k_to_score.items() if not np.isfinite(v)}
    if bad:
        r.errors.append(f"Silhouette score is non-finite for K={list(bad.keys())}")
    distinct = len({round(float(v), 6) for v in k_to_score.values() if np.isfinite(v)})
    if distinct == 1 and len(k_to_score) > 1:
        r.warnings.append(
            "Silhouette score is identical across all K — verify variance/standardization "
            "and that K-Means actually converged.",
        )
    return r


# ---------------------------------------------------------------------------
# Cluster assignment / centroid checks
# ---------------------------------------------------------------------------


def validate_clusters_assigned_all_rows(
    labels: np.ndarray,
    n_rows: int,
) -> ValidationResult:
    r = ValidationResult()
    if labels is None:
        r.errors.append("Cluster labels array is None")
        return r
    if labels.shape[0] != n_rows:
        r.errors.append(
            f"Cluster labels length {labels.shape[0]:,} != input rows {n_rows:,}",
        )
    if not np.issubdtype(labels.dtype, np.integer):
        r.errors.append(f"Cluster labels must be integer dtype (got {labels.dtype})")
    if (labels < 0).any():
        r.errors.append("Cluster labels contain negative values (unassigned rows)")
    return r


def validate_no_empty_cluster(labels: np.ndarray, k: int) -> ValidationResult:
    r = ValidationResult()
    counts = np.bincount(labels.astype(int), minlength=k)
    empty = [i for i, c in enumerate(counts) if c == 0]
    if empty:
        r.errors.append(f"Empty clusters detected: {empty} (k={k})")
    return r


def validate_centroids_shape(
    centroids: np.ndarray,
    *,
    k: int,
    n_features: int,
) -> ValidationResult:
    r = ValidationResult()
    if centroids is None:
        r.errors.append("Centroids array is None")
        return r
    if centroids.shape != (k, n_features):
        r.errors.append(
            f"Centroids shape {centroids.shape} != expected ({k}, {n_features})",
        )
    if not np.isfinite(centroids).all():
        n_bad = int((~np.isfinite(centroids)).sum())
        r.errors.append(f"Centroids have {n_bad:,} non-finite values")
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
        r.errors.append(
            f"{label}: row count mismatch file={n:,} expected={expected_rows:,} ({path})",
        )
    return r


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


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
        raise Step03ValidationError("; ".join(result.errors))


# ---------------------------------------------------------------------------
# Composite checks
# ---------------------------------------------------------------------------


def run_input_validations(df: pd.DataFrame) -> ValidationResult:
    out = ValidationResult()
    out.merge(validate_input_matrix(df))
    return out


def run_clustering_output_validations(
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    n_rows: int,
    k: int,
    n_features: int,
) -> ValidationResult:
    out = ValidationResult()
    out.merge(validate_clusters_assigned_all_rows(labels, n_rows))
    out.merge(validate_no_empty_cluster(labels, k))
    out.merge(validate_centroids_shape(centroids, k=k, n_features=n_features))
    return out


__all__ = [
    "Step03ValidationError",
    "ValidationResult",
    "apply_validation_result",
    "count_csv_data_rows",
    "run_clustering_output_validations",
    "run_input_validations",
    "validate_centroids_shape",
    "validate_clusters_assigned_all_rows",
    "validate_input_matrix",
    "validate_no_empty_cluster",
    "validate_pca_variance_threshold",
    "validate_silhouette_curve",
    "validate_written_csv",
]

