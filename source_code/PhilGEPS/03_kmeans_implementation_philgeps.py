"""
Step 03 — PhilGEPS k-means clustering with 3D PCA scatter visualizations.

Reads the step 02 final preprocessed matrix
(MedFlowPH/output_source/02/philgeps_preprocessed.csv) — already in [0, 1] across
all columns — and produces:

    output_source/03/
        philgeps_clusters.csv             (row_id, cluster, PC1..PC3)
        centroids_original_space.csv      (K rows × n_features in standardized space)
        centroids_pca_space.csv           (K rows × 3 PCs)
        pca_components.csv                (3 rows × n_features loadings)
        pca_explained_variance.json
        silhouette_by_k.csv               (K; silhouette±std; CH; DB; composite;
                                           separation_db_inv; inertia; sweep fit rows)
        kmeans_meta.json                  (best K, params, seeds, dataset shape, runtime)
        Cluster Profiles/
            cluster_<k>_numeric_means.csv
            cluster_<k>_dummy_prevalence.csv

Visualizations under MedFlowPH/results/03/ — all 3D PCA scatters use PC1, PC2, PC3:

    01_feature_variance_audit.png
    02_block_contribution_audit.png
    03_pca_explained_variance.png
    04_pca_loadings_top.png
    05_silhouette_score_vs_k.png
    06_silhouette_supporting_metrics.png
    07_cluster_sizes.png
    PCA Scatter/
        01_pca_3d_unlabeled.png
        02_pca_3d_clusters.png
        03_pca_3d_clusters_with_centroids.png
        04_clustering_results_by_k_pca2d.png   (grid: per k row = PC1–PC2, PC1–PC3, PC2–PC3)
    PCA Scatter per Cluster/
        cluster_<k>_pca_3d.png            (one figure per final cluster)
    Cluster Interpretation/
        01_numeric_centroid_heatmap.png
        02_top_dummies_per_cluster.png
        03_parallel_coords_centroids.png
        04_cluster_radar_numerics.png
        cluster_interpretation_summary.txt
    Dominating Feature Audit/
        01_pre_vs_post_standardize_var.png
        02_pca_loading_concentration.png

K is selected by a composite of mean sampled silhouette, Calinski–Harabasz, and
1/(1+Davies–Bouldin) (each min–max normalized across the K grid), matching the
same KMeans objective as the final fit.

Geometry pipeline (default = Option 1 + Option 4):
  1. Drop near-constant dummies (base rate < MIN_DUMMY_SUPPORT or > MAX_DUMMY_SUPPORT).
  2. StandardScaler over the kept matrix (avoids dominating features).
  3. PCA fit with enough components to hit PCA_FULL_VARIANCE_TARGET (e.g. 80 %).
  4. K-Means clusters in the PCA-reduced space  (CLUSTER_ON_PCA = True).
  5. The 3D scatters use PC1, PC2, PC3 of that same PCA basis (subspace of clustering).

Easy revert path:
  - Snapshot of the previous v1 outputs lives at MedFlowPH/results/03_v1_full312d_baseline/
    and MedFlowPH/output_source/03_v1_full312d_baseline/.
  - For full-D clustering like v1, set CLUSTER_ON_PCA = False below and re-run.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from philgeps_step03_validation import (
    Step03ValidationError,
    apply_validation_result,
    run_clustering_output_validations,
    run_input_validations,
    validate_pca_variance_threshold,
    validate_silhouette_curve,
    validate_written_csv,
)

# ---------------------------------------------------------------------------
# Paths (mirror step 02 layout, but for /03/)
# ---------------------------------------------------------------------------

_HERE = os.path.abspath(__file__)
MEDFLOW_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

PATH_INPUT_CSV = os.path.join(
    MEDFLOW_ROOT, "output_source", "02", "philgeps_preprocessed.csv",
)
PATH_FS_CSV = os.path.join(
    MEDFLOW_ROOT, "output_source", "02", "Feature Selection",
    "philgeps_features_selected.csv",
)

PATH_OUTPUT_03 = os.path.join(MEDFLOW_ROOT, "output_source", "03")
PATH_OUT_PROFILES = os.path.join(PATH_OUTPUT_03, "Cluster Profiles")

PATH_RESULTS_03 = os.path.join(MEDFLOW_ROOT, "results", "03")
PATH_RES_PCA_SCATTER = os.path.join(PATH_RESULTS_03, "PCA Scatter")
PATH_RES_PCA_PER_CLUSTER = os.path.join(PATH_RESULTS_03, "PCA Scatter per Cluster")
PATH_RES_INTERP = os.path.join(PATH_RESULTS_03, "Cluster Interpretation")
PATH_RES_DOM = os.path.join(PATH_RESULTS_03, "Dominating Feature Audit")

PATH_LOGS_03 = os.path.join(MEDFLOW_ROOT, "logs", "03")
PATH_LOG_TERMINAL = os.path.join(PATH_LOGS_03, "Terminal Logs")
PATH_LOG_ENTRIES = os.path.join(PATH_LOGS_03, "Log entries")

OUT_CLUSTERS_CSV = os.path.join(PATH_OUTPUT_03, "philgeps_clusters.csv")
OUT_CENTROIDS_ORIG_CSV = os.path.join(PATH_OUTPUT_03, "centroids_original_space.csv")
OUT_CENTROIDS_PCA_CSV = os.path.join(PATH_OUTPUT_03, "centroids_pca_space.csv")
OUT_CENTROIDS_PCA_FULL_CSV = os.path.join(PATH_OUTPUT_03, "centroids_pca_full_space.csv")
OUT_PCA_COMPONENTS_CSV = os.path.join(PATH_OUTPUT_03, "pca_components.csv")
OUT_PCA_FULL_COMPONENTS_CSV = os.path.join(PATH_OUTPUT_03, "pca_full_components.csv")
OUT_PCA_EXPLAINED_JSON = os.path.join(PATH_OUTPUT_03, "pca_explained_variance.json")
OUT_PCA_FULL_EXPLAINED_JSON = os.path.join(PATH_OUTPUT_03, "pca_full_explained_variance.json")
OUT_SILHOUETTE_CSV = os.path.join(PATH_OUTPUT_03, "silhouette_by_k.csv")
OUT_KMEANS_META_JSON = os.path.join(PATH_OUTPUT_03, "kmeans_meta.json")
OUT_DROPPED_DUMMIES_CSV = os.path.join(PATH_OUTPUT_03, "dropped_dummies.csv")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
K_RANGE: tuple[int, ...] = tuple(range(2, 11))            # K = 2..10
SILHOUETTE_SAMPLE_SIZE = 20_000
SILHOUETTE_N_RESAMPLES = 5
# K sweep uses KMeans (same family as final fit). Above this row count, fit on a fixed subsample.
K_SWEEP_FIT_MAX_ROWS: int = 50_000

# --- Geometry switches (flip CLUSTER_ON_PCA to False to match v1 full-matrix K-Means) ---
CLUSTER_ON_PCA: bool = True            # True: KMeans in PCA space (typical higher silhouette); False: full standardized matrix (v1)
MIN_DUMMY_SUPPORT: float = 0.0        # 0.0 -> keep all dummies (v1)
MAX_DUMMY_SUPPORT: float = 0.98        # symmetric upper bound on prevalence
PCA_FULL_VARIANCE_TARGET: float = 0.80  # cumulative variance for clustering basis
MAX_PCA_COMPONENTS: int = 150          # hard cap on PCA dimensionality
PCA_N_VIEW_COMPONENTS: int = 3         # 3D scatter always shows PC1, PC2, PC3
# Warn only if PC1..PC3 cumulative variance falls below this. High-D one-hot data often
# lands around 5–10%; 0.90 would false-alarm on every run.
PCA_VIEW_VARIANCE_TARGET: float = 0.05

SCATTER_SUBSAMPLE_FOR_PLOT = 30_000
# Matplotlib scatter ``s`` is marker area in points² — small values vanish at 150 DPI with ~30k points.
SCATTER_3D_S_UNLABELED = 22
SCATTER_3D_S_CLUSTERS = 22
SCATTER_3D_S_PER_CLUSTER_BACKGROUND = 14
SCATTER_3D_S_PER_CLUSTER_FOCUS = 52
# Fraction of each axis span added as symmetric padding around the data cloud (3D scatters).
SCATTER_3D_AXIS_MARGIN = 0.02
# Minimum padding as a fraction of span (avoids zero pad if margin is tiny).
SCATTER_3D_AXIS_PAD_MIN_FRAC = 0.002
# 3D axis limits: PC3 (z) often has long tails → wide-looking vertical axis; use inner percentiles on z.
SCATTER_3D_Z_LIMIT_P_LO = 0.5
SCATTER_3D_Z_LIMIT_P_HI = 99.5
# Multiply PC3 padding only (after span-based pad); lower = tighter vertical frame.
SCATTER_3D_Z_MARGIN_SCALE = 0.32
# Fixed PC1/PC2 limits (min, max) for all PCA 3D scatters. None = auto from data + padding.
SCATTER_3D_PC1_LIM: tuple[float, float] | None = (-6.35, 8.5)
SCATTER_3D_PC2_LIM: tuple[float, float] | None = (-4.35, 5.15)
# After limits are set, expand each axis outward by this fraction of its span (half per side).
# Small buffer avoids marker/edge clipping without large empty margins.
SCATTER_3D_LIM_OUTSET_FRAC = 0.055
# PC3 uses only this fraction of SCATTER_3D_LIM_OUTSET_FRAC (reduces empty top/bottom).
SCATTER_3D_Z_OUTSET_REL = 0.55
# Points shown in the K-by-k PCA figure (one subsample shared across all k rows and PC pair columns).
GRID_PCA2D_MAX_POINTS = 25_000
FINAL_KMEANS_N_INIT = 10
FINAL_KMEANS_MAX_ITER = 300
DUMMY_LIFT_MIN_SUPPORT = 0.02
TOP_DUMMIES_PER_CLUSTER = 10
TOP_LOADINGS_PER_PC = 15

# Engineered numerics produced by step 02 — used for cluster interpretation
ENGINEERED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "Year",
    "log1p_Approved_Budget_of_the_Contract",
    "log1p_Item_Budget",
    "log1p_Contract_Amount",
    "log1p_Quantity",
    "contract_duration_days",
    "pub_year",
    "pub_month",
    "pub_dow",
    "pub_epoch_days",
    "time_to_close_days",
    "prebid_lead_days",
    "award_publish_lag_days",
    "award_decision_lag_days",
    "ntp_lag_days",
    "contract_length_days",
    "contract_effectivity_lag_days",
    "City_Municipality_freq",
    "City_Municipality_of_Awardee_freq",
)


# ---------------------------------------------------------------------------
# Logging helpers (mirror step 02)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def tee_stdio_to_file(path: str) -> Any:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as logf:

        class _Tee:
            def __init__(self, *streams: TextIO) -> None:
                self._streams = streams

            def write(self, data: str) -> None:
                for s in self._streams:
                    s.write(data)
                    s.flush()

            def flush(self) -> None:
                for s in self._streams:
                    s.flush()

        old = sys.stdout
        sys.stdout = _Tee(old, logf)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old


def open_activity_log(activity_path: str) -> Callable[[str], None]:
    os.makedirs(os.path.dirname(activity_path) or ".", exist_ok=True)
    if os.path.isfile(activity_path):
        os.remove(activity_path)

    def _log(msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with open(activity_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{ts}] {msg}\n")

    return _log


def _ensure_tree() -> None:
    for p in (
        PATH_OUTPUT_03, PATH_OUT_PROFILES,
        PATH_RESULTS_03, PATH_RES_PCA_SCATTER, PATH_RES_PCA_PER_CLUSTER,
        PATH_RES_INTERP, PATH_RES_DOM,
        PATH_LOGS_03, PATH_LOG_TERMINAL, PATH_LOG_ENTRIES,
    ):
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 — Load + standardize
# ---------------------------------------------------------------------------


def _load_preprocessed(input_csv: str, *, log: Callable[[str], None]) -> pd.DataFrame:
    log(f"Reading preprocessed matrix: {input_csv}")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Step-02 preprocessed CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    log(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def _split_blocks(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Best-effort split: known engineered numerics vs. everything else (dummies)."""
    numeric_block = [c for c in ENGINEERED_NUMERIC_COLUMNS if c in df.columns]
    dummy_block = [c for c in df.columns if c not in numeric_block]
    return numeric_block, dummy_block


def _filter_low_support_dummies(
    df: pd.DataFrame,
    dummy_block: list[str],
    *,
    min_support: float,
    max_support: float,
    log: Callable[[str], None],
) -> tuple[list[str], pd.DataFrame]:
    """
    Drop dummy columns whose base rate is < ``min_support`` or > ``max_support``.

    Returns (kept_dummies, drop_report). The report has one row per dummy with
    columns: dummy, base_rate, kept (bool), reason.
    """
    rows: list[dict[str, Any]] = []
    kept: list[str] = []
    if not dummy_block:
        return kept, pd.DataFrame(columns=["dummy", "base_rate", "kept", "reason"])
    rates = df[dummy_block].mean(axis=0)
    for c in dummy_block:
        r = float(rates.loc[c])
        keep = (r >= min_support) and (r <= max_support)
        reason = "kept"
        if r < min_support:
            reason = f"base_rate<{min_support}"
        elif r > max_support:
            reason = f"base_rate>{max_support}"
        rows.append({"dummy": c, "base_rate": r, "kept": keep, "reason": reason})
        if keep:
            kept.append(c)
    report = pd.DataFrame(rows)
    n_dropped = int((~report["kept"]).sum())
    log(
        f"Dummy support filter [{min_support}, {max_support}]: kept={len(kept)}/"
        f"{len(dummy_block)}, dropped={n_dropped}",
    )
    return kept, report


def _plot_dropped_dummies(
    report: pd.DataFrame,
    *,
    min_support: float,
    max_support: float,
    out_path: str,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    if report.empty:
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    rates = report["base_rate"].to_numpy()
    ax0.hist(rates, bins=60, color="lightgray", alpha=0.85, edgecolor="white")
    ax0.axvline(min_support, color="red", linestyle="--", linewidth=1.0,
                label=f"min={min_support}")
    ax0.axvline(max_support, color="red", linestyle="--", linewidth=1.0,
                label=f"max={max_support}")
    ax0.set_yscale("log")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_title("Dummy base-rate distribution (all dummies)")
    ax0.set_xlabel("base rate")
    ax0.set_ylabel("# dummies (log scale)")
    ax0.legend()

    counts = report["kept"].value_counts()
    counts.index = counts.index.map({True: "kept", False: "dropped"})
    counts = counts.reindex(["kept", "dropped"], fill_value=0)
    ax1.bar(counts.index, counts.values, color=["steelblue", "indianred"])
    for i, v in enumerate(counts.values):
        ax1.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("Dummies kept vs dropped (Option 4)")
    ax1.set_ylabel("# dummies")

    plt.suptitle(
        f"Dummy support audit — drop dummies with base rate < {min_support} or > {max_support}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _variance_audit(
    df: pd.DataFrame,
    numeric_block: list[str],
    dummy_block: list[str],
    *,
    log: Callable[[str], None],
) -> pd.DataFrame:
    """Compute per-column variance and per-block variance share."""
    var = df.var(axis=0, ddof=0)
    is_numeric = pd.Series(
        ["numeric" if c in numeric_block else "dummy" for c in df.columns],
        index=df.columns,
        name="block",
    )
    audit = pd.DataFrame({"column": df.columns, "block": is_numeric.values, "variance": var.values})
    total = float(audit["variance"].sum())
    num_share = float(audit.loc[audit["block"] == "numeric", "variance"].sum()) / max(total, 1e-12)
    dum_share = float(audit.loc[audit["block"] == "dummy", "variance"].sum()) / max(total, 1e-12)
    log(
        f"Variance audit (pre-standardize): n_numeric={len(numeric_block)}, "
        f"n_dummies={len(dummy_block)}, "
        f"numeric_block_share={num_share:.3f}, dummy_block_share={dum_share:.3f}",
    )
    return audit


def _plot_feature_variance_audit(audit: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    top = audit.sort_values("variance", ascending=False).head(40).iloc[::-1]
    palette = {"numeric": "steelblue", "dummy": "lightgray"}
    colors = [palette[b] for b in top["block"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.25)))
    ax.barh(top["column"], top["variance"], color=colors)
    ax.set_title("Step 03 — top 40 columns by variance (pre-standardize)")
    ax.set_xlabel("Variance (on min-max [0,1] scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette]
    ax.legend(handles, list(palette.keys()), loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_block_contribution(audit: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    grouped = audit.groupby("block")["variance"].sum()
    counts = audit.groupby("block")["variance"].count()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    grouped.plot(kind="bar", ax=ax0, color=["lightgray", "steelblue"])
    ax0.set_title("Total variance by block")
    ax0.set_ylabel("Sum of column variances")
    counts.plot(kind="bar", ax=ax1, color=["lightgray", "steelblue"])
    ax1.set_title("Number of columns by block")
    ax1.set_ylabel("Column count")
    for ax in (ax0, ax1):
        ax.tick_params(axis="x", rotation=0)
    plt.suptitle("Step 03 — block contribution audit (numeric vs dummy)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _safe_hist(ax: Any, values: np.ndarray, *, color: str, label: str,
               bins: int = 40) -> None:
    """Histogram that tolerates near-constant inputs (post-standardize variance ≈ 1.0)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        ax.bar([lo], [arr.size], width=max(abs(lo) * 0.02, 1e-3),
               color=color, alpha=0.7, label=label)
        return
    ax.hist(arr, bins=bins, range=(lo, hi), alpha=0.7, color=color, label=label)


def _plot_pre_vs_post_standardize_var(
    pre_var: pd.Series,
    post_var: pd.Series,
    audit: pd.DataFrame,
    out_path: str,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    blocks = audit.set_index("column")["block"]
    palette = {"numeric": "steelblue", "dummy": "darkorange"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for b in ("numeric", "dummy"):
        cols = blocks[blocks == b].index
        _safe_hist(ax0, pre_var.loc[cols].to_numpy(), color=palette[b], label=b)
        _safe_hist(ax1, post_var.loc[cols].to_numpy(), color=palette[b], label=b)
    ax0.set_title("Pre-standardize variance distribution")
    ax0.set_xlabel("Variance"); ax0.set_ylabel("# columns"); ax0.legend()
    ax1.set_title("Post-standardize variance distribution")
    ax1.set_xlabel("Variance"); ax1.set_ylabel("# columns")
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax1.legend()
    plt.suptitle("Dominating-feature audit — variance equalization via StandardScaler",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def standardize_matrix(
    df: pd.DataFrame,
    *,
    log: Callable[[str], None],
) -> tuple[np.ndarray, StandardScaler, pd.Series, pd.Series]:
    """Return (X_std, scaler, pre_variance, post_variance) and free the dataframe early."""
    pre_var = df.var(axis=0, ddof=0)
    scaler = StandardScaler(copy=False)
    arr = df.to_numpy(dtype=np.float32, copy=True)
    X_std = scaler.fit_transform(arr)
    post_var = pd.Series(X_std.var(axis=0, ddof=0), index=df.columns)
    log(
        f"StandardScaler fit on {X_std.shape[0]:,} rows × {X_std.shape[1]} columns; "
        f"post-scale mean={float(X_std.mean()):.3e}, std mean={float(post_var.mean()):.3f}",
    )
    return X_std, scaler, pre_var, post_var


# ---------------------------------------------------------------------------
# Stage 2 — PCA
# ---------------------------------------------------------------------------


def fit_pca_auto(
    X_std: np.ndarray,
    feature_names: list[str],
    *,
    target_variance: float,
    max_components: int,
    log: Callable[[str], None],
) -> tuple[PCA, np.ndarray, int, dict[str, Any]]:
    """
    Fit a single randomized PCA with up to ``max_components`` components, then determine
    the smallest ``n_full`` whose cumulative variance >= ``target_variance``. Returns the
    fitted PCA, the transformed (n × n_full) matrix, n_full, and an info dict describing
    both the full-basis (clustering) and the first 3 view PCs.
    """
    n_features = X_std.shape[1]
    n_probe = int(min(max_components, n_features - 1))
    n_probe = max(n_probe, PCA_N_VIEW_COMPONENTS)
    pca = PCA(
        n_components=n_probe, svd_solver="randomized", random_state=RANDOM_SEED,
    )
    X_full = pca.fit_transform(X_std).astype(np.float32, copy=False)
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n_full = int(np.searchsorted(cum, target_variance) + 1)
    n_full = min(max(n_full, PCA_N_VIEW_COMPONENTS), n_probe)

    explained_full = [float(x) for x in ratios[:n_full]]
    cum_full = float(cum[n_full - 1])
    explained_view = explained_full[:PCA_N_VIEW_COMPONENTS]
    cum_view = float(np.sum(explained_view))

    log(
        f"PCA fit (probe={n_probe} components): n_full={n_full} explains "
        f"{cum_full:.4f} of variance (target {target_variance:.0%}); "
        f"first {PCA_N_VIEW_COMPONENTS} PCs explain {cum_view:.4f} (3D-view).",
    )

    info_full = {
        "n_components": n_full,
        "n_components_probe": n_probe,
        "explained_variance_ratio": explained_full,
        "cumulative_variance_ratio": cum_full,
        "all_probe_explained_variance_ratio": [float(x) for x in ratios],
        "all_probe_cumulative_variance_ratio": [float(x) for x in cum],
        "singular_values": [float(s) for s in pca.singular_values_[:n_full]],
        "feature_names": feature_names,
        "variance_target": target_variance,
    }
    info_view = {
        "n_components": PCA_N_VIEW_COMPONENTS,
        "explained_variance_ratio": explained_view,
        "cumulative_variance_ratio": cum_view,
        "feature_names": feature_names,
        "variance_target": PCA_VIEW_VARIANCE_TARGET,
    }
    info = {"full": info_full, "view": info_view}
    return pca, X_full[:, :n_full], n_full, info


def _save_pca_full_artifacts(
    pca: PCA, info: dict[str, Any], feature_names: list[str], n_full: int,
    *, components_csv: str, info_json: str,
) -> None:
    """Write the n_full × n_features components and the full info JSON."""
    comp_df = pd.DataFrame(
        pca.components_[:n_full],
        index=[f"PC{i+1}" for i in range(n_full)],
        columns=feature_names,
    )
    comp_df.to_csv(components_csv)
    with open(info_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)


def _save_pca_view_artifacts(
    pca: PCA, info_view: dict[str, Any], feature_names: list[str],
    *, components_csv: str, info_json: str,
) -> None:
    """Write the 3 × n_features view components and the view info JSON (3D scatter axes)."""
    comp_df = pd.DataFrame(
        pca.components_[:PCA_N_VIEW_COMPONENTS],
        index=[f"PC{i+1}" for i in range(PCA_N_VIEW_COMPONENTS)],
        columns=feature_names,
    )
    comp_df.to_csv(components_csv)
    with open(info_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info_view, f, indent=2)


def _plot_full_scree(info_full: dict[str, Any], n_full: int, out_path: str) -> None:
    """Scree plot over all probe components, with N_full + variance target highlighted."""
    sns.set_theme(style="whitegrid", context="notebook")
    ratios = info_full["all_probe_explained_variance_ratio"]
    cum = info_full["all_probe_cumulative_variance_ratio"]
    target = info_full["variance_target"]
    x = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, ratios, color="steelblue", alpha=0.7, label="per-PC variance ratio")
    ax.plot(x, cum, "-", color="darkorange", linewidth=2.0, label="cumulative")
    ax.axhline(target, color="red", linestyle="--", linewidth=1.0,
               label=f"target {target:.0%}")
    ax.axvline(n_full, color="black", linestyle=":", linewidth=1.0,
               label=f"N_full = {n_full} (cluster basis)")
    ax.axvline(PCA_N_VIEW_COMPONENTS, color="purple", linestyle=":", linewidth=1.0,
               label=f"view = first {PCA_N_VIEW_COMPONENTS} PCs (3D scatter)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Variance ratio")
    ax.set_title(
        f"PCA full scree: N_full={n_full} explains {cum[n_full - 1]:.3f} of variance "
        f"(probe up to {len(ratios)})",
    )
    ax.legend(loc="center right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pca_explained_variance(info_view: dict[str, Any], out_path: str) -> None:
    """Compact plot of just the 3 view PCs that drive the 3D scatter."""
    sns.set_theme(style="whitegrid", context="notebook")
    ratios = info_view["explained_variance_ratio"]
    cum = np.cumsum(ratios)
    x = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, ratios, color="steelblue", alpha=0.85, label="per-PC variance ratio")
    ax.plot(x, cum, "o-", color="darkorange", label="cumulative")
    ax.axhline(info_view["variance_target"], color="red", linestyle="--", linewidth=1.0,
               label=f"Warn if PC1–3 cum. below ({info_view['variance_target']:.0%}; high-D floor)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"PCA explained variance — view (3D scatter axes)\n"
                 f"cumulative across PC1..PC3 = {cum[-1]:.3f}")
    ax.set_ylabel("Variance ratio")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pca_top_loadings(
    pca: PCA, feature_names: list[str], out_path: str,
    *, top: int = TOP_LOADINGS_PER_PC, n_pc_to_show: int = PCA_N_VIEW_COMPONENTS,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    n_pc = min(n_pc_to_show, pca.n_components_)
    fig, axes = plt.subplots(1, n_pc, figsize=(6.0 * n_pc, max(6, top * 0.3)))
    if n_pc == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        idx = np.argsort(np.abs(loadings))[::-1][:top]
        sorted_idx = idx[np.argsort(loadings[idx])]
        names = [feature_names[j] for j in sorted_idx]
        vals = loadings[sorted_idx]
        colors = ["seagreen" if v >= 0 else "indianred" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"PC{i+1} — top {top} loadings by |value|")
        ax.tick_params(labelsize=8)
    plt.suptitle("PCA loadings (signed) — dominating-feature diagnostic (view PCs)",
                 fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pca_loading_concentration(
    pca: PCA, out_path: str, *, n_pc_to_show: int = PCA_N_VIEW_COMPONENTS,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    n_pc = min(n_pc_to_show, pca.n_components_)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_pc):
        sq = np.sort(pca.components_[i] ** 2)[::-1]
        cum = np.cumsum(sq)
        ax.plot(np.arange(1, len(cum) + 1), cum, label=f"PC{i+1}")
    ax.set_title("Cumulative squared loading by descending feature rank (view PCs)\n"
                 "(steeper curve = a few features dominate the PC)")
    ax.set_xlabel("Feature rank (per PC)")
    ax.set_ylabel("Cumulative squared loading")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 3 — K sweep + silhouette
# ---------------------------------------------------------------------------


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if size >= n:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def _kmeans_sweep_fit_matrix(
    X: np.ndarray,
    *,
    log: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Rows used to fit K during the sweep. Uses all rows up to K_SWEEP_FIT_MAX_ROWS,
    otherwise a fixed RNG subsample so sweep cost stays bounded.

    Returns (X_sweep, sweep_row_idx, n_sweep_rows, used_subsample) where sweep_row_idx
    indexes rows of the original X (for aligning PCA view coordinates).
    """
    n = int(X.shape[0])
    if n <= K_SWEEP_FIT_MAX_ROWS:
        log(f"K sweep: fitting on all n={n:,} rows (same KMeans as final).")
        idx = np.arange(n, dtype=np.int64)
        return X, idx, n, False
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(n, K_SWEEP_FIT_MAX_ROWS, replace=False)
    log(
        f"K sweep: fitting on fixed subsample n={K_SWEEP_FIT_MAX_ROWS:,}/{n:,} "
        f"(seed={RANDOM_SEED}); final fit still uses full data.",
    )
    return X[idx], idx, K_SWEEP_FIT_MAX_ROWS, True


def _minmax_normalize_series(values: np.ndarray) -> np.ndarray:
    """Per-column style min–max into [0,1]; constant column -> 1.0 where finite."""
    out = np.full_like(values, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return out
    lo = float(np.nanmin(values[mask]))
    hi = float(np.nanmax(values[mask]))
    if hi - lo < 1e-12:
        out[mask] = 1.0
    else:
        out[mask] = (values[mask] - lo) / (hi - lo)
    return out


def _add_composite_score_columns(sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Add separation_db_inv = 1/(1+DB) and composite_score: average of min–max
    normalized silhouette, CH, and separation_db_inv across the K grid.
    """
    df = sweep.copy()
    db = df["davies_bouldin_mean"].to_numpy(dtype=float)
    df["separation_db_inv"] = 1.0 / (1.0 + np.where(np.isfinite(db), db, np.nan))

    sil_n = _minmax_normalize_series(df["silhouette_mean"].to_numpy(dtype=float))
    ch_n = _minmax_normalize_series(df["calinski_harabasz_mean"].to_numpy(dtype=float))
    sep_n = _minmax_normalize_series(df["separation_db_inv"].to_numpy(dtype=float))

    comp = (sil_n + ch_n + sep_n) / 3.0
    bad = ~(np.isfinite(sil_n) & np.isfinite(ch_n) & np.isfinite(sep_n))
    comp[bad] = np.nan
    df["composite_silhouette_norm"] = sil_n
    df["composite_ch_norm"] = ch_n
    df["composite_db_sep_norm"] = sep_n
    df["composite_score"] = comp
    return df


def _pick_best_k_from_sweep(sweep: pd.DataFrame) -> tuple[int, str]:
    """Return (best_k, reason). Prefers composite_score argmax; falls back to silhouette."""
    if sweep["composite_score"].notna().any():
        sub = sweep.loc[sweep["composite_score"].notna()]
        k = int(sub.loc[sub["composite_score"].idxmax(), "k"])
        return k, "composite(sil_norm+ch_norm+db_sep_norm)"
    k = int(sweep.loc[sweep["silhouette_mean"].idxmax(), "k"])
    return k, "silhouette_only_fallback"


def k_sweep(
    X_std: np.ndarray,
    *,
    k_values: tuple[int, ...] = K_RANGE,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray]:
    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    rng_master = np.random.default_rng(RANDOM_SEED)
    X_sweep, sweep_row_idx, n_sweep_fit, sweep_subsampled = _kmeans_sweep_fit_matrix(
        X_std, log=log,
    )
    metric_cap = min(SILHOUETTE_SAMPLE_SIZE, n_sweep_fit)

    for k in k_values:
        if k >= n_sweep_fit:
            log(f"K={k}: skip (not enough rows in sweep matrix for k clusters)")
            continue
        t0 = time.time()
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=FINAL_KMEANS_N_INIT,
            max_iter=FINAL_KMEANS_MAX_ITER,
            random_state=RANDOM_SEED,
            algorithm="lloyd",
        )
        labels = model.fit_predict(X_sweep)
        labels_by_k[int(k)] = labels.astype(np.int32, copy=False)
        fit_secs = time.time() - t0

        sil_scores: list[float] = []
        ch_scores: list[float] = []
        db_scores: list[float] = []
        for _ in range(SILHOUETTE_N_RESAMPLES):
            seed = int(rng_master.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(seed)
            idx = _sample_indices(n_sweep_fit, metric_cap, sub_rng)
            X_sub = X_sweep[idx]
            y_sub = labels[idx]
            if np.unique(y_sub).size < 2:
                continue
            sil = float(silhouette_score(X_sub, y_sub, metric="euclidean", random_state=seed))
            ch = float(calinski_harabasz_score(X_sub, y_sub))
            db = float(davies_bouldin_score(X_sub, y_sub))
            sil_scores.append(sil)
            ch_scores.append(ch)
            db_scores.append(db)

        sil_mean = float(np.mean(sil_scores)) if sil_scores else float("nan")
        sil_std = float(np.std(sil_scores)) if sil_scores else float("nan")
        ch_mean = float(np.mean(ch_scores)) if ch_scores else float("nan")
        db_mean = float(np.mean(db_scores)) if db_scores else float("nan")
        rows.append(
            {
                "k": int(k),
                "silhouette_mean": sil_mean,
                "silhouette_std": sil_std,
                "calinski_harabasz_mean": ch_mean,
                "davies_bouldin_mean": db_mean,
                "inertia": float(model.inertia_),
                "n_iter": int(model.n_iter_),
                "fit_seconds": round(fit_secs, 2),
                "metric_sample_size": int(metric_cap),
                "n_sweep_fit_rows": int(n_sweep_fit),
                "sweep_on_subsample": bool(sweep_subsampled),
                "n_resamples": SILHOUETTE_N_RESAMPLES,
            },
        )
        log(
            f"K={k}: silhouette={sil_mean:.4f}±{sil_std:.4f}, "
            f"CH={ch_mean:.1f}, DB={db_mean:.4f}, "
            f"inertia={model.inertia_:.1f}, fit={fit_secs:.1f}s",
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, labels_by_k, sweep_row_idx
    out = out.sort_values("k").reset_index(drop=True)
    return _add_composite_score_columns(out), labels_by_k, sweep_row_idx


def _plot_silhouette_curve(
    sweep: pd.DataFrame,
    best_k: int,
    out_path: str,
    *,
    k_sil_argmax: int | None = None,
) -> None:
    if k_sil_argmax is None:
        k_sil_argmax = int(sweep.loc[sweep["silhouette_mean"].idxmax(), "k"])
    sil_by_k = sweep.set_index("k")["silhouette_mean"]
    sil_at_best = float(sil_by_k.loc[best_k])
    sil_at_sil_max = float(sil_by_k.loc[k_sil_argmax])

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        sweep["k"], sweep["silhouette_mean"], yerr=sweep["silhouette_std"],
        fmt="o-", color="steelblue", capsize=4, zorder=2,
        label="Mean silhouette ± std (subsamples)",
    )

    ax.scatter(
        [k_sil_argmax],
        [sil_at_sil_max],
        s=220,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=1.2,
        zorder=6,
        label=f"Argmax silhouette → K={k_sil_argmax} ({sil_at_sil_max:.4f})",
    )

    ax.axvline(
        best_k,
        color="darkorange",
        linestyle="-",
        linewidth=2.4,
        zorder=3,
        alpha=0.95,
        label=f"Selected K={best_k} (argmax composite score)",
    )
    if k_sil_argmax != best_k:
        ax.axvline(
            k_sil_argmax,
            color="#6c757d",
            linestyle=":",
            linewidth=2.0,
            zorder=1,
            alpha=0.9,
            label="Silhouette-only peak (not chosen when it differs)",
        )

    ax.scatter(
        [best_k],
        [sil_at_best],
        s=140,
        marker="o",
        facecolors="none",
        edgecolors="darkorange",
        linewidths=3,
        zorder=5,
        label=f"Silhouette at selected K = {sil_at_best:.4f}",
    )

    has_composite = (
        "composite_score" in sweep.columns
        and sweep["composite_score"].notna().any()
    )
    comp_title = ""
    if has_composite:
        ax2 = ax.twinx()
        ax2.plot(
            sweep["k"], sweep["composite_score"], "s--",
            color="darkgreen", alpha=0.9, markersize=8, zorder=2,
            label="Composite (equal-weight sil + CH + 1/(1+DB), min–max per grid)",
        )
        crow = sweep.loc[sweep["k"] == best_k, "composite_score"]
        if not crow.empty and pd.notna(crow.iloc[0]):
            cval = float(crow.iloc[0])
            comp_title = f" | composite@K={best_k} = {cval:.3f}"
            ax2.scatter(
                [best_k], [cval],
                s=120, marker="D", color="darkgreen", zorder=7,
                edgecolors="white", linewidths=1.2,
            )
        ax2.set_ylabel("Composite score", color="darkgreen")
        ax2.tick_params(axis="y", labelcolor="darkgreen")
        ax2.set_ylim(-0.05, 1.05)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(
            h1 + h2, l1 + l2,
            loc="lower left", fontsize=8, framealpha=0.95,
        )
    else:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    ax.set_xticks(list(sweep["k"]))
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Mean silhouette (higher is better)")
    ax.set_title(
        "K: gold star = highest mean silhouette; orange line = final K from composite."
        + comp_title,
        fontsize=11,
    )
    if has_composite:
        explain = (
            "How to read: The gold star is always at the K with the largest mean silhouette on this grid. "
            "The solid orange line is the K used for the final model (argmax of composite). "
            "The dotted gray vertical line appears only when that silhouette-best K differs from the chosen K."
        )
    else:
        explain = (
            "Gold star = K with highest mean silhouette. Orange line = selected K (fallback / same rule)."
        )
    fig.text(0.5, 0.02, explain, ha="center", fontsize=8.5, color="#333333")

    plt.tight_layout(rect=[0.02, 0.11, 0.98, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_silhouette_supporting_metrics(
    sweep: pd.DataFrame,
    best_k: int,
    out_path: str,
    *,
    k_sil_argmax: int | None = None,
) -> None:
    if k_sil_argmax is None:
        k_sil_argmax = int(sweep.loc[sweep["silhouette_mean"].idxmax(), "k"])

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(sweep["k"], sweep["calinski_harabasz_mean"], "o-", color="seagreen")
    axes[0].axvline(
        best_k, color="darkorange", linestyle="-", linewidth=2.2,
        label=f"Chosen K={best_k}",
    )
    if k_sil_argmax != best_k:
        axes[0].axvline(
            k_sil_argmax, color="#6c757d", linestyle=":", linewidth=2,
            label=f"Max-silhouette K={k_sil_argmax}",
        )
    axes[0].set_title("Calinski–Harabasz (higher better)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("CH score")
    axes[0].set_xticks(list(sweep["k"]))
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(sweep["k"], sweep["davies_bouldin_mean"], "o-", color="indianred")
    axes[1].axvline(
        best_k, color="darkorange", linestyle="-", linewidth=2.2,
        label=f"Chosen K={best_k}",
    )
    if k_sil_argmax != best_k:
        axes[1].axvline(
            k_sil_argmax, color="#6c757d", linestyle=":", linewidth=2,
            label=f"Max-silhouette K={k_sil_argmax}",
        )
    axes[1].set_title("Davies–Bouldin (lower better)")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("DB score")
    axes[1].set_xticks(list(sweep["k"]))
    axes[1].legend(loc="best", fontsize=8)

    plt.suptitle(
        "Metrics mixed into composite — solid orange = final K; dotted gray = silhouette-only argmax if different.",
        fontsize=11,
        y=1.03,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_clustering_results_grid_pca2d(
    X_pc123: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    sweep: pd.DataFrame,
    best_k: int,
    *,
    out_path: str,
    info_view: dict[str, Any],
    labels_best_k_final: np.ndarray | None = None,
) -> None:
    """
    For each candidate K, one row of three 2D projections matching the 3D scatter axes:
    PC1–PC2, PC1–PC3, PC2–PC3 (same PCA basis as 02_pca_3d_clusters). Colors follow
    _palette (same discrete hues as the 3D figures).

    labels_best_k_final: same row order as X_pc123; when set, the best_k row uses these labels
    (final full-data KMeans) instead of the sweep fit for that K.
    """
    if not labels_by_k:
        return
    ks = sorted(labels_by_k.keys())
    ratios = info_view.get("explained_variance_ratio", [0.0, 0.0, 0.0])
    r1 = float(ratios[0]) * 100.0 if len(ratios) > 0 else 0.0
    r2 = float(ratios[1]) * 100.0 if len(ratios) > 1 else 0.0
    r3 = float(ratios[2]) * 100.0 if len(ratios) > 2 else 0.0

    Xv = np.asarray(X_pc123, dtype=float)
    if Xv.ndim != 2 or Xv.shape[1] < 3:
        raise ValueError(
            "_plot_clustering_results_grid_pca2d expects X_pc123 with shape (n, 3+) "
            f"(got {Xv.shape})",
        )
    Xv = Xv[:, :3]
    n = Xv.shape[0]
    if n == 0:
        return
    plot_idx: np.ndarray
    if n > GRID_PCA2D_MAX_POINTS:
        rng = np.random.default_rng(int(RANDOM_SEED) + 7)
        plot_idx = rng.choice(n, GRID_PCA2D_MAX_POINTS, replace=False)
    else:
        plot_idx = np.arange(n, dtype=np.int64)
    Xp = Xv[plot_idx]

    sil_by_k = {int(r["k"]): float(r["silhouette_mean"]) for _, r in sweep.iterrows()}

    sil_colors = "#e8f5f0"
    best_edge = "#1b4332"
    sns.set_theme(style="whitegrid", context="notebook")
    n_k = len(ks)
    pair_axes = (
        (0, 1, f"PC1 ({r1:.1f}%)", f"PC2 ({r2:.1f}%)"),
        (0, 2, f"PC1 ({r1:.1f}%)", f"PC3 ({r3:.1f}%)"),
        (1, 2, f"PC2 ({r2:.1f}%)", f"PC3 ({r3:.1f}%)"),
    )
    pair_short = ("PC1–PC2", "PC1–PC3", "PC2–PC3")
    fig_w = min(4.0 * 3 + 1.0, 14.5)
    fig_h = min(2.55 * n_k + 1.2, 36)
    fig, axes = plt.subplots(
        n_k, 3, figsize=(fig_w, fig_h), dpi=150, facecolor=sil_colors,
        squeeze=False,
    )
    axes = np.atleast_2d(axes)

    for ri, k in enumerate(ks):
        if k == best_k and labels_best_k_final is not None:
            labels = labels_best_k_final[plot_idx].astype(np.int64, copy=False)
        else:
            labels = labels_by_k[k][plot_idx].astype(np.int64, copy=False)
        sil = sil_by_k.get(k, float("nan"))
        is_best = k == best_k
        palette_k = _palette(int(k))
        colors = palette_k[labels]

        for ci in range(3):
            ax = axes[ri, ci]
            i0, i1, xl, yl = pair_axes[ci]
            ax.scatter(
                Xp[:, i0], Xp[:, i1], c=colors,
                alpha=0.78, s=7, linewidths=0.12, edgecolors="white",
            )
            ax.set_facecolor(sil_colors)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.grid(True, alpha=0.35)
            if ci == 0:
                ax.set_title(
                    f"k={k}" + (" (best)" if is_best else "") + f"\nSil={sil:.3f}",
                    fontsize=10, fontweight="bold", loc="left",
                )
            else:
                ax.set_title(pair_short[ci], fontsize=9)

            if is_best:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.2)
                    spine.set_color(best_edge)

    same_final = labels_best_k_final is not None
    fig.suptitle(
        "K-Means clustering across candidate k — PhilGEPS (three 2D views of PC1–PC3)\n"
        "Each row: same k; columns = PC1–PC2, PC1–PC3, PC2–PC3 (axes match 02/03 PCA 3D scatters). "
        "Colors = cluster id (discrete palette). "
        f"'best' = K={best_k} (composite rule)"
        + ("; best row = final full-data KMeans labels." if same_final else ". ")
        + f"n_plot={Xp.shape[0]:,}.",
        fontsize=11.5, fontweight="bold", y=1.008,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.965])
    fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 4 — Final KMeans refit at best K
# ---------------------------------------------------------------------------


def fit_final_kmeans(
    X_std: np.ndarray,
    *,
    best_k: int,
    log: Callable[[str], None],
) -> tuple[KMeans, np.ndarray]:
    t0 = time.time()
    model = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=FINAL_KMEANS_N_INIT,
        max_iter=FINAL_KMEANS_MAX_ITER,
        random_state=RANDOM_SEED,
        algorithm="lloyd",
    )
    labels = model.fit_predict(X_std)
    log(
        f"Final KMeans(K={best_k}) fit in {time.time() - t0:.1f}s, "
        f"inertia={model.inertia_:.1f}, n_iter={model.n_iter_}",
    )
    return model, labels.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Stage 5 — 3D PCA scatters
# ---------------------------------------------------------------------------


def _palette(k: int) -> np.ndarray:
    base = plt.get_cmap("tab10").colors if k <= 10 else plt.get_cmap("tab20").colors
    if k <= len(base):
        return np.array(base[:k])
    cmap = plt.get_cmap("turbo", k)
    return np.array([cmap(i) for i in range(k)])


def _scatter_subsample(
    X_pca: np.ndarray, labels: np.ndarray | None, *, size: int, seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    n = X_pca.shape[0]
    if size >= n:
        return X_pca, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=size, replace=False)
    return X_pca[idx], (labels[idx] if labels is not None else None)


def _set_axes_labels_3d(ax: Any, info: dict[str, Any]) -> None:
    ratios = info["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratios[2] * 100:.1f}%)")


def _outset_lim_pair(lo: float, hi: float, frac: float) -> tuple[float, float]:
    """Widen [lo, hi] symmetrically by frac * span (split equally on both sides)."""
    if frac <= 0.0 or not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return lo, hi
    span = hi - lo
    d = span * frac * 0.5
    return float(lo - d), float(hi + d)


def _set_3d_axis_limits_tight(
    ax: Any,
    Xp: np.ndarray,
    *,
    extra_xyz: np.ndarray | None = None,
    margin: float | None = None,
) -> None:
    """
    PC1/PC2: use SCATTER_3D_PC1_LIM / SCATTER_3D_PC2_LIM when set; else data min/max + padding.
    PC3: inner percentiles on the cloud + SCATTER_3D_Z_MARGIN_SCALE padding; centroids stay inside z.
    All axes: PC1/PC2 get full SCATTER_3D_LIM_OUTSET_FRAC; PC3 uses a smaller fraction
    (see SCATTER_3D_Z_OUTSET_REL) so the vertical frame does not look overly tall.
    """
    m = SCATTER_3D_AXIS_MARGIN if margin is None else margin
    cloud = np.asarray(Xp, dtype=float).reshape(-1, 3)
    if cloud.shape[0] == 0:
        return

    raw_lo = cloud.min(axis=0)
    raw_hi = cloud.max(axis=0)
    if extra_xyz is not None and extra_xyz.size:
        ex = np.asarray(extra_xyz, dtype=float).reshape(-1, 3)
        raw_lo = np.minimum(raw_lo, ex.min(axis=0))
        raw_hi = np.maximum(raw_hi, ex.max(axis=0))

    lo = raw_lo.copy()
    hi = raw_hi.copy()
    z_lo_p = float(np.percentile(cloud[:, 2], SCATTER_3D_Z_LIMIT_P_LO))
    z_hi_p = float(np.percentile(cloud[:, 2], SCATTER_3D_Z_LIMIT_P_HI))
    lo[2] = z_lo_p
    hi[2] = z_hi_p
    if extra_xyz is not None and extra_xyz.size:
        ez = np.asarray(extra_xyz, dtype=float).reshape(-1, 3)[:, 2]
        lo[2] = float(min(lo[2], float(ez.min())))
        hi[2] = float(max(hi[2], float(ez.max())))

    z_span = float(np.maximum(hi[2] - lo[2], 1e-9))
    z_pad = float(
        np.maximum(z_span * m, z_span * SCATTER_3D_AXIS_PAD_MIN_FRAC) * SCATTER_3D_Z_MARGIN_SCALE,
    )
    z_lim = (float(lo[2] - z_pad), float(hi[2] + z_pad))

    if SCATTER_3D_PC1_LIM is not None:
        a, b = SCATTER_3D_PC1_LIM
        x_lim = (float(min(a, b)), float(max(a, b)))
    else:
        span_x = float(np.maximum(hi[0] - lo[0], 1e-9))
        pad_x = float(np.maximum(span_x * m, span_x * SCATTER_3D_AXIS_PAD_MIN_FRAC))
        x_lim = (float(lo[0] - pad_x), float(hi[0] + pad_x))

    if SCATTER_3D_PC2_LIM is not None:
        a, b = SCATTER_3D_PC2_LIM
        y_lim = (float(min(a, b)), float(max(a, b)))
    else:
        span_y = float(np.maximum(hi[1] - lo[1], 1e-9))
        pad_y = float(np.maximum(span_y * m, span_y * SCATTER_3D_AXIS_PAD_MIN_FRAC))
        y_lim = (float(lo[1] - pad_y), float(hi[1] + pad_y))

    ox0, ox1 = _outset_lim_pair(x_lim[0], x_lim[1], SCATTER_3D_LIM_OUTSET_FRAC)
    oy0, oy1 = _outset_lim_pair(y_lim[0], y_lim[1], SCATTER_3D_LIM_OUTSET_FRAC)
    oz0, oz1 = _outset_lim_pair(
        z_lim[0], z_lim[1],
        SCATTER_3D_LIM_OUTSET_FRAC * SCATTER_3D_Z_OUTSET_REL,
    )

    ax.set_xlim(ox0, ox1)
    ax.set_ylim(oy0, oy1)
    ax.set_zlim(oz0, oz1)
    if hasattr(ax, "set_box_aspect"):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        z0, z1 = ax.get_zlim()
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))


def plot_pca_3d_unlabeled(
    X_pca: np.ndarray, info: dict[str, Any], out_path: str,
) -> None:
    Xp, _ = _scatter_subsample(X_pca, None, size=SCATTER_SUBSAMPLE_FOR_PLOT, seed=RANDOM_SEED)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        Xp[:, 0], Xp[:, 1], Xp[:, 2],
        s=SCATTER_3D_S_UNLABELED,
        alpha=0.35,
        c="steelblue",
        linewidths=0,
        edgecolors="none",
    )
    _set_3d_axis_limits_tight(ax, Xp)
    _set_axes_labels_3d(ax, info)
    ax.set_title(
        f"PhilGEPS — PCA 3D scatter (n_plot={Xp.shape[0]:,}, "
        f"3-PC variance={sum(info['explained_variance_ratio']):.3f})",
    )
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_3d_clusters(
    X_pca: np.ndarray, labels: np.ndarray, info: dict[str, Any], out_path: str,
    *, with_centroids: bool, centroids_pca: np.ndarray | None = None,
) -> None:
    Xp, yp = _scatter_subsample(
        X_pca, labels, size=SCATTER_SUBSAMPLE_FOR_PLOT, seed=RANDOM_SEED,
    )
    k = int(labels.max()) + 1
    palette = _palette(k)
    colors = palette[yp]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        Xp[:, 0], Xp[:, 1], Xp[:, 2],
        s=SCATTER_3D_S_CLUSTERS,
        alpha=0.5,
        c=colors,
        linewidths=0,
        edgecolors="none",
    )

    sizes = np.bincount(labels, minlength=k)
    handles = []
    for j in range(k):
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                color=palette[j], markersize=8,
                label=f"C{j} (n={int(sizes[j]):,}, {sizes[j] / labels.size:.1%})",
            ),
        )

    if with_centroids and centroids_pca is not None:
        ax.scatter(
            centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2],
            s=300, marker="X", c=palette[: centroids_pca.shape[0]],
            edgecolors="black", linewidths=2.0, depthshade=False,
        )
        for j in range(centroids_pca.shape[0]):
            ax.text(
                centroids_pca[j, 0], centroids_pca[j, 1], centroids_pca[j, 2],
                f"  C{j}", fontsize=10, fontweight="bold", color="black",
            )

    _set_3d_axis_limits_tight(
        ax, Xp,
        extra_xyz=centroids_pca if with_centroids and centroids_pca is not None else None,
    )
    _set_axes_labels_3d(ax, info)
    title_suffix = " + final centroids" if with_centroids else ""
    ax.set_title(
        f"PhilGEPS — PCA 3D scatter colored by K-Means cluster{title_suffix} "
        f"(K={k}, n_plot={Xp.shape[0]:,})",
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
              fontsize=9, frameon=True)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_3d_per_cluster(
    X_pca: np.ndarray, labels: np.ndarray, info: dict[str, Any], out_dir: str,
    *, log: Callable[[str], None],
) -> None:
    Xp, yp = _scatter_subsample(
        X_pca, labels, size=SCATTER_SUBSAMPLE_FOR_PLOT, seed=RANDOM_SEED,
    )
    k = int(labels.max()) + 1
    palette = _palette(k)
    counts = np.bincount(labels, minlength=k)
    n_total = int(labels.size)

    for j in range(k):
        in_mask = yp == j
        out_mask = ~in_mask
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            Xp[out_mask, 0], Xp[out_mask, 1], Xp[out_mask, 2],
            s=SCATTER_3D_S_PER_CLUSTER_BACKGROUND,
            alpha=0.18,
            c="#b0b0b0",
            linewidths=0,
            edgecolors="none",
            label="other clusters",
        )
        ax.scatter(
            Xp[in_mask, 0], Xp[in_mask, 1], Xp[in_mask, 2],
            s=SCATTER_3D_S_PER_CLUSTER_FOCUS,
            alpha=0.78,
            c=[palette[j]],
            linewidths=0.4,
            edgecolors="white",
            label=f"cluster {j}",
        )
        _set_3d_axis_limits_tight(ax, Xp)
        _set_axes_labels_3d(ax, info)
        share = counts[j] / max(n_total, 1)
        ax.set_title(
            f"PhilGEPS — PCA 3D scatter — Cluster {j} highlighted "
            f"(n={int(counts[j]):,}, share={share:.1%})",
        )
        ax.legend(loc="upper left")
        ax.view_init(elev=20, azim=30)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"cluster_{j}_pca_3d.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log(f"Wrote per-cluster 3D scatter -> {out_path}")


# ---------------------------------------------------------------------------
# Stage 6 — Cluster interpretation
# ---------------------------------------------------------------------------


def _plot_cluster_sizes(labels: np.ndarray, out_path: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    counts = pd.Series(labels).value_counts().sort_index()
    palette = _palette(len(counts))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([f"C{i}" for i in counts.index], counts.values, color=palette)
    for x, v in enumerate(counts.values):
        ax.text(x, v, f"{int(v):,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Cluster sizes (K={len(counts)}, total n={counts.sum():,})")
    ax.set_ylabel("Row count")
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_engineered_numerics(
    fs_csv: str, *, log: Callable[[str], None],
) -> pd.DataFrame:
    """Load only the engineered numeric columns from the Layer A CSV (un-scaled)."""
    if not os.path.isfile(fs_csv):
        log(f"WARNING: feature-selected CSV not found: {fs_csv}")
        return pd.DataFrame()
    header = pd.read_csv(fs_csv, nrows=0)
    cols = [c for c in ENGINEERED_NUMERIC_COLUMNS if c in header.columns]
    df = pd.read_csv(fs_csv, usecols=cols, low_memory=False)
    log(f"Loaded engineered numerics for interpretation: shape={df.shape}")
    return df


def _numeric_cluster_profiles(
    df_num_orig: pd.DataFrame, labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_cluster_means, per_cluster_zscores) over original-unit numerics."""
    df = df_num_orig.copy()
    df["__cluster__"] = labels
    means = df.groupby("__cluster__").mean(numeric_only=True)
    global_mean = df.drop(columns=["__cluster__"]).mean(numeric_only=True)
    global_std = df.drop(columns=["__cluster__"]).std(numeric_only=True).replace(0.0, 1e-9)
    z = (means - global_mean) / global_std
    means.index.name = "cluster"
    z.index.name = "cluster"
    return means, z


def _plot_numeric_centroid_heatmap(z: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(max(10, len(z.columns) * 0.5), max(4, len(z) * 0.6)))
    sns.heatmap(
        z, ax=ax, cmap="vlag", center=0.0, annot=True, fmt=".2f",
        linewidths=0.3, cbar_kws={"shrink": 0.7, "label": "z-score vs global mean"},
    )
    ax.set_title("Numeric centroid heatmap — z-score of per-cluster mean vs global mean")
    ax.set_xlabel("Engineered numeric feature")
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_parallel_coords(z: pd.DataFrame, out_path: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = _palette(len(z))
    fig, ax = plt.subplots(figsize=(max(11, len(z.columns) * 0.5), 6))
    x = np.arange(len(z.columns))
    for i, (cluster, row) in enumerate(z.iterrows()):
        ax.plot(x, row.values, "o-", color=palette[i],
                label=f"C{cluster}", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--",
               label="global mean")
    ax.set_xticks(x)
    ax.set_xticklabels(z.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("z-score vs global mean")
    ax.set_title("Cluster centroids — parallel coordinates over engineered numerics")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_radar(z: pd.DataFrame, out_path: str) -> None:
    cols = list(z.columns)
    n_axes = len(cols)
    if n_axes < 3:
        return
    palette = _palette(len(z))
    angles = [n / n_axes * 2 * math.pi for n in range(n_axes)]
    angles += angles[:1]

    z_clip = z.clip(-3.0, 3.0)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="polar")
    for i, (cluster, row) in enumerate(z_clip.iterrows()):
        vals = list(row.values) + [row.values[0]]
        ax.plot(angles, vals, "o-", color=palette[i], linewidth=1.8,
                label=f"C{cluster}")
        ax.fill(angles, vals, color=palette[i], alpha=0.10)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_ylim(-3.0, 3.0)
    ax.set_title("Cluster radar (z-score, clipped to ±3) — engineered numerics")
    ax.legend(loc="upper left", bbox_to_anchor=(1.10, 1.05), frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dummy_lift_per_cluster(
    df_dummies: pd.DataFrame, labels: np.ndarray,
    *, top_k: int = TOP_DUMMIES_PER_CLUSTER,
    min_support: float = DUMMY_LIFT_MIN_SUPPORT,
) -> dict[int, pd.DataFrame]:
    """For each cluster, return a DataFrame of top dummies by lift (subject to min support)."""
    global_rate = df_dummies.mean(axis=0)
    out: dict[int, pd.DataFrame] = {}
    k = int(labels.max()) + 1
    for j in range(k):
        idx = np.where(labels == j)[0]
        if idx.size == 0:
            out[j] = pd.DataFrame()
            continue
        sub_rate = df_dummies.iloc[idx].mean(axis=0)
        diff = sub_rate - global_rate
        lift = sub_rate / global_rate.replace(0.0, np.nan)
        prof = pd.DataFrame(
            {
                "dummy": df_dummies.columns,
                "cluster_prevalence": sub_rate.values,
                "global_prevalence": global_rate.values,
                "lift": lift.values,
                "diff": diff.values,
            },
        )
        prof = prof[prof["cluster_prevalence"] >= min_support]
        prof = prof.replace([np.inf, -np.inf], np.nan).dropna(subset=["lift"])
        prof = prof.sort_values("lift", ascending=False).head(top_k).reset_index(drop=True)
        out[j] = prof
    return out


def _plot_top_dummies_per_cluster(
    profiles: dict[int, pd.DataFrame], out_path: str,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    k = len(profiles)
    if k == 0:
        return
    cols_per_row = 2
    nrows = int(math.ceil(k / cols_per_row))
    fig, axes = plt.subplots(nrows, cols_per_row, figsize=(cols_per_row * 8.5, nrows * 4.4))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[k:]:
        ax.axis("off")
    palette = _palette(k)
    for j, ax in zip(sorted(profiles.keys()), axes_flat[:k], strict=False):
        prof = profiles[j]
        if prof.empty:
            ax.set_title(f"Cluster {j} — no dummy meets support threshold")
            ax.axis("off")
            continue
        prof_show = prof.iloc[::-1]
        ax.barh(prof_show["dummy"], prof_show["lift"], color=palette[j])
        for y, (lift, prev) in enumerate(zip(prof_show["lift"], prof_show["cluster_prevalence"])):
            ax.text(lift, y, f" lift={lift:.2f}, p={prev:.2f}", va="center", fontsize=7)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Cluster {j} — top {len(prof_show)} dummies by lift")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("lift = cluster_prevalence / global_prevalence")
    plt.suptitle(
        f"Top one-hot dummies per cluster (min_support={DUMMY_LIFT_MIN_SUPPORT}, lift > 1 = enriched)",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _name_clusters(
    z: pd.DataFrame, dummy_profiles: dict[int, pd.DataFrame],
) -> dict[int, str]:
    names: dict[int, str] = {}
    for cluster in z.index:
        row = z.loc[cluster]
        top_pos = row.sort_values(ascending=False).head(2)
        top_neg = row.sort_values().head(1)
        descriptors: list[str] = []
        for feat, val in top_pos.items():
            if val > 0.5:
                descriptors.append(f"high {feat} (+{val:.1f}σ)")
        for feat, val in top_neg.items():
            if val < -0.5:
                descriptors.append(f"low {feat} ({val:.1f}σ)")
        prof = dummy_profiles.get(int(cluster), pd.DataFrame())
        cat_part = ""
        if not prof.empty:
            top_cat = prof.iloc[0]
            cat_part = (
                f"; top dummy: {top_cat['dummy']} (lift={top_cat['lift']:.2f}, "
                f"p={top_cat['cluster_prevalence']:.2f})"
            )
        text = ", ".join(descriptors) if descriptors else "near-global-average profile"
        names[int(cluster)] = f"{text}{cat_part}"
    return names


def _write_interpretation_summary(
    sizes: dict[int, int], names: dict[int, str], z: pd.DataFrame,
    dummy_profiles: dict[int, pd.DataFrame], out_path: str,
) -> None:
    lines: list[str] = [
        "PhilGEPS step 03 — cluster interpretation summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"K = {len(sizes)}, total n = {sum(sizes.values()):,}",
        "",
        "Heuristic descriptors are derived from z-scored numeric centroids and top dummy lift.",
        "Replace bracketed names with human-readable labels after review.",
        "",
    ]
    for cluster in sorted(sizes.keys()):
        share = sizes[cluster] / max(sum(sizes.values()), 1)
        lines.append(f"--- Cluster {cluster} (n={sizes[cluster]:,}, share={share:.1%}) ---")
        lines.append(f"  proposed name: [{names[cluster]}]")
        lines.append("  numeric centroid (z-score vs global, top movers):")
        row = z.loc[cluster]
        movers = row.reindex(row.abs().sort_values(ascending=False).index).head(5)
        for feat, val in movers.items():
            lines.append(f"    {feat:<40s} {val:+.3f}σ")
        prof = dummy_profiles.get(int(cluster), pd.DataFrame())
        if not prof.empty:
            lines.append("  top dummies by lift:")
            for _, p in prof.iterrows():
                lines.append(
                    f"    lift={p['lift']:6.2f}  p={p['cluster_prevalence']:.3f}  "
                    f"global={p['global_prevalence']:.3f}  {p['dummy']}",
                )
        lines.append("")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_step03(input_csv: str = PATH_INPUT_CSV) -> None:
    _ensure_tree()
    activity_path = os.path.join(
        PATH_LOG_ENTRIES, "03_kmeans_implementation_philgeps_activity.txt",
    )
    log = open_activity_log(activity_path)
    t_start = time.time()

    log(
        f"Mode: CLUSTER_ON_PCA={CLUSTER_ON_PCA}, MIN_DUMMY_SUPPORT={MIN_DUMMY_SUPPORT}, "
        f"PCA_FULL_VARIANCE_TARGET={PCA_FULL_VARIANCE_TARGET}, "
        f"MAX_PCA_COMPONENTS={MAX_PCA_COMPONENTS}",
    )

    # ---- Load + validate input ------------------------------------------------
    df = _load_preprocessed(input_csv, log=log)
    apply_validation_result(run_input_validations(df), log)
    n_rows = df.shape[0]
    n_features_in = df.shape[1]

    numeric_block, dummy_block_full = _split_blocks(df)
    log(f"Block split: numeric={len(numeric_block)}, dummies={len(dummy_block_full)}")

    # ---- Option 4: drop near-constant dummies --------------------------------
    if MIN_DUMMY_SUPPORT > 0.0 or MAX_DUMMY_SUPPORT < 1.0:
        kept_dummies, drop_report = _filter_low_support_dummies(
            df, dummy_block_full,
            min_support=MIN_DUMMY_SUPPORT, max_support=MAX_DUMMY_SUPPORT, log=log,
        )
        drop_report.to_csv(OUT_DROPPED_DUMMIES_CSV, index=False)
        _plot_dropped_dummies(
            drop_report,
            min_support=MIN_DUMMY_SUPPORT, max_support=MAX_DUMMY_SUPPORT,
            out_path=os.path.join(PATH_RESULTS_03, "08_dropped_dummies_by_support.png"),
        )
        df = df[numeric_block + kept_dummies]
        gc.collect()
    else:
        kept_dummies = list(dummy_block_full)
        drop_report = pd.DataFrame()

    feature_names = list(df.columns)

    # ---- Variance audit + standardize ----------------------------------------
    audit = _variance_audit(df, numeric_block, kept_dummies, log=log)
    _plot_feature_variance_audit(audit, os.path.join(PATH_RESULTS_03, "01_feature_variance_audit.png"))
    _plot_block_contribution(audit, os.path.join(PATH_RESULTS_03, "02_block_contribution_audit.png"))

    X_std, scaler, pre_var, post_var = standardize_matrix(df, log=log)
    n_features_kept = X_std.shape[1]
    del df
    gc.collect()

    _plot_pre_vs_post_standardize_var(
        pre_var, post_var, audit,
        os.path.join(PATH_RES_DOM, "01_pre_vs_post_standardize_var.png"),
    )

    # ---- PCA (auto-N to hit variance target) ---------------------------------
    pca, X_pca_full, n_full, pca_info = fit_pca_auto(
        X_std, feature_names,
        target_variance=PCA_FULL_VARIANCE_TARGET,
        max_components=MAX_PCA_COMPONENTS,
        log=log,
    )
    info_full = pca_info["full"]
    info_view = pca_info["view"]

    # The 3D scatter always uses the first 3 PCs of the same basis
    X_pca_view = X_pca_full[:, :PCA_N_VIEW_COMPONENTS]

    apply_validation_result(
        validate_pca_variance_threshold(
            info_view["explained_variance_ratio"],
            target=PCA_VIEW_VARIANCE_TARGET,
            n_components=PCA_N_VIEW_COMPONENTS,
        ),
        log, raise_on_error=False,
    )

    _save_pca_full_artifacts(
        pca, info_full, feature_names, n_full,
        components_csv=OUT_PCA_FULL_COMPONENTS_CSV,
        info_json=OUT_PCA_FULL_EXPLAINED_JSON,
    )
    _save_pca_view_artifacts(
        pca, info_view, feature_names,
        components_csv=OUT_PCA_COMPONENTS_CSV,
        info_json=OUT_PCA_EXPLAINED_JSON,
    )
    _plot_pca_explained_variance(
        info_view, os.path.join(PATH_RESULTS_03, "03_pca_explained_variance.png"),
    )
    _plot_full_scree(
        info_full, n_full, os.path.join(PATH_RESULTS_03, "09_pca_full_scree.png"),
    )
    _plot_pca_top_loadings(
        pca, feature_names,
        os.path.join(PATH_RESULTS_03, "04_pca_loadings_top.png"),
    )
    _plot_pca_loading_concentration(
        pca, os.path.join(PATH_RES_DOM, "02_pca_loading_concentration.png"),
    )
    log(
        f"Wrote PCA artifacts -> {OUT_PCA_FULL_COMPONENTS_CSV}, "
        f"{OUT_PCA_COMPONENTS_CSV}, {OUT_PCA_FULL_EXPLAINED_JSON}, {OUT_PCA_EXPLAINED_JSON}",
    )

    # ---- Unlabeled 3D scatter (before clustering) ----------------------------
    plot_pca_3d_unlabeled(
        X_pca_view, info_view,
        os.path.join(PATH_RES_PCA_SCATTER, "01_pca_3d_unlabeled.png"),
    )

    # ---- Choose clustering geometry ------------------------------------------
    if CLUSTER_ON_PCA:
        X_for_kmeans = X_pca_full
        log(f"Clustering on PCA-reduced matrix: shape={X_for_kmeans.shape}")
    else:
        X_for_kmeans = X_std
        log(f"Clustering on standardized full matrix (v1 mode): shape={X_for_kmeans.shape}")

    # ---- K sweep + composite K selection -----------------------------------
    sweep, labels_by_k, sweep_row_idx = k_sweep(X_for_kmeans, k_values=K_RANGE, log=log)
    if sweep.empty:
        raise Step03ValidationError("K sweep produced no rows (check n vs K_RANGE).")
    sweep.to_csv(OUT_SILHOUETTE_CSV, index=False)
    apply_validation_result(
        validate_silhouette_curve(dict(zip(sweep["k"], sweep["silhouette_mean"]))), log,
    )
    best_k, k_pick_reason = _pick_best_k_from_sweep(sweep)
    best_row = sweep.loc[sweep["k"] == best_k].iloc[0]
    best_composite = float(best_row["composite_score"]) if pd.notna(best_row.get("composite_score")) else float("nan")
    best_silhouette = float(best_row["silhouette_mean"])
    k_sil_only = int(sweep.loc[sweep["silhouette_mean"].idxmax(), "k"])
    log(
        f"Selected K={best_k} via {k_pick_reason}: silhouette_mean={best_silhouette:.4f}, "
        f"composite_score={best_composite:.4f}. "
        f"(Silhouette-only argmax would be K={k_sil_only}.)",
    )
    _plot_silhouette_curve(
        sweep,
        best_k,
        os.path.join(PATH_RESULTS_03, "05_silhouette_score_vs_k.png"),
        k_sil_argmax=k_sil_only,
    )
    _plot_silhouette_supporting_metrics(
        sweep,
        best_k,
        os.path.join(PATH_RESULTS_03, "06_silhouette_supporting_metrics.png"),
        k_sil_argmax=k_sil_only,
    )

    # ---- Final KMeans refit at best K ----------------------------------------
    final_model, labels = fit_final_kmeans(X_for_kmeans, best_k=best_k, log=log)
    cluster_centers_native = final_model.cluster_centers_

    grid_out = os.path.join(PATH_RES_PCA_SCATTER, "04_clustering_results_by_k_pca2d.png")
    _plot_clustering_results_grid_pca2d(
        X_pca_full[sweep_row_idx, :3],
        labels_by_k,
        sweep,
        best_k,
        out_path=grid_out,
        info_view=info_view,
        labels_best_k_final=labels[sweep_row_idx],
    )
    log(f"Wrote K-by-k PCA 2D grid -> {grid_out}")

    if CLUSTER_ON_PCA:
        # KMeans fit in n_full-D PCA space. Reconstruct standardized space using
        # only the first n_full PCs (the rest are zero in the cluster centers).
        centroids_pca_full = cluster_centers_native.astype(np.float64, copy=False)
        centroids_view = centroids_pca_full[:, :PCA_N_VIEW_COMPONENTS]
        centroids_std_space = (
            centroids_pca_full @ pca.components_[:n_full] + pca.mean_
        )
        centroids_orig_space = scaler.inverse_transform(centroids_std_space)
    else:
        centroids_std_space = cluster_centers_native.astype(np.float64, copy=False)
        # Project into the same n_full PCA space for downstream consistency.
        centroids_pca_full = (centroids_std_space - pca.mean_) @ pca.components_[:n_full].T
        centroids_view = centroids_pca_full[:, :PCA_N_VIEW_COMPONENTS]
        centroids_orig_space = scaler.inverse_transform(centroids_std_space)

    apply_validation_result(
        run_clustering_output_validations(
            labels, cluster_centers_native,
            n_rows=n_rows, k=best_k, n_features=X_for_kmeans.shape[1],
        ),
        log,
    )

    # ---- Persist clusters + centroids ----------------------------------------
    pd.DataFrame(
        {
            "row_id": np.arange(n_rows, dtype=np.int64),
            "cluster": labels,
            "PC1": X_pca_view[:, 0],
            "PC2": X_pca_view[:, 1],
            "PC3": X_pca_view[:, 2],
        },
    ).to_csv(OUT_CLUSTERS_CSV, index=False)
    apply_validation_result(
        validate_written_csv(OUT_CLUSTERS_CSV, n_rows, label="Clusters CSV"), log,
    )

    pd.DataFrame(
        centroids_orig_space, columns=feature_names,
        index=[f"C{i}" for i in range(best_k)],
    ).to_csv(OUT_CENTROIDS_ORIG_CSV, index_label="cluster")

    pd.DataFrame(
        centroids_view, columns=["PC1", "PC2", "PC3"],
        index=[f"C{i}" for i in range(best_k)],
    ).to_csv(OUT_CENTROIDS_PCA_CSV, index_label="cluster")

    pd.DataFrame(
        centroids_pca_full,
        columns=[f"PC{i+1}" for i in range(centroids_pca_full.shape[1])],
        index=[f"C{i}" for i in range(best_k)],
    ).to_csv(OUT_CENTROIDS_PCA_FULL_CSV, index_label="cluster")

    meta = {
        "mode": {
            "CLUSTER_ON_PCA": CLUSTER_ON_PCA,
            "MIN_DUMMY_SUPPORT": MIN_DUMMY_SUPPORT,
            "MAX_DUMMY_SUPPORT": MAX_DUMMY_SUPPORT,
            "PCA_FULL_VARIANCE_TARGET": PCA_FULL_VARIANCE_TARGET,
            "MAX_PCA_COMPONENTS": MAX_PCA_COMPONENTS,
            "PCA_N_VIEW_COMPONENTS": PCA_N_VIEW_COMPONENTS,
        },
        "best_k": best_k,
        "k_selection_rule": k_pick_reason,
        "best_composite_score": best_composite,
        "best_silhouette_at_selected_k": best_silhouette,
        "k_argmax_silhouette_alone": k_sil_only,
        "k_range": list(K_RANGE),
        "n_rows": int(n_rows),
        "n_features_input": int(n_features_in),
        "n_features_kept": int(n_features_kept),
        "n_dummies_dropped": int(len(dummy_block_full) - len(kept_dummies)),
        "pca_n_components_full": int(n_full),
        "pca_cumulative_variance_full": info_full["cumulative_variance_ratio"],
        "pca_view_explained_variance_ratio": info_view["explained_variance_ratio"],
        "pca_view_cumulative_variance": info_view["cumulative_variance_ratio"],
        "cluster_space_dim": int(X_for_kmeans.shape[1]),
        "random_seed": RANDOM_SEED,
        "k_sweep_fit_max_rows": K_SWEEP_FIT_MAX_ROWS,
        "k_sweep_fit_rows": int(sweep["n_sweep_fit_rows"].iloc[0]) if "n_sweep_fit_rows" in sweep.columns else int(n_rows),
        "k_sweep_used_fit_subsample": bool(sweep["sweep_on_subsample"].iloc[0]) if "sweep_on_subsample" in sweep.columns else False,
        "metric_sample_size": int(sweep["metric_sample_size"].iloc[0]) if "metric_sample_size" in sweep.columns else min(SILHOUETTE_SAMPLE_SIZE, n_rows),
        "silhouette_sample_size_cap": SILHOUETTE_SAMPLE_SIZE,
        "silhouette_n_resamples": SILHOUETTE_N_RESAMPLES,
        "scatter_subsample_for_plot": SCATTER_SUBSAMPLE_FOR_PLOT,
        "k_sweep_kmeans": {
            "algorithm": "lloyd",
            "init": "k-means++",
            "n_init": FINAL_KMEANS_N_INIT,
            "max_iter": FINAL_KMEANS_MAX_ITER,
        },
        "final_kmeans": {
            "n_init": FINAL_KMEANS_N_INIT,
            "max_iter": FINAL_KMEANS_MAX_ITER,
            "inertia": float(final_model.inertia_),
            "n_iter": int(final_model.n_iter_),
        },
        "runtime_seconds": round(time.time() - t_start, 2),
        "input_csv": input_csv,
    }
    with open(OUT_KMEANS_META_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    log(
        f"Wrote clusters -> {OUT_CLUSTERS_CSV}; centroids -> "
        f"{OUT_CENTROIDS_ORIG_CSV}, {OUT_CENTROIDS_PCA_CSV}, {OUT_CENTROIDS_PCA_FULL_CSV}; "
        f"meta -> {OUT_KMEANS_META_JSON}",
    )

    # ---- Cluster-colored 3D scatters -----------------------------------------
    plot_pca_3d_clusters(
        X_pca_view, labels, info_view,
        os.path.join(PATH_RES_PCA_SCATTER, "02_pca_3d_clusters.png"),
        with_centroids=False,
    )
    plot_pca_3d_clusters(
        X_pca_view, labels, info_view,
        os.path.join(PATH_RES_PCA_SCATTER, "03_pca_3d_clusters_with_centroids.png"),
        with_centroids=True, centroids_pca=centroids_view,
    )
    plot_pca_3d_per_cluster(X_pca_view, labels, info_view, PATH_RES_PCA_PER_CLUSTER, log=log)
    log(f"Wrote PCA 3D scatter set -> {PATH_RES_PCA_SCATTER} and {PATH_RES_PCA_PER_CLUSTER}")

    _plot_cluster_sizes(
        labels, os.path.join(PATH_RESULTS_03, "07_cluster_sizes.png"),
    )

    # ---- Cluster interpretation ---------------------------------------------
    df_num_orig = _load_engineered_numerics(PATH_FS_CSV, log=log)
    if not df_num_orig.empty and df_num_orig.shape[0] == n_rows:
        means, z = _numeric_cluster_profiles(df_num_orig, labels)
        _plot_numeric_centroid_heatmap(
            z, os.path.join(PATH_RES_INTERP, "01_numeric_centroid_heatmap.png"),
        )
        _plot_parallel_coords(
            z, os.path.join(PATH_RES_INTERP, "03_parallel_coords_centroids.png"),
        )
        _plot_cluster_radar(
            z, os.path.join(PATH_RES_INTERP, "04_cluster_radar_numerics.png"),
        )
        for cluster in means.index:
            means.loc[[cluster]].T.to_csv(
                os.path.join(PATH_OUT_PROFILES, f"cluster_{int(cluster)}_numeric_means.csv"),
                index_label="feature",
            )
    else:
        log("Skipped numeric interpretation (Layer A CSV missing or row mismatch).")
        means, z = pd.DataFrame(), pd.DataFrame()

    # Reload kept-dummies block for lift analysis (smaller now → faster IO)
    dummies_only = pd.read_csv(input_csv, usecols=kept_dummies, low_memory=False)
    dummy_profiles = _dummy_lift_per_cluster(
        dummies_only, labels,
        top_k=TOP_DUMMIES_PER_CLUSTER,
        min_support=DUMMY_LIFT_MIN_SUPPORT,
    )
    _plot_top_dummies_per_cluster(
        dummy_profiles, os.path.join(PATH_RES_INTERP, "02_top_dummies_per_cluster.png"),
    )
    for cluster, prof in dummy_profiles.items():
        if not prof.empty:
            prof.to_csv(
                os.path.join(PATH_OUT_PROFILES, f"cluster_{int(cluster)}_dummy_prevalence.csv"),
                index=False,
            )
    del dummies_only
    gc.collect()

    sizes = {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
    if not z.empty:
        names = _name_clusters(z, dummy_profiles)
        _write_interpretation_summary(
            sizes, names, z, dummy_profiles,
            os.path.join(PATH_RES_INTERP, "cluster_interpretation_summary.txt"),
        )

    log(f"Step 03 complete in {time.time() - t_start:.1f}s")
    print(
        "PhilGEPS step 03 done. "
        f"K*={best_k} (sil@K={best_silhouette:.4f}, composite={best_composite:.4f}); "
        f"cluster_space_dim={X_for_kmeans.shape[1]}; "
        f"clusters: {OUT_CLUSTERS_CSV}; "
        f"results: {PATH_RESULTS_03}; logs: {PATH_LOGS_03}",
        flush=True,
    )


def main() -> None:
    _ensure_tree()
    term_log = os.path.join(
        PATH_LOG_TERMINAL, "03_kmeans_implementation_philgeps_terminal.txt",
    )
    try:
        with tee_stdio_to_file(term_log):
            run_step03()
    except Step03ValidationError as e:
        print(f"Step 03 validation failed: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
