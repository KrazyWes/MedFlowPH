"""
Step 02 — PhilGEPS preprocessing for k-means clustering.

Reads the step 01 cleaned CSV (MedFlowPH/output_source/01/philgeps_medical_procurement.csv)
and produces:

    output_source/02/philgeps_preprocessed.csv      (FINAL merged matrix — input to step 03)
    output_source/02/Feature Selection/
        philgeps_features_selected.csv              (Layer A: keep cols + engineered, mixed)
    output_source/02/One-Hot Encoding/
        philgeps_one_hot.csv                        (Layer B: dummies only, uint8)
    output_source/02/Min-Max Scaling/
        philgeps_min_max_scaled.csv                 (Layer C: scaled numerics only, float32)
        scaler_params.json                          (clip bounds + scaler state)

The final merged matrix is the layer-B dummies concatenated with the layer-C scaled
numerics, all in ``[0, 1]``, ready for ``KMeans`` + silhouette in step 03.

EDA charts (matplotlib + seaborn) are written under MedFlowPH/results/02/ so each layer
has its own visual narrative (Feature Selection / One-Hot Encoding / Min-Max Scaling).

Step 02 does NOT re-run step 01 jobs (no medical filter, no row dedup, no null
imputation). It re-coerces dtypes the on-disk CSV demoted to object and re-validates
step 01 invariants (no nulls on the kept columns) as a read-side safety check.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, TextIO

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from philgeps_step02_validation import (
    DATE_COLUMNS,
    KEEP_COLUMNS,
    MONEY_COLUMNS,
    Step02ValidationError,
    apply_validation_result,
    run_layer_a_validations,
    run_layer_c_validations,
    validate_finite_numeric,
    validate_no_constant_dummies,
    validate_no_dropped_columns,
    validate_one_hot_partition,
    validate_scaled_in_unit_interval,
    validate_written_csv,
)

# ---------------------------------------------------------------------------
# Paths (mirror step 01 layout, but for /02/)
# ---------------------------------------------------------------------------

_HERE = os.path.abspath(__file__)
MEDFLOW_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

PATH_INPUT_CSV = os.path.join(
    MEDFLOW_ROOT, "output_source", "01", "philgeps_medical_procurement.csv",
)

PATH_OUTPUT_02 = os.path.join(MEDFLOW_ROOT, "output_source", "02")
PATH_OUT_FS = os.path.join(PATH_OUTPUT_02, "Feature Selection")
PATH_OUT_OHE = os.path.join(PATH_OUTPUT_02, "One-Hot Encoding")
PATH_OUT_SCALE = os.path.join(PATH_OUTPUT_02, "Min-Max Scaling")

PATH_RESULTS_02 = os.path.join(MEDFLOW_ROOT, "results", "02")
PATH_RES_FS = os.path.join(PATH_RESULTS_02, "Feature Selection")
PATH_RES_OHE = os.path.join(PATH_RESULTS_02, "One-Hot Encoding")
PATH_RES_SCALE = os.path.join(PATH_RESULTS_02, "Min-Max Scaling")

PATH_LOGS_02 = os.path.join(MEDFLOW_ROOT, "logs", "02")
PATH_LOG_TERMINAL = os.path.join(PATH_LOGS_02, "Terminal Logs")
PATH_LOG_ENTRIES = os.path.join(PATH_LOGS_02, "Log entries")

# Output CSVs — pure-blocks layout
OUT_PREPROC_CSV = os.path.join(PATH_OUTPUT_02, "philgeps_preprocessed.csv")
OUT_FS_CSV = os.path.join(PATH_OUT_FS, "philgeps_features_selected.csv")
OUT_OHE_CSV = os.path.join(PATH_OUT_OHE, "philgeps_one_hot.csv")
OUT_SCALED_CSV = os.path.join(PATH_OUT_SCALE, "philgeps_min_max_scaled.csv")
OUT_SCALER_JSON = os.path.join(PATH_OUT_SCALE, "scaler_params.json")

# ---------------------------------------------------------------------------
# Encoding policy (matches the agreed plan)
# ---------------------------------------------------------------------------

LOW_CARD_OHE: tuple[str, ...] = (
    "Government Branch",
    "PE Organization Type (Grouped)",
    "Notice Type",
    "Classification",
    "Procurement Mode",
    "Funding Source",
    "Funding Instrument",
    "Trade Agreement",
    "Calendar Type",
    "Notice Status",
    "Bid Notice Status",
    "Award Type",
    "Award Notice Status",
    "Award Status",
    "Country of Awardee",
    "Region",
    "Region of Awardee",
    "Awardee Size",
    "Client Agency",  # ~50 levels in this dataset
)

TOPK_PLUS_OTHER: dict[str, int] = {
    "Province": 25,
    "Province of Awardee": 25,
    "Area of Delivery": 25,
    "UOM": 20,
    "Business Category": 30,
}

FREQ_ENCODE: tuple[str, ...] = (
    "City/Municipality",
    "City/Municipality of Awardee",
)

CALENDAR_TO_DAYS: dict[str, float] = {"Day/s": 1.0, "Month/s": 30.0, "Year/s": 365.0}

ROBUST_CLIP_LOW = 0.01
ROBUST_CLIP_HIGH = 0.99


# ---------------------------------------------------------------------------
# Logging helpers (mirror step 01)
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
        PATH_OUTPUT_02, PATH_OUT_FS, PATH_OUT_OHE, PATH_OUT_SCALE,
        PATH_RESULTS_02, PATH_RES_FS, PATH_RES_OHE, PATH_RES_SCALE,
        PATH_LOGS_02, PATH_LOG_TERMINAL, PATH_LOG_ENTRIES,
    ):
        os.makedirs(p, exist_ok=True)


def _slug(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out).strip("_")
    return s or "col"


# ---------------------------------------------------------------------------
# Layer A — load + feature engineering
# ---------------------------------------------------------------------------


def _read_cleaned(input_csv: str) -> pd.DataFrame:
    """Load the step 01 cleaned CSV but only the keep columns (handles ' Client Agency' header)."""
    header = pd.read_csv(input_csv, nrows=0)
    rename = {c: c.strip() for c in header.columns if c != c.strip()}
    cols_in_file = [rename.get(c, c) for c in header.columns]
    keep_present = [c for c in KEEP_COLUMNS if c in cols_in_file]
    inv_rename = {v: k for k, v in rename.items()}
    usecols_ondisk = [inv_rename.get(c, c) for c in keep_present]

    df = pd.read_csv(input_csv, usecols=usecols_ondisk, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _coerce_datetime(s: pd.Series) -> pd.Series:
    """Parse to datetime, tolerating mixed ``dd/mm/yyyy`` + ISO formats (pandas-3.0 safe)."""
    import warnings as _warnings

    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True, utc=False)


def _diff_days(left: pd.Series, right: pd.Series) -> pd.Series:
    delta = (left - right).dt.total_seconds() / 86400.0
    return delta


def feature_engineer(
    df_raw: pd.DataFrame,
    *,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Layer A — keep cols + engineered numerics; drop raw dates/money/duration."""
    df = df_raw.copy()
    diag: dict[str, Any] = {}

    # 1) Money: numeric, clip negatives, log1p
    for c in MONEY_COLUMNS:
        if c in df.columns:
            s = _coerce_numeric(df[c])
            n_neg = int((s < 0).sum())
            if n_neg:
                log(f"Money {c!r}: clipped {n_neg:,} negative values to 0")
            s = s.clip(lower=0).fillna(0.0)
            df[c] = s
            df[f"log1p_{_slug(c)}"] = np.log1p(s.to_numpy())

    # Quantity -> log1p
    if "Quantity" in df.columns:
        q = _coerce_numeric(df["Quantity"]).clip(lower=0).fillna(0.0)
        df["Quantity"] = q
        df["log1p_Quantity"] = np.log1p(q.to_numpy())

    # 2) Contract Duration -> contract_duration_days using Calendar Type
    if "Contract Duration" in df.columns and "Calendar Type" in df.columns:
        dur = _coerce_numeric(df["Contract Duration"]).fillna(0.0)
        mult = df["Calendar Type"].map(CALENDAR_TO_DAYS).fillna(1.0).astype(float)
        df["contract_duration_days"] = (dur * mult).astype(float)

    # 3) Collapse the typo-duplicate effectivity columns into one canonical
    canonical = "Contract Efectivity Date"
    typo_dup = "Contract Effectivity Date"
    if canonical in df.columns and typo_dup in df.columns:
        a = _coerce_datetime(df[canonical])
        b = _coerce_datetime(df[typo_dup])
        merged = a.fillna(b)
        df[canonical] = merged
        both = a.notna() & b.notna()
        diff = (a[both] != b[both]).sum() if both.any() else 0
        diag["effectivity_dup"] = {
            "rows_both_present": int(both.sum()),
            "rows_disagreeing": int(diff),
        }
        log(
            f"Effectivity merge: both_present={diag['effectivity_dup']['rows_both_present']:,}, "
            f"disagreeing={diag['effectivity_dup']['rows_disagreeing']:,}",
        )
        df.drop(columns=[typo_dup], inplace=True)

    # 4) Parse all date columns once
    parsed_dates: dict[str, pd.Series] = {}
    for c in DATE_COLUMNS:
        if c in df.columns:
            parsed_dates[c] = _coerce_datetime(df[c])

    pub = parsed_dates.get("Published Date")
    close_ = parsed_dates.get("Closing Date")
    prebid = parsed_dates.get("PreBid Date")
    pub_award = parsed_dates.get("Published Date(Award)")
    award = parsed_dates.get("Award Date")
    ntp = parsed_dates.get("Notice to Proceed Date")
    eff = parsed_dates.get("Contract Efectivity Date")
    end = parsed_dates.get("Contract End Date")

    def _engineer(name: str, series: pd.Series, *, low: float, high: float) -> None:
        v = series.fillna(0.0).clip(lower=low, upper=high)
        df[name] = v.astype(float)

    if pub is not None:
        df["pub_year"] = pub.dt.year.fillna(0).astype(int)
        df["pub_month"] = pub.dt.month.fillna(0).astype(int)
        df["pub_dow"] = pub.dt.dayofweek.fillna(0).astype(int)
        epoch = (pub - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400.0
        df["pub_epoch_days"] = epoch.fillna(0.0).astype(float)

    if pub is not None and close_ is not None:
        _engineer("time_to_close_days", _diff_days(close_, pub), low=-30.0, high=3650.0)
    if pub is not None and prebid is not None:
        _engineer("prebid_lead_days", _diff_days(prebid, pub), low=-30.0, high=3650.0)
    if pub is not None and pub_award is not None:
        _engineer("award_publish_lag_days", _diff_days(pub_award, pub), low=-30.0, high=3650.0)
    if close_ is not None and award is not None:
        _engineer("award_decision_lag_days", _diff_days(award, close_), low=-30.0, high=3650.0)
    if award is not None and ntp is not None:
        _engineer("ntp_lag_days", _diff_days(ntp, award), low=-30.0, high=3650.0)
    if eff is not None and end is not None:
        _engineer("contract_length_days", _diff_days(end, eff), low=-30.0, high=3650.0)
    if award is not None and eff is not None:
        _engineer("contract_effectivity_lag_days", _diff_days(eff, award), low=-30.0, high=3650.0)

    # 5) Drop raw dates / contract duration / raw money / raw quantity
    drop_now: list[str] = []
    drop_now.extend(c for c in DATE_COLUMNS if c in df.columns)
    if "Contract Duration" in df.columns:
        drop_now.append("Contract Duration")
    drop_now.extend(c for c in MONEY_COLUMNS if c in df.columns)
    if "Quantity" in df.columns:
        drop_now.append("Quantity")
    df.drop(columns=drop_now, inplace=True)

    return df, diag


def topk_plus_other(series: pd.Series, k: int) -> tuple[pd.Series, pd.DataFrame]:
    vc = series.astype(str).value_counts(dropna=False)
    keep_levels = vc.head(k).index.tolist()
    coverage_rows = int(vc.head(k).sum())
    report = pd.DataFrame(
        {
            "level": vc.index,
            "count": vc.values,
            "kept": [lvl in keep_levels for lvl in vc.index],
        },
    )
    out = series.astype(str).where(series.astype(str).isin(keep_levels), other="Other")
    return out, report.assign(coverage_rows=coverage_rows, top_k=k)


def frequency_encode(series: pd.Series) -> pd.Series:
    counts = series.astype(str).value_counts(dropna=False)
    return series.astype(str).map(counts).astype(np.int64)


# ---------------------------------------------------------------------------
# Layer B — split into pure dummies + numerics carrier (no concat)
# ---------------------------------------------------------------------------


def encode_one_hot_split(
    df_a: pd.DataFrame,
    *,
    log: Callable[[str], None],
    res_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Encode all categoricals; return (dummies_only, numerics_only, width_rows).

    - High-cardinality cols become a numeric ``*_freq`` column (lives in numerics_only).
    - Top-K + "Other" reduces level set, then OHE.
    - Low/medium-card OHE with ``drop_first=True`` and ``dtype=uint8``.
    """
    df = df_a.copy()
    width_rows: list[dict[str, Any]] = []

    # 1) High-cardinality -> frequency encoding (numeric, lives with numerics)
    for c in FREQ_ENCODE:
        if c in df.columns:
            new_name = f"{_slug(c)}_freq"
            df[new_name] = frequency_encode(df[c])
            df.drop(columns=[c], inplace=True)
            log(f"Freq-encoded {c!r} -> {new_name}")
            width_rows.append({"source": c, "policy": "frequency", "new_columns": 1})

    # 2) Top-K + "Other" before OHE
    for c, k in TOPK_PLUS_OTHER.items():
        if c not in df.columns:
            continue
        collapsed, report = topk_plus_other(df[c], k=k)
        df[c] = collapsed
        report_path = os.path.join(res_dir, f"topk_{_slug(c)}.csv")
        report.to_csv(report_path, index=False)
        cov_top = float(collapsed.value_counts(normalize=True).drop(labels="Other", errors="ignore").sum())
        log(f"Top-K {c!r}: K={k} kept-levels coverage={cov_top:.1%} -> {report_path}")
        width_rows.append({"source": c, "policy": f"top_{k}_plus_other", "new_columns": -1})

    # 3) One-hot encode all categoricals (low-card + collapsed top-K)
    cat_cols = [c for c in (list(LOW_CARD_OHE) + list(TOPK_PLUS_OTHER.keys())) if c in df.columns]

    df_dummies_parts: list[pd.DataFrame] = []
    for c in cat_cols:
        block = pd.get_dummies(
            df[c].astype(str), prefix=_slug(c), drop_first=True, dtype=np.uint8,
        )
        v_part = validate_one_hot_partition(df[c], block, drop_first=True)
        apply_validation_result(v_part, log)
        df_dummies_parts.append(block)
        existing = next((r for r in width_rows if r["source"] == c), None)
        if existing:
            existing["new_columns"] = block.shape[1]
        else:
            width_rows.append({"source": c, "policy": "one_hot_drop_first", "new_columns": block.shape[1]})

    df_dummies = (
        pd.concat(df_dummies_parts, axis=1)
        if df_dummies_parts
        else pd.DataFrame(index=df.index)
    )

    # 4) Numerics carrier (everything that is NOT in cat_cols and NOT a dummy)
    df_numerics = df.drop(columns=cat_cols, errors="ignore").copy()

    # Persist width report
    width_df = pd.DataFrame(width_rows).sort_values(["policy", "source"])
    width_df.to_csv(os.path.join(res_dir, "ohe_width_per_source.csv"), index=False)

    # Constant-dummy advisory
    apply_validation_result(
        validate_no_constant_dummies(df_dummies, min_rate=0.0), log, raise_on_error=False,
    )

    log(
        f"OHE split: dummies={df_dummies.shape}, numerics={df_numerics.shape}, "
        f"sources_encoded={len(cat_cols)}",
    )
    return df_dummies, df_numerics, width_rows


# ---------------------------------------------------------------------------
# Layer C — min-max scale numerics only (already separated from dummies)
# ---------------------------------------------------------------------------


def min_max_scale_numerics(
    df_numerics: pd.DataFrame,
    *,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-numeric robust clip to ``[p1, p99]`` then ``MinMaxScaler`` to ``[0, 1]``."""
    df = df_numerics.copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    clip_bounds: dict[str, tuple[float, float]] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        lo = float(s.quantile(ROBUST_CLIP_LOW))
        hi = float(s.quantile(ROBUST_CLIP_HIGH))
        if not np.isfinite(lo):
            lo = float(s.min()) if np.isfinite(float(s.min())) else 0.0
        if not np.isfinite(hi):
            hi = float(s.max()) if np.isfinite(float(s.max())) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        df[c] = s.clip(lower=lo, upper=hi)
        clip_bounds[c] = (lo, hi)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    arr = df[numeric_cols].to_numpy(dtype=np.float64)
    scaled = scaler.fit_transform(arr)
    out = pd.DataFrame(scaled.astype(np.float32), index=df.index, columns=numeric_cols)
    out = out.clip(lower=0.0, upper=1.0)

    params = {
        "feature_range": [0.0, 1.0],
        "robust_clip": [ROBUST_CLIP_LOW, ROBUST_CLIP_HIGH],
        "columns": numeric_cols,
        "data_min_": list(map(float, scaler.data_min_.tolist())),
        "data_max_": list(map(float, scaler.data_max_.tolist())),
        "scale_": list(map(float, scaler.scale_.tolist())),
        "min_": list(map(float, scaler.min_.tolist())),
        "clip_bounds": {c: list(map(float, clip_bounds[c])) for c in numeric_cols},
    }
    log(f"Min-Max: scaled {len(numeric_cols)} numeric columns to [0, 1]")
    return out, params


# ---------------------------------------------------------------------------
# Reports / EDA — Feature Selection
# ---------------------------------------------------------------------------


def _write_cardinality_table(df: pd.DataFrame, out_dir: str) -> str:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_non_null": int(s.notna().sum()),
                "n_null": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "null_pct": (float(s.isna().sum()) / n * 100.0) if n else 0.0,
            },
        )
    snap = pd.DataFrame(rows).sort_values(["dtype", "n_unique"], ascending=[True, False])
    csv_path = os.path.join(out_dir, "feature_selection_cardinality.csv")
    txt_path = os.path.join(out_dir, "feature_selection_cardinality.txt")
    snap.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"PhilGEPS step 02 — feature-selection cardinality (rows={n:,})\n")
        f.write(snap.to_string(index=False))
        f.write("\n")
    return csv_path


def _eda_layer_a(df: pd.DataFrame, out_dir: str, *, log: Callable[[str], None]) -> None:
    """EDA visuals for Layer A (Feature Selection)."""
    sns.set_theme(style="whitegrid", context="notebook")
    n = len(df)

    # 01 dtype counts
    try:
        dtypes = df.dtypes.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        dtypes.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Layer A — column count by dtype")
        ax.set_xlabel("Number of columns")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_dtype_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA A 01 dtype_counts failed: {e}")

    # 02 missingness % per column
    try:
        miss = (df.isna().sum() / max(n, 1) * 100.0).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(10, len(miss) * 0.25), 5.5))
        miss.plot(kind="bar", ax=ax, color="darkorange")
        ax.set_title("Layer A — missingness (% of rows) per column")
        ax.set_ylabel("% missing")
        plt.xticks(rotation=90, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "02_missingness_pct.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA A 02 missingness failed: {e}")

    # 03 cardinality (top 30)
    try:
        nunique = df.nunique(dropna=True).sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10, max(6, len(nunique) * 0.22)))
        nunique.iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
        ax.set_title("Layer A — top 30 columns by unique values (cardinality)")
        ax.set_xlabel("Unique non-null values")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_cardinality_top30.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA A 03 cardinality failed: {e}")

    # 04 numeric correlation heatmap
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != np.uint8]
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr(numeric_only=True)
            corr.to_csv(os.path.join(out_dir, "numeric_correlation.csv"))
            sz = min(14, max(7, 0.5 + len(numeric_cols) * 0.6))
            fig, ax = plt.subplots(figsize=(sz, sz))
            sns.heatmap(
                corr, ax=ax, cmap="vlag", center=0,
                annot=len(numeric_cols) <= 16, fmt=".2f", linewidths=0.3, cbar_kws={"shrink": 0.7},
            )
            ax.set_title("Layer A — numeric correlation (engineered features)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "04_numeric_correlation.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA A 04 corr heatmap failed: {e}")

    # 05 numeric distributions (histograms small-multiples)
    if numeric_cols:
        try:
            cols_per_row = 4
            ncols = cols_per_row
            nrows = int(np.ceil(len(numeric_cols) / cols_per_row))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.8))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for ax in axes_flat[len(numeric_cols):]:
                ax.axis("off")
            for ax, c in zip(axes_flat, numeric_cols, strict=False):
                series = pd.to_numeric(df[c], errors="coerce").dropna()
                # Sample large columns for plotting speed
                if len(series) > 200_000:
                    series = series.sample(200_000, random_state=0)
                ax.hist(series, bins=40, color="teal", alpha=0.85)
                ax.set_title(c, fontsize=9)
                ax.tick_params(labelsize=7)
            fig.suptitle("Layer A — numeric distributions (engineered features)", fontsize=12)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            fig.savefig(os.path.join(out_dir, "05_numeric_distributions.png"), dpi=130)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA A 05 distributions failed: {e}")

        # 06 numeric boxplots small-multiples
        try:
            cols_per_row = 4
            ncols = cols_per_row
            nrows = int(np.ceil(len(numeric_cols) / cols_per_row))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.4))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for ax in axes_flat[len(numeric_cols):]:
                ax.axis("off")
            for ax, c in zip(axes_flat, numeric_cols, strict=False):
                series = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(series) > 200_000:
                    series = series.sample(200_000, random_state=0)
                ax.boxplot(series, vert=False, showfliers=True, patch_artist=True,
                           boxprops={"facecolor": "lightblue"})
                ax.set_title(c, fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_yticks([])
            fig.suptitle("Layer A — numeric boxplots", fontsize=12)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            fig.savefig(os.path.join(out_dir, "06_numeric_boxplots.png"), dpi=130)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA A 06 boxplots failed: {e}")

    # 07 top-K vs Other coverage (consolidated grid)
    topk_present = [c for c in TOPK_PLUS_OTHER.keys() if c in df.columns]
    if topk_present:
        try:
            cols_per_row = 2
            ncols = cols_per_row
            nrows = int(np.ceil(len(topk_present) / cols_per_row))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.0, nrows * 4.0))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for ax in axes_flat[len(topk_present):]:
                ax.axis("off")
            for ax, c in zip(axes_flat, topk_present, strict=False):
                vc = df[c].astype(str).value_counts().head(20)
                vc.iloc[::-1].plot(kind="barh", ax=ax, color="mediumpurple")
                ax.set_title(f"{c} (top {len(vc)} post top-K + Other)", fontsize=10)
                ax.set_xlabel("Row count", fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle("Layer A — Top-K + 'Other' coverage (high-card columns)", fontsize=12)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            fig.savefig(os.path.join(out_dir, "07_topk_grid.png"), dpi=140)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA A 07 topk grid failed: {e}")

    # 08 value counts (top 15) for low-card categoricals — consolidated grid
    low_present = [c for c in LOW_CARD_OHE if c in df.columns]
    if low_present:
        try:
            cols_per_row = 3
            ncols = cols_per_row
            nrows = int(np.ceil(len(low_present) / cols_per_row))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 3.6))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for ax in axes_flat[len(low_present):]:
                ax.axis("off")
            for ax, c in zip(axes_flat, low_present, strict=False):
                vc = df[c].astype(str).value_counts().head(10)
                vc.iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
                ax.set_title(f"{c} (top {len(vc)})", fontsize=10)
                ax.set_xlabel("Row count", fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle(
                "Layer A — Low-cardinality categorical value counts (top 10 each)", fontsize=12,
            )
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
            fig.savefig(os.path.join(out_dir, "08_categorical_grid.png"), dpi=140)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA A 08 categorical grid failed: {e}")


# ---------------------------------------------------------------------------
# Reports / EDA — One-Hot Encoding
# ---------------------------------------------------------------------------


def _eda_layer_b(
    df_dummies: pd.DataFrame,
    width_rows: list[dict[str, Any]],
    out_dir: str,
    *,
    log: Callable[[str], None],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    # 01 dummy count per source (from width_rows)
    try:
        wd = pd.DataFrame(width_rows).copy()
        wd = wd[wd["new_columns"] > 0]
        wd = wd.sort_values("new_columns", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(wd) * 0.28)))
        ax.barh(wd["source"], wd["new_columns"], color="steelblue")
        for y, v in enumerate(wd["new_columns"]):
            ax.text(v, y, f" {int(v)}", va="center", fontsize=8)
        ax.set_title("Layer B — number of new columns per source (after grouping + drop_first)")
        ax.set_xlabel("# columns produced")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_dummy_count_per_source.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA B 01 dummy count failed: {e}")

    if df_dummies.empty:
        return

    # 02 dummy base-rate distribution
    try:
        rates = df_dummies.mean(axis=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(rates.values, bins=40, color="coral", alpha=0.85)
        ax.set_title(f"Layer B — dummy base-rate distribution ({len(rates):,} dummies)")
        ax.set_xlabel("Mean (= base rate)")
        ax.set_ylabel("# dummies")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "02_dummy_base_rates.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA B 02 base-rate hist failed: {e}")

    # 03 sparsity per source group
    try:
        rates = df_dummies.mean(axis=0)
        prefixes: dict[str, list[float]] = {}
        for col, r in rates.items():
            pref = str(col).split("_", 1)[0]
            prefixes.setdefault(pref, []).append(float(r))
        rows = [
            {"source_prefix": p, "n_dummies": len(v), "mean_base_rate": float(np.mean(v))}
            for p, v in prefixes.items()
        ]
        sp = pd.DataFrame(rows).sort_values("n_dummies", ascending=True).tail(30)
        fig, ax = plt.subplots(figsize=(10, max(5, len(sp) * 0.28)))
        ax.barh(sp["source_prefix"], sp["mean_base_rate"], color="purple")
        ax.set_title("Layer B — mean dummy base-rate per source prefix (top 30)")
        ax.set_xlabel("Mean base rate")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_dummy_mean_rate_per_source.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA B 03 sparsity per source failed: {e}")

    # 04 top dummies by base rate (top 30)
    try:
        rates = df_dummies.mean(axis=0).sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10, max(5, len(rates) * 0.28)))
        rates.iloc[::-1].plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Layer B — top 30 dummies by base rate")
        ax.set_xlabel("Base rate")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "04_top_dummies_by_rate.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA B 04 top dummies failed: {e}")


# ---------------------------------------------------------------------------
# Reports / EDA — Min-Max Scaling
# ---------------------------------------------------------------------------


def _write_scaling_summary(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    scaler_params: dict[str, Any],
    out_dir: str,
) -> str:
    cols = scaler_params["columns"]
    rows: list[dict[str, Any]] = []
    for c in cols:
        lo, hi = scaler_params["clip_bounds"][c]
        s_pre = pd.to_numeric(df_pre[c], errors="coerce")
        s_post = df_post[c]
        rows.append(
            {
                "column": c,
                "pre_min": float(s_pre.min()),
                "pre_max": float(s_pre.max()),
                "pre_mean": float(s_pre.mean()),
                "pre_std": float(s_pre.std()),
                "clip_low": float(lo),
                "clip_high": float(hi),
                "post_min": float(s_post.min()),
                "post_max": float(s_post.max()),
                "post_mean": float(s_post.mean()),
                "post_std": float(s_post.std()),
            },
        )
    snap = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "scaling_summary.csv")
    txt_path = os.path.join(out_dir, "scaling_summary.txt")
    snap.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            f"PhilGEPS step 02 — Min-Max summary (n={len(df_post):,}, "
            f"scaled_columns={len(cols)})\n",
        )
        f.write(snap.to_string(index=False))
        f.write("\n")
    return csv_path


def _eda_layer_c(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    scaler_params: dict[str, Any],
    out_dir: str,
    *,
    log: Callable[[str], None],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    cols = scaler_params["columns"]
    if not cols:
        return

    # 01 pre/post histograms small-multiples
    try:
        cols_per_row = 4
        ncols = cols_per_row
        nrows = int(np.ceil(len(cols) / cols_per_row))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.0))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax in axes_flat[len(cols):]:
            ax.axis("off")
        for ax, c in zip(axes_flat, cols, strict=False):
            pre = pd.to_numeric(df_pre[c], errors="coerce").dropna()
            post = df_post[c].dropna()
            if len(pre) > 200_000:
                pre = pre.sample(200_000, random_state=0)
                post = post.loc[pre.index]
            ax2 = ax.twiny()
            ax.hist(pre, bins=40, color="lightcoral", alpha=0.7, label="pre (clipped)")
            ax2.hist(post, bins=40, color="navy", alpha=0.4, label="post [0,1]")
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
            ax2.tick_params(labelsize=7)
        fig.suptitle("Layer C — pre-clip vs post-scale distribution", fontsize=12)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(os.path.join(out_dir, "01_pre_post_distributions.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA C 01 pre/post failed: {e}")

    # 02 post-scale boxplots
    try:
        cols_per_row = 4
        ncols = cols_per_row
        nrows = int(np.ceil(len(cols) / cols_per_row))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.4))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax in axes_flat[len(cols):]:
            ax.axis("off")
        for ax, c in zip(axes_flat, cols, strict=False):
            post = df_post[c].dropna()
            if len(post) > 200_000:
                post = post.sample(200_000, random_state=0)
            ax.boxplot(post, vert=False, showfliers=True, patch_artist=True,
                       boxprops={"facecolor": "lightgreen"})
            ax.set_title(c, fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
        fig.suptitle("Layer C — post-scale boxplots (range = [0, 1])", fontsize=12)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(os.path.join(out_dir, "02_post_scale_boxplots.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA C 02 boxplots failed: {e}")

    # 03 mean and std bars
    try:
        means = df_post[cols].mean()
        stds = df_post[cols].std()
        order = means.sort_values().index
        fig, ax = plt.subplots(figsize=(11, max(5, len(cols) * 0.32)))
        x = np.arange(len(cols))
        ax.barh(order, means.loc[order], color="steelblue", label="mean", alpha=0.85)
        ax.barh(order, stds.loc[order], color="orange", label="std", alpha=0.55)
        ax.set_xlim(0.0, 1.0)
        ax.set_title("Layer C — post-scale mean and std per numeric")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_post_scale_summary.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"EDA C 03 summary bars failed: {e}")

    # 04 post-scale correlation heatmap
    if len(cols) >= 2:
        try:
            corr = df_post[cols].corr()
            sz = min(14, max(7, 0.5 + len(cols) * 0.6))
            fig, ax = plt.subplots(figsize=(sz, sz))
            sns.heatmap(
                corr, ax=ax, cmap="vlag", center=0,
                annot=len(cols) <= 16, fmt=".2f", linewidths=0.3, cbar_kws={"shrink": 0.7},
            )
            ax.set_title("Layer C — post-scale numeric correlation")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "04_post_scale_correlation.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            log(f"EDA C 04 correlation failed: {e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_step02(input_csv: str = PATH_INPUT_CSV) -> None:
    _ensure_tree()

    activity_path = os.path.join(PATH_LOG_ENTRIES, "02_data_preprocessing_philgeps_activity.txt")
    log = open_activity_log(activity_path)

    log(f"Reading cleaned step-01 CSV: {input_csv}")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Step-01 cleaned CSV not found: {input_csv}")

    df_in = _read_cleaned(input_csv)
    n_in = len(df_in)
    log(f"Loaded {n_in:,} rows × {df_in.shape[1]} kept columns")

    apply_validation_result(validate_no_dropped_columns(df_in), log)

    # ---------------- Layer A — feature engineering ---------------------------------
    log("Layer A: feature engineering")
    apply_validation_result(run_layer_a_validations(df_in), log, raise_on_error=False)
    df_a, diag = feature_engineer(df_in, log=log)
    del df_in
    gc.collect()

    df_a.to_csv(OUT_FS_CSV, index=False)
    apply_validation_result(
        validate_written_csv(OUT_FS_CSV, n_in, label="Feature Selection CSV"), log,
    )
    _write_cardinality_table(df_a, PATH_RES_FS)
    _eda_layer_a(df_a, PATH_RES_FS, log=log)
    log(f"Wrote Layer A CSV -> {OUT_FS_CSV} (shape={df_a.shape})")

    diag_path = os.path.join(PATH_RES_FS, "engineering_diag.json")
    with open(diag_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(diag, f, indent=2)
    log(f"Wrote engineering diag -> {diag_path}")

    # ---------------- Layer B — pure dummies + numerics carrier ---------------------
    log("Layer B: one-hot encoding (pure-blocks split)")
    df_dummies, df_numerics, width_rows = encode_one_hot_split(
        df_a, log=log, res_dir=PATH_RES_OHE,
    )
    del df_a
    gc.collect()

    df_dummies.to_csv(OUT_OHE_CSV, index=False)
    apply_validation_result(
        validate_written_csv(OUT_OHE_CSV, n_in, label="One-Hot CSV (dummies only)"), log,
    )
    _eda_layer_b(df_dummies, width_rows, PATH_RES_OHE, log=log)
    log(f"Wrote Layer B CSV (dummies only) -> {OUT_OHE_CSV} (shape={df_dummies.shape})")

    # ---------------- Layer C — scaled numerics only --------------------------------
    log("Layer C: min-max scaling (numerics only)")
    df_scaled, scaler_params = min_max_scale_numerics(df_numerics, log=log)
    df_scaled.to_csv(OUT_SCALED_CSV, index=False)
    with open(OUT_SCALER_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(scaler_params, f, indent=2)
    apply_validation_result(
        validate_written_csv(OUT_SCALED_CSV, n_in, label="Scaled CSV (numerics only)"), log,
    )
    apply_validation_result(run_layer_c_validations(df_scaled, scaler_params["columns"]), log)
    _write_scaling_summary(df_numerics, df_scaled, scaler_params, PATH_RES_SCALE)
    _eda_layer_c(df_numerics, df_scaled, scaler_params, PATH_RES_SCALE, log=log)
    log(f"Wrote Layer C CSV (scaled numerics only) -> {OUT_SCALED_CSV} (shape={df_scaled.shape})")
    log(f"Wrote scaler params -> {OUT_SCALER_JSON}")
    del df_numerics
    gc.collect()

    # ---------------- Final merge ---------------------------------------------------
    log("Merging dummies + scaled numerics into final preprocessed matrix")
    df_pre = pd.concat([df_scaled, df_dummies], axis=1)
    apply_validation_result(
        validate_finite_numeric(df_pre, df_pre.columns), log,
    )
    apply_validation_result(
        validate_scaled_in_unit_interval(df_pre, df_pre.columns), log,
    )
    df_pre.to_csv(OUT_PREPROC_CSV, index=False)
    apply_validation_result(
        validate_written_csv(OUT_PREPROC_CSV, n_in, label="Preprocessed (merged) CSV"), log,
    )
    log(f"Wrote final merged CSV -> {OUT_PREPROC_CSV} (shape={df_pre.shape})")

    print(
        "PhilGEPS step 02 done. "
        f"Final merged: {OUT_PREPROC_CSV} (shape={df_pre.shape}); "
        f"FS: {OUT_FS_CSV}; OHE: {OUT_OHE_CSV}; Scaled: {OUT_SCALED_CSV}; "
        f"results: {PATH_RESULTS_02}; logs: {PATH_LOGS_02}",
        flush=True,
    )


def main() -> None:
    _ensure_tree()
    term_log = os.path.join(
        PATH_LOG_TERMINAL, "02_data_preprocessing_philgeps_terminal.txt",
    )
    try:
        with tee_stdio_to_file(term_log):
            run_step02()
    except Step02ValidationError as e:
        print(f"Step 02 validation failed: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
