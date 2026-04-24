"""
Step 01 — PhilGEPS cleaning.

Loads procurement rows from MedFlowPH/raw_datasets/PhilGEPS/{2020..2025}/ (one file at a time),
normalizes dtypes, applies the step 00 medical keyword rule, drops duplicates within each file,
concatenates the filtered chunks, drops cross-file duplicates, imputes missing values, and exports
CSVs under MedFlowPH/output_source/01/ (combined + per-year). Avoids holding the full ~8M-row raw
merge in memory.

2022 raw exports are headerless: they are read with a fixed 46-column schema so
"Procuring Entity (PE)" and "Awardee Organization Name" match other years.

After cleaning:
- EDA (PNG) -> MedFlowPH/results/01/Exploratory Data Analysis/{year}/ (overview + numeric correlation only)
  and .../Exploratory Data Analysis/merged/ (missingness %, dtypes, dtype-family summary,
  numeric + categorical Cramér's V heatmaps, rows-by-year, raw vs cleaned comparison — no per-column PNGs)
- Summaries -> MedFlowPH/results/01/Summaries/ (TXT + philgeps_cleaning_summary_table.png)
- Cleaned schema table -> MedFlowPH/results/01/Data Schema/
- Terminal mirror -> MedFlowPH/logs/01/Terminal Logs/
- Timestamped activity -> MedFlowPH/logs/01/Log entries/

Medical row filter (aligned with step 00): keep if UNSPSC Description matches keywords OR
(if UNSPSC does not match) Item Name or Item Description matches — equivalent to union of
the three keyword tests; fallback recovers rows with empty/generic UNSPSC (e.g. some quarters).
"""

from __future__ import annotations

import contextlib
import gc
import os
import re
import sys
from datetime import datetime
from typing import Any, Callable, TextIO

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from philgeps_step01_validation import (
    Step01ValidationError,
    apply_validation_result,
    run_output_file_validations,
    run_post_imputation_validations,
    validate_after_concat,
    validate_raw_row_accounting,
)

# Low-cardinality categoricals only (pairwise Cramér's V is expensive / unreadable at high cardinality).
CRAMERS_V_MAX_CARDINALITY = 25
CRAMERS_V_MAX_COLUMNS = 16

# Project root: MedFlowPH/ (three levels up from source_code/PhilGEPS/this_file.py)
_HERE = os.path.abspath(__file__)
MEDFLOW_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# --- Paths (per MedFlowPH layout) ---
PATH_RAW_DATASETS = os.path.join(MEDFLOW_ROOT, "raw_datasets")
PATH_PHILGEPS_RAW = os.path.join(PATH_RAW_DATASETS, "PhilGEPS")
PATH_OUTPUT_01 = os.path.join(MEDFLOW_ROOT, "output_source", "01")
PATH_RESULTS_01 = os.path.join(MEDFLOW_ROOT, "results", "01")
PATH_LOGS_01 = os.path.join(MEDFLOW_ROOT, "logs", "01")
PATH_OUTPUT_YEARLY = os.path.join(PATH_OUTPUT_01, "PhilGEPS-yearly")

PATH_EDA_DIR = os.path.join(PATH_RESULTS_01, "Exploratory Data Analysis")
PATH_EDA_MERGED_DIR = os.path.join(PATH_EDA_DIR, "merged")
PATH_SUMMARIES_DIR = os.path.join(PATH_RESULTS_01, "Summaries")
PATH_DATA_SCHEMA_01 = os.path.join(PATH_RESULTS_01, "Data Schema")
PATH_TERMINAL_LOGS = os.path.join(PATH_LOGS_01, "Terminal Logs")
PATH_LOG_ENTRIES = os.path.join(PATH_LOGS_01, "Log entries")

# Raw exports sometimes read PE/geo/status fields as float64 (NaN for missing). Coerce to string before imputation.
FLOAT_LIKELY_TEXT_COLS: list[str] = [
    "Procuring Entity (PE)",
    "Region",
    "Province",
    "City/Municipality",
    "Government Branch",
    "PE Organization Type",
    "PE Organization Type (Grouped)",
    "Bid Notice Status",
    "Award Reference No.",
    "Award Notice Status",
    "Country of Awardee",
    "Region of Awardee",
    "Province of Awardee",
    "City/Municipality of Awardee",
    "Awardee Size",
    "Awardee Joint Venture",
    "Contract Effectivity Date",
]

CSV_EXTENSIONS = {".csv"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}

# Canonical PhilGEPS wide export (46 columns) — 2022 CSVs are shipped without a header row.
PHILGEPS_46_COLS: list[str] = [
    "Procuring Entity (PE)",
    "Region",
    "Province",
    "City/Municipality",
    "Government Branch",
    "PE Organization Type",
    "PE Organization Type (Grouped)",
    "Bid Reference No.",
    "Notice Title",
    "Classification",
    "Procurement Mode",
    "Business Category",
    "Funding Source",
    "Funding Instrument",
    "Trade Agreement",
    "Approved Budget of the Contract",
    "Published Date",
    "Closing Date",
    "Area of Delivery",
    "Contract Duration",
    "Calendar Type",
    "Line Item No",
    "Item Name",
    "Item Description",
    "Quantity",
    "UOM",
    "Item Budget",
    "Bid Notice Status",
    "Award Reference No.",
    "Award Title",
    "UNSPSC Code",
    "UNSPSC Description",
    "Published Date(Award)",
    "Award Date",
    "Contract Amount",
    "Award Notice Status",
    "Notice to Proceed Date",
    "Contract Effectivity Date",
    "Contract End Date",
    "Awardee Organization Name",
    "Country of Awardee",
    "Region of Awardee",
    "Province of Awardee",
    "City/Municipality of Awardee",
    "Awardee Size",
    "Awardee Joint Venture",
]

MEDICAL_KEYWORDS = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical",
]
PATTERN = "|".join(MEDICAL_KEYWORDS)

YEAR_FOLDERS = tuple(str(y) for y in range(2020, 2026))

SUMMARY_CAT_COLS = [
    "Procurement Mode",
    "Region",
    "PE Organization Type",
    "Awardee Size",
    "Funding Source",
]


def normalize_float_text_columns(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """Coerce columns that often arrive as float64 in raw exports to nullable string (step 00 schema)."""
    target = df if inplace else df.copy()
    for c in FLOAT_LIKELY_TEXT_COLS:
        if c not in target.columns:
            continue
        target[c] = target[c].map(lambda x: pd.NA if pd.isna(x) else str(x).strip())
        target[c] = target[c].replace("", pd.NA)
    return target


def _accumulate_med_stats(accum: dict[str, int] | None, chunk: dict[str, int]) -> dict[str, int]:
    if accum is None:
        return dict(chunk)
    return {k: accum[k] + chunk[k] for k in chunk}


def filter_medical_rows_step00_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Step 00 conservative rule: keep rows where UNSPSC Description matches keywords, or Item Name,
    or Item Description matches (fallback for null/generic UNSPSC, e.g. 2021_Q4-style exports).
    Returns (subset, stats) with overlap counts for logging.
    """
    n = len(df)
    if n == 0:
        return df.copy(), {"input_rows": 0, "output_rows": 0, "unspsc_match": 0, "item_name_match": 0, "item_desc_match": 0, "fallback_only_rows": 0}

    col_u = "UNSPSC Description"
    col_in = "Item Name"
    col_id = "Item Description"

    unspsc_match = (
        df[col_u].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_u in df.columns
        else pd.Series(False, index=df.index)
    )
    in_match = (
        df[col_in].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_in in df.columns
        else pd.Series(False, index=df.index)
    )
    id_match = (
        df[col_id].astype(str).str.contains(PATTERN, case=False, na=False)
        if col_id in df.columns
        else pd.Series(False, index=df.index)
    )

    keep = unspsc_match | in_match | id_match
    out = df.loc[keep].copy()

    # Rows that would be dropped under UNSPSC-only rule
    fallback_only = (~unspsc_match) & (in_match | id_match)

    stats = {
        "input_rows": n,
        "output_rows": len(out),
        "unspsc_match": int(unspsc_match.sum()),
        "item_name_match": int(in_match.sum()),
        "item_desc_match": int(id_match.sum()),
        "fallback_only_rows": int(fallback_only.sum()),
    }
    return out, stats


def _safe_filename_fragment(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return (s[:max_len] if s else "column") or "column"


@contextlib.contextmanager
def tee_stdio_to_file(path: str) -> Any:
    """Mirror stdout to a log file (creates parent dirs)."""
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


def _ensure_tree() -> None:
    for p in (
        PATH_PHILGEPS_RAW,
        PATH_OUTPUT_01,
        PATH_RESULTS_01,
        PATH_LOGS_01,
        PATH_OUTPUT_YEARLY,
        PATH_EDA_DIR,
        PATH_EDA_MERGED_DIR,
        PATH_SUMMARIES_DIR,
        PATH_DATA_SCHEMA_01,
        PATH_TERMINAL_LOGS,
        PATH_LOG_ENTRIES,
    ):
        os.makedirs(p, exist_ok=True)


def build_step01_layout() -> dict[str, str]:
    """MedFlowPH-local result folders for step 01."""
    return {
        "eda_dir": PATH_EDA_DIR,
        "eda_merged_dir": PATH_EDA_MERGED_DIR,
        "summaries_dir": PATH_SUMMARIES_DIR,
        "schema_dir": PATH_DATA_SCHEMA_01,
    }


def open_activity_log(activity_path: str) -> Callable[[str], None]:
    """Append-only activity log with ISO-8601 timestamps; truncates file at run start."""
    os.makedirs(os.path.dirname(activity_path) or ".", exist_ok=True)
    if os.path.isfile(activity_path):
        os.remove(activity_path)

    def _log(msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with open(activity_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{ts}] {msg}\n")

    return _log


def get_supported_files_by_year() -> list[tuple[str, int]]:
    """
    Return (path, year) for each file under .../PhilGEPS/2020 .. 2025/
    (only direct children of the year folder).
    """
    out: list[tuple[str, int]] = []
    for ydir in YEAR_FOLDERS:
        folder = os.path.join(PATH_PHILGEPS_RAW, ydir)
        if not os.path.isdir(folder):
            continue
        year = int(ydir)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in CSV_EXTENSIONS or ext in EXCEL_EXTENSIONS:
                out.append((path, year))
    return out


def _read_csv_resilient(path: str, header: int | None, names: list[str] | None) -> pd.DataFrame:
    return pd.read_csv(path, header=header, names=names, low_memory=False, encoding="utf-8")


def _load_excel_all_sheets(path: str) -> list[pd.DataFrame]:
    return [df.copy() for _, df in pd.read_excel(path, sheet_name=None, dtype=object).items()]


def _first_cell_looks_like_header(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 1:
        return False
    v = str(df.iloc[0, 0]).strip()
    return v.lower().startswith("procuring") or "procuring entity" in v.lower()


def load_philgeps_dataframe(path: str, year: int) -> list[pd.DataFrame]:
    """
    Load one raw file. 2022 CSVs have no header row — use PHILGEPS_46_COLS.
    For Excel, use the year folder: 2022 uses header=None when the sheet does not look headed.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in CSV_EXTENSIONS:
        if year == 2022:
            df = _read_csv_resilient(path, header=None, names=PHILGEPS_46_COLS)
        else:
            df = _read_csv_resilient(path, header=0, names=None)
        return [df]
    if ext in EXCEL_EXTENSIONS:
        sheets = _load_excel_all_sheets(path)
        if year == 2022:
            out: list[pd.DataFrame] = []
            for s in sheets:
                if s.shape[1] == len(PHILGEPS_46_COLS) and not _first_cell_looks_like_header(s):
                    s.columns = PHILGEPS_46_COLS
                elif s.shape[1] == len(PHILGEPS_46_COLS) and all(str(c) in PHILGEPS_46_COLS for c in s.columns):
                    s = s.reindex(columns=PHILGEPS_46_COLS)
                out.append(s)
            return out
        return sheets
    return []


def generate_eda_visualizations(
    df: pd.DataFrame,
    out_dir: str,
    dataset_label: str = "PhilGEPS",
    *,
    include_per_column_charts: bool = True,
) -> None:
    """
    Write EDA PNGs: overview (nulls, dtypes), optional correlation heatmap, and optionally
    per-column histogram/box/bar charts (omit for merged folder to avoid duplicating per-year detail).
    """
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # --- Overview: null counts (all columns) ---
    try:
        nulls = df.isnull().sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(10, len(nulls) * 0.25), 6))
        nulls.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{dataset_label} — missing values per column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Null count")
        plt.xticks(rotation=90, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "00_overview_null_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA: overview null counts failed: {e}", flush=True)

    # --- Overview: dtype counts ---
    try:
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        dtype_counts.plot(kind="barh", ax=ax, color="coral")
        ax.set_title(f"{dataset_label} — column count by dtype")
        ax.set_xlabel("Number of columns")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_overview_dtype_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA: dtype counts failed: {e}", flush=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in datetime_cols
    ]

    # --- Correlation heatmap (numeric only) ---
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(min(14, 0.5 + len(numeric_cols)), min(12, 0.5 + len(numeric_cols))))
            sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=len(numeric_cols) <= 12, fmt=".2f", linewidths=0.3)
            ax.set_title(f"{dataset_label} — numeric correlation heatmap")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "02_correlation_heatmap_numeric.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA: correlation heatmap failed: {e}", flush=True)

    if not include_per_column_charts:
        return

    # --- Per numeric column: histogram + KDE, boxplot ---
    for col in numeric_cols:
        slug = _safe_filename_fragment(col)
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(series, kde=True, ax=ax, color="teal")
            ax.set_title(f"{dataset_label} — {col}\n(histogram + KDE)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"num_{slug}_histogram_kde.png"), dpi=120)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA: histogram for {col!r} failed: {e}", flush=True)

        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x=series, ax=ax, color="lightblue")
            ax.set_title(f"{dataset_label} — {col} (boxplot)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"num_{slug}_boxplot.png"), dpi=120)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA: boxplot for {col!r} failed: {e}", flush=True)

    # --- Datetime columns: treat as categorical (top 15 stringified values) ---
    for col in datetime_cols:
        slug = _safe_filename_fragment(col)
        try:
            vc = df[col].dropna().astype(str).value_counts().head(15)
            if vc.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, max(4, len(vc) * 0.35)))
            vc.sort_values().plot(kind="barh", ax=ax, color="mediumpurple")
            ax.set_title(f"{dataset_label} — {col}\n(top 15 values)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"dt_{slug}_top15.png"), dpi=120)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA: datetime bar for {col!r} failed: {e}", flush=True)

    # --- Categorical / object / bool: top 15 value counts ---
    for col in categorical_cols:
        slug = _safe_filename_fragment(col)
        try:
            vc = df[col].astype(str).replace("nan", "NaN").value_counts().head(15)
            if vc.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, max(4, len(vc) * 0.35)))
            vc.sort_values().plot(kind="barh", ax=ax, color="seagreen")
            ax.set_title(f"{dataset_label} — {col}\n(top 15 categories)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"cat_{slug}_top15.png"), dpi=120)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA: categorical bar for {col!r} failed: {e}", flush=True)


def chart_rows_by_year(df: pd.DataFrame, out_dir: str, dataset_label: str) -> None:
    """Bar chart of row counts by Year (merged cleaned data only)."""
    if "Year" not in df.columns or df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    try:
        vc = df["Year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        vc.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{dataset_label} — row counts by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rows")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_rows_by_year.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA: rows-by-year chart failed: {e}", flush=True)


def _dtype_family(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string/object"


def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """Cramér's V association for two categorical columns (0..1)."""
    a = col_a.astype(str).fillna("<NA>")
    b = col_b.astype(str).fillna("<NA>")
    ct = pd.crosstab(a, b)
    if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    try:
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
    except ValueError:
        return 0.0
    n = float(ct.values.sum())
    if n <= 0 or not np.isfinite(chi2):
        return 0.0
    r, k = ct.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(np.sqrt(max(chi2, 0.0) / denom))


def _categorical_columns_for_cramers(df: pd.DataFrame) -> list[str]:
    num = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    out: list[tuple[str, int]] = []
    for c in df.columns:
        if c in num or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        try:
            nu = int(df[c].nunique(dropna=True))
        except (TypeError, ValueError):
            nu = int(df[c].astype(str).nunique(dropna=True))
        if nu < 2 or nu > CRAMERS_V_MAX_CARDINALITY:
            continue
        out.append((c, nu))
    out.sort(key=lambda x: x[1])
    return [c for c, _ in out[:CRAMERS_V_MAX_COLUMNS]]


def generate_merged_eda_charts(
    df: pd.DataFrame,
    out_dir: str,
    raw_rows_by_year: dict[int, int],
    dataset_label: str = "PhilGEPS merged (cleaned medical)",
) -> None:
    """
    Merged-folder EDA only: missingness %, dtype counts, dtype-family overview,
    numeric Pearson heatmap, categorical Cramér's V heatmap (low-cardinality cols),
    cleaned rows-by-year, grouped raw vs cleaned bars, stacked retention by year.
    """
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    n = len(df)

    # 01 — missingness % (counts are redundant: pct = 100 * count / n)
    try:
        if n:
            miss_pct = (df.isnull().sum() / n * 100.0).sort_values(ascending=False)
        else:
            miss_pct = pd.Series(dtype=float)
        fig, ax = plt.subplots(figsize=(max(10, len(miss_pct) * 0.25), 6))
        miss_pct.plot(kind="bar", ax=ax, color="darkorange")
        ax.set_title(f"{dataset_label} — missingness (% of rows) per column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Percent missing")
        plt.xticks(rotation=90, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_overview_missingness_pct.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA merged: missingness % failed: {e}", flush=True)

    # 02 — dtype column counts
    try:
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        dtype_counts.plot(kind="barh", ax=ax, color="coral")
        ax.set_title(f"{dataset_label} — number of columns by pandas dtype")
        ax.set_xlabel("Number of columns")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "02_overview_dtype_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA merged: dtype counts failed: {e}", flush=True)

    # 03 — dtype family: column counts + mean missing % per family
    try:
        families: list[str] = []
        miss_rates: list[float] = []
        fam_counts: dict[str, int] = {}
        fam_miss_sum: dict[str, float] = {}
        for col in df.columns:
            fam = _dtype_family(df[col].dtype)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            mrate = float(df[col].isna().mean()) if n else 0.0
            fam_miss_sum[fam] = fam_miss_sum.get(fam, 0.0) + mrate
        fam_order = sorted(fam_counts.keys())
        counts = [fam_counts[f] for f in fam_order]
        avg_miss = [fam_miss_sum[f] / fam_counts[f] for f in fam_order]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.bar(fam_order, counts, color="seagreen")
        ax0.set_title("Columns per dtype family")
        ax0.set_ylabel("Column count")
        ax0.tick_params(axis="x", rotation=25)

        ax1.bar(fam_order, [x * 100.0 for x in avg_miss], color="slateblue")
        ax1.set_title("Mean % missing (within columns of each family)")
        ax1.set_ylabel("Avg. % missing")
        ax1.tick_params(axis="x", rotation=25)
        plt.suptitle(f"{dataset_label} — dtype families", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_dtype_families_overview.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA merged: dtype families failed: {e}", flush=True)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # 04 — numeric correlation
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(min(14, 0.5 + len(numeric_cols)), min(12, 0.5 + len(numeric_cols))))
            sns.heatmap(
                corr,
                ax=ax,
                cmap="vlag",
                center=0,
                annot=len(numeric_cols) <= 12,
                fmt=".2f",
                linewidths=0.3,
            )
            ax.set_title(f"{dataset_label} — numeric (Pearson) correlation")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "04_correlation_numeric.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA merged: numeric correlation failed: {e}", flush=True)

    # 05 — Cramér's V heatmap
    cat_cols = _categorical_columns_for_cramers(df)
    if len(cat_cols) >= 2:
        try:
            mat = np.eye(len(cat_cols))
            for i, ci in enumerate(cat_cols):
                for j in range(i + 1, len(cat_cols)):
                    v = _cramers_v(df[ci], df[cat_cols[j]])
                    mat[i, j] = v
                    mat[j, i] = v
            fig, ax = plt.subplots(figsize=(max(8, len(cat_cols) * 0.6), max(7, len(cat_cols) * 0.55)))
            sns.heatmap(
                mat,
                ax=ax,
                xticklabels=[_safe_filename_fragment(c, 40) for c in cat_cols],
                yticklabels=[_safe_filename_fragment(c, 40) for c in cat_cols],
                cmap="magma",
                vmin=0,
                vmax=1,
                annot=len(cat_cols) <= 10,
                fmt=".2f",
                linewidths=0.3,
            )
            ax.set_title(
                f"{dataset_label} — categorical association (Cramér's V)\n"
                f"columns: nunique ≤ {CRAMERS_V_MAX_CARDINALITY}, up to {CRAMERS_V_MAX_COLUMNS} cols",
            )
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.yticks(rotation=0, fontsize=7)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "05_correlation_categorical_cramers_v.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA merged: Cramér's V heatmap failed: {e}", flush=True)

    # 06 — cleaned medical rows by year
    if "Year" in df.columns and not df.empty:
        try:
            vc = df["Year"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            vc.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(f"{dataset_label} — row counts by Year (cleaned medical)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Rows")
            plt.xticks(rotation=0)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "06_rows_by_year_cleaned_medical.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA merged: rows-by-year failed: {e}", flush=True)

    # 07 — grouped: raw loaded vs cleaned by year
    if raw_rows_by_year and "Year" in df.columns and not df.empty:
        try:
            clean_vc = df["Year"].value_counts()
            years = sorted(set(raw_rows_by_year.keys()) | {int(y) for y in clean_vc.index})
            raw_counts = [raw_rows_by_year.get(y, 0) for y in years]
            clean_counts = [int(clean_vc.get(y, 0)) for y in years]
            x = np.arange(len(years))
            w = 0.35
            fig, ax = plt.subplots(figsize=(max(9, len(years) * 0.9), 5))
            ax.bar(x - w / 2, raw_counts, w, label="Raw loaded (all rows, step 01)", color="gray")
            ax.bar(x + w / 2, clean_counts, w, label="Medical cleaned (final)", color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels([str(y) for y in years])
            ax.set_xlabel("Year")
            ax.set_ylabel("Row count")
            ax.set_title(f"{dataset_label} — raw vs cleaned row counts by Year")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "07_raw_vs_cleaned_rows_by_year_grouped.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA merged: grouped raw vs cleaned failed: {e}", flush=True)

    # 08 — stacked: cleaned vs remainder of raw per year
    if raw_rows_by_year and "Year" in df.columns and not df.empty:
        try:
            clean_vc = df["Year"].value_counts()
            years = sorted(set(raw_rows_by_year.keys()) | {int(y) for y in clean_vc.index})
            raw_counts = [raw_rows_by_year.get(y, 0) for y in years]
            clean_counts = [int(clean_vc.get(y, 0)) for y in years]
            not_in_final = [max(0, r - c) for r, c in zip(raw_counts, clean_counts)]
            fig, ax = plt.subplots(figsize=(max(9, len(years) * 0.9), 5))
            x = np.arange(len(years))
            ax.bar(x, clean_counts, label="In final cleaned medical", color="steelblue")
            ax.bar(x, not_in_final, bottom=clean_counts, label="Raw minus final (filtered + deduped out)", color="lightgray")
            ax.set_xticks(x)
            ax.set_xticklabels([str(y) for y in years])
            ax.set_xlabel("Year")
            ax.set_ylabel("Row count")
            ax.set_title(f"{dataset_label} — raw load stacked: kept vs not in final output")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "08_raw_vs_cleaned_stacked_by_year.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA merged: stacked raw vs cleaned failed: {e}", flush=True)


def write_summary_table_png(
    df: pd.DataFrame,
    summaries_dir: str,
    *,
    medical_stats: dict[str, int] | None,
    raw_rows_by_year: dict[int, int],
    total_raw_rows: int,
    rows_final: int,
) -> str:
    """PNG table of key cleaning summary metrics for the merged medical dataset."""
    os.makedirs(summaries_dir, exist_ok=True)
    path = os.path.join(summaries_dir, "philgeps_cleaning_summary_table.png")
    ts = datetime.now().isoformat(timespec="seconds")
    rows_table: list[list[str]] = [
        ["Generated", ts],
        ["Final cleaned rows (medical)", f"{rows_final:,}"],
        ["Columns", str(df.shape[1])],
        ["Total raw rows loaded (step 01)", f"{total_raw_rows:,}"],
    ]
    if raw_rows_by_year:
        for y in sorted(raw_rows_by_year.keys()):
            rows_table.append([f"Raw rows loaded {y}", f"{raw_rows_by_year[y]:,}"])
    if medical_stats:
        for k, v in medical_stats.items():
            rows_table.append([k.replace("_", " "), f"{v:,}"])
    if "Year" in df.columns and not df.empty:
        vc = df["Year"].value_counts().sort_index()
        for y in vc.index:
            rows_table.append([f"Cleaned rows {int(y)}", f"{int(vc[y]):,}"])

    fig_h = max(4.0, 0.32 * len(rows_table) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows_table,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.35)
    ax.set_title("PhilGEPS step 01 — merged medical cleaning summary", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_cleaned_schema_table(df: pd.DataFrame, schema_dir: str) -> tuple[str, str]:
    """Write cleaned merged schema as TXT + CSV (aligned with step 00 raw schema layout). Returns (txt_path, csv_path)."""
    os.makedirs(schema_dir, exist_ok=True)
    n = len(df)
    rows: list[dict[str, Any]] = []
    canonical_set = set(PHILGEPS_46_COLS)

    for pos, col in enumerate(df.columns, start=1):
        s = df[col]
        null_count = int(s.isna().sum())
        nn = n - null_count
        try:
            nu = int(s.nunique(dropna=True))
        except (TypeError, ValueError):
            nu = int(s.astype(str).nunique())
        rows.append({
            "canonical_position": pos,
            "column_name": col,
            "in_canonical_46": col in canonical_set or col == "Year",
            "pandas_dtype": str(s.dtype),
            "row_count": n,
            "null_count": null_count,
            "null_pct": round(100.0 * null_count / n, 4) if n else 0.0,
            "non_null_count": nn,
            "nunique_non_null": nu,
        })

    schema_df = pd.DataFrame(rows)
    path_txt = os.path.join(schema_dir, "philgeps_cleaned_schema_table.txt")
    path_csv = os.path.join(schema_dir, "philgeps_cleaned_schema_table.csv")
    header = (
        "PhilGEPS cleaned merged schema (post step 01)\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Rows: {n:,}\n\n"
    )
    with open(path_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(schema_df.to_string(index=False))
        f.write("\n")
    schema_df.to_csv(path_csv, index=False)
    return path_txt, path_csv


def write_summary_txt(
    df: pd.DataFrame,
    summaries_dir: str,
    *,
    medical_stats: dict[str, int] | None = None,
    pipeline_notes: list[str] | None = None,
) -> str:
    """Write text summary of the cleaned merged dataset; returns output path."""
    os.makedirs(summaries_dir, exist_ok=True)
    path = os.path.join(summaries_dir, "philgeps_cleaning_summary.txt")
    lines: list[str] = [
        "PhilGEPS — merged medical procurement (after cleaning)",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Medical filter (step 00 aligned): rows kept if UNSPSC Description OR Item Name OR Item Description",
        "matches MEDICAL_KEYWORDS (regex). Recovers rows with missing/generic UNSPSC when item text matches.",
        "",
    ]
    if medical_stats:
        lines.append("--- Medical filter counts (summed over source files; pre within-file dedup) ---")
        for k, v in medical_stats.items():
            lines.append(f"  {k}: {v:,}")
        lines.append("")
    if pipeline_notes:
        lines.append("--- Pipeline ---")
        lines.extend(pipeline_notes)
        lines.append("")
    lines.extend([
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        "",
        "--- Column dtypes ---",
        df.dtypes.to_string(),
        "",
        "--- Null counts per column ---",
        df.isnull().sum().sort_values(ascending=False).to_string(),
        "",
    ])
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if num:
        lines.extend(["--- Numeric describe ---", df[num].describe().to_string(), ""])
    for col in SUMMARY_CAT_COLS:
        if col not in df.columns:
            continue
        lines.append(f"--- Value counts (top 10): {col} ---")
        lines.append(df[col].astype(str).value_counts().head(10).to_string())
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return path


def handle_missing_values_philgeps(
    df: pd.DataFrame,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Handle Missing Values: numeric->median, categorical->mode or 'N/A'."""
    df = df.copy()
    numeric_cols = [
        "Contract Amount", "Item Budget", "Quantity",
        "Approved Budget of the Contract", "Line Item No",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        n_null = int(series.isna().sum())
        if n_null:
            median_val = series.median()
            df[col] = series.fillna(median_val if pd.notna(median_val) else 0)
            if log:
                log(f"Imputed {n_null:,} nulls in {col!r} (median or 0)")

    cat_cols = [
        c for c in df.columns
        if c not in numeric_cols and not pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in cat_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n_null = int(df[col].isna().sum())
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "N/A"
        df[col] = (
            df[col]
            .fillna(fill_val)
            .replace("", fill_val)
            .replace("nan", "N/A")
            .replace("None", "N/A")
        )
        if n_null and log:
            log(f"Imputed {n_null:,} nulls in {col!r} (mode or 'N/A')")
    return df


def _attach_year_tag(df: pd.DataFrame, year: int) -> None:
    """
    Tag rows with the source year in place (no copy). 2022 alignment is handled at load time via
    PHILGEPS_46_COLS.
    """
    if "Year" in df.columns:
        df.drop(columns=["Year"], inplace=True)
    df["Year"] = year


def _run_cleaning_philgeps(
    output_combined: str,
    vis_paths: dict[str, str],
) -> None:
    activity_path = os.path.join(PATH_LOG_ENTRIES, "01_data_cleaning_philgeps_activity.txt")
    log_activity = open_activity_log(activity_path)

    files = get_supported_files_by_year()
    if not files:
        msg = f"PhilGEPS: No CSV/Excel files under {PATH_PHILGEPS_RAW}/{{2020..2025}}. Skipping."
        log_activity(msg)
        print(msg, flush=True)
        return

    medical_chunks: list[pd.DataFrame] = []
    med_stats_total: dict[str, int] | None = None
    total_raw_rows = 0
    rows_after_chunk_dedup = 0
    raw_rows_by_year: dict[int, int] = {}

    for filepath, year in files:
        try:
            for part_i, df in enumerate(load_philgeps_dataframe(filepath, year)):
                _attach_year_tag(df, year)
                yk = int(year)
                raw_rows_by_year[yk] = raw_rows_by_year.get(yk, 0) + len(df)
                total_raw_rows += len(df)
                base = os.path.basename(filepath)
                label = f"PhilGEPS - {base} (year {year})" + (f" / part {part_i}" if part_i else "")

                normalize_float_text_columns(df, inplace=True)
                med_part, st = filter_medical_rows_step00_rule(df)
                med_stats_total = _accumulate_med_stats(med_stats_total, st)

                before_part_dedup = len(med_part)
                med_part = med_part.drop_duplicates()
                removed_part = before_part_dedup - len(med_part)
                if len(med_part):
                    medical_chunks.append(med_part)
                    rows_after_chunk_dedup += len(med_part)

                log_activity(
                    f"Chunk {label}: raw={len(df):,}, after_medical={before_part_dedup:,}, "
                    f"after_within_file_dedup={len(med_part):,} (dropped {removed_part:,})",
                )
                print(f"PhilGEPS: Loaded {label} ({len(df):,} raw rows)", flush=True)
                del df
                gc.collect()
        except Exception as e:  # noqa: BLE001
            log_activity(f"Error loading {filepath}: {e}")
            print(f"PhilGEPS: Error loading {filepath}: {e}", flush=True)

    if not medical_chunks:
        log_activity("No medical rows retained from any file.")
        print("PhilGEPS: No medical rows retained from any file.", flush=True)
        return

    log_activity(
        f"Chunked pipeline totals: raw_rows={total_raw_rows:,}, "
        f"sum_medical_pre_chunk_dedup={med_stats_total['output_rows']:,}, "
        f"rows_after_within_file_dedup={rows_after_chunk_dedup:,}",
    )

    v_acct = validate_raw_row_accounting(med_stats_total, total_raw_rows, raw_rows_by_year)
    apply_validation_result(v_acct, log_activity)

    medical_df = pd.concat(medical_chunks, ignore_index=True)
    del medical_chunks
    gc.collect()
    log_activity(f"Concatenated medical chunks: {len(medical_df):,} rows (before cross-file dedup)")

    v_cat = validate_after_concat(med_stats_total, rows_after_chunk_dedup, len(medical_df))
    apply_validation_result(v_cat, log_activity)

    if med_stats_total:
        log_activity(
            "Medical filter (summed over files): "
            f"input={med_stats_total['input_rows']:,}, output_pre_chunk_dedup={med_stats_total['output_rows']:,}, "
            f"unspsc_kw={med_stats_total['unspsc_match']:,}, item_name_kw={med_stats_total['item_name_match']:,}, "
            f"item_desc_kw={med_stats_total['item_desc_match']:,}, "
            f"rows_with_fallback_text_match={med_stats_total['fallback_only_rows']:,}",
        )
    print(
        f"PhilGEPS: Concatenated medical rows (before cross-file dedup): {len(medical_df):,}",
        flush=True,
    )

    before_dedup = len(medical_df)
    medical_df = medical_df.drop_duplicates()
    log_activity(
        f"Cross-file dedup: removed {before_dedup - len(medical_df):,} rows; "
        f"{len(medical_df):,} rows remain",
    )

    medical_df = handle_missing_values_philgeps(medical_df, log=log_activity)
    log_activity("Completed missing-value handling (numeric median / categorical mode or N/A)")

    v_post = run_post_imputation_validations(medical_df, PATTERN)
    apply_validation_result(v_post, log_activity)
    log_activity("Post-imputation validations passed")

    eda_root = vis_paths["eda_dir"]
    eda_merged = vis_paths["eda_merged_dir"]
    years_iter = sorted(medical_df["Year"].dropna().unique()) if "Year" in medical_df.columns else []
    for y in years_iter:
        ydir = os.path.join(eda_root, str(int(y)))
        ysub = medical_df[medical_df["Year"] == y]
        generate_eda_visualizations(
            ysub,
            ydir,
            dataset_label=f"PhilGEPS {int(y)} (cleaned)",
            include_per_column_charts=False,
        )
        log_activity(f"Wrote per-year EDA -> {ydir} ({len(ysub):,} rows)")

    generate_merged_eda_charts(
        medical_df,
        eda_merged,
        raw_rows_by_year,
        dataset_label="PhilGEPS merged (cleaned medical)",
    )
    log_activity(f"Wrote merged EDA (dtype/missingness/correlations/raw vs cleaned) -> {eda_merged}")

    pipeline_notes = [
        f"Total raw rows loaded (all files): {total_raw_rows:,}",
        f"After medical filter (sum over files, pre within-file dedup): {med_stats_total['output_rows']:,}",
        f"After within-file dedup (sum of chunk sizes): {rows_after_chunk_dedup:,}",
        f"After concat + cross-file drop_duplicates: {len(medical_df):,}",
        f"Per-year EDA under: {eda_root}/<year>/",
        f"Merged EDA under: {eda_merged}/",
    ]
    summary_path = write_summary_txt(
        medical_df,
        vis_paths["summaries_dir"],
        medical_stats=med_stats_total,
        pipeline_notes=pipeline_notes,
    )
    log_activity(f"Wrote summary: {summary_path}")

    summary_png = write_summary_table_png(
        medical_df,
        vis_paths["summaries_dir"],
        medical_stats=med_stats_total,
        raw_rows_by_year=raw_rows_by_year,
        total_raw_rows=total_raw_rows,
        rows_final=len(medical_df),
    )
    log_activity(f"Wrote summary table PNG: {summary_png}")

    schema_txt, schema_csv = write_cleaned_schema_table(medical_df, vis_paths["schema_dir"])
    log_activity(f"Wrote cleaned schema: {schema_txt}, {schema_csv}")

    os.makedirs(PATH_OUTPUT_YEARLY, exist_ok=True)
    yearly_write_plan: list[tuple[str, int]] = []
    for year, ysub in medical_df.groupby("Year", sort=True):
        ypath = os.path.join(PATH_OUTPUT_YEARLY, f"philgeps_{int(year)}_medical_procurement.csv")
        yearly_write_plan.append((ypath, len(ysub)))
        ysub.to_csv(ypath, index=False)
        log_activity(f"Saved {len(ysub):,} records (year {int(year)}) -> {ypath}")
        print(f"PhilGEPS: Saved {len(ysub):,} records (year {int(year)}) -> {ypath}", flush=True)

    medical_df.to_csv(output_combined, index=False)
    log_activity(f"Saved {len(medical_df):,} records (all years) -> {output_combined}")
    print(f"PhilGEPS: Saved {len(medical_df):,} records (all years) -> {output_combined}", flush=True)

    v_files = run_output_file_validations(output_combined, yearly_write_plan)
    apply_validation_result(v_files, log_activity)
    log_activity("Output CSV validations passed")


def main() -> None:
    _ensure_tree()
    vis_paths = build_step01_layout()
    output_combined = os.path.join(PATH_OUTPUT_01, "philgeps_medical_procurement.csv")
    term_log = os.path.join(PATH_TERMINAL_LOGS, "01_data_cleaning_philgeps_terminal.txt")
    try:
        with tee_stdio_to_file(term_log):
            _run_cleaning_philgeps(output_combined, vis_paths)
            print(
                f"Done. Terminal log: {PATH_TERMINAL_LOGS}, activity: {PATH_LOG_ENTRIES}, "
                f"EDA: {PATH_EDA_DIR} (per-year + merged/), schema: {PATH_DATA_SCHEMA_01}, "
                f"summaries: {PATH_SUMMARIES_DIR}, CSV: {PATH_OUTPUT_01}",
                flush=True,
            )
    except Step01ValidationError as e:
        print(f"Step 01 validation failed: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
