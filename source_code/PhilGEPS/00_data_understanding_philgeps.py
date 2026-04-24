"""
Step 00 — PhilGEPS data understanding (pre-cleaning).

Scans raw CSVs under MedFlowPH/raw_datasets/PhilGEPS/{2020..2025}/, profiles columns,
types, nulls, duplicates, k-means feature suitability, and medicine-filter candidates.
Processes one quarter at a time (per-quarter concat only; quarter inferred from filenames).
No all-years merge; full merge is left to step 01. Writes EDA PNGs under
results/00/Exploratory Data Analysis/{year}/{Qn}/ (three overview charts: nulls, dtypes, row counts).

2022 raw exports are headerless: read with fixed PHILGEPS_46_COLS.

Outputs:
- EDA / schema -> MedFlowPH/results/00/{Exploratory Data Analysis|Raw Dataset Schema}/{year}/{Qn}/
- One consolidated summary TXT -> MedFlowPH/results/00/Summaries/philgeps_understanding_summary.txt (overview tables + compact per-segment detail)
- Raw schema table (CSV + TXT) -> MedFlowPH/results/00/Raw Dataset Schema/
- Quarter manifest -> MedFlowPH/logs/00/step00_quarter_manifest.txt
- Terminal mirror -> MedFlowPH/logs/00/Terminal Logs/
- Activity log -> MedFlowPH/logs/00/Log entries/
"""

from __future__ import annotations

import contextlib
import gc
import os
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.abspath(__file__)
MEDFLOW_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

PATH_RAW_DATASETS = os.path.join(MEDFLOW_ROOT, "raw_datasets")
PATH_PHILGEPS_RAW = os.path.join(PATH_RAW_DATASETS, "PhilGEPS")

PATH_RESULTS_00 = os.path.join(MEDFLOW_ROOT, "results", "00")
PATH_EDA_00 = os.path.join(PATH_RESULTS_00, "Exploratory Data Analysis")
PATH_SUMMARIES_00 = os.path.join(PATH_RESULTS_00, "Summaries")
PATH_SUMMARY_CONSOLIDATED_00 = os.path.join(PATH_SUMMARIES_00, "philgeps_understanding_summary.txt")
PATH_RAW_SCHEMA_00 = os.path.join(PATH_RESULTS_00, "Raw Dataset Schema")

PATH_LOGS_00 = os.path.join(MEDFLOW_ROOT, "logs", "00")
PATH_TERMINAL_LOGS_00 = os.path.join(PATH_LOGS_00, "Terminal Logs")
PATH_LOG_ENTRIES_00 = os.path.join(PATH_LOGS_00, "Log entries")

CSV_EXTENSIONS = {".csv"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}

ENCODING_TRIES = ("utf-8", "latin-1", "cp1252")

# Canonical PhilGEPS wide export (46 columns) — 2022 CSVs have no header row.
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
    "medical",
    "medicine",
    "pharmaceutical",
    "drug",
    "vaccine",
    "hospital",
    "laboratory",
    "diagnostic",
    "surgical",
    "clinic",
    "health",
    "therapeutic",
    "antibiotic",
    "syringe",
    "test kit",
    "reagent",
    "biomedical",
]
# Same keyword union as step 01 (substring match via regex alternation)
PATTERN = "|".join(MEDICAL_KEYWORDS)

YEAR_FOLDERS = tuple(str(y) for y in range(2020, 2026))

# Compact summary: top columns by null % per segment
SUMMARY_TOP_NULL_COLUMNS = 10

# For per-year null breakdown in summary
CRITICAL_NULL_FIELDS = [
    "UNSPSC Description",
    "Item Name",
    "Item Description",
    "Contract Amount",
    "Item Budget",
    "Award Reference No.",
    "Bid Reference No.",
    "UNSPSC Code",
]

KMEANS_NUMERIC_READY = {
    "Contract Amount",
    "Item Budget",
    "Quantity",
    "Approved Budget of the Contract",
    "Line Item No",
}
KMEANS_DATE_DERIVABLE = {
    "Published Date",
    "Closing Date",
    "Award Date",
    "Published Date(Award)",
    "Contract Effectivity Date",
    "Contract End Date",
    "Notice to Proceed Date",
}
KMEANS_LOW_CARD_ENCODE = {
    "Region",
    "Procurement Mode",
    "PE Organization Type",
    "Awardee Size",
    "Funding Source",
    "Classification",
    "Calendar Type",
    "Government Branch",
    "PE Organization Type (Grouped)",
    "Business Category",
    "Bid Notice Status",
    "Award Notice Status",
    "Country of Awardee",
    "Region of Awardee",
    "Awardee Joint Venture",
}
KMEANS_HIGH_CARD_TEXT = {
    "Procuring Entity (PE)",
    "Awardee Organization Name",
    "Item Name",
    "Item Description",
    "UNSPSC Description",
    "Notice Title",
    "Award Title",
    "Province",
    "City/Municipality",
    "Area of Delivery",
    "Funding Instrument",
    "Trade Agreement",
    "Province of Awardee",
    "City/Municipality of Awardee",
    "UOM",
}
KMEANS_IDENTIFIER_SKIP = {
    "Bid Reference No.",
    "Award Reference No.",
    "UNSPSC Code",
}

MEDICINE_FILTER_CANDIDATES = [
    "UNSPSC Description",
    "Item Name",
    "Item Description",
    "UNSPSC Code",
    "Classification",
    "Business Category",
    "Notice Title",
    "Award Title",
]


def _safe_filename_fragment(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return (s[:max_len] if s else "column") or "column"


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
        PATH_PHILGEPS_RAW,
        PATH_RESULTS_00,
        PATH_EDA_00,
        PATH_SUMMARIES_00,
        PATH_RAW_SCHEMA_00,
        PATH_LOGS_00,
        PATH_TERMINAL_LOGS_00,
        PATH_LOG_ENTRIES_00,
    ):
        os.makedirs(p, exist_ok=True)
    # EDA subfolders are created per quarter segment on demand (generate_eda_for_segment).


# Old step-00 builds used a single folder name "2024_Q1" instead of "2024/Q1".
_LEGACY_FLAT_QUARTER_DIR = re.compile(r"^\d{4}_Q[1-4]$")


def _remove_legacy_flat_quarter_dirs() -> None:
    """Delete obsolete top-level YYYY_Qn folders under results/00 (EDA, schema, summaries)."""
    for root in (PATH_EDA_00, PATH_RAW_SCHEMA_00, PATH_SUMMARIES_00):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if _LEGACY_FLAT_QUARTER_DIR.fullmatch(name):
                legacy = os.path.join(root, name)
                if os.path.isdir(legacy):
                    shutil.rmtree(legacy, ignore_errors=True)


def _remove_summaries_per_year_trees() -> None:
    """Remove Summaries/2020 .. Summaries/2025 (old per-quarter summary layout); keep only root TXT."""
    if not os.path.isdir(PATH_SUMMARIES_00):
        return
    for name in os.listdir(PATH_SUMMARIES_00):
        if name not in YEAR_FOLDERS:
            continue
        p = os.path.join(PATH_SUMMARIES_00, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def _clear_eda_png_stale(out_dir: str) -> None:
    """Remove every PNG in the quarter EDA folder so reruns do not keep old per-column charts."""
    if not os.path.isdir(out_dir):
        return
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".png"):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(out_dir, fn))


def infer_quarter_from_basename(path: str) -> int | None:
    """
    Map PhilGEPS export filenames to calendar quarter 1–4.
    Returns None if the period cannot be inferred (caller uses a file-specific segment folder).
    """
    b = os.path.basename(path).lower().replace("(", " ").replace(")", " ")
    if re.search(r"jan(?:uary)?[\s.\-_]*mar(?:ch)?", b) or re.search(r"\bjan[\s.\-_]*mar\b", b):
        return 1
    if re.search(r"apr\w*[\s.\-_]*jun\w*", b) or ("april" in b and "june" in b):
        return 2
    if re.search(r"jul\w*[\s.\-_]*sep\w*", b) or (
        "july" in b and ("sep" in b or "sept" in b or "september" in b)
    ):
        return 3
    if re.search(r"oct\w*[\s.\-_]*dec\w*", b) or ("october" in b and "december" in b):
        return 4
    return None


def segment_key_for_file(path: str, year_from_folder: int) -> str:
    """Stable folder name for this raw file's quarter bucket, e.g. 2023_Q2 or 2020__April-June_2020."""
    q = infer_quarter_from_basename(path)
    if q is not None:
        return f"{year_from_folder}_Q{q}"
    slug = _safe_filename_fragment(os.path.splitext(os.path.basename(path))[0], max_len=48)
    return f"{year_from_folder}__{slug}"


def segment_sort_key(segment_key: str) -> tuple[Any, ...]:
    """Sort 2024_Q1 before 2024_Q2; unknown segments sort after by name."""
    m = re.match(r"^(\d{4})_Q([1-4])$", segment_key)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (int(segment_key[:4]) if segment_key[:4].isdigit() else 9999, 99, segment_key)


def year_and_quarter_subdir(segment_key: str) -> tuple[int, str]:
    """
    Output path nesting: results/00/.../{calendar_year}/{quarter_folder}/.
    quarter_folder is Q1..Q4 or a sanitized slug for unknown-inferred periods.
    """
    m = re.match(r"^(\d{4})_Q([1-4])$", segment_key)
    if m:
        return int(m.group(1)), f"Q{m.group(2)}"
    m2 = re.match(r"^(\d{4})__(.+)$", segment_key)
    if m2:
        y = int(m2.group(1))
        sub = _safe_filename_fragment(m2.group(2), max_len=80) or "unknown"
        return y, sub
    if len(segment_key) >= 4 and segment_key[:4].isdigit():
        return int(segment_key[:4]), _safe_filename_fragment(segment_key, max_len=80)
    return 9999, "unknown"


def get_supported_files_by_year() -> list[tuple[str, int]]:
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


def _read_csv_one_encoding(
    path: str,
    encoding: str,
    header: int | None,
    names: list[str] | None,
) -> pd.DataFrame:
    kw: dict[str, Any] = {
        "header": header,
        "names": names,
        "low_memory": False,
        "encoding": encoding,
        "on_bad_lines": "skip",
    }
    if encoding == "utf-8":
        kw["encoding_errors"] = "replace"
    try:
        return pd.read_csv(path, **kw)
    except Exception:  # noqa: BLE001
        kw["engine"] = "python"
        return pd.read_csv(path, **kw)


def load_file_resilient(path: str, year: int) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext not in CSV_EXTENSIONS:
        raise ValueError(f"Unsupported extension for resilient load: {path}")

    last_err: Exception | None = None
    if year == 2022:
        header: int | None = None
        names: list[str] | None = PHILGEPS_46_COLS
    else:
        header = 0
        names = None

    for enc in ENCODING_TRIES:
        try:
            df = _read_csv_one_encoding(path, enc, header, names)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV {path}: {last_err}")


def _load_excel_all_sheets(path: str) -> list[tuple[str, pd.DataFrame]]:
    book = pd.read_excel(path, sheet_name=None, dtype=object)
    return [(str(name), df.copy()) for name, df in book.items()]


def _first_cell_looks_like_header(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 1:
        return False
    v = str(df.iloc[0, 0]).strip()
    return v.lower().startswith("procuring") or "procuring entity" in v.lower()


def load_philgeps_frames(path: str, year: int) -> list[tuple[pd.DataFrame, str]]:
    """
    Load one raw file into one or more frames (Excel = one per sheet).
    Returns (dataframe, inventory_label).
    """
    ext = os.path.splitext(path)[1].lower()
    base = os.path.basename(path)
    if ext in CSV_EXTENSIONS:
        return [(load_file_resilient(path, year), base)]
    if ext in EXCEL_EXTENSIONS:
        out: list[tuple[pd.DataFrame, str]] = []
        for sheet_name, s in _load_excel_all_sheets(path):
            part_label = f"{base}::{sheet_name}"
            s2 = s
            if year == 2022:
                if s2.shape[1] == len(PHILGEPS_46_COLS) and not _first_cell_looks_like_header(s2):
                    s2 = s2.copy()
                    s2.columns = PHILGEPS_46_COLS
                elif s2.shape[1] == len(PHILGEPS_46_COLS) and all(
                    str(c) in PHILGEPS_46_COLS for c in s2.columns
                ):
                    s2 = s2.reindex(columns=PHILGEPS_46_COLS).copy()
            out.append((s2, part_label))
        return out
    return []


PROFILE_CHUNK_ROWS = 250_000


def _object_empty_whitespace_count(s: pd.Series, chunk_size: int = PROFILE_CHUNK_ROWS) -> int:
    """Count non-null cells that are '' or whitespace-only without materializing full-column str arrays."""
    n = len(s)
    empty = 0
    for start in range(0, n, chunk_size):
        part = s.iloc[start : start + chunk_size]
        nn = part.notna()
        if not nn.any():
            continue
        sub = part.loc[nn]
        empty += int((sub.astype(str).str.strip() == "").sum())
    return empty


def check_column_drift(df: pd.DataFrame, _filepath: str) -> str:
    cols = list(df.columns)
    canon = PHILGEPS_46_COLS
    if cols == canon:
        return "OK"
    s_df = set(cols)
    s_c = set(canon)
    if s_df == s_c:
        return "REORDERED"
    if s_df < s_c:
        return "MISSING"
    if s_df > s_c:
        return "EXTRA"
    return "MISMATCH"


def profile_dtypes_and_nulls(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    n = len(df)
    out: dict[str, dict[str, Any]] = {}
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        # Treat empty string as missing for object / string columns (chunked for large series)
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            so = s.astype(object) if pd.api.types.is_string_dtype(s.dtype) else s
            nulls += _object_empty_whitespace_count(so)
        out[c] = {
            "dtype": str(s.dtype),
            "null_count": nulls,
            "null_pct": (nulls / n * 100.0) if n else 0.0,
        }
    return out


def _nulls_critical_for_segment(
    ydf: pd.DataFrame,
    segment_key: str,
    fields: list[str],
    out: dict[str, dict[str, int]],
) -> None:
    """Update out[field][segment_key] with null+empty counts for one quarter's frame."""
    for f in fields:
        if f not in ydf.columns:
            out[f][segment_key] = -1
            continue
        s = ydf[f]
        na = int(s.isna().sum())
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            so = s.astype(object) if pd.api.types.is_string_dtype(s.dtype) else s
            na += _object_empty_whitespace_count(so)
        out[f][segment_key] = na


def count_duplicates_one_segment(ydf: pd.DataFrame) -> dict[str, Any]:
    """Exact duplicate rows and key-column duplication within one quarter's data only."""
    n = len(ydf)
    nd = ydf.drop_duplicates().shape[0]
    key_stats: dict[str, Any] = {}
    for key_col in ("Award Reference No.", "Bid Reference No."):
        if key_col not in ydf.columns:
            key_stats[key_col] = {"missing_column": True}
            continue
        s = ydf[key_col]
        non_null = s.notna() & (s.astype(str).str.strip() != "")
        dup_mask = s.duplicated(keep=False) & non_null
        key_stats[key_col] = {
            "rows_with_duplicate_key": int(dup_mask.sum()),
            "unique_non_null_keys": int(s[non_null].nunique()),
        }
    return {
        "rows": n,
        "exact_duplicate_rows": n - nd,
        "key_columns": key_stats,
    }


def kmeans_feature_report(df: pd.DataFrame) -> dict[str, str]:
    report: dict[str, str] = {}
    for c in df.columns:
        if c in KMEANS_NUMERIC_READY:
            report[c] = "numeric_ready"
        elif c in KMEANS_DATE_DERIVABLE:
            report[c] = "date_derivable"
        elif c in KMEANS_LOW_CARD_ENCODE:
            report[c] = "low_cardinality_encodable"
        elif c in KMEANS_HIGH_CARD_TEXT:
            report[c] = "high_cardinality_text"
        elif c in KMEANS_IDENTIFIER_SKIP:
            report[c] = "identifier_skip"
        else:
            report[c] = "other_review"
    return report


def _series_keyword_match(s: pd.Series, pattern: str) -> pd.Series:
    # Object columns may hold numbers/complex; always stringify for regex match.
    return s.astype(str).str.contains(pattern, case=False, na=False, regex=True)


def evaluate_medicine_filter(df: pd.DataFrame) -> dict[str, Any]:
    baseline_col = "UNSPSC Description"
    if baseline_col not in df.columns:
        baseline = pd.Series(False, index=df.index)
    else:
        baseline = _series_keyword_match(df[baseline_col], PATTERN)

    n = len(df)
    baseline_count = int(baseline.sum())

    rows: list[dict[str, Any]] = []
    for col in MEDICINE_FILTER_CANDIDATES:
        if col not in df.columns:
            rows.append(
                {
                    "column": col,
                    "coverage_pct": 0.0,
                    "keyword_match_count": 0,
                    "overlap_with_baseline": 0,
                    "match_outside_baseline": 0,
                    "precision_risk_note": "column missing",
                },
            )
            continue
        s = df[col]
        non_null = s.notna() & (s.astype(str).str.strip() != "")
        coverage_pct = float(non_null.sum() / n * 100.0) if n else 0.0
        kw = _series_keyword_match(s, PATTERN)
        kw_count = int(kw.sum())
        overlap = int((kw & baseline).sum())
        outside = int((kw & ~baseline).sum())
        if col == baseline_col:
            risk = "baseline; broad keywords may include non-medicine uses of 'health' etc."
        elif col in ("Notice Title", "Award Title", "Business Category"):
            risk = "high — procurement titles often mention health facilities without medical goods"
        elif col == "Classification":
            risk = "medium — category labels can be broad"
        elif col == "UNSPSC Code":
            risk = "low–medium if validated against UNSPSC segments; codes alone need segment rules"
        else:
            risk = "medium — free text; review sample of match_outside_baseline rows"
        rows.append(
            {
                "column": col,
                "coverage_pct": round(coverage_pct, 2),
                "keyword_match_count": kw_count,
                "overlap_with_baseline": overlap,
                "match_outside_baseline": outside,
                "precision_risk_note": risk,
            },
        )

    # Recommendation
    item_outside = next(
        (r["match_outside_baseline"] for r in rows if r["column"] == "Item Name"),
        0,
    )
    desc_outside = next(
        (r["match_outside_baseline"] for r in rows if r["column"] == "Item Description"),
        0,
    )
    if item_outside + desc_outside > max(500, int(0.01 * baseline_count)):
        recommendation = (
            "(c) Conservative multi-field rule: keep UNSPSC Description OR "
            "(Item Name / Item Description keyword match only when UNSPSC Description is null/empty "
            "or does not match), to recover rows with generic UNSPSC text. "
            "Validate with manual spot-checks on match_outside_baseline."
        )
    elif baseline_count > 0:
        recommendation = (
            "(a) UNSPSC Description keyword filter remains primary; additional fields add limited "
            "recall vs precision trade-off for this dataset snapshot."
        )
    else:
        recommendation = (
            "(b) Baseline produced zero matches — expand to Item Name + Item Description with same "
            "keywords until UNSPSC coverage is confirmed in raw files."
        )

    return {
        "baseline_unspsc_match_count": baseline_count,
        "per_column": rows,
        "recommendation": recommendation,
    }


def write_raw_schema_table(
    profile_merged: dict[str, dict[str, Any]],
    kmeans_report: dict[str, str],
    out_dir: str,
    total_row_count: int,
    segment_key: str | None = None,
) -> tuple[str, str]:
    """
    Write a tabular schema for one loaded quarter: column order, dtypes, null rates, k-means role.
    """
    os.makedirs(out_dir, exist_ok=True)
    canon_set = set(PHILGEPS_46_COLS)
    canon_pos = {name: i + 1 for i, name in enumerate(PHILGEPS_46_COLS)}
    present = set(profile_merged.keys())
    ordered: list[str] = [c for c in PHILGEPS_46_COLS if c in present]
    ordered.extend(sorted(c for c in present if c not in canon_set))

    rows: list[dict[str, Any]] = []
    for col in ordered:
        p = profile_merged[col]
        null_ct = int(p["null_count"])
        null_pct = float(p["null_pct"])
        rows.append(
            {
                "canonical_position": canon_pos.get(col, ""),
                "column_name": col,
                "in_canonical_46": col in canon_set,
                "pandas_dtype": p.get("dtype", ""),
                "row_count": total_row_count,
                "null_count": null_ct,
                "null_pct": round(null_pct, 4),
                "non_null_count": max(0, total_row_count - null_ct),
                "nunique_non_null": "",
                "nunique_how": "N/A (see EDA charts in this quarter folder)",
                "kmeans_feature_role": kmeans_report.get(col, ""),
            },
        )

    schema_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "philgeps_raw_schema_table.csv")
    txt_path = os.path.join(out_dir, "philgeps_raw_schema_table.txt")
    schema_df.to_csv(csv_path, index=False)
    scope = (
        f"PhilGEPS raw schema — quarter {segment_key} (pre-cleaning, one segment in memory)"
        if segment_key
        else "PhilGEPS raw schema (pre-cleaning)"
    )
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{scope}\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Row count (this quarter): {total_row_count:,}\n\n")
        f.write(schema_df.to_string(index=False))
        f.write("\n")
    return csv_path, txt_path


def generate_eda_for_segment(
    df: pd.DataFrame,
    segment_key: str,
    out_dir: str,
    file_row_counts: dict[str, int] | None = None,
    dataset_label: str | None = None,
) -> None:
    """
    Overview PNGs only: null counts, dtype counts, row counts per source file (if provided).
    """
    os.makedirs(out_dir, exist_ok=True)
    _clear_eda_png_stale(out_dir)
    if df.empty:
        return
    label = dataset_label or f"PhilGEPS raw {segment_key}"

    try:
        nulls = df.isnull().sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(10, len(nulls) * 0.25), 6))
        nulls.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{label} — missing values per column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Null count")
        plt.xticks(rotation=90, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "00_overview_null_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA {segment_key}: null counts failed: {e}", flush=True)

    try:
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        dtype_counts.plot(kind="barh", ax=ax, color="coral")
        ax.set_title(f"{label} — column count by dtype")
        ax.set_xlabel("Number of columns")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "01_overview_dtype_counts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"EDA {segment_key}: dtype counts failed: {e}", flush=True)

    if file_row_counts:
        try:
            names = list(file_row_counts.keys())
            vals = [file_row_counts[k] for k in names]
            fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
            ax.barh(names, vals, color="darkslateblue")
            ax.set_title(f"{label} — row counts per source file")
            ax.set_xlabel("Rows")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "02_row_counts_per_file.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"EDA {segment_key}: row counts per file failed: {e}", flush=True)


def _path_for_display(abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, MEDFLOW_ROOT).replace("\\", "/")
    except ValueError:
        return abs_path


def _profile_top_nulls(profile: dict[str, dict[str, Any]], k: int) -> list[tuple[str, float, str]]:
    ranked = sorted(profile.items(), key=lambda x: x[1]["null_pct"], reverse=True)
    out: list[tuple[str, float, str]] = []
    for col, p in ranked[:k]:
        out.append((col, float(p["null_pct"]), str(p["dtype"])))
    return out


def _best_overlap_column_label(filter_eval: dict[str, Any]) -> str:
    best_col, best_v = "", -1
    for r in filter_eval["per_column"]:
        if r["column"] == "UNSPSC Description":
            continue
        ov = int(r["overlap_with_baseline"])
        if ov > best_v:
            best_v = ov
            best_col = str(r["column"])
    return f"{best_col} (overlap {best_v:,})" if best_col else "—"


def write_consolidated_understanding_summary(
    out_path: str,
    *,
    run_started: str,
    run_finished: str,
    kmeans_report: dict[str, str],
    segment_snapshots: list[dict[str, Any]],
) -> str:
    """
    Single readable report: overview tables, cross-segment matrices, k-means once,
    then compact per-segment sections. Full column-level profiling stays in schema CSV/TXT per quarter.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines: list[str] = [
        "PhilGEPS — step 00 data understanding (pre-cleaning)",
        f"Run started: {run_started}",
        f"Run finished: {run_finished}",
        "File paths below use paths relative to the MedFlowPH project root when possible.",
        "",
    ]

    if not segment_snapshots:
        lines.extend(
            [
                "No segments produced data (no files loaded).",
                "",
            ],
        )
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        return out_path

    schema_rel = _path_for_display(PATH_RAW_SCHEMA_00)

    lines.extend(
        [
            "=" * 72,
            "OVERVIEW — ALL QUARTERS",
            "=" * 72,
            "",
        ],
    )
    skw = max(len(s["segment_key"]) for s in segment_snapshots)
    skw = max(skw, 12)
    for snap in segment_snapshots:
        invs = snap["inventory"]
        drifts = sorted({str(x["drift"]) for x in invs})
        drift_s = ",".join(drifts) if len(drifts) <= 3 else f"{len(drifts)} mixed"
        col_counts = sorted({x["cols"] for x in invs})
        cols_note = (
            str(col_counts[0])
            if len(col_counts) == 1
            else f"{min(col_counts)}–{max(col_counts)}"
        )
        lines.append(
            f"  {snap['segment_key']:<{skw}}  {snap['folder']:<10}  {snap['rows']:>12,}  "
            f"{len(invs):>5} file(s)  drift={drift_s:<14}  cols={cols_note}",
        )

    total_rows = sum(int(s["rows"]) for s in segment_snapshots)
    sum_exact_dup = sum(int(s["dup"]["exact_duplicate_rows"]) for s in segment_snapshots)
    lines.extend(
        [
            "",
            f"  Total rows (sum of segments): {total_rows:,}",
            f"  Segments processed: {len(segment_snapshots)}",
            f"  Sum of within-segment exact duplicate rows: {sum_exact_dup:,}",
            f"  Per-column schema/null detail: {schema_rel}/{{year}}/{{Qn}}/",
            "",
        ],
    )

    # Critical nulls matrix
    lines.extend(
        [
            "=" * 72,
            "CRITICAL FIELDS — NULL+EMPTY COUNTS BY SEGMENT",
            "=" * 72,
            "",
        ],
    )
    segs = [s["segment_key"] for s in segment_snapshots]
    fcw = max(len(f) for f in CRITICAL_NULL_FIELDS)
    scw = max(max(len(s) for s in segs), 10)
    head = " " * (fcw + 2) + "".join(s.ljust(scw + 1) for s in segs)
    lines.append(head)
    for field in CRITICAL_NULL_FIELDS:
        row = f"  {field.ljust(fcw)}"
        for snap in segment_snapshots:
            v = snap["critical_nulls"].get(field, -1)
            cell = "absent" if v < 0 else f"{v:,}"
            row += cell.rjust(scw + 1)
        lines.append(row)
    lines.append("")

    # Duplicates
    scope_dup = (
        "No full merge in step 00. Duplicate counts are within each quarter segment only; "
        "cross-segment / cross-year duplicates are not evaluated here (see step 01)."
    )
    lines.extend(
        [
            "=" * 72,
            "DUPLICATES — WITHIN SEGMENT ONLY",
            "=" * 72,
            "",
            f"  {scope_dup}",
            "",
            f"  {'Segment':<12}  {'Rows':>12}  {'Exact dup rows':>14}  Key columns (dup keys / unique keys)",
        ],
    )
    for snap in segment_snapshots:
        d = snap["dup"]
        kc = d.get("key_columns", {})
        bits: list[str] = []
        for key_col in ("Award Reference No.", "Bid Reference No."):
            kv = kc.get(key_col, {})
            if kv.get("missing_column"):
                bits.append(f"{key_col}: missing")
            else:
                bits.append(
                    f"{key_col}: {kv['rows_with_duplicate_key']:,} / "
                    f"{kv['unique_non_null_keys']:,}",
                )
        lines.append(
            f"  {snap['segment_key']:<12}  {d['rows']:>12,}  {d['exact_duplicate_rows']:>14,}  "
            f"{'; '.join(bits)}",
        )
    lines.append("")

    # Medicine filter snapshot
    lines.extend(
        [
            "=" * 72,
            "MEDICINE FILTER — SNAPSHOT BY SEGMENT",
            "=" * 72,
            "",
            f"  {'Segment':<12}  {'Baseline UNSPSC':>16}  {'Top overlap (non-baseline)':<38}  Recommendation (abridged)",
        ],
    )
    for snap in segment_snapshots:
        fe = snap["filter_eval"]
        rec = fe["recommendation"]
        short = (rec[:56] + "…") if len(rec) > 57 else rec
        lines.append(
            f"  {snap['segment_key']:<12}  {fe['baseline_unspsc_match_count']:>16,}  "
            f"{_best_overlap_column_label(fe):<38}  {short}",
        )
    lines.append("")

    # K-means once
    lines.extend(
        [
            "=" * 72,
            "K-MEANS FEATURE ROLES (canonical 46 columns — static taxonomy)",
            "=" * 72,
            "",
        ],
    )
    by_cat: dict[str, list[str]] = defaultdict(list)
    for col, cat in kmeans_report.items():
        by_cat[cat].append(col)
    for cat in sorted(by_cat.keys()):
        lines.append(f"  [{cat}]")
        for c in sorted(by_cat[cat]):
            lines.append(f"    - {c}")
        lines.append("")

    # Per-segment detail
    lines.extend(
        [
            "=" * 72,
            "SEGMENT DETAIL (compact)",
            "=" * 72,
            "",
        ],
    )
    for snap in segment_snapshots:
        fe = snap["filter_eval"]
        lines.extend(
            [
                "-" * 72,
                f"{snap['segment_key']}  ({snap['folder']})  —  {snap['rows']:,} rows",
                "-" * 72,
                "  Files:",
            ],
        )
        for inv in snap["inventory"]:
            label = inv.get("inventory_label", os.path.basename(inv["path"]))
            disp = _path_for_display(inv["path"])
            lines.append(
                f"    - {disp}  :: {label}  | rows={inv['rows']:,}  cols={inv['cols']}  "
                f"drift={inv['drift']}",
            )
        lines.append("  Top null % (columns, this segment):")
        for col, pct, dt in _profile_top_nulls(snap["profile"], SUMMARY_TOP_NULL_COLUMNS):
            lines.append(f"    - {col}: {pct:.2f}% null  ({dt})")
        d = snap["dup"]
        lines.extend(
            [
                "  Duplicates:",
                f"    exact_duplicate_rows={d['exact_duplicate_rows']:,}  (within this segment)",
            ],
        )
        kc = d.get("key_columns", {})
        for key_col in ("Award Reference No.", "Bid Reference No."):
            kv = kc.get(key_col, {})
            if kv.get("missing_column"):
                lines.append(f"    {key_col}: (column missing)")
            else:
                lines.append(
                    f"    {key_col}: rows_with_duplicate_key={kv['rows_with_duplicate_key']:,}  "
                    f"unique_non_null_keys={kv['unique_non_null_keys']:,}",
                )
        lines.extend(
            [
                f"  Medicine filter: baseline UNSPSC keyword matches = {fe['baseline_unspsc_match_count']:,}",
                "  Per candidate column:",
            ],
        )
        for r in fe["per_column"]:
            lines.append(
                f"    {r['column']}: kw={r['keyword_match_count']:,}  "
                f"overlap={r['overlap_with_baseline']:,}  outside={r['match_outside_baseline']:,}  "
                f"risk: {r['precision_risk_note']}",
            )
        lines.append("  Recommendation:")
        for para in textwrap.wrap(
            fe["recommendation"],
            width=68,
            initial_indent="    ",
            subsequent_indent="    ",
        ):
            lines.append(para)
        lines.append("")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return out_path


def main() -> None:
    _ensure_tree()
    _remove_legacy_flat_quarter_dirs()
    _remove_summaries_per_year_trees()
    activity_path = os.path.join(PATH_LOG_ENTRIES_00, "00_data_understanding_philgeps_activity.txt")
    log_activity = open_activity_log(activity_path)
    term_log = os.path.join(PATH_TERMINAL_LOGS_00, "00_data_understanding_philgeps_terminal.txt")
    manifest_path = os.path.join(PATH_LOGS_00, "step00_quarter_manifest.txt")

    with tee_stdio_to_file(term_log):
        files = get_supported_files_by_year()
        if not files:
            msg = f"No CSV/Excel files under {PATH_PHILGEPS_RAW}/{{2020..2025}}."
            log_activity(msg)
            print(msg, flush=True)
            return

        segment_files: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for filepath, year in files:
            seg = segment_key_for_file(filepath, year)
            segment_files[seg].append((filepath, year))

        if not segment_files:
            log_activity("No file paths grouped.")
            print("No file paths grouped.", flush=True)
            return

        run_started = datetime.now().isoformat(timespec="seconds")
        segment_snapshots: list[dict[str, Any]] = []

        with open(manifest_path, "w", encoding="utf-8", newline="\n") as mf:
            mf.write("Step 00 — one quarter in RAM at a time (EDA + schema per folder)\n")
            mf.write(f"Run started: {run_started}\n")
            mf.write(f"Consolidated summary (all quarters): {PATH_SUMMARY_CONSOLIDATED_00}\n\n")

        km = kmeans_feature_report(pd.DataFrame(columns=PHILGEPS_46_COLS))

        for seg in sorted(segment_files.keys(), key=segment_sort_key):
            yr, qdir = year_and_quarter_subdir(seg)
            eda_dir = os.path.join(PATH_EDA_00, str(yr), qdir)
            schema_dir = os.path.join(PATH_RAW_SCHEMA_00, str(yr), qdir)

            parts: list[pd.DataFrame] = []
            inventory_q: list[dict[str, Any]] = []
            seg_file_rows: dict[str, int] = {}

            for filepath, year in segment_files[seg]:
                try:
                    frames = load_philgeps_frames(filepath, year)
                    if not frames:
                        log_activity(f"No loader for file: {filepath}")
                        continue
                    for df, inv_label in frames:
                        drift = check_column_drift(df, filepath)
                        inventory_q.append(
                            {
                                "path": filepath,
                                "inventory_label": inv_label,
                                "year": year,
                                "segment_key": seg,
                                "inferred_quarter": infer_quarter_from_basename(filepath),
                                "rows": len(df),
                                "cols": df.shape[1],
                                "drift": drift,
                            },
                        )
                        seg_file_rows[inv_label] = len(df)
                        parts.append(df.reindex(columns=PHILGEPS_46_COLS).copy())
                        log_activity(
                            f"Loaded {inv_label} year={year} segment={seg} rows={len(df)} drift={drift}",
                        )
                        print(
                            f"Loaded {inv_label} ({year} / {yr}/{qdir}): {len(df):,} rows, drift={drift}",
                            flush=True,
                        )
                except Exception as e:  # noqa: BLE001
                    log_activity(f"Error loading {filepath}: {e}")
                    print(f"Error loading {filepath}: {e}", flush=True)

            if not parts:
                log_activity(f"Segment {seg}: no frames loaded; skip outputs.")
                continue

            try:
                try:
                    ydf = pd.concat(parts, ignore_index=True, copy=False)
                except TypeError:
                    ydf = pd.concat(parts, ignore_index=True)
            finally:
                del parts
            gc.collect()

            ny = len(ydf)
            log_activity(
                f"Segment {seg} ({yr}/{qdir}): {ny:,} rows in memory — EDA, schema, consolidated summary then release",
            )

            generate_eda_for_segment(
                ydf,
                seg,
                eda_dir,
                file_row_counts=seg_file_rows,
                dataset_label=f"PhilGEPS raw {yr}/{qdir}",
            )
            log_activity(f"Wrote EDA -> {eda_dir}")

            profile = profile_dtypes_and_nulls(ydf)
            csv_schema, txt_schema = write_raw_schema_table(
                profile,
                km,
                schema_dir,
                ny,
                segment_key=seg,
            )
            log_activity(f"Wrote schema CSV: {csv_schema}")

            nulls_one: dict[str, dict[str, int]] = {f: {} for f in CRITICAL_NULL_FIELDS}
            _nulls_critical_for_segment(ydf, seg, CRITICAL_NULL_FIELDS, nulls_one)
            dup_inner = count_duplicates_one_segment(ydf)
            filt = evaluate_medicine_filter(ydf)
            critical_for_snap = {f: nulls_one[f].get(seg, -1) for f in CRITICAL_NULL_FIELDS}
            segment_snapshots.append(
                {
                    "segment_key": seg,
                    "folder": f"{yr}/{qdir}",
                    "rows": ny,
                    "inventory": inventory_q,
                    "critical_nulls": critical_for_snap,
                    "dup": dup_inner,
                    "filter_eval": filt,
                    "profile": profile,
                },
            )

            with open(manifest_path, "a", encoding="utf-8", newline="\n") as mf:
                mf.write(f"{yr}/{qdir}  (segment_key={seg})\n")
                mf.write(f"  EDA: {eda_dir}\n")
                mf.write(f"  Schema: {schema_dir}\n\n")

            del ydf
            gc.collect()

        run_finished = datetime.now().isoformat(timespec="seconds")
        write_consolidated_understanding_summary(
            PATH_SUMMARY_CONSOLIDATED_00,
            run_started=run_started,
            run_finished=run_finished,
            kmeans_report=km,
            segment_snapshots=segment_snapshots,
        )
        log_activity(f"Wrote consolidated summary: {PATH_SUMMARY_CONSOLIDATED_00}")

        print(
            f"Done. Quarter manifest: {manifest_path}\n"
            f"Consolidated summary: {PATH_SUMMARY_CONSOLIDATED_00}\n"
            f"Results root: {PATH_RESULTS_00}\n"
            f"Logs: {PATH_LOGS_00}",
            flush=True,
        )


if __name__ == "__main__":
    main()
