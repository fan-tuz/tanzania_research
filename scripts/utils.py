"""Shared helper functions for the Tanzania NPS analysis codebase.

The project started as several independent scripts, so small helpers such as
``to_num`` and ``first_existing`` were repeated many times.  Keep cross-script,
side-effect-free utilities here; keep dataset-specific logic inside the scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re

import numpy as np
import pandas as pd

# Local projected CRS for Tanzania distance calculations.
# EPSG:32736 = WGS 84 / UTM zone 36S.  It is a better default than Web Mercator
# for km thresholds in Tanzania, while still being simple and widely supported.
# For projects concentrated in far eastern Tanzania, pass --projected-crs EPSG:32737.
DEFAULT_PROJECTED_CRS = "EPSG:32736"


def split_csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI value and drop empty/none tokens."""
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip() and x.strip().lower() != "none"]


def clean_id_series(values) -> pd.Series:
    """Normalize household IDs read from CSV/Excel without losing leading text.

    Numeric-looking IDs are often read as floats (e.g. ``123.0``).  This removes
    only the artificial trailing ``.0`` and standardizes common missing tokens.
    """
    s = pd.Series(values) if not isinstance(values, pd.Series) else values.copy()
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "NaN": np.nan, "<NA>": np.nan})
    )


def norm_text(value) -> str:
    """Lowercase, punctuation-insensitive text key for place-name matching."""
    if pd.isna(value):
        return ""
    s = str(value).strip().lower().replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_existing(columns_or_df, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate present in columns, case-insensitively.

    ``columns_or_df`` may be a DataFrame or any iterable of column names.
    """
    columns = columns_or_df.columns if hasattr(columns_or_df, "columns") else columns_or_df
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if not cand:
            continue
        if cand in cols:
            return cand
        found = lower.get(str(cand).lower())
        if found is not None:
            return found
    return None


def existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return requested columns that exist in ``df``, preserving order."""
    return [c for c in columns if c in df.columns]


def existing_nonconstant(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return existing columns with at least one non-missing value and variation."""
    out: list[str] = []
    for c in columns:
        if c in df.columns and df[c].notna().any() and df[c].nunique(dropna=True) > 1:
            out.append(c)
    return out


def to_num(values) -> pd.Series:
    """Coerce a Series-like object to numeric, using NaN for parse failures."""
    return pd.to_numeric(values, errors="coerce")


def to_float(values) -> pd.Series:
    """Coerce a Series-like object to float numeric values."""
    return pd.to_numeric(values, errors="coerce").astype("float64")


def to_binary(values) -> pd.Series:
    """Robust binary coercion: positive numeric values become 1, zero becomes 0.

    Missing and non-numeric values remain missing.  This avoids pandas nullable
    boolean ambiguity around ``pd.NA``.
    """
    x = to_float(values)
    return pd.Series(np.where(x.notna(), np.where(x > 0, 1.0, 0.0), np.nan), index=getattr(values, "index", None))


def yes1(values) -> pd.Series:
    """Binary convention for raw survey yes/no fields where exactly 1 means yes."""
    x = to_num(values)
    return pd.Series(np.where(x == 1, 1, 0), index=getattr(values, "index", None))


def safe_log(values) -> pd.Series:
    """Natural log of strictly positive numeric values; non-positive -> NaN."""
    x = to_num(values)
    return np.log(x.where(x > 0))


def safe_ratio(numerator, denominator) -> pd.Series:
    """Numeric ratio with zero denominators converted to missing."""
    n = to_num(numerator)
    d = to_num(denominator)
    return n / d.replace(0, np.nan)


def combine_first_columns(df: pd.DataFrame, new_col: str, candidates: Iterable[str]) -> pd.DataFrame:
    """Create ``new_col`` from the first non-missing value across candidates."""
    present = existing(df, candidates)
    if not present:
        return df
    out = df[present[0]].copy()
    for c in present[1:]:
        out = out.combine_first(df[c])
    df[new_col] = out
    return df


def find_file(folder: Path, candidates: Iterable[str], required: bool = False) -> Optional[Path]:
    """Find a file by exact or case-insensitive name within ``folder``."""
    folder = Path(folder)
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    lower_map = {p.name.lower(): p for p in folder.glob("*") if p.is_file()}
    for name in candidates:
        found = lower_map.get(str(name).lower())
        if found is not None:
            return found
    if required:
        raise FileNotFoundError(f"Could not find any of {list(candidates)} in {folder}")
    return None


def read_csv_optional(folder: Path, candidates: Iterable[str]) -> tuple[pd.DataFrame, Optional[Path]]:
    """Read the first matching CSV, returning an empty frame when absent."""
    path = find_file(folder, candidates, required=False)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path, low_memory=False), path


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected DataFrame columns to numeric in place and return ``df``."""
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_int_list(value: str | None) -> list[int]:
    """Parse a comma-separated list of numeric-looking values into integers."""
    out: list[int] = []
    for part in split_csv(value):
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return out


def add_post_flags(df: pd.DataFrame, year_col: str = "survey_year", wave_col: str = "wave") -> pd.DataFrame:
    """Add mechanical post_2009 and post_2013 indicators.

    Uses survey_year when available; falls back to wave numbers where possible.
    """
    out = df.copy()
    year = to_num(out[year_col]) if year_col in out.columns else pd.Series(np.nan, index=out.index)
    if year.isna().all() and wave_col in out.columns:
        wave = to_num(out[wave_col])
        year = wave.map({1: 2008, 2: 2010, 3: 2012, 4: 2014, 5: 2020})
    out["post_2009"] = np.where(year.notna(), (year >= 2010).astype(float), np.nan)
    out["post_2013"] = np.where(year.notna(), (year >= 2014).astype(float), np.nan)
    return out


def within_distance_threshold(distance_km, threshold_km: float) -> pd.Series:
    """Indicator for non-missing distances within a km threshold."""
    d = to_float(distance_km)
    return pd.Series(np.where(d.notna(), (d <= float(threshold_km)).astype(float), np.nan), index=getattr(distance_km, "index", None))


def make_interaction_column(df: pd.DataFrame, left: str, right: str, name: str | None = None) -> tuple[pd.DataFrame, str]:
    """Create a two-way interaction column and return ``(df, column_name)``."""
    out = df.copy()
    col = name or f"{left}_X_{right}"
    out[col] = to_num(out[left]) * to_num(out[right])
    return out, col


def add_bh_q_values(df: pd.DataFrame, p_col: str = "p_value", q_col: str = "q_value_bh") -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values to result tables with p-values.

    This is intentionally tiny and dependency-free.  It does not change p-values;
    it adds a companion column for multiple-testing awareness.
    """
    if df is None or df.empty or p_col not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out[p_col], errors="coerce")
    valid = p.notna()
    q = pd.Series(np.nan, index=out.index, dtype="float64")
    if valid.any():
        pv = p[valid].sort_values()
        m = len(pv)
        adjusted = pv * m / np.arange(1, m + 1)
        adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        q.loc[adjusted.index] = adjusted
    out[q_col] = q
    return out


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
