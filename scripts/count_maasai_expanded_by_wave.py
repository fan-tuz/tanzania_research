#!/usr/bin/env python3
"""
count_maasai_expanded_by_wave.py

Count, by NPS wave, how many household-wave observations fall under the
expanded Maasai treatment umbrella built from the 83-row locality dataset.

The script is designed to run AFTER add_ward_and_village_treatment_v4.py.
It reads spatialized analysis files such as:
  - analysis_core_1_4_spatial.csv
  - analysis_land_2_4_spatial.csv
  - analysis_pastoral_3_4_spatial.csv
or the restricted versions produced after adding game-reserve exposure:
  - analysis_core_1_4_spatial_restricted.csv
  - analysis_land_2_4_spatial_restricted.csv
  - analysis_pastoral_3_4_spatial_restricted.csv

Key definitions reported when available:
  - maasai_expanded_reference_ward: cluster falls inside a matched 2022 ward
    corresponding to one of the expanded 83 Maasai localities / aliases.
  - maasai_expanded_10km / 25km / 50km / 100km: cluster is within X km of the
    matched expanded Maasai reference ward union.
  - maasai_expanded_area_proxy: default main treatment alias from v4, equal to
    maasai_expanded_25km unless changed upstream.
  - treated_area: backward-compatible alias, also equal to maasai_expanded_25km
    in v4.

This script does not redefine Maasai status using livestock, grazing, TLU, or
pastoral livelihood variables. It only summarizes the geography-based expanded
Maasai treatment variables already created upstream.

Example:
  python3 scripts/count_maasai_expanded_by_wave.py \
    --input-dir output/spatial_restricted \
    --output diagnostics/maasai_expanded_counts_by_wave.csv \
    --boundary-bands 50,75,100

For exact matched-ward counts only:
  python3 scripts/count_maasai_expanded_by_wave.py \
    --input-dir output/spatial_restricted \
    --primary-col maasai_expanded_reference_ward
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FILES = [
    "analysis_core_1_4_spatial_restricted.csv",
    "analysis_land_2_4_spatial_restricted.csv",
    "analysis_pastoral_3_4_spatial_restricted.csv",
    "analysis_core_1_4_spatial.csv",
    "analysis_land_2_4_spatial.csv",
    "analysis_pastoral_3_4_spatial.csv",
]

MAASAI_COLUMNS_ORDERED = [
    "maasai_expanded_reference_ward",
    "maasai_expanded_10km",
    "maasai_expanded_25km",
    "maasai_expanded_area_proxy",
    "treated_area",
    "maasai_expanded_50km",
    "maasai_expanded_100km",
]

ID_CANDIDATES = [
    "hhid_wave",
    "hhid",
    "original_hhid",
]

CLUSTER_CANDIDATES = [
    "clusterid",
    "community_id",
    "ward2022_name",
]


def split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip() and x.strip().lower() != "none"]


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        found = lower.get(str(cand).lower())
        if found is not None:
            return found
    return None


def to_binary(series: pd.Series) -> pd.Series:
    """Positive numeric values are 1, zero is 0, missing remains missing."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(x.notna(), np.where(x > 0, 1.0, 0.0), np.nan), index=series.index)


def read_input_files(input_dir: Path, explicit_files: list[str]) -> list[Path]:
    if explicit_files:
        paths = []
        for f in explicit_files:
            p = Path(f)
            if not p.is_absolute():
                p = input_dir / p
            if not p.exists():
                raise FileNotFoundError(p)
            paths.append(p)
        return paths

    # Prefer restricted files if present. If not, use spatial files.
    restricted = [input_dir / f for f in DEFAULT_FILES[:3] if (input_dir / f).exists()]
    if restricted:
        return restricted
    spatial = [input_dir / f for f in DEFAULT_FILES[3:] if (input_dir / f).exists()]
    if spatial:
        return spatial

    # Final fallback: any plausible spatial analysis file.
    discovered = sorted(input_dir.glob("analysis_*_spatial*.csv"))
    if discovered:
        return discovered

    raise FileNotFoundError(
        f"No analysis_*_spatial*.csv files found in {input_dir}. "
        "Run add_ward_and_village_treatment_v4.py first, and optionally "
        "add_restricted_area_exposure.py before using spatial_restricted."
    )


def parse_boundary_bands(value: str | None) -> list[int]:
    out = []
    for x in split_csv(value):
        try:
            out.append(int(float(x)))
        except ValueError:
            raise ValueError(f"Invalid boundary band: {x}")
    return out


def make_sample_masks(df: pd.DataFrame, boundary_bands: list[int]) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"all": pd.Series(True, index=df.index)}
    if boundary_bands and "distance_to_game_reserve_boundary_km" in df.columns:
        bd = pd.to_numeric(df["distance_to_game_reserve_boundary_km"], errors="coerce")
        for b in boundary_bands:
            masks[f"boundary{b}km"] = bd.le(b).fillna(False)
    return masks


def summarize_one_definition(
    df: pd.DataFrame,
    dataset_name: str,
    sample_name: str,
    sample_mask: pd.Series,
    wave_value,
    maasai_col: str,
    id_col: str | None,
    cluster_col: str | None,
    primary_col: str,
) -> dict:
    d = df.loc[sample_mask & df["wave"].eq(wave_value)].copy()
    n_total = int(len(d))
    if n_total == 0:
        return {
            "dataset": dataset_name,
            "sample": sample_name,
            "wave": wave_value,
            "maasai_definition": maasai_col,
            "is_primary_definition": int(maasai_col == primary_col),
            "n_total_rows": 0,
            "n_maasai_rows": 0,
            "share_maasai_rows": np.nan,
            "n_nonmissing_maasai_flag": 0,
            "n_missing_maasai_flag": 0,
            "n_unique_households_total": np.nan,
            "n_unique_households_maasai": np.nan,
            "n_unique_clusters_total": np.nan,
            "n_unique_clusters_maasai": np.nan,
        }

    flag = to_binary(d[maasai_col])
    maasai_mask = flag.eq(1)
    nonmissing = flag.notna()
    n_maasai = int(maasai_mask.sum())

    row = {
        "dataset": dataset_name,
        "sample": sample_name,
        "wave": wave_value,
        "maasai_definition": maasai_col,
        "is_primary_definition": int(maasai_col == primary_col),
        "n_total_rows": n_total,
        "n_maasai_rows": n_maasai,
        "share_maasai_rows": float(n_maasai / n_total) if n_total else np.nan,
        "n_nonmissing_maasai_flag": int(nonmissing.sum()),
        "n_missing_maasai_flag": int((~nonmissing).sum()),
    }

    if id_col is not None:
        row["n_unique_households_total"] = int(d[id_col].nunique(dropna=True))
        row["n_unique_households_maasai"] = int(d.loc[maasai_mask, id_col].nunique(dropna=True))
    else:
        row["n_unique_households_total"] = np.nan
        row["n_unique_households_maasai"] = np.nan

    if cluster_col is not None:
        row["n_unique_clusters_total"] = int(d[cluster_col].nunique(dropna=True))
        row["n_unique_clusters_maasai"] = int(d.loc[maasai_mask, cluster_col].nunique(dropna=True))
    else:
        row["n_unique_clusters_total"] = np.nan
        row["n_unique_clusters_maasai"] = np.nan

    return row


def summarize_file(path: Path, primary_col: str | None, boundary_bands: list[int]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    dataset_name = path.name
    warnings: list[str] = []

    if "wave" not in df.columns:
        raise ValueError(f"{path} has no 'wave' column.")

    available_maasai_cols = [c for c in MAASAI_COLUMNS_ORDERED if c in df.columns]
    if not available_maasai_cols:
        raise ValueError(
            f"{path} has none of the expected expanded Maasai columns: {MAASAI_COLUMNS_ORDERED}. "
            "Run add_ward_and_village_treatment_v4.py before counting."
        )

    if primary_col is None:
        if "maasai_expanded_area_proxy" in available_maasai_cols:
            primary_col = "maasai_expanded_area_proxy"
        elif "maasai_expanded_25km" in available_maasai_cols:
            primary_col = "maasai_expanded_25km"
        else:
            primary_col = available_maasai_cols[0]
    elif primary_col not in df.columns:
        raise ValueError(f"Requested --primary-col {primary_col!r} is not in {path}.")

    if primary_col not in available_maasai_cols:
        available_maasai_cols = [primary_col] + available_maasai_cols

    id_col = first_existing(df.columns, ID_CANDIDATES)
    cluster_col = first_existing(df.columns, CLUSTER_CANDIDATES)
    if id_col is None:
        warnings.append(f"{path.name}: no household id column found among {ID_CANDIDATES}.")
    if cluster_col is None:
        warnings.append(f"{path.name}: no cluster column found among {CLUSTER_CANDIDATES}.")

    sample_masks = make_sample_masks(df, boundary_bands)
    if boundary_bands and "distance_to_game_reserve_boundary_km" not in df.columns:
        warnings.append(
            f"{path.name}: distance_to_game_reserve_boundary_km not found; boundary samples skipped. "
            "Use output/spatial_restricted after add_restricted_area_exposure.py for boundary counts."
        )

    waves = sorted(pd.to_numeric(df["wave"], errors="coerce").dropna().unique())
    # Use numeric comparison but keep a clean integer-looking wave in output.
    df = df.copy()
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")

    rows = []
    for sample_name, mask in sample_masks.items():
        for w in waves:
            wave_out = int(w) if float(w).is_integer() else w
            for c in available_maasai_cols:
                rows.append(
                    summarize_one_definition(
                        df=df,
                        dataset_name=dataset_name,
                        sample_name=sample_name,
                        sample_mask=mask,
                        wave_value=w,
                        maasai_col=c,
                        id_col=id_col,
                        cluster_col=cluster_col,
                        primary_col=primary_col,
                    ) | {"wave": wave_out}
                )

    return pd.DataFrame(rows), warnings


def print_primary_table(out: pd.DataFrame) -> None:
    primary = out[out["is_primary_definition"].eq(1)].copy()
    if primary.empty:
        return
    # Keep console output compact.
    cols = [
        "dataset", "sample", "wave", "maasai_definition",
        "n_total_rows", "n_maasai_rows", "share_maasai_rows",
        "n_unique_clusters_total", "n_unique_clusters_maasai",
    ]
    cols = [c for c in cols if c in primary.columns]
    primary = primary[cols].sort_values(["dataset", "sample", "wave"])
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    print("\nPrimary Maasai-umbrella counts by wave:")
    print(primary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="output/spatial_restricted", type=Path,
                    help="Directory containing spatialized analysis CSVs. Default: output/spatial_restricted")
    ap.add_argument("--files", default=None,
                    help="Optional comma-separated file names or paths. If omitted, standard analysis_*_spatial*.csv files are used.")
    ap.add_argument("--primary-col", default=None,
                    help="Main Maasai definition to highlight. Default: maasai_expanded_area_proxy if available, else maasai_expanded_25km.")
    ap.add_argument("--boundary-bands", default="50,75,100",
                    help="Optional comma-separated reserve-boundary samples in km. Set to none to skip. Requires distance_to_game_reserve_boundary_km.")
    ap.add_argument("--output", default="diagnostics/maasai_expanded_counts_by_wave.csv", type=Path,
                    help="Output CSV path.")
    args = ap.parse_args()

    explicit_files = split_csv(args.files)
    boundary_bands = parse_boundary_bands(args.boundary_bands)
    paths = read_input_files(args.input_dir, explicit_files)

    all_frames = []
    all_warnings = []
    for p in paths:
        frame, warnings = summarize_file(p, args.primary_col, boundary_bands)
        all_frames.append(frame)
        all_warnings.extend(warnings)

    out = pd.concat(all_frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print_primary_table(out)
    print(f"\nSaved counts: {args.output}")
    if all_warnings:
        print("\nWarnings:")
        for w in all_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
