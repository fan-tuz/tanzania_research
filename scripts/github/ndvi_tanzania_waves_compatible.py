#!/usr/bin/env python3
"""
Build an EA-wave NDVI control compatible with the Tanzania NPS / Maasai pipeline.

Version v5 changes the assignment logic:
1. NDVI is sampled at all available georeferenced survey coordinate points.
2. The sampled NDVI values are merged back to household rows.
3. The final regression-ready control is collapsed to EA-by-wave level.

This avoids giving Wave 3 a different final unit of analysis when that wave has
household-level coordinates while other waves have cluster-level coordinates.
The regression can still cluster standard errors at EA level, and the NDVI
control is also merged at EA-wave level.

Main regression-ready output
----------------------------
    data/ndvi/ndvi_ea_wave_compatible.csv

Recommended merge key
---------------------
    ea, wave

Control variable
----------------
    ndvi_control

Example: sample from Google Earth Engine
----------------------------------------
python scripts/github/ndvi_tanzania_waves_compatible.py \
  --mode sample-ee \
  --nps-household output/nps_final/nps_relevant_household_wave.csv \
  --output-dir data/ndvi \
  --gee-project YOUR_GEE_PROJECT_ID \
  --scale 1000

Example: convert existing wide NDVI CSV
---------------------------------------
python scripts/github/ndvi_tanzania_waves_compatible.py \
  --mode from-wide-csv \
  --nps-household output/nps_final/nps_relevant_household_wave.csv \
  --wide-ndvi-csv data/ndvi/ndvi_by_ea.csv \
  --output-dir data/ndvi
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-06-27-ndvi-ea-wave-compatible-v5"

# Wave definitions follow the user's original Landsat NDVI setup.
WAVES: dict[int, dict[str, Any]] = {
    1: {"label": "W1", "years": [2008, 2009], "collections": ["LANDSAT/LT05/C02/T1_L2"]},
    2: {"label": "W2", "years": [2010, 2011], "collections": ["LANDSAT/LT05/C02/T1_L2", "LANDSAT/LE07/C02/T1_L2"]},
    3: {"label": "W3", "years": [2012, 2013], "collections": ["LANDSAT/LE07/C02/T1_L2"]},
    4: {"label": "W4", "years": [2014, 2015], "collections": ["LANDSAT/LC08/C02/T1_L2"]},
    5: {"label": "W5", "years": [2020, 2021], "collections": ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an EA-wave NDVI control for Tanzania NPS robustness regressions."
    )
    parser.add_argument("--mode", choices=["sample-ee", "from-wide-csv"], required=True)
    parser.add_argument("--nps-household", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    # NPS column names used by the current pipeline.
    parser.add_argument("--ea-col", default="ea")
    parser.add_argument("--wave-col", default="wave")
    parser.add_argument("--lon-col", default="cluster_lon")
    parser.add_argument("--lat-col", default="cluster_lat")

    # Coordinate precision used only to define unique sampling points.
    # It does not define the final regression merge key, which is EA-wave.
    parser.add_argument(
        "--sample-coordinate-decimals",
        default=6,
        type=int,
        help=(
            "Decimals used to identify unique coordinate points before GEE sampling. "
            "Default 6 keeps the available survey coordinates. The final output is still collapsed to EA-wave."
        ),
    )
    parser.add_argument(
        "--coord-decimals",
        default=None,
        type=int,
        help=(
            "Backward-compatible alias for --sample-coordinate-decimals. "
            "Use only if you intentionally want coarser point sampling; final output remains EA-wave."
        ),
    )

    # Earth Engine mode.
    parser.add_argument("--gee-project", default=None, help="Google Earth Engine project id.")
    parser.add_argument(
        "--scale",
        default=1000,
        type=int,
        help="Sampling scale in metres. 1000 is recommended with jittered NPS coordinates.",
    )
    parser.add_argument("--crs", default="EPSG:4326")

    # Existing wide NDVI mode.
    parser.add_argument("--wide-ndvi-csv", type=Path, default=None)
    parser.add_argument("--wide-id-col", default="ea_id")
    parser.add_argument("--wide-prefix", default="NDVI_W")

    parser.add_argument(
        "--write-merged-household",
        action="store_true",
        help="Also write nps_relevant_household_wave_with_ndvi.csv for diagnostics.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def wave_int(value: Any) -> int | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    rounded = int(round(float(value)))
    return rounded if rounded in WAVES else None


def coordinate_key_part(series: pd.Series, decimals: int) -> pd.Series:
    return numeric(series).round(decimals)


def make_point_wave_id(wave: pd.Series, lon: pd.Series, lat: pd.Series, decimals: int) -> pd.Series:
    lon_txt = lon.round(decimals).map(lambda x: f"{x:.{decimals}f}")
    lat_txt = lat.round(decimals).map(lambda x: f"{x:.{decimals}f}")
    return "W" + wave.astype(int).astype(str) + "__lon" + lon_txt + "__lat" + lat_txt


def active_sample_decimals(args: argparse.Namespace) -> int:
    if args.coord_decimals is not None:
        return int(args.coord_decimals)
    return int(args.sample_coordinate_decimals)


def load_clean_nps(nps_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    require_file(nps_path, "NPS household-wave file")
    nps = pd.read_csv(nps_path, low_memory=False)
    require_columns(nps, [args.ea_col, args.wave_col, args.lon_col, args.lat_col], "NPS household-wave file")

    work = nps[[args.ea_col, args.wave_col, args.lon_col, args.lat_col]].copy()
    work = work.rename(
        columns={
            args.ea_col: "ea",
            args.wave_col: "wave",
            args.lon_col: "cluster_lon",
            args.lat_col: "cluster_lat",
        }
    )
    work["original_row_id"] = np.arange(len(work))
    work["ea"] = clean_id(work["ea"])
    work["wave"] = work["wave"].map(wave_int)
    work["cluster_lon"] = numeric(work["cluster_lon"])
    work["cluster_lat"] = numeric(work["cluster_lat"])

    work["valid_ndvi_key_input"] = (
        work["ea"].notna()
        & work["wave"].notna()
        & work["cluster_lon"].notna()
        & work["cluster_lat"].notna()
    )
    work = work.loc[work["valid_ndvi_key_input"]].copy()
    work["wave"] = work["wave"].astype(int)
    work["ea_wave_id"] = work["ea"].astype(str) + "__W" + work["wave"].astype(str)

    decimals = active_sample_decimals(args)
    work["sample_lon"] = coordinate_key_part(work["cluster_lon"], decimals)
    work["sample_lat"] = coordinate_key_part(work["cluster_lat"], decimals)
    work["ndvi_point_wave_id"] = make_point_wave_id(work["wave"], work["sample_lon"], work["sample_lat"], decimals)
    work["sample_coordinate_decimals"] = decimals
    return work


def build_sampling_and_diagnostics(nps_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nps_points = load_clean_nps(nps_path, args)

    # Unique coordinate-wave points sent to Earth Engine.
    point_wave = (
        nps_points.groupby(["wave", "sample_lon", "sample_lat", "ndvi_point_wave_id"], as_index=False)
        .agg(
            cluster_lon=("cluster_lon", "median"),
            cluster_lat=("cluster_lat", "median"),
            n_household_rows_at_point_wave=("ea", "size"),
            n_unique_ea_at_point_wave=("ea", "nunique"),
            example_ea=("ea", "first"),
        )
    )
    ea_lists = (
        nps_points.groupby("ndvi_point_wave_id")["ea"]
        .apply(lambda x: "+".join(sorted(set(map(str, x.dropna())))))
        .rename("ea_list_at_point_wave")
        .reset_index()
    )
    point_wave = point_wave.merge(ea_lists, on="ndvi_point_wave_id", how="left")

    # Diagnostics on the original coordinate structure by wave.
    raw_unique_points = (
        nps_points.assign(raw_lon=nps_points["cluster_lon"].round(6), raw_lat=nps_points["cluster_lat"].round(6))
        [["wave", "raw_lon", "raw_lat"]]
        .drop_duplicates()
        .groupby("wave", as_index=False)
        .size()
        .rename(columns={"size": "raw_coordinate_points"})
    )
    wave_input_diag = (
        nps_points.groupby("wave", as_index=False)
        .agg(
            household_rows=("ea", "size"),
            ea_codes=("ea", "nunique"),
            ea_wave_rows=("ea_wave_id", "nunique"),
            sampled_coordinate_points=("ndvi_point_wave_id", "nunique"),
        )
        .merge(raw_unique_points, on="wave", how="left")
    )
    wave_input_diag["raw_coordinate_points"] = wave_input_diag["raw_coordinate_points"].fillna(0).astype(int)
    wave_input_diag["raw_to_sampled_point_ratio"] = (
        wave_input_diag["raw_coordinate_points"]
        / wave_input_diag["sampled_coordinate_points"].replace(0, np.nan)
    )

    # EA-wave diagnostic: how many sampling points are collapsed into each final EA-wave value.
    ea_wave_diag = (
        nps_points.groupby(["ea", "wave", "ea_wave_id"], as_index=False)
        .agg(
            n_household_rows_in_ea_wave=("ea", "size"),
            n_unique_sample_points_in_ea_wave=("ndvi_point_wave_id", "nunique"),
            n_unique_raw_lon=("cluster_lon", lambda x: int(pd.Series(x).round(6).nunique())),
            n_unique_raw_lat=("cluster_lat", lambda x: int(pd.Series(x).round(6).nunique())),
        )
    )
    ea_wave_diag["ea_wave_has_multiple_sample_points"] = ea_wave_diag["n_unique_sample_points_in_ea_wave"].gt(1).astype(int)

    input_diag = pd.DataFrame(
        [
            {"check": "household_rows_with_valid_wave_coordinates", "value": int(len(nps_points))},
            {"check": "coordinate_wave_points_sent_to_gee", "value": int(len(point_wave))},
            {"check": "ea_wave_rows_final_output", "value": int(len(ea_wave_diag))},
            {"check": "sample_coordinate_decimals_used", "value": int(active_sample_decimals(args))},
            {"check": "ea_wave_rows_with_multiple_sample_points", "value": int(ea_wave_diag["ea_wave_has_multiple_sample_points"].sum())},
        ]
    )
    return nps_points, point_wave, ea_wave_diag, pd.concat([input_diag, wave_input_diag.assign(check="wave_diagnostic")], ignore_index=True, sort=False)


# ----------------------------- Earth Engine logic -----------------------------

def initialize_earth_engine(project: str | None):
    try:
        import ee  # type: ignore
    except ImportError as error:
        raise ImportError("earthengine-api is not installed. Run: pip install earthengine-api") from error
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def build_tanzania_geometry(ee):
    return (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "United Republic of Tanzania"))
        .geometry()
    )


def apply_scale_and_mask_factory(ee, sensor: str):
    def apply_scale_and_mask(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        qa = image.select("QA_PIXEL")
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.addBands(optical, overwrite=True).updateMask(cloud_mask)
    return apply_scale_and_mask


def compute_ndvi_factory(ee, sensor: str):
    def compute_ndvi(image):
        if sensor in ("L5", "L7"):
            nir, red = "SR_B4", "SR_B3"
        else:
            nir, red = "SR_B5", "SR_B4"
        return image.normalizedDifference([nir, red]).rename("NDVI")
    return compute_ndvi


def build_wave_ndvi(ee, tanzania, wave: int):
    config = WAVES[wave]
    date_start = f"{config['years'][0]}-01-01"
    date_end = f"{config['years'][-1]}-12-31"

    collections = []
    for coll_id in config["collections"]:
        if "LT05" in coll_id:
            sensor = "L5"
        elif "LE07" in coll_id:
            sensor = "L7"
        else:
            sensor = "L89"

        coll = (
            ee.ImageCollection(coll_id)
            .filterBounds(tanzania)
            .filterDate(date_start, date_end)
            .map(apply_scale_and_mask_factory(ee, sensor))
            .map(compute_ndvi_factory(ee, sensor))
        )
        collections.append(coll)

    merged = collections[0]
    for extra in collections[1:]:
        merged = merged.merge(extra)
    return merged.median().clip(tanzania)


def sample_ee_points(point_wave: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ee = initialize_earth_engine(args.gee_project)
    tanzania = build_tanzania_geometry(ee)
    frames: list[pd.DataFrame] = []

    for wave in sorted(point_wave["wave"].dropna().unique()):
        if int(wave) not in WAVES:
            print(f"Skipping unsupported wave: {wave}")
            continue
        points = point_wave.loc[point_wave["wave"].eq(wave)].copy()
        print(f"Sampling NDVI for W{int(wave)}: {len(points):,} coordinate-wave points")
        image = build_wave_ndvi(ee, tanzania, int(wave))

        features = [
            ee.Feature(
                ee.Geometry.Point([float(row.cluster_lon), float(row.cluster_lat)]),
                {"ndvi_point_wave_id": str(row.ndvi_point_wave_id), "wave": int(row.wave)},
            )
            for row in points.itertuples(index=False)
        ]
        fc = ee.FeatureCollection(features)
        sampled = image.sampleRegions(
            collection=fc,
            properties=["ndvi_point_wave_id", "wave"],
            scale=int(args.scale),
            geometries=False,
        )
        data = sampled.getInfo()
        rows = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            rows.append({"ndvi_point_wave_id": props.get("ndvi_point_wave_id"), "ndvi_control_point": props.get("NDVI")})
        sampled_df = pd.DataFrame(rows)
        merged = points.merge(sampled_df, on="ndvi_point_wave_id", how="left")
        merged["ndvi_source_wave"] = f"W{int(wave)}"
        merged["ndvi_years"] = "-".join(str(y) for y in WAVES[int(wave)]["years"])
        merged["ndvi_sensor_collections"] = "+".join(WAVES[int(wave)]["collections"])
        merged["ndvi_sampling_scale_m"] = int(args.scale)
        merged["ndvi_source"] = "Google Earth Engine Landsat median NDVI sampled at georeferenced survey coordinate points"
        frames.append(merged)

    if not frames:
        raise ValueError("No waves were sampled. Check the NPS wave column.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["ndvi_control_point"] = numeric(out["ndvi_control_point"])
    return out


# -------------------------- Existing wide CSV conversion -----------------------

def from_wide_csv_ea_wave(args: argparse.Namespace, ea_wave_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.wide_ndvi_csv is None:
        raise ValueError("--wide-ndvi-csv is required when --mode from-wide-csv")
    require_file(args.wide_ndvi_csv, "wide NDVI CSV")
    ndvi = pd.read_csv(args.wide_ndvi_csv, low_memory=False)
    require_columns(ndvi, [args.wide_id_col], "wide NDVI CSV")

    ndvi = ndvi.copy()
    ndvi["ea"] = clean_id(ndvi[args.wide_id_col])
    wave_cols = [f"{args.wide_prefix}{wave}" for wave in WAVES if f"{args.wide_prefix}{wave}" in ndvi.columns]
    if not wave_cols:
        raise ValueError(f"No columns like {args.wide_prefix}1, {args.wide_prefix}2, ... were found in {args.wide_ndvi_csv}")

    long = ndvi[["ea", *wave_cols]].melt(id_vars=["ea"], var_name="ndvi_source_wave", value_name="ndvi_control")
    long["wave"] = long["ndvi_source_wave"].str.extract(r"(\d+)").astype(int)
    long["ndvi_control"] = numeric(long["ndvi_control"])

    ea_wave = ea_wave_diag.merge(long[["ea", "wave", "ndvi_source_wave", "ndvi_control"]], on=["ea", "wave"], how="left")
    ea_wave["ndvi_years"] = ea_wave["wave"].map(lambda w: "-".join(str(y) for y in WAVES[int(w)]["years"]) if int(w) in WAVES else "")
    ea_wave["ndvi_sensor_collections"] = ea_wave["wave"].map(lambda w: "+".join(WAVES[int(w)]["collections"]) if int(w) in WAVES else "")
    ea_wave["ndvi_sampling_scale_m"] = np.nan
    ea_wave["ndvi_source"] = f"Converted from wide EA-level NDVI CSV: {args.wide_ndvi_csv.name}"
    ea_wave["ndvi_assignment_level"] = "ea_wave_from_wide_csv"
    ea_wave["ndvi_control_median"] = ea_wave["ndvi_control"]
    ea_wave["ndvi_control_sd_within_ea_wave"] = np.nan
    ea_wave["n_household_rows_with_ndvi"] = np.where(
        ea_wave["ndvi_control"].notna(), ea_wave["n_household_rows_in_ea_wave"], 0
    ).astype(int)
    ea_wave["n_household_rows_missing_ndvi"] = (
        ea_wave["n_household_rows_in_ea_wave"] - ea_wave["n_household_rows_with_ndvi"]
    ).astype(int)
    ea_wave["household_ndvi_missing_rate"] = (
        ea_wave["n_household_rows_missing_ndvi"] / ea_wave["n_household_rows_in_ea_wave"].replace(0, np.nan)
    )
    ea_wave["n_nonmissing_sample_points_in_ea_wave"] = np.where(
        ea_wave["ndvi_control"].notna(), ea_wave["n_unique_sample_points_in_ea_wave"], 0
    ).astype(int)

    point_samples = pd.DataFrame(columns=["ndvi_point_wave_id", "wave", "ndvi_control_point"])
    return ea_wave, point_samples


# ------------------------------- Collapsing logic ------------------------------

def collapse_points_to_ea_wave(nps_points: pd.DataFrame, point_samples: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "ndvi_point_wave_id", "ndvi_control_point", "ndvi_source_wave", "ndvi_years",
        "ndvi_sensor_collections", "ndvi_sampling_scale_m", "ndvi_source",
    ]
    merge_cols = [c for c in merge_cols if c in point_samples.columns]
    point_values = point_samples[merge_cols].drop_duplicates("ndvi_point_wave_id").copy()
    hh = nps_points.merge(point_values, on="ndvi_point_wave_id", how="left")
    hh["ndvi_control_point"] = numeric(hh["ndvi_control_point"])

    ea_wave = (
        hh.groupby(["ea", "wave", "ea_wave_id"], as_index=False)
        .agg(
            ndvi_control=("ndvi_control_point", "mean"),
            ndvi_control_median=("ndvi_control_point", "median"),
            ndvi_control_sd_within_ea_wave=("ndvi_control_point", "std"),
            n_household_rows_in_ea_wave=("ea", "size"),
            n_household_rows_with_ndvi=("ndvi_control_point", lambda x: int(pd.Series(x).notna().sum())),
            n_household_rows_missing_ndvi=("ndvi_control_point", lambda x: int(pd.Series(x).isna().sum())),
            n_unique_sample_points_in_ea_wave=("ndvi_point_wave_id", "nunique"),
            n_nonmissing_sample_points_in_ea_wave=("ndvi_point_wave_id", lambda x: int(hh.loc[x.index].drop_duplicates("ndvi_point_wave_id")["ndvi_control_point"].notna().sum())),
            ndvi_source_wave=("ndvi_source_wave", "first"),
            ndvi_years=("ndvi_years", "first"),
            ndvi_sensor_collections=("ndvi_sensor_collections", "first"),
            ndvi_sampling_scale_m=("ndvi_sampling_scale_m", "first"),
            ndvi_source=("ndvi_source", "first"),
        )
    )
    ea_wave["household_ndvi_missing_rate"] = ea_wave["n_household_rows_missing_ndvi"] / ea_wave["n_household_rows_in_ea_wave"].replace(0, np.nan)
    ea_wave["ndvi_assignment_level"] = "ea_wave_mean_of_household_assigned_coordinate_ndvi"
    return ea_wave


def merged_household_with_ea_wave_ndvi(nps_path: Path, ea_wave: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    original = pd.read_csv(nps_path, low_memory=False)
    require_columns(original, [args.ea_col, args.wave_col], "NPS household-wave file")
    out = original.copy()
    out["ea"] = clean_id(out[args.ea_col])
    out["wave"] = out[args.wave_col].map(wave_int)
    merge_cols = [
        "ea", "wave", "ndvi_control", "ndvi_control_median", "n_unique_sample_points_in_ea_wave",
        "n_household_rows_in_ea_wave", "n_household_rows_with_ndvi", "ndvi_assignment_level",
    ]
    merge_cols = [c for c in merge_cols if c in ea_wave.columns]
    out = out.merge(ea_wave[merge_cols].drop_duplicates(["ea", "wave"]), on=["ea", "wave"], how="left")
    return out


def write_outputs(
    ea_wave: pd.DataFrame,
    point_samples: pd.DataFrame,
    ea_wave_diag: pd.DataFrame,
    input_diag: pd.DataFrame,
    nps_points: pd.DataFrame,
    nps_path: Path,
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_path = args.output_dir / "ndvi_ea_wave_compatible.csv"
    point_path = args.output_dir / "ndvi_coordinate_point_samples.csv"
    ea_diag_path = args.output_dir / "ndvi_ea_wave_coordinate_diagnostics.csv"
    diag_path = args.output_dir / "ndvi_assignment_diagnostics.csv"

    ea_order = [
        "ea", "wave", "ea_wave_id", "ndvi_control", "ndvi_control_median", "ndvi_control_sd_within_ea_wave",
        "n_household_rows_in_ea_wave", "n_household_rows_with_ndvi", "n_household_rows_missing_ndvi",
        "household_ndvi_missing_rate", "n_unique_sample_points_in_ea_wave", "n_nonmissing_sample_points_in_ea_wave",
        "ndvi_source_wave", "ndvi_years", "ndvi_sensor_collections", "ndvi_sampling_scale_m", "ndvi_source",
        "ndvi_assignment_level",
    ]
    ea_order = [c for c in ea_order if c in ea_wave.columns]
    ea_wave = ea_wave[ea_order + [c for c in ea_wave.columns if c not in ea_order]].sort_values(["wave", "ea"]).copy()
    ea_wave.to_csv(main_path, index=False)

    point_samples.sort_values(["wave", "ndvi_point_wave_id"] if "ndvi_point_wave_id" in point_samples.columns else ["wave"]).to_csv(point_path, index=False)
    ea_wave_diag.to_csv(ea_diag_path, index=False)

    wave_point_diag = (
        point_samples.groupby("wave", as_index=False)
        .agg(
            coordinate_points_sampled=("ndvi_point_wave_id", "nunique"),
            point_ndvi_nonmissing=("ndvi_control_point", lambda x: int(pd.Series(x).notna().sum())),
            point_ndvi_missing=("ndvi_control_point", lambda x: int(pd.Series(x).isna().sum())),
            point_ndvi_mean=("ndvi_control_point", "mean"),
        )
        if not point_samples.empty and "ndvi_control_point" in point_samples.columns
        else pd.DataFrame()
    )
    if not wave_point_diag.empty:
        wave_point_diag["point_ndvi_missing_rate"] = wave_point_diag["point_ndvi_missing"] / wave_point_diag["coordinate_points_sampled"].replace(0, np.nan)

    wave_ea_diag = (
        ea_wave.groupby("wave", as_index=False)
        .agg(
            ea_wave_rows=("ea_wave_id", "nunique"),
            ea_wave_rows_with_ndvi=("ndvi_control", lambda x: int(pd.Series(x).notna().sum())),
            ea_wave_rows_missing_ndvi=("ndvi_control", lambda x: int(pd.Series(x).isna().sum())),
            household_rows_represented=("n_household_rows_in_ea_wave", "sum"),
            household_rows_with_ndvi=("n_household_rows_with_ndvi", "sum"),
            mean_sample_points_per_ea_wave=("n_unique_sample_points_in_ea_wave", "mean"),
            max_sample_points_per_ea_wave=("n_unique_sample_points_in_ea_wave", "max"),
            ndvi_mean=("ndvi_control", "mean"),
        )
    )
    wave_ea_diag["ea_wave_missing_rate"] = wave_ea_diag["ea_wave_rows_missing_ndvi"] / wave_ea_diag["ea_wave_rows"].replace(0, np.nan)
    wave_ea_diag["household_missing_rate_after_ea_wave_merge"] = 1 - (wave_ea_diag["household_rows_with_ndvi"] / wave_ea_diag["household_rows_represented"].replace(0, np.nan))

    diagnostics = pd.concat(
        [
            input_diag,
            pd.DataFrame([
                {"check": "rows_in_final_ndvi_ea_wave_file", "value": int(len(ea_wave))},
                {"check": "rows_in_coordinate_point_sample_file", "value": int(len(point_samples))},
            ]),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics.to_csv(diag_path, index=False)

    if args.write_merged_household:
        merged = merged_household_with_ea_wave_ndvi(nps_path, ea_wave, args)
        merged_path = args.output_dir / "nps_relevant_household_wave_with_ndvi.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Wrote optional household merge: {merged_path}")

    print("\nDone.")
    print(f"Regression-ready EA-wave NDVI:    {main_path}")
    print(f"Coordinate point sample file:     {point_path}")
    print(f"EA-wave coordinate diagnostics:   {ea_diag_path}")
    print(f"Assignment diagnostics:           {diag_path}")
    print("\nRecommended regression merge key:")
    print("  ['ea', 'wave']")
    print("Control variable name:")
    print("  ndvi_control")

    print("\nShort diagnostic: coordinate points sampled by wave")
    if not wave_point_diag.empty:
        print(wave_point_diag.to_string(index=False))
    else:
        print("  Point-level diagnostics unavailable in this mode.")

    print("\nShort diagnostic: final EA-wave NDVI coverage by wave")
    print(wave_ea_diag.to_string(index=False))

    multi = int(ea_wave_diag["ea_wave_has_multiple_sample_points"].sum())
    total = int(len(ea_wave_diag))
    print("\nShort diagnostic: collapsed coordinate heterogeneity")
    print(f"  EA-wave groups with multiple sampled coordinate points: {multi:,} / {total:,}")
    if multi > 0:
        print("  Those multiple coordinate values are now averaged into one EA-wave NDVI control.")


def main() -> None:
    args = parse_args()
    print(f"ndvi_tanzania_waves_compatible version: {SCRIPT_VERSION}")

    nps_points, point_wave, ea_wave_diag, input_diag = build_sampling_and_diagnostics(args.nps_household, args)

    if args.mode == "sample-ee":
        point_samples = sample_ee_points(point_wave, args)
        ea_wave = collapse_points_to_ea_wave(nps_points, point_samples)
    elif args.mode == "from-wide-csv":
        ea_wave, point_samples = from_wide_csv_ea_wave(args, ea_wave_diag)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    write_outputs(ea_wave, point_samples, ea_wave_diag, input_diag, nps_points, args.nps_household, args)


if __name__ == "__main__":
    main()
