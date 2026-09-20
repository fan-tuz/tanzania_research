#!/usr/bin/env python3
"""
Add game-reserve exposure and locality-based Maasai-associated treatment
variables to NPS files.

The Maasai-associated treatment is built from the two shapefiles stored in:

    data/github/maasai_parakuyo_datoga_spatial_dataset_expanded/

Expected files:
    community_wards_expanded.shp
    community_points_expanded.shp

By default, the treatment uses only features with:
    conf_score >= 0.80
    manual_chk == 0

Required household columns:
    hhid_wave, cluster_lon, cluster_lat

Variables added:
    distance_to_game_reserve_boundary_km
    inside_game_reserve
    within_50km_game_reserve_boundary
    within_75km_game_reserve_boundary
    within_100km_game_reserve_boundary

    distance_to_maasai_feature_km
    inside_maasai_ward
    is_maasai_10km
    is_maasai_25km

Example:
    python scripts/github/georeferencing.py \
        --input-dir output/nps_relevant \
        --output-dir output/nps_final \
        --game-reserves data/geo/Tanzania_Game_Reserves.shp \
        --maasai-data-dir \
            data/github/maasai_parakuyo_datoga_spatial_dataset_expanded
        --include_manual_checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-06-25-locality-treatment-v4"
PROJECTED_CRS = "EPSG:32736"

RESERVE_DISTANCE_BANDS_KM = (50, 75, 100)
MAASAI_DISTANCE_BANDS_KM = (10, 25)

DEFAULT_MAASAI_DATA_DIR = Path(
    "data/github/maasai_parakuyo_datoga_spatial_dataset_expanded"
)
DEFAULT_MAASAI_GROUPS = ("maasai", "parakuyo", "datoga_barabaig")

RESERVE_COLUMNS = [
    "distance_to_game_reserve_boundary_km",
    "inside_game_reserve",
    "within_50km_game_reserve_boundary",
    "within_75km_game_reserve_boundary",
    "within_100km_game_reserve_boundary",
]
MAASAI_COLUMNS = [
    "distance_to_maasai_feature_km",
    "inside_maasai_ward",
    "is_maasai_10km",
    "is_maasai_25km",
]
SPATIAL_COLUMNS = RESERVE_COLUMNS + MAASAI_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add reserve-boundary exposure and treatment variables based on "
            "the expanded Maasai/Parakuyo/Datoga spatial dataset."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--household-file",
        default="nps_relevant_household_wave.csv",
    )
    parser.add_argument(
        "--sales-file",
        default="nps_livestock_sales_long.csv",
    )

    parser.add_argument("--game-reserves", required=True, type=Path)
    parser.add_argument("--game-layer", default=None)

    parser.add_argument(
        "--maasai-data-dir",
        type=Path,
        default=DEFAULT_MAASAI_DATA_DIR,
    )
    parser.add_argument(
        "--maasai-wards-file",
        default="community_wards_expanded.shp",
    )
    parser.add_argument(
        "--maasai-points-file",
        default="community_points_expanded.shp",
    )
    parser.add_argument("--maasai-wards-layer", default=None)
    parser.add_argument("--maasai-points-layer", default=None)

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.80,
        help=(
            "Minimum conf_score used for treatment features. "
            "Default: 0.80."
        ),
    )
    parser.add_argument(
        "--include-manual-checks",
        action="store_true",
        help=(
            "Include features with manual_chk == 1. By default they are "
            "excluded from the treatment."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_MAASAI_GROUPS),
        help=(
            "Values of group_nm included in the treatment. "
            "Default: maasai parakuyo datoga_barabaig."
        ),
    )

    parser.add_argument("--projected-crs", default=PROJECTED_CRS)
    parser.add_argument(
        "--assume-crs",
        default=None,
        help="CRS assigned only when an input vector file has no CRS metadata.",
    )
    args = parser.parse_args()

    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1.")

    return args


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def read_vector(
    path: Path,
    layer: str | None,
    assume_crs: str | None,
    label: str,
) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    data = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if data.empty:
        raise ValueError(f"{label} file is empty: {path}")
    if data.crs is None:
        if assume_crs is None:
            raise ValueError(f"{label} has no CRS. Pass --assume-crs.")
        data = data.set_crs(assume_crs)

    data = data.loc[data.geometry.notna() & ~data.geometry.is_empty].copy()
    if data.empty:
        raise ValueError(f"{label} contains no usable geometry.")
    return data


def select_treatment_features(
    data: gpd.GeoDataFrame,
    label: str,
    min_confidence: float,
    include_manual_checks: bool,
    groups: list[str],
) -> gpd.GeoDataFrame:
    """
    Keep only credible features from the expanded community dataset.

    The shapefiles generated for this project use:
        conf_score : confidence score between 0 and 1
        manual_chk : 1 when manual validation is still required
        group_nm   : maasai, parakuyo, or datoga_barabaig
    """
    required = {"conf_score", "manual_chk", "group_nm"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

    selected_groups = {str(group).strip().casefold() for group in groups}
    confidence = pd.to_numeric(data["conf_score"], errors="coerce")
    manual_check = pd.to_numeric(data["manual_chk"], errors="coerce").fillna(1)
    group_name = data["group_nm"].astype("string").str.strip().str.casefold()

    keep = confidence.ge(min_confidence) & group_name.isin(selected_groups)
    if not include_manual_checks:
        keep &= manual_check.eq(0)

    selected = data.loc[keep].copy()
    if selected.empty:
        raise ValueError(
            f"No {label.lower()} remain after applying the treatment filters."
        )

    print(f"\n{label} selection")
    print(f"  input features       : {len(data):,}")
    print(f"  selected features    : {len(selected):,}")
    print(f"  minimum confidence   : {min_confidence:.2f}")
    print(f"  include manual checks: {include_manual_checks}")
    print(
        "  groups               : "
        + ", ".join(sorted(selected["group_nm"].astype(str).unique()))
    )
    print(
        selected.groupby("group_nm", dropna=False)
        .size()
        .rename("features")
        .to_string()
    )
    return selected


def union_geometry(data: gpd.GeoDataFrame):
    if hasattr(data.geometry, "union_all"):
        return data.geometry.union_all()
    return data.geometry.unary_union


def household_points(household: pd.DataFrame) -> gpd.GeoDataFrame:
    required = {"hhid_wave", "cluster_lon", "cluster_lat"}
    missing = sorted(required.difference(household.columns))
    if missing:
        raise ValueError(f"Household file is missing columns: {missing}")

    longitude = pd.to_numeric(household["cluster_lon"], errors="coerce")
    latitude = pd.to_numeric(household["cluster_lat"], errors="coerce")
    valid = longitude.between(-180, 180) & latitude.between(-90, 90)
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum()):,} household rows do not have valid coordinates."
        )

    return gpd.GeoDataFrame(
        {"_rowid": np.arange(len(household))},
        geometry=gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    )


def minimum_distance_km(
    points_metric: gpd.GeoDataFrame,
    geometries_metric: list,
) -> np.ndarray:
    """Return the minimum distance to any supplied geometry collection."""
    distances: list[np.ndarray] = []
    for geometry in geometries_metric:
        if geometry is None or geometry.is_empty:
            continue
        distances.append(
            points_metric.geometry.distance(geometry).to_numpy() / 1000.0
        )

    if not distances:
        raise ValueError("No usable treatment geometry was supplied.")

    return np.minimum.reduce(distances)


def add_spatial_variables(
    household: pd.DataFrame,
    reserves: gpd.GeoDataFrame,
    maasai_wards: gpd.GeoDataFrame,
    maasai_points: gpd.GeoDataFrame,
    projected_crs: str,
) -> pd.DataFrame:
    out = household.drop(columns=SPATIAL_COLUMNS, errors="ignore").copy()
    points = household_points(out)
    points_metric = points.to_crs(projected_crs)

    # Game-reserve exposure.
    reserves_metric = reserves.to_crs(projected_crs)
    reserve_union = union_geometry(reserves_metric)
    reserve_boundaries = union_geometry(
        gpd.GeoDataFrame(
            geometry=reserves_metric.geometry.boundary,
            crs=projected_crs,
        )
    )
    reserve_distance_km = (
        points_metric.geometry.distance(reserve_boundaries).to_numpy() / 1000.0
    )
    inside_reserve = points_metric.geometry.intersects(reserve_union)

    # Locality-based treatment. Distance to a polygon is zero when the
    # household lies inside it; otherwise it is measured from the polygon edge.
    maasai_wards_metric = maasai_wards.to_crs(projected_crs)
    maasai_points_metric = maasai_points.to_crs(projected_crs)

    ward_union = union_geometry(maasai_wards_metric)
    point_union = union_geometry(maasai_points_metric)
    maasai_distance_km = minimum_distance_km(
        points_metric,
        [ward_union, point_union],
    )
    inside_maasai_ward = points_metric.geometry.intersects(ward_union)

    out["distance_to_game_reserve_boundary_km"] = reserve_distance_km
    out["inside_game_reserve"] = inside_reserve.astype("int8").to_numpy()
    for km in RESERVE_DISTANCE_BANDS_KM:
        out[f"within_{km}km_game_reserve_boundary"] = (
            reserve_distance_km <= km
        ).astype("int8")

    out["distance_to_maasai_feature_km"] = maasai_distance_km
    out["inside_maasai_ward"] = inside_maasai_ward.astype("int8").to_numpy()
    for km in MAASAI_DISTANCE_BANDS_KM:
        out[f"is_maasai_{km}km"] = (
            maasai_distance_km <= km
        ).astype("int8")

    return out


def merge_household_variables(
    sales: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.DataFrame:
    if "hhid_wave" not in sales.columns:
        raise ValueError("Livestock file is missing hhid_wave.")

    household_keys = clean_id(household["hhid_wave"])
    if household_keys.isna().any() or household_keys.duplicated().any():
        raise ValueError("Household hhid_wave must be complete and unique.")

    crosswalk = household[["hhid_wave"] + SPATIAL_COLUMNS].copy()
    crosswalk["hhid_wave"] = household_keys

    out = sales.drop(columns=SPATIAL_COLUMNS, errors="ignore").copy()
    out["hhid_wave"] = clean_id(out["hhid_wave"])
    return out.merge(crosswalk, on="hhid_wave", how="left", validate="many_to_one")


def print_diagnostics(household: pd.DataFrame) -> None:
    print("\nSpatial-variable diagnostics")
    print(f"  household rows                  : {len(household):,}")
    print(
        "  reserve distance available     : "
        f"{household['distance_to_game_reserve_boundary_km'].notna().sum():,}"
    )
    print(
        "  community distance available   : "
        f"{household['distance_to_maasai_feature_km'].notna().sum():,}"
    )

    diagnostic_columns = [
        "inside_game_reserve",
        "inside_maasai_ward",
        "is_maasai_10km",
        "is_maasai_25km",
    ] + [
        f"within_{km}km_game_reserve_boundary"
        for km in RESERVE_DISTANCE_BANDS_KM
    ]
    for column in diagnostic_columns:
        print(f"  {column:<35}: {household[column].mean():.4f}")

    if "survey_round" in household.columns:
        table = household.groupby("survey_round", sort=False).agg(
            households=("hhid_wave", "size"),
            inside_maasai_ward=("inside_maasai_ward", "sum"),
            is_maasai_10km=("is_maasai_10km", "sum"),
            is_maasai_25km=("is_maasai_25km", "sum"),
            inside_game_reserve=("inside_game_reserve", "sum"),
            within_75km=("within_75km_game_reserve_boundary", "sum"),
        )
        print("\n" + table.to_string())


def main() -> None:
    args = parse_args()
    print(f"georeferencing version: {SCRIPT_VERSION}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    household_path = args.input_dir / args.household_file
    sales_path = args.input_dir / args.sales_file
    if not household_path.exists():
        raise FileNotFoundError(household_path)
    if not sales_path.exists():
        raise FileNotFoundError(sales_path)

    household = pd.read_csv(household_path, low_memory=False)
    sales = pd.read_csv(sales_path, low_memory=False)

    reserves = read_vector(
        args.game_reserves,
        args.game_layer,
        args.assume_crs,
        "Game reserves",
    )

    maasai_wards_path = args.maasai_data_dir / args.maasai_wards_file
    maasai_points_path = args.maasai_data_dir / args.maasai_points_file

    maasai_wards = read_vector(
        maasai_wards_path,
        args.maasai_wards_layer,
        args.assume_crs,
        "Maasai-associated wards",
    )
    maasai_points = read_vector(
        maasai_points_path,
        args.maasai_points_layer,
        args.assume_crs,
        "Maasai-associated points",
    )

    maasai_wards = select_treatment_features(
        maasai_wards,
        "Maasai-associated wards",
        args.min_confidence,
        args.include_manual_checks,
        args.groups,
    )
    maasai_points = select_treatment_features(
        maasai_points,
        "Maasai-associated points",
        args.min_confidence,
        args.include_manual_checks,
        args.groups,
    )

    household_out = add_spatial_variables(
        household,
        reserves,
        maasai_wards,
        maasai_points,
        args.projected_crs,
    )
    sales_out = merge_household_variables(sales, household_out)

    household_output = args.output_dir / args.household_file
    sales_output = args.output_dir / args.sales_file
    household_out.to_csv(household_output, index=False)
    sales_out.to_csv(sales_output, index=False)

    print_diagnostics(household_out)
    print(f"\nHousehold output: {household_output}")
    print(f"Livestock output: {sales_output}")


if __name__ == "__main__":
    main()
