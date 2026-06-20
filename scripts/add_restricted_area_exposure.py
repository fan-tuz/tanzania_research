#!/usr/bin/env python3
"""
add_restricted_area_exposure.py

Adds spatial exposure variables from external restricted-area polygons to already-spatialized
NPS analysis subsets.

Designed for game reserves and forest reserves, but it works with any polygon vector file
readable by GeoPandas (.shp, .gpkg, .geojson). CSV inputs are accepted only if they contain
a WKT geometry column (geometry/wkt/WKT). Attribute-only CSV exports are not enough.

Typical run:
python scripts/add_restricted_area_exposure.py \
  --spatial-dir output/spatial \
  --game-reserves data/geo/Tanzania_Game_Reserves.shp \
  --output-dir output/spatial_restricted \
  --diagnostics-dir diagnostics

Outputs keep the same filenames as the input spatial subsets. Main variables:
- distance_to_game_reserve_km, inside_game_reserve, within_10/25/50km_game_reserve
- nearest_game_reserve_name, nearest_game_reserve_designation
- forest-reserve and combined restricted-area variables only if --forest-reserves is passed
- *_x_pastoral_reference_area interactions when pastoral_reference_area/treated_area exists
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt

from utils import DEFAULT_PROJECTED_CRS, first_existing

PROJECTED_CRS = DEFAULT_PROJECTED_CRS


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s




def read_polygon_file(path: Path, label: str, layer: Optional[str] = None, assume_crs: Optional[str] = None) -> gpd.GeoDataFrame:
    if path is None:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
        geom_col = first_existing(df.columns, ["geometry", "geom", "wkt", "WKT", "the_geom"])
        if geom_col is None:
            raise ValueError(
                f"{label}: {path} looks like an attribute-only CSV. It has no WKT/geometry column. "
                "Please provide the original .shp/.gpkg/.geojson, not only the CSV attribute table."
            )
        geom = df[geom_col].apply(lambda x: wkt.loads(x) if pd.notna(x) and str(x).strip() else None)
        gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=assume_crs or "EPSG:4326")
    else:
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if gdf.crs is None:
            if assume_crs is None:
                raise ValueError(f"{label}: {path} has no CRS. Pass --assume-crs, e.g. EPSG:4326")
            gdf = gdf.set_crs(assume_crs)

    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError(f"{label}: no valid geometries found in {path}")
    # Fix invalid polygons where possible.
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def pick_name_cols(gdf: gpd.GeoDataFrame, kind: str) -> Tuple[str, Optional[str]]:
    if kind == "game":
        name = first_existing(gdf.columns, ["Name", "NAME", "English", "Short", "Jina", "ORIG_NAME"])
        desig = first_existing(gdf.columns, ["DESIG_ENG", "DESIG", "English", "Status", "IUCN_CAT"])
    else:
        name = first_existing(gdf.columns, ["NAME", "Name", "ORIG_NAME", "orig_name"])
        desig = first_existing(gdf.columns, ["DESIG_ENG", "DESIG", "IUCN_CAT", "STATUS"])
    if name is None:
        name = "__area_name"
        gdf[name] = kind
    return name, desig


def union_geometry(gdf: gpd.GeoDataFrame):
    g = gdf.to_crs(PROJECTED_CRS)
    if hasattr(g.geometry, "union_all"):
        return g.geometry.union_all()
    return g.geometry.unary_union


def add_area_exposure(df: pd.DataFrame, areas: gpd.GeoDataFrame, prefix: str, name_col: str, desig_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    dist_col = f"distance_to_{prefix}_km"
    inside_col = f"inside_{prefix}"
    nearest_name_col = f"nearest_{prefix}_name"
    nearest_desig_col = f"nearest_{prefix}_designation"

    for c in [dist_col, inside_col, nearest_name_col, nearest_desig_col]:
        out[c] = pd.NA

    if areas.empty:
        return out
    if "cluster_lon" not in out.columns or "cluster_lat" not in out.columns:
        raise ValueError("Input subset must contain cluster_lon and cluster_lat")

    out["__rowid"] = np.arange(len(out))
    coord_mask = out[["cluster_lon", "cluster_lat"]].notna().all(axis=1)
    if not coord_mask.any():
        out = out.drop(columns="__rowid")
        return out

    points = gpd.GeoDataFrame(
        out.loc[coord_mask, ["__rowid", "cluster_lon", "cluster_lat"]].copy(),
        geometry=gpd.points_from_xy(out.loc[coord_mask, "cluster_lon"], out.loc[coord_mask, "cluster_lat"]),
        crs="EPSG:4326",
    )

    areas_small_cols = [name_col, "geometry"] + ([desig_col] if desig_col else [])
    areas_small = areas[areas_small_cols].copy()
    areas_small = areas_small.rename(columns={name_col: nearest_name_col})
    if desig_col:
        areas_small = areas_small.rename(columns={desig_col: nearest_desig_col})
    else:
        areas_small[nearest_desig_col] = pd.NA

    # Inside flag.
    within = gpd.sjoin(points, areas_small[[nearest_name_col, nearest_desig_col, "geometry"]], how="left", predicate="within")
    inside_rows = within.dropna(subset=["index_right"]).drop_duplicates("__rowid", keep="first")
    out.loc[out["__rowid"].isin(inside_rows["__rowid"]), inside_col] = 1
    out.loc[coord_mask & out[inside_col].isna(), inside_col] = 0

    # Nearest area + distance.
    nearest = gpd.sjoin_nearest(
        points.to_crs(PROJECTED_CRS),
        areas_small.to_crs(PROJECTED_CRS),
        how="left",
        distance_col="__distance_m",
    ).drop_duplicates("__rowid", keep="first")
    nearest = pd.DataFrame(nearest.drop(columns=["geometry", "index_right"], errors="ignore"))
    nearest[dist_col] = nearest["__distance_m"] / 1000.0

    cols = ["__rowid", dist_col, nearest_name_col, nearest_desig_col]
    out = out.merge(nearest[cols], on="__rowid", how="left", suffixes=("", "_new"))
    for c in [dist_col, nearest_name_col, nearest_desig_col]:
        new = f"{c}_new"
        if new in out.columns:
            out[c] = out[new].combine_first(out[c])
            out = out.drop(columns=new)

    d = pd.to_numeric(out[dist_col], errors="coerce")
    for km in [10, 25, 50]:
        col = f"within_{km}km_{prefix}"
        out[col] = (d <= km).astype("Int64")
        out.loc[d.isna(), col] = pd.NA

    # Convenience alias for main threshold.
    out[f"near_{prefix}_25km"] = out[f"within_25km_{prefix}"]
    out[f"neg_distance_to_{prefix}_10km"] = -d / 10.0
    out[f"inverse_distance_to_{prefix}"] = 1.0 / (1.0 + d.clip(lower=0) / 10.0)
    out = out.drop(columns="__rowid")
    return out


def add_combined_restricted(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    dg = pd.to_numeric(out.get("distance_to_game_reserve_km"), errors="coerce") if "distance_to_game_reserve_km" in out.columns else pd.Series(np.nan, index=out.index)
    df = pd.to_numeric(out.get("distance_to_forest_reserve_km"), errors="coerce") if "distance_to_forest_reserve_km" in out.columns else pd.Series(np.nan, index=out.index)
    dist = pd.concat([dg, df], axis=1).min(axis=1)
    out["distance_to_restricted_area_km"] = dist
    out["nearest_restricted_area_type"] = np.where(dg <= df, "game_reserve", "forest_reserve")
    out.loc[dg.isna() & df.notna(), "nearest_restricted_area_type"] = "forest_reserve"
    out.loc[df.isna() & dg.notna(), "nearest_restricted_area_type"] = "game_reserve"
    out.loc[dist.isna(), "nearest_restricted_area_type"] = pd.NA

    out["nearest_restricted_area_name"] = np.where(
        out["nearest_restricted_area_type"].eq("game_reserve"),
        out.get("nearest_game_reserve_name"),
        out.get("nearest_forest_reserve_name"),
    )
    inside_g = pd.to_numeric(out.get("inside_game_reserve"), errors="coerce").fillna(0) if "inside_game_reserve" in out.columns else 0
    inside_f = pd.to_numeric(out.get("inside_forest_reserve"), errors="coerce").fillna(0) if "inside_forest_reserve" in out.columns else 0
    out["inside_restricted_area"] = ((inside_g == 1) | (inside_f == 1)).astype("Int64")

    for km in [10, 25, 50]:
        col = f"within_{km}km_restricted_area"
        out[col] = (dist <= km).astype("Int64")
        out.loc[dist.isna(), col] = pd.NA
    out["near_restricted_area_25km"] = out["within_25km_restricted_area"]
    out["neg_distance_to_restricted_area_10km"] = -dist / 10.0
    out["inverse_distance_to_restricted_area"] = 1.0 / (1.0 + dist.clip(lower=0) / 10.0)

    # Heterogeneity variables with pastoral reference geography.
    if "pastoral_reference_area" not in out.columns and "treated_area" in out.columns:
        out["pastoral_reference_area"] = out["treated_area"]
    if "pastoral_reference_area" in out.columns:
        p = pd.to_numeric(out["pastoral_reference_area"], errors="coerce")
        for base in ["within_25km_game_reserve", "within_25km_forest_reserve", "within_25km_restricted_area"]:
            if base in out.columns:
                out[f"{base}_x_pastoral_reference_area"] = pd.to_numeric(out[base], errors="coerce") * p
    return out


def diagnostics(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    vars_ = [
        "distance_to_game_reserve_km", "inside_game_reserve", "within_10km_game_reserve", "within_25km_game_reserve", "within_50km_game_reserve",
        "distance_to_forest_reserve_km", "inside_forest_reserve", "within_10km_forest_reserve", "within_25km_forest_reserve", "within_50km_forest_reserve",
        "distance_to_restricted_area_km", "inside_restricted_area", "within_10km_restricted_area", "within_25km_restricted_area", "within_50km_restricted_area",
        "within_25km_game_reserve_x_pastoral_reference_area", "within_25km_forest_reserve_x_pastoral_reference_area", "within_25km_restricted_area_x_pastoral_reference_area",
    ]
    rows = []
    for v in vars_:
        if v not in df.columns:
            continue
        s = pd.to_numeric(df[v], errors="coerce")
        rows.append({
            "dataset": dataset,
            "variable": v,
            "n_rows": len(df),
            "non_missing": int(df[v].notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "sum": float(s.sum()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial-dir", required=True, type=Path)
    ap.add_argument("--game-reserves", required=True, type=Path)
    ap.add_argument("--forest-reserves", default=None, type=Path, help="Optional. Use for appendix/robustness only; omitted by default to keep the main dataset game-reserve focused.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--diagnostics-dir", default="diagnostics", type=Path, help="Folder for diagnostics; separate from regression outputs.")
    ap.add_argument("--projected-crs", default=DEFAULT_PROJECTED_CRS, help="Projected CRS for distance calculations, e.g. EPSG:32736 or EPSG:32737.")
    ap.add_argument("--game-layer", default=None)
    ap.add_argument("--forest-layer", default=None)
    ap.add_argument("--assume-crs", default=None, help="Use only if the input vector has no CRS, e.g. EPSG:4326")
    args = ap.parse_args()

    global PROJECTED_CRS
    PROJECTED_CRS = args.projected_crs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print("Reading game reserves...")
    game = read_polygon_file(args.game_reserves, "game_reserves", layer=args.game_layer, assume_crs=args.assume_crs)
    game_name, game_desig = pick_name_cols(game, "game")
    print(f"  game reserves: {len(game):,}; name_col={game_name}; desig_col={game_desig}")

    forest = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    forest_name = forest_desig = None
    if args.forest_reserves is not None:
        print("Reading forest reserves for appendix/robustness...")
        forest = read_polygon_file(args.forest_reserves, "forest_reserves", layer=args.forest_layer, assume_crs=args.assume_crs)
        forest_name, forest_desig = pick_name_cols(forest, "forest")
        print(f"  forest reserves: {len(forest):,}; name_col={forest_name}; desig_col={forest_desig}")

    files = sorted(args.spatial_dir.glob("analysis_*_spatial.csv"))
    if not files:
        raise FileNotFoundError(f"No analysis_*_spatial.csv files found in {args.spatial_dir}")

    all_diag = []
    for f in files:
        print(f"\nAdding restricted-area exposure to {f.name}...")
        df = pd.read_csv(f, low_memory=False)
        out = add_area_exposure(df, game, "game_reserve", game_name, game_desig)
        if args.forest_reserves is not None:
            out = add_area_exposure(out, forest, "forest_reserve", forest_name, forest_desig)
            out = add_combined_restricted(out)
        out_path = args.output_dir / f.name
        out.to_csv(out_path, index=False)
        print(f"  saved {out_path}: {len(out):,} rows x {out.shape[1]:,} cols")
        all_diag.append(diagnostics(out, f.stem))

    diag = pd.concat(all_diag, ignore_index=True, sort=False) if all_diag else pd.DataFrame()
    diag.to_csv(args.diagnostics_dir / "restricted_area_exposure_diagnostics.csv", index=False)

    notes = f"""Restricted-area exposure added.\n\nInput spatial directory: {args.spatial_dir}\nOutput directory: {args.output_dir}\nGame reserves file: {args.game_reserves}\nForest reserves file: {args.forest_reserves if args.forest_reserves is not None else "not used in main dataset"}\n\nMain variables added:\n- distance_to_game_reserve_km, within_10/25/50km_game_reserve, inside_game_reserve\n- forest-reserve and combined restricted-area variables are added only when --forest-reserves is passed\n- *_x_pastoral_reference_area interactions for the 25km thresholds\n\nImportant caveat:\nDistances are computed from modified NPS cluster coordinates. Treat them as approximate spatial exposure measures, not exact household locations.\n"""
    (args.output_dir / "restricted_area_exposure_notes.txt").write_text(notes, encoding="utf-8")
    print("\nDone.")
    print(f"Diagnostics: {args.diagnostics_dir / 'restricted_area_exposure_diagnostics.csv'}")


if __name__ == "__main__":
    main()


"""
python scripts/add_restricted_area_exposure.py \
  --spatial-dir output/spatial \
  --game-reserves data/geo/Tanzania_Game_Reserves/Tanzania_Game_Reserves.shp \
  --output-dir output/spatial_restricted \
  --diagnostics-dir diagnostics \
  --assume-crs EPSG:4326
"""
