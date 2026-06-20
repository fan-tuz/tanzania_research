#!/usr/bin/env python3
"""
Add a 2022-ward localization layer and an expanded Maasai-area treatment to
Tanzania NPS analysis subset files.

Version 4 changes relative to v3
--------------------------------
- Uses the expanded 83-row Maasai locality CSV as the treatment reference list.
- Replaces the previous "reference places + pastoral characteristics" treatment
  logic. Livestock, TLU, grazing land, and mobility variables are no longer used
  to identify Maasai-associated treatment status.
- Keeps backward-compatible columns such as treated_area, treat_25km,
  pastoral_reference_25km, and maasai_pastoral_composite_proxy, but these are now
  aliases for the expanded Maasai geography, not composites with livelihood or
  landscape characteristics.
- Preserves spelling variants for matching by using match_name plus place_name
  and tier2_place_name aliases when present.

What this script does
---------------------
  NPS cluster coordinates
  -> 2022 ward polygon
  -> 2022 ward/district/region/council names
  -> expanded Maasai reference wards matched from the 83-row CSV
  -> distance to matched expanded Maasai reference wards
  -> treatment thresholds at 10 km, 25 km, 50 km, and 100 km

Main treatment convention
-------------------------
  treated_area = maasai_expanded_25km
  is_masai_25km = maasai_expanded_25km
  is_masai_50km = maasai_expanded_50km

Important caveats
-----------------
  - NPS cluster coordinates are modified/offset locations, not exact households.
  - 2022 ward boundaries are used as a spatial harmonization layer for older waves.
  - The expanded 83-row dataset contains villages/settlements as well as ward-like
    localities. This script can only match rows to 2022 ward polygons by name,
    district, and region. Rows that cannot be matched should be reviewed in the
    unmatched diagnostics and manually crosswalked when necessary.
  - This is a geographic Maasai-associated-area proxy, not individual ethnicity.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from utils import DEFAULT_PROJECTED_CRS, first_existing, norm_text, to_binary

PROJECTED_CRS = DEFAULT_PROJECTED_CRS

CORE_PASTORAL_DISTRICTS = {"longido", "monduli", "ngorongoro", "simanjiro", "kiteto"}
CORE_PASTORAL_REGIONS = {"arusha", "manyara"}

SUBSET_FILES = [
    "analysis_core_1_4.csv",
    "analysis_land_2_4.csv",
    "analysis_pastoral_3_4.csv",
]


def read_wards(path: Path, layer: str) -> gpd.GeoDataFrame:
    wards = gpd.read_file(path, layer=layer).to_crs("EPSG:4326").copy()

    mapping = {
        "ward2022_name": ["ward_name", "Ward_Name", "WARD_NAME", "ward", "Ward", "WARD"],
        "ward2022_code": ["ward_code", "Ward_Code", "WARD_CODE"],
        "district2022_name": ["dist_name", "district_name", "District", "DISTRICT"],
        "district2022_code": ["dist_code", "district_code", "DIST_CODE"],
        "region2022_name": ["reg_name", "region_name", "Region", "REGION"],
        "region2022_code": ["reg_code", "region_code", "REG_CODE"],
        "council2022_name": ["counc_name", "council_name", "Council", "COUNCIL"],
        "council2022_code": ["counc_code", "council_code", "COUNC_CODE"],
    }
    for out, candidates in mapping.items():
        src = first_existing(wards.columns, candidates)
        wards[out] = wards[src] if src else pd.NA

    wards["_ward_norm"] = wards["ward2022_name"].map(norm_text)
    wards["_district_norm"] = wards["district2022_name"].map(norm_text)
    wards["_region_norm"] = wards["region2022_name"].map(norm_text)

    # Kept as geography descriptors only. They are NOT treatment identifiers.
    wards["core_pastoral_region"] = wards["_region_norm"].isin(CORE_PASTORAL_REGIONS).astype(int)
    wards["core_pastoral_district"] = wards["_district_norm"].isin(CORE_PASTORAL_DISTRICTS).astype(int)
    wards["core_pastoral_ward"] = wards["core_pastoral_district"]

    return wards


def split_aliases(value) -> list[str]:
    """Split slash/semicolon/comma-separated spelling variants, preserving recall."""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    # Most variants in the expanded CSV are written as "A / B / C".
    parts = re.split(r"\s*/\s*|\s*;\s*|\s*,\s*", s)
    out = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p and p.lower() not in {"nan", "none", "<na>"}:
            out.append(p)
    return out


def read_reference_places(path: Path) -> pd.DataFrame:
    """Read either the legacy reference-place CSV or the expanded 83-row CSV.

    Required fields remain match_name, district_name, and region_name so that the
    same command-line option can be used. If main_treatment_expanded exists, it is
    the preferred inclusion flag; otherwise use_for_treatment is used.
    """
    ref = pd.read_csv(path, low_memory=False)
    required = {"match_name", "district_name", "region_name"}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Reference CSV missing columns: {sorted(missing)}")

    if "main_treatment_expanded" in ref.columns:
        ref = ref[pd.to_numeric(ref["main_treatment_expanded"], errors="coerce").fillna(0).eq(1)].copy()
    elif "use_for_treatment" in ref.columns:
        ref = ref[pd.to_numeric(ref["use_for_treatment"], errors="coerce").fillna(0).eq(1)].copy()

    alias_columns = [c for c in ["match_name", "place_name", "tier2_place_name"] if c in ref.columns]
    alias_lists = []
    for _, row in ref.iterrows():
        aliases: list[str] = []
        for c in alias_columns:
            aliases.extend(split_aliases(row.get(c)))
        # De-duplicate after normalization while preserving the original order.
        seen = set()
        clean_aliases = []
        for a in aliases:
            key = norm_text(a)
            if key and key not in seen:
                clean_aliases.append(a)
                seen.add(key)
        alias_lists.append(clean_aliases)

    ref["_aliases"] = alias_lists
    ref["_match_norm"] = ref["match_name"].map(norm_text)
    ref["_district_norm"] = ref["district_name"].map(norm_text)
    ref["_region_norm"] = ref["region_name"].map(norm_text)
    return ref


def reference_alias_table(ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in ref.iterrows():
        aliases = row.get("_aliases") or [row.get("match_name")]
        for alias in aliases:
            alias_norm = norm_text(alias)
            if not alias_norm:
                continue
            rows.append({
                "reference_row_index": idx,
                "alias_original": alias,
                "_alias_norm": alias_norm,
                "_district_norm": row.get("_district_norm"),
                "_region_norm": row.get("_region_norm"),
                "dataset_source": row.get("dataset_source", "reference_places"),
                "evidence_status": row.get("evidence_status", "reference_places"),
                "expanded_row_id": row.get("expanded_row_id", pd.NA),
            })
    return pd.DataFrame(rows).drop_duplicates()


def flag_reference_wards(wards: gpd.GeoDataFrame, ref: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag ward polygons matching the expanded Maasai locality reference list."""
    wards = wards.copy()
    aliases = reference_alias_table(ref)

    wards["maasai_expanded_reference_ward"] = 0
    wards["maasai_reference_ward"] = 0  # backward-compatible alias
    wards["maasai_expanded_match_aliases"] = pd.NA
    wards["maasai_expanded_evidence_statuses"] = pd.NA

    keys3 = set(zip(aliases["_alias_norm"], aliases["_district_norm"], aliases["_region_norm"]))
    keys2 = set(zip(aliases["_alias_norm"], aliases["_district_norm"]))

    def match_key(row) -> tuple[bool, str, str]:
        k3 = (row["_ward_norm"], row["_district_norm"], row["_region_norm"])
        k2 = (row["_ward_norm"], row["_district_norm"])
        ok = k3 in keys3 or k2 in keys2
        if not ok:
            return False, "", ""
        subset = aliases.loc[
            ((aliases["_alias_norm"].eq(row["_ward_norm"])) & (aliases["_district_norm"].eq(row["_district_norm"])) & (aliases["_region_norm"].eq(row["_region_norm"])))
            | ((aliases["_alias_norm"].eq(row["_ward_norm"])) & (aliases["_district_norm"].eq(row["_district_norm"])))
        ]
        return (
            True,
            "; ".join(sorted(set(subset["alias_original"].dropna().astype(str)))),
            "; ".join(sorted(set(subset["evidence_status"].dropna().astype(str)))),
        )

    matched_info = wards.apply(match_key, axis=1, result_type="expand")
    wards["maasai_expanded_reference_ward"] = matched_info[0].astype(int)
    wards["maasai_reference_ward"] = wards["maasai_expanded_reference_ward"]
    wards.loc[wards["maasai_expanded_reference_ward"].eq(1), "maasai_expanded_match_aliases"] = matched_info.loc[wards["maasai_expanded_reference_ward"].eq(1), 1]
    wards.loc[wards["maasai_expanded_reference_ward"].eq(1), "maasai_expanded_evidence_statuses"] = matched_info.loc[wards["maasai_expanded_reference_ward"].eq(1), 2]

    matched_alias_norms = set(wards.loc[wards["maasai_expanded_reference_ward"].eq(1), "_ward_norm"])
    unmatched = aliases.loc[~aliases["_alias_norm"].isin(matched_alias_norms)].copy()
    return wards, aliases, unmatched


def union_geometry(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def add_expanded_maasai_aliases(out: pd.DataFrame) -> pd.DataFrame:
    """Create treatment aliases expected by downstream scripts.

    These aliases intentionally use only expanded Maasai geography. They do not
    combine treatment with livestock, grazing, TLU, sales, or mobility variables.
    """
    out = out.copy()

    if "maasai_expanded_25km" in out.columns:
        out["treated_area"] = out["maasai_expanded_25km"]
        out["maasai_expanded_area_proxy"] = out["maasai_expanded_25km"]
        out["maasai_expanded_area_25km"] = out["maasai_expanded_25km"]
        # Explicit regression-ready Maasai identifiers.
        # Main definition: within 25 km of the expanded 83-row Maasai geography.
        out["is_masai_25km"] = out["maasai_expanded_25km"]
    if "maasai_expanded_50km" in out.columns:
        out["maasai_expanded_area_50km"] = out["maasai_expanded_50km"]
        # Robustness definition: wider 50 km buffer around the same 83-row geography.
        out["is_masai_50km"] = out["maasai_expanded_50km"]
    if "maasai_expanded_100km" in out.columns:
        out["maasai_expanded_area_100km"] = out["maasai_expanded_100km"]

    # Backward-compatible names. Despite the historical names, these are no longer
    # pastoral/livelihood composites in v4.
    if "maasai_expanded_25km" in out.columns:
        out["treat_25km"] = out["maasai_expanded_25km"]
        out["pastoral_reference_25km"] = out["maasai_expanded_25km"]
        out["maasai_pastoral_area_proxy_25km"] = out["maasai_expanded_25km"]
        out["maasai_pastoral_area_proxy"] = out["maasai_expanded_25km"]
        out["maasai_pastoral_composite_25km"] = out["maasai_expanded_25km"]
        out["maasai_pastoral_composite_proxy"] = out["maasai_expanded_25km"]
    if "maasai_expanded_50km" in out.columns:
        out["treat_50km"] = out["maasai_expanded_50km"]
        out["pastoral_reference_50km"] = out["maasai_expanded_50km"]
        out["maasai_pastoral_area_proxy_50km"] = out["maasai_expanded_50km"]
        out["maasai_pastoral_composite_50km"] = out["maasai_expanded_50km"]
    if "maasai_expanded_10km" in out.columns:
        out["treat_10km"] = out["maasai_expanded_10km"]

    for post in ["post_2009", "post_2013"]:
        if post in out.columns and "treated_area" in out.columns:
            out[f"treated_x_{post}"] = pd.to_numeric(out["treated_area"], errors="coerce") * pd.to_numeric(out[post], errors="coerce")
        if post in out.columns and "maasai_expanded_area_proxy" in out.columns:
            out[f"maasai_expanded_area_x_{post}"] = pd.to_numeric(out["maasai_expanded_area_proxy"], errors="coerce") * pd.to_numeric(out[post], errors="coerce")
            out[f"maasai_pastoral_area_x_{post}"] = out[f"maasai_expanded_area_x_{post}"]
            out[f"maasai_pastoral_composite_x_{post}"] = out[f"maasai_expanded_area_x_{post}"]
        if post in out.columns and "is_masai_25km" in out.columns:
            out[f"is_masai_25km_x_{post}"] = pd.to_numeric(out["is_masai_25km"], errors="coerce") * pd.to_numeric(out[post], errors="coerce")
        if post in out.columns and "is_masai_50km" in out.columns:
            out[f"is_masai_50km_x_{post}"] = pd.to_numeric(out["is_masai_50km"], errors="coerce") * pd.to_numeric(out[post], errors="coerce")

    return out


def attach_ward_attrs(df: pd.DataFrame, wards: gpd.GeoDataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__rowid"] = np.arange(len(out))

    needed = ["cluster_lon", "cluster_lat"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"Input dataset lacks {c}")

    coord_mask = out[needed].notna().all(axis=1)
    base_cols = [
        "ward2022_name", "ward2022_code",
        "district2022_name", "district2022_code",
        "region2022_name", "region2022_code",
        "council2022_name", "council2022_code",
        "core_pastoral_region", "core_pastoral_district", "core_pastoral_ward",
        "maasai_expanded_reference_ward", "maasai_reference_ward",
        "maasai_expanded_match_aliases", "maasai_expanded_evidence_statuses",
    ]

    for c in base_cols + ["ward_match_method", "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km"]:
        if c not in out.columns:
            out[c] = pd.NA

    if not coord_mask.any():
        out = add_expanded_maasai_aliases(out)
        out = out.drop(columns="__rowid")
        return out

    points = gpd.GeoDataFrame(
        out.loc[coord_mask, ["__rowid", "cluster_lon", "cluster_lat"]].copy(),
        geometry=gpd.points_from_xy(out.loc[coord_mask, "cluster_lon"], out.loc[coord_mask, "cluster_lat"]),
        crs="EPSG:4326",
    )

    wards_small = wards[base_cols + ["geometry"]].copy()

    within = gpd.sjoin(points, wards_small, how="left", predicate="within")
    within = within.drop_duplicates("__rowid", keep="first")
    within["ward_match_method"] = np.where(within["ward2022_name"].notna(), "within", pd.NA)

    # Nearest fallback for points that fall just outside polygons because of offsets/boundaries.
    missing_ids = within.loc[within["ward2022_name"].isna(), "__rowid"].tolist()
    if missing_ids:
        nearest = gpd.sjoin_nearest(
            points[points["__rowid"].isin(missing_ids)].to_crs(PROJECTED_CRS),
            wards_small.to_crs(PROJECTED_CRS),
            how="left",
            distance_col="distance_to_nearest_ward_m",
        ).drop_duplicates("__rowid", keep="first")
        nearest["ward_match_method"] = "nearest"
        nearest = nearest.to_crs("EPSG:4326")
        within = within[~within["__rowid"].isin(missing_ids)]
        within = pd.concat([within, nearest], ignore_index=True, sort=False)

    # Distance to expanded Maasai reference ward polygons.
    ref_geom = union_geometry(wards[wards["maasai_expanded_reference_ward"].eq(1)].to_crs(PROJECTED_CRS))
    if ref_geom is None:
        raise ValueError("No expanded Maasai reference wards matched. Check the expanded 83-row CSV against the 2022 ward layer.")

    points_proj = points.to_crs(PROJECTED_CRS).copy()
    dist = points_proj[["__rowid", "geometry"]].copy()
    dist["distance_to_maasai_expanded_km"] = dist.geometry.distance(ref_geom) / 1000.0
    dist = pd.DataFrame(dist.drop(columns="geometry"))

    joined = pd.DataFrame(within.drop(columns=["geometry", "index_right"], errors="ignore"))
    joined = joined.merge(dist, on="__rowid", how="left", suffixes=("", "_dist"))
    if "distance_to_maasai_expanded_km_dist" in joined.columns:
        joined["distance_to_maasai_expanded_km"] = joined["distance_to_maasai_expanded_km_dist"]
        joined = joined.drop(columns=["distance_to_maasai_expanded_km_dist"])
    joined["distance_to_maasai_reference_km"] = joined["distance_to_maasai_expanded_km"]

    out = out.merge(joined[["__rowid"] + base_cols + ["ward_match_method", "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km"]],
                    on="__rowid", how="left", suffixes=("", "_new"))

    for c in base_cols + ["ward_match_method", "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km"]:
        new = f"{c}_new"
        if new in out.columns:
            out[c] = out[new].combine_first(out[c])
            out = out.drop(columns=new)

    # Treatment thresholds. Main treatment is 25 km from expanded Maasai reference geography.
    d = pd.to_numeric(out["distance_to_maasai_expanded_km"], errors="coerce")
    for km in [10, 25, 50, 100]:
        col = f"maasai_expanded_{km}km"
        out[col] = (d <= km).astype("Int64")
        out.loc[d.isna(), col] = pd.NA

    out["treatment_bucket"] = pd.NA
    out.loc[d == 0, "treatment_bucket"] = "inside_expanded_maasai_reference_ward"
    out.loc[(d > 0) & (d <= 10), "treatment_bucket"] = "0_10km"
    out.loc[(d > 10) & (d <= 25), "treatment_bucket"] = "10_25km"
    out.loc[(d > 25) & (d <= 50), "treatment_bucket"] = "25_50km"
    out.loc[(d > 50) & (d <= 100), "treatment_bucket"] = "50_100km"
    out.loc[d > 100, "treatment_bucket"] = "gt_100km"

    out = add_expanded_maasai_aliases(out)
    out = out.drop(columns="__rowid")
    return out


def diagnostics(df: pd.DataFrame, dataset_name: str) -> list[dict]:
    vars_to_check = [
        "ward2022_name", "district2022_name", "region2022_name",
        "core_pastoral_region", "core_pastoral_district", "core_pastoral_ward",
        "maasai_expanded_reference_ward", "maasai_reference_ward",
        "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km",
        "maasai_expanded_10km", "maasai_expanded_25km", "maasai_expanded_50km", "maasai_expanded_100km",
        "is_masai_25km", "is_masai_50km",
        "treat_10km", "treat_25km", "treat_50km", "treated_area",
        "pastoral_reference_25km", "pastoral_reference_50km",
        "maasai_expanded_area_proxy", "maasai_expanded_area_25km", "maasai_expanded_area_50km", "maasai_expanded_area_100km",
        "maasai_pastoral_area_proxy", "maasai_pastoral_composite_proxy",
        "pastoral_livelihood_proxy", "pastoral_landscape_proxy",
    ]
    rows = []
    n = len(df)
    for v in vars_to_check:
        if v not in df.columns:
            continue
        s = df[v]
        sn = pd.to_numeric(s, errors="coerce")
        rows.append({
            "dataset": dataset_name,
            "variable": v,
            "n_rows": n,
            "non_missing": int(s.notna().sum()),
            "share_non_missing": float(s.notna().mean()) if n else np.nan,
            "mean": float(sn.mean()) if sn.notna().any() else np.nan,
            "min": float(sn.min()) if sn.notna().any() else np.nan,
            "max": float(sn.max()) if sn.notna().any() else np.nan,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets-dir", required=True, help="Directory containing analysis_core_1_4.csv, analysis_land_2_4.csv, analysis_pastoral_3_4.csv")
    ap.add_argument("--wards-gpkg", required=True, help="Path to tanzania_2022_wards.gpkg")
    ap.add_argument("--wards-layer", default="wards")
    ap.add_argument("--reference-csv", required=True, help="Expanded 83-row Maasai treatment CSV; legacy reference-place CSV also accepted")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--diagnostics-dir", default="diagnostics", help="Folder for spatial diagnostics; separate from regression outputs.")
    ap.add_argument("--projected-crs", default=DEFAULT_PROJECTED_CRS, help="Projected CRS used for distance calculations, e.g. EPSG:32736 or EPSG:32737.")
    args = ap.parse_args()

    global PROJECTED_CRS
    PROJECTED_CRS = args.projected_crs

    subsets_dir = Path(args.subsets_dir)
    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print("Reading 2022 wards...")
    wards = read_wards(Path(args.wards_gpkg), args.wards_layer)
    print(f"  wards: {len(wards):,}")

    print("Reading expanded Maasai treatment places...")
    ref = read_reference_places(Path(args.reference_csv))
    print(f"  treatment rows: {len(ref):,}")

    wards, aliases, unmatched = flag_reference_wards(wards, ref)
    matched = wards[wards["maasai_expanded_reference_ward"].eq(1)].copy()
    print(f"  matched expanded Maasai reference wards: {len(matched):,}")
    if matched.empty:
        raise SystemExit("No expanded Maasai reference wards matched. Check CSV names against 2022 ward layer.")

    matched_out = output_dir / "matched_maasai_expanded_reference_wards_2022.csv"
    matched.drop(columns="geometry").to_csv(matched_out, index=False)
    print(f"  saved matched expanded reference wards: {matched_out}")

    aliases_out = diagnostics_dir / "maasai_expanded_reference_aliases_used_for_matching.csv"
    aliases.to_csv(aliases_out, index=False)
    unmatched_out = diagnostics_dir / "unmatched_maasai_expanded_reference_aliases.csv"
    unmatched.to_csv(unmatched_out, index=False)
    print(f"  saved alias diagnostics: {aliases_out}")
    print(f"  saved unmatched alias diagnostics: {unmatched_out}")

    # Also save a GPKG for easy GIS inspection.
    try:
        matched.to_file(output_dir / "matched_maasai_expanded_reference_wards_2022.gpkg", layer="reference_wards", driver="GPKG")
    except Exception as e:
        print(f"  warning: could not write GPKG reference wards: {e}")

    all_diag = []
    for fname in SUBSET_FILES:
        in_path = subsets_dir / fname
        if not in_path.exists():
            print(f"Skipping missing file: {in_path}")
            continue
        print(f"\nSpatializing {fname}...")
        df = pd.read_csv(in_path, low_memory=False)
        out = attach_ward_attrs(df, wards)
        out_path = output_dir / fname.replace(".csv", "_spatial.csv")
        out.to_csv(out_path, index=False)
        print(f"  {len(out):,} rows x {out.shape[1]:,} cols -> {out_path}")
        all_diag.extend(diagnostics(out, fname.replace(".csv", "_spatial")))

    diag = pd.DataFrame(all_diag)
    diag_out = diagnostics_dir / "spatial_treatment_diagnostics.csv"
    diag.to_csv(diag_out, index=False)
    print(f"\nDiagnostics -> {diag_out}")
    print("Done.")


if __name__ == "__main__":
    main()
