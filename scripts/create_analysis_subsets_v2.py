#!/usr/bin/env python3
"""
Create three analysis-ready subsets from the giant NPS multi-wave file.

Outputs:
  1. analysis_core_1_4.csv       waves 1-4, welfare / shocks / geography backbone
  2. analysis_land_2_4.csv       waves 2-4, land-event exposure + welfare
  3. analysis_pastoral_3_4.csv   waves 3-4, pastoral mechanisms + welfare
  4. analysis_subset_diagnostics.csv

V2 additions:
  - keeps survey weights/design variables
  - keeps geo_* climate/environment/market-access controls
  - keeps herd composition/TLU variables
  - builds pastoral livelihood/landscape proxies for later Maasai-area composite flags

Usage:
  python scripts/create_analysis_subsets_v2.py \
    --input data/refined_multiwave/nps_analysis_household_wave.csv \
    --output output/analysis_subsets \
    --diagnostics-dir diagnostics
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from utils import combine_first_columns, existing, first_existing, safe_log, safe_ratio, to_num





def matching(df: pd.DataFrame, prefixes: list[str] | None = None, contains: list[str] | None = None) -> list[str]:
    """Return columns matching prefixes or substrings, preserving original order."""
    prefixes = prefixes or []
    contains = contains or []
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(lc.startswith(p.lower()) for p in prefixes) or any(x.lower() in lc for x in contains):
            out.append(c)
    return out










def clean_pct(series: pd.Series) -> pd.Series:
    x = to_num(series)
    return x.where((x >= 0) & (x <= 100))


def add_constructed_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of harmonized/analysis variables. Nothing fancy."""
    df = df.copy()

    # Resolve duplicate IDs created by joins in the giant build.
    df = combine_first_columns(df, "community_id", ["community_id", "community_id_x", "community_id_y", "clusterid"])
    df = combine_first_columns(df, "cons_hhsize", ["cons_hhsize", "cons_hhsize_x", "cons_hhsize_y", "hh_size"])
    df = combine_first_columns(df, "cons_adulteq", ["cons_adulteq", "cons_adulteq_x", "cons_adulteq_y"])

    # Basic logs for welfare outcomes.
    if "cons_total_monthly_real_pc" in df.columns:
        df["log_cons_pc"] = safe_log(df["cons_total_monthly_real_pc"])
    if "cons_total_monthly_real_ae" in df.columns:
        df["log_cons_ae"] = safe_log(df["cons_total_monthly_real_ae"])
    if "cons_total_monthly_real" in df.columns:
        df["log_cons_total"] = safe_log(df["cons_total_monthly_real"])

    # Time flags. These are mechanical; treatment definitions come later.
    if "survey_year" in df.columns:
        yr = to_num(df["survey_year"])
        df["post_2009"] = (yr >= 2010).astype("Int64")
        df["post_2013"] = (yr >= 2014).astype("Int64")

    # Clean percentages. Keep raw variables in giant DB, use *_clean in analysis.
    pct_vars = [
        "grazing_land_pct",
        "cultivation_villagers_pct",
        "outsider_agrobusiness_pct",
        "forest_pct",
        "wetland_pct",
    ]
    for c in pct_vars:
        if c in df.columns:
            df[f"{c}_clean"] = clean_pct(df[c])

    # Land-event timing variables.
    if "first_land_event_year" in df.columns and "survey_year" in df.columns:
        event_year = to_num(df["first_land_event_year"])
        survey_year = to_num(df["survey_year"])
        df["event_already_occurred"] = np.where(
            event_year.notna(),
            (event_year <= survey_year).astype(int),
            np.nan,
        )
        df["years_since_land_event"] = survey_year - event_year

    # Pastoral mechanisms. Use log1p because zero livestock/sales is meaningful.
    if "livestock_total_current" in df.columns:
        x = to_num(df["livestock_total_current"])
        df["log1p_livestock_total_current"] = np.log1p(x.where(x >= 0))

    if "livestock_sales_value" in df.columns:
        x = to_num(df["livestock_sales_value"])
        df["log1p_livestock_sales_value"] = np.log1p(x.where(x >= 0))

    if {"livestock_total_current", "livestock_total_12m_ago"}.issubset(df.columns):
        current = to_num(df["livestock_total_current"])
        past = to_num(df["livestock_total_12m_ago"])
        df["livestock_change_12m"] = current - past
        df["livestock_decline_12m"] = np.where(
            df["livestock_change_12m"].notna(),
            (df["livestock_change_12m"] < 0).astype(int),
            np.nan,
        )

    # Herd/TLU intensity controls and outcomes. These are used only when build_db_v5
    # could construct them from LF/AG livestock modules.
    if "livestock_total_current" in df.columns and "cons_adulteq" in df.columns:
        df["livestock_total_per_ae"] = safe_ratio(df["livestock_total_current"], df["cons_adulteq"])
    if "livestock_total_current" in df.columns and "hh_size" in df.columns:
        df["livestock_total_per_capita"] = safe_ratio(df["livestock_total_current"], df["hh_size"])
    if "livestock_tlu_default" in df.columns and "cons_adulteq" in df.columns:
        df["livestock_tlu_per_ae_default"] = safe_ratio(df["livestock_tlu_default"], df["cons_adulteq"])
    if "livestock_tlu_default" in df.columns and "hh_size" in df.columns:
        df["livestock_tlu_per_capita_default"] = safe_ratio(df["livestock_tlu_default"], df["hh_size"])

    # Livelihood/landscape proxies. These are NOT ethnicity identifiers. They are
    # used later with Maasai/pastoral reference geography to create a conservative
    # Maasai-associated pastoral-area proxy.
    livestock_signals = []
    for c in ["has_livestock_current", "livestock_migration_any"]:
        if c in df.columns:
            livestock_signals.append(to_num(df[c]).gt(0))
    if "livestock_total_current" in df.columns:
        livestock_signals.append(to_num(df["livestock_total_current"]).ge(10))
    if "livestock_tlu_default" in df.columns:
        livestock_signals.append(to_num(df["livestock_tlu_default"]).ge(3))
    if "livestock_sales_value" in df.columns:
        livestock_signals.append(to_num(df["livestock_sales_value"]).gt(0))
    if livestock_signals:
        sig = livestock_signals[0]
        for x in livestock_signals[1:]:
            sig = sig | x
        df["pastoral_livelihood_proxy"] = np.where(sig.notna(), sig.astype(int), np.nan)

    landscape_signals = []
    if "grazing_land_pct_clean" in df.columns:
        landscape_signals.append(to_num(df["grazing_land_pct_clean"]).ge(50))
    if "livestock_migration_any" in df.columns:
        landscape_signals.append(to_num(df["livestock_migration_any"]).gt(0))
    if landscape_signals:
        sig = landscape_signals[0]
        for x in landscape_signals[1:]:
            sig = sig | x
        df["pastoral_landscape_proxy"] = np.where(sig.notna(), sig.astype(int), np.nan)

    return df


def build_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the three lean analysis datasets."""

    id_geo = [
        "hhid", "hhid_wave", "original_hhid", "wave", "survey_year",
        "weight", "rural", "clusterid", "strataid", "community_id",
        "region", "district", "ward",
        "cluster_lat", "cluster_lon", "has_cluster_coordinates",
        # These may be added later through the 2022 ward join / treatment build.
        "ward_name", "district_name", "region_name", "council_name",
        "ward2022_name", "district2022_name", "region2022_name", "council2022_name",
        "core_pastoral_region", "core_pastoral_ward", "core_pastoral_district", "arusha_manyara",
        "distance_to_core_pastoral_ward_km", "distance_to_treated_area_km",
        "distance_to_maasai_reference_km", "maasai_reference_ward",
        "treat_10km", "treat_25km", "treat_50km", "treated_area", "treated_x_post",
        "pastoral_reference_25km", "pastoral_reference_50km",
        "maasai_pastoral_area_proxy", "maasai_pastoral_composite_proxy",
        "maasai_pastoral_composite_25km", "maasai_pastoral_composite_50km",
    ]

    hh_controls = [
        "hh_size", "cons_hhsize", "cons_adulteq",
        "mean_age", "max_age", "n_children_u15", "n_adults_15plus",
        "n_female", "share_female", "head_age", "head_sex",
    ]

    welfare = [
        "cons_total_monthly_real", "cons_total_monthly_nominal",
        "cons_total_monthly_real_pc", "cons_total_monthly_real_ae",
        "cons_foodbev_real", "cons_food_in_real", "cons_food_out_real",
        "cons_food_share_real", "log_cons_total", "log_cons_pc", "log_cons_ae",
    ]

    shocks_time = [
        "n_shock_records", "n_shocks_occurred",
        "post_2009", "post_2013",
    ] + matching(df, prefixes=["shock_"])

    # build_db_v5 creates geo_* variables from LSMS geovariables. Keep them in all
    # subsets so climate/environment and market-access controls can be added later.
    geo_controls = matching(df, prefixes=["geo_"])

    land_events = [
        "any_land_event",
        "event_outside_investor_appropriation",
        "event_reserve_land_declaration",
        "event_public_use_allocation",
        "first_land_event_year",
        "event_already_occurred",
        "years_since_land_event",
        "outside_investor_appropriation_year",
        "reserve_land_declaration_year",
        "public_use_allocation_year",
        "hh_affected_outside_investor_appropriation",
        "hh_affected_reserve_land_declaration",
        "hh_affected_public_use_allocation",
        "compensation_outside_investor_appropriation",
        "compensation_reserve_land_declaration",
        "compensation_public_use_allocation",
    ]

    land_composition = [
        "grazing_land_pct", "grazing_land_pct_clean", "grazing_land_acres",
        "cultivation_villagers_pct", "cultivation_villagers_pct_clean",
        "outsider_agrobusiness_pct", "outsider_agrobusiness_pct_clean",
        "forest_pct", "forest_pct_clean",
        "wetland_pct", "wetland_pct_clean",
        "has_village_land_certificate",
    ]

    community_migration = [
        "perm_migration_out_hh_12m",
        "perm_migration_in_hh_12m",
        "livestock_migration_any",
        "livestock_migration_changed_vs_historical",
        "livestock_migration_months_count",
    ] + matching(df, prefixes=["livestock_migration_month_"])

    pastoral = [
        "has_livestock_current",
        "livestock_total_current",
        "livestock_total_12m_ago",
        "livestock_total_2y_ago",
        "livestock_sales_value",
        "log1p_livestock_total_current",
        "log1p_livestock_sales_value",
        "livestock_change_12m",
        "livestock_decline_12m",
        "livestock_tlu_default",
        "livestock_total_per_ae",
        "livestock_total_per_capita",
        "livestock_tlu_per_ae_default",
        "livestock_tlu_per_capita_default",
        "livestock_indigenous_current",
        "livestock_exotic_current",
        "livestock_cattle_current",
        "livestock_cattle_share",
        "livestock_small_ruminants_current",
        "livestock_small_ruminants_share",
        "livestock_goat_current",
        "livestock_sheep_current",
        "livestock_poultry_current",
        "pastoral_livelihood_proxy",
        "pastoral_landscape_proxy",
    ] + matching(df, prefixes=["livestock_current_code_", "livestock_milk_", "lf_sec_06_", "lf_sec_07_", "lf_sec_08_"])

    # Dataset A: broad backbone, waves 1-4.
    core_cols = existing(df, id_geo + hh_controls + welfare + shocks_time + geo_controls + ["pastoral_livelihood_proxy", "pastoral_landscape_proxy"])
    core = df.loc[df["wave"].isin([1, 2, 3, 4]), core_cols].copy()

    # Dataset B: land-event exposure, waves 2-4.
    land_cols = existing(df, id_geo + hh_controls + welfare + shocks_time + geo_controls + land_events + land_composition + ["pastoral_livelihood_proxy", "pastoral_landscape_proxy"])
    land = df.loc[df["wave"].isin([2, 3, 4]), land_cols].copy()

    # Dataset C: pastoral mechanisms, waves 3-4.
    pastoral_cols = existing(
        df,
        id_geo + hh_controls + welfare + shocks_time + geo_controls + land_events + land_composition + community_migration + pastoral,
    )
    pastoral_df = df.loc[df["wave"].isin([3, 4]), pastoral_cols].copy()

    return {
        "analysis_core_1_4": core,
        "analysis_land_2_4": land,
        "analysis_pastoral_3_4": pastoral_df,
    }

def diagnostics(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in subsets.items():
        waves = sorted(pd.to_numeric(d["wave"], errors="coerce").dropna().unique().tolist()) if "wave" in d else []
        rows.append({
            "dataset": name,
            "rows": len(d),
            "columns": d.shape[1],
            "waves": ",".join(str(int(w)) for w in waves),
            "unique_hhid": d["hhid"].nunique() if "hhid" in d else np.nan,
            "unique_clusters": d["clusterid"].nunique() if "clusterid" in d else np.nan,
        })

        for c in d.columns:
            rows.append({
                "dataset": name,
                "variable": c,
                "rows": len(d),
                "columns": d.shape[1],
                "non_missing": d[c].notna().sum(),
                "share_non_missing": d[c].notna().mean(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create lean analysis subsets from the giant NPS household-wave file.")
    parser.add_argument(
        "--input",
        default="data/refined_multiwave/nps_analysis_household_wave.csv",
        help="Path to nps_analysis_household_wave.csv",
    )
    parser.add_argument(
        "--output",
        default="output/analysis_subsets",
        help="Output folder for lean datasets",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="diagnostics",
        help="Folder for diagnostics; kept separate from regression outputs.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_path}")
    df = pd.read_csv(
        in_path,
        low_memory=False,
        dtype={
            "hhid": "string",
            "hhid_wave": "string",
            "original_hhid": "string",
            "clusterid": "string",
            "community_id": "string",
            "community_id_x": "string",
            "community_id_y": "string",
        },
    )

    if "wave" not in df.columns:
        raise ValueError("Input file has no 'wave' column.")

    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype("Int64")

    print("Adding constructed variables...")
    df = add_constructed_variables(df)

    print("Building subsets...")
    subsets = build_subsets(df)

    for name, subset in subsets.items():
        out_path = out_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        print(f"  - {name}: {subset.shape[0]:,} rows x {subset.shape[1]:,} cols -> {out_path}")

    diag = diagnostics(subsets)
    diag_dir = Path(args.diagnostics_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / "analysis_subset_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    print(f"  - diagnostics -> {diag_path}")

    print("\nDone.")
    print("Recommended next step: merge ward/treatment/NDVI variables into these lean files, not into the giant DB.")


if __name__ == "__main__":
    main()
