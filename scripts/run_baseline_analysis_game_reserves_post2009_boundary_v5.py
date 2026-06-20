#!/usr/bin/env python3
"""
run_baseline_analysis_game_reserves_post2009_boundary_v5.py

Game-reserve baseline regressions with compact, readable output management.

Design in brief
---------------
The script estimates repeated-cross-section spatial exposure models for NPS
waves 1-4. The main estimand is:

  Among households in a credible local risk set around game reserves, did
  post-2009 reserve proximity differentially affect households located in
  Maasai-associated areas?

The main sample is households within 75 km of a game-reserve boundary.
The main heterogeneity proxy is the expanded 83-row Maasai locality dataset
constructed upstream; livestock/pastoral characteristics are mechanisms and
robustness descriptors, not Maasai identifiers.

Control convention
------------------
The preferred/main specification intentionally excludes head_age, because its
coverage is materially lower than the other demographic controls and it can drop
a large share of observations.  head_age is reported as robustness only.

Main controls:
  hh_size, share_female, n_children_u15, n_adults_15plus

Robustness controls:
  + head_age
  + n_shocks_occurred
  + head_age + n_shocks_occurred
  + survey weights / geospatial / market-access controls

Output layout
-------------
The output directory is no longer crowded with every CSV in one place:

  output/regressions/baseline_game_reserves/
    dashboard_main_results.csv        # first file to read after every run
    dashboard_main_results.md         # human-readable summary
    output_manifest.csv               # where every table was written
    game_reserve_post2009_boundary_notes.txt
    main/
      table_0_support_game_reserve.csv
      table_1_main_welfare_game_reserve.csv
      table_3_maasai_expanded_heterogeneity.csv
      table_5_controls_robustness.csv
    appendix/
      table_2_boundary_welfare_game_reserve.csv
      table_3b_maasai_expanded50_heterogeneity_robustness.csv
      table_3c_maasai_expanded_leave_one_district_out.csv
      table_4_mechanisms_game_reserve.csv
      table_4b_livestock_narrative.csv
      table_6_weighted_welfare.csv
      table_7_geospatial_market_controls.csv
      table_8_land_event_mechanisms.csv
      table_8b_sparse_land_event_appendix.csv

Typical run:
python scripts/run_baseline_analysis_game_reserves_post2009_boundary_v5.py \
  --spatial-dir output/spatial_restricted \
  --output-dir output/regressions/baseline_game_reserves
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import add_bh_q_values, coerce_numeric_columns, existing, find_file, parse_int_list

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


CORE_CANDIDATES = ["analysis_core_1_4_spatial.csv", "analysis_core_1_4_spatial_restricted.csv"]
LAND_CANDIDATES = ["analysis_land_2_4_spatial.csv", "analysis_land_2_4_spatial_restricted.csv"]
PASTORAL_CANDIDATES = ["analysis_pastoral_3_4_spatial.csv", "analysis_pastoral_3_4_spatial_restricted.csv"]

GAME_CUTOFFS = [25, 50]
SUPPORT_CUTOFFS = [10, 25, 50]
BOUNDARY_BANDS_KM = [50, 75, 100]

WELFARE_OUTCOMES = ["log_cons_pc", "log_cons_ae", "cons_food_share_real"]

MECHANISM_OUTCOMES = [
    "log1p_livestock_total_current",
    "livestock_change_12m",        # signed herd change: negative = net decline
    "livestock_decline_12m",
    "log1p_livestock_sales_value",
    "livestock_migration_any",
    "cons_food_share_real",
]

HERD_COMPOSITION_OUTCOMES = [
    "livestock_tlu_default",
    "livestock_tlu_per_ae_default",
    "livestock_total_per_ae",
    "livestock_cattle_current",
    "livestock_cattle_share",
    "livestock_small_ruminants_current",
    "livestock_small_ruminants_share",
    "livestock_indigenous_current",
    "livestock_exotic_current",
]

LAND_MECHANISM_OUTCOMES = [
    "any_land_event",
    "event_reserve_land_declaration",
    "event_outside_investor_appropriation",
    "event_public_use_allocation",
    "hh_affected_reserve_land_declaration",
    "hh_affected_outside_investor_appropriation",
    "hh_affected_public_use_allocation",
    "compensation_reserve_land_declaration",
    "compensation_outside_investor_appropriation",
    "compensation_public_use_allocation",
    "grazing_land_pct_clean",
    "cultivation_villagers_pct_clean",
    "has_village_land_certificate",
]

# Main land-mechanism table: broad land events and land-use/tenure outcomes only.
# Household-affected and compensation variables are usually sparse in the NPS and
# are pushed to an appendix table so they do not drive the core narrative.
LAND_MECHANISM_MAIN_OUTCOMES = [
    "any_land_event",
    "event_reserve_land_declaration",
    "event_outside_investor_appropriation",
    "event_public_use_allocation",
    "grazing_land_pct_clean",
    "cultivation_villagers_pct_clean",
    "has_village_land_certificate",
]

LAND_MECHANISM_SPARSE_OUTCOMES = [
    "hh_affected_reserve_land_declaration",
    "hh_affected_outside_investor_appropriation",
    "hh_affected_public_use_allocation",
    "compensation_reserve_land_declaration",
    "compensation_outside_investor_appropriation",
    "compensation_public_use_allocation",
]

# Ordered sequence for the narrative livestock table (table_4b).
# Each entry: (narrative_step, label, outcome_variable)
LIVESTOCK_NARRATIVE = [
    (1, "baseline_herd_size",    "log1p_livestock_total_current"),
    (2, "net_herd_change_12m",   "livestock_change_12m"),
    (3, "prob_herd_decline",     "livestock_decline_12m"),
    (4, "livestock_sales_value", "log1p_livestock_sales_value"),
    (5, "herd_mobility",         "livestock_migration_any"),
    (6, "tlu_intensity",         "livestock_tlu_per_ae_default"),
    (7, "cattle_share",          "livestock_cattle_share"),
]

# Main demographic controls: pre-determined/slow-moving and high coverage.
# head_age is intentionally NOT in the main specification: diagnostics show it
# can drop a large share of observations, especially outside the core welfare
# sample.  It is reported in table_5 as robustness only.
CONTROLS_MAIN = [
    "hh_size",
    "share_female",
    "n_children_u15",
    "n_adults_15plus",
]
CONTROLS_WITH_HEAD_AGE = CONTROLS_MAIN + ["head_age"]
CONTROLS_WITH_SHOCKS = CONTROLS_MAIN + ["n_shocks_occurred"]
CONTROLS_WITH_HEAD_AGE_SHOCKS = CONTROLS_MAIN + ["head_age", "n_shocks_occurred"]
CONTROL_COLUMNS = list(dict.fromkeys(CONTROLS_MAIN + ["head_age", "n_shocks_occurred"]))

# Default controls used in fit_model unless overridden.
CONTROLS = CONTROLS_MAIN

# Added robustness/control blocks. These are selected dynamically from geo_* variables
# that build_db_v5 extracts from NPS geovariable files.
LANDSCAPE_CONTROLS = ["grazing_land_pct_clean"]
CLIMATE_ENV_PATTERNS = ["geo_clim", "geo_hh_clim", "geo_hh_envi", "geo_land", "geo_soil", "geo_hh_soil_con", "geo_crops", "geo_srtm", "geo_twi", "geo_afmnslp"]
MARKET_ACCESS_PATTERNS = ["geo_dist", "geo_hh_geo"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--post", default="post_2009", choices=["post_2009", "post_2013"])
    p.add_argument("--cluster", default="ward2022_name")
    p.add_argument("--weight-col", default="weight", help="Survey weight column for weighted robustness tables.")
    p.add_argument("--max-extra-controls", default=8, type=int, help="Max geospatial/environment/market controls selected by non-missing coverage.")
    p.add_argument("--min-main-land-n", default=500, type=int, help="Minimum observations for rows retained in the main land-mechanism table.")
    p.add_argument("--min-main-land-clusters", default=50, type=int, help="Minimum clusters for rows retained in the main land-mechanism table.")
    p.add_argument("--skip-leave-one-district-out", action="store_true", help="Skip Maasai-expanded leave-one-district-out robustness table.")
    p.add_argument("--leave-one-district-all", action="store_true", help="Drop every district instead of only districts containing treated-post Maasai-expanded observations.")
    p.add_argument("--fe-level", default="district2022_name",
                   choices=["district2022_name", "region2022_name", "ward2022_name", "none"])
    p.add_argument("--northern-regions", default="Arusha,Manyara")
    p.add_argument("--risk-km", default=100.0, type=float, help="Backward-compatible alias for --maasai-risk-km.")
    p.add_argument("--maasai-risk-km", default=None, type=float, help="Km radius around expanded Maasai geography for the boundary+Maasai robustness sample. Defaults to --risk-km.")
    p.add_argument("--main-boundary-km", default=75, type=int, help="Main local risk-set sample: households within this many km of a game-reserve boundary.")
    p.add_argument("--boundary-bands", default="50,75,100",
                   help="Comma-separated km radii for local boundary-band samples. Should include --main-boundary-km.")
    args = p.parse_args()
    if args.maasai_risk_km is None:
        args.maasai_risk_km = args.risk_km
    return args






def read_csv_auto(directory: Path, candidates: List[str]) -> Tuple[pd.DataFrame, str]:
    path = find_file(directory, candidates, required=True)
    return pd.read_csv(path, low_memory=False), path.name




def matching_patterns(df: pd.DataFrame, patterns: Iterable[str]) -> List[str]:
    patterns = [p.lower() for p in patterns]
    return [c for c in df.columns if any(c.lower().startswith(p) for p in patterns)]


def select_controls_by_patterns(df: pd.DataFrame, patterns: Iterable[str], max_controls: int = 8) -> List[str]:
    """Select controls with the highest non-missing coverage to avoid overfitting."""
    candidates = matching_patterns(df, patterns)
    scored = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 50 or s.nunique(dropna=True) <= 1:
            continue
        scored.append((int(s.notna().sum()), c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:max_controls]]




def reconstruct_post(df: pd.DataFrame, post_name: str) -> pd.DataFrame:
    df = df.copy()
    wave = pd.to_numeric(df.get("wave"), errors="coerce")
    year = pd.to_numeric(df.get("survey_year"), errors="coerce")
    if post_name == "post_2009":
        df[post_name] = ((wave >= 2) | (year >= 2010)).astype(float)
    elif post_name == "post_2013":
        df[post_name] = ((wave >= 4) | (year >= 2014)).astype(float)
    else:
        raise ValueError(post_name)
    return df


def prepare(df: pd.DataFrame, post_name: str) -> pd.DataFrame:
    df = df.copy()
    df = reconstruct_post(df, post_name)

    numeric = set(
        ["wave", "survey_year", post_name,
         "distance_to_game_reserve_km",
         "distance_to_game_reserve_boundary_km",
         "treated_area", "treat_10km", "treat_25km", "treat_50km",
         "distance_to_maasai_reference_km", "distance_to_maasai_expanded_km",
         "maasai_expanded_10km", "maasai_expanded_25km", "maasai_expanded_50km", "maasai_expanded_100km",
         "maasai_expanded_area_proxy", "maasai_expanded_area_25km", "maasai_expanded_area_50km", "maasai_expanded_area_100km",
         "maasai_pastoral_area_proxy", "maasai_pastoral_area_proxy_25km", "maasai_pastoral_area_proxy_50km",
         "maasai_pastoral_composite_proxy", "maasai_pastoral_composite_25km", "maasai_pastoral_composite_50km",
         "pastoral_livelihood_proxy", "pastoral_landscape_proxy",
         "weight"]
        + [f"within_{k}km_game_reserve" for k in SUPPORT_CUTOFFS]
        + WELFARE_OUTCOMES
        + MECHANISM_OUTCOMES
        + HERD_COMPOSITION_OUTCOMES
        + LAND_MECHANISM_OUTCOMES
        + CONTROL_COLUMNS
        + matching_patterns(df, CLIMATE_ENV_PATTERNS + MARKET_ACCESS_PATTERNS)
    )
    df = coerce_numeric_columns(df, numeric)

    if "distance_to_game_reserve_km" in df.columns:
        d = pd.to_numeric(df["distance_to_game_reserve_km"], errors="coerce")
        for k in SUPPORT_CUTOFFS:
            col = f"within_{k}km_game_reserve"
            if col not in df.columns:
                df[col] = (d <= k).astype(float)
                df.loc[d.isna(), col] = np.nan
        df["neg_distance_to_game_reserve_10km"] = -d / 10.0

        # Boundary fallback: for observations outside a reserve this equals polygon distance.
        # For inside observations, upstream true boundary distance is better if available.
        if "distance_to_game_reserve_boundary_km" not in df.columns:
            df["distance_to_game_reserve_boundary_km"] = d.copy()

    if "distance_to_game_reserve_boundary_km" in df.columns:
        bd = pd.to_numeric(df["distance_to_game_reserve_boundary_km"], errors="coerce").clip(lower=0)
        df["neg_distance_to_game_reserve_boundary_10km"] = -bd / 10.0
        for band in BOUNDARY_BANDS_KM:
            df[f"boundary_band_{band}km"] = (bd <= band).astype(float)
            df.loc[bd.isna(), f"boundary_band_{band}km"] = np.nan

    # Expanded Maasai geography aliases. In v5 these are the treatment
    # identifiers. Livestock/pastoral characteristics are NOT used to define
    # Maasai status. Legacy columns are kept only for compatibility.
    if "distance_to_maasai_expanded_km" not in df.columns and "distance_to_maasai_reference_km" in df.columns:
        df["distance_to_maasai_expanded_km"] = pd.to_numeric(df["distance_to_maasai_reference_km"], errors="coerce")
    if "distance_to_maasai_reference_km" not in df.columns and "distance_to_maasai_expanded_km" in df.columns:
        df["distance_to_maasai_reference_km"] = pd.to_numeric(df["distance_to_maasai_expanded_km"], errors="coerce")

    d_maasai = pd.to_numeric(df.get("distance_to_maasai_expanded_km"), errors="coerce") if "distance_to_maasai_expanded_km" in df.columns else None
    if d_maasai is not None:
        for km in [10, 25, 50, 100]:
            col = f"maasai_expanded_{km}km"
            if col not in df.columns:
                df[col] = (d_maasai <= km).astype(float)
                df.loc[d_maasai.isna(), col] = np.nan

    alias_pairs = {
        "treated_area": "maasai_expanded_25km",
        "treat_10km": "maasai_expanded_10km",
        "treat_25km": "maasai_expanded_25km",
        "treat_50km": "maasai_expanded_50km",
        "pastoral_reference_25km": "maasai_expanded_25km",
        "pastoral_reference_50km": "maasai_expanded_50km",
        "maasai_expanded_area_proxy": "maasai_expanded_25km",
        "maasai_expanded_area_25km": "maasai_expanded_25km",
        "maasai_expanded_area_50km": "maasai_expanded_50km",
        "maasai_expanded_area_100km": "maasai_expanded_100km",
        "maasai_pastoral_area_proxy": "maasai_expanded_25km",
        "maasai_pastoral_area_proxy_25km": "maasai_expanded_25km",
        "maasai_pastoral_area_proxy_50km": "maasai_expanded_50km",
        "maasai_pastoral_composite_proxy": "maasai_expanded_25km",
        "maasai_pastoral_composite_25km": "maasai_expanded_25km",
        "maasai_pastoral_composite_50km": "maasai_expanded_50km",
    }
    for alias, src in alias_pairs.items():
        if src in df.columns:
            df[alias] = pd.to_numeric(df[src], errors="coerce")

    return df


def maasai_distance_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["distance_to_maasai_expanded_km", "distance_to_maasai_reference_km"]:
        if c in df.columns:
            return c
    return None


def make_base_samples(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    samples = {"full": df.copy()}

    regions = {x.strip().lower() for x in args.northern_regions.split(",") if x.strip()}
    if "region2022_name" in df.columns and regions:
        reg = df["region2022_name"].astype(str).str.strip().str.lower()
        samples["northern_regions"] = df.loc[reg.isin(regions)].copy()

    dcol = maasai_distance_column(df)
    if "region2022_name" in df.columns and dcol is not None and regions:
        reg = df["region2022_name"].astype(str).str.strip().str.lower()
        dist = pd.to_numeric(df[dcol], errors="coerce")
        samples[f"northern_or_within{int(args.maasai_risk_km)}km_maasai_expanded"] = df.loc[
            reg.isin(regions) | (dist <= args.maasai_risk_km)
        ].copy()

    return samples


def make_boundary_samples(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    samples = {}
    bands = parse_int_list(args.boundary_bands) or BOUNDARY_BANDS_KM
    if args.main_boundary_km not in bands:
        bands = [args.main_boundary_km] + bands
    if "distance_to_game_reserve_boundary_km" not in df.columns:
        return samples
    bd = pd.to_numeric(df["distance_to_game_reserve_boundary_km"], errors="coerce")
    for b in bands:
        samples[f"boundary{b}km"] = df.loc[bd <= b].copy()
    return samples


def make_model_samples(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Samples used in the main regression tables.

    Main sample first: boundary <= --main-boundary-km, default 75 km.
    Robustness samples: full, northern regions, boundary 50/100 km, and
    boundary main sample plus within --maasai-risk-km of expanded Maasai geography.
    """
    out: dict[str, pd.DataFrame] = {}
    base = make_base_samples(df, args)
    boundary = make_boundary_samples(df, args)

    main_name = f"boundary{int(args.main_boundary_km)}km"
    if main_name in boundary:
        out[main_name] = boundary[main_name]

    for name in ["full", "northern_regions"]:
        if name in base:
            out[name] = base[name]

    for b in [50, 100]:
        name = f"boundary{b}km"
        if name in boundary and name not in out:
            out[name] = boundary[name]

    dcol = maasai_distance_column(df)
    if main_name in boundary and dcol is not None:
        dist = pd.to_numeric(boundary[main_name][dcol], errors="coerce")
        out[f"{main_name}_maasai{int(args.maasai_risk_km)}km"] = boundary[main_name].loc[dist <= args.maasai_risk_km].copy()

    # Add any extra user-specified boundary bands after the standard set.
    for name, sample in boundary.items():
        if name not in out:
            out[name] = sample
    for name, sample in base.items():
        if name not in out:
            out[name] = sample
    return out


def parse_boundary_band_from_sample(sample_name: str) -> Optional[int]:
    """Return numeric X from sample labels like boundary50km."""
    import re
    m = re.search(r"boundary(\d+)km", str(sample_name))
    return int(m.group(1)) if m else None


def restrict_to_model_sample(
    df: pd.DataFrame,
    outcome: str,
    terms: List[str],
    controls: List[str],
    args: argparse.Namespace,
    weighted: bool = False,
) -> pd.DataFrame:
    """Return exactly the observations that a given model specification can use.

    This is used for same-sample baseline rows in the geospatial/market robustness
    table. Without this, adding market-access controls can change both the
    coefficient and the estimation sample, making the comparison hard to read.
    """
    needed = [outcome] + list(terms) + existing(df, controls) + ["wave"]
    if args.fe_level != "none" and args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    if weighted and args.weight_col in df.columns:
        needed.append(args.weight_col)
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))
    d = df.dropna(subset=needed).copy()
    if weighted and args.weight_col in d.columns:
        d[args.weight_col] = pd.to_numeric(d[args.weight_col], errors="coerce")
        d = d.loc[d[args.weight_col].notna() & (d[args.weight_col] > 0)].copy()
    return d


def binary_cell_counts(df: pd.DataFrame, var_map: dict[str, str]) -> dict:
    """Cell-count diagnostics for binary heterogeneity designs.

    var_map example: {"game": within_25km_game_reserve, "proxy": maasai_proxy,
    "post": post_2009}. Returns margins and full 0/1 combination cell counts.
    """
    out: dict[str, float | int | str] = {}
    present = {k: v for k, v in var_map.items() if v in df.columns}
    if not present:
        return out
    tmp = pd.DataFrame(index=df.index)
    for short, col in present.items():
        tmp[short] = pd.to_numeric(df[col], errors="coerce")
    tmp = tmp.dropna()
    out["cell_count_vars"] = ",".join(f"{k}={v}" for k, v in present.items())
    out["cell_n_nonmissing"] = int(len(tmp))
    for short in present:
        vals = tmp[short]
        out[f"cell_sum_{short}"] = int((vals > 0).sum())
        out[f"cell_mean_{short}"] = float((vals > 0).mean()) if len(vals) else np.nan
    keys = list(present.keys())
    if set(["game", "proxy", "post"]).issubset(keys):
        b = tmp[["game", "proxy", "post"]].copy()
        for c in ["game", "proxy", "post"]:
            b[c] = (b[c] > 0).astype(int)
        for g in [0, 1]:
            for pxy in [0, 1]:
                for post in [0, 1]:
                    out[f"cell_n_g{g}_p{pxy}_post{post}"] = int(((b["game"] == g) & (b["proxy"] == pxy) & (b["post"] == post)).sum())
        out["cell_n_treated_proxy_post"] = out.get("cell_n_g1_p1_post1", 0)
        out["cell_n_treated_proxy_total"] = int(((b["game"] == 1) & (b["proxy"] == 1)).sum())
    return out


def rhs_with_fe(df: pd.DataFrame, terms: List[str], controls: List[str], fe_level: str) -> List[str]:
    rhs = list(terms) + existing(df, controls)
    if "wave" in df.columns:
        rhs.append("C(wave)")
    if fe_level != "none" and fe_level in df.columns:
        rhs.append(f"C({fe_level})")
    return rhs


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    terms: List[str],
    report: str,
    sample: str,
    model: str,
    args: argparse.Namespace,
    meta: Optional[dict] = None,
    controls_override: Optional[List[str]] = None,
    weighted: bool = False,
) -> dict:
    meta = meta or {}
    # Use override if provided, otherwise fall back to global CONTROLS.
    controls_to_use = controls_override if controls_override is not None else CONTROLS
    controls = existing(df, controls_to_use)
    missing = [c for c in [outcome] + terms if c not in df.columns]
    out = {
        "sample": sample,
        "model": model,
        "outcome": outcome,
        "reported_term": report,
        "controls": ",".join(controls) if controls else "none",
        "head_age_in_controls": "head_age" in controls,
        "head_age_available_input": int(pd.to_numeric(df["head_age"], errors="coerce").notna().sum()) if "head_age" in df.columns else np.nan,
        "fe_level": args.fe_level,
        "weighted": bool(weighted),
        "weight_col": args.weight_col if weighted else "none",
        "n_input_rows": int(len(df)),
        **meta,
    }
    if missing:
        out["status"] = "missing_columns:" + ",".join(missing)
        return out

    needed = [outcome] + terms + controls + ["wave"]
    if args.fe_level != "none" and args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    if weighted and args.weight_col in df.columns:
        needed.append(args.weight_col)
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    d = df.dropna(subset=needed).copy()
    out["n_after_missing_filter"] = int(len(d))
    out["n_dropped_missing_model_vars"] = int(len(df) - len(d))
    out["share_dropped_missing_model_vars"] = float((len(df) - len(d)) / len(df)) if len(df) else np.nan
    if "head_age" in df.columns:
        h = pd.to_numeric(df["head_age"], errors="coerce")
        out["head_age_missing_input"] = int(h.isna().sum())
        out["head_age_missing_share_input"] = float(h.isna().mean()) if len(h) else np.nan
    if weighted:
        if args.weight_col not in d.columns:
            out["status"] = f"missing_weight_col:{args.weight_col}"
            return out
        d[args.weight_col] = pd.to_numeric(d[args.weight_col], errors="coerce")
        d = d.loc[d[args.weight_col].notna() & (d[args.weight_col] > 0)].copy()
    out["n"] = int(len(d))
    out["n_clusters"] = int(d[args.cluster].nunique(dropna=True)) if args.cluster in d.columns else np.nan
    if report in d.columns:
        r = pd.to_numeric(d[report], errors="coerce")
        out["reported_term_n_nonmissing"] = int(r.notna().sum())
        out["reported_term_n_positive"] = int((r.fillna(0) > 0).sum())
        out["reported_term_mean"] = float(r.mean(skipna=True)) if r.notna().any() else np.nan

    if len(d) < 50:
        out["status"] = "too_few_obs"
        return out
    if report in d.columns and d[report].nunique(dropna=True) <= 1:
        out["status"] = "reported_term_no_variation"
        out["term_unique_values"] = int(d[report].nunique(dropna=True))
        return out

    rhs = rhs_with_fe(d, terms, controls, args.fe_level)
    formula = f"{outcome} ~ " + " + ".join(rhs)
    out["formula"] = formula

    try:
        if weighted:
            mod = smf.wls(formula, data=d, weights=d[args.weight_col])
        else:
            mod = smf.ols(formula, data=d)
        if args.cluster in d.columns and d[args.cluster].nunique(dropna=True) >= 2:
            fit = mod.fit(cov_type="cluster", cov_kwds={"groups": d[args.cluster]})
            out["se_type"] = f"cluster:{args.cluster}"
        else:
            fit = mod.fit(cov_type="HC1")
            out["se_type"] = "HC1"
    except Exception as e:
        out["status"] = "fit_error"
        out["error"] = repr(e)
        return out

    coef = fit.params.get(report, np.nan)
    se = fit.bse.get(report, np.nan)
    p = fit.pvalues.get(report, np.nan)

    out.update({
        "status": "ok",
        "coef": float(coef) if pd.notna(coef) else np.nan,
        "se": float(se) if pd.notna(se) else np.nan,
        "p_value": float(p) if pd.notna(p) else np.nan,
        "ci95_low": float(coef - 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan,
        "ci95_high": float(coef + 1.96 * se) if pd.notna(coef) and pd.notna(se) else np.nan,
        "r2": float(fit.rsquared),
    })

    for c in terms:
        if c in d.columns:
            x = pd.to_numeric(d[c], errors="coerce")
            if x.dropna().isin([0, 1]).all():
                out[f"sum_{c}"] = float(x.sum())
                out[f"mean_{c}"] = float(x.mean())
    return out


def support_table(samples: dict[str, pd.DataFrame], post: str, include_boundary: bool = False) -> pd.DataFrame:
    rows = []
    vars_ = (
        [f"within_{k}km_game_reserve" for k in SUPPORT_CUTOFFS]
        + ["distance_to_game_reserve_km",
           "distance_to_game_reserve_boundary_km",
           "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km",
           "maasai_expanded_25km", "maasai_expanded_50km", "maasai_expanded_100km",
           "maasai_expanded_area_proxy", "maasai_expanded_area_50km", "maasai_expanded_area_100km",
           "pastoral_reference_25km", "pastoral_reference_50km",
           "pastoral_livelihood_proxy", "pastoral_landscape_proxy",
           "maasai_pastoral_area_proxy", "maasai_pastoral_composite_proxy",
           "maasai_pastoral_composite_25km", "maasai_pastoral_composite_50km"]
    )
    if include_boundary:
        vars_ += [f"boundary_band_{b}km" for b in BOUNDARY_BANDS_KM]

    for sample, df in samples.items():
        d = df.copy()
        for gk in SUPPORT_CUTOFFS:
            g = f"within_{gk}km_game_reserve"
            if g not in d.columns:
                continue
            for pk in [25, 50]:
                p = f"maasai_expanded_{pk}km"
                if p in d.columns:
                    d[f"game{gk}_x_maasai_expanded{pk}"] = d[g] * d[p]
                    d[f"game{gk}_x_maasai_expanded{pk}_x_{post}"] = d[g] * d[p] * d[post]
                    vars_.extend([f"game{gk}_x_maasai_expanded{pk}", f"game{gk}_x_maasai_expanded{pk}_x_{post}"])

        for var in list(dict.fromkeys(vars_)):
            if var not in d.columns:
                continue
            for wave, gw in d.groupby("wave", dropna=False):
                s = pd.to_numeric(gw[var], errors="coerce")
                rows.append({
                    "sample": sample,
                    "variable": var,
                    "wave": wave,
                    "n_rows": len(gw),
                    "n_nonmissing": int(s.notna().sum()),
                    "sum": float(s.sum(skipna=True)) if s.notna().any() else np.nan,
                    "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                    "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                    "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
                })
    return pd.DataFrame(rows)


def main_welfare(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for sample, df in samples.items():
        for k in GAME_CUTOFFS:
            treat = f"within_{k}km_game_reserve"
            if treat not in df.columns:
                continue
            d = df.copy()
            inter = f"{treat}_X_{args.post}"
            d[inter] = d[treat] * d[args.post]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, [treat, inter], inter, sample,
                    "binary_game_reserve", args,
                    {"game_cutoff_km": k},
                    controls_override=CONTROLS_MAIN,
                ))

        if "neg_distance_to_game_reserve_10km" in df.columns:
            d = df.copy()
            intensity = "neg_distance_to_game_reserve_10km"
            inter = f"{intensity}_X_{args.post}"
            d[inter] = d[intensity] * d[args.post]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, [intensity, inter], inter, sample,
                    "continuous_game_distance", args,
                    {"game_cutoff_km": "continuous"},
                    controls_override=CONTROLS_MAIN,
                ))
    return pd.DataFrame(rows)


def boundary_welfare(boundary_samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    skipped_rows = []
    for sample, df in boundary_samples.items():
        band_km = parse_boundary_band_from_sample(sample)
        # In local boundary bands, estimate standard 25/50km proximity effects.
        # Skip mechanically uninformative combinations, e.g. within_50km_game_reserve
        # inside the boundary50km sample, where almost all outside observations are
        # necessarily within 50km of the reserve boundary.
        for k in GAME_CUTOFFS:
            if band_km is not None and k >= band_km:
                skipped_rows.append({
                    "sample": sample,
                    "model": "boundary_band_binary_game_reserve",
                    "game_cutoff_km": k,
                    "status": "skipped_mechanical_cutoff_ge_boundary_band",
                    "reason": f"within_{k}km is mechanically uninformative inside boundary{band_km}km sample",
                })
                continue
            treat = f"within_{k}km_game_reserve"
            if treat not in df.columns:
                continue
            d = df.copy()
            inter = f"{treat}_X_{args.post}"
            d[inter] = d[treat] * d[args.post]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, [treat, inter], inter, sample,
                    "boundary_band_binary_game_reserve", args,
                    {"game_cutoff_km": k, "boundary_band_km": band_km},
                    controls_override=CONTROLS_MAIN,
                ))

        # Boundary-distance intensity within boundary-band sample.
        if "neg_distance_to_game_reserve_boundary_10km" in df.columns:
            d = df.copy()
            intensity = "neg_distance_to_game_reserve_boundary_10km"
            inter = f"{intensity}_X_{args.post}"
            d[inter] = d[intensity] * d[args.post]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, [intensity, inter], inter, sample,
                    "boundary_band_continuous_boundary_distance", args,
                    {"game_cutoff_km": "boundary_distance", "boundary_band_km": band_km},
                    controls_override=CONTROLS_MAIN,
                ))
    out = pd.DataFrame(rows)
    if skipped_rows:
        skipped = pd.DataFrame(skipped_rows)
        skipped.attrs["skipped_specs"] = skipped
    return out

def maasai_expanded50_heterogeneity(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Triple interaction using the 50 km expanded Maasai geography as robustness."""
    rows = []
    for sample, df in samples.items():
        maasai = "maasai_expanded_50km" if "maasai_expanded_50km" in df.columns else "pastoral_reference_50km"
        if maasai not in df.columns:
            continue
        for k in GAME_CUTOFFS:
            game = f"within_{k}km_game_reserve"
            if game not in df.columns:
                continue
            d = df.copy()
            gp = f"{game}_X_{maasai}"
            gpost = f"{game}_X_{args.post}"
            ppost = f"{maasai}_X_{args.post}"
            triple = f"{game}_X_{maasai}_X_{args.post}"
            d[gp] = d[game] * d[maasai]
            d[gpost] = d[game] * d[args.post]
            d[ppost] = d[maasai] * d[args.post]
            d[triple] = d[game] * d[maasai] * d[args.post]
            terms = [game, maasai, gp, gpost, ppost, triple]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, terms, triple, sample,
                    "triple_game_x_maasai_expanded50", args,
                    {"game_cutoff_km": k, "maasai_cutoff_km": 50, "maasai_proxy": maasai, **binary_cell_counts(d, {"game": game, "proxy": maasai, "post": args.post})},
                    controls_override=CONTROLS_MAIN,
                ))
    return pd.DataFrame(rows)


def mechanisms(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Standard mechanism table. Uses CONTROLS_MAIN + LANDSCAPE_CONTROLS where available."""
    rows = []
    for sample, df in samples.items():
        for k in GAME_CUTOFFS:
            treat = f"within_{k}km_game_reserve"
            if treat not in df.columns:
                continue
            d = df.copy()
            inter = f"{treat}_X_{args.post}"
            d[inter] = d[treat] * d[args.post]
            # Add landscape control where present in this dataset.
            controls = CONTROLS_MAIN + existing(df, LANDSCAPE_CONTROLS)
            for y in existing(d, MECHANISM_OUTCOMES + HERD_COMPOSITION_OUTCOMES):
                rows.append(fit_model(
                    d, y, [treat, inter], inter, sample,
                    "mechanism_binary_game_reserve", args,
                    {"game_cutoff_km": k},
                    controls_override=controls,
                ))
    return pd.DataFrame(rows)


def livestock_narrative(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """
    table_4b: Ordered livestock outcome sequence designed to tell a coherent story.

    Runs only within_25km_game_reserve × post_2009 on the 'full' and
    'northern_regions' samples. Each row gets a narrative_step and a
    step_label so the table can be read top-to-bottom as a causal chain:

      1  baseline_herd_size    Does near-reserve status correlate with larger herds?
      2  net_herd_change_12m   Are those herds shrinking more after 2009?
      3  prob_herd_decline     Is a binary decline more likely?
      4  livestock_sales_value Are they selling more (liquidation)?
      5  herd_mobility         Are they moving herds more (adaptation)?

    Interpretation note: if step 1 is positive (larger herds near reserves)
    but steps 2–3 are negative/positive (net decline / higher decline prob)
    and step 4 is positive (more sales), the coherent reading is:
    households near reserves hold relatively larger herds but are in accelerated
    liquidation/stress — not that they are better off. Include this framing in
    the paper's mechanism section.
    """
    NARRATIVE_SAMPLES = [f"boundary{int(args.main_boundary_km)}km", "full", "northern_regions"]
    treat = "within_25km_game_reserve"

    rows = []
    for sample_name in NARRATIVE_SAMPLES:
        df = samples.get(sample_name)
        if df is None:
            continue
        if treat not in df.columns:
            continue
        d = df.copy()
        inter = f"{treat}_X_{args.post}"
        d[inter] = d[treat] * d[args.post]
        controls = CONTROLS_MAIN + existing(df, LANDSCAPE_CONTROLS)

        for step, label, outcome in LIVESTOCK_NARRATIVE:
            if outcome not in d.columns:
                continue
            row = fit_model(
                d, outcome, [treat, inter], inter, sample_name,
                "livestock_narrative_25km", args,
                {"game_cutoff_km": 25, "narrative_step": step, "step_label": label},
                controls_override=controls,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def controls_robustness(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Key control-set robustness for the headline welfare specification.

    table_5 now explicitly treats head_age as robustness, not as part of the
    preferred specification.  This makes sample attrition visible instead of
    silently baking it into every main result.

    Main sample only, within_25km_game_reserve × post variable, welfare outcomes.
    """
    rows = []
    sample_name = f"boundary{int(args.main_boundary_km)}km"
    df = samples.get(sample_name)
    if df is None:
        return pd.DataFrame(rows)

    treat = "within_25km_game_reserve"
    if treat not in df.columns:
        return pd.DataFrame(rows)

    d = df.copy()
    inter = f"{treat}_X_{args.post}"
    d[inter] = d[treat] * d[args.post]

    robustness_specs = [
        ("main_no_head_age", CONTROLS_MAIN),
        ("plus_head_age", CONTROLS_WITH_HEAD_AGE),
        ("plus_shocks", CONTROLS_WITH_SHOCKS),
        ("plus_head_age_and_shocks", CONTROLS_WITH_HEAD_AGE_SHOCKS),
    ]

    for ctrl_label, ctrl_set in robustness_specs:
        for y in existing(d, WELFARE_OUTCOMES):
            row = fit_model(
                d, y, [treat, inter], inter, sample_name,
                "controls_robustness_25km", args,
                {"game_cutoff_km": 25, "control_specification": ctrl_label},
                controls_override=ctrl_set,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def weighted_welfare_robustness(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Survey-weighted version of the headline welfare specification."""
    rows = []
    sample_name = f"boundary{int(args.main_boundary_km)}km"
    df = samples.get(sample_name)
    if df is None:
        return pd.DataFrame(rows)
    treat = "within_25km_game_reserve"
    if treat not in df.columns:
        return pd.DataFrame(rows)
    d = df.copy()
    inter = f"{treat}_X_{args.post}"
    d[inter] = d[treat] * d[args.post]
    for y in existing(d, WELFARE_OUTCOMES):
        rows.append(fit_model(
            d, y, [treat, inter], inter, sample_name,
            "weighted_welfare_25km", args,
            {"game_cutoff_km": 25, "control_specification": "main_no_head_age"},
            controls_override=CONTROLS_MAIN,
            weighted=True,
        ))
    return pd.DataFrame(rows)


def geospatial_market_controls_robustness(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Compare headline welfare results with geospatial/environment and market controls.

    For each augmented specification, the table now includes a paired baseline
    estimated on exactly the same non-missing sample. This avoids interpreting a
    coefficient change that is actually due to sample attrition when market-access
    variables are only available for some waves/observations.
    """
    rows = []
    sample_name = f"boundary{int(args.main_boundary_km)}km"
    df = samples.get(sample_name)
    if df is None:
        return pd.DataFrame(rows)
    treat = "within_25km_game_reserve"
    if treat not in df.columns:
        return pd.DataFrame(rows)
    d = df.copy()
    inter = f"{treat}_X_{args.post}"
    d[inter] = d[treat] * d[args.post]
    terms = [treat, inter]

    climate_env = select_controls_by_patterns(d, CLIMATE_ENV_PATTERNS, max_controls=args.max_extra_controls)
    market = select_controls_by_patterns(d, MARKET_ACCESS_PATTERNS, max_controls=args.max_extra_controls)
    augmented_specs = [
        ("geo_env", CONTROLS_MAIN + climate_env, climate_env),
        ("market", CONTROLS_MAIN + market, market),
        ("geo_env_market", CONTROLS_MAIN + climate_env + market, climate_env + market),
    ]

    # Main boundary-sample baseline for reference.
    for y in existing(d, WELFARE_OUTCOMES):
        rows.append(fit_model(
            d, y, terms, inter, sample_name,
            "geospatial_market_controls_25km", args,
            {"game_cutoff_km": 25, "control_specification": "main_no_head_age_full_available_sample", "same_sample_pair_id": "none"},
            controls_override=CONTROLS_MAIN,
        ))

    for spec_label, aug_controls, added_controls in augmented_specs:
        pair_id = f"same_sample_{spec_label}"
        for y in existing(d, WELFARE_OUTCOMES):
            # Same-sample baseline: restrict to the augmented-spec sample, but estimate demo-only controls.
            d_same = restrict_to_model_sample(d, y, terms, aug_controls, args, weighted=False)
            rows.append(fit_model(
                d_same, y, terms, inter, sample_name,
                "geospatial_market_controls_25km", args,
                {
                    "game_cutoff_km": 25,
                    "control_specification": f"main_no_head_age_{pair_id}",
                    "same_sample_pair_id": pair_id,
                    "added_controls_for_pair": ",".join(added_controls) if added_controls else "none",
                },
                controls_override=CONTROLS_MAIN,
            ))
            rows.append(fit_model(
                d_same, y, terms, inter, sample_name,
                "geospatial_market_controls_25km", args,
                {
                    "game_cutoff_km": 25,
                    "control_specification": f"main_plus_{spec_label}",
                    "same_sample_pair_id": pair_id,
                    "added_controls_for_pair": ",".join(added_controls) if added_controls else "none",
                },
                controls_override=aug_controls,
            ))
    return pd.DataFrame(rows)

def land_event_mechanisms(
    samples: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    outcomes: Optional[List[str]] = None,
    table_role: str = "main",
) -> pd.DataFrame:
    """Land-event and tenure mechanisms using the land subset.

    Main table uses broad land-event / land-use / tenure variables. Sparse
    household-affected and compensation variables are written to an appendix table.
    """
    rows = []
    outcomes = outcomes or LAND_MECHANISM_MAIN_OUTCOMES
    for sample, df in samples.items():
        for k in GAME_CUTOFFS:
            treat = f"within_{k}km_game_reserve"
            if treat not in df.columns:
                continue
            d = df.copy()
            inter = f"{treat}_X_{args.post}"
            d[inter] = d[treat] * d[args.post]
            climate_env = select_controls_by_patterns(d, CLIMATE_ENV_PATTERNS, max_controls=min(4, args.max_extra_controls))
            market = select_controls_by_patterns(d, MARKET_ACCESS_PATTERNS, max_controls=min(4, args.max_extra_controls))
            controls = CONTROLS_MAIN + climate_env + market
            for y in existing(d, outcomes):
                row = fit_model(
                    d, y, [treat, inter], inter, sample,
                    "land_event_mechanism_game_reserve", args,
                    {"game_cutoff_km": k, "table_role": table_role},
                    controls_override=controls,
                )
                if table_role == "main":
                    n = row.get("n", 0) or 0
                    nc = row.get("n_clusters", 0) or 0
                    if row.get("status") == "ok" and (n < args.min_main_land_n or nc < args.min_main_land_clusters):
                        row["status"] = "filtered_sparse_main_land_mechanism"
                        row["sparse_filter_reason"] = f"n<{args.min_main_land_n} or n_clusters<{args.min_main_land_clusters}"
                rows.append(row)
    out = pd.DataFrame(rows)
    if table_role == "main" and not out.empty:
        # Keep filtered rows out of the main CSV; they can be inspected in the appendix logic if needed.
        out = out.loc[out["status"].ne("filtered_sparse_main_land_mechanism")].copy()
    return out

def maasai_expanded_heterogeneity(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Triple interaction using expanded 83-row Maasai geography, not pastoral characteristics."""
    rows = []
    for sample, df in samples.items():
        candidates = ["maasai_expanded_area_proxy", "maasai_expanded_25km", "treated_area", "maasai_pastoral_composite_proxy"]
        maasai = next((c for c in candidates if c in df.columns), None)
        if maasai is None:
            continue
        for k in GAME_CUTOFFS:
            game = f"within_{k}km_game_reserve"
            if game not in df.columns:
                continue
            d = df.copy()
            gp = f"{game}_X_{maasai}"
            gpost = f"{game}_X_{args.post}"
            ppost = f"{maasai}_X_{args.post}"
            triple = f"{game}_X_{maasai}_X_{args.post}"
            d[gp] = d[game] * d[maasai]
            d[gpost] = d[game] * d[args.post]
            d[ppost] = d[maasai] * d[args.post]
            d[triple] = d[game] * d[maasai] * d[args.post]
            terms = [game, maasai, gp, gpost, ppost, triple]
            for y in existing(d, WELFARE_OUTCOMES):
                rows.append(fit_model(
                    d, y, terms, triple, sample,
                    "triple_game_x_maasai_expanded", args,
                    {"game_cutoff_km": k, "maasai_proxy": maasai, "main_sample": sample == f"boundary{int(args.main_boundary_km)}km", **binary_cell_counts(d, {"game": game, "proxy": maasai, "post": args.post})},
                    controls_override=CONTROLS_MAIN,
                ))
    return pd.DataFrame(rows)


def maasai_expanded_leave_one_district_out(samples: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    """Leave-one-district-out robustness for the expanded Maasai heterogeneity result."""
    rows = []
    if args.skip_leave_one_district_out:
        return pd.DataFrame(rows)
    sample_name = f"boundary{int(args.main_boundary_km)}km"
    df = samples.get(sample_name)
    if df is None:
        df = samples.get("full")
        sample_name = "full"
    if df is None or "district2022_name" not in df.columns:
        return pd.DataFrame(rows)
    candidates = ["maasai_expanded_area_proxy", "maasai_expanded_25km", "treated_area", "maasai_pastoral_composite_proxy"]
    maasai = next((c for c in candidates if c in df.columns), None)
    if maasai is None:
        return pd.DataFrame(rows)
    game = "within_25km_game_reserve"
    if game not in df.columns:
        return pd.DataFrame(rows)

    d0 = df.copy()
    gp = f"{game}_X_{maasai}"
    gpost = f"{game}_X_{args.post}"
    ppost = f"{maasai}_X_{args.post}"
    triple = f"{game}_X_{maasai}_X_{args.post}"
    d0[gp] = d0[game] * d0[maasai]
    d0[gpost] = d0[game] * d0[args.post]
    d0[ppost] = d0[maasai] * d0[args.post]
    d0[triple] = d0[game] * d0[maasai] * d0[args.post]
    terms = [game, maasai, gp, gpost, ppost, triple]

    districts = sorted([x for x in d0["district2022_name"].dropna().unique()])
    if not args.leave_one_district_all:
        keep = []
        for dist in districts:
            sub = d0.loc[d0["district2022_name"].eq(dist)]
            if pd.to_numeric(sub[triple], errors="coerce").fillna(0).gt(0).any():
                keep.append(dist)
        districts = keep

    for dropped in districts:
        d = d0.loc[~d0["district2022_name"].eq(dropped)].copy()
        dropped_n = int(d0["district2022_name"].eq(dropped).sum())
        dropped_triple_n = int(pd.to_numeric(d0.loc[d0["district2022_name"].eq(dropped), triple], errors="coerce").fillna(0).gt(0).sum())
        for y in existing(d, WELFARE_OUTCOMES):
            rows.append(fit_model(
                d, y, terms, triple, sample_name,
                "leave_one_district_maasai_expanded_25km", args,
                {
                    "game_cutoff_km": 25,
                    "maasai_proxy": maasai,
                    "dropped_district2022_name": dropped,
                    "dropped_n": dropped_n,
                    "dropped_cell_n_treated_proxy_post": dropped_triple_n,
                    **binary_cell_counts(d, {"game": game, "proxy": maasai, "post": args.post}),
                },
                controls_override=CONTROLS_MAIN,
            ))
    return pd.DataFrame(rows)


MAIN_TABLE_FILES = {
    "table_0_support_game_reserve.csv",
    "table_1_main_welfare_game_reserve.csv",
    "table_3_maasai_expanded_heterogeneity.csv",
    "table_5_controls_robustness.csv",
}

APPENDIX_TABLE_FILES = {
    "table_2_boundary_welfare_game_reserve.csv",
    "table_3b_maasai_expanded50_heterogeneity_robustness.csv",
    "table_3c_maasai_expanded_leave_one_district_out.csv",
    "table_4_mechanisms_game_reserve.csv",
    "table_4b_livestock_narrative.csv",
    "table_6_weighted_welfare.csv",
    "table_7_geospatial_market_controls.csv",
    "table_8_land_event_mechanisms.csv",
    "table_8b_sparse_land_event_appendix.csv",
}


def output_subdir_for_table(fname: str) -> str:
    """Return output subfolder for a result table."""
    if fname in MAIN_TABLE_FILES:
        return "main"
    if fname in APPENDIX_TABLE_FILES:
        return "appendix"
    return "."


def select_rows(df: pd.DataFrame, **equals) -> pd.Series:
    """Boolean mask helper that tolerates missing columns and numeric typing."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)
    for col, value in equals.items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[col]
        try:
            value_num = float(value)
            series_num = pd.to_numeric(series, errors="coerce")
            if series_num.notna().any():
                mask &= series_num.eq(value_num)
                continue
        except Exception:
            pass
        mask &= series.astype(str).eq(str(value))
    return mask


def interpretation_flag(row: pd.Series) -> str:
    """Compact interpretation label for dashboard rows."""
    status = str(row.get("status", ""))
    if status and status != "ok":
        return f"check_status:{status}"
    q = pd.to_numeric(pd.Series([row.get("q_value_bh")]), errors="coerce").iloc[0]
    p = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
    if pd.notna(q) and q < 0.10:
        return "passes_bh_10pct"
    if pd.notna(p) and p < 0.10:
        return "nominal_only"
    return "weak_or_null"


def build_dashboard(tables: dict[str, pd.DataFrame], main_sample: str = "boundary75km") -> pd.DataFrame:
    """Build a one-screen dashboard from the many regression tables.

    The point is not to replace appendix tables.  The point is to make every run
    readable: first inspect this file, then drill down only if something moved.
    """
    pieces: list[pd.DataFrame] = []

    def add_piece(fname: str, df: pd.DataFrame, mask: pd.Series, section: str, priority: int) -> None:
        if df is None or df.empty or mask.empty:
            return
        part = df.loc[mask].copy()
        if part.empty:
            return
        part.insert(0, "priority", priority)
        part.insert(1, "section", section)
        part.insert(2, "source_table", fname)
        pieces.append(part)

    fname = "table_1_main_welfare_game_reserve.csv"
    df = tables.get(fname, pd.DataFrame())
    add_piece(
        fname,
        df,
        select_rows(df, sample=main_sample, model="binary_game_reserve", game_cutoff_km="25"),
        "main_average_welfare_25km",
        1,
    )

    fname = "table_3_maasai_expanded_heterogeneity.csv"
    df = tables.get(fname, pd.DataFrame())
    add_piece(
        fname,
        df,
        select_rows(df, sample=main_sample, model="triple_game_x_maasai_expanded", game_cutoff_km="25"),
        "main_maasai_expanded_heterogeneity_25km",
        2,
    )

    fname = "table_5_controls_robustness.csv"
    df = tables.get(fname, pd.DataFrame())
    add_piece(
        fname,
        df,
        select_rows(df, sample=main_sample, model="controls_robustness_25km", game_cutoff_km="25"),
        "key_control_robustness_head_age_and_shocks",
        3,
    )

    fname = "table_6_weighted_welfare.csv"
    df = tables.get(fname, pd.DataFrame())
    add_piece(
        fname,
        df,
        select_rows(df, sample=main_sample, model="weighted_welfare_25km", game_cutoff_km="25"),
        "appendix_weighted_welfare_25km",
        4,
    )

    if not pieces:
        return pd.DataFrame()

    dash = pd.concat(pieces, ignore_index=True, sort=False)
    dash["interpretation_flag"] = dash.apply(interpretation_flag, axis=1)
    if "coef" in dash.columns:
        coef = pd.to_numeric(dash["coef"], errors="coerce")
        is_log = dash.get("outcome", pd.Series(index=dash.index, dtype=object)).astype(str).str.startswith("log_")
        dash["approx_log_effect_pct"] = np.where(is_log & coef.notna(), 100 * (np.exp(coef) - 1), np.nan)

    preferred = [
        "priority", "section", "source_table", "sample", "model", "outcome", "reported_term",
        "control_specification", "controls", "head_age_in_controls", "weighted", "coef", "se",
        "p_value", "q_value_bh", "approx_log_effect_pct", "n", "n_clusters",
        "n_input_rows", "n_dropped_missing_model_vars", "share_dropped_missing_model_vars",
        "cell_n_treated_proxy_post", "cell_n_treated_proxy_total", "interpretation_flag",
        "status", "formula",
    ]
    cols = [c for c in preferred if c in dash.columns] + [c for c in dash.columns if c not in preferred]
    return dash[cols].sort_values(["priority", "outcome", "control_specification"], na_position="last")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Tiny dependency-free markdown table writer."""
    if df.empty:
        return "_No rows._"
    safe = df.fillna("").astype(str)
    headers = list(safe.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in safe.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_dashboard_markdown(path: Path, dashboard: pd.DataFrame) -> None:
    """Write a compact human-readable dashboard."""
    lines = [
        "# Baseline game-reserve run dashboard",
        "",
        "Leggi questo file prima delle singole tabelle. Le tabelle dettagliate sono in `main/` e `appendix/`.",
        "",
        "## Controlli",
        "",
        f"Main controls: `{', '.join(CONTROLS_MAIN)}`.",
        "`head_age` è robustness, non main, per evitare attrition nascosta del campione.",
        "",
    ]
    if dashboard.empty:
        lines += ["Nessuna riga dashboard disponibile: controlla gli status nelle tabelle CSV.", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    show_cols = [c for c in [
        "section", "outcome", "control_specification", "coef", "se", "p_value", "q_value_bh",
        "approx_log_effect_pct", "n", "n_clusters", "share_dropped_missing_model_vars",
        "cell_n_treated_proxy_post", "interpretation_flag",
    ] if c in dashboard.columns]
    view = dashboard[show_cols].copy()
    for c in ["coef", "se", "p_value", "q_value_bh", "approx_log_effect_pct", "share_dropped_missing_model_vars"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
    lines += ["## Righe chiave", "", dataframe_to_markdown(view), ""]

    warnings_ = []
    if "share_dropped_missing_model_vars" in dashboard.columns:
        max_drop = pd.to_numeric(dashboard["share_dropped_missing_model_vars"], errors="coerce").max()
        if pd.notna(max_drop) and max_drop > 0.20:
            warnings_.append(f"Almeno una specificazione perde più del 20% del campione dopo il filtro missing ({max_drop:.1%}).")
    if "cell_n_treated_proxy_post" in dashboard.columns:
        min_cell = pd.to_numeric(dashboard["cell_n_treated_proxy_post"], errors="coerce").dropna()
        if not min_cell.empty and min_cell.min() < 100:
            warnings_.append("La cella treated × Maasai-expanded proxy × post è piccola in almeno una riga dashboard: controlla table_0 e table_3.")

    lines += ["## Warning automatici", ""]
    if warnings_:
        lines += [f"- {w}" for w in warnings_]
    else:
        lines += ["- Nessun warning automatico forte nella dashboard."]
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables_organized(outdir: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Save result tables into main/ and appendix/ and return a manifest."""
    manifest_rows = []
    for fname, table in tables.items():
        sub = output_subdir_for_table(fname)
        target_dir = outdir if sub == "." else outdir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        table_out = add_bh_q_values(table)
        target = target_dir / fname
        table_out.to_csv(target, index=False)
        manifest_rows.append({
            "file": fname,
            "section": sub,
            "relative_path": str(target.relative_to(outdir)),
            "n_rows": int(len(table_out)) if table_out is not None else 0,
            "n_columns": int(table_out.shape[1]) if table_out is not None and hasattr(table_out, "shape") else 0,
        })
    manifest = pd.DataFrame(manifest_rows).sort_values(["section", "file"])
    manifest.to_csv(outdir / "output_manifest.csv", index=False)
    return manifest

def write_notes(outdir: Path, args: argparse.Namespace, core_file: str, land_file: str, pastoral_file: str) -> None:
    notes = f"""Game-reserve baseline with expanded Maasai treatment and local reserve-risk main sample.

Research question:
- Among households in a credible local risk set around game reserves, did post-2009 reserve proximity differentially affect households located in Maasai-associated areas?

Input directory: {args.spatial_dir}
Output directory: {args.output_dir}

Detected files:
- core: {core_file}
- land: {land_file}
- pastoral: {pastoral_file}

Timing:
- Main post variable: {args.post}
- If post_2009: wave 1 is pre; waves 2+ are post.

Main game-reserve exposure:
- within_25km_game_reserve
- within_50km_game_reserve
- continuous: neg_distance_to_game_reserve_10km

Main Maasai identifier:
- maasai_expanded_area_proxy / maasai_expanded_25km
- Built upstream from the 83-row expanded Maasai locality dataset.
- Livestock, TLU, grazing land, and mobility variables are NOT used to identify Maasai status in this version.
- Legacy variables such as maasai_pastoral_composite_proxy are treated as backward-compatible aliases for expanded Maasai geography when produced by add_ward_and_village_treatment_v4.py.

Main sample:
- boundary{int(args.main_boundary_km)}km: households within {int(args.main_boundary_km)} km of a game-reserve boundary.

Robustness samples:
- full
- northern_regions: {args.northern_regions}
- boundary50km
- boundary100km
- boundary{int(args.main_boundary_km)}km_maasai{int(args.maasai_risk_km)}km
- northern_or_within{int(args.maasai_risk_km)}km_maasai_expanded

Boundary variables:
- distance_to_game_reserve_boundary_km
- neg_distance_to_game_reserve_boundary_10km
- boundary-band samples requested: {args.boundary_bands} km
- If true boundary distance was not created upstream, the script uses distance_to_game_reserve_km as a fallback. This is valid for outside points, but less informative for inside points.

Organized outputs:
- dashboard_main_results.md / dashboard_main_results.csv = first files to read after each run.
- output_manifest.csv = complete list of written tables.
- main/ = small set of tables to inspect routinely.
- appendix/ = robustness/mechanism tables to inspect only when needed.

Main tables:
1. main/table_0_support_game_reserve.csv
2. main/table_1_main_welfare_game_reserve.csv
3. main/table_3_maasai_expanded_heterogeneity.csv
4. main/table_5_controls_robustness.csv

Appendix tables:
- appendix/table_2_boundary_welfare_game_reserve.csv
- appendix/table_3b_maasai_expanded50_heterogeneity_robustness.csv
- appendix/table_3c_maasai_expanded_leave_one_district_out.csv
- appendix/table_4_mechanisms_game_reserve.csv
- appendix/table_4b_livestock_narrative.csv
- appendix/table_6_weighted_welfare.csv
- appendix/table_7_geospatial_market_controls.csv
- appendix/table_8_land_event_mechanisms.csv
- appendix/table_8b_sparse_land_event_appendix.csv

Specifications:
- Main welfare:
  Y ~ near_game_reserve + near_game_reserve × {args.post} + controls + wave FE + {args.fe_level} FE

- Main heterogeneity:
  Y ~ game + expanded_Maasai + game×expanded_Maasai + game×post + expanded_Maasai×post + game×expanded_Maasai×post
      + controls + wave FE + {args.fe_level} FE

Controls:
- Main baseline: {", ".join(CONTROLS_MAIN)}
- head_age is excluded from main controls and shown only in table_5 robustness.
- Robustness adds head_age, shocks, head_age+shocks, survey weights, geospatial/environment controls, and market-access controls.

Clustered SE:
- {args.cluster}

Interpretation warning:
These are repeated-cross-section spatial exposure models. Game-reserve proximity and expanded Maasai-area proximity are approximate spatial exposure measures because NPS coordinates are modified/offset. The boundary sample improves counterfactual credibility but is not a sharp RD.

Recommended reading:
- First read dashboard_main_results.md.
- Then inspect main/table_1 and main/table_3 if the dashboard changed.
- main/table_5 is the key place to check head_age attrition and robustness.
"""
    (outdir / "game_reserve_post2009_boundary_notes.txt").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core, core_file = read_csv_auto(args.spatial_dir, CORE_CANDIDATES)
    land, land_file = read_csv_auto(args.spatial_dir, LAND_CANDIDATES)
    pastoral, pastoral_file = read_csv_auto(args.spatial_dir, PASTORAL_CANDIDATES)

    core = prepare(core, args.post)
    land = prepare(land, args.post)
    pastoral = prepare(pastoral, args.post)

    core_samples = make_model_samples(core, args)
    land_samples = make_model_samples(land, args)
    pastoral_samples = make_model_samples(pastoral, args)
    boundary_samples = make_boundary_samples(core, args)

    support = pd.concat([
        support_table(core_samples, args.post, include_boundary=True).assign(dataset="core"),
        support_table(boundary_samples, args.post, include_boundary=True).assign(dataset="core_boundary"),
        support_table(land_samples, args.post, include_boundary=False).assign(dataset="land"),
        support_table(pastoral_samples, args.post, include_boundary=False).assign(dataset="pastoral"),
    ], ignore_index=True, sort=False)

    welfare = main_welfare(core_samples, args)
    bwelfare = boundary_welfare(boundary_samples, args)
    hetero_comp = maasai_expanded_heterogeneity(core_samples, args)
    hetero = maasai_expanded50_heterogeneity(core_samples, args)
    mech = mechanisms(pastoral_samples, args)
    livestock_story = livestock_narrative(pastoral_samples, args)
    ctrl_rob = controls_robustness(core_samples, args)
    weighted = weighted_welfare_robustness(core_samples, args)
    geospatial_market = geospatial_market_controls_robustness(core_samples, args)
    land_mech = land_event_mechanisms(land_samples, args, outcomes=LAND_MECHANISM_MAIN_OUTCOMES, table_role="main")
    land_sparse = land_event_mechanisms(land_samples, args, outcomes=LAND_MECHANISM_SPARSE_OUTCOMES, table_role="appendix_sparse")
    loo = maasai_expanded_leave_one_district_out(core_samples, args)

    tables = {
        "table_0_support_game_reserve.csv": support,
        "table_1_main_welfare_game_reserve.csv": welfare,
        "table_2_boundary_welfare_game_reserve.csv": bwelfare,
        "table_3_maasai_expanded_heterogeneity.csv": hetero_comp,
        "table_3b_maasai_expanded50_heterogeneity_robustness.csv": hetero,
        "table_4_mechanisms_game_reserve.csv": mech,
        "table_4b_livestock_narrative.csv": livestock_story,
        "table_5_controls_robustness.csv": ctrl_rob,
        "table_6_weighted_welfare.csv": weighted,
        "table_7_geospatial_market_controls.csv": geospatial_market,
        "table_8_land_event_mechanisms.csv": land_mech,
        "table_8b_sparse_land_event_appendix.csv": land_sparse,
        "table_3c_maasai_expanded_leave_one_district_out.csv": loo,
    }

    # Apply q-values once, write tables into main/ and appendix/, then build a
    # compact run dashboard from those q-value augmented tables.
    tables_q = {fname: add_bh_q_values(table) for fname, table in tables.items()}
    manifest = write_tables_organized(args.output_dir, tables_q)
    dashboard = build_dashboard(tables_q, main_sample=f"boundary{int(args.main_boundary_km)}km")
    dashboard.to_csv(args.output_dir / "dashboard_main_results.csv", index=False)
    write_dashboard_markdown(args.output_dir / "dashboard_main_results.md", dashboard)

    write_notes(args.output_dir, args, core_file, land_file, pastoral_file)

    print("Done.")
    print(f"Saved organized outputs to: {args.output_dir}")
    print("Read first:")
    print(args.output_dir / "dashboard_main_results.md")
    print(args.output_dir / "dashboard_main_results.csv")
    print(args.output_dir / "output_manifest.csv")
    print("Main tables:")
    for rel in manifest.loc[manifest["section"].eq("main"), "relative_path"].tolist():
        print(args.output_dir / rel)


if __name__ == "__main__":
    main()
