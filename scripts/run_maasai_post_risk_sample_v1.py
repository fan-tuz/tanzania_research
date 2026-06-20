#!/usr/bin/env python3
"""
run_maasai_post_risk_sample_v1.py

Estimate lower-sparsity Maasai-area post-2009 models inside local game-reserve
risk samples.

Research question
-----------------
Among households in a credible local risk set around game reserves, did
Maasai-associated areas experience differentially worse post-2009 outcomes?

Preferred specification
-----------------------
For each outcome Y_iwt:

    Y_iwt = beta * (is_masai_25km_i x Post_t)
          + controls
          + wave FE
          + district FE
          + f(distance_to_game_reserve_boundary_km)
          + error_iwt

Main sample:
    distance_to_game_reserve_boundary_km <= 75

Robustness:
    - boundary <= 50 km
    - boundary <= 100 km
    - full sample
    - northern regions only
    - boundary <= 75 km and within 100 km of expanded Maasai geography
    - is_masai_50km x Post_t instead of is_masai_25km x Post_t

This script intentionally removes the near-reserve binary from the main model.
The local reserve-risk sample itself defines exposure to the conservation-risk
environment. Livestock/pastoral variables are not used to identify Maasai status.

Typical run:
python scripts/run_maasai_post_risk_sample_v1.py \
  --spatial-dir output/spatial_restricted \
  --output-dir output/regressions/maasai_post_risk_sample
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import coerce_numeric_columns, existing, find_file, parse_int_list, split_csv

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CORE_CANDIDATES = ["analysis_core_1_4_spatial_restricted.csv", "analysis_core_1_4_spatial.csv"]

DEFAULT_OUTCOMES = ["log_cons_pc", "log_cons_ae", "cons_food_share_real"]

CONTROLS_MAIN = [
    "hh_size",
    "share_female",
    "n_children_u15",
    "n_adults_15plus",
]
CONTROL_COLUMNS = CONTROLS_MAIN + ["head_age", "n_shocks_occurred"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--input-file", default=None, help="Optional explicit CSV file inside --spatial-dir.")
    p.add_argument("--post", default="post_2009", choices=["post_2009", "post_2013"])
    p.add_argument("--cluster", default="ward2022_name")
    p.add_argument("--fe-level", default="district2022_name",
                   choices=["district2022_name", "region2022_name", "ward2022_name", "none"])
    p.add_argument("--main-boundary-km", default=75, type=int)
    p.add_argument("--boundary-bands", default="50,75,100",
                   help="Comma-separated boundary risk-sample radii, in km.")
    p.add_argument("--maasai-risk-km", default=100.0, type=float,
                   help="For the boundary+Maasai robustness sample.")
    p.add_argument("--primary-maasai-col", default="is_masai_25km")
    p.add_argument("--robust-maasai-col", default="is_masai_50km")
    p.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES),
                   help="Comma-separated outcome list.")
    p.add_argument("--distance-poly", default=2, type=int, choices=[0, 1, 2, 3],
                   help="Polynomial order for f(distance_to_game_reserve_boundary_km). Default: quadratic.")
    p.add_argument("--northern-regions", default="Arusha,Manyara")
    p.add_argument("--min-n", default=100, type=int)
    p.add_argument("--min-clusters", default=20, type=int)
    return p.parse_args()


def read_core_csv(spatial_dir: Path, input_file: Optional[str]) -> tuple[pd.DataFrame, str]:
    if input_file:
        path = spatial_dir / input_file
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        path = find_file(spatial_dir, CORE_CANDIDATES, required=True)
    return pd.read_csv(path, low_memory=False), path.name


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


def ensure_maasai_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Create is_masai_25km and is_masai_50km if upstream script used older names."""
    df = df.copy()

    if "is_masai_25km" not in df.columns:
        for c in ["maasai_expanded_25km", "maasai_expanded_area_proxy", "treated_area", "treat_25km"]:
            if c in df.columns:
                df["is_masai_25km"] = pd.to_numeric(df[c], errors="coerce")
                break

    if "is_masai_50km" not in df.columns:
        for c in ["maasai_expanded_50km", "maasai_expanded_area_50km", "treat_50km", "pastoral_reference_50km"]:
            if c in df.columns:
                df["is_masai_50km"] = pd.to_numeric(df[c], errors="coerce")
                break

    return df


def prepare(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = reconstruct_post(df, args.post)
    df = ensure_maasai_aliases(df)
    df = df.copy()

    numeric = set(
        ["wave", "survey_year", args.post,
         "distance_to_game_reserve_boundary_km", "distance_to_game_reserve_km",
         "distance_to_maasai_expanded_km", "distance_to_maasai_reference_km",
         "is_masai_25km", "is_masai_50km"]
        + DEFAULT_OUTCOMES
        + CONTROL_COLUMNS
    )
    df = coerce_numeric_columns(df, numeric)

    if "distance_to_game_reserve_boundary_km" not in df.columns:
        if "distance_to_game_reserve_km" in df.columns:
            df["distance_to_game_reserve_boundary_km"] = pd.to_numeric(df["distance_to_game_reserve_km"], errors="coerce")
        else:
            raise ValueError("Need distance_to_game_reserve_boundary_km or distance_to_game_reserve_km.")

    bd = pd.to_numeric(df["distance_to_game_reserve_boundary_km"], errors="coerce").clip(lower=0)
    df["boundary_distance_10km"] = bd / 10.0
    df["boundary_distance_10km_sq"] = df["boundary_distance_10km"] ** 2
    df["boundary_distance_10km_cu"] = df["boundary_distance_10km"] ** 3

    # Harmonize Maasai-distance alias for boundary+Maasai robustness.
    if "distance_to_maasai_expanded_km" not in df.columns and "distance_to_maasai_reference_km" in df.columns:
        df["distance_to_maasai_expanded_km"] = pd.to_numeric(df["distance_to_maasai_reference_km"], errors="coerce")

    return df


def make_samples(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    bd = pd.to_numeric(df["distance_to_game_reserve_boundary_km"], errors="coerce")

    # Main first.
    samples[f"boundary{int(args.main_boundary_km)}km_main"] = df.loc[bd <= args.main_boundary_km].copy()

    for b in parse_int_list(args.boundary_bands):
        name = f"boundary{b}km"
        if name not in samples:
            samples[name] = df.loc[bd <= b].copy()

    samples["full"] = df.copy()

    regions = {x.strip().lower() for x in split_csv(args.northern_regions)}
    if regions and "region2022_name" in df.columns:
        reg = df["region2022_name"].astype(str).str.strip().str.lower()
        samples["northern_regions"] = df.loc[reg.isin(regions)].copy()

    if "distance_to_maasai_expanded_km" in df.columns:
        md = pd.to_numeric(df["distance_to_maasai_expanded_km"], errors="coerce")
        samples[f"boundary{int(args.main_boundary_km)}km_within{int(args.maasai_risk_km)}km_maasai"] = df.loc[
            (bd <= args.main_boundary_km) & (md <= args.maasai_risk_km)
        ].copy()

    return samples


def distance_terms(args: argparse.Namespace) -> list[str]:
    if args.distance_poly == 0:
        return []
    terms = ["boundary_distance_10km"]
    if args.distance_poly >= 2:
        terms.append("boundary_distance_10km_sq")
    if args.distance_poly >= 3:
        terms.append("boundary_distance_10km_cu")
    return terms


def coefficient_name(maasai_col: str, post_col: str, params_index: Iterable[str]) -> Optional[str]:
    candidates = [f"{maasai_col}:{post_col}", f"{post_col}:{maasai_col}"]
    params = set(params_index)
    for c in candidates:
        if c in params:
            return c
    return None


def fit_one(
    df: pd.DataFrame,
    outcome: str,
    maasai_col: str,
    sample_name: str,
    args: argparse.Namespace,
) -> dict:
    controls = existing(df, CONTROLS_MAIN)
    dterms = distance_terms(args)
    terms = [maasai_col, args.post, f"{maasai_col}:{args.post}"] + controls + dterms

    needed = [outcome, maasai_col, args.post] + controls + dterms + ["wave"]
    if args.fe_level != "none" and args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    d = df.dropna(subset=needed).copy()
    n = int(len(d))
    n_clusters = int(d[args.cluster].nunique(dropna=True)) if args.cluster in d.columns else np.nan

    base = {
        "sample": sample_name,
        "outcome": outcome,
        "maasai_col": maasai_col,
        "post_col": args.post,
        "distance_function": f"poly_{args.distance_poly}",
        "controls": "+".join(controls),
        "n": n,
        "clusters": n_clusters,
        "n_maasai": int(pd.to_numeric(d.get(maasai_col), errors="coerce").eq(1).sum()) if n else 0,
        "n_post": int(pd.to_numeric(d.get(args.post), errors="coerce").eq(1).sum()) if n else 0,
        "n_maasai_post": int((pd.to_numeric(d.get(maasai_col), errors="coerce").eq(1) & pd.to_numeric(d.get(args.post), errors="coerce").eq(1)).sum()) if n else 0,
        "mean_outcome": float(pd.to_numeric(d[outcome], errors="coerce").mean()) if n else np.nan,
        "status": "not_estimated",
    }

    if n < args.min_n:
        base["status"] = f"too_few_obs_min_{args.min_n}"
        return base
    if args.cluster in d.columns and pd.notna(n_clusters) and n_clusters < args.min_clusters:
        base["status"] = f"too_few_clusters_min_{args.min_clusters}"
        return base
    if pd.to_numeric(d[maasai_col], errors="coerce").nunique(dropna=True) < 2:
        base["status"] = "no_maasai_variation"
        return base
    if pd.to_numeric(d[args.post], errors="coerce").nunique(dropna=True) < 2:
        base["status"] = "no_post_variation"
        return base

    rhs = terms.copy()
    rhs.append("C(wave)")
    if args.fe_level != "none" and args.fe_level in d.columns:
        rhs.append(f"C({args.fe_level})")
    formula = f"{outcome} ~ " + " + ".join(rhs)

    try:
        model = smf.ols(formula, data=d)
        if args.cluster in d.columns and d[args.cluster].nunique(dropna=True) >= 2:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": d[args.cluster].astype(str)})
            cov_type = f"cluster:{args.cluster}"
        else:
            res = model.fit(cov_type="HC1")
            cov_type = "HC1"
        cname = coefficient_name(maasai_col, args.post, res.params.index)
        if cname is None:
            base["status"] = "interaction_coef_not_found"
            base["formula"] = formula
            return base
        base.update({
            "status": "ok",
            "term": cname,
            "coef": float(res.params[cname]),
            "se": float(res.bse[cname]),
            "t": float(res.tvalues[cname]),
            "p": float(res.pvalues[cname]),
            "r2": float(getattr(res, "rsquared", np.nan)),
            "cov_type": cov_type,
            "formula": formula,
        })
        return base
    except Exception as e:
        base["status"] = f"error:{type(e).__name__}: {e}"
        base["formula"] = formula
        return base


def run_models(df: pd.DataFrame, args: argparse.Namespace, maasai_col: str) -> pd.DataFrame:
    outcomes = split_csv(args.outcomes) or DEFAULT_OUTCOMES
    samples = make_samples(df, args)
    rows = []
    for sample_name, sdf in samples.items():
        for outcome in outcomes:
            if outcome not in sdf.columns:
                rows.append({
                    "sample": sample_name,
                    "outcome": outcome,
                    "maasai_col": maasai_col,
                    "status": "missing_outcome",
                })
                continue
            if maasai_col not in sdf.columns:
                rows.append({
                    "sample": sample_name,
                    "outcome": outcome,
                    "maasai_col": maasai_col,
                    "status": "missing_maasai_col",
                })
                continue
            rows.append(fit_one(sdf, outcome, maasai_col, sample_name, args))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, source_name = read_core_csv(args.spatial_dir, args.input_file)
    df = prepare(df, args)

    primary = run_models(df, args, args.primary_maasai_col)
    primary["specification_set"] = "main_is_masai_25km" if args.primary_maasai_col == "is_masai_25km" else "main"

    robust = run_models(df, args, args.robust_maasai_col)
    robust["specification_set"] = "robustness_is_masai_50km" if args.robust_maasai_col == "is_masai_50km" else "robustness"

    combined = pd.concat([primary, robust], ignore_index=True, sort=False)

    primary_out = args.output_dir / "table_maasai25_post_risk_sample.csv"
    robust_out = args.output_dir / "table_maasai50_post_robustness.csv"
    combined_out = args.output_dir / "table_maasai_post_risk_sample_combined.csv"
    support_out = args.output_dir / "table_maasai_post_risk_sample_support.csv"

    primary.to_csv(primary_out, index=False)
    robust.to_csv(robust_out, index=False)
    combined.to_csv(combined_out, index=False)

    # Compact support table by sample and Maasai definition.
    support_cols = [
        "specification_set", "sample", "maasai_col", "outcome", "n", "clusters",
        "n_maasai", "n_post", "n_maasai_post", "status"
    ]
    existing_support_cols = [c for c in support_cols if c in combined.columns]
    combined[existing_support_cols].to_csv(support_out, index=False)

    notes = args.output_dir / "maasai_post_risk_sample_notes.txt"
    notes.write_text(
        "Maasai post risk-sample models\n"
        "================================\n"
        f"Input file: {source_name}\n"
        f"Post indicator: {args.post}\n"
        f"Primary Maasai column: {args.primary_maasai_col}\n"
        f"Robust Maasai column: {args.robust_maasai_col}\n"
        f"Main boundary sample: <= {args.main_boundary_km} km\n"
        f"Distance function: polynomial order {args.distance_poly} in distance_to_game_reserve_boundary_km / 10\n"
        "Near-reserve binary is intentionally excluded from these models.\n"
    )

    print(f"Wrote {primary_out}")
    print(f"Wrote {robust_out}")
    print(f"Wrote {combined_out}")
    print(f"Wrote {support_out}")


if __name__ == "__main__":
    main()
