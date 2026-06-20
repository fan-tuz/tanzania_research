#!/usr/bin/env python3
"""
run_wma_active_inactive_staggered.py

Staggered WMA active-vs-future design for Tanzania NPS analysis subsets.

Purpose
-------
Use WMA gazettement years as institutional timing:
    active_wma = NPS unit belongs to a WMA whose gazettement year is <= survey_year
    inactive_future_wma = NPS unit belongs to a WMA whose gazettement year is > survey_year
    never_wma = omitted baseline in the main specification

Preferred contrast:
    active_wma - inactive_future_wma

Reserve-specific contrast:
    (game_reserve × active_wma) - (game_reserve × inactive_future_wma)

This mirrors the "active vs future/inactive" script but uses the externally
reconstructed WMA→village→gazettement-year crosswalk.

Important caveat
----------------
The script can only be as good as the match between NPS geography and the WMA
crosswalk. If your NPS file has only ward names, matching village_name to
ward2022_name is a conservative/partial name match, not proof that the sampled
cluster is inside the village. Prefer village polygons or WMA polygons if you have
them. The script writes match diagnostics before running regressions.

Typical run
-----------
python scripts/run_wma_active_inactive_staggered.py \
  --spatial-dir output/spatial_restricted \
  --wma-crosswalk data/tanzania_wma_village_year_crosswalk_best_effort.csv \
  --output-dir output/wma_active_inactive_results \
  --dataset core \
  --nps-unit-col ward2022_name \
  --nps-district-col district2022_name \
  --wma-unit-col village_name \
  --wma-district-col district_strategy \
  --game-col within_25km_game_reserve \
  --cluster ward2022_name \
  --fe-level district2022_name
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import existing_nonconstant as existing, norm_text, to_num, find_file, add_bh_q_values


DATASET_CANDIDATES = {
    "core": ["analysis_core_1_4_spatial.csv", "analysis_core_1_4_spatial_restricted.csv"],
    "land": ["analysis_land_2_5_spatial.csv", "analysis_land_2_4_spatial.csv"],
    "pastoral": ["analysis_pastoral_3_5_spatial.csv", "analysis_pastoral_3_4_spatial.csv"],
}

WELFARE_OUTCOMES = ["log_cons_pc", "log_cons_ae", "cons_food_share_real"]
MECHANISM_OUTCOMES = [
    "log1p_livestock_total_current",
    "livestock_tlu_default",
    "livestock_tlu_per_ae_default",
    "livestock_change_12m",
    "livestock_decline_12m",
    "log1p_livestock_sales_value",
    "livestock_migration_any",
    "grazing_land_pct_clean",
    "cultivation_villagers_pct_clean",
    "has_village_land_certificate",
]
BASE_CONTROLS = ["hh_size", "head_age", "share_female", "n_children_u15", "n_adults_15plus"]
ROBUST_CONTROLS = BASE_CONTROLS + ["n_shocks_occurred"]








def find_input(spatial_dir: Path, dataset: str) -> Path:
    candidates = DATASET_CANDIDATES[dataset]
    for c in candidates:
        p = find_file(spatial_dir, [c], required=False)
        if p is not None:
            return p
    # fallback: any matching prefix
    prefix = f"analysis_{dataset}"
    matches = sorted(spatial_dir.glob(f"{prefix}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No input file for dataset={dataset} in {spatial_dir}")


def ensure_survey_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "survey_year" in df.columns and to_num(df["survey_year"]).notna().any():
        df["survey_year"] = to_num(df["survey_year"])
        return df
    if "wave" not in df.columns:
        raise ValueError("Need survey_year or wave.")
    wave_to_year = {1: 2008, 2: 2010, 3: 2012, 4: 2014, 5: 2020}
    df["survey_year"] = to_num(df["wave"]).map(wave_to_year)
    return df


def prepare_wma_crosswalk(args) -> pd.DataFrame:
    w = pd.read_csv(args.wma_crosswalk, low_memory=False).copy()
    for col in [args.wma_unit_col, args.wma_district_col, "wma_name", "gazettement_year_strategy"]:
        if col not in w.columns:
            raise ValueError(f"WMA crosswalk missing required column: {col}")

    w["wma_event_year"] = to_num(w["gazettement_year_strategy"])
    w = w[w["wma_event_year"].notna()].copy()
    if args.drop_partial:
        w = w[w.get("coverage_status", "").astype(str).str.contains("complete", case=False, na=False)].copy()
    if args.min_confidence:
        keep = set(args.min_confidence.split(","))
        if "source_confidence" in w.columns:
            w = w[w["source_confidence"].astype(str).isin(keep)].copy()

    w["_unit_norm"] = w[args.wma_unit_col].map(norm_text)
    w["_district_norm"] = w[args.wma_district_col].map(norm_text)
    w = w[w["_unit_norm"].ne("")].copy()

    # For duplicate village/name matches, keep earliest gazettement year.
    agg = (
        w.groupby(["_unit_norm", "_district_norm"], as_index=False)
         .agg(
             wma_event_year=("wma_event_year", "min"),
             wma_names=("wma_name", lambda s: "; ".join(sorted(set(map(str, s.dropna()))))),
             wma_source_rows=("wma_name", "size"),
             wma_coverage_status=("coverage_status", lambda s: "; ".join(sorted(set(map(str, s.dropna())))) if "coverage_status" in w.columns else ""),
             wma_source_confidence=("source_confidence", lambda s: "; ".join(sorted(set(map(str, s.dropna())))) if "source_confidence" in w.columns else ""),
         )
    )
    return agg


def attach_wma_event(df: pd.DataFrame, wma: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    for c in [args.nps_unit_col, args.nps_district_col]:
        if c not in out.columns:
            raise ValueError(f"NPS dataset missing match column: {c}")

    out["_unit_norm"] = out[args.nps_unit_col].map(norm_text)
    out["_district_norm"] = out[args.nps_district_col].map(norm_text)

    if args.match_on_district:
        out = out.merge(wma, on=["_unit_norm", "_district_norm"], how="left")
    else:
        # Name-only fallback. If duplicates exist across districts, use earliest event year and concatenated WMA names.
        w2 = (
            wma.groupby("_unit_norm", as_index=False)
               .agg(
                   wma_event_year=("wma_event_year", "min"),
                   wma_names=("wma_names", lambda s: "; ".join(sorted(set("; ".join(s).split("; "))))),
                   wma_source_rows=("wma_source_rows", "sum"),
                   wma_coverage_status=("wma_coverage_status", lambda s: "; ".join(sorted(set("; ".join(s).split("; "))))),
                   wma_source_confidence=("wma_source_confidence", lambda s: "; ".join(sorted(set("; ".join(s).split("; "))))),
               )
        )
        out = out.merge(w2, on="_unit_norm", how="left")

    out["ever_wma"] = out["wma_event_year"].notna().astype(float)
    out["wma_active"] = np.where(
        out["wma_event_year"].notna() & out["survey_year"].notna() & (out["wma_event_year"] <= out["survey_year"]),
        1.0, 0.0
    )
    out["wma_inactive_future"] = np.where(
        out["wma_event_year"].notna() & out["survey_year"].notna() & (out["wma_event_year"] > out["survey_year"]),
        1.0, 0.0
    )
    out["wma_event_time"] = out["survey_year"] - out["wma_event_year"]
    out["wma_cohort_2009"] = (out["wma_event_year"] == 2009).astype(float)
    out["wma_post_2009_cohort_active"] = ((out["wma_event_year"] == 2009) & (out["survey_year"] >= 2009)).astype(float)

    if args.game_col in out.columns:
        out["game_x_wma_active"] = to_num(out[args.game_col]) * out["wma_active"]
        out["game_x_wma_inactive_future"] = to_num(out[args.game_col]) * out["wma_inactive_future"]
        out["game_x_wma_cohort2009_active"] = to_num(out[args.game_col]) * out["wma_post_2009_cohort_active"]

    return out.drop(columns=["_unit_norm", "_district_norm"], errors="ignore")


def wma_distribution(wma_raw_path: Path) -> pd.DataFrame:
    w = pd.read_csv(wma_raw_path, low_memory=False).copy()
    w["gazettement_year_strategy"] = to_num(w.get("gazettement_year_strategy"))
    w = w[w["gazettement_year_strategy"].notna()].copy()
    rows = []
    for yr, g in w.groupby("gazettement_year_strategy"):
        rows.append({
            "gazettement_year": int(yr),
            "n_wma": g["wma_name"].dropna().nunique() if "wma_name" in g.columns else np.nan,
            "wma_names": "; ".join(sorted(g["wma_name"].dropna().astype(str).unique())) if "wma_name" in g.columns else "",
            "village_rows": int(g["village_name"].notna().sum()) if "village_name" in g.columns else np.nan,
            "unique_villages_simple_norm": g["village_name"].dropna().map(norm_text).nunique() if "village_name" in g.columns else np.nan,
            "complete_or_high_conf_village_rows": int(g.loc[
                g.get("village_name", pd.Series(index=g.index)).notna()
                & (
                    g.get("coverage_status", pd.Series("", index=g.index)).astype(str).str.contains("complete", case=False, na=False)
                    | g.get("source_confidence", pd.Series("", index=g.index)).astype(str).isin(["high", "medium_high"])
                ),
                "village_name"
            ].count()) if "village_name" in g.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values("gazettement_year")


def counts_by_wave(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for wave, g in df.groupby("wave", dropna=False):
        rows.append({
            "wave": wave,
            "survey_year_min": to_num(g["survey_year"]).min(),
            "survey_year_max": to_num(g["survey_year"]).max(),
            "n_obs": len(g),
            "n_units": g[args.nps_unit_col].nunique(dropna=True) if args.nps_unit_col in g.columns else np.nan,
            "ever_wma_obs": int(g["ever_wma"].sum()),
            "wma_active_obs": int(g["wma_active"].sum()),
            "wma_inactive_future_obs": int(g["wma_inactive_future"].sum()),
            "never_wma_obs": int(((g["wma_active"] == 0) & (g["wma_inactive_future"] == 0)).sum()),
            "game_x_wma_active_obs": int(g.get("game_x_wma_active", pd.Series(0, index=g.index)).sum()),
            "game_x_wma_inactive_future_obs": int(g.get("game_x_wma_inactive_future", pd.Series(0, index=g.index)).sum()),
        })
    return pd.DataFrame(rows)


def fit_active_inactive(df, outcome, controls, args, interact_game=False):
    game = args.game_col
    terms = ["wma_active", "wma_inactive_future"]
    report_active = "wma_active"
    report_inactive = "wma_inactive_future"
    if interact_game:
        if game not in df.columns:
            return {"status": "missing_game_col", "outcome": outcome, "interact_game": interact_game}
        terms = [game, "wma_active", "wma_inactive_future", f"{game}:wma_active", f"{game}:wma_inactive_future"]
        report_active = f"{game}:wma_active"
        report_inactive = f"{game}:wma_inactive_future"

    needed = [outcome] + [game if t == game else t for t in terms if ":" not in t] + controls
    if "wave" in df.columns:
        needed.append("wave")
    if args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    missing = [c for c in [outcome, "wma_active", "wma_inactive_future"] if c not in df.columns]
    if interact_game and game not in df.columns:
        missing.append(game)
    if missing:
        return {"status": "missing_columns", "missing_columns": ",".join(missing), "outcome": outcome}

    d = df.dropna(subset=needed).copy()
    if len(d) < args.min_n:
        return {"status": "too_few_obs", "outcome": outcome, "n": len(d), "interact_game": interact_game}
    if d["wma_active"].sum() == 0 or d["wma_inactive_future"].sum() == 0:
        return {
            "status": "no_active_or_future_support",
            "outcome": outcome,
            "n": len(d),
            "active_n": float(d["wma_active"].sum()),
            "inactive_future_n": float(d["wma_inactive_future"].sum()),
            "interact_game": interact_game,
        }
    if interact_game and (d[game].fillna(0).mul(d["wma_active"]).sum() == 0 or d[game].fillna(0).mul(d["wma_inactive_future"]).sum() == 0):
        return {
            "status": "no_game_active_or_future_support",
            "outcome": outcome,
            "n": len(d),
            "game_active_n": float(d[game].fillna(0).mul(d["wma_active"]).sum()),
            "game_inactive_future_n": float(d[game].fillna(0).mul(d["wma_inactive_future"]).sum()),
            "interact_game": interact_game,
        }

    rhs = terms + controls
    if "wave" in d.columns:
        rhs.append("C(wave)")
    if args.fe_level in d.columns:
        rhs.append(f"C({args.fe_level})")
    formula = outcome + " ~ " + " + ".join(rhs)

    try:
        mod = smf.ols(formula, data=d)
        if args.cluster in d.columns and d[args.cluster].nunique(dropna=True) >= 2:
            fit = mod.fit(cov_type="cluster", cov_kwds={"groups": d[args.cluster]})
            se_type = f"cluster:{args.cluster}"
        else:
            fit = mod.fit(cov_type="HC1")
            se_type = "HC1"
    except Exception as exc:
        return {"status": "fit_error", "error": repr(exc), "outcome": outcome, "formula": formula, "interact_game": interact_game}

    b_a = fit.params.get(report_active, np.nan)
    b_f = fit.params.get(report_inactive, np.nan)
    cov = fit.cov_params()
    diff = b_a - b_f
    if report_active in cov.index and report_inactive in cov.index:
        var = cov.loc[report_active, report_active] + cov.loc[report_inactive, report_inactive] - 2 * cov.loc[report_active, report_inactive]
        se = math.sqrt(var) if var >= 0 else np.nan
        t = diff / se if se and se > 0 else np.nan
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if not np.isnan(t) else np.nan
    else:
        se, t, p = np.nan, np.nan, np.nan

    return {
        "status": "ok",
        "outcome": outcome,
        "interact_game": interact_game,
        "reported_active_term": report_active,
        "reported_inactive_future_term": report_inactive,
        "formula": formula,
        "n": int(fit.nobs),
        "n_clusters": int(d[args.cluster].nunique(dropna=True)) if args.cluster in d.columns else np.nan,
        "se_type": se_type,
        "active_n": float(d["wma_active"].sum()),
        "inactive_future_n": float(d["wma_inactive_future"].sum()),
        "game_active_n": float(d[args.game_col].fillna(0).mul(d["wma_active"]).sum()) if args.game_col in d.columns else np.nan,
        "game_inactive_future_n": float(d[args.game_col].fillna(0).mul(d["wma_inactive_future"]).sum()) if args.game_col in d.columns else np.nan,
        "coef_active": float(b_a) if pd.notna(b_a) else np.nan,
        "se_active": float(fit.bse.get(report_active, np.nan)) if report_active in fit.bse else np.nan,
        "p_active": float(fit.pvalues.get(report_active, np.nan)) if report_active in fit.pvalues else np.nan,
        "coef_inactive_future": float(b_f) if pd.notna(b_f) else np.nan,
        "se_inactive_future": float(fit.bse.get(report_inactive, np.nan)) if report_inactive in fit.bse else np.nan,
        "p_inactive_future": float(fit.pvalues.get(report_inactive, np.nan)) if report_inactive in fit.pvalues else np.nan,
        "active_minus_inactive_future": float(diff) if pd.notna(diff) else np.nan,
        "se_active_minus_inactive_future": float(se) if pd.notna(se) else np.nan,
        "p_active_minus_inactive_future": float(p) if pd.notna(p) else np.nan,
        "controls": ",".join(controls),
    }


def run_models(df, outcomes, args, label):
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for controls_name, controls_raw in {
            "demo": BASE_CONTROLS,
            "demo_shock": ROBUST_CONTROLS,
        }.items():
            controls = existing(df, controls_raw)
            for interact in [False, True]:
                res = fit_active_inactive(df, outcome, controls, args, interact_game=interact)
                res["controls_name"] = controls_name
                res["dataset"] = label
                rows.append(res)
    return pd.DataFrame(rows)


def save_notes(outdir: Path, args, input_path: Path):
    txt = f"""WMA active-vs-future analysis completed.

Input NPS file: {input_path}
WMA crosswalk: {args.wma_crosswalk}

Matching:
- NPS unit column: {args.nps_unit_col}
- NPS district column: {args.nps_district_col}
- WMA unit column: {args.wma_unit_col}
- WMA district column: {args.wma_district_col}
- Match on district: {args.match_on_district}

Design:
- wma_active = 1 if matched WMA gazettement year <= survey_year.
- wma_inactive_future = 1 if matched WMA gazettement year > survey_year.
- never-WMA observations are omitted baseline in the main active/future model.
- Preferred contrast is active_minus_inactive_future.
- If --game-col is available, the reserve-specific preferred contrast is
  (game × wma_active) - (game × wma_inactive_future).

Read first:
1. table_A_wma_gazettement_distribution.csv
2. table_B_wma_match_diagnostics.csv
3. table_C_wma_active_inactive_counts_by_wave.csv
4. table_D_wma_active_inactive_welfare.csv
5. table_E_wma_active_inactive_mechanisms.csv

Caveats:
- If NPS only has ward names and WMA crosswalk is village-level, this is a partial name match.
- 2006 and 2007 WMA cohorts are already active before NPS wave 1, so they do not provide
  future/inactive support.
- The strongest staggered support should come from WMA gazetted in 2009, 2010, 2012, 2013,
  2015 and 2016, depending on which NPS waves are in the input file.
"""
    (outdir / "wma_active_inactive_notes.txt").write_text(txt, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-dir", required=True, type=Path)
    p.add_argument("--wma-crosswalk", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset", default="core", choices=["core", "land", "pastoral"])
    p.add_argument("--nps-unit-col", default="ward2022_name")
    p.add_argument("--nps-district-col", default="district2022_name")
    p.add_argument("--wma-unit-col", default="village_name")
    p.add_argument("--wma-district-col", default="district_strategy")
    p.add_argument("--match-on-district", action="store_true", default=True)
    p.add_argument("--name-only-match", action="store_false", dest="match_on_district")
    p.add_argument("--drop-partial", action="store_true", help="Keep only coverage_status containing 'complete'.")
    p.add_argument("--min-confidence", default="", help="Comma-separated source_confidence values to keep, e.g. high,medium_high.")
    p.add_argument("--game-col", default="within_25km_game_reserve")
    p.add_argument("--cluster", default="ward2022_name")
    p.add_argument("--fe-level", default="district2022_name")
    p.add_argument("--min-n", type=int, default=50)
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = find_input(args.spatial_dir, args.dataset)
    print(f"Reading NPS input: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df = ensure_survey_year(df)

    print("Preparing WMA crosswalk...")
    wma = prepare_wma_crosswalk(args)

    print("Attaching WMA event timing to NPS observations...")
    out = attach_wma_event(df, wma, args)

    out.to_csv(outdir / f"analysis_{args.dataset}_wma_active_inactive.csv", index=False)

    print("Writing distribution and diagnostics...")
    wma_distribution(args.wma_crosswalk).to_csv(outdir / "table_A_wma_gazettement_distribution.csv", index=False)

    diag = pd.DataFrame([{
        "input_rows": len(out),
        "nps_units": out[args.nps_unit_col].nunique(dropna=True) if args.nps_unit_col in out.columns else np.nan,
        "matched_rows_ever_wma": int(out["ever_wma"].sum()),
        "matched_units_ever_wma": out.loc[out["ever_wma"].eq(1), args.nps_unit_col].nunique(dropna=True) if args.nps_unit_col in out.columns else np.nan,
        "active_rows": int(out["wma_active"].sum()),
        "inactive_future_rows": int(out["wma_inactive_future"].sum()),
        "game_active_rows": int(out.get("game_x_wma_active", pd.Series(0, index=out.index)).sum()),
        "game_inactive_future_rows": int(out.get("game_x_wma_inactive_future", pd.Series(0, index=out.index)).sum()),
        "matching_note": "village-to-NPS-unit name match; inspect matched file carefully",
    }])
    diag.to_csv(outdir / "table_B_wma_match_diagnostics.csv", index=False)
    counts_by_wave(out, args).to_csv(outdir / "table_C_wma_active_inactive_counts_by_wave.csv", index=False)

    print("Running welfare models...")
    welfare = add_bh_q_values(run_models(out, WELFARE_OUTCOMES, args, args.dataset), p_col="p_active_minus_inactive_future", q_col="q_active_minus_inactive_future_bh")
    welfare.to_csv(outdir / "table_D_wma_active_inactive_welfare.csv", index=False)

    print("Running mechanism models...")
    mech = add_bh_q_values(run_models(out, MECHANISM_OUTCOMES, args, args.dataset), p_col="p_active_minus_inactive_future", q_col="q_active_minus_inactive_future_bh")
    mech.to_csv(outdir / "table_E_wma_active_inactive_mechanisms.csv", index=False)

    save_notes(outdir, args, input_path)

    print("Done.")
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()
