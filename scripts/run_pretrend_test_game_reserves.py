#!/usr/bin/env python3
"""
run_pretrend_test_game_reserves.py

Dynamic pre-trend / event-study diagnostics for the game-reserve ×
Maasai/pastoral proxy design.

Important identification note
-----------------------------
If the event is post_2009 and the available NPS waves are 1=2008/09,
2=2010/11, 3=2012/13, 4=2014/15, there is only ONE clean pre-period wave.
A conventional parallel-trends test requires at least two pre-period waves.
This script therefore:
  1) estimates dynamic wave-specific interactions relative to a base wave;
  2) runs a joint pre-trend test only when at least one non-base lead exists;
  3) writes explicit notes when post_2009 has no true pre-trend test;
  4) adds baseline-balance diagnostics in the base wave.

Typical run:
python scripts/run_pretrend_test_game_reserves.py \
  --spatial-dir data/analysis_subsets_spatial_restricted_v2 \
  --output-dir data/pretrend_results_game_reserves \
  --event-wave 2 \
  --base-wave 1 \
  --game-col within_25km_game_reserve \
  --proxy-col maasai_pastoral_composite_proxy \
  --cluster ward2022_name \
  --fe-level district2022_name

For a true mechanical pre-trend test with waves 1-3 as leads, use event-wave 4
as a placebo/event-date diagnostic:
python scripts/run_pretrend_test_game_reserves.py ... --event-wave 4 --base-wave 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import existing, find_file, split_csv, to_binary, to_float, add_bh_q_values

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CORE_CANDIDATES = ["analysis_core_1_4_spatial.csv", "analysis_core_1_4_spatial_restricted.csv"]

DEFAULT_OUTCOMES = ["log_cons_pc", "log_cons_ae", "cons_food_share_real"]
DEFAULT_CONTROLS = ["hh_size", "head_age", "share_female", "n_children_u15", "n_adults_15plus"]
DEFAULT_BALANCE = DEFAULT_CONTROLS + [
    "rural", "cons_total_monthly_real_pc", "cons_total_monthly_real_ae",
    "livestock_total_current", "livestock_tlu_default", "grazing_land_pct_clean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None,
                   help="Optional direct CSV input. If omitted, searches --spatial-dir for analysis_core files.")
    p.add_argument("--event-wave", type=int, default=2,
                   help="First treated/post wave. For post_2009 use 2; for placebo post_2013 use 4.")
    p.add_argument("--base-wave", type=int, default=1)
    p.add_argument("--game-col", default="within_25km_game_reserve")
    p.add_argument("--proxy-col", default="maasai_pastoral_composite_proxy")
    p.add_argument("--cluster", default="ward2022_name")
    p.add_argument("--fe-level", default="district2022_name")
    p.add_argument("--weight-col", default=None,
                   help="Optional survey weight column for WLS dynamic diagnostics.")
    p.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES))
    p.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    p.add_argument("--northern-regions", default="Arusha,Manyara")
    p.add_argument("--risk-km", type=float, default=100.0)
    p.add_argument("--min-n", type=int, default=100)
    return p.parse_args()






def read_input(args: argparse.Namespace) -> pd.DataFrame:
    path = args.data if args.data else find_file(args.spatial_dir, CORE_CANDIDATES, required=True)
    df = pd.read_csv(path, low_memory=False)
    df["_source_file"] = path.name
    return df








def add_missing_exposure_aliases(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.game_col not in out.columns and "distance_to_game_reserve_km" in out.columns:
        if args.game_col.startswith("within_") and "km_game_reserve" in args.game_col:
            km = float(args.game_col.split("within_", 1)[1].split("km", 1)[0])
            d = to_float(out["distance_to_game_reserve_km"])
            out[args.game_col] = np.where(d.notna(), (d <= km).astype(float), np.nan)
    if args.proxy_col not in out.columns:
        fallback = None
        for c in ["maasai_pastoral_composite_proxy", "pastoral_reference_50km", "treated_area", "treat_50km"]:
            if c in out.columns:
                fallback = c
                break
        if fallback:
            out[args.proxy_col] = to_binary(out[fallback])
    return out


def make_samples(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    samples = {"full": df.copy()}
    regions = {x.strip().lower() for x in args.northern_regions.split(",") if x.strip()}
    if regions and "region2022_name" in df.columns:
        reg = df["region2022_name"].astype(str).str.strip().str.lower()
        samples["northern_regions"] = df.loc[reg.isin(regions)].copy()
    if regions and {"region2022_name", "distance_to_maasai_reference_km"}.issubset(df.columns):
        reg = df["region2022_name"].astype(str).str.strip().str.lower()
        dist = to_float(df["distance_to_maasai_reference_km"])
        samples[f"northern_or_within{int(args.risk_km)}km_pastoral_ref"] = df.loc[
            reg.isin(regions) | (dist <= args.risk_km)
        ].copy()
    return samples


def prepare_dynamic_terms(df: pd.DataFrame, args: argparse.Namespace, model_kind: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return prepared df, RHS terms, reported dynamic terms."""
    d = df.copy()
    d["wave_num"] = to_float(d["wave"])
    waves = sorted(int(w) for w in d["wave_num"].dropna().unique())
    if args.base_wave not in waves:
        raise ValueError(f"base_wave={args.base_wave} not present. Available waves: {waves}")

    d["game_pretrend"] = to_binary(d[args.game_col])
    terms: list[str] = []
    report_terms: list[str] = []

    if model_kind == "game_dynamic":
        terms.append("game_pretrend")
        for w in waves:
            if w == args.base_wave:
                continue
            col = f"game_wave_{w}"
            d[col] = d["game_pretrend"] * (d["wave_num"] == w).astype(float)
            terms.append(col)
            report_terms.append(col)
        return d, terms, report_terms

    if model_kind == "composite_triple_dynamic":
        d["proxy_pretrend"] = to_binary(d[args.proxy_col])
        d["game_proxy_pretrend"] = d["game_pretrend"] * d["proxy_pretrend"]
        terms += ["game_pretrend", "proxy_pretrend", "game_proxy_pretrend"]
        for w in waves:
            if w == args.base_wave:
                continue
            wd = (d["wave_num"] == w).astype(float)
            g = f"game_wave_{w}"
            p = f"proxy_wave_{w}"
            gp = f"game_proxy_wave_{w}"
            d[g] = d["game_pretrend"] * wd
            d[p] = d["proxy_pretrend"] * wd
            d[gp] = d["game_proxy_pretrend"] * wd
            terms += [g, p, gp]
            report_terms.append(gp)
        return d, terms, report_terms

    raise ValueError(model_kind)


def fit_dynamic(df: pd.DataFrame, outcome: str, terms: list[str], report_terms: list[str],
                controls: list[str], sample: str, model_kind: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    out_rows: list[dict] = []
    meta = {
        "sample": sample,
        "model": model_kind,
        "outcome": outcome,
        "event_wave": args.event_wave,
        "base_wave": args.base_wave,
        "game_col": args.game_col,
        "proxy_col": args.proxy_col if model_kind == "composite_triple_dynamic" else "",
    }

    missing = [c for c in [outcome] + terms if c not in df.columns]
    if missing:
        return [{**meta, "status": "missing_columns:" + ",".join(missing)}], {**meta, "status": "missing_columns"}

    controls_e = existing(df, controls)
    needed = [outcome] + terms + controls_e + ["wave"]
    if args.fe_level != "none" and args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    if args.weight_col and args.weight_col in df.columns:
        needed.append(args.weight_col)
    needed = list(dict.fromkeys(needed))
    d = df.dropna(subset=needed).copy()

    if len(d) < args.min_n:
        return [{**meta, "status": "too_few_obs", "n": int(len(d))}], {**meta, "status": "too_few_obs", "n": int(len(d))}

    rhs = list(terms) + controls_e + ["C(wave)"]
    if args.fe_level != "none" and args.fe_level in d.columns:
        rhs.append(f"C({args.fe_level})")
    formula = f"{outcome} ~ " + " + ".join(rhs)

    try:
        if args.weight_col and args.weight_col in d.columns:
            w = to_float(d[args.weight_col]).clip(lower=0)
            mod = smf.wls(formula, data=d, weights=w.fillna(w.mean()))
            weight_type = args.weight_col
        else:
            mod = smf.ols(formula, data=d)
            weight_type = "none"
        if args.cluster in d.columns and d[args.cluster].nunique(dropna=True) >= 2:
            fit = mod.fit(cov_type="cluster", cov_kwds={"groups": d[args.cluster]})
            se_type = f"cluster:{args.cluster}"
        else:
            fit = mod.fit(cov_type="HC1")
            se_type = "HC1"
    except Exception as e:
        return [{**meta, "status": "fit_error", "error": repr(e), "formula": formula}], {**meta, "status": "fit_error", "error": repr(e)}

    for term in report_terms:
        wave = int(term.rsplit("_", 1)[1])
        out_rows.append({
            **meta,
            "status": "ok",
            "reported_term": term,
            "reported_wave": wave,
            "relative_to_wave": args.base_wave,
            "period_type": "lead" if wave < args.event_wave else "post_or_event",
            "coef": float(fit.params.get(term, np.nan)),
            "se": float(fit.bse.get(term, np.nan)),
            "p_value": float(fit.pvalues.get(term, np.nan)),
            "n": int(len(d)),
            "n_clusters": int(d[args.cluster].nunique(dropna=True)) if args.cluster in d.columns else np.nan,
            "controls": ",".join(controls_e) if controls_e else "none",
            "fe_level": args.fe_level,
            "se_type": se_type,
            "weight_type": weight_type,
            "formula": formula,
        })

    # Joint lead test where possible.
    lead_terms = [t for t in report_terms if int(t.rsplit("_", 1)[1]) < args.event_wave]
    joint = {**meta, "n": int(len(d)), "n_lead_terms": len(lead_terms), "lead_terms": ",".join(lead_terms)}
    if not lead_terms:
        joint.update({
            "status": "no_true_pretrend_test",
            "note": "No non-base pre-event waves are available. For post_2009 with NPS waves 1-4, wave 1 is the only pre-period."
        })
    else:
        try:
            constraint = ", ".join([f"{t} = 0" for t in lead_terms])
            wt = fit.wald_test(constraint, scalar=True)
            joint.update({
                "status": "ok",
                "wald_stat": float(wt.statistic),
                "p_value": float(wt.pvalue),
                "df_num": int(getattr(wt, "df_num", len(lead_terms))),
                "constraint": constraint,
            })
        except Exception as e:
            joint.update({"status": "wald_error", "error": repr(e)})
    return out_rows, joint


def baseline_balance(df: pd.DataFrame, args: argparse.Namespace, controls: list[str], outcomes: list[str]) -> pd.DataFrame:
    d = df[pd.to_numeric(df.get("wave"), errors="coerce").eq(args.base_wave)].copy()
    if d.empty:
        return pd.DataFrame([{"status": "no_base_wave"}])
    d["game_bal"] = to_binary(d[args.game_col]) if args.game_col in d.columns else np.nan
    if args.proxy_col in d.columns:
        d["proxy_bal"] = to_binary(d[args.proxy_col])
        d["target_bal"] = d["game_bal"] * d["proxy_bal"]
    else:
        d["target_bal"] = d["game_bal"]
    variables = existing(d, list(dict.fromkeys(outcomes + controls + DEFAULT_BALANCE)))
    rows = []
    for v in variables:
        x = to_float(d[v])
        t = d["target_bal"]
        treated = x[t.eq(1)]
        control = x[t.eq(0)]
        mean_t = treated.mean()
        mean_c = control.mean()
        sd_pool = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        rows.append({
            "variable": v,
            "base_wave": args.base_wave,
            "target_definition": f"{args.game_col} x {args.proxy_col}" if args.proxy_col in d.columns else args.game_col,
            "n_target": int(t.eq(1).sum()),
            "n_control": int(t.eq(0).sum()),
            "mean_target": mean_t,
            "mean_control": mean_c,
            "diff": mean_t - mean_c,
            "std_diff": (mean_t - mean_c) / sd_pool if pd.notna(sd_pool) and sd_pool > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = add_missing_exposure_aliases(read_input(args), args)
    outcomes = split_csv(args.outcomes)
    controls = split_csv(args.controls)

    all_rows: list[dict] = []
    joint_rows: list[dict] = []
    samples = make_samples(df, args)
    for sample_name, sdf in samples.items():
        for model_kind in ["game_dynamic", "composite_triple_dynamic"]:
            if model_kind == "composite_triple_dynamic" and args.proxy_col not in sdf.columns:
                continue
            try:
                prepared, terms, report_terms = prepare_dynamic_terms(sdf, args, model_kind)
            except Exception as e:
                all_rows.append({"sample": sample_name, "model": model_kind, "status": "prepare_error", "error": repr(e)})
                continue
            for outcome in outcomes:
                rows, joint = fit_dynamic(prepared, outcome, terms, report_terms, controls, sample_name, model_kind, args)
                all_rows.extend(rows)
                joint_rows.append(joint)

    event_df = add_bh_q_values(pd.DataFrame(all_rows))
    joint_df = add_bh_q_values(pd.DataFrame(joint_rows))
    bal_df = baseline_balance(df, args, controls, outcomes)

    event_df.to_csv(args.output_dir / "table_pretrend_dynamic_event_study.csv", index=False)
    joint_df.to_csv(args.output_dir / "table_pretrend_joint_lead_tests.csv", index=False)
    bal_df.to_csv(args.output_dir / "table_pretrend_baseline_balance.csv", index=False)

    note_lines = [
        "Pre-trend diagnostics completed.",
        f"Input source: {df['_source_file'].iloc[0] if '_source_file' in df.columns and len(df) else 'direct'}",
        f"Event wave: {args.event_wave}; base wave: {args.base_wave}",
        f"Game exposure: {args.game_col}",
        f"Proxy: {args.proxy_col}",
        "",
        "Interpretation:",
        "- table_pretrend_dynamic_event_study.csv reports wave-specific interactions relative to the base wave.",
        "- table_pretrend_joint_lead_tests.csv tests equality of pre-event lead coefficients to zero only if at least one non-base pre-event wave exists.",
        "- table_pretrend_baseline_balance.csv compares baseline-wave characteristics for the eventual game × proxy target group.",
        "",
    ]
    if args.event_wave <= args.base_wave + 1:
        note_lines += [
            "WARNING: With event_wave <= base_wave+1 there is no conventional pre-trend test.",
            "For post_2009 in NPS waves 1-4, wave 1 is the only pre-period, so parallel trends cannot be directly tested.",
            "Use the dynamic coefficients as post-treatment dynamics and the baseline balance table as indirect support.",
            "For a placebo diagnostic, rerun with --event-wave 4 to treat waves 1-3 as pre-periods.",
        ]
    (args.output_dir / "pretrend_notes.txt").write_text("\n".join(note_lines) + "\n")
    print(f"Saved pretrend outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
