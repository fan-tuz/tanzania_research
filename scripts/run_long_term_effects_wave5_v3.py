#!/usr/bin/env python3
"""
run_long_term_effects_wave5_v3.py

Long-term (2020/21, NPS Wave 5) follow-up analysis — v2 fixed Wave4 ID link for game-reserve ×
Maasai/pastoral policy exposure.

Design logic
------------
Wave 5 can be used to study persistence/long-run associations by linking 2020/21
households back to their Wave 4 household IDs and inheriting spatial exposure
from the already-spatialized Wave 4 analysis dataset. This avoids re-computing
spatial proxies from Wave 5 when household GPS/public geovariables are not as
harmonized as waves 1-4.

The script estimates:
  1) Wave-5 cross-section: Y_2020 ~ game + proxy + game×proxy + controls + FE
  2) Y4→Y5 changes: ΔY = Y_2020 - Y_2014 ~ game + proxy + game×proxy + controls + FE
  3) ANCOVA: Y_2020 ~ Y_2014 + game + proxy + game×proxy + controls + FE
  4) Wave-5 mechanisms: livestock stock, TLU, herd composition, shocks.

Interpretation warning
----------------------
This is a long-run persistence/heterogeneity design, not a new pre/post causal
DID by itself. The causal argument still rests on the exogeneity of the 2009
policy-targeted Maasai/pastoral villages and the validity of comparable control
areas. Use this as evidence of long-term persistence and mechanisms.

Typical run:
python scripts/run_long_term_effects_wave5_v3.py \
  --wave5-dir data/wave5 \
  --baseline-spatial-dir output/spatial_restricted \
  --output-dir output/lt_results_wave5 \
  --game-col within_25km_game_reserve \
  --proxy-col maasai_pastoral_composite_proxy \
  --cluster ward2022_name \
  --fe-level district2022_name \
  --weight-col weight
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import clean_id_series, existing, first_existing, find_file, read_csv_optional, safe_log, split_csv, to_binary, to_num, add_bh_q_values

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CORE_CANDIDATES = ["analysis_core_1_5_spatial.csv", "analysis_core_1_4_spatial.csv"]
LAND_CANDIDATES = ["analysis_land_2_5_spatial.csv", "analysis_land_2_4_spatial.csv"]
PASTORAL_CANDIDATES = ["analysis_pastoral_3_5_spatial.csv", "analysis_pastoral_3_4_spatial.csv"]

WELFARE_OUTCOMES = ["log_cons_pc", "log_cons_ae", "cons_food_share_real"]
MECHANISM_OUTCOMES = [
    "log1p_livestock_total_current",
    "livestock_tlu_default",
    "livestock_tlu_per_ae_default",
    "livestock_cattle_share",
    "livestock_small_ruminants_share",
    "n_shocks_occurred",
]
CONTROLS = ["hh_size", "head_age", "share_female", "n_children_u15", "n_adults_15plus"]

TLU_WEIGHTS_BY_GROUP = {
    "cattle": 0.70,
    "camel": 1.00,
    "donkey_horse": 0.50,
    "goat": 0.10,
    "sheep": 0.10,
    "pig": 0.20,
    "poultry": 0.01,
    "rabbit": 0.01,
}
DEFAULT_LIVESTOCK_CODE_GROUP = {
    1: "cattle", 2: "cattle", 3: "cattle",
    4: "goat", 5: "sheep", 6: "pig",
    7: "poultry", 8: "poultry",
    9: "donkey_horse", 10: "camel",
    11: "rabbit",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wave5-dir", type=Path, required=True)
    p.add_argument("--baseline-spatial-dir", type=Path, required=True,
                   help="Directory with waves 1-4 spatial/restricted analysis files.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--game-col", default="within_25km_game_reserve")
    p.add_argument("--proxy-col", default="maasai_pastoral_composite_proxy")
    p.add_argument("--cluster", default="ward2022_name")
    p.add_argument("--fe-level", default="district2022_name")
    p.add_argument("--weight-col", default="weight",
                   help="Use weight column in wave5 output; set none to disable.")
    p.add_argument("--outcomes", default=",".join(WELFARE_OUTCOMES))
    p.add_argument("--mechanisms", default=",".join(MECHANISM_OUTCOMES))
    p.add_argument("--controls", default=",".join(CONTROLS))
    p.add_argument("--min-n", type=int, default=80)
    p.add_argument("--restrict-panel", action="store_true", default=True,
                   help="Keep only households with a y4_hhid link. Default true.")
    p.add_argument("--include-booster", action="store_true", default=False,
                   help="Include booster households if they can be spatially linked; normally false for LT panel design.")
    return p.parse_args()






def norm_name(x: str) -> str:
    return str(x).strip().lower()
















def infer_livestock_group(code, name=None) -> str:
    txt = str(name).lower() if name is not None and not pd.isna(name) else ""
    if any(k in txt for k in ["cattle", "cow", "bull", "calf", "ng'ombe", "ngombe"]):
        return "cattle"
    if "camel" in txt:
        return "camel"
    if any(k in txt for k in ["donkey", "horse", "mule"]):
        return "donkey_horse"
    if any(k in txt for k in ["goat", "mbuzi"]):
        return "goat"
    if any(k in txt for k in ["sheep", "kondoo"]):
        return "sheep"
    if "pig" in txt:
        return "pig"
    if any(k in txt for k in ["chicken", "poultry", "duck", "turkey", "bird"]):
        return "poultry"
    if "rabbit" in txt:
        return "rabbit"
    try:
        return DEFAULT_LIVESTOCK_CODE_GROUP.get(int(float(code)), "other")
    except Exception:
        return "other"


def current_livestock_count(df: pd.DataFrame) -> pd.Series:
    if "lf02_04_1" in df.columns or "lf02_04_2" in df.columns:
        return to_num(df.get("lf02_04_1", 0)).fillna(0) + to_num(df.get("lf02_04_2", 0)).fillna(0)
    return pd.Series(np.nan, index=df.index)


def build_wave5_household(wave5_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    report: list[dict] = []

    # Core / tracking / weights.
    core, path = read_csv_optional(wave5_dir, ["hh_sec_a.csv", "HH_SEC_A.csv"])
    if core.empty:
        raise FileNotFoundError("Wave 5 hh_sec_a.csv not found")
    report.append({"module": "core", "file": path.name if path else "", "rows": len(core), "columns": core.shape[1]})
    core = core.copy()
    core["y5_hhid"] = clean_id_series(core["y5_hhid"])
    if "y4_hhid" in core.columns:
        core["y4_hhid"] = clean_id_series(core["y4_hhid"])

    out = pd.DataFrame({
        "hhid": core["y5_hhid"],
        "hhid_wave": core["y5_hhid"],
        "y5_hhid": core["y5_hhid"],
        "y4_hhid": core["y4_hhid"] if "y4_hhid" in core.columns else np.nan,
        "wave": 5,
        "survey_year": 2020,
    })
    for new, candidates in {
        "weight": ["y5_panelweight", "y5_crossweight"],
        "cross_weight": ["y5_crossweight"],
        "panel_weight": ["y5_panelweight"],
        "rural": ["y5_rural"],
        "clusterid": ["clusterid", "y5_cluster"],
        "strataid": ["strataid"],
        "domain": ["domain"],
        "booster": ["booster"],
    }.items():
        c = first_existing(core.columns, candidates)
        if c:
            out[new] = core[c]

    # Consumption.
    cons, path = read_csv_optional(wave5_dir, ["consumption_real_y5.csv", "Consumption_real_y5.csv", "consumptionNPS5.csv"])
    if not cons.empty:
        report.append({"module": "consumption", "file": path.name if path else "", "rows": len(cons), "columns": cons.shape[1]})
        cons = cons.copy()
        cons["y5_hhid"] = clean_id_series(cons["y5_hhid"])
        for c in cons.columns:
            if c != "y5_hhid":
                cons[c] = to_num(cons[c])
        tmp = cons[["y5_hhid"]].copy()
        if "hhsize" in cons.columns:
            tmp["cons_hhsize"] = cons["hhsize"]
        if "adulteq" in cons.columns:
            tmp["cons_adulteq"] = cons["adulteq"]
        # expmR_pae is per-adult-equivalent real total expenditure in the Wave 5 aggregate.
        if "expmR_pae" in cons.columns:
            tmp["cons_total_monthly_real_ae"] = cons["expmR_pae"]
            tmp["log_cons_ae"] = safe_log(cons["expmR_pae"])
            if {"adulteq", "hhsize"}.issubset(cons.columns):
                total_real = cons["expmR_pae"] * cons["adulteq"]
                tmp["cons_total_monthly_real"] = total_real
                tmp["cons_total_monthly_real_pc"] = total_real / cons["hhsize"].replace(0, np.nan)
                tmp["log_cons_pc"] = safe_log(tmp["cons_total_monthly_real_pc"])
        # Food share from per-AE real food components when available.
        food_cols = [c for c in ["foodINR_pae", "foodOUTR_pae"] if c in cons.columns]
        if food_cols and "expmR_pae" in cons.columns:
            tmp["cons_food_share_real"] = cons[food_cols].sum(axis=1, min_count=1) / cons["expmR_pae"].replace(0, np.nan)
        for wcol in ["hhweight_pan", "hhweight", "hhweight2"]:
            if wcol in cons.columns and "weight" not in tmp.columns:
                tmp["weight"] = cons[wcol]
        out = out.merge(tmp.drop_duplicates("y5_hhid"), on="y5_hhid", how="left", suffixes=("", "_cons"))
        if "weight_cons" in out.columns:
            out["weight"] = out.get("weight").combine_first(out["weight_cons"])
            out = out.drop(columns=["weight_cons"])

    # Demographics.
    hh_b, path = read_csv_optional(wave5_dir, ["hh_sec_b.csv", "HH_SEC_B.csv"])
    if not hh_b.empty:
        report.append({"module": "demographics", "file": path.name if path else "", "rows": len(hh_b), "columns": hh_b.shape[1]})
        hh_b = hh_b.copy()
        hh_b["y5_hhid"] = clean_id_series(hh_b["y5_hhid"])
        # Wave 5 roster variable names differ from earlier waves. In NPS5,
        # hh_b04 is the most reliable age-in-years field in the user's header dump,
        # while hh_b05 is usually relationship-to-head. Earlier draft code used
        # hh_b03_1 as age and hh_b04 as sex, which left age missing and treated
        # age as sex. We therefore select age/sex/relation by simple sanity checks.
        def pick_age_col(frame: pd.DataFrame) -> Optional[str]:
            candidates = ["hh_b04", "hh_b03_1", "hh_b03_2", "age", "age_years"]
            best = None
            best_valid = -1
            for cand in candidates:
                if cand not in frame.columns:
                    continue
                x = to_num(frame[cand])
                valid = x.between(0, 120).sum()
                if valid > best_valid and valid > 0:
                    best = cand
                    best_valid = int(valid)
            return best

        def pick_relation_col(frame: pd.DataFrame) -> Optional[str]:
            candidates = ["hh_b05", "hh_b06", "hh_b07", "relation"]
            n_hh = frame["y5_hhid"].nunique(dropna=True)
            best = None
            best_score = -1
            for cand in candidates:
                if cand not in frame.columns:
                    continue
                x = to_num(frame[cand])
                # Relationship-to-head should identify roughly one head per household.
                head_hh = frame.loc[x.eq(1), "y5_hhid"].nunique(dropna=True)
                score = head_hh / n_hh if n_hh else 0
                if 0.50 <= score <= 1.20 and score > best_score:
                    best = cand
                    best_score = score
            return best

        def pick_sex_col(frame: pd.DataFrame, used: set[str]) -> tuple[Optional[str], Optional[int]]:
            candidates = ["hh_b06", "hh_b05", "hh_b04", "sex", "gender"]
            best = None
            best_female_code = None
            best_score = -1
            for cand in candidates:
                if cand not in frame.columns or cand in used:
                    continue
                x = to_num(frame[cand])
                vals = set(x.dropna().astype(int).unique().tolist()) if x.notna().any() else set()
                if not vals.issubset({1, 2}) or len(vals) < 2:
                    continue
                share1 = float((x == 1).mean())
                share2 = float((x == 2).mean())
                # Choose the code whose overall share is most plausibly female.
                for code, sh in [(1, share1), (2, share2)]:
                    score = 1.0 - abs(sh - 0.50)
                    if 0.25 <= sh <= 0.75 and score > best_score:
                        best = cand
                        best_female_code = code
                        best_score = score
            return best, best_female_code

        age = pick_age_col(hh_b)
        rel = pick_relation_col(hh_b)
        sex, female_code = pick_sex_col(hh_b, used={c for c in [age, rel] if c})
        ind = "indidy5" if "indidy5" in hh_b.columns else None
        if ind is None:
            hh_b["_person"] = np.arange(len(hh_b))
            ind = "_person"
        demo = hh_b.groupby("y5_hhid", as_index=False).agg(hh_size=(ind, "count"))

        if age:
            hh_b[age] = to_num(hh_b[age])
            demo = demo.merge(hh_b.groupby("y5_hhid", as_index=False).agg(
                mean_age=(age, "mean"),
                max_age=(age, "max"),
                n_children_u15=(age, lambda x: int((x < 15).sum())),
                n_adults_15plus=(age, lambda x: int((x >= 15).sum())),
            ), on="y5_hhid", how="left")

        if sex and female_code is not None:
            hh_b[sex] = to_num(hh_b[sex])
            demo = demo.merge(hh_b.groupby("y5_hhid", as_index=False).agg(
                n_female=(sex, lambda x: int((x == female_code).sum())),
                share_female=(sex, lambda x: np.nan if x.notna().sum() == 0 else float((x == female_code).mean())),
            ), on="y5_hhid", how="left")

        if rel and age:
            hh_b[rel] = to_num(hh_b[rel])
            head_cols = ["y5_hhid", age] + ([sex] if sex else [])
            heads = hh_b.loc[hh_b[rel].eq(1), head_cols].drop_duplicates("y5_hhid")
            ren = {age: "head_age"}
            if sex:
                ren[sex] = "head_sex"
            demo = demo.merge(heads.rename(columns=ren), on="y5_hhid", how="left")

        # Store selected roster mapping for diagnostics/debugging.
        report.append({
            "module": "demographics_mapping",
            "file": path.name if path else "",
            "rows": 0,
            "columns": 0,
            "age_col": age or "",
            "relation_col": rel or "",
            "sex_col": sex or "",
            "female_code": female_code if female_code is not None else "",
        })
        out = out.merge(demo, on="y5_hhid", how="left", suffixes=("", "_demo"))
        if "hh_size_demo" in out.columns:
            out["hh_size"] = out["hh_size"].combine_first(out["hh_size_demo"]) if "hh_size" in out.columns else out["hh_size_demo"]
            out = out.drop(columns=["hh_size_demo"])

    # Shocks.
    shocks, path = read_csv_optional(wave5_dir, ["hh_sec_r.csv", "HH_SEC_R.csv"])
    if not shocks.empty:
        report.append({"module": "shocks", "file": path.name if path else "", "rows": len(shocks), "columns": shocks.shape[1]})
        shocks = shocks.copy()
        shocks["y5_hhid"] = clean_id_series(shocks["y5_hhid"])
        shocks["shock_occurred"] = to_binary(shocks["hh_r01"]) if "hh_r01" in shocks.columns else np.nan
        sid = "shockid" if "shockid" in shocks.columns else "shock_occurred"
        sh = shocks.groupby("y5_hhid", as_index=False).agg(
            n_shock_records=(sid, "count"),
            n_shocks_occurred=("shock_occurred", "sum"),
        )
        out = out.merge(sh, on="y5_hhid", how="left")

    # Livestock stock and TLU.
    lf, path = read_csv_optional(wave5_dir, ["lf_sec_02.csv", "LF_SEC_02.csv"])
    if not lf.empty:
        report.append({"module": "livestock", "file": path.name if path else "", "rows": len(lf), "columns": lf.shape[1]})
        lf = lf.copy()
        lf["y5_hhid"] = clean_id_series(lf["y5_hhid"])
        lf["_lv_current"] = current_livestock_count(lf)
        lf["_lv_group"] = [infer_livestock_group(c) for c in lf.get("lvstckid", pd.Series(np.nan, index=lf.index))]
        lf["_tlu_default"] = lf["_lv_current"] * lf["_lv_group"].map(TLU_WEIGHTS_BY_GROUP).fillna(0.0)
        lv = lf.groupby("y5_hhid", as_index=False).agg(
            n_livestock_records=("lvstckid", "count") if "lvstckid" in lf.columns else ("_lv_current", "count"),
            livestock_total_current=("_lv_current", "sum"),
            livestock_tlu_default=("_tlu_default", "sum"),
        )
        if "lf02_03" in lf.columns:
            lv = lv.merge(lf.groupby("y5_hhid", as_index=False).agg(livestock_total_12m_ago=("lf02_03", "sum")), on="y5_hhid", how="left")
            lv["livestock_change_12m"] = lv["livestock_total_current"] - lv["livestock_total_12m_ago"]
            lv["livestock_decline_12m"] = np.where(lv["livestock_change_12m"].notna(), (lv["livestock_change_12m"] < 0).astype(float), np.nan)
        if "lf02_26" in lf.columns:
            lv = lv.merge(lf.groupby("y5_hhid", as_index=False).agg(livestock_sales_value=("lf02_26", "sum")), on="y5_hhid", how="left")
        comp = lf.pivot_table(index="y5_hhid", columns="_lv_group", values="_lv_current", aggfunc="sum").reset_index()
        comp = comp.rename(columns={c: f"livestock_{c}_current" for c in comp.columns if c != "y5_hhid"})
        lv = lv.merge(comp, on="y5_hhid", how="left")
        if "livestock_cattle_current" in lv.columns:
            lv["livestock_cattle_share"] = lv["livestock_cattle_current"] / lv["livestock_total_current"].replace(0, np.nan)
        goat = lv["livestock_goat_current"] if "livestock_goat_current" in lv.columns else 0
        sheep = lv["livestock_sheep_current"] if "livestock_sheep_current" in lv.columns else 0
        if isinstance(goat, pd.Series) or isinstance(sheep, pd.Series):
            lv["livestock_small_ruminants_current"] = (goat if isinstance(goat, pd.Series) else 0).fillna(0) + (sheep if isinstance(sheep, pd.Series) else 0).fillna(0)
            lv["livestock_small_ruminants_share"] = lv["livestock_small_ruminants_current"] / lv["livestock_total_current"].replace(0, np.nan)
        out = out.merge(lv, on="y5_hhid", how="left")
        out["log1p_livestock_total_current"] = np.log1p(to_num(out["livestock_total_current"]).where(to_num(out["livestock_total_current"]) >= 0))
        if "livestock_sales_value" in out.columns:
            out["log1p_livestock_sales_value"] = np.log1p(to_num(out["livestock_sales_value"]).where(to_num(out["livestock_sales_value"]) >= 0))

    # Derived ratios.
    ae = out.get("cons_adulteq", out.get("hh_size"))
    if "livestock_tlu_default" in out.columns and ae is not None:
        out["livestock_tlu_per_ae_default"] = to_num(out["livestock_tlu_default"]) / to_num(ae).replace(0, np.nan)
    return out, report


def read_baseline_spatial(baseline_dir: Path) -> pd.DataFrame:
    blocks = []
    link_reports = []
    for candidates in [CORE_CANDIDATES, LAND_CANDIDATES, PASTORAL_CANDIDATES]:
        p = None
        for c in candidates:
            q = baseline_dir / c
            if q.exists():
                p = q
                break
        if not p:
            continue
        df = pd.read_csv(p, low_memory=False)
        if "wave" in df.columns:
            df = df[pd.to_numeric(df["wave"], errors="coerce").eq(4)].copy()
        if df.empty:
            continue

        # IMPORTANT: in the waves 1-4 analysis file, `hhid` can be a harmonized/global
        # ID that does not match Wave 5's `y4_hhid`.  Diagnostics on the user's data
        # showed zero matches on `hhid` but ~3,042 matches on `hhid_wave` and
        # `original_hhid`.  Therefore prefer wave-specific IDs for the bridge.
        preferred_ids = ["y4_hhid", "hhid_wave", "original_hhid", "hhid"]
        id_candidates = [c for c in preferred_ids if c in df.columns]
        if not id_candidates:
            continue

        # Create y4_hhid_bridge from the first available preferred ID. Do NOT start
        # from `hhid` and combine_first, because a non-matching non-missing `hhid`
        # would block the matching `hhid_wave` from ever being used.
        df["y4_hhid_bridge"] = clean_id_series(df[id_candidates[0]])
        for c in id_candidates[1:]:
            df["y4_hhid_bridge"] = df["y4_hhid_bridge"].combine_first(clean_id_series(df[c]))
        link_reports.append({
            "file": p.name,
            "bridge_source_first": id_candidates[0],
            "candidate_ids": ",".join(id_candidates),
            "rows_wave4": int(len(df)),
            "unique_bridge_ids": int(df["y4_hhid_bridge"].nunique(dropna=True)),
        })
        keep_cols = ["y4_hhid_bridge"]
        useful_patterns = [
            r"^log_cons_", r"^cons_", r"^within_.*game_reserve", r"^distance_to_.*game_reserve",
            r"^inside_.*reserve", r"^nearest_.*reserve", r"maasai", r"pastoral", r"^treat_", r"treated_area",
            r"ward2022", r"district2022", r"region2022", r"council2022", r"clusterid", r"strataid",
            r"grazing_land", r"cultivation_villagers", r"forest_pct", r"wetland_pct", r"village_land_certificate",
            r"^geo_", r"livestock_", r"n_shocks_occurred",
            r"hh_size", r"head_age", r"share_female", r"n_children", r"n_adults",
        ]
        for c in df.columns:
            if c in keep_cols:
                continue
            if any(re.search(pat, c) for pat in useful_patterns):
                keep_cols.append(c)
        block = df[keep_cols].drop_duplicates("y4_hhid_bridge")
        blocks.append(block)
    if not blocks:
        raise FileNotFoundError(f"No wave-4 spatial baseline files found in {baseline_dir}")
    out = blocks[0]
    for b in blocks[1:]:
        out = out.merge(b, on="y4_hhid_bridge", how="outer", suffixes=("", "__dup"))
        # Combine duplicates.
        for c in list(out.columns):
            if c.endswith("__dup"):
                base = c[:-5]
                if base in out.columns:
                    out[base] = out[base].combine_first(out[c])
                else:
                    out[base] = out[c]
                out = out.drop(columns=[c])
    # Prefix y4 outcome/mechanism columns to allow change and ANCOVA.
    rename = {}
    for c in out.columns:
        if c == "y4_hhid_bridge":
            continue
        if c in WELFARE_OUTCOMES or c in MECHANISM_OUTCOMES or c in ["livestock_total_current", "livestock_tlu_default", "n_shocks_occurred"]:
            rename[c] = f"y4_{c}"
    return out.rename(columns=rename)


def attach_baseline_exposure(y5: pd.DataFrame, y4: pd.DataFrame) -> pd.DataFrame:
    out = y5.copy()
    out["y4_hhid_bridge"] = clean_id_series(out["y4_hhid"])
    merged = out.merge(y4, on="y4_hhid_bridge", how="left", suffixes=("", "_y4base"))
    # If controls missing in y5, use wave-4 values as backup with explicit names too.
    for c in CONTROLS:
        yc = f"y4_{c}"
        if c not in merged.columns and yc in merged.columns:
            merged[c] = merged[yc]
        elif c in merged.columns and yc in merged.columns:
            merged[c] = merged[c].combine_first(merged[yc])
    return merged


def ensure_treatment_aliases(df: pd.DataFrame, game_col: str, proxy_col: str) -> pd.DataFrame:
    out = df.copy()
    if game_col not in out.columns and "distance_to_game_reserve_km" in out.columns:
        if game_col.startswith("within_") and "km_game_reserve" in game_col:
            km = float(game_col.split("within_", 1)[1].split("km", 1)[0])
            d = to_num(out["distance_to_game_reserve_km"])
            out[game_col] = np.where(d.notna(), (d <= km).astype(float), np.nan)
    if proxy_col not in out.columns:
        for c in ["maasai_pastoral_composite_proxy", "pastoral_reference_50km", "treated_area", "treat_50km"]:
            if c in out.columns:
                out[proxy_col] = to_binary(out[c])
                break
    if game_col in out.columns:
        out["lt_game"] = to_binary(out[game_col])
    if proxy_col in out.columns:
        out["lt_proxy"] = to_binary(out[proxy_col])
    if {"lt_game", "lt_proxy"}.issubset(out.columns):
        out["lt_game_x_proxy"] = out["lt_game"] * out["lt_proxy"]
    return out


def fit_model(df: pd.DataFrame, outcome: str, terms: list[str], report: str, controls: list[str],
              model: str, args: argparse.Namespace, extra_terms: Optional[list[str]] = None) -> dict:
    extra_terms = extra_terms or []
    row = {
        "model": model,
        "outcome": outcome,
        "reported_term": report,
        "game_col": args.game_col,
        "proxy_col": args.proxy_col,
    }
    missing = [c for c in [outcome] + terms + extra_terms if c not in df.columns]
    if missing:
        return {**row, "status": "missing_columns:" + ",".join(missing)}
    # Keep only controls that are actually usable in the estimation sample.
    # This prevents all-missing or constant controls from wiping out Wave 5 models.
    controls_e = []
    for c in existing(df, controls):
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= args.min_n and x.nunique(dropna=True) > 1:
            controls_e.append(c)
    dropped_controls = [c for c in existing(df, controls) if c not in controls_e]
    needed = [outcome] + terms + extra_terms + controls_e
    if args.fe_level != "none" and args.fe_level in df.columns:
        needed.append(args.fe_level)
    if args.cluster in df.columns:
        needed.append(args.cluster)
    weight_col = None if str(args.weight_col).lower() == "none" else args.weight_col
    if weight_col and weight_col in df.columns:
        needed.append(weight_col)
    d = df.dropna(subset=list(dict.fromkeys(needed))).copy()
    row["n"] = int(len(d))
    row["n_clusters"] = int(d[args.cluster].nunique(dropna=True)) if args.cluster in d.columns else np.nan
    row["controls"] = ",".join(controls_e) if controls_e else "none"
    row["dropped_controls"] = ",".join(dropped_controls) if dropped_controls else ""
    row["cell_n_game_proxy"] = int(((d.get("lt_game") == 1) & (d.get("lt_proxy") == 1)).sum()) if {"lt_game", "lt_proxy"}.issubset(d.columns) else np.nan
    if len(d) < args.min_n:
        return {**row, "status": "too_few_obs"}
    rhs = list(terms) + extra_terms + controls_e
    if args.fe_level != "none" and args.fe_level in d.columns:
        rhs.append(f"C({args.fe_level})")
    formula = f"{outcome} ~ " + " + ".join(rhs)
    try:
        if weight_col and weight_col in d.columns:
            w = to_num(d[weight_col]).clip(lower=0)
            mod = smf.wls(formula, data=d, weights=w.fillna(w.mean()))
            weight_type = weight_col
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
        return {**row, "status": "fit_error", "error": repr(e), "formula": formula}
    return {
        **row,
        "status": "ok",
        "coef": float(fit.params.get(report, np.nan)),
        "se": float(fit.bse.get(report, np.nan)),
        "p_value": float(fit.pvalues.get(report, np.nan)),
        "controls": ",".join(controls_e) if controls_e else "none",
        "dropped_controls": ",".join(dropped_controls) if dropped_controls else "",
        "fe_level": args.fe_level,
        "se_type": se_type,
        "weight_type": weight_type,
        "formula": formula,
    }


def add_change_variables(df: pd.DataFrame, outcomes: list[str] + list[str]) -> pd.DataFrame:
    out = df.copy()
    for y in outcomes:
        y4 = f"y4_{y}"
        if y in out.columns and y4 in out.columns:
            out[f"delta_{y}_y5_y4"] = to_num(out[y]) - to_num(out[y4])
    return out


def run_models(df: pd.DataFrame, outcomes: list[str], mechanisms: list[str], controls: list[str], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    terms = ["lt_game", "lt_proxy", "lt_game_x_proxy"]
    rows_cross = []
    rows_change = []
    rows_mech = []
    for y in outcomes:
        rows_cross.append(fit_model(df, y, terms, "lt_game_x_proxy", controls, "wave5_cross_section", args))
        y4 = f"y4_{y}"
        if y4 in df.columns:
            rows_cross.append(fit_model(df, y, terms, "lt_game_x_proxy", controls, "wave5_ancova_y4_control", args, extra_terms=[y4]))
        dy = f"delta_{y}_y5_y4"
        if dy in df.columns:
            rows_change.append(fit_model(df, dy, terms, "lt_game_x_proxy", controls, "delta_y5_minus_y4", args))
    for y in mechanisms:
        if y in df.columns:
            rows_mech.append(fit_model(df, y, terms, "lt_game_x_proxy", controls, "wave5_mechanism_cross_section", args))
            y4 = f"y4_{y}"
            if y4 in df.columns:
                rows_mech.append(fit_model(df, y, terms, "lt_game_x_proxy", controls, "wave5_mechanism_ancova_y4_control", args, extra_terms=[y4]))
                dy = f"delta_{y}_y5_y4"
                if dy in df.columns:
                    rows_mech.append(fit_model(df, dy, terms, "lt_game_x_proxy", controls, "mechanism_delta_y5_minus_y4", args))
    return {
        "lt_table_1_wave5_welfare_cross_section_ancova": pd.DataFrame(rows_cross),
        "lt_table_2_y4_y5_welfare_changes": pd.DataFrame(rows_change),
        "lt_table_3_wave5_mechanisms": pd.DataFrame(rows_mech),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = split_csv(args.outcomes)
    mechanisms = split_csv(args.mechanisms)
    controls = split_csv(args.controls)

    print("Building Wave 5 household file...")
    y5, report = build_wave5_household(args.wave5_dir)
    print(f"  Wave5 rows: {len(y5):,}")
    print("Reading Wave 4 spatial baseline exposures...")
    y4base = read_baseline_spatial(args.baseline_spatial_dir)
    print(f"  Wave4 baseline rows: {len(y4base):,}")
    df = attach_baseline_exposure(y5, y4base)
    if args.restrict_panel and not args.include_booster:
        df = df[df["y4_hhid"].notna()].copy()
    df = ensure_treatment_aliases(df, args.game_col, args.proxy_col)
    df = add_change_variables(df, outcomes + mechanisms)

    # Drop observations that cannot be linked to spatial exposure.
    linked = df["lt_game"].notna() if "lt_game" in df.columns else pd.Series(False, index=df.index)
    df_linked = df[linked].copy()

    df.to_csv(args.output_dir / "wave5_analysis_household_linked_attempt.csv", index=False)
    df_linked.to_csv(args.output_dir / "wave5_analysis_household_spatial_linked.csv", index=False)
    pd.DataFrame(report).to_csv(args.output_dir / "wave5_build_report.csv", index=False)

    tables = run_models(df_linked, outcomes, mechanisms, controls, args)
    for name, tab in tables.items():
        add_bh_q_values(tab).to_csv(args.output_dir / f"{name}.csv", index=False)

    # Diagnostics.
    diag = {
        "wave5_rows_total": int(len(df)),
        "wave5_rows_spatial_linked": int(len(df_linked)),
        "n_with_game": int((df_linked.get("lt_game") == 1).sum()) if "lt_game" in df_linked.columns else np.nan,
        "n_with_proxy": int((df_linked.get("lt_proxy") == 1).sum()) if "lt_proxy" in df_linked.columns else np.nan,
        "n_game_x_proxy": int(((df_linked.get("lt_game") == 1) & (df_linked.get("lt_proxy") == 1)).sum()) if {"lt_game", "lt_proxy"}.issubset(df_linked.columns) else np.nan,
        "game_col": args.game_col,
        "proxy_col": args.proxy_col,
        "cluster": args.cluster,
        "fe_level": args.fe_level,
        "weight_col": args.weight_col,
    }
    pd.DataFrame([diag]).to_csv(args.output_dir / "lt_wave5_linkage_diagnostics.csv", index=False)

    notes = [
        "Long-term Wave 5 analysis completed.",
        "",
        "Design:",
        "- Wave 5 outcomes are linked to Wave 4 spatial treatment/proxy variables through y4_hhid.",
        "- The main reported long-term term is lt_game_x_proxy = game-reserve exposure × composite Maasai/pastoral proxy.",
        "- Welfare tables include cross-section, ANCOVA with Y4 outcome control, and Y5-Y4 change specifications when Y4 is available.",
        "",
        "Interpretation:",
        "- Cross-section estimates are long-run associations conditional on controls and inherited spatial exposure.",
        "- Delta and ANCOVA specifications are stronger for persistence because they use the Y4→Y5 panel link.",
        "- This does not replace the post_2009 identification design; it tests persistence and mechanisms in 2020/21.",
        "",
        "Diagnostics:",
    ] + [f"- {k}: {v}" for k, v in diag.items()]
    (args.output_dir / "lt_wave5_notes.txt").write_text("\n".join(notes) + "\n")
    print(f"Saved Wave 5 LT outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
