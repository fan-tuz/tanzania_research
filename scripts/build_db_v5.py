#!/usr/bin/env python3
"""
Build harmonized Tanzania NPS analytical datasets for Waves 1-4.

Version 5:
- Adds stronger cross-wave harmonized aliases for consumption, land events, migration,
  shocks, livestock, and coordinates.
- Adds source comments with DDI/documentation page references at harmonization points.

Output unit of the main file:
    household observation x wave

Typical usage:
    python scripts/build_db_v5.py \
        --wave1 data/wave1 \
        --wave2 data/wave2 \
        --wave3 data/wave3 \
        --wave4 data/wave4 \
        --output output/refined_multiwave \
        --only-main

Main outputs:
    - y1_analysis_household.csv ... y4_analysis_household.csv
    - nps_analysis_household_wave.csv
    - nps_variable_inventory.csv
    - nps_column_presence_by_wave.csv
    - nps_build_report.csv

Design notes:
    - Waves 3-4 are highly comparable and use the LSMS-style module names.
    - Waves 1-2 use older naming conventions. This script therefore keeps a
      rich set of raw aggregated variables and creates only conservative aliases.
    - The main appended file is a repeated cross-section style dataset. Panel
      linking is intentionally not imposed here; use panel key files separately.
    - Yes/no coding assumes 1 = yes. Validate with the codebooks before final regressions.

Source notes used in harmonization comments below
-------------------------------------------------
Wave 1 DDI: ddi-documentation-english_microdata-76.pdf
  - p.3: sample design, nationally representative household survey.
  - p.4: household, agriculture, and community questionnaires.
  - p.5: fieldwork dates 2008-10 to 2009-09.
Wave 2 DDI: ddi-documentation-english_microdata-1050.pdf
  - p.3: panel follow-up, split-off tracking, sample size, response rates.
  - p.4: weights and panel-weight methodology.
  - p.5-6: household/agriculture/fisheries/community questionnaires.
  - p.17: community questionnaire sections CB-CD-CE etc.
  - p.66-67 approximately: geovariables and EA offset confidentiality strategy.
Wave 3 DDI: ddi-documentation-english_microdata-2252.pdf
  - p.3-4: sample, y3_weight, clusterid, strataid in HH_SEC_A.
  - p.14-15: AG_SEC_2A/2B plot roster and GPS variables.
  - p.54-55: COM_SEC_CD land use and land-event variables.
  - p.99-112: LF_SEC_03-08 livestock mechanisms, milk, dung, by-products.
Wave 4 DDI: ddi-documentation-english_microdata-2862.pdf
  - p.3: refresh sample, y4_weight, clusterid, strataid in HH_SEC_A.
  - p.14-15: AG_SEC_2A/2B plot roster and GPS variables.
  - p.50, p.314-316: COM_SEC_CD land use and land-event variables.
  - p.99-104: LF_SEC_03-07 livestock mechanisms, milk, animal power/dung.
Header dumps supplied by user:
  - wave1/wave2 headers: Pasted text(9).txt / Pasted text (2).txt.
  - wave3 headers: Pasted text(7).txt.
  - wave4 headers: Pasted text(5).txt.

"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from utils import first_existing, find_file, yes1


# ============================================================
# Configuration
# ============================================================

@dataclass
class WaveConfig:
    wave: int
    survey_year: int
    hhid_col: str
    indid_col: str | None
    core_files: list[str]
    weight_candidates: list[str]
    rural_candidates: list[str]
    cluster_candidates: list[str]
    strata_candidates: list[str]
    region_candidates: list[str]
    district_candidates: list[str]
    ward_candidates: list[str]
    community_id_from_core: list[str]
    demographics_files: list[str] = field(default_factory=list)
    age_candidates: list[str] = field(default_factory=list)
    sex_candidates: list[str] = field(default_factory=list)
    relation_candidates: list[str] = field(default_factory=list)
    labour_files: list[str] = field(default_factory=list)
    housing_files: list[str] = field(default_factory=list)
    consumption_files: list[str] = field(default_factory=list)
    food_files: list[str] = field(default_factory=list)
    assets_files: list[str] = field(default_factory=list)
    nonfarm_files: list[str] = field(default_factory=list)
    remittance_files: list[str] = field(default_factory=list)
    shock_files: list[str] = field(default_factory=list)
    shock_id_candidates: list[str] = field(default_factory=list)
    shock_occurred_candidates: list[str] = field(default_factory=list)
    geo_files: list[str] = field(default_factory=list)
    geo_lat_candidates: list[str] = field(default_factory=list)
    geo_lon_candidates: list[str] = field(default_factory=list)
    geo_ea_candidates: list[str] = field(default_factory=list)
    community_exposure_files: list[str] = field(default_factory=list)
    community_controls_files: list[str] = field(default_factory=list)
    community_key_candidates: list[str] = field(default_factory=list)
    community_composite_key: list[str] = field(default_factory=list)
    household_community_link_files: list[str] = field(default_factory=list)
    livestock_files: list[str] = field(default_factory=list)
    # Optional auxiliary livestock modules. These are especially useful for waves 3-4,
    # where the LF questionnaire is split into stock, animal health, feed/water/housing,
    # labour, milk, dung/animal power, and by-products.
    livestock_health_files: list[str] = field(default_factory=list)
    livestock_milk_files: list[str] = field(default_factory=list)
    livestock_id_candidates: list[str] = field(default_factory=list)
    livestock_name_candidates: list[str] = field(default_factory=list)
    agriculture_files: list[str] = field(default_factory=list)
    plot_geo_files: list[str] = field(default_factory=list)


WAVE_CONFIGS: dict[int, WaveConfig] = {
    1: WaveConfig(
        wave=1,
        survey_year=2008,
        hhid_col="hhid",
        indid_col="sbmemno",
        core_files=["SEC_A_T.csv"],
        weight_candidates=["hh_weight_trimmed", "hh_weight", "hhweight"],
        rural_candidates=["rural", "urban"],
        cluster_candidates=["clusterid", "ea_id"],
        strata_candidates=["strataid", "strata"],
        region_candidates=["region"],
        district_candidates=["district"],
        ward_candidates=["ward"],
        community_id_from_core=["clusterid", "ea_id"],
        demographics_files=["SEC_B_C_D_E1_F_G1_U.csv"],
        age_candidates=["sbq3yr"],
        sex_candidates=["sbq4"],
        relation_candidates=["sbq5"],
        labour_files=["SEC_B_C_D_E1_F_G1_U.csv"],
        housing_files=["SEC_H1_J_K2_O2_P1_Q1_S1.csv"],
        consumption_files=["TZY1.HH.Consumption.csv", "ConsumptionNPS1.csv"],
        food_files=[],
        assets_files=["SEC_M.csv"],
        nonfarm_files=["SEC_E2.csv"],
        remittance_files=["SEC_Q2.csv", "SEC_Q3.csv"],
        shock_files=["SEC_R.csv"],
        shock_id_candidates=["srcode", "shockid"],
        shock_occurred_candidates=["srq1", "hh_r01"],
        geo_files=["HH.Geovariables_Y1.csv", "EA.Offsets.csv"],
        geo_lat_candidates=["lat_modified", "lat_dd_mod"],
        geo_lon_candidates=["lon_modified", "lon_dd_mod"],
        geo_ea_candidates=["ea_id"],
        community_exposure_files=["SEC_D.csv"],
        community_controls_files=["SEC_E_F_G.csv"],
        community_key_candidates=["ea_id", "clusterid"],
        livestock_files=["SEC_10A.csv"],
        livestock_id_candidates=["animal"],
        livestock_name_candidates=[],
        agriculture_files=[
            "SEC_2A.csv", "SEC_2B.csv", "SEC_3A.csv", "SEC_3B.csv",
            "SEC_4A.csv", "SEC_4B.csv", "SEC_5A.csv", "SEC_5B.csv",
            "SEC_6A.csv", "SEC_6B.csv", "SEC_7A.csv", "SEC_7B.csv",
            "SEC_8A.csv", "SEC_8B.csv", "SEC_8C.csv", "SEC_9_ALL.csv",
        ],
        plot_geo_files=["Plot.Geovariables_Y1_revised2.csv", "PlotGeovars_Y1.csv"],
    ),
    2: WaveConfig(
        wave=2,
        survey_year=2010,
        hhid_col="y2_hhid",
        indid_col="indidy2",
        core_files=["HH_SEC_A.csv"],
        weight_candidates=["y2_weight", "y2_panelweight", "hhweight"],
        rural_candidates=["y2_rural", "urban"],
        cluster_candidates=["clusterid", "ea"],
        strata_candidates=["strataid", "strata"],
        region_candidates=["region"],
        district_candidates=["district"],
        ward_candidates=["ward"],
        community_id_from_core=["clusterid", "ea"],
        demographics_files=["HH_SEC_B.csv"],
        age_candidates=["hh_b03_1"],
        sex_candidates=["hh_b04"],
        relation_candidates=["hh_b05"],
        labour_files=["HH_SEC_E1.csv", "HH_SEC_E2.csv"],
        housing_files=["HH_SEC_I1.csv", "HH_SEC_I2.csv"],
        consumption_files=["TZY2.HH.Consumption.csv", "ConsumptionNPS2.csv", "consumptionNPS2.csv"],
        food_files=["HH_SEC_I1.csv"],
        assets_files=["HH_SEC_M.csv"],
        nonfarm_files=["HH_SEC_E2.csv", "HH_SEC_N.csv"],
        remittance_files=["HH_SEC_Q.csv"],
        shock_files=["HH_SEC_R.csv"],
        shock_id_candidates=["shockid"],
        shock_occurred_candidates=["hh_r01"],
        geo_files=["HH.Geovariables_Y2.csv", "TZY2.HH.Geovariables.csv", "TZY2.EA.Offsets.csv", "EA.Offsets.csv"],
        geo_lat_candidates=["lat_modified", "lat_dd_mod"],
        geo_lon_candidates=["lon_modified", "lon_dd_mod"],
        geo_ea_candidates=["ea_id"],
        community_exposure_files=["COMSEC_CD.csv"],
        community_controls_files=["COMSEC_CE.csv"],
        community_composite_key=["id_01", "id_02", "id_03", "id_04"],
        household_community_link_files=["y2commlink.csv", "Y2COMMLINK.csv"],
        livestock_files=["AG_SEC10A.csv"],
        livestock_id_candidates=["lvstkcode"],
        livestock_name_candidates=[],
        agriculture_files=[
            "AG_SEC2A.csv", "AG_SEC2B.csv", "AG_SEC3A.csv", "AG_SEC3B.csv",
            "AG_SEC4A.csv", "AG_SEC4B.csv", "AG_SEC5A.csv", "AG_SEC5B.csv",
            "AG_SEC6A.csv", "AG_SEC6B.csv", "AG_SEC7A.csv", "AG_SEC7B.csv",
            "AG_SEC8A.csv", "AG_SEC8B.csv", "AG_SEC8C.csv", "AG_SEC09.csv",
        ],
        plot_geo_files=["Plot.Geovariables_Y2.csv", "PlotGeovars_Y2.csv"],
    ),
    3: WaveConfig(
        wave=3,
        survey_year=2012,
        hhid_col="y3_hhid",
        indid_col="indidy3",
        core_files=["HH_SEC_A.csv"],
        weight_candidates=["y3_weight", "y3_panelweight"],
        rural_candidates=["y3_rural", "urban"],
        cluster_candidates=["clusterid", "y3_cluster"],
        strata_candidates=["strataid", "strata"],
        region_candidates=["region", "id_01"],
        district_candidates=["district", "id_02"],
        ward_candidates=["ward", "id_03"],
        community_id_from_core=["y3_cluster", "clusterid"],
        demographics_files=["HH_SEC_B.csv"],
        age_candidates=["hh_b03_1"],
        sex_candidates=["hh_b04"],
        relation_candidates=["hh_b05"],
        labour_files=["HH_SEC_E.csv"],
        housing_files=["HH_SEC_I.csv"],
        consumption_files=["ConsumptionNPS3.csv", "consumptionNPS3.csv"],
        food_files=["HH_SEC_H.csv"],
        assets_files=["HH_SEC_M.csv"],
        nonfarm_files=["HH_SEC_N.csv"],
        remittance_files=["HH_SEC_Q2.csv"],
        shock_files=["HH_SEC_R.csv"],
        shock_id_candidates=["shockid"],
        shock_occurred_candidates=["hh_r01"],
        geo_files=["HouseholdGeovars_Y3.csv"],
        geo_lat_candidates=["lat_dd_mod", "lat_modified"],
        geo_lon_candidates=["lon_dd_mod", "lon_modified"],
        geo_ea_candidates=[],
        community_exposure_files=["COM_SEC_CD.csv"],
        community_controls_files=["COM_SEC_CE.csv"],
        community_key_candidates=["y3_cluster"],
        livestock_files=["LF_SEC_02.csv"],
        livestock_health_files=["LF_SEC_03.csv", "LF_SEC_04.csv", "LF_SEC_05.csv"],
        livestock_milk_files=["LF_SEC_06.csv", "LF_SEC_07.csv", "LF_SEC_08.csv"],
        livestock_id_candidates=["lvstckid"],
        livestock_name_candidates=["lvstckname"],
        agriculture_files=[
            "AG_SEC_2A.csv", "AG_SEC_2B.csv", "AG_SEC_3A.csv", "AG_SEC_3B.csv",
            "AG_SEC_4A.csv", "AG_SEC_4B.csv", "AG_SEC_5A.csv", "AG_SEC_5B.csv",
            "AG_SEC_6A.csv", "AG_SEC_6B.csv", "AG_SEC_7A.csv", "AG_SEC_7B.csv",
            "AG_SEC_8.csv", "AG_SEC_9A.csv", "AG_SEC_9B.csv", "AG_SEC_9C.csv",
            "AG_SEC_10.csv", "AG_SEC_11.csv", "AG_SEC_12A.csv", "AG_SEC_12B.csv",
        ],
        plot_geo_files=["PlotGeovars_Y3.csv"],
    ),
    4: WaveConfig(
        wave=4,
        survey_year=2014,
        hhid_col="y4_hhid",
        indid_col="indidy4",
        core_files=["hh_sec_a.csv", "HH_SEC_A.csv"],
        weight_candidates=["y4_weight", "y4_weights", "y4_panelweight", "y4_crossweight"],
        rural_candidates=["y4_rural", "urban"],
        cluster_candidates=["clusterid", "y4_cluster"],
        strata_candidates=["strataid", "strata"],
        region_candidates=["region", "id_01"],
        district_candidates=["district", "id_02"],
        ward_candidates=["ward", "id_03"],
        community_id_from_core=["clusterid", "y4_cluster"],
        demographics_files=["hh_sec_b.csv", "HH_SEC_B.csv"],
        age_candidates=["hh_b03_1"],
        sex_candidates=["hh_b04"],
        relation_candidates=["hh_b05"],
        labour_files=["hh_sec_e.csv", "HH_SEC_E.csv"],
        housing_files=["hh_sec_i.csv", "HH_SEC_I.csv"],
        consumption_files=["consumptionNPS4.CSV", "consumptionNPS4.csv", "ConsumptionNPS4.csv"],
        food_files=["hh_sec_h.csv", "HH_SEC_H.csv"],
        assets_files=["hh_sec_m.csv", "HH_SEC_M.csv"],
        nonfarm_files=["hh_sec_n.csv", "HH_SEC_N.csv"],
        remittance_files=["hh_sec_q2.csv", "HH_SEC_Q2.csv"],
        shock_files=["hh_sec_r.csv", "HH_SEC_R.csv"],
        shock_id_candidates=["shockid"],
        shock_occurred_candidates=["hh_r01"],
        geo_files=["npsy4.ea.offset.csv", "NPSY4.EA.OFFSET.csv", "EA.Offsets.csv"],
        geo_lat_candidates=["lat_modified", "lat_dd_mod"],
        geo_lon_candidates=["lon_modified", "lon_dd_mod"],
        geo_ea_candidates=["clusterid"],
        community_exposure_files=["com_sec_cd.csv", "COM_SEC_CD.csv"],
        community_controls_files=["com_sec_ce.csv", "COM_SEC_CE.csv"],
        community_key_candidates=["y4_cluster", "clusterid"],
        livestock_files=["lf_sec_02.csv", "LF_SEC_02.csv"],
        livestock_health_files=["lf_sec_03.csv", "LF_SEC_03.csv", "lf_sec_04.csv", "LF_SEC_04.csv", "lf_sec_05.csv", "LF_SEC_05.csv"],
        livestock_milk_files=["lf_sec_06.csv", "LF_SEC_06.csv", "lf_sec_07.csv", "LF_SEC_07.csv", "lf_sec_08.csv", "LF_SEC_08.csv"],
        livestock_id_candidates=["lvstckid"],
        livestock_name_candidates=["lvstckname"],
        agriculture_files=[
            "ag_sec_02.csv", "ag_sec_2a.csv", "ag_sec_2b.csv",
            "ag_sec_3a.csv", "ag_sec_3b.csv", "ag_sec_4a.csv", "ag_sec_4b.csv",
            "ag_sec_5a.csv", "ag_sec_5b.csv", "ag_sec_6a.csv", "ag_sec_6b.csv",
            "ag_sec_7a.csv", "ag_sec_7b.csv", "ag_sec_10.csv", "ag_sec_11.csv",
            "ag_sec_12a.csv", "ag_sec_12b.csv",
        ],
        plot_geo_files=[],
    ),
}


# ============================================================
# Utilities
# ============================================================

def clean_id(x: pd.Series | object) -> pd.Series:
    s = pd.Series(x) if not isinstance(x, pd.Series) else x
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "NaN": np.nan})
    )








def read_optional(folder: Path, candidates: list[str]) -> tuple[pd.DataFrame, Path | None]:
    path = find_file(folder, candidates, required=False)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path, low_memory=False), path


def to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().any():
            df[c] = converted
            cols.append(c)
    return cols


def make_composite_key(df: pd.DataFrame, cols: list[str], name: str = "community_id") -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if len(existing) != len(cols):
        return pd.Series(np.nan, index=df.index)
    parts = [clean_id(df[c]).fillna("") for c in existing]
    out = parts[0]
    for p in parts[1:]:
        out = out + "_" + p
    return out.replace({"___": np.nan, "__": np.nan, "_": np.nan})


def prefix_except(df: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    keep_set = set(keep)
    rename = {c: f"{prefix}_{c}" for c in df.columns if c not in keep_set and not c.startswith(f"{prefix}_")}
    return df.rename(columns=rename)


def group_numeric_sum(df: pd.DataFrame, id_col: str, prefix: str, exclude: list[str] | None = None) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return pd.DataFrame()
    exclude = set(exclude or []) | {id_col}
    tmp = df.copy()
    cols = numeric_columns(tmp, exclude=list(exclude))
    if not cols:
        return tmp[[id_col]].drop_duplicates().copy()
    out = tmp[[id_col] + cols].groupby(id_col, as_index=False)[cols].sum(min_count=1)
    return out.rename(columns={c: f"{prefix}_sum_{c}" for c in cols})


def group_numeric_mean(df: pd.DataFrame, id_col: str, prefix: str, exclude: list[str] | None = None) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return pd.DataFrame()
    exclude = set(exclude or []) | {id_col}
    tmp = df.copy()
    cols = numeric_columns(tmp, exclude=list(exclude))
    if not cols:
        return pd.DataFrame()
    out = tmp[[id_col] + cols].groupby(id_col, as_index=False)[cols].mean()
    return out.rename(columns={c: f"{prefix}_mean_{c}" for c in cols})


def merge_blocks(base: pd.DataFrame, blocks: list[pd.DataFrame], on: str = "hhid") -> pd.DataFrame:
    out = base.copy()
    for block in blocks:
        if block is not None and not block.empty and on in block.columns:
            # defensive de-duplication to avoid accidental many-to-many merges
            if block[on].duplicated().any():
                block = block.groupby(on, as_index=False).first()
            out = out.merge(block, on=on, how="left")
    return out


def add_source_report(report: list[dict], wave: int, module: str, path: Path | None, rows: int | None = None, cols: int | None = None):
    report.append({
        "wave": wave,
        "module": module,
        "file": str(path.name) if path is not None else "MISSING/SKIPPED",
        "rows": rows,
        "columns": cols,
    })


# ============================================================
# Builders
# ============================================================

def standardize_hhid(df: pd.DataFrame, cfg: WaveConfig) -> pd.DataFrame:
    col = first_existing(df.columns, [cfg.hhid_col, "hhid", f"y{cfg.wave}_hhid"])
    if col is None:
        raise KeyError(f"Wave {cfg.wave}: missing household ID. Columns start: {list(df.columns)[:20]}")
    out = df.copy()
    out["hhid_wave"] = clean_id(out[col])
    out["original_hhid"] = out["hhid_wave"]
    out["hhid"] = out["hhid_wave"]
    return out


def build_household_core(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.core_files)
    if df.empty:
        raise FileNotFoundError(f"Wave {cfg.wave}: household core missing. Tried: {cfg.core_files}")
    add_source_report(report, cfg.wave, "household_core", path, len(df), df.shape[1])
    df = standardize_hhid(df, cfg)

    out = df[["hhid", "hhid_wave", "original_hhid"]].copy()
    out["wave"] = cfg.wave
    out["survey_year"] = cfg.survey_year

    mappings = {
        "weight": cfg.weight_candidates,
        "rural": cfg.rural_candidates,
        "clusterid": cfg.cluster_candidates,
        "strataid": cfg.strata_candidates,
        "region": cfg.region_candidates,
        "district": cfg.district_candidates,
        "ward": cfg.ward_candidates,
    }
    for new, cands in mappings.items():
        col = first_existing(df.columns, cands)
        if col is not None:
            out[new] = df[col]

    # Harmonize rural when only urban is present: common coding assumed 1=urban.
    if "rural" in out.columns:
        out["rural"] = pd.to_numeric(out["rural"], errors="coerce")
        # If the source column was urban, invert it.
        source_col = first_existing(df.columns, cfg.rural_candidates)
        if source_col and source_col.lower() == "urban":
            out["rural"] = np.where(out["rural"] == 1, 0, np.where(out["rural"].notna(), 1, np.nan))

    comm_col = first_existing(df.columns, cfg.community_id_from_core)
    if comm_col is not None:
        out["community_id"] = clean_id(df[comm_col])
    elif "clusterid" in out.columns:
        out["community_id"] = clean_id(out["clusterid"])
    else:
        out["community_id"] = np.nan

    for c in ["clusterid", "strataid", "region", "district", "ward", "community_id"]:
        if c in out.columns:
            out[c] = clean_id(out[c])

    # Keep a few useful raw household/core fields, prefixed.
    keep_cols = ["hhid", "hhid_wave", "original_hhid", "wave", "survey_year", "weight", "rural", "clusterid", "strataid", "region", "district", "ward", "community_id"]
    raw_extra = [c for c in df.columns if c not in out.columns and c not in [cfg.hhid_col, "hhid"]]
    raw_extra = raw_extra[:40]
    raw = df[["hhid"] + raw_extra].copy() if raw_extra else df[["hhid"]].copy()
    raw = prefix_except(raw, "core_raw", ["hhid"])
    out = out.merge(raw, on="hhid", how="left")
    return out


def build_demographics(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.demographics_files)
    if df.empty:
        return pd.DataFrame()
    add_source_report(report, cfg.wave, "demographics", path, len(df), df.shape[1])
    df = standardize_hhid(df, cfg)

    ind_col = first_existing(df.columns, [cfg.indid_col or "", "indid", "personid", "memid", "sbmemno"])
    age_col = first_existing(df.columns, cfg.age_candidates)
    sex_col = first_existing(df.columns, cfg.sex_candidates)
    rel_col = first_existing(df.columns, cfg.relation_candidates)

    if ind_col is None:
        df["_person_counter"] = np.arange(len(df))
        ind_col = "_person_counter"

    out = df.groupby("hhid", as_index=False).agg(hh_size=(ind_col, "count"))

    if age_col:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        tmp = df.groupby("hhid", as_index=False).agg(
            mean_age=(age_col, "mean"),
            max_age=(age_col, "max"),
            n_children_u15=(age_col, lambda x: int((x < 15).sum())),
            n_adults_15plus=(age_col, lambda x: int((x >= 15).sum())),
        )
        out = out.merge(tmp, on="hhid", how="left")

    if sex_col:
        df[sex_col] = pd.to_numeric(df[sex_col], errors="coerce")
        tmp = df.groupby("hhid", as_index=False).agg(
            n_female=(sex_col, lambda x: int((x == 2).sum())),
            share_female=(sex_col, lambda x: np.nan if x.notna().sum() == 0 else float((x == 2).mean())),
        )
        out = out.merge(tmp, on="hhid", how="left")

    if rel_col:
        df[rel_col] = pd.to_numeric(df[rel_col], errors="coerce")
        heads = df[df[rel_col] == 1].copy()
        keep = ["hhid"]
        ren = {}
        if age_col:
            keep.append(age_col); ren[age_col] = "head_age"
        if sex_col:
            keep.append(sex_col); ren[sex_col] = "head_sex"
        if keep != ["hhid"]:
            heads = heads[keep].rename(columns=ren).drop_duplicates("hhid")
            out = out.merge(heads, on="hhid", how="left")
    return out


def build_module_raw(folder: Path, cfg: WaveConfig, files: list[str], module: str, prefix: str, report: list[dict], aggregate: str = "sum") -> pd.DataFrame:
    blocks = []
    for file in files:
        df, path = read_optional(folder, [file])
        if df.empty:
            continue
        add_source_report(report, cfg.wave, module, path, len(df), df.shape[1])
        try:
            df = standardize_hhid(df, cfg)
        except Exception:
            continue
        id_cols = ["hhid", "hhid_wave", "original_hhid", cfg.hhid_col, "hhid"]
        if aggregate == "first":
            tmp = df.drop_duplicates("hhid").copy()
            keep = ["hhid"] + [c for c in tmp.columns if c not in id_cols][:80]
            tmp = prefix_except(tmp[keep], prefix, ["hhid"])
        elif aggregate == "mean":
            tmp = group_numeric_mean(df, "hhid", prefix, exclude=id_cols)
        else:
            tmp = group_numeric_sum(df, "hhid", prefix, exclude=id_cols)
        if not tmp.empty:
            blocks.append(tmp)
    if not blocks:
        return pd.DataFrame()
    out = blocks[0]
    for b in blocks[1:]:
        out = out.merge(b, on="hhid", how="outer")
    return out


def build_labour(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    out = build_module_raw(folder, cfg, cfg.labour_files, "labour", "labour", report, aggregate="sum")
    if out.empty:
        return out
    # HARMONIZATION SOURCE: labour activity variables are harmonized only as proxies.
    # Wave 2 DDI p.5-6 describes household labour in the household questionnaire.
    # Wave 3/4 headers show hh_e04* variants; DDI p.5 for wave 3 and p.5 for wave 4
    # document comparable household questionnaire structure, but labels differ across waves.
    # Therefore these are conservative proxy aliases; validate labels before final publication.
    candidates_nonfarm = [c for c in out.columns if re.search(r"sum_hh_e04c|sum_hh_e04d|sum_hh_e04cd", c)]
    candidates_farm = [c for c in out.columns if re.search(r"sum_hh_e04e$", c)]
    candidates_anywork = [c for c in out.columns if re.search(r"sum_hh_e04$|sum_hh_e04a|sum_hh_e04b|sum_hh_e04ab|sum_hh_e04cd|sum_hh_e04e", c)]
    if candidates_nonfarm:
        out["labour_nonfarm_activity_proxy_sum"] = out[candidates_nonfarm].sum(axis=1, min_count=1)
    if candidates_farm:
        out["labour_farm_livestock_activity_proxy_sum"] = out[candidates_farm].sum(axis=1, min_count=1)
    if candidates_anywork:
        out["labour_any_activity_proxy_sum"] = out[candidates_anywork].sum(axis=1, min_count=1)
    return out


def build_consumption(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.consumption_files)
    if df.empty:
        return pd.DataFrame()
    add_source_report(report, cfg.wave, "consumption", path, len(df), df.shape[1])

    # Consumption files for wave 1 may use plain hhid while wave 2-4 use y*_hhid.
    id_col = first_existing(df.columns, [cfg.hhid_col, "hhid", "HHID", f"y{cfg.wave}_hhid"])
    if id_col is None:
        warnings.warn(f"Wave {cfg.wave}: consumption file has no household ID: {path.name}")
        return pd.DataFrame()
    df = df.copy()
    df["hhid"] = clean_id(df[id_col])
    for c in df.columns:
        if c != "hhid":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("hhid")

    out = df[["hhid"]].copy()
    # HARMONIZATION SOURCE: consumption aggregate files in wave 1/2/3/4 share expm, expmR,
    # foodbevR, foodINR, foodOUTR, hhsize, and adulteq headers. See header dumps for
    # TZY1.HH.Consumption, TZY2.HH.Consumption, ConsumptionNPS3, and consumptionNPS4.
    # The R suffix is treated as real/spatially adjusted, consistent with the consumption
    # aggregate descriptions used in the NPS documentation.
    aliases = {
        "expmR": "cons_total_monthly_real",
        "expm": "cons_total_monthly_nominal",
        "foodbevR": "cons_foodbev_real",
        "foodINR": "cons_food_in_real",
        "foodOUTR": "cons_food_out_real",
        "hhsize": "cons_hhsize",
        "adulteq": "cons_adulteq",
    }
    for raw, new in aliases.items():
        if raw in df.columns:
            out[new] = df[raw]
    if "expmR" in df.columns and "hhsize" in df.columns:
        out["cons_total_monthly_real_pc"] = df["expmR"] / df["hhsize"].replace(0, np.nan)
    if "expmR" in df.columns and "adulteq" in df.columns:
        out["cons_total_monthly_real_ae"] = df["expmR"] / df["adulteq"].replace(0, np.nan)
    if "foodbevR" in df.columns and "expmR" in df.columns:
        out["cons_food_share_real"] = df["foodbevR"] / df["expmR"].replace(0, np.nan)

    raw = prefix_except(df, "cons", ["hhid"])
    out = out.merge(raw, on="hhid", how="left")
    return out


def build_shocks(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.shock_files)
    if df.empty:
        return pd.DataFrame()
    add_source_report(report, cfg.wave, "shocks", path, len(df), df.shape[1])
    df = standardize_hhid(df, cfg)
    shockid = first_existing(df.columns, cfg.shock_id_candidates)
    occurred = first_existing(df.columns, cfg.shock_occurred_candidates)
    if shockid is None:
        shockid = "_shock_record"
        df[shockid] = np.arange(len(df))
    if occurred is None:
        df["shock_occurred"] = np.nan
    else:
        # HARMONIZATION SOURCE: wave 1 SEC_R uses srcode/srq1; wave 2-4 HH_SEC_R use
        # shockid/hh_r01 or close variants. Header dumps confirm this mapping.
        # Yes/no coding assumes 1=yes and creates comparable shock occurrence dummies.
        df["shock_occurred"] = yes1(df[occurred])
    df[shockid] = pd.to_numeric(df[shockid], errors="coerce")

    out = df.groupby("hhid", as_index=False).agg(
        n_shock_records=(shockid, "count"),
        n_shocks_occurred=("shock_occurred", "sum"),
    )
    if shockid in df.columns and df[shockid].notna().any():
        wide = df.pivot_table(index="hhid", columns=shockid, values="shock_occurred", aggfunc="max")
        wide.columns = [f"shock_{int(c)}_occurred" if pd.notna(c) else "shock_missing_occurred" for c in wide.columns]
        wide = wide.reset_index()
        out = out.merge(wide, on="hhid", how="left")
    raw = group_numeric_sum(df, "hhid", "shock_raw", exclude=[cfg.hhid_col, "hhid", "hhid_wave", "original_hhid"])
    if not raw.empty:
        out = out.merge(raw, on="hhid", how="left")
    return out


def merge_combine(out: pd.DataFrame, tmp: pd.DataFrame, on: str = "hhid") -> pd.DataFrame:
    """Merge tmp into out, combining same-named columns with first non-missing values."""
    if tmp.empty:
        return out
    tmp = tmp.drop_duplicates(on)
    merged = out.merge(tmp, on=on, how="left", suffixes=("", "__new"))
    for c in tmp.columns:
        if c == on:
            continue
        new = f"{c}__new"
        if new in merged.columns:
            merged[c] = merged[c].combine_first(merged[new])
            merged = merged.drop(columns=new)
    return merged


def standardize_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep harmonized environmental and market-access geovariables from LSMS/NPS
    geovariable files. These include distance variables (dist* / hh_geo*), climate
    and vegetation/environment variables (clim* / hh_clim* / hh_envi*), terrain and
    soil variables (srtm*, twi*, soil*), and land/crop suitability variables.

    The DDI/header dumps show these are not identically named in every wave:
      - Waves 1/3: dist01-dist05, clim01-clim05, land*, soil*, crops*.
      - Wave 2: both the above and hh_clim*, hh_envi*, hh_geo*, hh_soil_con*,
        afmnslp_pct, srtm_eaf_5_15, twi_tza.
    We prefix everything with geo_* to make the subset and regression scripts
    able to detect the controls automatically without relying on wave-specific names.
    """
    keep = {}
    for c in df.columns:
        lc = c.lower()
        if re.fullmatch(r"dist\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"clim\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"land\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"soil\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"crops\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"hh_clim\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"hh_envi\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"hh_geo\d+", lc):
            keep[c] = f"geo_{lc}"
        elif re.fullmatch(r"hh_soil_con\d+", lc):
            keep[c] = f"geo_{lc}"
        elif lc in {"afmnslp_pct", "srtm_eaf_5_15", "twi_tza", "dist_y1y2", "dist_y1y3"}:
            keep[c] = f"geo_{lc}"
    if not keep:
        return pd.DataFrame(index=df.index)
    out = df[list(keep)].copy().rename(columns=keep)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_spatial_coordinates(folder: Path, cfg: WaveConfig, household_core: pd.DataFrame, report: list[dict]) -> pd.DataFrame:
    """
    Harmonize coordinates and LSMS geovariables to household-level variables.

    HARMONIZATION SOURCE:
    - Wave 2 DDI p.66-67 explains that exact GPS locations are confidential and
      public geovariables/EA offsets use modified cluster centerpoints.
    - Wave 1/2/3 header dumps show HH geovariable files include distance, climate,
      land, soil and crop/vegetation variables. Wave 2 additionally has hh_clim*,
      hh_envi*, hh_geo*, hh_soil_con*, afmnslp_pct, srtm_eaf_5_15 and twi_tza.
    - Wave 3 DDI p.14-15 and Wave 4 DDI p.14-15 document plot GPS fields, but for
      household/community exposure we use offset EA/cluster coordinates.

    Output includes:
      cluster_lat, cluster_lon, geo_* controls.
    """
    out = household_core[["hhid"]].drop_duplicates().copy()
    out["cluster_lat"] = np.nan
    out["cluster_lon"] = np.nan

    candidate_core_keys = ["clusterid", "community_id", "geo_ea_id", "core_raw_ea_id", "core_raw_ea"]

    for file in cfg.geo_files:
        df, path = read_optional(folder, [file])
        if df.empty:
            continue
        add_source_report(report, cfg.wave, "spatial_coordinates_geovars", path, len(df), df.shape[1])

        lat = first_existing(df.columns, cfg.geo_lat_candidates + ["lat_modified", "lat_dd_mod"])
        lon = first_existing(df.columns, cfg.geo_lon_candidates + ["lon_modified", "lon_dd_mod"])
        geo = standardize_geo_columns(df)

        # Household-level geovariables/coordinates.
        id_col = first_existing(df.columns, [cfg.hhid_col, "hhid", f"y{cfg.wave}_hhid"])
        if id_col is not None:
            cols = [id_col]
            if lat and lon:
                cols += [lat, lon]
            tmp = df[cols].copy()
            tmp["hhid"] = clean_id(tmp[id_col])
            if lat and lon:
                tmp["cluster_lat"] = pd.to_numeric(tmp[lat], errors="coerce")
                tmp["cluster_lon"] = pd.to_numeric(tmp[lon], errors="coerce")
            tmp = tmp[["hhid"] + [c for c in ["cluster_lat", "cluster_lon"] if c in tmp.columns]]
            if not geo.empty:
                tmp = pd.concat([tmp.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
            out = merge_combine(out, tmp, on="hhid")
            continue

        # EA/cluster-level offsets or geovariables.
        offset_key = first_existing(df.columns, ["clusterid", "ea_id", "ea", "y3_cluster", "y4_cluster"])
        if offset_key is None:
            continue

        tmp = pd.DataFrame({"_geo_key": clean_id(df[offset_key])})
        if lat and lon:
            tmp["cluster_lat"] = pd.to_numeric(df[lat], errors="coerce")
            tmp["cluster_lon"] = pd.to_numeric(df[lon], errors="coerce")
        if not geo.empty:
            tmp = pd.concat([tmp.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
        tmp = tmp.drop_duplicates("_geo_key")

        for core_key in candidate_core_keys:
            if core_key not in household_core.columns:
                continue
            bridge = household_core[["hhid", core_key]].copy()
            bridge["_geo_key"] = clean_id(bridge[core_key])
            m = bridge[["hhid", "_geo_key"]].merge(tmp, on="_geo_key", how="left").drop(columns="_geo_key")
            value_cols = [c for c in m.columns if c != "hhid"]
            if not value_cols or m[value_cols].notna().sum().sum() == 0:
                continue
            out = merge_combine(out, m, on="hhid")
            break

    # Return even if coordinates are missing, as geovariables may still be useful.
    value_cols = [c for c in out.columns if c != "hhid"]
    if not value_cols or out[value_cols].notna().sum().sum() == 0:
        return pd.DataFrame()
    return out

def standardize_community(df: pd.DataFrame, cfg: WaveConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.community_composite_key:
        out["community_id"] = make_composite_key(out, cfg.community_composite_key)
    else:
        key = first_existing(out.columns, cfg.community_key_candidates)
        if key is not None:
            out["community_id"] = clean_id(out[key])
        else:
            # Wave 1 fallback: EA-style key
            key = first_existing(out.columns, ["ea_id", "clusterid"])
            out["community_id"] = clean_id(out[key]) if key else np.nan
    return out


def build_community_exposure(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.community_exposure_files)
    if df.empty:
        return pd.DataFrame()
    add_source_report(report, cfg.wave, "community_exposure", path, len(df), df.shape[1])
    df = standardize_community(df, cfg)

    out = df[["community_id"]].copy()
    # HARMONIZATION SOURCE: community land variables are documented in the community
    # questionnaire land-use section. Wave 3 DDI p.54-55 lists COM_SEC_CD with id_01-id_04,
    # y3_cluster, land shares, land-event months/years, affected households and compensation.
    # Wave 4 DDI p.50 and p.314-316 list the equivalent com_sec_cd variables with cd* names.
    # Wave 2 DDI p.17 confirms the community questionnaire includes SECTION CD: LAND USE.
    # Header dumps confirm wave 2 uses cm_d* and wave 4 uses cd*.
    # Land-use / land-event variables differ mainly in prefix: cd* in wave1/4, cm_d* in wave2/3.
    mapping_candidates = {
        "cultivation_villagers_acres": ["cd1a1", "cm_d01a1"],
        "cultivation_villagers_pct": ["cd1a2", "cm_d01a2"],
        "outsider_agrobusiness_acres": ["cd1b1", "cm_d01b1"],
        "outsider_agrobusiness_pct": ["cd1b2", "cm_d01b2"],
        "forest_acres": ["cd1c1", "cm_d01c1"],
        "forest_pct": ["cd1c2", "cm_d01c2"],
        "grazing_land_acres": ["cd1d1", "cm_d01d1"],
        "grazing_land_pct": ["cd1d2", "cm_d01d2"],
        "has_village_land_certificate": ["cd2", "cm_d02"],
        "village_land_certificate_month": ["cd3m", "cm_d03m"],
        "village_land_certificate_year": ["cd3y", "cm_d03y"],
        "outside_investor_appropriation_month": ["cd5am", "cm_d05am"],
        "outside_investor_appropriation_year": ["cd5ay", "cm_d05ay"],
        "reserve_land_declaration_month": ["cd5bm", "cm_d05bm"],
        "reserve_land_declaration_year": ["cd5by", "cm_d05by"],
        "public_use_allocation_month": ["cd5cm", "cm_d05cm"],
        "public_use_allocation_year": ["cd5cy", "cm_d05cy"],
        "hh_affected_outside_investor_appropriation": ["cd6a", "cm_d06a"],
        "hh_affected_reserve_land_declaration": ["cd6b", "cm_d06b"],
        "hh_affected_public_use_allocation": ["cd6c", "cm_d06c"],
        "compensation_outside_investor_appropriation": ["cd7a", "cm_d07a"],
        "compensation_reserve_land_declaration": ["cd7b", "cm_d07b"],
        "compensation_public_use_allocation": ["cd7c", "cm_d07c"],
    }
    for new, cands in mapping_candidates.items():
        col = first_existing(df.columns, cands)
        if col:
            out[new] = pd.to_numeric(df[col], errors="coerce")

    event_mapping = {
        "event_outside_investor_appropriation": ["cd4a", "cm_d04a"],
        "event_reserve_land_declaration": ["cd4b", "cm_d04b"],
        "event_public_use_allocation": ["cd4c", "cm_d04c"],
    }
    for new, cands in event_mapping.items():
        col = first_existing(df.columns, cands)
        if col:
            out[new] = yes1(df[col])
    event_cols = [c for c in event_mapping if c in out.columns]
    if event_cols:
        out["any_land_event"] = (out[event_cols].sum(axis=1, min_count=1) > 0).astype(float)
    year_cols = [c for c in ["outside_investor_appropriation_year", "reserve_land_declaration_year", "public_use_allocation_year"] if c in out.columns]
    if year_cols:
        out["first_land_event_year"] = out[year_cols].min(axis=1, skipna=True)

    raw = prefix_except(df, "community_land_raw", ["community_id"])
    out = out.merge(raw.drop_duplicates("community_id"), on="community_id", how="left")
    return out.drop_duplicates("community_id")


def build_community_controls(folder: Path, cfg: WaveConfig, report: list[dict]) -> pd.DataFrame:
    df, path = read_optional(folder, cfg.community_controls_files)
    if df.empty:
        return pd.DataFrame()
    add_source_report(report, cfg.wave, "community_controls", path, len(df), df.shape[1])
    df = standardize_community(df, cfg)
    out = df[["community_id"]].copy()

    # HARMONIZATION SOURCE: Wave 3 community questionnaire includes Section CE:
    # Demographics, Land & Livestock (DDI p.635 / table of contents) and the headers show
    # cm_e09/cm_e10 for migration and cm_e18-cm_e21 for livestock mobility.
    # Wave 4 has the same com_sec_ce structure in the user-provided headers.
    # Wave 2 CE is shorter and lacks the later livestock mobility variables.
    # Migration / livestock mobility variables. Wave 2 lacks the later livestock mobility variables.
    aliases = {
        "perm_migration_out_hh_12m": ["cm_e09", "ce09"],
        "perm_migration_in_hh_12m": ["cm_e10", "ce10"],
        "livestock_migration_any": ["cm_e18", "ce18"],
        "livestock_migration_changed_vs_historical": ["cm_e21", "ce21"],
    }
    for new, cands in aliases.items():
        col = first_existing(df.columns, cands)
        if col:
            if new == "livestock_migration_any":
                out[new] = yes1(df[col])
            else:
                out[new] = pd.to_numeric(df[col], errors="coerce")

    month_cols = []
    for m in range(1, 13):
        col = first_existing(df.columns, [f"cm_e20_{m}", f"ce20_{m}", f"ce20{m}"])
        if col:
            new = f"livestock_migration_month_{m:02d}"
            out[new] = pd.to_numeric(df[col], errors="coerce")
            month_cols.append(new)
    if month_cols:
        out["livestock_migration_months_count"] = out[month_cols].sum(axis=1, min_count=1)

    raw = prefix_except(df, "community_ctrl_raw", ["community_id"])
    out = out.merge(raw.drop_duplicates("community_id"), on="community_id", how="left")
    return out.drop_duplicates("community_id")


def attach_community_to_households(folder: Path, cfg: WaveConfig, community_full: pd.DataFrame, household_core: pd.DataFrame, report: list[dict]) -> pd.DataFrame:
    if community_full.empty:
        return pd.DataFrame()

    # Wave 2 community files are keyed by id_01...id_04 and require y2commlink.
    if cfg.household_community_link_files:
        link, path = read_optional(folder, cfg.household_community_link_files)
        if not link.empty:
            add_source_report(report, cfg.wave, "household_community_link", path, len(link), link.shape[1])
            link = standardize_hhid(link, cfg)
            link["community_id"] = make_composite_key(link, cfg.community_composite_key)
            bridge = link[["hhid", "community_id"]].drop_duplicates("hhid")
            return bridge.merge(community_full, on="community_id", how="left")

    # Normal case: household_core already has community_id comparable with community_full.
    bridge = household_core[["hhid", "community_id"]].drop_duplicates("hhid")
    return bridge.merge(community_full, on="community_id", how="left")


# Conservative default TLU weights. These are only used for a clearly-labelled
# *_default estimate and should be validated against the official Tanzania NPS
# livestock code list before becoming a headline outcome.
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

# Fallback for waves without livestock-name labels. This is deliberately marked
# "default" in outputs and should be overridden if the codebook says otherwise.
DEFAULT_LIVESTOCK_CODE_GROUP = {
    1: "cattle", 2: "cattle", 3: "cattle",
    4: "goat", 5: "sheep", 6: "pig",
    7: "poultry", 8: "poultry",
    9: "donkey_horse", 10: "camel",
    11: "rabbit",
}


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
        c = int(float(code))
        return DEFAULT_LIVESTOCK_CODE_GROUP.get(c, "other")
    except Exception:
        return "other"


def current_livestock_count(df: pd.DataFrame) -> pd.Series:
    """Best-effort current stock count across modules."""
    # Waves 3-4 LF_SEC_02: current indigenous + exotic/improved counts.
    if "lf02_04_1" in df.columns or "lf02_04_2" in df.columns:
        return pd.to_numeric(df.get("lf02_04_1", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("lf02_04_2", 0), errors="coerce").fillna(0)
    # Wave 2 AG_SEC10A likely uses ag10a_05_* for current stock counts. Kept as best-effort.
    if "ag10a_05_1" in df.columns or "ag10a_05_2" in df.columns:
        return pd.to_numeric(df.get("ag10a_05_1", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("ag10a_05_2", 0), errors="coerce").fillna(0)
    # Wave 1 SEC_10A likely uses s10aq4_* for stock counts. Kept as best-effort.
    if "s10aq4_1" in df.columns or "s10aq4_2" in df.columns:
        return pd.to_numeric(df.get("s10aq4_1", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("s10aq4_2", 0), errors="coerce").fillna(0)
    return pd.Series(np.nan, index=df.index)


def build_livestock(folder: Path, cfg: WaveConfig, report: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, path = read_optional(folder, cfg.livestock_files)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    add_source_report(report, cfg.wave, "livestock_stock", path, len(df), df.shape[1])
    df = standardize_hhid(df, cfg)
    livestock_id = first_existing(df.columns, cfg.livestock_id_candidates)
    livestock_name = first_existing(df.columns, cfg.livestock_name_candidates)
    if livestock_id is None:
        livestock_id = "_livestock_record"
        df[livestock_id] = np.arange(len(df))

    long_keep = ["hhid", "hhid_wave", livestock_id]
    if livestock_name:
        long_keep.append(livestock_name)
    long_keep += [c for c in df.columns if c not in long_keep and c not in [cfg.hhid_col, "original_hhid"]][:100]
    livestock_long = df[long_keep].copy()

    df["_lv_current"] = current_livestock_count(df)
    df["_lv_group"] = [infer_livestock_group(c, n if livestock_name else None) for c, n in zip(df[livestock_id], df[livestock_name] if livestock_name else [None] * len(df))]
    df["_tlu_weight_default"] = df["_lv_group"].map(TLU_WEIGHTS_BY_GROUP).fillna(0.0)
    df["_tlu_default"] = pd.to_numeric(df["_lv_current"], errors="coerce") * df["_tlu_weight_default"]

    out = df.groupby("hhid", as_index=False).agg(n_livestock_records=(livestock_id, "count"))

    tmp = df.groupby("hhid", as_index=False).agg(
        livestock_total_current=("_lv_current", "sum"),
        livestock_tlu_default=("_tlu_default", "sum"),
    )

    # More precise aliases for waves 3-4, where lf02_* is comparable.
    if "lf02_04_1" in df.columns:
        tmp = tmp.merge(df.groupby("hhid", as_index=False).agg(livestock_indigenous_current=("lf02_04_1", "sum")), on="hhid", how="left")
    if "lf02_04_2" in df.columns:
        tmp = tmp.merge(df.groupby("hhid", as_index=False).agg(livestock_exotic_current=("lf02_04_2", "sum")), on="hhid", how="left")
    if "lf02_03" in df.columns:
        tmp = tmp.merge(df.groupby("hhid", as_index=False).agg(livestock_total_12m_ago=("lf02_03", "sum")), on="hhid", how="left")
    if "lf02_02" in df.columns:
        tmp = tmp.merge(df.groupby("hhid", as_index=False).agg(livestock_total_2y_ago=("lf02_02", "sum")), on="hhid", how="left")
    if "lf02_26" in df.columns:
        tmp = tmp.merge(df.groupby("hhid", as_index=False).agg(livestock_sales_value=("lf02_26", "sum")), on="hhid", how="left")
    if "livestock_total_12m_ago" in tmp.columns:
        tmp["livestock_change_12m"] = tmp["livestock_total_current"] - tmp["livestock_total_12m_ago"]

    # Herd composition by inferred group.
    comp = df.pivot_table(index="hhid", columns="_lv_group", values="_lv_current", aggfunc="sum").reset_index()
    comp = comp.rename(columns={c: f"livestock_{c}_current" for c in comp.columns if c != "hhid"})
    tmp = tmp.merge(comp, on="hhid", how="left")
    if "livestock_cattle_current" in tmp.columns:
        tmp["livestock_cattle_share"] = tmp["livestock_cattle_current"] / tmp["livestock_total_current"].replace(0, np.nan)
    if "livestock_goat_current" in tmp.columns or "livestock_sheep_current" in tmp.columns:
        tmp["livestock_small_ruminants_current"] = tmp.get("livestock_goat_current", 0).fillna(0) + tmp.get("livestock_sheep_current", 0).fillna(0)
        tmp["livestock_small_ruminants_share"] = tmp["livestock_small_ruminants_current"] / tmp["livestock_total_current"].replace(0, np.nan)

    # Herd composition by raw code for transparency and validation.
    code_counts = df.copy()
    code_counts["_lv_code"] = pd.to_numeric(code_counts[livestock_id], errors="coerce")
    wide = code_counts.pivot_table(index="hhid", columns="_lv_code", values="_lv_current", aggfunc="sum")
    if not wide.empty:
        wide.columns = [f"livestock_current_code_{int(c):02d}" if pd.notna(c) else "livestock_current_code_missing" for c in wide.columns]
        wide = wide.reset_index()
        tmp = tmp.merge(wide, on="hhid", how="left")

    out = out.merge(tmp, on="hhid", how="left")

    # Auxiliary livestock modules: raw aggregates, kept for mechanisms/descriptives.
    for module_files, label in [(cfg.livestock_health_files, "livestock_health"), (cfg.livestock_milk_files, "livestock_milk")]:
        for f in module_files:
            aux, aux_path = read_optional(folder, [f])
            if aux.empty:
                continue
            add_source_report(report, cfg.wave, label, aux_path, len(aux), aux.shape[1])
            aux = standardize_hhid(aux, cfg)
            raw_aux = group_numeric_sum(aux, "hhid", Path(f).stem.lower(), exclude=[cfg.hhid_col, "hhid", "hhid_wave", "original_hhid"])
            if not raw_aux.empty:
                out = out.merge(raw_aux, on="hhid", how="left")

    # Conservative raw aggregates from the stock file.
    raw = group_numeric_sum(df, "hhid", "livestock_raw", exclude=[cfg.hhid_col, "hhid", "hhid_wave", "original_hhid"])
    if not raw.empty:
        out = out.merge(raw, on="hhid", how="left")
    return livestock_long, out

def build_agriculture(folder: Path, cfg: WaveConfig, report: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = []
    plot_blocks = []
    for f in cfg.agriculture_files:
        df, path = read_optional(folder, [f])
        if df.empty:
            continue
        add_source_report(report, cfg.wave, "agriculture", path, len(df), df.shape[1])
        try:
            df = standardize_hhid(df, cfg)
        except Exception:
            continue
        prefix = Path(f).stem.lower().replace(".", "_")
        raw = group_numeric_sum(df, "hhid", prefix, exclude=[cfg.hhid_col, "hhid", "hhid_wave", "original_hhid"])
        if not raw.empty:
            blocks.append(raw)

        # HARMONIZATION SOURCE: Wave 3 DDI p.14-15 and Wave 4 DDI p.14-15 document
        # AG_SEC_2A/2B plot GPS degree/minute variables. We convert those to decimal
        # degrees for plot summaries when present; household treatment exposure still uses
        # offset cluster coordinates from build_spatial_coordinates().
        # Plot GPS when present in ag_sec_2a/2b or similar.
        lat_deg = first_existing(df.columns, ["ag2a_06_1", "ag2b_17_1"])
        lat_min = first_existing(df.columns, ["ag2a_06_2", "ag2b_17_2"])
        lon_deg = first_existing(df.columns, ["ag2a_06_3", "ag2b_17_3"])
        lon_min = first_existing(df.columns, ["ag2a_06_4", "ag2b_17_4"])
        if lat_deg and lat_min and lon_deg and lon_min:
            p = df[["hhid"]].copy()
            if "plotnum" in df.columns:
                p["plotnum"] = df["plotnum"]
            p["plot_lat"] = -(pd.to_numeric(df[lat_deg], errors="coerce").abs() + pd.to_numeric(df[lat_min], errors="coerce") / 60)
            p["plot_lon"] = pd.to_numeric(df[lon_deg], errors="coerce") + pd.to_numeric(df[lon_min], errors="coerce") / 60
            plot_blocks.append(p)

    # Additional plot geovars files.
    for f in cfg.plot_geo_files:
        df, path = read_optional(folder, [f])
        if df.empty:
            continue
        add_source_report(report, cfg.wave, "plot_geovars", path, len(df), df.shape[1])
        id_col = first_existing(df.columns, [cfg.hhid_col, "hhid", f"y{cfg.wave}_hhid"])
        if not id_col:
            continue
        p = df.copy()
        p["hhid"] = clean_id(p[id_col])
        plot_blocks.append(p)

    if blocks:
        ag = blocks[0]
        for b in blocks[1:]:
            ag = ag.merge(b, on="hhid", how="outer")
    else:
        ag = pd.DataFrame()

    if plot_blocks:
        plot = pd.concat(plot_blocks, ignore_index=True, sort=False)
        if "plot_lat" in plot.columns and "plot_lon" in plot.columns:
            plot_summary = plot.groupby("hhid", as_index=False).agg(
                n_plot_gps=("plot_lat", "count"),
                mean_plot_lat=("plot_lat", "mean"),
                mean_plot_lon=("plot_lon", "mean"),
            )
            ag = plot_summary if ag.empty else ag.merge(plot_summary, on="hhid", how="outer")
    else:
        plot = pd.DataFrame()
    return plot, ag



def add_final_harmonized_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add final harmonized variables that are safe across waves.

    HARMONIZATION SOURCE:
    - Consumption: common aggregate headers expmR/foodbevR/hhsize/adulteq across waves.
    - Land events: Wave 3 DDI p.54-55 and Wave 4 DDI p.50/p.314-316 document equivalent
      land-event variables and event year/month fields; wave 2 community questionnaire includes
      Section CD: LAND USE (Wave 2 DDI p.17).
    - Coordinates: Wave 2 DDI p.66-67 explains public coordinates are modified EA/cluster
      coordinates; use buffers/distance rather than exact household location.
    """
    out = df.copy()

    # Binary pastoral-relevance proxy based on livestock counts where comparable aliases exist.
    if "livestock_total_current" in out.columns:
        out["has_livestock_current"] = np.where(pd.to_numeric(out["livestock_total_current"], errors="coerce") > 0, 1, 0)
    else:
        raw_livestock_current_cols = [c for c in out.columns if re.search(r"livestock_raw_sum_.*(04_1|04_2|10a_05_1|10a_05_2)", c)]
        if raw_livestock_current_cols:
            tmp = out[raw_livestock_current_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            out["has_livestock_current"] = np.where(tmp > 0, 1, np.where(tmp.notna(), 0, np.nan))

    # Land-event exposure summary.
    event_cols = [c for c in [
        "event_outside_investor_appropriation",
        "event_reserve_land_declaration",
        "event_public_use_allocation",
    ] if c in out.columns]
    if event_cols and "any_land_event" not in out.columns:
        out["any_land_event"] = (out[event_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) > 0).astype(float)

    if "cluster_lat" in out.columns and "cluster_lon" in out.columns:
        out["has_cluster_coordinates"] = np.where(out["cluster_lat"].notna() & out["cluster_lon"].notna(), 1, 0)

    return out

def build_one_wave(folder: Path, wave: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    cfg = WAVE_CONFIGS[wave]
    report: list[dict] = []
    print(f"\n=== Building Wave {wave}: {folder} ===")

    household_core = build_household_core(folder, cfg, report)
    demographics = build_demographics(folder, cfg, report)
    labour = build_labour(folder, cfg, report)
    housing = build_module_raw(folder, cfg, cfg.housing_files, "housing_water", "housing_water", report, aggregate="first")
    consumption = build_consumption(folder, cfg, report)
    food = build_module_raw(folder, cfg, cfg.food_files, "food_security", "foodsec", report, aggregate="first")
    assets = build_module_raw(folder, cfg, cfg.assets_files, "assets", "asset", report, aggregate="sum")
    nonfarm = build_module_raw(folder, cfg, cfg.nonfarm_files, "nonfarm", "nonfarm", report, aggregate="sum")
    remit = build_module_raw(folder, cfg, cfg.remittance_files, "remittances", "remit", report, aggregate="sum")
    shocks = build_shocks(folder, cfg, report)
    geovars = build_spatial_coordinates(folder, cfg, household_core, report)

    community_exposure = build_community_exposure(folder, cfg, report)
    community_controls = build_community_controls(folder, cfg, report)
    if not community_exposure.empty and not community_controls.empty:
        community_full = community_exposure.merge(community_controls, on="community_id", how="outer")
    elif not community_exposure.empty:
        community_full = community_exposure
    else:
        community_full = community_controls
    community_hh = attach_community_to_households(folder, cfg, community_full, household_core, report)

    livestock_long, livestock = build_livestock(folder, cfg, report)
    plot_gps, agriculture = build_agriculture(folder, cfg, report)

    analysis = merge_blocks(
        household_core,
        [demographics, labour, housing, consumption, food, assets, nonfarm, remit, shocks, geovars, livestock, agriculture, community_hh],
        on="hhid",
    )

    analysis = add_final_harmonized_flags(analysis)

    # Final globally unique repeated-cross-section ID.
    analysis["hhid_wave"] = analysis["hhid"]
    analysis["hhid"] = analysis["wave"].astype(str) + "_" + analysis["hhid_wave"].astype(str)

    # Move key columns first.
    key_cols = [c for c in ["hhid", "hhid_wave", "original_hhid", "wave", "survey_year", "weight", "rural", "clusterid", "strataid", "region", "district", "ward", "community_id", "cluster_lat", "cluster_lon"] if c in analysis.columns]
    other_cols = [c for c in analysis.columns if c not in key_cols]
    analysis = analysis[key_cols + other_cols]

    outputs = {
        f"y{wave}_household_core": household_core,
        f"y{wave}_community_full": community_full,
        f"y{wave}_community_hh": community_hh,
        f"y{wave}_livestock_outcomes_long": livestock_long,
        f"y{wave}_livestock_full": livestock,
        f"y{wave}_plot_gps": plot_gps,
        f"y{wave}_agriculture_full": agriculture,
        f"y{wave}_analysis_household": analysis,
    }
    print(f"Wave {wave} analysis: {analysis.shape[0]:,} rows x {analysis.shape[1]:,} columns")
    return outputs, report


# ============================================================
# Documentation outputs
# ============================================================

def write_inventory(outputs: dict[str, pd.DataFrame], out_folder: Path) -> pd.DataFrame:
    rows = []
    for name, df in outputs.items():
        if df is None or df.empty:
            continue
        for i, col in enumerate(df.columns, start=1):
            rows.append({
                "dataset": name,
                "position": i,
                "variable": col,
                "dtype": str(df[col].dtype),
                "non_missing": int(df[col].notna().sum()),
                "n_rows": int(len(df)),
                "share_non_missing": float(df[col].notna().mean()) if len(df) else np.nan,
            })
    inv = pd.DataFrame(rows)
    inv.to_csv(out_folder / "nps_variable_inventory.csv", index=False)
    return inv


def write_column_presence(main_df: pd.DataFrame, out_folder: Path) -> pd.DataFrame:
    rows = []
    for wave, dfw in main_df.groupby("wave"):
        for col in main_df.columns:
            rows.append({
                "wave": int(wave),
                "variable": col,
                "non_missing": int(dfw[col].notna().sum()),
                "n_rows_wave": int(len(dfw)),
                "share_non_missing": float(dfw[col].notna().mean()) if len(dfw) else np.nan,
            })
    pres = pd.DataFrame(rows)
    pres.to_csv(out_folder / "nps_column_presence_by_wave.csv", index=False)
    return pres


def write_balanced_candidate_list(main_df: pd.DataFrame, out_folder: Path, min_share: float = 0.5) -> pd.DataFrame:
    pres = []
    for col in main_df.columns:
        if col in ["hhid", "hhid_wave", "original_hhid", "wave"]:
            continue
        shares = main_df.groupby("wave")[col].apply(lambda x: x.notna().mean())
        pres.append({
            "variable": col,
            "waves_present": int((shares >= min_share).sum()),
            "min_share_non_missing": float(shares.min()),
            "max_share_non_missing": float(shares.max()),
            "recommended_balanced_3_4": bool(all(shares.get(w, 0) >= min_share for w in [3, 4])),
            "recommended_balanced_1_4": bool(all(shares.get(w, 0) >= min_share for w in [1, 2, 3, 4])),
        })
    out = pd.DataFrame(pres).sort_values(["recommended_balanced_1_4", "recommended_balanced_3_4", "waves_present", "variable"], ascending=[False, False, False, True])
    out.to_csv(out_folder / "nps_balanced_variable_candidates.csv", index=False)
    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build harmonized Tanzania NPS multi-wave household dataset.")
    parser.add_argument("--wave1", type=str, default=None, help="Folder containing Wave 1 CSV files")
    parser.add_argument("--wave2", type=str, default=None, help="Folder containing Wave 2 CSV files")
    parser.add_argument("--wave3", type=str, default=None, help="Folder containing Wave 3 CSV files")
    parser.add_argument("--wave4", type=str, default=None, help="Folder containing Wave 4 CSV files")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--only-main", action="store_true", help="Write only main dataset and documentation")
    args = parser.parse_args()

    out_folder = Path(args.output)
    out_folder.mkdir(parents=True, exist_ok=True)

    wave_folders = {}
    for w in [1, 2, 3, 4]:
        p = getattr(args, f"wave{w}")
        if p:
            wave_folders[w] = Path(p)

    if not wave_folders:
        raise ValueError("Provide at least one wave folder: --wave1, --wave2, --wave3, --wave4")

    all_outputs: dict[str, pd.DataFrame] = {}
    build_report_rows: list[dict] = []
    analyses = []

    for wave, folder in sorted(wave_folders.items()):
        if not folder.exists():
            raise FileNotFoundError(f"Wave {wave} folder not found: {folder}")
        outputs, report = build_one_wave(folder, wave)
        all_outputs.update(outputs)
        build_report_rows.extend(report)
        analysis = outputs[f"y{wave}_analysis_household"]
        analyses.append(analysis)

        if not args.only_main:
            for name, df in outputs.items():
                if df is not None and not df.empty:
                    df.to_csv(out_folder / f"{name}.csv", index=False)

    main_df = pd.concat(analyses, ignore_index=True, sort=False)
    main_df.to_csv(out_folder / "nps_analysis_household_wave.csv", index=False)
    all_outputs["nps_analysis_household_wave"] = main_df

    build_report = pd.DataFrame(build_report_rows)
    build_report.to_csv(out_folder / "nps_build_report.csv", index=False)
    write_inventory(all_outputs, out_folder)
    write_column_presence(main_df, out_folder)
    write_balanced_candidate_list(main_df, out_folder)

    print("\nDone.")
    print(f"Output folder: {out_folder.resolve()}")
    print(f"Main dataset: nps_analysis_household_wave.csv = {main_df.shape[0]:,} rows x {main_df.shape[1]:,} columns")
    print("\nDocumentation files:")
    print("  - nps_variable_inventory.csv")
    print("  - nps_column_presence_by_wave.csv")
    print("  - nps_balanced_variable_candidates.csv")
    print("  - nps_build_report.csv")
    print("\nRecommended next checks:")
    print("  1. Open nps_build_report.csv and confirm all expected modules were read.")
    print("  2. Use nps_column_presence_by_wave.csv to choose balanced outcomes.")
    print("  3. Treat wave 1-2 livestock aliases as raw until variable labels are confirmed.")
    print("  4. Add Maasai-area / treatment / NDVI variables after this build.")


if __name__ == "__main__":
    main()
