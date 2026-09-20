#!/usr/bin/env python3
"""
NDVI robustness check for the Maasai / conservation Tanzania NPS analysis.

This is a deliberately simple standalone script. It starts from the same
main two-shock specification used in the baseline regressions and adds one
additional control: ndvi_control.

Expected inputs
---------------
1. Household-wave dataset, usually:
   output/nps_final/nps_relevant_household_wave.csv
2. EA-wave NDVI dataset created by ndvi_tanzania_waves_compatible.py, usually:
   data/ndvi/ndvi_ea_wave_compatible.csv

Merge key
---------
The NDVI file must be at EA-wave level and contain:
    ea, wave, ndvi_control

Main specification with NDVI
----------------------------
Y_iw = alpha + beta_1 Maasai_i + beta_2 Post2009_w + beta_3 Post2013_w
       + delta_1 Maasai_i x Post2009_w
       + delta_2 Maasai_i x Post2013_w
       + controls_iw + NDVI_ew + wave FE + district FE + error_iw

Standard errors are clustered at EA level.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    # Reuse the same wild-bootstrap routine as the main specification script
    # when both scripts are located in scripts/github.
    from run_maasai_main_specifications import wild_bootstrap_inference
except Exception:  # pragma: no cover - fallback for standalone inspection
    wild_bootstrap_inference = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_VERSION = "2026-06-28-ndvi-robustness-main-25km-v2"

HOUSEHOLD_FILE = "nps_relevant_household_wave.csv"
HHID = "hhid_wave"
WAVE = "wave"
FE_COLUMN = "district"
CLUSTER_COLUMN = "ea"
DEFAULT_TREATMENT = "is_maasai_25km"
DISTANCE = "distance_to_game_reserve_boundary_km"
INSIDE_RESERVE = "inside_game_reserve"
POST_2009 = "post_2009"
POST_2013 = "post_2013"
NDVI_CONTROL = "ndvi_control"
MAIN_BOUNDARY_KM = 75.0

SOURCE_CONTROL_COLUMNS = ["hh_members", "share_female", "n_children_u15", "n_adults_15plus"]
BASE_CONTROL_COLUMNS = ["hh_members", "share_female", "child_share"]

WELFARE_OUTCOMES = ["log_cons_pc", "log_cons_ae", "poverty_under_npl"]
LIVESTOCK_ASSET_OUTCOMES = [
    "livestock_tlu_per_ae",
    "livestock_total_per_ae",
    "cattle_change_1y",
    "cattle_decline_1y",
]
AGRICULTURE_RAW_OUTCOMES = [
    "ag_plot_count",
    "ag_land_ha",
    "ag_family_labor_days",
    "ag_hired_labor_days",
    "ag_total_labor_days",
    "ag_harvest_value_2ppp",
]
AGRICULTURE_OUTCOMES = [
    "ag_participation",
    "asinh_ag_plot_count",
    "asinh_ag_land_ha",
    "asinh_ag_family_labor_days",
    "asinh_ag_hired_labor_days",
    "asinh_ag_total_labor_days",
    "asinh_ag_harvest_value_2ppp",
]
SALES_RAW_OUTCOMES = [
    "livestock_number_sold_total",
    "livestock_sales_revenue_total",
    "cattle_number_sold_total",
    "cattle_sales_revenue_total",
]
SALES_OUTCOMES = [
    "livestock_any_sale",
    "asinh_livestock_number_sold_total",
    "asinh_livestock_sales_revenue_total",
    "cattle_any_sale",
    "asinh_cattle_number_sold_total",
    "asinh_cattle_sales_revenue_total",
]
PRODUCT_RAW_OUTCOMES = ["livestock_product_sales_revenue_total"]
PRODUCT_OUTCOMES = [
    "livestock_product_any_sale",
    "asinh_livestock_product_sales_revenue_total",
    "livestock_product_distress_sale_any",
    "livestock_product_log_price_gap_mean",
]
LIVESTOCK_STRESS_RAW_OUTCOMES = [
    "cost_fodder_livestock",
    "cost_lrum",
    "cost_hired_labor_livestock",
    "livestock_expenses",
    "value_milk_produced",
]
LIVESTOCK_STRESS_OUTCOMES = [
    "cattle_exit_1y",
    "cattle_loss_share_1y",
    "any_large_ruminant_disease",
    "asinh_cost_fodder_livestock",
    "asinh_cost_lrum",
    "asinh_cost_hired_labor_livestock",
    "asinh_livestock_expenses",
    "asinh_value_milk_produced",
    "share_livestock_prod_sold",
]
INCOME_REALLOCATION_OUTCOMES = [
    "livestock_income_share",
    "crop_income_share",
    "nonfarm_income_share",
    "self_employment_income_share",
    "nonagwage_income_share",
]

OUTCOME_FAMILIES = [
    ("welfare", WELFARE_OUTCOMES),
    ("livestock_assets", LIVESTOCK_ASSET_OUTCOMES),
    ("agriculture_switch", AGRICULTURE_OUTCOMES),
    ("livestock_sales", SALES_OUTCOMES),
    ("livestock_product_distress", PRODUCT_OUTCOMES),
    ("livestock_stress_productivity", LIVESTOCK_STRESS_OUTCOMES),
    ("income_reallocation", INCOME_REALLOCATION_OUTCOMES),
]

KEY_TABLE_OUTCOMES = [
    "log_cons_pc",
    "log_cons_ae",
    "poverty_under_npl",
    "cattle_change_1y",
    "cattle_decline_1y",
    "cattle_exit_1y",
    "cattle_loss_share_1y",
    "livestock_total_per_ae",
    "asinh_cost_lrum",
    "asinh_livestock_expenses",
    "any_large_ruminant_disease",
    "asinh_value_milk_produced",
    "livestock_income_share",
]

OUTCOME_LABELS = {
    "log_cons_pc": "Log consumption per capita",
    "log_cons_ae": "Log consumption per adult equivalent",
    "poverty_under_npl": "Poverty indicator",
    "livestock_tlu_per_ae": "Livestock TLU per adult equivalent",
    "livestock_total_per_ae": "Total livestock per adult equivalent",
    "cattle_change_1y": "Net cattle change",
    "cattle_decline_1y": "Any cattle decline",
    "cattle_exit_1y": "Cattle exit",
    "cattle_loss_share_1y": "Cattle loss share",
    "asinh_cost_lrum": "Large-ruminant treatment cost",
    "asinh_livestock_expenses": "Livestock expenses",
    "any_large_ruminant_disease": "Large-ruminant disease",
    "asinh_value_milk_produced": "Milk value produced",
    "livestock_income_share": "Livestock income share",
}

SHOCK_LABELS = {
    "post_2009_first_shift": "Post-2009 first shift",
    "post_2013_additional_shift": "Additional post-2013 shift",
    "total_post_2013": "Total post-2013 effect",
}

LOG_SCALE_OUTCOMES = {"log_cons_pc", "log_cons_ae", "livestock_product_log_price_gap_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NDVI-control robustness check.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ndvi-csv", required=True, type=Path)
    parser.add_argument("--household-file", default=HOUSEHOLD_FILE)
    parser.add_argument("--treatment-column", default=DEFAULT_TREATMENT)
    parser.add_argument("--main-boundary-km", default=MAIN_BOUNDARY_KM, type=float)
    parser.add_argument("--min-n", default=100, type=int)
    parser.add_argument("--min-clusters", default=2, type=int)
    parser.add_argument("--exclude-wave5", action="store_true")
    parser.add_argument("--wild-bootstrap-reps", default=999, type=int)
    parser.add_argument("--wild-bootstrap-seed", default=20260625, type=int)
    parser.add_argument("--wild-bootstrap-type", default="31", choices=["11", "13", "31", "33", "fnw11"])
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def add_asinh_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw in AGRICULTURE_RAW_OUTCOMES + SALES_RAW_OUTCOMES + PRODUCT_RAW_OUTCOMES + LIVESTOCK_STRESS_RAW_OUTCOMES:
        if raw in out.columns:
            out[f"asinh_{raw}"] = np.arcsinh(numeric(out[raw]).clip(lower=0))
    return out


def load_and_prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    household_path = args.input_dir / args.household_file
    if not household_path.exists():
        raise FileNotFoundError(household_path)
    if not args.ndvi_csv.exists():
        raise FileNotFoundError(args.ndvi_csv)

    hh = pd.read_csv(household_path, low_memory=False)
    ndvi = pd.read_csv(args.ndvi_csv, low_memory=False)

    require_columns(
        hh,
        [
            HHID,
            WAVE,
            FE_COLUMN,
            CLUSTER_COLUMN,
            args.treatment_column,
            POST_2009,
            POST_2013,
            DISTANCE,
            INSIDE_RESERVE,
            *SOURCE_CONTROL_COLUMNS,
        ],
        "Household dataset",
    )
    require_columns(ndvi, ["ea", "wave", NDVI_CONTROL], "NDVI dataset")

    hh["ea_key"] = clean_id(hh[CLUSTER_COLUMN])
    hh["wave_key"] = numeric(hh[WAVE]).astype("Int64")
    ndvi["ea_key"] = clean_id(ndvi["ea"])
    ndvi["wave_key"] = numeric(ndvi["wave"]).astype("Int64")
    ndvi[NDVI_CONTROL] = numeric(ndvi[NDVI_CONTROL])

    duplicates = int(ndvi.duplicated(["ea_key", "wave_key"]).sum())
    if duplicates:
        raise ValueError(
            f"NDVI file has {duplicates:,} duplicated ea-wave rows. "
            "Use the EA-wave collapsed NDVI dataset."
        )

    keep_ndvi = ["ea_key", "wave_key", NDVI_CONTROL]
    for optional in [
        "n_sample_points_with_ndvi",
        "n_sample_points_total",
        "mean_sample_points_per_ea_wave",
        "max_sample_points_per_ea_wave",
    ]:
        if optional in ndvi.columns:
            keep_ndvi.append(optional)

    before = len(hh)
    df = hh.merge(ndvi[keep_ndvi], on=["ea_key", "wave_key"], how="left")
    if len(df) != before:
        raise ValueError("Merge changed the number of household rows. Check NDVI merge keys.")

    if args.exclude_wave5:
        df = df.loc[numeric(df[WAVE]).ne(5)].copy()

    df = add_asinh_variables(df)
    df["fe_group"] = clean_id(df[FE_COLUMN])
    df["cluster_group"] = clean_id(df[CLUSTER_COLUMN])
    df["boundary_distance_10km"] = numeric(df[DISTANCE]).clip(lower=0) / 10.0

    for column in [args.treatment_column, POST_2009, POST_2013, INSIDE_RESERVE]:
        df[column] = numeric(df[column])

    roster_total = numeric(df["n_children_u15"]) + numeric(df["n_adults_15plus"])
    df["child_share"] = numeric(df["n_children_u15"]) / roster_total.where(roster_total > 0)

    diagnostics = []
    for wave, group in df.groupby(WAVE, dropna=False):
        diagnostics.append(
            {
                "wave": wave,
                "household_rows": int(len(group)),
                "rows_with_ndvi": int(group[NDVI_CONTROL].notna().sum()),
                "rows_missing_ndvi": int(group[NDVI_CONTROL].isna().sum()),
                "missing_rate": float(group[NDVI_CONTROL].isna().mean()),
                "unique_ea": int(group[CLUSTER_COLUMN].nunique(dropna=True)),
                "unique_ea_with_ndvi": int(group.loc[group[NDVI_CONTROL].notna(), CLUSTER_COLUMN].nunique(dropna=True)),
            }
        )
    diagnostic_df = pd.DataFrame(diagnostics)

    return df, diagnostic_df


def common_support_districts(sample: pd.DataFrame, treatment: str) -> pd.DataFrame:
    rows = []
    for district, group in sample.groupby("fe_group", sort=True):
        treated = int(numeric(group[treatment]).eq(1).sum())
        control = int(numeric(group[treatment]).eq(0).sum())
        rows.append({"district": district, "kept": int(treated > 0 and control > 0)})
    return pd.DataFrame(rows)


def scalar_value(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def coefficient_row(result, term: str, base: dict) -> dict:
    row = dict(base)
    row["reported_term"] = term
    if term not in result.params.index:
        row["status"] = "term_not_found"
        return row
    coef = float(result.params[term])
    se = float(result.bse[term])
    ci = result.conf_int(alpha=0.05)
    row.update(
        {
            "status": "ok",
            "coef": coef,
            "se": se,
            "t": float(result.tvalues[term]),
            "p_value": float(result.pvalues[term]),
            "ci95_low": float(ci.loc[term].iloc[0]),
            "ci95_high": float(ci.loc[term].iloc[1]),
            "r2": float(getattr(result, "rsquared", np.nan)),
        }
    )
    if base.get("outcome") in LOG_SCALE_OUTCOMES:
        row["percent_effect"] = 100.0 * (np.exp(coef) - 1.0)
    return row


def total_post2013_row(result, early_term: str, additional_term: str, base: dict) -> dict:
    row = dict(base)
    row["reported_term"] = "maasai_associated_X_total_post2013"
    row["wild_bootstrap_status"] = "not_computed_for_linear_combination"
    if early_term not in result.params.index or additional_term not in result.params.index:
        row["status"] = "term_not_found"
        return row
    names = list(result.params.index)
    restriction = np.zeros(len(names))
    restriction[names.index(early_term)] = 1.0
    restriction[names.index(additional_term)] = 1.0
    try:
        test = result.t_test(restriction)
        coef = scalar_value(test.effect)
        se = scalar_value(test.sd)
        ci_low, ci_high = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)[:2]
        row.update(
            {
                "status": "ok",
                "coef": coef,
                "se": se,
                "t": scalar_value(test.tvalue),
                "p_value": scalar_value(test.pvalue),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "r2": float(getattr(result, "rsquared", np.nan)),
            }
        )
        if base.get("outcome") in LOG_SCALE_OUTCOMES:
            row["percent_effect"] = 100.0 * (np.exp(coef) - 1.0)
    except Exception as error:
        row["status"] = f"linear_combination_error:{type(error).__name__}"
        row["error"] = str(error)
    return row


def run_one_outcome(
    df: pd.DataFrame,
    outcome: str,
    outcome_family: str,
    sample_name: str,
    args: argparse.Namespace,
) -> list[dict]:
    treatment = args.treatment_column
    interaction_2009 = "maasai_associated_X_post2009"
    interaction_2013 = "maasai_associated_X_post2013"
    report_terms = [interaction_2009, interaction_2013]

    required = [
        outcome,
        treatment,
        POST_2009,
        POST_2013,
        INSIDE_RESERVE,
        *BASE_CONTROL_COLUMNS,
        "boundary_distance_10km",
        NDVI_CONTROL,
        WAVE,
        "fe_group",
        "cluster_group",
    ]
    if any(c not in df.columns for c in required):
        return []

    work = df.copy()
    work[interaction_2009] = work[treatment] * work[POST_2009]
    work[interaction_2013] = work[treatment] * work[POST_2013]

    rhs = [
        treatment,
        POST_2009,
        POST_2013,
        INSIDE_RESERVE,
        interaction_2009,
        interaction_2013,
        *BASE_CONTROL_COLUMNS,
        "boundary_distance_10km",
        NDVI_CONTROL,
        f"C({WAVE})",
        "C(fe_group)",
    ]
    formula = f"{outcome} ~ " + " + ".join(rhs)

    base = {
        "model": "main_two_shock_with_ndvi_control",
        "outcome_family": outcome_family,
        "outcome": outcome,
        "sample": sample_name,
        "treatment": treatment,
        "formula": formula,
        "ndvi_control_included": 1,
        "ndvi_control_level": "ea_wave",
        "se_type": f"cluster:{CLUSTER_COLUMN}",
    }

    try:
        model = smf.ols(formula, data=work.reset_index(drop=True), missing="drop")
        estimation_data = work.reset_index(drop=True).loc[pd.Index(model.data.row_labels)].copy()
        n = int(model.endog.shape[0])
        clusters = int(estimation_data["cluster_group"].nunique(dropna=True))
        base.update(
            {
                "n": n,
                "clusters": clusters,
                "mean_outcome": float(numeric(estimation_data[outcome]).mean()),
                "mean_ndvi_control": float(numeric(estimation_data[NDVI_CONTROL]).mean()),
            }
        )
        if n < args.min_n:
            return [{**base, "reported_term": term, "status": f"too_few_obs_min_{args.min_n}"} for term in report_terms]
        if clusters < args.min_clusters:
            return [{**base, "reported_term": term, "status": "too_few_clusters"} for term in report_terms]

        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": estimation_data["cluster_group"].astype(str), "use_correction": True, "df_correction": True},
            use_t=True,
        )
        rows = [coefficient_row(result, term, base) for term in report_terms]
        if wild_bootstrap_inference is not None:
            boot = wild_bootstrap_inference(outcome, rhs, estimation_data, report_terms, args)
            for row in rows:
                row.update(boot.get(row.get("reported_term"), {}))
        else:
            for row in rows:
                row["wild_bootstrap_status"] = "main_bootstrap_function_not_available"
        rows.append(total_post2013_row(result, report_terms[0], report_terms[1], base))
        return rows
    except Exception as error:
        return [{**base, "reported_term": term, "status": f"fit_error:{type(error).__name__}", "error": str(error)} for term in report_terms]


def run_sample(df: pd.DataFrame, sample_name: str, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    for family, outcomes in OUTCOME_FAMILIES:
        for outcome in outcomes:
            rows.extend(run_one_outcome(df, outcome, family, sample_name, args))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["shock"] = out["reported_term"].map(
        {
            "maasai_associated_X_post2009": "post_2009_first_shift",
            "maasai_associated_X_post2013": "post_2013_additional_shift",
            "maasai_associated_X_total_post2013": "total_post_2013",
        }
    )
    return out


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


def latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(str(outcome), str(outcome).replace("_", " "))


def shock_label(shock: str) -> str:
    return SHOCK_LABELS.get(str(shock), str(shock).replace("_", " "))


def format_cell(row: pd.Series | None) -> tuple[str, str, str]:
    if row is None or row.empty:
        return "", "", ""
    coef = pd.to_numeric(pd.Series([row.get("coef")]), errors="coerce").iloc[0]
    se = pd.to_numeric(pd.Series([row.get("se")]), errors="coerce").iloc[0]
    p = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
    wb_p = pd.to_numeric(pd.Series([row.get("wild_bootstrap_p_value")]), errors="coerce").iloc[0]
    wb_status = str(row.get("wild_bootstrap_status", "")).lower()
    if pd.isna(coef):
        return "", "", ""
    coef_text = f"{coef:.3f}{significance_stars(p)}"
    se_text = f"({se:.3f})" if pd.notna(se) else ""
    wb_text = f"[{wb_p:.3f}]" if pd.notna(wb_p) and wb_status == "ok" else ""
    return coef_text, se_text, wb_text


def make_latex_table(results: pd.DataFrame, path: Path) -> None:
    if results.empty:
        path.write_text("% No results available.\n", encoding="utf-8")
        return
    df = results.loc[results["status"].fillna("ok").eq("ok")].copy()
    samples = [s for s in ["full_risk_set", "districts_with_treated_and_control"] if s in set(df["sample"])]
    if not samples:
        samples = sorted(df["sample"].dropna().unique().tolist())
    sample_labels = {
        "full_risk_set": "Full risk set",
        "districts_with_treated_and_control": "Common-support districts",
    }

    lines = []
    lines.append(r"\begin{table}[!htbp]\centering")
    lines.append(r"\caption{Robustness check with NDVI control}")
    lines.append(r"\label{tab:ndvi-robustness}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l" + "c" * len(samples) + r"}")
    lines.append(r"\toprule")
    lines.append("Outcome and coefficient & " + " & ".join(latex_escape(sample_labels.get(s, s)) for s in samples) + r" \\")
    lines.append(r"\midrule")

    for outcome in KEY_TABLE_OUTCOMES:
        for shock in ["post_2009_first_shift", "post_2013_additional_shift", "total_post_2013"]:
            coef_cells = []
            se_cells = []
            wb_cells = []
            for sample in samples:
                sub = df.loc[df["sample"].eq(sample) & df["outcome"].eq(outcome) & df["shock"].eq(shock)]
                row = sub.iloc[0] if not sub.empty else None
                coef, se, wb = format_cell(row)
                coef_cells.append(coef)
                se_cells.append(se)
                wb_cells.append(wb)
            if not any(coef_cells):
                continue
            row_label = latex_escape(f"{outcome_label(outcome)} -- {shock_label(shock)}")
            lines.append(row_label + " & " + " & ".join(coef_cells) + " \\")
            lines.append(" " + " & " + " & ".join(se_cells) + " \\")
            if any(wb_cells):
                lines.append(r"\emph{Wild p-value} & " + " & ".join(wb_cells) + " \\")

    lines.append(r"\midrule")
    for stat, label in [("n", "Observations"), ("clusters", "EA clusters")]:
        cells = []
        for sample in samples:
            sub = df.loc[df["sample"].eq(sample)]
            values = pd.to_numeric(sub[stat], errors="coerce").dropna() if stat in sub.columns else pd.Series(dtype=float)
            cells.append(f"{int(values.iloc[0]):,}" if len(values) else "")
        lines.append(latex_escape(label) + " & " + " & ".join(cells) + r" \\")
    cells = []
    for sample in samples:
        sub = df.loc[df["sample"].eq(sample)]
        values = pd.to_numeric(sub["mean_ndvi_control"], errors="coerce").dropna() if "mean_ndvi_control" in sub.columns else pd.Series(dtype=float)
        cells.append(f"{values.iloc[0]:.3f}" if len(values) else "")
    lines.append(r"Mean NDVI control & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\begin{minipage}{0.96\textwidth}\vspace{0.3em}\footnotesize "
        r"Notes: this table re-estimates the main two-shock specification after adding NDVI as an EA-wave control. "
        r"Standard errors clustered at enumeration-area level are in parentheses. "
        r"Wild-cluster-bootstrap p-values, when available, are reported in square brackets. "
        r"Significance stars use ordinary clustered p-values: * p$<$0.10, ** p$<$0.05, *** p$<$0.01. "
        r"\end{minipage}"
    )
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    print(f"run_maasai_ndvi_robustness version: {SCRIPT_VERSION}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, diagnostics = load_and_prepare_data(args)
    # Diagnostics are printed and embedded in the raw results where possible;
    # no separate raw diagnostics CSV is written to keep one raw CSV output.

    main_sample = df.loc[numeric(df[DISTANCE]).le(args.main_boundary_km)].copy()
    if main_sample.empty:
        raise ValueError("No observations remain in the reserve-boundary risk set.")

    district_table = common_support_districts(main_sample, args.treatment_column)
    # No separate raw support CSV: raw regression output below is the single CSV.
    kept = district_table.loc[district_table["kept"].eq(1), "district"]
    support_sample = main_sample.loc[main_sample["fe_group"].isin(kept)].copy()

    frames = [run_sample(main_sample, "full_risk_set", args)]
    if not support_sample.empty:
        frames.append(run_sample(support_sample, "districts_with_treated_and_control", args))
    results = pd.concat(frames, ignore_index=True, sort=False)

    results.to_csv(args.output_dir / "table_ndvi_control_robustness_raw.csv", index=False)
    make_latex_table(results, args.output_dir / "table_ndvi_control_robustness.tex")

    print("\nDone.")
    print(f"Raw results:        {args.output_dir / 'table_ndvi_control_robustness_raw.csv'}")
    print(f"LaTeX table:        {args.output_dir / 'table_ndvi_control_robustness.tex'}")
    print("Merge diagnostics:  printed below")
    print("\nNDVI merge diagnostics by wave:")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
