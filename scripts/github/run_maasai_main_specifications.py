#!/usr/bin/env python3
"""
Core repeated-cross-section regressions for the Tanzania NPS conservation and
community-exposure project.

The script estimates one parsimonious two-shock specification: treated-by-post-2009 and the additional treated-by-post-2013 shift for:

1. household welfare;
2. livestock assets;
3. agricultural participation and input reallocation;
4. live-animal sale margins;
5. livestock-product distress pricing relative to supplied EA medians;
6. livestock disease, maintenance costs, and pastoral productivity;
7. income-source reallocation.

Each outcome is estimated in the full reserve-boundary risk set and in the
subset of districts containing both treated and control observations. No
household fixed effects, pressure gradients, triple interactions, quadratic
distance terms, or inside-specific slopes are used.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_VERSION = "2026-06-28-two-shock-main-25km-v11"

HOUSEHOLD_FILE = "nps_relevant_household_wave.csv"
HHID = "hhid_wave"
WAVE = "wave"
FE_COLUMN = "district"
CLUSTER_COLUMN = "ea"
REGION_COLUMN = "region"

DEFAULT_TREATMENT = "is_maasai_25km"
ALLOWED_TREATMENTS = ["is_maasai_25km", "is_maasai_10km", "inside_maasai_ward"]
DISTANCE = "distance_to_game_reserve_boundary_km"
INSIDE_RESERVE = "inside_game_reserve"
POST_2009 = "post_2009"
POST_2013 = "post_2013"
POST_COLUMNS = [POST_2009, POST_2013]
MAIN_BOUNDARY_KM = 75.0

DEFAULT_WILD_BOOTSTRAP_REPS = 999
DEFAULT_WILD_BOOTSTRAP_SEED = 20260625
DEFAULT_WILD_BOOTSTRAP_TYPE = "31"

WELFARE_OUTCOMES = [
    "log_cons_pc",
    "log_cons_ae",
    "poverty_under_npl",
]

LIVESTOCK_ASSET_OUTCOMES = [
    "livestock_tlu_per_ae",
    "livestock_total_per_ae",
    "cattle_change_1y",
    "cattle_decline_1y",
]

AGRICULTURE_RAW_OUTCOMES = [
    "ag_participation",
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
    "livestock_any_sale",
    "livestock_number_sold_total",
    "livestock_sales_revenue_total",
    "cattle_any_sale",
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

PRODUCT_PRICE_RAW_OUTCOMES = [
    "livestock_product_any_sale",
    "livestock_product_sales_revenue_total",
    "livestock_product_distress_sale_any",
    "livestock_product_log_price_gap_mean",
]

PRODUCT_PRICE_OUTCOMES = [
    "livestock_product_any_sale",
    "asinh_livestock_product_sales_revenue_total",
    "livestock_product_distress_sale_any",
    "livestock_product_log_price_gap_mean",
]

LIVESTOCK_STRESS_RAW_OUTCOMES = [
    "cattle_exit_1y",
    "cattle_loss_share_1y",
    "any_large_ruminant_disease",
    "cost_fodder_livestock",
    "cost_lrum",
    "cost_hired_labor_livestock",
    "livestock_expenses",
    "value_milk_produced",
    "share_livestock_prod_sold",
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

SOURCE_CONTROL_COLUMNS = [
    "hh_members",
    "share_female",
    "n_children_u15",
    "n_adults_15plus",
]

BASE_CONTROL_COLUMNS = [
    "hh_members",
    "share_female",
    "child_share",
]

LOG_SCALE_OUTCOMES = {
    "log_cons_pc",
    "log_cons_ae",
    "livestock_product_log_price_gap_mean",
}

PRODUCT_BENCHMARK_OUTCOMES = {
    "livestock_product_distress_sale_any",
    "livestock_product_log_price_gap_mean",
}
PREVIOUS_CATTLE_OWNER_OUTCOMES = {"cattle_exit_1y", "cattle_loss_share_1y"}
INCOME_SHARE_OUTCOMES = set(INCOME_REALLOCATION_OUTCOMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated-cross-section welfare and mechanism regressions."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--household-file", default=HOUSEHOLD_FILE)
    parser.add_argument(
        "--treatment-column",
        default=DEFAULT_TREATMENT,
        choices=ALLOWED_TREATMENTS,
    )
    parser.add_argument("--main-boundary-km", default=MAIN_BOUNDARY_KM, type=float)
    parser.add_argument("--min-n", default=100, type=int)
    parser.add_argument(
        "--min-clusters",
        default=20,
        type=int,
        help="Warning threshold only; estimation requires at least two EA clusters.",
    )
    parser.add_argument(
        "--exclude-wave5",
        action="store_true",
        help="Drop Wave 5 from the regression sample without changing the extracted dataset.",
    )
    parser.add_argument(
        "--wild-bootstrap-reps",
        default=DEFAULT_WILD_BOOTSTRAP_REPS,
        type=int,
        help="Set to 0 to disable. Use 9999 for final inference.",
    )
    parser.add_argument(
        "--wild-bootstrap-seed",
        default=DEFAULT_WILD_BOOTSTRAP_SEED,
        type=int,
    )
    parser.add_argument(
        "--wild-bootstrap-type",
        default=DEFAULT_WILD_BOOTSTRAP_TYPE,
        choices=["11", "13", "31", "33", "fnw11"],
    )
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def add_bh_q_values(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    out["q_value_bh"] = np.nan
    if out.empty or p_col not in out.columns:
        return out

    groups = out.groupby("sample", dropna=False).groups if "sample" in out.columns else {"all": out.index}
    for indices in groups.values():
        valid = numeric(out.loc[indices, p_col]).notna()
        valid_indices = pd.Index(indices)[valid.to_numpy()]
        if len(valid_indices) == 0:
            continue
        p = numeric(out.loc[valid_indices, p_col]).to_numpy()
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        restored = np.empty(m)
        restored[order] = adjusted
        out.loc[valid_indices, "q_value_bh"] = restored
    return out


def prepare_data(household: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    df = household.copy()
    treatment = args.treatment_column

    required = [
        HHID,
        WAVE,
        FE_COLUMN,
        CLUSTER_COLUMN,
        REGION_COLUMN,
        treatment,
        *POST_COLUMNS,
        DISTANCE,
        INSIDE_RESERVE,
        *SOURCE_CONTROL_COLUMNS,
        *WELFARE_OUTCOMES,
        *LIVESTOCK_ASSET_OUTCOMES,
        *AGRICULTURE_RAW_OUTCOMES,
        *SALES_RAW_OUTCOMES,
        *PRODUCT_PRICE_RAW_OUTCOMES,
        *LIVESTOCK_STRESS_RAW_OUTCOMES,
        *INCOME_REALLOCATION_OUTCOMES,
    ]
    require_columns(df, required, "Household dataset")

    df[HHID] = clean_id(df[HHID])
    if df[HHID].isna().any():
        raise ValueError(f"{HHID} contains missing values.")
    duplicated = int(df[HHID].duplicated().sum())
    if duplicated:
        raise ValueError(f"Household dataset has {duplicated:,} duplicated {HHID} values.")

    df["fe_group"] = clean_id(df[FE_COLUMN])
    df["cluster_group"] = clean_id(df[CLUSTER_COLUMN])
    df["region_group"] = clean_id(df[REGION_COLUMN])

    for column in [treatment, *POST_COLUMNS, INSIDE_RESERVE]:
        df[column] = numeric(df[column])
        invalid = df[column].notna() & ~df[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{column} contains {int(invalid.sum())} values outside 0/1.")

    numeric_columns = [
        WAVE,
        DISTANCE,
        *SOURCE_CONTROL_COLUMNS,
        *WELFARE_OUTCOMES,
        *LIVESTOCK_ASSET_OUTCOMES,
        *AGRICULTURE_RAW_OUTCOMES,
        *SALES_RAW_OUTCOMES,
        *PRODUCT_PRICE_RAW_OUTCOMES,
        *LIVESTOCK_STRESS_RAW_OUTCOMES,
        *INCOME_REALLOCATION_OUTCOMES,
    ]
    for column in numeric_columns:
        df[column] = numeric(df[column])

    roster_total = df["n_children_u15"] + df["n_adults_15plus"]
    df["child_share"] = df["n_children_u15"] / roster_total.where(roster_total > 0)

    distance = df[DISTANCE].clip(lower=0)
    df[DISTANCE] = distance
    # Distance is measured as the shortest distance to the nearest reserve
    # boundary and rescaled so one unit corresponds to 10 km.
    df["boundary_distance_10km"] = distance / 10.0

    for raw in AGRICULTURE_RAW_OUTCOMES[1:]:
        df[f"asinh_{raw}"] = np.arcsinh(df[raw].clip(lower=0))

    asinh_outcomes = [
        "livestock_number_sold_total",
        "livestock_sales_revenue_total",
        "cattle_number_sold_total",
        "cattle_sales_revenue_total",
        "livestock_product_sales_revenue_total",
        "cost_fodder_livestock",
        "cost_lrum",
        "cost_hired_labor_livestock",
        "livestock_expenses",
        "value_milk_produced",
    ]
    for raw in asinh_outcomes:
        df[f"asinh_{raw}"] = np.arcsinh(df[raw].clip(lower=0))

    metadata = {
        "treatment": treatment,
        "fe_source": FE_COLUMN,
        "cluster_source": CLUSTER_COLUMN,
        "region_source": REGION_COLUMN,
        "controls": "+".join(BASE_CONTROL_COLUMNS),
    }
    return df, metadata


def support_counts_for_post(df: pd.DataFrame, treatment: str, post_column: str, prefix: str) -> dict:
    treated = numeric(df[treatment]).eq(1)
    control = numeric(df[treatment]).eq(0)
    post = numeric(df[post_column]).eq(1)
    pre = numeric(df[post_column]).eq(0)
    return {
        f"n_treated_pre_{prefix}": int((treated & pre).sum()),
        f"n_treated_post_{prefix}": int((treated & post).sum()),
        f"n_control_pre_{prefix}": int((control & pre).sum()),
        f"n_control_post_{prefix}": int((control & post).sum()),
        f"n_treated_pre_clusters_{prefix}": int(
            df.loc[treated & pre, "cluster_group"].nunique(dropna=True)
        ),
        f"n_treated_post_clusters_{prefix}": int(
            df.loc[treated & post, "cluster_group"].nunique(dropna=True)
        ),
    }


def support_counts(df: pd.DataFrame, treatment: str) -> dict:
    treated = numeric(df[treatment]).eq(1)
    out = {
        "n_treated_clusters": int(df.loc[treated, "cluster_group"].nunique(dropna=True)),
    }
    out.update(support_counts_for_post(df, treatment, POST_2009, "post2009"))
    out.update(support_counts_for_post(df, treatment, POST_2013, "post2013"))
    return out


def empty_did_cells(counts: dict) -> list[str]:
    names = [
        "n_treated_pre_post2009",
        "n_treated_post_post2009",
        "n_control_pre_post2009",
        "n_control_post_post2009",
        "n_treated_pre_post2013",
        "n_treated_post_post2013",
        "n_control_pre_post2013",
        "n_control_post_post2013",
    ]
    return [name for name in names if counts.get(name, 0) == 0]


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
    if not np.isfinite(se):
        row.update({"status": "invalid_cluster_covariance", "coef": coef, "se": se})
        return row

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
            "cluster_df_inference": float(
                getattr(result, "df_resid_inference", np.nan)
            ),
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
        t_stat = scalar_value(test.tvalue)
        p_value = scalar_value(test.pvalue)
        ci_low, ci_high = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)[:2]
        row.update(
            {
                "status": "ok",
                "coef": coef,
                "se": se,
                "t": t_stat,
                "p_value": p_value,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "r2": float(getattr(result, "rsquared", np.nan)),
                "cluster_df_inference": float(getattr(result, "df_resid_inference", np.nan)),
            }
        )
        if base.get("outcome") in LOG_SCALE_OUTCOMES:
            row["percent_effect"] = 100.0 * (np.exp(coef) - 1.0)
        return row
    except Exception as error:
        row["status"] = f"linear_combination_error:{type(error).__name__}"
        row["error"] = str(error)
        return row


def build_wild_bootstrap_model(
    outcome: str,
    rhs_terms: list[str],
    estimation_data: pd.DataFrame,
    report_terms: list[str],
):
    bootstrap_rhs = [term for term in rhs_terms if term not in POST_COLUMNS]
    model = smf.ols(
        f"{outcome} ~ " + " + ".join(bootstrap_rhs),
        data=estimation_data,
        missing="drop",
    )
    x = np.asarray(model.exog, dtype=float)
    rank = np.linalg.matrix_rank(x)
    if rank == x.shape[1]:
        return model, "post_removed_as_wave_fe_collinear"

    names = list(model.exog_names)
    priority_names = [
        *[term for term in report_terms if term in names],
        *(["Intercept"] if "Intercept" in names else []),
        *names,
    ]
    priority_indices: list[int] = []
    for name in priority_names:
        index = names.index(name)
        if index not in priority_indices:
            priority_indices.append(index)

    selected: list[int] = []
    current_rank = 0
    for index in priority_indices:
        candidate = selected + [index]
        candidate_rank = np.linalg.matrix_rank(x[:, candidate])
        if candidate_rank > current_rank:
            selected.append(index)
            current_rank = candidate_rank
        if current_rank == rank:
            break

    reduced_x = pd.DataFrame(
        x[:, selected],
        columns=[names[index] for index in selected],
        index=estimation_data.index,
    )
    return (
        sm.OLS(np.asarray(model.endog, dtype=float), reduced_x),
        "post_removed_and_full_rank_basis_used",
    )


def wild_bootstrap_inference(
    outcome: str,
    rhs_terms: list[str],
    estimation_data: pd.DataFrame,
    terms: list[str],
    args: argparse.Namespace,
) -> dict[str, dict]:
    if args.wild_bootstrap_reps <= 0:
        return {term: {"wild_bootstrap_status": "disabled"} for term in terms}
    try:
        from wildboottest.wildboottest import wildboottest
    except ImportError:
        return {term: {"wild_bootstrap_status": "package_not_installed"} for term in terms}

    try:
        model, design_note = build_wild_bootstrap_model(
            outcome, rhs_terms, estimation_data, terms
        )
        names = list(model.exog_names)
        clusters = pd.factorize(
            estimation_data["cluster_group"].astype(str), sort=True
        )[0].astype(np.int64)

        output: dict[str, dict] = {}
        for offset, term in enumerate(terms):
            if term not in names:
                output[term] = {
                    "wild_bootstrap_status": "term_not_found_after_rank_reduction",
                    "wild_bootstrap_design_note": design_note,
                }
                continue
            result = wildboottest(
                model,
                param=term,
                cluster=clusters,
                B=args.wild_bootstrap_reps,
                bootstrap_type=args.wild_bootstrap_type,
                weights_type="rademacher",
                impose_null=True,
                seed=args.wild_bootstrap_seed + offset,
                adj=True,
                cluster_adj=True,
                parallel=False,
                show=False,
            )
            output[term] = {
                "wild_bootstrap_status": "ok",
                "wild_bootstrap_design_note": design_note,
                "wild_bootstrap_reps": int(args.wild_bootstrap_reps),
                "wild_bootstrap_type": args.wild_bootstrap_type,
                "wild_bootstrap_t": scalar_value(result.loc[term, "statistic"]),
                "wild_bootstrap_p_value": scalar_value(result.loc[term, "p-value"]),
            }
        return output
    except Exception as error:
        return {
            term: {
                "wild_bootstrap_status": f"error:{type(error).__name__}",
                "wild_bootstrap_error": str(error),
            }
            for term in terms
        }


def fit_ols(
    df: pd.DataFrame,
    outcome: str,
    rhs_terms: list[str],
    report_terms: list[str],
    outcome_family: str,
    sample_name: str,
    args: argparse.Namespace,
    metadata: Optional[dict] = None,
) -> list[dict]:
    metadata = metadata or {}
    base = {
        "model": "main_two_shock_linear_boundary_distance",
        "outcome_family": outcome_family,
        "outcome": outcome,
        "sample": sample_name,
        "treatment": args.treatment_column,
        "post_specification": "post_2009_plus_additional_post_2013",
        "outcome_sample_definition": (
            "ea_benchmarked_livestock_product_sellers_only"
            if outcome in PRODUCT_BENCHMARK_OUTCOMES
            else "previous_cattle_owners_only"
            if outcome in PREVIOUS_CATTLE_OWNER_OUTCOMES
            else "positive_total_income_only"
            if outcome in INCOME_SHARE_OUTCOMES
            else "all_complete_household_observations"
        ),
        **metadata,
    }

    formula = f"{outcome} ~ " + " + ".join(rhs_terms)
    base["formula"] = formula

    try:
        working = df.reset_index(drop=True).copy()
        model = smf.ols(formula, data=working, missing="drop")
        estimation_data = working.loc[pd.Index(model.data.row_labels)].copy()
        n = int(model.endog.shape[0])
        n_clusters = int(estimation_data["cluster_group"].nunique(dropna=True))
        counts = support_counts(estimation_data, args.treatment_column)
        unsupported = empty_did_cells(counts)

        base.update(
            {
                "n": n,
                "clusters": n_clusters,
                **counts,
                "unsupported_did_cells": "+".join(unsupported),
                "few_cluster_warning": int(2 <= n_clusters < args.min_clusters),
                "few_treated_cluster_warning": int(
                    0 < counts["n_treated_clusters"] < args.min_clusters
                ),
                "mean_outcome": float(numeric(estimation_data[outcome]).mean()),
            }
        )

        if n < args.min_n:
            return [
                {**base, "reported_term": term, "status": f"too_few_obs_min_{args.min_n}"}
                for term in report_terms
            ]
        if n_clusters < 2:
            return [
                {**base, "reported_term": term, "status": "cannot_cluster_fewer_than_2_clusters"}
                for term in report_terms
            ]
        if unsupported:
            return [
                {**base, "reported_term": term, "status": "unsupported_empty_did_cell"}
                for term in report_terms
            ]

        result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": estimation_data["cluster_group"].astype(str),
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )
        base["se_type"] = f"cluster:{CLUSTER_COLUMN}"
        base["post_role"] = "post_2009_and_post_2013_explicit_but_absorbed_by_wave_fixed_effects"

        wild = wild_bootstrap_inference(
            outcome, rhs_terms, estimation_data, report_terms, args
        )
        rows: list[dict] = []
        for term in report_terms:
            row = coefficient_row(result, term, base)
            row.update(wild.get(term, {}))
            rows.append(row)
        if len(report_terms) == 2:
            rows.append(total_post2013_row(result, report_terms[0], report_terms[1], base))
        return rows
    except Exception as error:
        return [
            {
                **base,
                "reported_term": term,
                "status": f"fit_error:{type(error).__name__}",
                "error": str(error),
            }
            for term in report_terms
        ]


def run_outcome_family(
    sample: pd.DataFrame,
    outcomes: Iterable[str],
    family: str,
    sample_name: str,
    args: argparse.Namespace,
    common_metadata: dict,
) -> pd.DataFrame:
    df = sample.copy()
    treatment = args.treatment_column
    interaction_2009 = "maasai_associated_X_post2009"
    interaction_2013 = "maasai_associated_X_post2013"
    report_terms = [interaction_2009, interaction_2013]

    df[interaction_2009] = df[treatment] * df[POST_2009]
    df[interaction_2013] = df[treatment] * df[POST_2013]

    rhs = [
        treatment,
        POST_2009,
        POST_2013,
        INSIDE_RESERVE,
        interaction_2009,
        interaction_2013,
        *BASE_CONTROL_COLUMNS,
        "boundary_distance_10km",
        f"C({WAVE})",
        "C(fe_group)",
    ]

    metadata = {
        **common_metadata,
        "distance_function": "linear_distance_to_nearest_reserve_boundary",
        "inside_reserve_role": "level_control_only",
        "reported_parameter": "maasai_associated_x_post2009_and_additional_post2013",
    }

    rows: list[dict] = []
    for outcome in outcomes:
        rows.extend(
            fit_ols(
                df,
                outcome,
                rhs,
                report_terms,
                family,
                sample_name,
                args,
                metadata,
            )
        )
    out = add_bh_q_values(pd.DataFrame(rows))
    if not out.empty and "reported_term" in out.columns:
        out["shock"] = out["reported_term"].map(
            {
                interaction_2009: "post_2009_first_shift",
                interaction_2013: "post_2013_additional_shift",
                "maasai_associated_X_total_post2013": "total_post_2013",
            }
        )
        out["shock_interpretation"] = out["reported_term"].map(
            {
                interaction_2009: "Differential change after the 2009 reform period",
                interaction_2013: "Additional differential change after the 2013 reform period",
                "maasai_associated_X_total_post2013": "Total differential change in the post-2013 period",
            }
        )
    return out


def common_support_districts(sample: pd.DataFrame, treatment: str) -> pd.DataFrame:
    rows = []
    for district, group in sample.groupby("fe_group", sort=True):
        treated = int(numeric(group[treatment]).eq(1).sum())
        control = int(numeric(group[treatment]).eq(0).sum())
        rows.append(
            {
                "district": district,
                "treated_observations": treated,
                "control_observations": control,
                "kept": int(treated > 0 and control > 0),
            }
        )
    return pd.DataFrame(rows)


def build_support_table(
    samples: dict[str, pd.DataFrame],
    metadata: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for name, data in samples.items():
        rows.append(
            {
                "sample": name,
                "n": len(data),
                "clusters": int(data["cluster_group"].nunique(dropna=True)),
                "fixed_effect_groups": int(data["fe_group"].nunique(dropna=True)),
                **support_counts(data, args.treatment_column),
                **metadata,
            }
        )
    return pd.DataFrame(rows)


OUTCOME_LABELS_FOR_TABLES = {
    "log_cons_pc": "Log consumption per capita",
    "log_cons_ae": "Log consumption per adult equivalent",
    "poverty_under_npl": "Poverty indicator",
    "livestock_tlu_per_ae": "Livestock TLU per adult equivalent",
    "livestock_total_per_ae": "Total livestock per adult equivalent",
    "cattle_change_1y": "Net cattle change",
    "cattle_decline_1y": "Any cattle decline",
    "cattle_exit_1y": "Cattle exit",
    "cattle_loss_share_1y": "Cattle loss share",
    "any_large_ruminant_disease": "Large-ruminant disease",
    "asinh_cost_lrum": "Large-ruminant treatment cost",
    "asinh_livestock_expenses": "Livestock expenses",
    "asinh_value_milk_produced": "Milk value produced",
    "livestock_income_share": "Livestock income share",
    "crop_income_share": "Crop income share",
    "nonfarm_income_share": "Non-farm income share",
    "self_employment_income_share": "Self-employment income share",
    "nonagwage_income_share": "Non-agricultural wage income share",
    "ag_participation": "Agricultural participation",
    "asinh_ag_plot_count": "Number of agricultural plots",
    "asinh_ag_land_ha": "Agricultural land",
    "asinh_ag_family_labor_days": "Family farm labour days",
    "asinh_ag_hired_labor_days": "Hired farm labour days",
    "asinh_ag_total_labor_days": "Total farm labour days",
    "asinh_ag_harvest_value_2ppp": "Harvest value",
    "livestock_any_sale": "Any livestock sale",
    "asinh_livestock_number_sold_total": "Number of livestock sold",
    "asinh_livestock_sales_revenue_total": "Livestock sales revenue",
    "cattle_any_sale": "Any cattle sale",
    "asinh_cattle_number_sold_total": "Number of cattle sold",
    "asinh_cattle_sales_revenue_total": "Cattle sales revenue",
    "livestock_product_any_sale": "Any livestock-product sale",
    "asinh_livestock_product_sales_revenue_total": "Livestock-product sales revenue",
    "livestock_product_distress_sale_any": "Distress livestock-product sale",
    "livestock_product_log_price_gap_mean": "Livestock-product price gap",
    "share_livestock_prod_sold": "Share of livestock products sold",
    "asinh_cost_fodder_livestock": "Fodder cost",
    "asinh_cost_hired_labor_livestock": "Livestock hired-labour cost",
}

SHOCK_LABELS_FOR_TABLES = {
    "post_2009_first_shift": "Post-2009",
    "post_2013_additional_shift": "Additional post-2013",
    "total_post_2013": "Total post-2013",
}

SAMPLE_LABELS_FOR_TABLES = {
    "full_risk_set": "Full risk set",
    "districts_with_treated_and_control": "Common-support districts",
}


def latex_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)
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
        text = text.replace(old, new)
    return text


def paper_stars(p_value) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fmt_number(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def wild_p_text(row: pd.Series) -> str:
    """Return wild-bootstrap p-value text for LaTeX cells.

    Stars in all tables are deliberately based on ordinary clustered p-values;
    wild-cluster-bootstrap p-values, when available, are shown separately in
    square brackets under the standard error.
    """
    if row is None or row.empty:
        return ""
    p = pd.to_numeric(pd.Series([row.get("wild_bootstrap_p_value")]), errors="coerce").iloc[0]
    status = str(row.get("wild_bootstrap_status", "")).lower()
    if pd.notna(p) and (status in {"ok", ""} or status.startswith("ok")):
        return f"[{p:.3f}]"
    return ""


def coef_cell(row: pd.Series) -> str:
    if row is None or row.empty or pd.isna(row.get("coef")):
        return ""
    coef = fmt_number(row.get("coef"), 3) + paper_stars(row.get("p_value"))
    se = fmt_number(row.get("se"), 3)
    wb = wild_p_text(row)
    lines = [coef, f"({se})" if se else ""]
    if wb:
        lines.append(wb)
    return r"\begin{tabular}[c]{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


def outcome_label_for_table(outcome: str) -> str:
    return OUTCOME_LABELS_FOR_TABLES.get(str(outcome), str(outcome).replace("_", " ").title())


def sample_label_for_table(sample: str) -> str:
    return SAMPLE_LABELS_FOR_TABLES.get(str(sample), str(sample).replace("_", " ").title())


def write_latex_document(table_body: str, title: str, note: str, path: Path) -> None:
    text = "\n".join([
        r"\begin{table}[!htbp] \centering",
        r"\caption{" + latex_escape(title) + r"}",
        table_body,
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize \textit{Notes:} " + latex_escape(note),
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def format_main_results_latex(df: pd.DataFrame, title: str, path: Path) -> None:
    display = df.copy()
    display = display.loc[display.get("status", "ok").eq("ok")].copy() if "status" in display.columns else display
    if display.empty:
        write_latex_document(r"\begin{tabular}{l} No results. \\ \end{tabular}", title, "No estimable results.", path)
        return

    shocks = [s for s in ["post_2009_first_shift", "post_2013_additional_shift", "total_post_2013"] if s in set(display["shock"].dropna())]
    colspec = "ll" + "c" * len(shocks) + "cc"
    lines = [r"\begin{tabular}{" + colspec + r"}", r"\toprule"]
    header = ["Outcome", "Sample"] + [SHOCK_LABELS_FOR_TABLES.get(s, s) for s in shocks] + ["Obs.", "EA clusters"]
    lines.append(" & ".join(latex_escape(x) for x in header) + r" \\")
    lines.append(r"\midrule")

    sort_cols = [c for c in ["outcome_family", "outcome", "sample"] if c in display.columns]
    for (family, outcome), family_outcome in display.sort_values(sort_cols).groupby(["outcome_family", "outcome"], dropna=False):
        first = True
        for sample, group in family_outcome.groupby("sample", dropna=False, sort=False):
            cells = []
            for shock in shocks:
                row = group.loc[group["shock"].eq(shock)]
                cells.append(coef_cell(row.iloc[0]) if not row.empty else "")
            n = fmt_int(group["n"].dropna().iloc[0]) if "n" in group.columns and group["n"].notna().any() else ""
            clusters = fmt_int(group["clusters"].dropna().iloc[0]) if "clusters" in group.columns and group["clusters"].notna().any() else ""
            outcome_text = outcome_label_for_table(outcome) if first else ""
            lines.append(" & ".join([latex_escape(outcome_text), latex_escape(sample_label_for_table(sample)), *cells, n, clusters]) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    note = (
        "Each cell reports the coefficient and the EA-clustered standard error in parentheses. "
        "The specification includes wave fixed effects, district fixed effects, household controls, "
        "inside-reserve status, and linear distance to the nearest reserve boundary. "
        "The post-2013 coefficient is the additional shift; total post-2013 is the linear combination of the post-2009 and additional post-2013 terms. "
        "Wild-cluster-bootstrap p-values, when available, are reported in square brackets under clustered standard errors. "
        "Significance stars are based on ordinary clustered p-values: * p<0.10, ** p<0.05, *** p<0.01."
    )
    write_latex_document("\n".join(lines), title, note, path)


def format_event_results_latex(df: pd.DataFrame, title: str, path: Path) -> None:
    display = df.copy()
    display = display.loc[display.get("status", "ok").eq("ok")].copy() if "status" in display.columns else display
    if display.empty:
        write_latex_document(r"\begin{tabular}{l} No results. \\ \end{tabular}", title, "No estimable results.", path)
        return

    waves = sorted(display["event_wave_label"].dropna().astype(str).unique(), key=lambda x: float(str(x).replace("W", "")) if str(x).replace("W", "").replace(".", "", 1).isdigit() else 999)
    colspec = "ll" + "c" * len(waves) + "cc"
    lines = [r"\begin{tabular}{" + colspec + r"}", r"\toprule"]
    header = ["Outcome", "Sample"] + waves + ["Obs.", "EA clusters"]
    lines.append(" & ".join(latex_escape(x) for x in header) + r" \\")
    lines.append(r"\midrule")

    sort_cols = [c for c in ["outcome_family", "outcome", "sample"] if c in display.columns]
    for (family, outcome), family_outcome in display.sort_values(sort_cols).groupby(["outcome_family", "outcome"], dropna=False):
        first = True
        for sample, group in family_outcome.groupby("sample", dropna=False, sort=False):
            cells = []
            for wave in waves:
                row = group.loc[group["event_wave_label"].astype(str).eq(wave)]
                cells.append(coef_cell(row.iloc[0]) if not row.empty else "")
            n = fmt_int(group["n"].dropna().iloc[0]) if "n" in group.columns and group["n"].notna().any() else ""
            clusters = fmt_int(group["clusters"].dropna().iloc[0]) if "clusters" in group.columns and group["clusters"].notna().any() else ""
            outcome_text = outcome_label_for_table(outcome) if first else ""
            lines.append(" & ".join([latex_escape(outcome_text), latex_escape(sample_label_for_table(sample)), *cells, n, clusters]) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    reference = display["reference_wave_label"].dropna().astype(str).iloc[0] if "reference_wave_label" in display.columns and display["reference_wave_label"].notna().any() else "reference wave"
    note = (
        f"Each cell reports the treated-by-wave coefficient relative to {reference}, with the EA-clustered standard error in parentheses. "
        "The specification includes district fixed effects, wave fixed effects, household controls, inside-reserve status, and linear distance to the nearest reserve boundary. "
        "* p<0.10, ** p<0.05, *** p<0.01."
    )
    write_latex_document("\n".join(lines), title, note, path)


def format_generic_latex(df: pd.DataFrame, title: str, path: Path) -> None:
    display = df.copy()
    if display.empty:
        write_latex_document(r"\begin{tabular}{l} No rows. \\ \end{tabular}", title, "No rows.", path)
        return
    preferred = [
        "sample", "wave_label", "n", "clusters", "fixed_effect_groups", "n_treated", "n_control",
        "n_treated_clusters", "treated_clusters", "control_clusters", "kept",
        "treated_observations", "control_observations",
    ]
    keep = [c for c in preferred if c in display.columns]
    if not keep:
        keep = list(display.columns[:8])
    short = display[keep].copy()
    body = short.to_latex(index=False, escape=True, longtable=False)
    write_latex_document(body, title, "Support and diagnostic table generated by the pipeline.", path)


def save_table(df: pd.DataFrame, output_dir: Path, stem: str, title: str) -> None:
    # Raw CSV is kept only for reproducibility and downstream figure generation.
    # The thesis-facing table is the LaTeX file.
    df.to_csv(output_dir / f"{stem}_raw.csv", index=False)
    if "shock" in df.columns and "coef" in df.columns:
        format_main_results_latex(df, title, output_dir / f"{stem}.tex")
    elif "event_wave_label" in df.columns and "coef" in df.columns:
        format_event_results_latex(df, title, output_dir / f"{stem}.tex")
    else:
        format_generic_latex(df, title, output_dir / f"{stem}.tex")

PAPER_TABLE_GROUPS = [
    ("table_1_welfare", "Welfare", [
        "log_cons_pc", "log_cons_ae", "poverty_under_npl",
    ]),
    ("table_2_herd_dynamics", "Herd dynamics", [
        "livestock_tlu_per_ae", "livestock_total_per_ae", "cattle_change_1y",
        "cattle_decline_1y", "cattle_exit_1y", "cattle_loss_share_1y",
    ]),
    ("table_3_cost_productivity", "Mechanisms: livestock costs and productivity", [
        "any_large_ruminant_disease", "asinh_cost_fodder_livestock", "asinh_cost_lrum",
        "asinh_livestock_expenses", "asinh_value_milk_produced",
        "livestock_product_any_sale", "asinh_livestock_product_sales_revenue_total",
        "livestock_product_log_price_gap_mean", "share_livestock_prod_sold",
    ]),
    ("table_4_agricultural_mechanisms", "Agricultural mechanisms", [
        "ag_participation", "asinh_ag_land_ha", "asinh_ag_hired_labor_days",
        "asinh_ag_total_labor_days", "asinh_ag_harvest_value_2ppp",
    ]),
    ("table_5_income_reallocation", "Income reallocation", [
        "livestock_income_share", "crop_income_share", "nonfarm_income_share",
        "self_employment_income_share", "nonagwage_income_share",
    ]),
]


def save_paper_group_tables(combined: pd.DataFrame, output_dir: Path) -> None:
    for stem, title, outcomes in PAPER_TABLE_GROUPS:
        table_df = combined.loc[combined["outcome"].isin(outcomes)].copy()
        format_main_results_latex(table_df, title, output_dir / f"{stem}.tex")


def main() -> None:
    args = parse_args()
    print(f"run_maasai_main_specifications version: {SCRIPT_VERSION}")
    if args.main_boundary_km <= 0:
        raise ValueError("--main-boundary-km must be positive.")
    if args.wild_bootstrap_reps < 0:
        raise ValueError("--wild-bootstrap-reps cannot be negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    household_path = args.input_dir / args.household_file
    if not household_path.exists():
        raise FileNotFoundError(household_path)

    household = pd.read_csv(household_path, low_memory=False)
    df, metadata = prepare_data(household, args)
    if args.exclude_wave5:
        df = df.loc[numeric(df[WAVE]).ne(5)].copy()
    metadata["wave5_included"] = int(not args.exclude_wave5)
    main_sample = df.loc[df[DISTANCE].le(args.main_boundary_km)].copy()
    if main_sample.empty:
        raise ValueError("No observations remain in the reserve-boundary risk set.")

    common_metadata = {
        "treatment_definition": args.treatment_column,
        "risk_set_definition": f"distance_to_game_reserve_boundary_km_le_{args.main_boundary_km:g}",
        "inside_reserve_distance_reset_to_zero": 0,
        **metadata,
        "wild_bootstrap_reps_requested": args.wild_bootstrap_reps,
        "wild_bootstrap_type_requested": args.wild_bootstrap_type,
    }

    district_table = common_support_districts(main_sample, args.treatment_column)
    kept_districts = district_table.loc[district_table["kept"].eq(1), "district"]
    support_sample = main_sample.loc[main_sample["fe_group"].isin(kept_districts)].copy()
    if support_sample.empty:
        raise ValueError("No districts contain both treated and control observations.")

    samples = {
        "full_risk_set": main_sample,
        "districts_with_treated_and_control": support_sample,
    }

    family_definitions = [
        ("welfare", WELFARE_OUTCOMES),
        ("livestock_assets", LIVESTOCK_ASSET_OUTCOMES),
        ("agriculture_switch", AGRICULTURE_OUTCOMES),
        ("livestock_sales", SALES_OUTCOMES),
        ("livestock_product_distress", PRODUCT_PRICE_OUTCOMES),
        ("livestock_stress_productivity", LIVESTOCK_STRESS_OUTCOMES),
        ("income_reallocation", INCOME_REALLOCATION_OUTCOMES),
    ]

    results: dict[str, pd.DataFrame] = {}
    for family, outcomes in family_definitions:
        frames = []
        for sample_name, sample_data in samples.items():
            frames.append(
                run_outcome_family(
                    sample_data,
                    outcomes,
                    family,
                    sample_name,
                    args,
                    common_metadata,
                )
            )
        results[family] = add_bh_q_values(
            pd.concat(frames, ignore_index=True, sort=False)
        )

    combined = pd.concat(results.values(), ignore_index=True, sort=False)
    support = build_support_table(samples, metadata, args)

    # One raw CSV contains the full set of estimated outcomes and coefficients.
    combined.to_csv(args.output_dir / "table_main_specifications_raw.csv", index=False)
    save_paper_group_tables(combined, args.output_dir)
    # Diagnostics are kept as LaTeX-only tables to avoid proliferating raw CSV outputs.
    format_generic_latex(district_table, "District common-support diagnostics", args.output_dir / "common_support_districts.tex")
    format_generic_latex(support, "Analysis sample support", args.output_dir / "table_0_sample_support.tex")

    notes = args.output_dir / "main_specifications_notes.txt"
    notes.write_text(
        "Repeated-cross-section core specifications\n"
        "==========================================\n"
        f"Script version: {SCRIPT_VERSION}\n"
        f"Household input: {household_path}\n"
        f"Treatment: {args.treatment_column}\n"
        f"Risk set: distance to reserve boundary <= {args.main_boundary_km:g} km\n"
        "Samples: full risk set; districts containing both treated and control observations\n"
        f"Fixed effects: wave + {FE_COLUMN}\n"
        f"Clustered standard errors: {CLUSTER_COLUMN}\n"
        f"Controls: {', '.join(BASE_CONTROL_COLUMNS)}\n\n"
        "Specification:\n"
        "Y = MaasaiAssociated + Post2009 + InsideReserve + "
        "MaasaiAssociated*Post2009 + controls + linear distance to the nearest "
        "reserve boundary (per 10 km) + wave FE + district FE.\n\n"
        "The common-support sample keeps districts with at least one treated and "
        "one control observation in the reserve-boundary risk set.\n\n"
        "Livestock-product distress pricing compares price_per_unit_hh with the supplied "
        "price_median_ea for the same product and EA. No district/region hierarchy is estimated.\n\n"
        "Non-farm, self-employment, and non-agricultural wage income shares are "
        "broad proxies for income diversification; they do not identify craft or "
        "tourism-guide income separately.\n",
        encoding="utf-8",
    )

    print(f"Household input: {household_path}")
    print(f"Full risk-set observations: {len(main_sample):,}")
    print(f"Common-support observations: {len(support_sample):,}")
    print(f"Common-support districts: {len(kept_districts):,}")
    print(f"Treatment: {args.treatment_column}")
    print(f"Saved results in: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
