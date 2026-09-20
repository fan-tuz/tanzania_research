#!/usr/bin/env python3
"""
Create thesis-oriented PNG figures for the final Maasai results.

Inputs
------
- main raw CSV from the main specification script
- event-study raw CSV
- robustness raw CSV (optional but recommended)
- NDVI robustness raw CSV (optional)

Outputs
-------
PNG figures only.
No PDF, HTML, CSV, or README side products.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_VERSION = "2026-06-28-final-thesis-figures-v7-png-only"

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
    "any_large_ruminant_disease": "Large-ruminant disease",
    "asinh_cost_fodder_livestock": "Fodder cost",
    "asinh_cost_lrum": "Large-ruminant treatment cost",
    "asinh_livestock_expenses": "Livestock expenses",
    "asinh_value_milk_produced": "Milk value produced",
    "share_livestock_prod_sold": "Share of livestock products sold",
    "ag_participation": "Agricultural participation",
    "asinh_ag_plot_count": "Agricultural plot count",
    "asinh_ag_land_ha": "Agricultural land (ha)",
    "asinh_ag_family_labor_days": "Family farm labour days",
    "asinh_ag_hired_labor_days": "Hired farm labour days",
    "asinh_ag_total_labor_days": "Total farm labour days",
    "asinh_ag_harvest_value_2ppp": "Harvest value",
    "livestock_income_share": "Livestock income share",
    "crop_income_share": "Crop income share",
    "nonfarm_income_share": "Non-farm income share",
    "self_employment_income_share": "Self-employment income share",
    "nonagwage_income_share": "Non-agricultural wage income share",
}

SHOCK_LABELS = {
    "post_2009_first_shift": "Post-2009",
    "post_2013_additional_shift": "Additional post-2013",
    "total_post_2013": "Total post-2013",
}

EVENT_WAVE_ORDER = {"W1": 1, "W2": 2, "W3": 3, "W4": 4, "W5": 5}

OUTCOME_GROUPS = {
    "welfare": {
        "title": "Welfare outcomes",
        "filename": "fig_01_main_welfare",
        "outcomes": ["log_cons_pc", "log_cons_ae", "poverty_under_npl"],
    },
    "herd": {
        "title": "Herd dynamics",
        "filename": "fig_02_main_herd_dynamics",
        "outcomes": [
            "livestock_total_per_ae",
            "livestock_tlu_per_ae",
            "cattle_change_1y",
            "cattle_decline_1y",
            "cattle_exit_1y",
            "cattle_loss_share_1y",
        ],
    },
    "mechanisms": {
        "title": "Cost and productivity mechanisms",
        "filename": "fig_03_main_cost_productivity",
        "outcomes": [
            "asinh_cost_lrum",
            "asinh_livestock_expenses",
            "any_large_ruminant_disease",
            "asinh_value_milk_produced",
            "share_livestock_prod_sold",
            "asinh_cost_fodder_livestock",
        ],
    },
    "agriculture": {
        "title": "Agricultural mechanisms",
        "filename": "fig_04_main_agricultural_mechanisms",
        "outcomes": [
            "ag_participation",
            "asinh_ag_plot_count",
            "asinh_ag_land_ha",
            "asinh_ag_family_labor_days",
            "asinh_ag_hired_labor_days",
            "asinh_ag_total_labor_days",
            "asinh_ag_harvest_value_2ppp",
        ],
    },
    "income": {
        "title": "Income reallocation",
        "filename": "fig_05_main_income_reallocation",
        "outcomes": [
            "livestock_income_share",
            "crop_income_share",
            "nonfarm_income_share",
            "self_employment_income_share",
            "nonagwage_income_share",
        ],
    },
}

KEY_COMPARISON_ITEMS = [
    ("log_cons_ae", "post_2009_first_shift"),
    ("poverty_under_npl", "post_2009_first_shift"),
    ("cattle_change_1y", "post_2009_first_shift"),
    ("cattle_decline_1y", "post_2009_first_shift"),
    ("cattle_exit_1y", "post_2009_first_shift"),
    ("livestock_total_per_ae", "post_2013_additional_shift"),
    ("asinh_cost_lrum", "post_2009_first_shift"),
    ("asinh_cost_lrum", "post_2013_additional_shift"),
    ("asinh_livestock_expenses", "post_2013_additional_shift"),
    ("asinh_value_milk_produced", "post_2013_additional_shift"),
    ("livestock_income_share", "post_2013_additional_shift"),
]

EVENT_WELFARE_HERD = [
    "log_cons_ae",
    "poverty_under_npl",
    "cattle_change_1y",
    "cattle_decline_1y",
    "livestock_total_per_ae",
]

EVENT_MECHANISMS = [
    "asinh_cost_lrum",
    "asinh_livestock_expenses",
    "asinh_value_milk_produced",
    "livestock_income_share",
]

# vivid but thesis-usable colors
SHOCK_COLORS = {
    "post_2009_first_shift": "#D94841",
    "post_2013_additional_shift": "#2C7FB8",
    "total_post_2013": "#2CA25F",
}

SCENARIO_COLORS = {
    "main": "#D94841",
    "treatment_10km": "#6A3D9A",
    "treatment_25km": "#2C7FB8",
    "riskset_50km": "#E67E22",
    "ndvi_control": "#2CA25F",
}

CATEGORY_COLORS = {
    "welfare": "#1F77B4",
    "herd": "#D94841",
    "mechanisms": "#2CA25F",
    "agriculture": "#9467BD",
    "income": "#E67E22",
}

ZERO_COLOR = "#404040"
GRID_COLOR = "#D8D8D8"
TEXT_COLOR = "#1E1E1E"
NS_COLOR = "#BFBFBF"
EVENT_COLOR = "#2C7FB8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create thesis-oriented PNG figures.")
    p.add_argument("--main-csv", required=True, type=Path)
    p.add_argument("--event-csv", required=True, type=Path)
    p.add_argument("--robustness-csv", required=False, type=Path)
    p.add_argument("--ndvi-csv", required=False, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sample", default="full_risk_set")
    p.add_argument("--comparison-sample", default="full_risk_set")
    p.add_argument("--p-threshold", default=0.10, type=float)
    p.add_argument("--clean-output-dir", action="store_true")
    return p.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "axes.titleweight": "bold",
        }
    )


def read_csv(path: Path | None, label: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    for col in ["coef", "se", "p_value", "wild_bootstrap_p_value", "ci95_low", "ci95_high", "n", "clusters"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" in df.columns:
        df = df.loc[df["status"].fillna("ok").eq("ok")].copy()
    return df


def filter_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    if df.empty or "sample" not in df.columns:
        return df.copy()
    out = df.loc[df["sample"].eq(sample)].copy()
    if out.empty:
        raise ValueError(f"No rows found for sample={sample!r}")
    return out


def clean_output_dir(path: Path, clean: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if clean:
        for item in path.glob("*.png"):
            item.unlink()


def save_png(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(str(outcome), str(outcome).replace("_", " ").title())


def shock_label(shock: str) -> str:
    return SHOCK_LABELS.get(str(shock), str(shock).replace("_", " "))


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


def ordinary_ci(row: pd.Series) -> tuple[float, float]:
    low = row.get("ci95_low", np.nan)
    high = row.get("ci95_high", np.nan)
    if pd.notna(low) and pd.notna(high):
        return float(low), float(high)
    coef = float(row.get("coef", np.nan))
    se = float(row.get("se", np.nan))
    return coef - 1.96 * se, coef + 1.96 * se


def short_wrap(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def extract_one(df: pd.DataFrame, outcome: str, shock: str) -> pd.Series | None:
    out = df.loc[df["outcome"].eq(outcome) & df["shock"].eq(shock)]
    if out.empty:
        return None
    return out.iloc[0]


def make_plot_label(outcome: str, shock: str) -> str:
    return short_wrap(f"{outcome_label(outcome)} — {shock_label(shock)}", 38)


# ---------------------------------------------------------------------------
# Main category plots
# ---------------------------------------------------------------------------


def main_group_plot(main_df: pd.DataFrame, group_key: str, group: dict, output_dir: Path) -> None:
    outcomes = group["outcomes"]
    shock_order = ["post_2009_first_shift", "post_2013_additional_shift", "total_post_2013"]
    df = main_df.loc[main_df["outcome"].isin(outcomes) & main_df["shock"].isin(shock_order)].copy()
    if df.empty:
        return

    outcome_order = {o: i for i, o in enumerate(outcomes)}
    shock_order_map = {s: i for i, s in enumerate(shock_order)}
    df["outcome_order"] = df["outcome"].map(outcome_order)
    df["shock_order"] = df["shock"].map(shock_order_map)
    df = df.dropna(subset=["coef", "se", "outcome_order", "shock_order"]).sort_values(["outcome_order", "shock_order"])
    if df.empty:
        return

    rows = []
    for _, row in df.iterrows():
        r = row.copy()
        r["plot_label"] = make_plot_label(r["outcome"], r["shock"])
        rows.append(r)
    plot_df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig_h = max(5.4, 0.52 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))

    for i, row in plot_df.iterrows():
        low, high = ordinary_ci(row)
        color = SHOCK_COLORS.get(row["shock"], "#333333")
        alpha = 1.0 if row.get("p_value", np.nan) <= 0.10 else 0.38
        ax.errorbar(
            row["coef"],
            i,
            xerr=np.array([[row["coef"] - low], [high - row["coef"]]]),
            fmt="o",
            markersize=7.8,
            capsize=3.5,
            elinewidth=2.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            alpha=alpha,
            zorder=3,
        )

    ax.axvline(0, color=ZERO_COLOR, linewidth=1.15)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.60, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"])
    ax.tick_params(axis="y", pad=8)
    ax.set_xlabel("Estimated effect with 95% confidence interval")
    ax.set_title(group["title"], pad=14)

    ci_lows = []
    ci_highs = []
    for _, row in plot_df.iterrows():
        low, high = ordinary_ci(row)
        ci_lows.append(low)
        ci_highs.append(high)
    xmin = float(np.nanmin(ci_lows))
    xmax = float(np.nanmax(ci_highs))
    span = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.38 * span)
    label_x = xmax + 0.04 * span

    for i, row in plot_df.iterrows():
        label = f"{row['coef']:.3f}{significance_stars(row.get('p_value', np.nan))}"
        ax.text(label_x, i, label, va="center", fontsize=8.8, color=TEXT_COLOR)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=shock_label(s), markerfacecolor=c, markersize=8)
        for s, c in SHOCK_COLORS.items()
    ]
    legend_handles.append(Patch(facecolor=NS_COLOR, alpha=0.38, label="Non-significant estimates are faded"))
    ax.legend(handles=legend_handles, loc="lower right")

    fig.tight_layout()
    save_png(fig, output_dir / group["filename"])


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------


def narrative_summary_plot(main_df: pd.DataFrame, output_path: Path) -> None:
    items = [
        ("log_cons_ae", "post_2009_first_shift", "Consumption"),
        ("poverty_under_npl", "post_2009_first_shift", "Poverty"),
        ("cattle_change_1y", "post_2009_first_shift", "Net cattle change"),
        ("cattle_decline_1y", "post_2009_first_shift", "Cattle decline"),
        ("asinh_cost_lrum", "post_2009_first_shift", "Treatment cost"),
        ("livestock_total_per_ae", "post_2013_additional_shift", "Livestock rebuilding"),
        ("asinh_livestock_expenses", "post_2013_additional_shift", "Livestock expenses"),
        ("asinh_value_milk_produced", "post_2013_additional_shift", "Milk production"),
        ("livestock_income_share", "post_2013_additional_shift", "Livestock income share"),
    ]
    rows = []
    for outcome, shock, label in items:
        row = extract_one(main_df, outcome, shock)
        if row is not None:
            row = row.copy()
            row["summary_label"] = label
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["plot_color"] = df["shock"].map(SHOCK_COLORS)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.bar(x, df["coef"], color=df["plot_color"], alpha=0.92, width=0.72)
    ax.axhline(0, color=ZERO_COLOR, linewidth=1.1)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([short_wrap(v, 14) for v in df["summary_label"]], rotation=0)
    ax.set_ylabel("Estimated effect")
    ax.set_title("Narrative summary of the main findings", pad=14)

    span = max(df["coef"].max() - df["coef"].min(), 1.0)
    for i, row in df.reset_index(drop=True).iterrows():
        stars = significance_stars(row.get("p_value", np.nan))
        y = row["coef"]
        offset = 0.06 * span
        ax.text(
            i,
            y + (offset if y >= 0 else -offset),
            f"{y:.2f}{stars}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    legend_handles = [
        Patch(facecolor=SHOCK_COLORS[s], label=shock_label(s))
        for s in ["post_2009_first_shift", "post_2013_additional_shift"]
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()
    save_png(fig, output_path)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------


def scenario_frame(df: pd.DataFrame, scenario: str, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["scenario"] = scenario
    out["scenario_label_clean"] = label
    return out


def comparison_plot(frames: list[pd.DataFrame], scenario_order: list[str], scenario_labels: dict[str, str], title: str, output_path: Path) -> None:
    df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True, sort=False)
    if df.empty:
        return
    keep_items = set(KEY_COMPARISON_ITEMS)
    df = df.loc[df[["outcome", "shock"]].apply(tuple, axis=1).isin(keep_items)].copy()
    if df.empty:
        return

    item_labels = [make_plot_label(o, s) for o, s in KEY_COMPARISON_ITEMS]
    item_map = {(o, s): make_plot_label(o, s) for o, s in KEY_COMPARISON_ITEMS}
    df["item"] = df[["outcome", "shock"]].apply(tuple, axis=1).map(item_map)
    items_present = [x for x in item_labels if x in set(df["item"])]
    df["item_order"] = df["item"].map({item: i for i, item in enumerate(items_present)})
    df["scenario_order"] = df["scenario"].map({s: i for i, s in enumerate(scenario_order)})
    df = df.dropna(subset=["coef", "se", "item_order", "scenario_order"]).sort_values(["item_order", "scenario_order"])
    if df.empty:
        return

    y_base = np.arange(len(items_present))[::-1]
    offsets = np.linspace(-0.22, 0.22, max(len(scenario_order), 2))[: len(scenario_order)]

    fig_h = max(5.6, 0.55 * len(items_present) + 1.6)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))

    for s_idx, scenario in enumerate(scenario_order):
        sub = df.loc[df["scenario"].eq(scenario)].copy()
        if sub.empty:
            continue
        color = SCENARIO_COLORS.get(scenario, "#333333")
        for _, row in sub.iterrows():
            y = y_base[items_present.index(row["item"])] + offsets[s_idx]
            low, high = ordinary_ci(row)
            alpha = 1.0 if row.get("p_value", np.nan) <= 0.10 else 0.42
            ax.errorbar(
                row["coef"],
                y,
                xerr=np.array([[row["coef"] - low], [high - row["coef"]]]),
                fmt="o",
                markersize=7.2,
                capsize=3,
                elinewidth=1.85,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                alpha=alpha,
                zorder=3,
            )

    ax.axvline(0, color=ZERO_COLOR, linewidth=1.1)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
    ax.set_yticks(y_base)
    ax.set_yticklabels(items_present)
    ax.tick_params(axis="y", pad=8)
    ax.set_xlabel("Estimated effect with 95% confidence interval")
    ax.set_title(title, pad=14)
    legend = [
        Line2D([0], [0], marker="o", color="w", label=scenario_labels[s], markerfacecolor=SCENARIO_COLORS.get(s, "#333333"), markersize=8)
        for s in scenario_order if s in set(df["scenario"])
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    save_png(fig, output_path)


# ---------------------------------------------------------------------------
# Event-study plots
# ---------------------------------------------------------------------------


def event_panel_plot(event_df: pd.DataFrame, outcomes: list[str], title: str, output_path: Path) -> None:
    df = event_df.loc[event_df["outcome"].isin(outcomes)].copy()
    if df.empty:
        return

    present = [o for o in outcomes if o in set(df["outcome"])]
    if not present:
        return

    n = len(present)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.4, max(4.2, 3.4 * nrows)), squeeze=False)
    axes_flat = axes.flatten()

    for ax, outcome in zip(axes_flat, present):
        sub = df.loc[df["outcome"].eq(outcome)].copy()
        sub = sub.dropna(subset=["coef", "se", "event_wave_label"])
        if sub.empty:
            ax.axis("off")
            continue
        sub["wave_order"] = sub["event_wave_label"].map(EVENT_WAVE_ORDER)
        sub = sub.sort_values("wave_order")
        x = np.arange(len(sub))
        y = sub["coef"].to_numpy(dtype=float)
        lows, highs = [], []
        for _, row in sub.iterrows():
            low, high = ordinary_ci(row)
            lows.append(low)
            highs.append(high)
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        yerr = np.vstack([y - lows, highs - y])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=EVENT_COLOR,
            linewidth=1.8,
            elinewidth=1.7,
            markersize=6.6,
            capsize=3,
            markerfacecolor=EVENT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.axhline(0, color=ZERO_COLOR, linewidth=1.0)
        ax.grid(axis="y", color=GRID_COLOR, linestyle="--", alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["event_wave_label"].astype(str))
        ax.set_title(outcome_label(outcome), fontsize=12.2, pad=8)

        # annotate only significant estimates with stars to avoid clutter
        yspan = max(np.nanmax(highs) - np.nanmin(lows), 1.0)
        for i, row in sub.reset_index(drop=True).iterrows():
            stars = significance_stars(row.get("p_value", np.nan))
            if stars:
                ax.text(i, row["coef"] + 0.06 * yspan, stars, ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    for ax in axes_flat[len(present):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.supxlabel("Wave (reference = W3)", fontsize=11)
    fig.supylabel("Estimated treated-control difference relative to W3", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_png(fig, output_path)


# ---------------------------------------------------------------------------
# Robustness dashboard
# ---------------------------------------------------------------------------


def robustness_dashboard(main_df: pd.DataFrame, rob_df: pd.DataFrame, ndvi_df: pd.DataFrame, output_path: Path) -> None:
    key_items = [
        ("poverty_under_npl", "post_2009_first_shift"),
        ("cattle_change_1y", "post_2009_first_shift"),
        ("cattle_decline_1y", "post_2009_first_shift"),
        ("cattle_exit_1y", "post_2009_first_shift"),
        ("livestock_total_per_ae", "post_2013_additional_shift"),
        ("asinh_cost_lrum", "post_2013_additional_shift"),
        ("asinh_value_milk_produced", "post_2013_additional_shift"),
    ]

    frames = []
    frames.append(scenario_frame(main_df.copy(), "main", "Main (25 km, 75 km risk set)"))
    if not rob_df.empty:
        if "scenario" in rob_df.columns:
            ten = rob_df.loc[rob_df["scenario"].eq("treatment_maasai_10km")].copy()
            if not ten.empty:
                frames.append(scenario_frame(ten, "treatment_10km", "10 km treatment"))
            fifty = rob_df.loc[rob_df["scenario"].eq("riskset_50km")].copy()
            if not fifty.empty:
                frames.append(scenario_frame(fifty, "riskset_50km", "50 km risk set"))
    if not ndvi_df.empty:
        frames.append(scenario_frame(ndvi_df.copy(), "ndvi_control", "With NDVI control"))

    df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True, sort=False)
    if df.empty:
        return
    keep_items = set(key_items)
    df = df.loc[df[["outcome", "shock"]].apply(tuple, axis=1).isin(keep_items)].copy()
    if df.empty:
        return

    item_names = [make_plot_label(o, s) for o, s in key_items]
    item_map = {(o, s): make_plot_label(o, s) for o, s in key_items}
    df["item"] = df[["outcome", "shock"]].apply(tuple, axis=1).map(item_map)
    df["item_order"] = df["item"].map({k: i for i, k in enumerate(item_names)})
    scenario_order = ["main", "treatment_10km", "riskset_50km", "ndvi_control"]
    df["scenario_order"] = df["scenario"].map({k: i for i, k in enumerate(scenario_order)})
    df = df.dropna(subset=["item_order", "scenario_order", "coef"]).sort_values(["item_order", "scenario_order"])

    y_base = np.arange(len(item_names))[::-1]
    offsets = np.linspace(-0.27, 0.27, 4)
    fig_h = max(5.6, 0.58 * len(item_names) + 1.6)
    fig, ax = plt.subplots(figsize=(13.2, fig_h))

    for s_idx, scenario in enumerate(scenario_order):
        sub = df.loc[df["scenario"].eq(scenario)]
        if sub.empty:
            continue
        color = SCENARIO_COLORS.get(scenario, "#333333")
        for _, row in sub.iterrows():
            y = y_base[item_names.index(row["item"])] + offsets[s_idx]
            low, high = ordinary_ci(row)
            alpha = 1.0 if row.get("p_value", np.nan) <= 0.10 else 0.40
            ax.errorbar(
                row["coef"], y,
                xerr=np.array([[row["coef"] - low], [high - row["coef"]]]),
                fmt="o",
                markersize=7.1,
                capsize=3,
                elinewidth=1.8,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                alpha=alpha,
            )

    ax.axvline(0, color=ZERO_COLOR, linewidth=1.1)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
    ax.set_yticks(y_base)
    ax.set_yticklabels(item_names)
    ax.tick_params(axis="y", pad=8)
    ax.set_xlabel("Estimated effect with 95% confidence interval")
    ax.set_title("Robustness snapshot for the core outcomes", pad=14)
    legend = [
        Line2D([0], [0], marker="o", color="w", label=lbl, markerfacecolor=SCENARIO_COLORS.get(key, "#333333"), markersize=8)
        for key, lbl in [
            ("main", "Main"),
            ("treatment_10km", "10 km treatment"),
            ("riskset_50km", "50 km risk set"),
            ("ndvi_control", "NDVI control"),
        ] if key in set(df["scenario"])
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    save_png(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_style()
    clean_output_dir(args.output_dir, args.clean_output_dir)

    print(f"visualize_maasai_main_findings version: {SCRIPT_VERSION}")

    main_raw = read_csv(args.main_csv, "main CSV")
    event_raw = read_csv(args.event_csv, "event CSV")
    robustness_raw = read_csv(args.robustness_csv, "robustness CSV") if args.robustness_csv else pd.DataFrame()
    ndvi_raw = read_csv(args.ndvi_csv, "NDVI CSV") if args.ndvi_csv else pd.DataFrame()

    main_df = filter_sample(main_raw, args.sample)
    event_df = filter_sample(event_raw, args.sample)
    rob_df = filter_sample(robustness_raw, args.comparison_sample) if not robustness_raw.empty else pd.DataFrame()
    ndvi_df = filter_sample(ndvi_raw, args.comparison_sample) if not ndvi_raw.empty else pd.DataFrame()

    # main category figures
    for group_key, group in OUTCOME_GROUPS.items():
        main_group_plot(main_df, group_key, group, args.output_dir)

    # narrative summary
    narrative_summary_plot(main_df, args.output_dir / "fig_06_main_narrative_summary")

    # comparisons
    baseline = scenario_frame(main_df.copy(), "main", "Main (25 km, 75 km risk set)")

    # 25 km versus 10 km treatment comparison
    tenkm = pd.DataFrame()
    if not rob_df.empty and "scenario" in rob_df.columns:
        tenkm = rob_df.loc[rob_df["scenario"].eq("treatment_maasai_10km")].copy()
    if not tenkm.empty:
        tenkm = scenario_frame(tenkm, "treatment_10km", "10 km treatment")
        comparison_plot(
            [baseline, tenkm],
            scenario_order=["main", "treatment_10km"],
            scenario_labels={"main": "25 km treatment", "treatment_10km": "10 km treatment"},
            title="Treatment-definition comparison",
            output_path=args.output_dir / "fig_07_comparison_maasai_10km_vs_25km",
        )

    # 75 km versus 50 km risk-set comparison
    risk50 = pd.DataFrame()
    if not rob_df.empty and "scenario" in rob_df.columns:
        risk50 = rob_df.loc[rob_df["scenario"].eq("riskset_50km")].copy()
    if not risk50.empty:
        risk50 = scenario_frame(risk50, "riskset_50km", "50 km risk set")
        comparison_plot(
            [baseline, risk50],
            scenario_order=["main", "riskset_50km"],
            scenario_labels={"main": "75 km risk set", "riskset_50km": "50 km risk set"},
            title="Risk-set comparison",
            output_path=args.output_dir / "fig_08_comparison_riskset_75km_vs_50km",
        )

    # NDVI comparison
    if not ndvi_df.empty:
        ndvi_comp = scenario_frame(ndvi_df.copy(), "ndvi_control", "With NDVI control")
        comparison_plot(
            [baseline, ndvi_comp],
            scenario_order=["main", "ndvi_control"],
            scenario_labels={"main": "Baseline", "ndvi_control": "With NDVI control"},
            title="NDVI-control robustness",
            output_path=args.output_dir / "fig_09_comparison_baseline_vs_ndvi_control",
        )

    # event-study figures
    event_panel_plot(
        event_df,
        EVENT_WELFARE_HERD,
        "Event-study dynamics: welfare and herd outcomes",
        args.output_dir / "fig_10_event_study_welfare_herd",
    )
    event_panel_plot(
        event_df,
        EVENT_MECHANISMS,
        "Event-study dynamics: mechanisms",
        args.output_dir / "fig_11_event_study_mechanisms",
    )

    # additional robustness dashboard
    robustness_dashboard(main_df, rob_df, ndvi_df, args.output_dir / "fig_12_robustness_dashboard")

    print(f"PNG figures saved in: {args.output_dir}")
    for p in sorted(args.output_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
