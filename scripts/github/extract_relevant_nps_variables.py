#!/usr/bin/env python3
"""
Extract lean Tanzania NPS household, agriculture, live-animal sales, and
livestock-product price datasets for Waves 1-5.

All input paths are deterministic. For each wave W1-W5 the script reads:

    data/github/Tanzania NPS W*/final_data/
        Tanzania_NPS_W*_household_variables.csv
        Tanzania_NPS_W*_individual_variables.csv
        Tanzania_NPS_W*_field_plot_variables.csv

    data/github/Tanzania NPS W*/created_data/
        Tanzania_NPS_W*_hh_livestock_sales.csv
        Tanzania_NPS_W*_livestock_products.csv
        Tanzania_NPS_W*_livestock_products_prices_ea.csv
        Tanzania_NPS_W*_livestock_income.csv
        Tanzania_NPS_W*_livestock_expenses.csv
        Tanzania_NPS_W*_livestock_diseases.csv

Wave 5 GPS coordinates are confidential. The script does not read any W5
coordinate file. When --include-wave5 is supplied, W5 is kept only through the
official NPSY5 panel key linking y5_hhid to y4_hhid, and W4 coordinates are
inherited as baseline W4 locations. Booster households and W5 households not
linked to W4 through the official key are excluded.

The EA price file benchmarks livestock-product prices, not prices of live
animals. Consequently:

- live-animal sales are summarized through participation, quantities and
  revenues only;
- livestock-product distress pricing is measured by comparing each observed
  product price with the supplied EA median for the same product code;
- no hierarchical district/region/country benchmark is constructed;
- no livestock-purchase variable is assumed to exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-06-27-w1-w5-official-panel-key-v18"

SOURCES = {
    "W1": {"folder": "Tanzania NPS W1", "wave": 1.0, "year": 2008, "plot_has_season": False},
    "W2": {"folder": "Tanzania NPS W2", "wave": 2.0, "year": 2010, "plot_has_season": True},
    "W3": {"folder": "Tanzania NPS W3", "wave": 3.0, "year": 2012, "plot_has_season": True},
    "W4": {"folder": "Tanzania NPS W4", "wave": 4.0, "year": 2014, "plot_has_season": False},
    "W5": {"folder": "Tanzania NPS W5", "wave": 5.0, "year": 2021, "plot_has_season": False},
}

FIXED_GEOREFERENCING = {
    "W1": {
        "coordinate_file": "wave1/HH.Geovariables_Y1.csv",
        "household_key": "hhid",
        "coordinate_key": "hhid",
        "latitude_column": "lat_modified",
        "longitude_column": "lon_modified",
        "expected_household_rows": 3277,
        "expected_georeferenced_households": 2990,
    },
    "W2": {
        "coordinate_file": "wave2/TZY2.EA.Offsets.csv",
        "household_key": "clusterid",
        "coordinate_key": "clusterid",
        "latitude_column": "lat_modified",
        "longitude_column": "lon_modified",
        "expected_household_rows": 3924,
        "expected_georeferenced_households": 3924,
    },
    "W3": {
        "coordinate_file": "wave3/HouseholdGeovars_Y3.csv",
        "household_key": "hhid",
        "coordinate_key": "y3_hhid",
        "latitude_column": "lat_dd_mod",
        "longitude_column": "lon_dd_mod",
        "expected_household_rows": 5010,
        "expected_georeferenced_households": 4988,
    },
    "W4": {
        "coordinate_file": "wave4/npsy4.ea.offset.csv",
        "household_key": "clusterid",
        "coordinate_key": "clusterid",
        "latitude_column": "lat_modified",
        "longitude_column": "lon_modified",
        "expected_household_rows": 3353,
        "expected_georeferenced_households": 3352,
    },
}

HOUSEHOLD_COLUMNS = [
    "hhid",
    "hh_members",
    "clusterid",
    "district",
    "ward",
    "ea",
    "region",
    "ag_hh",
    "adulteq",
    "percapita_cons",
    "peraeq_cons",
    "poverty_under_npl",
    "total_income",
    "crop_income",
    "nonfarm_income",
    "self_employment_income",
    "nonagwage_income",
    "lvstck_holding_all",
    "lvstck_holding_tlu",
    "nb_cattle_today",
    "nb_cattle_1yearago",
]

INDIVIDUAL_COLUMNS = ["hhid", "indid", "female", "age"]

PLOT_BASE_COLUMNS = [
    "hhid",
    "plot_id",
    "area_meas_hectares",
    "plot_value_harvest_2ppp",
    "labor_hired",
    "labor_family",
    "labor_total",
]

LIVE_SALES_COLUMNS = [
    "hhid",
    "livestock_code",
    "number_sold",
    "income_live_sales",
    "price_per_animal",
    "district",
    "ward",
    "ea",
    "region",
]

PRODUCT_COLUMNS = [
    "hhid",
    "livestock_code",
    "earnings_sales",
    "quantity_produced",
    "price_per_unit_hh",
    "district",
    "ward",
    "ea",
    "region",
]

PRODUCT_EA_PRICE_COLUMNS = [
    "livestock_code",
    "district",
    "ward",
    "ea",
    "region",
    "obs_ea",
    "price_median_ea",
]

LIVESTOCK_INCOME_COLUMNS = [
    "hhid",
    "value_milk_produced",
    "sales_livestock_products",
    "value_livestock_products",
    "share_livestock_prod_sold",
    "livestock_income",
]

HH_LIVESTOCK_PRODUCTS_COLUMNS = [
    "hhid",
    "value_milk_produced",
    "sales_livestock_products",
    "value_livestock_products",
    "share_livestock_prod_sold",
]

LIVESTOCK_EXPENSE_BASE_COLUMNS = [
    "hhid",
    "cost_fodder_livestock",
    "cost_hired_labor_livestock",
]

LIVESTOCK_EXPENSE_COLUMNS = [
    *LIVESTOCK_EXPENSE_BASE_COLUMNS,
    "cost_lrum",
]

LRUM_EXPENSE_COLUMNS = [
    "hhid",
    "cost_lrum",
]

LIVESTOCK_DISEASE_COLUMNS = [
    "hhid",
    "disease_fmd_lrum",
    "disease_lump_lrum",
    "disease_bruc_lrum",
    "disease_cbpp_lrum",
    "disease_bq_lrum",
]

AGRICULTURE_OUTCOMES = [
    "ag_participation",
    "ag_plot_count",
    "ag_land_ha",
    "ag_family_labor_days",
    "ag_hired_labor_days",
    "ag_total_labor_days",
    "ag_harvest_value_2ppp",
]

HOUSEHOLD_OUTPUT_COLUMNS = [
    "survey_round",
    "wave",
    "survey_year",
    "hhid",
    "hhid_wave",
    "clusterid",
    "region",
    "district",
    "ward",
    "ea",
    "hh_members",
    "share_female",
    "n_children_u15",
    "n_adults_15plus",
    "log_cons_pc",
    "log_cons_ae",
    "poverty_under_npl",
    "livestock_tlu_per_ae",
    "livestock_total_per_ae",
    "cattle_change_1y",
    "cattle_decline_1y",
    "cattle_exit_1y",
    "cattle_loss_share_1y",
    "livestock_any_sale",
    "livestock_number_sold_total",
    "livestock_sales_revenue_total",
    "cattle_any_sale",
    "cattle_number_sold_total",
    "cattle_sales_revenue_total",
    "livestock_product_any_sale",
    "livestock_product_sales_revenue_total",
    "livestock_product_price_records_benchmarked",
    "livestock_product_log_price_gap_mean",
    "livestock_product_distress_sale_any",
    "any_large_ruminant_disease",
    "cost_fodder_livestock",
    "cost_lrum",
    "cost_hired_labor_livestock",
    "livestock_expenses",
    "value_milk_produced",
    "share_livestock_prod_sold",
    "livestock_income_share",
    "crop_income_share",
    "nonfarm_income_share",
    "self_employment_income_share",
    "nonagwage_income_share",
    *AGRICULTURE_OUTCOMES,
    "post_2009",
    "post_2013",
    "cluster_lat",
    "cluster_lon",
]

AGRICULTURE_OUTPUT_COLUMNS = [
    "survey_round",
    "wave",
    "survey_year",
    "hhid",
    "hhid_wave",
    "region",
    "district",
    "ward",
    "ea",
    *AGRICULTURE_OUTCOMES,
    "cluster_lat",
    "cluster_lon",
]

LIVE_SALES_OUTPUT_COLUMNS = [
    "survey_round",
    "wave",
    "survey_year",
    "hhid",
    "hhid_wave",
    "livestock_code",
    "is_cattle",
    "district",
    "ward",
    "ea",
    "region",
    "number_sold",
    "income_live_sales",
    "actual_livestock_sale",
    "price_per_animal",
    "cluster_lat",
    "cluster_lon",
]

PRODUCT_PRICE_OUTPUT_COLUMNS = [
    "survey_round",
    "wave",
    "survey_year",
    "hhid",
    "hhid_wave",
    "livestock_code",
    "district",
    "ward",
    "ea",
    "region",
    "earnings_sales",
    "price_per_unit_hh",
    "obs_ea",
    "price_median_ea",
    "valid_ea_product_price_benchmark",
    "livestock_product_log_price_gap",
    "livestock_product_distress_sale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--legacy-data-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    wave5_group = parser.add_mutually_exclusive_group()
    wave5_group.add_argument(
        "--include-wave5",
        dest="include_wave5",
        action="store_true",
        help=(
            "Include Wave 5 using the official NPSY5 panel key and inherited W4 baseline coordinates."
        ),
    )
    wave5_group.add_argument(
        "--exclude-wave5",
        dest="include_wave5",
        action="store_false",
        help="Exclude Wave 5 from the extracted dataset.",
    )
    parser.set_defaults(include_wave5=True)
    parser.add_argument(
        "--wave5-panel-key",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to npsy5.panel.key.csv. If omitted, the script "
            "uses data-root/Tanzania NPS W5/created_data/npsy5.panel.key.csv."
        ),
    )
    parser.add_argument(
        "--wave5-max-households",
        type=int,
        default=4200,
        help=(
            "Safety cap for W5 households included through the official NPSY5 panel key. "
            "The public documentation reports about 4,164 panel-analysis households."
        ),
    )
    parser.add_argument(
        "--cattle-codes",
        default="1,2,3,4,5,6",
        help="Comma-separated live-animal codes classified as cattle.",
    )
    parser.add_argument(
        "--min-ea-price-observations",
        type=int,
        default=2,
        help="Minimum obs_ea required to use the supplied EA product-price median.",
    )
    parser.add_argument(
        "--distress-price-ratio",
        type=float,
        default=0.80,
        help="Classify a livestock-product sale as distressed when its household price is below this share of the supplied EA median.",
    )
    return parser.parse_args()


def source_paths(
    data_root: Path,
    source: str,
    config: dict,
) -> dict[str, Path]:
    folder = data_root / config["folder"]
    prefix = f"Tanzania_NPS_{source}"

    paths = {
        "household": (
            folder
            / "final_data"
            / f"{prefix}_household_variables.csv"
        ),
        "individual": (
            folder
            / "final_data"
            / f"{prefix}_individual_variables.csv"
        ),
        "plot": (
            folder
            / "final_data"
            / f"{prefix}_field_plot_variables.csv"
        ),
        "live_sales": (
            folder
            / "created_data"
            / f"{prefix}_hh_livestock_sales.csv"
        ),
        "products": (
            folder
            / "created_data"
            / f"{prefix}_livestock_products.csv"
        ),
        "product_ea_prices": (
            folder
            / "created_data"
            / f"{prefix}_livestock_products_prices_ea.csv"
        ),
        "livestock_income": (
            folder
            / "created_data"
            / f"{prefix}_livestock_income.csv"
        ),
        "hh_livestock_products": (
            folder
            / "created_data"
            / f"{prefix}_hh_livestock_products.csv"
        ),
        "livestock_expenses": (
            folder
            / "created_data"
            / f"{prefix}_livestock_expenses.csv"
        ),
        "livestock_diseases": (
            folder
            / "created_data"
            / f"{prefix}_livestock_diseases.csv"
        ),
    }

    if source in {"W3", "W4"}:
        paths["lrum_expenses"] = (
            folder
            / "created_data"
            / f"{prefix}_lrum_expenses.csv"
        )
    if source == "W5":
        paths["lrum_expenses"] = (
            folder
            / "created_data"
            / f"{prefix}_livestock_expenses_animal.csv"
        )
        paths["panel_key"] = (
            folder
            / "created_data"
            / "npsy5.panel.key.csv"
        )

    return paths


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_required(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    require_file(path, label)
    available = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    return pd.read_csv(path, usecols=columns, low_memory=False)


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_code_set(value: str) -> set[str]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    return set(clean_id(pd.Series(parts)).dropna().astype(str))


def valid_coordinate_pair(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    return latitude.between(-90, 90) & longitude.between(-180, 180)


def apply_fixed_georeferencing(
    household: pd.DataFrame,
    legacy_root: Path,
    source: str,
) -> tuple[pd.DataFrame, dict]:
    specification = FIXED_GEOREFERENCING[source]
    path = legacy_root / specification["coordinate_file"]
    require_file(path, f"{source} coordinate file")

    expected_rows = specification["expected_household_rows"]
    if expected_rows is not None:
        expected_rows = int(expected_rows)
        if len(household) != expected_rows:
            raise ValueError(
                f"{source}: expected {expected_rows:,} household rows, found {len(household):,}."
            )

    household_key = specification["household_key"]
    coordinate_key = specification["coordinate_key"]
    latitude_column = specification["latitude_column"]
    longitude_column = specification["longitude_column"]

    coordinates = read_required(
        path,
        [coordinate_key, latitude_column, longitude_column],
        f"{source} coordinate file",
    )
    coordinates["_key"] = clean_id(coordinates[coordinate_key])
    coordinates["cluster_lat"] = numeric(coordinates[latitude_column])
    coordinates["cluster_lon"] = numeric(coordinates[longitude_column])
    coordinates = coordinates.loc[
        coordinates["_key"].notna()
        & valid_coordinate_pair(coordinates["cluster_lat"], coordinates["cluster_lon"]),
        ["_key", "cluster_lat", "cluster_lon"],
    ].drop_duplicates()

    conflicts = (
        coordinates.groupby("_key")[["cluster_lat", "cluster_lon"]]
        .apply(lambda x: len(x.drop_duplicates()), include_groups=False)
        .gt(1)
        .sum()
    )
    if conflicts:
        raise ValueError(f"{source}: {int(conflicts)} coordinate keys have conflicts.")
    coordinates = coordinates.drop_duplicates("_key")

    out = household.copy()
    out["_key"] = clean_id(out[household_key])
    out = out.merge(coordinates, on="_key", how="left", validate="many_to_one")
    out = out.drop(columns="_key")

    valid = valid_coordinate_pair(out["cluster_lat"], out["cluster_lon"])
    out.loc[~valid, ["cluster_lat", "cluster_lon"]] = np.nan
    observed = int(valid.sum())
    expected = specification["expected_georeferenced_households"]
    if expected is not None:
        expected = int(expected)
        if observed != expected:
            raise ValueError(
                f"{source}: expected {expected:,} georeferenced households, found {observed:,}."
            )

    return out, {
        "survey_round": source,
        "coordinate_file": str(path.resolve()),
        "household_rows": len(out),
        "georeferenced_households": observed,
        "share_georeferenced": observed / len(out),
    }




def apply_w5_coordinates_from_w4(
    household: pd.DataFrame,
    w4_coordinates: pd.DataFrame,
    panel_key_path: Path,
    max_households: int,
) -> tuple[pd.DataFrame, dict]:
    """Keep W5 households safely linked to W4 through the official panel key.

    W5 GPS coordinates are confidential and are never read. Instead, this
    function uses npsy5.panel.key.csv, which explicitly links y5_hhid to
    y4_hhid. Only W5 households with a non-ambiguous y5_hhid -> y4_hhid link
    and a matched W4 coordinate are retained. Booster households and W5
    households without an official W4 link are excluded.

    The inherited coordinates represent baseline W4 location, not current W5
    residence. This is appropriate only for analyses that treat W5 as a later
    outcome for the W4 baseline spatial exposure.
    """
    require_file(panel_key_path, "W5 panel key file")

    required_coordinates = ["hhid", "cluster_lat", "cluster_lon"]
    missing = [column for column in required_coordinates if column not in w4_coordinates.columns]
    if missing:
        raise ValueError(f"W4 coordinate lookup is missing columns: {missing}")

    coordinates = w4_coordinates[required_coordinates].copy()
    coordinates["y4_hhid"] = clean_id(coordinates["hhid"])
    coordinates["cluster_lat"] = numeric(coordinates["cluster_lat"])
    coordinates["cluster_lon"] = numeric(coordinates["cluster_lon"])
    coordinates = coordinates.loc[
        coordinates["y4_hhid"].notna()
        & valid_coordinate_pair(coordinates["cluster_lat"], coordinates["cluster_lon"]),
        ["y4_hhid", "cluster_lat", "cluster_lon"],
    ].drop_duplicates()

    if coordinates["y4_hhid"].duplicated().any():
        raise ValueError("W4 coordinate lookup contains duplicated hhid values.")

    panel = read_required(
        panel_key_path,
        ["y4_hhid", "y5_hhid"],
        "W5 panel key file",
    )
    panel["y4_hhid"] = clean_id(panel["y4_hhid"])
    panel["y5_hhid"] = clean_id(panel["y5_hhid"])
    panel = panel.loc[panel["y4_hhid"].notna() & panel["y5_hhid"].notna(), ["y4_hhid", "y5_hhid"]]
    panel = panel.drop_duplicates()

    # A few W5 household IDs can appear linked to more than one W4 household in
    # the individual-level key. Those are not safe for assigning W4 coordinates.
    y5_link_counts = panel.groupby("y5_hhid", sort=False)["y4_hhid"].nunique()
    ambiguous_y5 = set(y5_link_counts.loc[y5_link_counts.gt(1)].index.astype(str))
    if ambiguous_y5:
        panel = panel.loc[~panel["y5_hhid"].astype(str).isin(ambiguous_y5)].copy()

    if panel["y5_hhid"].duplicated().any():
        # Remaining duplicates are repeated individuals within the same W4-W5
        # household link; collapse them to one household-level link.
        panel = panel.drop_duplicates("y5_hhid")

    out = household.copy()
    before = len(out)
    out["hhid"] = clean_id(out["hhid"])

    out = out.merge(
        panel,
        left_on="hhid",
        right_on="y5_hhid",
        how="inner",
        validate="one_to_one",
    )
    panel_matched = len(out)
    if panel_matched == 0:
        raise ValueError(
            "W5 produced zero households after merging with npsy5.panel.key.csv. "
            "Check that y5_hhid in the panel key uses the same format as the W5 household hhid."
        )

    out = out.merge(
        coordinates,
        on="y4_hhid",
        how="inner",
        validate="many_to_one",
    )
    after = len(out)
    matched_w4_households = int(out["y4_hhid"].nunique(dropna=True))

    if after == 0:
        raise ValueError(
            "W5 produced zero households after matching official y4_hhid links to W4 coordinates."
        )
    if after > max_households:
        raise ValueError(
            f"W5 official panel-key merge kept {after:,} households, above the safety cap "
            f"of {max_households:,}. Check the panel key before using W5."
        )

    return out, {
        "survey_round": "W5",
        "coordinate_file": "none_confidential_w5_coordinates_not_used",
        "coordinate_rule": "official_npsy5_panel_key_y5_hhid_to_y4_hhid_inherit_w4_coordinates",
        "panel_key_file": str(panel_key_path.resolve()),
        "household_rows": before,
        "panel_key_unique_y5_links": int(panel["y5_hhid"].nunique(dropna=True)),
        "ambiguous_y5_links_dropped": int(len(ambiguous_y5)),
        "panel_matched_households": panel_matched,
        "georeferenced_households": after,
        "matched_w4_households_unique": matched_w4_households,
        "share_georeferenced": after / before if before else np.nan,
        "w5_coordinate_interpretation": "baseline_w4_location_not_current_w5_location",
    }

def summarize_individuals(path: Path) -> pd.DataFrame:
    data = read_required(path, INDIVIDUAL_COLUMNS, "individual file")
    data["hhid"] = clean_id(data["hhid"])
    data["female"] = numeric(data["female"])
    data["age"] = numeric(data["age"])
    return (
        data.groupby("hhid", as_index=False, sort=False)
        .agg(
            share_female=("female", "mean"),
            n_children_u15=("age", lambda x: int(numeric(x).lt(15).sum())),
            n_adults_15plus=("age", lambda x: int(numeric(x).ge(15).sum())),
        )
    )


def summarize_agriculture(path: Path, has_season: bool) -> pd.DataFrame:
    columns = PLOT_BASE_COLUMNS + (["season"] if has_season else [])
    data = read_required(path, columns, "field-plot file")
    data["hhid"] = clean_id(data["hhid"])
    data["plot_id"] = clean_id(data["plot_id"])
    data = data.loc[data["hhid"].notna() & data["plot_id"].notna()].copy()

    if data.empty:
        return pd.DataFrame(columns=["hhid", *AGRICULTURE_OUTCOMES[1:]])

    data["_season"] = clean_id(data["season"]).fillna("annual") if has_season else "annual"
    value_columns = [
        "area_meas_hectares",
        "plot_value_harvest_2ppp",
        "labor_hired",
        "labor_family",
        "labor_total",
    ]
    for column in value_columns:
        data[column] = numeric(data[column]).where(lambda x: x >= 0)

    plot_season = (
        data.groupby(["hhid", "plot_id", "_season"], as_index=False, sort=False)[value_columns]
        .max()
    )
    plot_land = (
        plot_season.groupby(["hhid", "plot_id"], as_index=False, sort=False)["area_meas_hectares"]
        .max()
    )
    land = (
        plot_land.groupby("hhid", as_index=False, sort=False)
        .agg(
            ag_plot_count=("plot_id", "nunique"),
            ag_land_ha=("area_meas_hectares", lambda x: numeric(x).sum(min_count=1)),
        )
    )
    flows = (
        plot_season.groupby("hhid", as_index=False, sort=False)
        .agg(
            ag_family_labor_days=("labor_family", lambda x: numeric(x).sum(min_count=1)),
            ag_hired_labor_days=("labor_hired", lambda x: numeric(x).sum(min_count=1)),
            ag_total_labor_days=("labor_total", lambda x: numeric(x).sum(min_count=1)),
            ag_harvest_value_2ppp=("plot_value_harvest_2ppp", lambda x: numeric(x).sum(min_count=1)),
        )
    )
    return land.merge(flows, on="hhid", how="outer", validate="one_to_one")


def add_agriculture(household: pd.DataFrame, agriculture: pd.DataFrame) -> pd.DataFrame:
    out = household.merge(agriculture, on="hhid", how="left", validate="one_to_one")
    out["ag_participation"] = numeric(out["ag_hh"])
    invalid = out["ag_participation"].notna() & ~out["ag_participation"].isin([0, 1])
    if invalid.any():
        raise ValueError(f"ag_hh contains {int(invalid.sum())} values outside 0/1.")

    zero_columns = AGRICULTURE_OUTCOMES[1:]
    non_agricultural = out["ag_participation"].eq(0)
    for column in zero_columns:
        out.loc[non_agricultural, column] = 0.0
    return out


def summarize_live_sales(path: Path, cattle_codes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = read_required(path, LIVE_SALES_COLUMNS, "live-animal sales file")
    for column in ["hhid", "livestock_code", "district", "ward", "ea", "region"]:
        sales[column] = clean_id(sales[column])
    for column in ["number_sold", "income_live_sales", "price_per_animal"]:
        sales[column] = numeric(sales[column])

    sales["actual_livestock_sale"] = (
        sales["number_sold"].gt(0) | sales["income_live_sales"].gt(0)
    ).astype("int8")
    sales["is_cattle"] = sales["livestock_code"].isin(cattle_codes).astype("int8")

    rows = []
    for hhid, group in sales.groupby("hhid", sort=False):
        actual = group["actual_livestock_sale"].eq(1)
        cattle = actual & group["is_cattle"].eq(1)
        rows.append(
            {
                "hhid": hhid,
                "livestock_any_sale": int(actual.any()),
                "livestock_number_sold_total": float(group["number_sold"].where(actual).sum(min_count=1)),
                "livestock_sales_revenue_total": float(group["income_live_sales"].where(actual).sum(min_count=1)),
                "cattle_any_sale": int(cattle.any()),
                "cattle_number_sold_total": float(group["number_sold"].where(cattle).sum(min_count=1)),
                "cattle_sales_revenue_total": float(group["income_live_sales"].where(cattle).sum(min_count=1)),
            }
        )
    return sales, pd.DataFrame(rows)


def summarize_product_prices(
    products_path: Path,
    ea_prices_path: Path,
    min_ea_observations: int,
    distress_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = read_required(products_path, PRODUCT_COLUMNS, "livestock-products file")
    ea_prices = read_required(
        ea_prices_path,
        PRODUCT_EA_PRICE_COLUMNS,
        "EA livestock-product price file",
    )

    keys = ["livestock_code", "district", "ward", "ea", "region"]
    for column in ["hhid", *keys]:
        products[column] = clean_id(products[column])
    for column in keys:
        ea_prices[column] = clean_id(ea_prices[column])

    for column in ["earnings_sales", "quantity_produced", "price_per_unit_hh"]:
        products[column] = numeric(products[column])
    ea_prices["obs_ea"] = numeric(ea_prices["obs_ea"])
    ea_prices["price_median_ea"] = numeric(ea_prices["price_median_ea"])

    if ea_prices.duplicated(keys).any():
        raise ValueError("EA livestock-product price file contains duplicated merge keys.")

    data = products.merge(ea_prices, on=keys, how="left", validate="many_to_one")
    data["actual_livestock_product_sale"] = data["earnings_sales"].gt(0).astype("int8")
    valid = (
        data["actual_livestock_product_sale"].eq(1)
        & data["price_per_unit_hh"].gt(0)
        & data["price_median_ea"].gt(0)
        & data["obs_ea"].ge(min_ea_observations)
    )
    data["valid_ea_product_price_benchmark"] = valid.astype("int8")
    data["livestock_product_log_price_gap"] = np.where(
        valid,
        np.log(data["price_per_unit_hh"]) - np.log(data["price_median_ea"]),
        np.nan,
    )
    data["livestock_product_distress_sale"] = np.where(
        valid,
        data["price_per_unit_hh"].lt(distress_ratio * data["price_median_ea"]).astype(int),
        np.nan,
    )

    rows = []
    for hhid, group in data.groupby("hhid", sort=False):
        sold = group["actual_livestock_product_sale"].eq(1)
        benchmarked = group["valid_ea_product_price_benchmark"].eq(1)
        rows.append(
            {
                "hhid": hhid,
                "livestock_product_any_sale": int(sold.any()),
                "livestock_product_sales_revenue_total": float(
                    group["earnings_sales"].where(sold).sum(min_count=1)
                ),
                "livestock_product_price_records_benchmarked": int(benchmarked.sum()),
                "livestock_product_log_price_gap_mean": (
                    float(group.loc[benchmarked, "livestock_product_log_price_gap"].mean())
                    if benchmarked.any()
                    else np.nan
                ),
                "livestock_product_distress_sale_any": (
                    int(group.loc[benchmarked, "livestock_product_distress_sale"].eq(1).any())
                    if benchmarked.any()
                    else np.nan
                ),
            }
        )
    return data, pd.DataFrame(rows)


def merge_one_to_one(household: pd.DataFrame, module: pd.DataFrame, label: str) -> pd.DataFrame:
    module = module.copy()
    module["hhid"] = clean_id(module["hhid"])
    if module["hhid"].duplicated().any():
        raise ValueError(f"{label} contains duplicated hhid values.")
    return household.merge(module, on="hhid", how="left", validate="one_to_one")


def read_livestock_expenses(
    paths: dict[str, Path],
    source: str,
) -> pd.DataFrame:
    if source not in {"W3", "W4", "W5"}:
        return read_required(
            paths["livestock_expenses"],
            LIVESTOCK_EXPENSE_COLUMNS,
            "livestock-expenses file",
        )

    expenses = read_required(
        paths["livestock_expenses"],
        LIVESTOCK_EXPENSE_BASE_COLUMNS,
        f"{source} livestock-expenses file",
    )
    lrum = read_required(
        paths["lrum_expenses"],
        LRUM_EXPENSE_COLUMNS,
        f"{source} large-ruminant expenses file",
    )

    expenses["hhid"] = clean_id(expenses["hhid"])
    lrum["hhid"] = clean_id(lrum["hhid"])
    if expenses["hhid"].duplicated().any():
        raise ValueError(f"{source} livestock-expenses file contains duplicated hhid values.")
    if lrum["hhid"].duplicated().any():
        raise ValueError(f"{source} large-ruminant expenses file contains duplicated hhid values.")

    return expenses.merge(lrum, on="hhid", how="outer", validate="one_to_one")


def read_livestock_income(paths: dict[str, Path], source: str) -> pd.DataFrame:
    if paths["livestock_income"].exists():
        return read_required(
            paths["livestock_income"],
            LIVESTOCK_INCOME_COLUMNS,
            "livestock-income file",
        )

    if source != "W2":
        raise FileNotFoundError(
            f"livestock-income file not found: {paths['livestock_income']}"
        )

    products = read_required(
        paths["hh_livestock_products"],
        HH_LIVESTOCK_PRODUCTS_COLUMNS,
        "W2 household livestock-products file",
    )
    sales = read_required(
        paths["live_sales"],
        ["hhid", "income_live_sales", "income_slaughtered"],
        "W2 live-animal sales file",
    )
    expenses = read_livestock_expenses(paths, source)

    for frame in [products, sales, expenses]:
        frame["hhid"] = clean_id(frame["hhid"])

    sales["income_live_sales"] = numeric(sales["income_live_sales"])
    sales["income_slaughtered"] = numeric(sales["income_slaughtered"])
    sales["value_livestock_sales_row"] = (
        sales["income_live_sales"].fillna(0)
        + sales["income_slaughtered"].fillna(0)
    )
    live_sales = (
        sales.groupby("hhid", as_index=False, sort=False)
        .agg(
            value_livestock_sales=(
                "value_livestock_sales_row",
                lambda x: numeric(x).sum(min_count=1),
            )
        )
    )

    income = products.merge(live_sales, on="hhid", how="outer", validate="one_to_one")
    income = income.merge(expenses, on="hhid", how="outer", validate="one_to_one")

    components = [
        "value_livestock_sales",
        "value_livestock_products",
        "cost_fodder_livestock",
        "cost_hired_labor_livestock",
    ]
    for column in components:
        income[column] = numeric(income[column]).fillna(0)

    income["livestock_income"] = (
        income["value_livestock_sales"]
        + income["value_livestock_products"]
        - income["cost_fodder_livestock"]
        - income["cost_hired_labor_livestock"]
    )
    print("W2: reconstructed livestock_income from sales, products, and expenses.")
    return income[LIVESTOCK_INCOME_COLUMNS].copy()


def add_livestock_modules(
    household: pd.DataFrame,
    paths: dict[str, Path],
    source: str,
) -> pd.DataFrame:
    income = read_livestock_income(paths, source)
    expenses = read_livestock_expenses(paths, source)
    diseases = read_required(paths["livestock_diseases"], LIVESTOCK_DISEASE_COLUMNS, "livestock-diseases file")

    out = merge_one_to_one(household, income, "livestock-income file")
    out = merge_one_to_one(out, expenses, "livestock-expenses file")
    out = merge_one_to_one(out, diseases, "livestock-diseases file")

    disease_columns = LIVESTOCK_DISEASE_COLUMNS[1:]
    for column in disease_columns:
        out[column] = numeric(out[column])
    disease_observed = out[disease_columns].notna().any(axis=1)
    out["any_large_ruminant_disease"] = np.where(
        disease_observed,
        out[disease_columns].fillna(0).max(axis=1),
        np.nan,
    )

    for column in LIVESTOCK_EXPENSE_COLUMNS[1:]:
        out[column] = numeric(out[column])
    out["livestock_expenses"] = out[LIVESTOCK_EXPENSE_COLUMNS[1:]].sum(axis=1, min_count=1)
    return out


def add_constructed_variables(household: pd.DataFrame, year: int) -> pd.DataFrame:
    out = household.copy()
    out["log_cons_pc"] = np.log(numeric(out["percapita_cons"]).where(lambda x: x > 0))
    out["log_cons_ae"] = np.log(numeric(out["peraeq_cons"]).where(lambda x: x > 0))

    cattle_today = numeric(out["nb_cattle_today"])
    cattle_previous = numeric(out["nb_cattle_1yearago"])
    out["cattle_change_1y"] = cattle_today - cattle_previous
    out["cattle_decline_1y"] = np.where(
        out["cattle_change_1y"].notna(),
        out["cattle_change_1y"].lt(0).astype(int),
        np.nan,
    )
    previous_owner = cattle_previous.gt(0)
    out["cattle_exit_1y"] = np.where(previous_owner, cattle_today.eq(0).astype(int), np.nan)
    out["cattle_loss_share_1y"] = np.where(
        previous_owner,
        (cattle_previous - cattle_today).clip(lower=0) / cattle_previous,
        np.nan,
    )

    adulteq = numeric(out["adulteq"])
    out["livestock_total_per_ae"] = numeric(out["nb_ls_today"]) / adulteq.where(adulteq > 0)
    out["livestock_tlu_per_ae"] = numeric(out["lvstck_holding_tlu"]) / adulteq.where(adulteq > 0)

    total_income = numeric(out["total_income"])
    positive_total = total_income.gt(0)
    income_sources = {
        "livestock_income_share": "livestock_income",
        "crop_income_share": "crop_income",
        "nonfarm_income_share": "nonfarm_income",
        "self_employment_income_share": "self_employment_income",
        "nonagwage_income_share": "nonagwage_income",
    }
    for output, source in income_sources.items():
        out[output] = (numeric(out[source]) / total_income).where(positive_total)

    out["post_2009"] = int(year >= 2010)
    out["post_2013"] = int(year >= 2014)
    return out


def add_structural_zeros(household: pd.DataFrame) -> pd.DataFrame:
    out = household.copy()
    zero_columns = [
        "livestock_any_sale",
        "livestock_number_sold_total",
        "livestock_sales_revenue_total",
        "cattle_any_sale",
        "cattle_number_sold_total",
        "cattle_sales_revenue_total",
        "livestock_product_any_sale",
        "livestock_product_sales_revenue_total",
        "livestock_product_price_records_benchmarked",
    ]
    for column in zero_columns:
        out[column] = numeric(out[column]).fillna(0)

    no_cattle = numeric(out["nb_cattle_today"]).fillna(0).eq(0) & numeric(
        out["nb_cattle_1yearago"]
    ).fillna(0).eq(0)
    out.loc[no_cattle & out["any_large_ruminant_disease"].isna(), "any_large_ruminant_disease"] = 0

    no_livestock = numeric(out["nb_ls_today"]).fillna(0).eq(0)
    expense_columns = [
        "cost_fodder_livestock",
        "cost_lrum",
        "cost_hired_labor_livestock",
        "livestock_expenses",
        "value_milk_produced",
        "sales_livestock_products",
        "value_livestock_products",
    ]
    for column in expense_columns:
        out.loc[no_livestock & out[column].isna(), column] = 0
    return out


def attach_coordinates(frame: pd.DataFrame, household: pd.DataFrame) -> pd.DataFrame:
    crosswalk = household[["hhid_wave", "cluster_lat", "cluster_lon"]].drop_duplicates("hhid_wave")
    return frame.merge(crosswalk, on="hhid_wave", how="inner", validate="many_to_one")


def select_columns(data: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing output columns: {missing}")
    return data[columns].copy()


def build_source(
    data_root: Path,
    legacy_root: Path,
    source: str,
    config: dict,
    cattle_codes: set[str],
    min_ea_observations: int,
    distress_ratio: float,
    w4_coordinates: pd.DataFrame | None = None,
    wave5_panel_key: Path | None = None,
    wave5_max_households: int = 4200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], dict]:
    paths = source_paths(data_root, source, config)

    household = read_required(paths["household"], HOUSEHOLD_COLUMNS, "household file")
    household["hhid"] = clean_id(household["hhid"])
    household["nb_ls_today"] = numeric(household["lvstck_holding_all"])
    if household["hhid"].duplicated().any():
        raise ValueError(f"{source}: household file contains duplicated hhid values.")

    individuals = summarize_individuals(paths["individual"])
    household = household.merge(individuals, on="hhid", how="left", validate="one_to_one")

    agriculture = summarize_agriculture(paths["plot"], config["plot_has_season"])
    household = add_agriculture(household, agriculture)
    household = add_livestock_modules(household, paths, source)

    live_sales, live_sales_household = summarize_live_sales(paths["live_sales"], cattle_codes)
    products, product_household = summarize_product_prices(
        paths["products"],
        paths["product_ea_prices"],
        min_ea_observations,
        distress_ratio,
    )

    household = merge_one_to_one(household, live_sales_household, "live-sales summary")
    household = merge_one_to_one(household, product_household, "product-price summary")
    household = add_structural_zeros(household)

    household.insert(0, "survey_round", source)
    household.insert(1, "wave", config["wave"])
    household.insert(2, "survey_year", config["year"])
    household["hhid_wave"] = source + "_" + household["hhid"].astype("string")
    household = add_constructed_variables(household, config["year"])
    if source == "W5":
        if w4_coordinates is None:
            raise ValueError("W5 requires W4 coordinates from previously matched W4 households.")
        household, coordinate_report = apply_w5_coordinates_from_w4(
            household,
            w4_coordinates,
            panel_key_path=wave5_panel_key or paths["panel_key"],
            max_households=wave5_max_households,
        )
    else:
        household, coordinate_report = apply_fixed_georeferencing(household, legacy_root, source)
    household = household.loc[household[["cluster_lat", "cluster_lon"]].notna().all(axis=1)].copy()

    for frame in [live_sales, products]:
        frame.insert(0, "survey_round", source)
        frame.insert(1, "wave", config["wave"])
        frame.insert(2, "survey_year", config["year"])
        frame["hhid_wave"] = source + "_" + frame["hhid"].astype("string")

    live_sales = attach_coordinates(live_sales, household)
    products = products.loc[products["hhid_wave"].isin(household["hhid_wave"])].copy()

    reports = [
        {"survey_round": source, "module": name, "file": str(path.resolve())}
        for name, path in paths.items()
    ]
    return household, live_sales, products, reports, coordinate_report


def main() -> None:
    args = parse_args()
    print(f"extract_relevant_nps_variables version: {SCRIPT_VERSION}")
    if args.min_ea_price_observations < 2:
        raise ValueError("--min-ea-price-observations must be at least 2.")
    if not 0 < args.distress_price_ratio < 1:
        raise ValueError("--distress-price-ratio must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_root = args.legacy_data_root or args.data_root
    cattle_codes = parse_code_set(args.cattle_codes)

    household_frames = []
    live_sales_frames = []
    product_frames = []
    reports = []
    coordinate_reports = []
    w4_coordinates = None

    for source, config in SOURCES.items():
        if source == "W5" and not args.include_wave5:
            print("Skipping W5 because --include-wave5 was not supplied.")
            continue
        print(f"Extracting {source}...")
        household, live_sales, products, source_reports, coordinate_report = build_source(
            args.data_root,
            legacy_root,
            source,
            config,
            cattle_codes,
            args.min_ea_price_observations,
            args.distress_price_ratio,
            w4_coordinates=w4_coordinates,
            wave5_panel_key=args.wave5_panel_key,
            wave5_max_households=args.wave5_max_households,
        )
        if source == "W4":
            w4_coordinates = household[["hhid", "cluster_lat", "cluster_lon"]].copy()

        lean_household = select_columns(household, HOUSEHOLD_OUTPUT_COLUMNS, "household output")
        lean_live_sales = select_columns(live_sales, LIVE_SALES_OUTPUT_COLUMNS, "live-sales output")
        lean_products = select_columns(products, PRODUCT_PRICE_OUTPUT_COLUMNS, "product-price output")

        lean_household.to_csv(args.output_dir / f"nps_relevant_{source.lower()}.csv", index=False)
        lean_live_sales.to_csv(
            args.output_dir / f"nps_livestock_sales_{source.lower()}_long.csv", index=False
        )
        lean_products.to_csv(
            args.output_dir / f"nps_livestock_product_prices_{source.lower()}_long.csv", index=False
        )

        household_frames.append(lean_household)
        live_sales_frames.append(lean_live_sales)
        product_frames.append(lean_products)
        reports.extend(source_reports)
        coordinate_reports.append(coordinate_report)
        print(
            f"  households: {len(lean_household):,}; "
            f"live-sale rows: {len(lean_live_sales):,}; "
            f"product rows: {len(lean_products):,}"
        )

    household = pd.concat(household_frames, ignore_index=True, sort=False)
    live_sales = pd.concat(live_sales_frames, ignore_index=True, sort=False)
    products = pd.concat(product_frames, ignore_index=True, sort=False)

    if household["hhid_wave"].duplicated().any():
        raise ValueError("Combined household output contains duplicated hhid_wave values.")

    household.to_csv(args.output_dir / "nps_relevant_household_wave.csv", index=False)
    select_columns(household, AGRICULTURE_OUTPUT_COLUMNS, "agriculture output").to_csv(
        args.output_dir / "nps_agriculture_household_wave.csv", index=False
    )
    live_sales.to_csv(args.output_dir / "nps_livestock_sales_long.csv", index=False)
    products.to_csv(args.output_dir / "nps_livestock_product_prices_long.csv", index=False)
    pd.DataFrame(reports).to_csv(args.output_dir / "nps_relevant_build_report.csv", index=False)
    pd.DataFrame(coordinate_reports).to_csv(
        args.output_dir / "nps_fixed_georeferencing_sources.csv", index=False
    )

    product_diagnostic = (
        products.groupby("survey_round", as_index=False)
        .agg(
            product_rows=("hhid_wave", "size"),
            observed_product_sales=("earnings_sales", lambda x: int(numeric(x).gt(0).sum())),
            benchmarked_product_sales=(
                "valid_ea_product_price_benchmark",
                lambda x: int(numeric(x).eq(1).sum()),
            ),
        )
    )
    product_diagnostic.to_csv(
        args.output_dir / "nps_livestock_product_price_diagnostic.csv", index=False
    )

    print(f"Household output: {args.output_dir / 'nps_relevant_household_wave.csv'}")
    print(f"Live-animal sales output: {args.output_dir / 'nps_livestock_sales_long.csv'}")
    print(
        "Livestock-product price output: "
        f"{args.output_dir / 'nps_livestock_product_prices_long.csv'}"
    )


if __name__ == "__main__":
    main()
