from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


SHAPEFILE = Path("./GeoEPR-2021.shp")
OUTPUT_CSV = Path("./geoepr_maasai_changes.csv")
OUTPUT_GPKG = Path("./geoepr_maasai_periods.gpkg")
OUTPUT_PNG = Path("./geoepr_maasai_periods.png")


def normalise_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(columns, candidates):
    lookup = {normalise_name(c): c for c in columns}

    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]

    return None


if not SHAPEFILE.exists():
    raise FileNotFoundError(
        f"{SHAPEFILE} non trovato nella directory corrente: {Path.cwd()}"
    )

print(f"Reading {SHAPEFILE}...")
gdf = gpd.read_file(SHAPEFILE)

print("\nGENERAL INFORMATION")
print("-" * 70)
print(f"Rows: {len(gdf):,}")
print(f"CRS: {gdf.crs}")
print("\nColumns:")
for column in gdf.columns:
    print(f"  - {column}")

if gdf.crs is None:
    raise ValueError(
        "The shapefile has no CRS. Check that GeoEPR-2021.prj is present."
    )


# ------------------------------------------------------------
# 1. Find every row mentioning Maasai or Masai
# ------------------------------------------------------------

text_columns = [
    column
    for column in gdf.columns
    if column != gdf.geometry.name
    and (
        pd.api.types.is_object_dtype(gdf[column])
        or pd.api.types.is_string_dtype(gdf[column])
    )
]

maasai_pattern = r"\bmaasai\b|\bmasai\b"

maasai_mask = pd.Series(False, index=gdf.index)

for column in text_columns:
    maasai_mask |= (
        gdf[column]
        .astype("string")
        .str.contains(maasai_pattern, case=False, regex=True, na=False)
    )

maasai = gdf.loc[maasai_mask].copy()

if maasai.empty:
    raise ValueError(
        "No row containing 'Maasai' or 'Masai' was found. "
        "Inspect the printed columns and the group-name coding."
    )

print("\nMAASAI MATCHES")
print("-" * 70)
print(f"Rows containing Maasai/Masai: {len(maasai):,}")

for column in text_columns:
    matched_values = (
        maasai.loc[
            maasai[column]
            .astype("string")
            .str.contains(maasai_pattern, case=False, regex=True, na=False),
            column,
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(matched_values):
        print(f"{column}: {list(matched_values)}")


# ------------------------------------------------------------
# 2. Retain Tanzania when a country-name variable is available
# ------------------------------------------------------------

country_columns = [
    column
    for column in text_columns
    if any(
        token in normalise_name(column)
        for token in ["country", "state", "statename", "cntry"]
    )
]

tanzania_mask = pd.Series(False, index=maasai.index)

for column in country_columns:
    tanzania_mask |= (
        maasai[column]
        .astype("string")
        .str.contains("tanzania", case=False, na=False)
    )

if tanzania_mask.any():
    maasai = maasai.loc[tanzania_mask].copy()
    print(f"\nRestricted to Tanzania: {len(maasai):,} rows")
else:
    print(
        "\nWARNING: no country-name field identifying Tanzania was found."
        "\nAll Maasai/Masai records are retained. Check whether Kenya is also present."
    )

    for column in country_columns:
        print(f"{column}: {maasai[column].dropna().unique().tolist()}")


# ------------------------------------------------------------
# 3. Identify temporal variables
# ------------------------------------------------------------

start_column = find_column(
    maasai.columns,
    [
        "from",
        "start",
        "start_year",
        "year_from",
        "begin",
        "begin_year",
        "first_year",
    ],
)

end_column = find_column(
    maasai.columns,
    [
        "to",
        "end",
        "end_year",
        "year_to",
        "until",
        "last_year",
    ],
)

single_year_column = find_column(
    maasai.columns,
    ["year", "date", "period_year"],
)

print("\nTEMPORAL COLUMNS")
print("-" * 70)
print(f"Start column: {start_column}")
print(f"End column:   {end_column}")
print(f"Single year:  {single_year_column}")

possible_time_columns = [
    column
    for column in maasai.columns
    if any(
        token in normalise_name(column)
        for token in ["year", "from", "start", "begin", "to", "end", "until"]
    )
]

print(f"All possible temporal columns: {possible_time_columns}")

if start_column and end_column:
    maasai["start_year"] = pd.to_numeric(
        maasai[start_column], errors="coerce"
    )
    maasai["end_year"] = pd.to_numeric(
        maasai[end_column], errors="coerce"
    )
    period_columns = ["start_year", "end_year"]

elif single_year_column:
    maasai["year"] = pd.to_numeric(
        maasai[single_year_column], errors="coerce"
    )
    period_columns = ["year"]

else:
    raw_output = Path("geoepr_maasai_raw_records.gpkg")
    maasai.to_file(raw_output, layer="maasai_raw", driver="GPKG")

    raise ValueError(
        "No temporal columns were automatically identified. "
        f"The Maasai records were saved to {raw_output}. "
        "Inspect the printed columns and assign the correct start/end variables."
    )


# ------------------------------------------------------------
# 4. Validate and dissolve all polygon parts by period
# ------------------------------------------------------------

maasai = maasai.loc[maasai.geometry.notna()].copy()

try:
    maasai["geometry"] = maasai.geometry.make_valid()
except Exception:
    maasai["geometry"] = maasai.geometry.buffer(0)

feature_counts = (
    maasai.groupby(period_columns, dropna=False)
    .size()
    .rename("n_source_features")
    .reset_index()
)

periods = (
    maasai[period_columns + ["geometry"]]
    .dissolve(by=period_columns, as_index=False)
    .merge(feature_counts, on=period_columns, how="left")
)

periods = periods.sort_values(period_columns).reset_index(drop=True)

if periods.empty:
    raise ValueError("No valid Maasai geometries remained after cleaning.")


# ------------------------------------------------------------
# 5. Compare consecutive polygons in an equal-area projection
# ------------------------------------------------------------

periods_metric = periods.to_crs("EPSG:6933").copy()

periods["area_km2"] = periods_metric.geometry.area / 1_000_000
periods["geometry_equal_previous"] = pd.NA
periods["symmetric_difference_km2"] = np.nan
periods["changed_share_of_union_pct"] = np.nan
periods["intersection_over_union"] = np.nan
periods["area_change_pct"] = np.nan

for index in range(len(periods)):
    if index == 0:
        continue

    previous = periods_metric.geometry.iloc[index - 1]
    current = periods_metric.geometry.iloc[index]

    previous_area = previous.area
    current_area = current.area

    intersection_area = previous.intersection(current).area
    union_area = previous.union(current).area
    symmetric_difference_area = previous.symmetric_difference(current).area

    periods.loc[index, "geometry_equal_previous"] = bool(
        previous.equals(current)
    )
    periods.loc[index, "symmetric_difference_km2"] = (
        symmetric_difference_area / 1_000_000
    )
    periods.loc[index, "changed_share_of_union_pct"] = (
        100 * symmetric_difference_area / union_area
        if union_area > 0
        else np.nan
    )
    periods.loc[index, "intersection_over_union"] = (
        intersection_area / union_area
        if union_area > 0
        else np.nan
    )
    periods.loc[index, "area_change_pct"] = (
        100 * (current_area - previous_area) / previous_area
        if previous_area > 0
        else np.nan
    )

periods["geometry_equal_previous"] = (
    periods["geometry_equal_previous"].astype("boolean")
)


# ------------------------------------------------------------
# 6. Explicitly flag boundaries around 2009
# ------------------------------------------------------------

if "start_year" in periods.columns:
    periods["starts_in_2009"] = periods["start_year"].eq(2009)
    periods["covers_2009"] = (
        periods["start_year"].le(2009)
        & periods["end_year"].ge(2009)
    )
    periods["previous_end_year"] = periods["end_year"].shift(1)
    periods["transition_around_2009"] = (
        periods["start_year"].isin([2009, 2010])
        | periods["end_year"].isin([2008, 2009])
    )
else:
    periods["is_2009"] = periods["year"].eq(2009)
    periods["transition_around_2009"] = periods["year"].isin(
        [2008, 2009, 2010]
    )


# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

periods.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

try:
    periods.to_file(
        OUTPUT_GPKG,
        layer="maasai_periods",
        driver="GPKG",
    )
except Exception as error:
    print(f"\nCould not write {OUTPUT_GPKG}: {error}")

try:
    ax = periods.boundary.plot(figsize=(9, 9), linewidth=2)
    ax.set_title("GeoEPR Maasai settlement polygon, Tanzania, 1992–2021")
    ax.set_axis_off()

    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(
        "geoepr_maasai_periods.png",
        dpi=200,
        bbox_inches="tight"
    )
except Exception as error:
    print(f"\nCould not create plot: {error}")


# ------------------------------------------------------------
# 8. Print interpretation table
# ------------------------------------------------------------

display_columns = period_columns + [
    "n_source_features",
    "area_km2",
    "geometry_equal_previous",
    "symmetric_difference_km2",
    "changed_share_of_union_pct",
    "intersection_over_union",
    "area_change_pct",
    "transition_around_2009",
]

print("\nMAASAI POLYGON CHANGES OVER TIME")
print("=" * 100)
print(
    periods[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}",
    )
)

around_2009 = periods.loc[
    periods["transition_around_2009"].fillna(False),
    display_columns,
]

print("\nPERIODS OR TRANSITIONS AROUND 2009")
print("=" * 100)

if around_2009.empty:
    print("No period boundary was found around 2009.")
else:
    print(
        around_2009.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

print("\nOUTPUTS")
print("-" * 70)
print(f"Summary:  {OUTPUT_CSV}")
print(f"Polygons: {OUTPUT_GPKG}")
print(f"Map:      {OUTPUT_PNG}")