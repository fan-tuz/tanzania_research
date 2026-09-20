# Expanded AI-assisted georeferenced dataset of Maasai-, Parakuyo- and Datoga/Barabaig-associated localities in Tanzania

## Contents
- `maasai_parakuyo_datoga_georeferenced_expanded.csv`: full simplified audit table (88 rows; unchanged 19-column schema).
- `community_wards_expanded.*`: 59 official 2022 ward polygons.
- `community_points_expanded.*`: 29 locality or facility point estimates.
- `maasai_parakuyo_datoga_georeferenced_expanded.gpkg`: both layers with the complete attributes.
- `additions_and_recoveries.csv`: the 21 rows added to the previous 67-row version and their recommended role.
- `source_register_expanded.csv`: unique documentary sources.
- `excluded_or_not_promoted_expanded.csv`: exclusions and disputed cases.
- `qa_summary_expanded.json`: automated quality checks.

## What changed
The previous version was intentionally conservative and removed recoverable observations. This version restores documented wards including Engaresero, Olbalbal, Naberera, Orkesumet, Kitwai and Naisinyai; retains mixed or less certain places such as Orgosorok, Selela, Lolkisale, Monduli Juu and Moita in a separately flagged sensitivity layer; and expands southern Kiteto with the explicitly Maasai-dominated villages Olkitikiti and Ndotoi. Bassotu and Bassodesh are added separately as Datoga/Barabaig-associated wards.

The dataset contains 77 Maasai, 3 Parakuyo and 8 Datoga/Barabaig observations. It is an evidence-coded spatial proxy, not an ethnic census.

## Interpretation
`point_estimate = 0` means an official 2022 ward polygon. The coordinates in the CSV are representative points only; spatial treatment must use the polygon geometry.

`point_estimate = 1` means a settlement or facility coordinate. Points were spatially joined back to the official 2022 ward layer to standardise ward, district and region names.

`needs_manual_check = 1` marks evidence that is mixed, territorial rather than locality-specific, based on a weaker publication type, or geocoded through an approximate proxy. These rows are deliberately retained rather than deleted, but they should not enter the preferred baseline.

## Recommended treatment definitions
- Preferred baseline: `confidence_score >= 0.80 & needs_manual_check == 0` (70 rows).
- Conservative baseline: `confidence_score >= 0.90 & needs_manual_check == 0` (47 rows).
- Expanded sensitivity: `confidence_score >= 0.75`, including manual-check rows only in a specification explicitly labelled expanded (79 rows).
- Manual-review-only archive: `confidence_score < 0.75` (9 rows). Do not use these in a main regression before manual validation.
- Keep Maasai, Parakuyo and Datoga/Barabaig as separate groups or interactions. Do not silently pool them.

## Use with NPS coordinates
For NPS privacy-displaced household coordinates, construct both 10 km and 25 km proximity definitions. Report 10 km as the narrower exposure and 25 km as a displacement-robust definition. For polygon rows, measure distance to the polygon boundary or intersection with a buffer; do not measure distance to the representative CSV coordinate.

## Inclusion rules
1. Direct locality-level documentary evidence receives the highest confidence.
2. Ward polygons are retained only where the evidence plausibly applies to the named ward or a very high local concentration is documented.
3. Mixed towns or territorial evidence are marked for manual checking and excluded from the preferred baseline.
4. Official Tanzania 2022 PHC spelling and geometry take precedence over historical spellings; spelling variants are retained in `alternate_names`.
5. Point estimates require an actual locality/facility coordinate. Ward centroids are not passed off as village points.
6. Sonjo/Batemi localities are not promoted into the Maasai treatment merely because they lie in Ngorongoro District.

## Shapefile note
ESRI Shapefile field names are shortened because of the DBF 10-character limit. Use the CSV or GeoPackage for the complete documentary fields. Both spatial layers use EPSG:4326.

## Limits
This dataset records documentary association, not individual ethnicity. Community composition changes over time, ward polygons can be large, pastoral mobility crosses administrative boundaries, and some source dates predate the 2022 ward geography. The defensible description is: **AI-assisted, source-audited and manually reviewable georeferenced dataset of localities explicitly or strongly associated with Maasai, Parakuyo and Datoga/Barabaig communities.**
