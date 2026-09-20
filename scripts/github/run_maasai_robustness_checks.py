#!/usr/bin/env python3
"""
Run robustness checks for the Maasai / conservation Tanzania NPS analysis.

This script is intentionally a wrapper around the main regression script. It
keeps the identifying specification unchanged and varies only the robustness
margin requested in the thesis:

1. reserve-boundary risk set <= 50 km, keeping the main is_maasai_25km treatment;
2. more selective Maasai-associated-area treatment: is_maasai_10km.

The script writes one folder per robustness check and also creates a compact
paper-style LaTeX table comparing the key coefficients across checks.

Example
-------
python scripts/github/run_maasai_robustness_checks.py \
  --input-dir output/nps_final \
  --output-dir output/regressions/robustness_checks \
  --main-script scripts/github/run_maasai_main_specifications.py \
  --exclude-wave5 \
  --wild-bootstrap-reps 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-06-28-robustness-main-25km-v3"
HOUSEHOLD_FILE = "nps_relevant_household_wave.csv"
INSIDE_RESERVE = "inside_game_reserve"

SCENARIOS = [
    {
        "name": "riskset_50km",
        "label": "Risk set $\\leq$ 50 km",
        "main_boundary_km": 50.0,
        "treatment_column": "is_maasai_25km",
        "exclude_inside_reserves": False,
    },
    {
        "name": "treatment_maasai_10km",
        "label": "Maasai treatment 10 km",
        "main_boundary_km": 75.0,
        "treatment_column": "is_maasai_10km",
        "exclude_inside_reserves": False,
    },
]

OUTCOME_LABELS = {
    "log_cons_pc": "Log consumption per capita",
    "log_cons_ae": "Log consumption per adult equivalent",
    "poverty_under_npl": "Poverty indicator",
    "livestock_total_per_ae": "Total livestock per adult equivalent",
    "livestock_tlu_per_ae": "Livestock TLU per adult equivalent",
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

KEY_OUTCOMES = [
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

SHOCK_ORDER = [
    "post_2009_first_shift",
    "post_2013_additional_shift",
    "total_post_2013",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness checks for the Maasai NPS analysis.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--main-script", default=Path("scripts/github/run_maasai_main_specifications.py"), type=Path)
    parser.add_argument("--household-file", default=HOUSEHOLD_FILE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sample-for-summary", default="full_risk_set")
    parser.add_argument("--min-n", default=100, type=int)
    parser.add_argument("--min-clusters", default=20, type=int)
    parser.add_argument("--wild-bootstrap-reps", default=0, type=int)
    parser.add_argument("--wild-bootstrap-seed", default=20260625, type=int)
    parser.add_argument("--wild-bootstrap-type", default="31", choices=["11", "13", "31", "33", "fnw11"])
    parser.add_argument("--exclude-wave5", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def clean_previous_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def prepare_no_inside_input(input_dir: Path, household_file: str, temp_dir: Path) -> tuple[Path, str, dict]:
    source = input_dir / household_file
    require_file(source, "household file")
    temp_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source, low_memory=False)
    if INSIDE_RESERVE not in data.columns:
        raise ValueError(f"Cannot exclude inside-reserve households because {INSIDE_RESERVE!r} is missing.")

    inside = pd.to_numeric(data[INSIDE_RESERVE], errors="coerce").eq(1)
    out = data.loc[~inside].copy()
    filename = "nps_relevant_household_wave_no_inside_reserves.csv"
    out_path = temp_dir / filename
    out.to_csv(out_path, index=False)
    diagnostics = {
        "original_rows": int(len(data)),
        "dropped_inside_reserve_rows": int(inside.sum()),
        "remaining_rows": int(len(out)),
    }
    return temp_dir, filename, diagnostics


def scenario_command(args: argparse.Namespace, scenario: dict, scenario_input_dir: Path, household_file: str, scenario_output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(args.main_script),
        "--input-dir", str(scenario_input_dir),
        "--output-dir", str(scenario_output_dir),
        "--household-file", household_file,
        "--treatment-column", scenario["treatment_column"],
        "--main-boundary-km", str(scenario["main_boundary_km"]),
        "--min-n", str(args.min_n),
        "--min-clusters", str(args.min_clusters),
        "--wild-bootstrap-reps", str(args.wild_bootstrap_reps),
        "--wild-bootstrap-seed", str(args.wild_bootstrap_seed),
        "--wild-bootstrap-type", args.wild_bootstrap_type,
    ]
    if args.exclude_wave5:
        cmd.append("--exclude-wave5")
    return cmd


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_scenario_results(scenario_output_dir: Path) -> pd.DataFrame | None:
    candidates = [
        scenario_output_dir / "table_main_specifications_raw.csv",
        scenario_output_dir / "table_main_specifications_combined_raw.csv",
        scenario_output_dir / "table_main_specifications_combined.csv",
    ]
    existing = first_existing(candidates)
    if existing is None:
        # Some versions write one CSV per outcome family only. Fall back to concatenating them.
        family_files = sorted(scenario_output_dir.glob("table_*_raw.csv"))
        if not family_files:
            family_files = sorted(scenario_output_dir.glob("table_*.csv"))
        family_files = [p for p in family_files if "support" not in p.name and "district" not in p.name]
        if not family_files:
            return None
        frames = [pd.read_csv(path, low_memory=False) for path in family_files]
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.read_csv(existing, low_memory=False)


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


def bootstrap_results_available(df: pd.DataFrame) -> bool:
    """Return True when at least one valid wild-bootstrap p-value is available."""
    if "wild_bootstrap_p_value" not in df.columns:
        return False
    p = pd.to_numeric(df["wild_bootstrap_p_value"], errors="coerce")
    if "wild_bootstrap_status" in df.columns:
        status = df["wild_bootstrap_status"].astype(str).str.lower().eq("ok")
        return bool((p.notna() & status).any())
    return bool(p.notna().any())


def first_nonmissing(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    values = df[column].dropna().astype(str)
    values = values.loc[values.ne("")]
    return values.iloc[0] if len(values) else ""


def format_cell(row: pd.Series | None, use_bootstrap: bool = False) -> tuple[str, str, str]:
    """
    Format one regression-table cell.

    Stars are based on ordinary clustered p-values. Wild-bootstrap p-values,
    when available, are reported separately in square brackets.
    """
    if row is None or row.empty:
        return "", "", ""

    coef = pd.to_numeric(pd.Series([row.get("coef")]), errors="coerce").iloc[0]
    se = pd.to_numeric(pd.Series([row.get("se")]), errors="coerce").iloc[0]
    p = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
    wb_p = pd.to_numeric(pd.Series([row.get("wild_bootstrap_p_value")]), errors="coerce").iloc[0]
    wb_status = str(row.get("wild_bootstrap_status", "")).lower()

    if pd.isna(coef):
        return "", "", ""

    bootstrap_ok = use_bootstrap and pd.notna(wb_p) and wb_status == "ok"

    coef_text = f"{coef:.3f}{significance_stars(p)}"
    se_text = f"({se:.3f})" if pd.notna(se) else ""
    wb_text = f"[{wb_p:.3f}]" if bootstrap_ok else ""
    return coef_text, se_text, wb_text


def make_robustness_table(combined: pd.DataFrame, scenario_labels: dict[str, str], sample: str, path: Path) -> None:
    required = {"scenario", "sample", "outcome", "shock", "coef", "se", "p_value"}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"Cannot make robustness table; missing columns: {missing}")

    df = combined.loc[combined["sample"].eq(sample)].copy()
    if "status" in df.columns:
        df = df.loc[df["status"].fillna("ok").eq("ok")].copy()

    show_bootstrap = bootstrap_results_available(df)
    bootstrap_reps = first_nonmissing(df, "wild_bootstrap_reps")
    bootstrap_type = first_nonmissing(df, "wild_bootstrap_type")

    scenario_order = [scenario["name"] for scenario in SCENARIOS]
    cols = "l" + "c" * len(scenario_order)
    header = "Outcome and coefficient & " + " & ".join(scenario_labels[s] for s in scenario_order) + r" \\" 
    lines = []
    lb = r"\\"
    lines.append(r"\begin{table}[!htbp]\centering")
    lines.append(r"\caption{Robustness checks for the main specification}")
    lines.append(r"\label{tab:robustness-main}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{" + cols + r"}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for outcome in KEY_OUTCOMES:
        wrote_outcome = False
        for shock in SHOCK_ORDER:
            row_label = latex_escape(f"{outcome_label(outcome)} -- {shock_label(shock)}")
            coef_cells = []
            se_cells = []
            wb_cells = []
            for scenario_name in scenario_order:
                subset = df.loc[
                    df["scenario"].eq(scenario_name)
                    & df["outcome"].eq(outcome)
                    & df["shock"].eq(shock)
                ]
                row = subset.iloc[0] if not subset.empty else None
                coef_text, se_text, wb_text = format_cell(row, use_bootstrap=show_bootstrap)
                coef_cells.append(coef_text)
                se_cells.append(se_text)
                wb_cells.append(wb_text)
            if not any(coef_cells):
                continue
            lines.append(row_label + " & " + " & ".join(coef_cells) + " " + lb)
            lines.append(" " + " & " + " & ".join(se_cells) + " " + lb)
            if show_bootstrap and any(wb_cells):
                lines.append(r"\emph{Wild p-value} & " + " & ".join(wb_cells) + " " + lb)
            wrote_outcome = True
        if wrote_outcome:
            lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    for stat, label in [("n", "Observations"), ("clusters", "EA clusters")]:
        cells = []
        for scenario_name in scenario_order:
            sub = df.loc[df["scenario"].eq(scenario_name)]
            value = pd.to_numeric(sub[stat], errors="coerce").dropna() if stat in sub.columns else pd.Series(dtype=float)
            cells.append(f"{int(value.iloc[0]):,}" if len(value) else "")
        lines.append(latex_escape(label) + " & " + " & ".join(cells) + " " + lb)
    if show_bootstrap:
        rep_label = f"Wild bootstrap reps"
        rep_cells = []
        for scenario_name in scenario_order:
            sub = df.loc[df["scenario"].eq(scenario_name)]
            reps = pd.to_numeric(sub.get("wild_bootstrap_reps", pd.Series(dtype=float)), errors="coerce").dropna()
            rep_cells.append(f"{int(reps.iloc[0]):,}" if len(reps) else bootstrap_reps)
        lines.append(latex_escape(rep_label) + " & " + " & ".join(rep_cells) + " " + lb)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if show_bootstrap:
        bootstrap_note = (
            r"Wild-cluster bootstrap p-values are reported in square brackets when available. "
            r"Significance stars are based on ordinary clustered p-values, not wild-bootstrap p-values. "
            r"Conventional EA-clustered standard errors remain in parentheses. "
            r"For linear-combination rows, bootstrap p-values may be unavailable. "
        )
        if bootstrap_reps or bootstrap_type:
            bootstrap_note += latex_escape(f" Bootstrap settings: reps={bootstrap_reps or 'NA'}, type={bootstrap_type or 'NA'}. ")
    else:
        bootstrap_note = r"Ordinary p-values are used for significance stars because no valid wild-bootstrap p-values were found. "

    lines.append(
        r"\begin{minipage}{0.96\textwidth}\vspace{0.3em}\footnotesize "
        r"Notes: each column re-estimates the main two-shock repeated-cross-section specification under a different robustness choice. "
        r"The reported coefficients are treatment-by-post interactions. "
        + bootstrap_note
        + r"Significance levels: * p$<$0.10, ** p$<$0.05, *** p$<$0.01. "
        r"q-values are intentionally not reported.\end{minipage}"
    )
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scenario_manifest(output_dir: Path, diagnostics_rows: list[dict]) -> None:
    # No separate raw CSV is written here: robustness_checks_combined_raw.csv
    # is the single raw output for this script. Scenario metadata are also
    # embedded in that combined CSV.
    return None


def main() -> None:
    args = parse_args()
    print(f"run_maasai_robustness_checks version: {SCRIPT_VERSION}")
    require_file(args.main_script, "main regression script")
    require_file(args.input_dir / args.household_file, "household file")
    clean_previous_outputs(args.output_dir)

    all_results = []
    manifest_rows = []
    temp_root = args.output_dir / "_temporary_inputs"

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        scenario_output_dir = args.output_dir / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)

        input_dir = args.input_dir
        household_file = args.household_file
        extra_diag = {}
        if scenario["exclude_inside_reserves"]:
            input_dir, household_file, extra_diag = prepare_no_inside_input(
                args.input_dir,
                args.household_file,
                temp_root / scenario_name,
            )

        cmd = scenario_command(args, scenario, input_dir, household_file, scenario_output_dir)
        run_command(cmd, args.dry_run)

        manifest_row = {
            "scenario": scenario_name,
            "scenario_label": scenario["label"],
            "main_boundary_km": scenario["main_boundary_km"],
            "treatment_column": scenario["treatment_column"],
            "exclude_inside_reserves": int(scenario["exclude_inside_reserves"]),
            "wild_bootstrap_reps": args.wild_bootstrap_reps,
            "wild_bootstrap_seed": args.wild_bootstrap_seed,
            "wild_bootstrap_type": args.wild_bootstrap_type,
            "output_dir": str(scenario_output_dir),
            **extra_diag,
        }
        manifest_rows.append(manifest_row)

        if not args.dry_run:
            result = read_scenario_results(scenario_output_dir)
            if result is not None and not result.empty:
                result.insert(0, "scenario", scenario_name)
                result.insert(1, "scenario_label", scenario["label"])
                result.insert(2, "robustness_boundary_km", scenario["main_boundary_km"])
                result.insert(3, "robustness_treatment", scenario["treatment_column"])
                result.insert(4, "robustness_exclude_inside_reserves", int(scenario["exclude_inside_reserves"]))
                all_results.append(result)

    write_scenario_manifest(args.output_dir, manifest_rows)

    if args.dry_run:
        print("\nDry run complete. No regressions were estimated.")
        return

    if all_results:
        combined = pd.concat(all_results, ignore_index=True, sort=False)
        combined.to_csv(args.output_dir / "robustness_checks_combined_raw.csv", index=False)
        scenario_labels = {row["scenario"]: row["scenario_label"] for row in manifest_rows}
        make_robustness_table(
            combined,
            scenario_labels,
            args.sample_for_summary,
            args.output_dir / "table_robustness_checks_main.tex",
        )
        print(f"\nWrote combined robustness CSV: {args.output_dir / 'robustness_checks_combined_raw.csv'}")
        print(f"Wrote LaTeX robustness table: {args.output_dir / 'table_robustness_checks_main.tex'}")
    else:
        print("\nNo scenario result CSVs were found. Individual scenario folders may still contain LaTeX tables.")


if __name__ == "__main__":
    main()
