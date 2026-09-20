import csv
import os
import sys
from pathlib import Path

import pandas as pd

"""
https://github.com/EvansSchoolPolicyAnalysisAndResearch/LSMS-Data-Dissemination/tree/main/Tanzania%20NPS
"""

ROOT = Path('/home/mf/masai/nps_codebase_refactored/data/github')
REPORT_PATH = ROOT / "folder_contents.txt"

converted_files = []
conversion_errors = []


# ------------------------------------------------------------
# 1. Convert every .dta file recursively into .csv
# ------------------------------------------------------------

dta_files = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() == ".dta"
)

for dta_path in dta_files:
    csv_path = dta_path.with_suffix(".csv")
    temporary_csv = dta_path.parent / f".{dta_path.stem}.temporary.csv"

    try:
        print(f"Converting: {dta_path.relative_to(ROOT)}")

        # Numeric Stata codes are retained rather than replaced by value labels.
        dataframe = pd.read_stata(
            dta_path,
            convert_categoricals=False
        )

        # Write to a temporary file first, so the .dta is not deleted if
        # CSV creation fails.
        dataframe.to_csv(
            temporary_csv,
            index=False,
            encoding="utf-8"
        )

        # Replace any existing CSV with the newly converted file.
        os.replace(temporary_csv, csv_path)

        # Delete the original only after successful CSV creation.
        dta_path.unlink()

        converted_files.append(dta_path.relative_to(ROOT))

    except Exception as error:
        if temporary_csv.exists():
            temporary_csv.unlink()

        conversion_errors.append(
            (dta_path.relative_to(ROOT), str(error))
        )

        print(
            f"ERROR converting {dta_path.relative_to(ROOT)}: {error}",
            file=sys.stderr
        )


# ------------------------------------------------------------
# 2. Read the header of a CSV file
# ------------------------------------------------------------

def read_csv_header(csv_path):
    encodings = ["utf-8-sig", "utf-8", "latin-1"]

    for encoding in encodings:
        try:
            with csv_path.open(
                "r",
                encoding=encoding,
                newline=""
            ) as file:
                reader = csv.reader(file)
                return next(reader, [])
        except UnicodeDecodeError:
            continue
        except Exception as error:
            return [f"[Unable to read header: {error}]"]

    return ["[Unable to determine file encoding]"]


# ------------------------------------------------------------
# 3. Create the folder-content report
# ------------------------------------------------------------

directories = sorted(
    [ROOT] + [
        path
        for path in ROOT.rglob("*")
        if path.is_dir()
    ],
    key=lambda path: str(path.relative_to(ROOT)).lower()
)

with REPORT_PATH.open("w", encoding="utf-8") as report:
    report.write("FOLDER AND CSV VARIABLE STRUCTURE\n")
    report.write("=" * 80 + "\n")
    report.write(f"Root directory: {ROOT}\n\n")

    for directory in directories:
        relative_directory = directory.relative_to(ROOT)

        if directory == ROOT:
            folder_label = ".  [ROOT DIRECTORY]"
        else:
            folder_label = str(relative_directory)

        report.write(f"FOLDER: {folder_label}\n")
        report.write("-" * 80 + "\n")

        csv_files = sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".csv"
            ],
            key=lambda path: path.name.lower()
        )

        if not csv_files:
            report.write("  No CSV files in this folder.\n\n")
            continue

        for csv_path in csv_files:
            report.write(f"  CSV: {csv_path.name}\n")
            report.write("    Variables:\n")

            variables = read_csv_header(csv_path)

            if variables:
                for variable in variables:
                    report.write(f"      - {variable}\n")
            else:
                report.write("      - [No variables found: empty file]\n")

            report.write("\n")

    report.write("=" * 80 + "\n")
    report.write("CONVERSION SUMMARY\n")
    report.write("=" * 80 + "\n")
    report.write(f"Successfully converted: {len(converted_files)}\n")
    report.write(f"Conversion errors: {len(conversion_errors)}\n")

    if conversion_errors:
        report.write("\nFILES NOT CONVERTED OR DELETED:\n")

        for path, error in conversion_errors:
            report.write(f"  - {path}: {error}\n")


print()
print(f"Successfully converted and deleted {len(converted_files)} .dta file(s).")
print(f"Conversion errors: {len(conversion_errors)}")
print(f"Report written to: {REPORT_PATH}")