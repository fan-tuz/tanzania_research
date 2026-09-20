SHELL := /bin/bash
.DELETE_ON_ERROR:

# =============================================================================
# Tanzania NPS Maasai / game-reserve pipeline
#
# Pipeline:
#   1. extract_relevant_nps_variables.py
#   2. georeferencing.py
#   3. run_maasai_main_specifications.py
#   4. run_maasai_wave_event_study.py
#
# All Python scripts are expected in scripts/github.
# =============================================================================

# -----------------------------------------------------------------------------
# Commands and scripts
# Override PYTHON when using a virtual or Conda environment, for example:
#   make all PYTHON="conda run -n masai_env python"
# -----------------------------------------------------------------------------
PYTHON          ?= python3
SCRIPTS_DIR     ?= scripts/github

EXTRACT_SCRIPT  ?= $(SCRIPTS_DIR)/extract_relevant_nps_variables.py
GEO_SCRIPT      ?= $(SCRIPTS_DIR)/georeferencing.py
REGRESSION_SCRIPT ?= $(SCRIPTS_DIR)/run_maasai_main_specifications.py
EVENT_STUDY_SCRIPT ?= $(SCRIPTS_DIR)/run_maasai_wave_event_study.py
ROBUSTNESS_SCRIPT ?= $(SCRIPTS_DIR)/run_maasai_robustness_checks.py
NDVI_SCRIPT ?= $(SCRIPTS_DIR)/ndvi_tanzania_waves_compatible.py
NDVI_ROBUSTNESS_SCRIPT ?= $(SCRIPTS_DIR)/run_maasai_ndvi_robustness.py

# -----------------------------------------------------------------------------
# Input data
# -----------------------------------------------------------------------------
DATA_ROOT        ?= data/github
LEGACY_DATA_ROOT ?= data
GAME_RESERVES    ?= data/geo/Tanzania_Game_Reserves.shp
MAASAI_DATA_DIR  ?= data/github/maasai_parakuyo_datoga_spatial_dataset_expanded
WAVE5_PANEL_KEY ?= data/wave5/npsy5.panel.key.csv

MAASAI_WARDS     ?= $(MAASAI_DATA_DIR)/community_wards_expanded.shp
MAASAI_POINTS    ?= $(MAASAI_DATA_DIR)/community_points_expanded.shp

# -----------------------------------------------------------------------------
# Output directories and deterministic filenames
# -----------------------------------------------------------------------------
EXTRACT_OUTPUT_DIR    ?= output/nps_relevant
FINAL_OUTPUT_DIR      ?= output/nps_final
REGRESSION_OUTPUT_DIR ?= output/regressions/main_specifications
EVENT_STUDY_OUTPUT_DIR ?= output/regressions/wave_event_study
ROBUSTNESS_OUTPUT_DIR ?= output/regressions/robustness_checks
NDVI_OUTPUT_DIR ?= data/ndvi
NDVI_ROBUSTNESS_OUTPUT_DIR ?= output/regressions/ndvi_robustness

HOUSEHOLD_FILE ?= nps_relevant_household_wave.csv
SALES_FILE     ?= nps_livestock_sales_long.csv

EXTRACT_HOUSEHOLD := $(EXTRACT_OUTPUT_DIR)/$(HOUSEHOLD_FILE)
EXTRACT_SALES     := $(EXTRACT_OUTPUT_DIR)/$(SALES_FILE)
FINAL_HOUSEHOLD   := $(FINAL_OUTPUT_DIR)/$(HOUSEHOLD_FILE)
FINAL_SALES       := $(FINAL_OUTPUT_DIR)/$(SALES_FILE)
FINAL_TABLE       := $(REGRESSION_OUTPUT_DIR)/table_1_welfare.tex
FINAL_TABLE_RAW   := $(REGRESSION_OUTPUT_DIR)/table_main_specifications_raw.csv
EVENT_TABLE       := $(EVENT_STUDY_OUTPUT_DIR)/table_1_wave_event_study_welfare.tex
EVENT_TABLE_RAW   := $(EVENT_STUDY_OUTPUT_DIR)/table_wave_event_study_raw.csv
ROBUSTNESS_TABLE  := $(ROBUSTNESS_OUTPUT_DIR)/table_robustness_checks_main.tex
ROBUSTNESS_RAW    := $(ROBUSTNESS_OUTPUT_DIR)/robustness_checks_combined_raw.csv
NDVI_EA_WAVE      := $(NDVI_OUTPUT_DIR)/ndvi_ea_wave_compatible.csv
NDVI_ROBUSTNESS_TABLE := $(NDVI_ROBUSTNESS_OUTPUT_DIR)/table_ndvi_control_robustness.tex
NDVI_ROBUSTNESS_RAW   := $(NDVI_ROBUSTNESS_OUTPUT_DIR)/table_ndvi_control_robustness_raw.csv

# -----------------------------------------------------------------------------
# Extractor settings
# -----------------------------------------------------------------------------
MIN_EA_PRICE_OBSERVATIONS ?= 2
DISTRESS_PRICE_RATIO ?= 0.80
CATTLE_CODES   ?= 1,2,3,4,5,6

# -----------------------------------------------------------------------------
# Georeferencing settings
# -----------------------------------------------------------------------------
MIN_CONFIDENCE ?= 0.80
MAASAI_GROUPS  ?= maasai parakuyo datoga_barabaig
PROJECTED_CRS  ?= EPSG:32736

# -----------------------------------------------------------------------------
# Regression settings
# -----------------------------------------------------------------------------
TREATMENT_COLUMN     ?= is_maasai_25km
MAIN_BOUNDARY_KM     ?= 75
MIN_N                ?= 100
MIN_CLUSTERS         ?= 20
WILD_BOOTSTRAP_REPS  ?= 999
WILD_BOOTSTRAP_SEED  ?= 20260625
WILD_BOOTSTRAP_TYPE  ?= 31
REFERENCE_WAVE       ?= 3
GEE_PROJECT          ?= python2103
NDVI_SCALE           ?= 1000
INCLUDE_W5           ?= 1

ifeq ($(INCLUDE_W5),0)
EXTRACT_W5_OPTION := --exclude-wave5
WAVE5_OPTION := --exclude-wave5
else
EXTRACT_W5_OPTION := --include-wave5 --wave5-panel-key "$(WAVE5_PANEL_KEY)"
WAVE5_OPTION :=
endif

.PHONY: all extract georeference ndvi regressions event-study robustness ndvi-robustness check-scripts check-inputs clean help

# Run the complete pipeline (without figures).
all: event-study robustness ndvi-robustness
	@echo "Pipeline completed successfully."
	@echo "Main paper table: $(FINAL_TABLE)"
	@echo "Event-study paper table: $(EVENT_TABLE)"

# Check that scripts and fixed external inputs exist before running.
check-scripts:
	@test -f "$(EXTRACT_SCRIPT)" || { echo "ERROR: missing $(EXTRACT_SCRIPT)"; exit 1; }
	@test -f "$(GEO_SCRIPT)" || { echo "ERROR: missing $(GEO_SCRIPT)"; exit 1; }
	@test -f "$(REGRESSION_SCRIPT)" || { echo "ERROR: missing $(REGRESSION_SCRIPT)"; exit 1; }
	@test -f "$(EVENT_STUDY_SCRIPT)" || { echo "ERROR: missing $(EVENT_STUDY_SCRIPT)"; exit 1; }
	@test -f "$(ROBUSTNESS_SCRIPT)" || { echo "ERROR: missing $(ROBUSTNESS_SCRIPT)"; exit 1; }
	@test -f "$(NDVI_SCRIPT)" || { echo "ERROR: missing $(NDVI_SCRIPT)"; exit 1; }
	@test -f "$(NDVI_ROBUSTNESS_SCRIPT)" || { echo "ERROR: missing $(NDVI_ROBUSTNESS_SCRIPT)"; exit 1; }

check-inputs: check-scripts
	@test -d "$(DATA_ROOT)" || { echo "ERROR: missing data directory $(DATA_ROOT)"; exit 1; }
	@test -d "$(LEGACY_DATA_ROOT)" || { echo "ERROR: missing legacy data directory $(LEGACY_DATA_ROOT)"; exit 1; }
	@test -f "$(GAME_RESERVES)" || { echo "ERROR: missing game-reserve file $(GAME_RESERVES)"; exit 1; }
	@test -f "$(MAASAI_WARDS)" || { echo "ERROR: missing Maasai ward file $(MAASAI_WARDS)"; exit 1; }
	@test -f "$(MAASAI_POINTS)" || { echo "ERROR: missing Maasai point file $(MAASAI_POINTS)"; exit 1; }
	@if [ "$(INCLUDE_W5)" = "1" ]; \
	then test -f "$(WAVE5_PANEL_KEY)" || { echo "ERROR: missing W5 panel key $(WAVE5_PANEL_KEY)"; exit 1; }; \
	fi

# 1. Extract household-wave and livestock-sale datasets.
extract: check-inputs
	@mkdir -p "$(EXTRACT_OUTPUT_DIR)"
	$(PYTHON) "$(EXTRACT_SCRIPT)" \
		--data-root "$(DATA_ROOT)" \
		--legacy-data-root "$(LEGACY_DATA_ROOT)" \
		--output-dir "$(EXTRACT_OUTPUT_DIR)" \
		$(EXTRACT_W5_OPTION) \
		--cattle-codes "$(CATTLE_CODES)" \
		--min-ea-price-observations "$(MIN_EA_PRICE_OBSERVATIONS)" \
		--distress-price-ratio "$(DISTRESS_PRICE_RATIO)"
	@test -f "$(EXTRACT_HOUSEHOLD)" || { echo "ERROR: extractor did not create $(EXTRACT_HOUSEHOLD)"; exit 1; }
	@test -f "$(EXTRACT_SALES)" || { echo "ERROR: extractor did not create $(EXTRACT_SALES)"; exit 1; }

# 2. Add reserve exposure and locality-based treatment variables.
georeference: extract
	@mkdir -p "$(FINAL_OUTPUT_DIR)"
	$(PYTHON) "$(GEO_SCRIPT)" \
		--input-dir "$(EXTRACT_OUTPUT_DIR)" \
		--output-dir "$(FINAL_OUTPUT_DIR)" \
		--household-file "$(HOUSEHOLD_FILE)" \
		--sales-file "$(SALES_FILE)" \
		--game-reserves "$(GAME_RESERVES)" \
		--maasai-data-dir "$(MAASAI_DATA_DIR)" \
		--min-confidence "$(MIN_CONFIDENCE)" \
		--groups $(MAASAI_GROUPS) \
		--projected-crs "$(PROJECTED_CRS)"
	@test -f "$(FINAL_HOUSEHOLD)" || { echo "ERROR: georeferencing did not create $(FINAL_HOUSEHOLD)"; exit 1; }
	@test -f "$(FINAL_SALES)" || { echo "ERROR: georeferencing did not create $(FINAL_SALES)"; exit 1; }

# 3a. Create EA-wave NDVI control used for the NDVI robustness check.
ndvi: georeference
	@mkdir -p "$(NDVI_OUTPUT_DIR)"
	$(PYTHON) "$(NDVI_SCRIPT)" \
		--mode sample-ee \
		--nps-household "$(FINAL_HOUSEHOLD)" \
		--output-dir "$(NDVI_OUTPUT_DIR)" \
		--gee-project "$(GEE_PROJECT)" \
		--scale "$(NDVI_SCALE)"
	@test -f "$(NDVI_EA_WAVE)" || { echo "ERROR: NDVI script did not create $(NDVI_EA_WAVE)"; exit 1; }
	@echo "NDVI control completed: $(NDVI_EA_WAVE)"

# 3. Run the deterministic main and core pressure specifications.
regressions: georeference
	@mkdir -p "$(REGRESSION_OUTPUT_DIR)"
	$(PYTHON) "$(REGRESSION_SCRIPT)" \
		--input-dir "$(FINAL_OUTPUT_DIR)" \
		--output-dir "$(REGRESSION_OUTPUT_DIR)" \
		--household-file "$(HOUSEHOLD_FILE)" \
		--treatment-column "$(TREATMENT_COLUMN)" \
		--main-boundary-km "$(MAIN_BOUNDARY_KM)" \
		$(WAVE5_OPTION) --min-n "$(MIN_N)" \
		--min-clusters "$(MIN_CLUSTERS)" \
		--wild-bootstrap-reps "$(WILD_BOOTSTRAP_REPS)" \
		--wild-bootstrap-seed "$(WILD_BOOTSTRAP_SEED)" \
		--wild-bootstrap-type "$(WILD_BOOTSTRAP_TYPE)"
	@test -f "$(FINAL_TABLE)" || { echo "ERROR: regressions did not create $(FINAL_TABLE)"; exit 1; }
	@test -f "$(FINAL_TABLE_RAW)" || { echo "ERROR: regressions did not create $(FINAL_TABLE_RAW)"; exit 1; }
	@echo "Main specification completed. Paper table: $(FINAL_TABLE)"

# 4. Run wave-by-wave event-study specifications.
event-study: regressions
	@mkdir -p "$(EVENT_STUDY_OUTPUT_DIR)"
	$(PYTHON) "$(EVENT_STUDY_SCRIPT)" \
		--input-dir "$(FINAL_OUTPUT_DIR)" \
		--output-dir "$(EVENT_STUDY_OUTPUT_DIR)" \
		--household-file "$(HOUSEHOLD_FILE)" \
		--treatment-column "$(TREATMENT_COLUMN)" \
		--reference-wave "$(REFERENCE_WAVE)" \
		--main-boundary-km "$(MAIN_BOUNDARY_KM)" \
		$(WAVE5_OPTION) --min-n "$(MIN_N)" \
		--min-clusters "$(MIN_CLUSTERS)" \
		--wild-bootstrap-reps "$(WILD_BOOTSTRAP_REPS)" \
		--wild-bootstrap-seed "$(WILD_BOOTSTRAP_SEED)" \
		--wild-bootstrap-type "$(WILD_BOOTSTRAP_TYPE)"
	@test -f "$(EVENT_TABLE)" || { echo "ERROR: event study did not create $(EVENT_TABLE)"; exit 1; }
	@test -f "$(EVENT_TABLE_RAW)" || { echo "ERROR: event study did not create $(EVENT_TABLE_RAW)"; exit 1; }
	@echo "Event study completed. Paper table: $(EVENT_TABLE)"

# 5a. Run robustness checks: 50 km risk set and narrower 10 km treatment.
robustness: regressions
	@mkdir -p "$(ROBUSTNESS_OUTPUT_DIR)"
	$(PYTHON) "$(ROBUSTNESS_SCRIPT)" \
		--input-dir "$(FINAL_OUTPUT_DIR)" \
		--output-dir "$(ROBUSTNESS_OUTPUT_DIR)" \
		--main-script "$(REGRESSION_SCRIPT)" \
		--household-file "$(HOUSEHOLD_FILE)" \
		$(WAVE5_OPTION) --min-n "$(MIN_N)" \
		--min-clusters "$(MIN_CLUSTERS)" \
		--wild-bootstrap-reps "$(WILD_BOOTSTRAP_REPS)" \
		--wild-bootstrap-seed "$(WILD_BOOTSTRAP_SEED)" \
		--wild-bootstrap-type "$(WILD_BOOTSTRAP_TYPE)"
	@test -f "$(ROBUSTNESS_TABLE)" || { echo "ERROR: robustness did not create $(ROBUSTNESS_TABLE)"; exit 1; }
	@test -f "$(ROBUSTNESS_RAW)" || { echo "ERROR: robustness did not create $(ROBUSTNESS_RAW)"; exit 1; }
	@echo "Robustness checks completed. Paper table: $(ROBUSTNESS_TABLE)"

# 5b. Run NDVI-control robustness check using the main 25 km treatment.
ndvi-robustness: ndvi regressions
	@mkdir -p "$(NDVI_ROBUSTNESS_OUTPUT_DIR)"
	$(PYTHON) "$(NDVI_ROBUSTNESS_SCRIPT)" \
		--input-dir "$(FINAL_OUTPUT_DIR)" \
		--ndvi-csv "$(NDVI_EA_WAVE)" \
		--output-dir "$(NDVI_ROBUSTNESS_OUTPUT_DIR)" \
		--household-file "$(HOUSEHOLD_FILE)" \
		--treatment-column "$(TREATMENT_COLUMN)" \
		--main-boundary-km "$(MAIN_BOUNDARY_KM)" \
		$(WAVE5_OPTION) --min-n "$(MIN_N)" \
		--min-clusters "$(MIN_CLUSTERS)" \
		--wild-bootstrap-reps "$(WILD_BOOTSTRAP_REPS)" \
		--wild-bootstrap-seed "$(WILD_BOOTSTRAP_SEED)" \
		--wild-bootstrap-type "$(WILD_BOOTSTRAP_TYPE)"
	@test -f "$(NDVI_ROBUSTNESS_TABLE)" || { echo "ERROR: NDVI robustness did not create $(NDVI_ROBUSTNESS_TABLE)"; exit 1; }
	@test -f "$(NDVI_ROBUSTNESS_RAW)" || { echo "ERROR: NDVI robustness did not create $(NDVI_ROBUSTNESS_RAW)"; exit 1; }
	@echo "NDVI robustness completed. Paper table: $(NDVI_ROBUSTNESS_TABLE)"

# Remove generated outputs only.
clean:
	rm -rf "$(EXTRACT_OUTPUT_DIR)" "$(FINAL_OUTPUT_DIR)" "$(REGRESSION_OUTPUT_DIR)" "$(EVENT_STUDY_OUTPUT_DIR)" "$(ROBUSTNESS_OUTPUT_DIR)" "$(NDVI_ROBUSTNESS_OUTPUT_DIR)" "$(NDVI_OUTPUT_DIR)"

help:
	@echo "Targets:"
	@echo "  make all             Run extraction, georeferencing, main/event, robustness, and NDVI"
	@echo "  INCLUDE_W5=0         Drop Wave 5 from regressions and event study"
	@echo "  TREATMENT_COLUMN=is_maasai_25km   Main treatment default"
	@echo "  WILD_BOOTSTRAP_REPS=9999          Final inference run"
	@echo "  make extract         Run only the extractor"
	@echo "  make georeference    Run extraction and georeferencing"
	@echo "  make regressions     Run extraction, georeferencing, and main regressions"
	@echo "  make event-study     Run the complete pipeline through event study"
	@echo "  make robustness      Run 50 km and 10 km-treatment robustness checks"
	@echo "  make ndvi-robustness Run NDVI-control robustness check"
	@echo "  make clean           Remove generated pipeline outputs"
	@echo ""
	@echo "Common overrides:"
	@echo '  make all PYTHON="conda run -n masai_env python"'
	@echo "  make all WILD_BOOTSTRAP_REPS=9999"
	@echo "  make all MIN_EA_PRICE_OBSERVATIONS=3"
	@echo "  make all TREATMENT_COLUMN=is_maasai_10km  # optional high-intensity run"