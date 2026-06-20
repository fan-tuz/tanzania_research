PYTHON ?= python
DATA ?= data
OUTPUT ?= output
DIAGNOSTICS ?= diagnostics
SCRIPTS ?= scripts

.PHONY: setup test dirs build subsets spatial restricted baseline pretrend wma diagnostics clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

dirs:
	mkdir -p $(DATA) $(OUTPUT) $(OUTPUT)/regressions $(DIAGNOSTICS)

test:
	PYTHONPATH=$(SCRIPTS) pytest -q

build: dirs
	$(PYTHON) $(SCRIPTS)/build_db_v5.py 		--wave1 $(DATA)/wave1 		--wave2 $(DATA)/wave2 		--wave3 $(DATA)/wave3 		--wave4 $(DATA)/wave4 		--output $(OUTPUT)/refined_multiwave 		--only-main

subsets: dirs
	$(PYTHON) $(SCRIPTS)/create_analysis_subsets_v2.py 		--input $(OUTPUT)/refined_multiwave/nps_analysis_household_wave.csv 		--output $(OUTPUT)/analysis_subsets 		--diagnostics-dir $(DIAGNOSTICS)

spatial: dirs
	$(PYTHON) $(SCRIPTS)/add_ward_and_village_treatment_v5.py 		--subsets-dir $(OUTPUT)/analysis_subsets 		--wards-gpkg $(DATA)/geo/tanzania_2022_wards.gpkg 		--wards-layer wards 		--reference-csv $(DATA)/reference/maasai_treatment_expanded_reference_plus_tier2_83obs.csv 		--output-dir $(OUTPUT)/spatial 		--diagnostics-dir $(DIAGNOSTICS)

restricted: dirs
	$(PYTHON) $(SCRIPTS)/add_restricted_area_exposure.py 		--spatial-dir $(OUTPUT)/spatial 		--game-reserves $(DATA)/geo/Tanzania_Game_Reserves.shp 		--output-dir $(OUTPUT)/spatial_restricted 		--diagnostics-dir $(DIAGNOSTICS)

baseline: dirs
	$(PYTHON) $(SCRIPTS)/run_baseline_analysis_game_reserves_post2009_boundary_v5.py 		--spatial-dir $(OUTPUT)/spatial_restricted 		--output-dir $(OUTPUT)/regressions/baseline_game_reserves

pretrend: dirs
	$(PYTHON) $(SCRIPTS)/run_pretrend_test_game_reserves.py 		--spatial-dir $(OUTPUT)/spatial_restricted 		--output-dir $(OUTPUT)/regressions/pretrend_game_reserves

wma: dirs
	$(PYTHON) $(SCRIPTS)/run_wma_active_inactive_staggered.py 		--spatial-dir $(OUTPUT)/spatial_restricted 		--wma-crosswalk $(DATA)/reference/tanzania_wma_village_year_crosswalk_best_effort.csv 		--output-dir $(OUTPUT)/regressions/wma_active_inactive 		--dataset core

diagnostics: dirs
	jupyter notebook $(DIAGNOSTICS)/validation_diagnostics.ipynb

clean:
	rm -rf $(OUTPUT)/* $(DIAGNOSTICS)/*.csv $(DIAGNOSTICS)/*.png
