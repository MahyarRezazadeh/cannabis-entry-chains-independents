#!/usr/bin/env bash
# run_all.sh -- reproduce every table, figure, model estimate, and counterfactual of the paper (macOS / Linux).
# Usage (from the replication_package folder):  bash run_all.sh [--rebuild]
#   --rebuild  also rebuilds data/intermediate/build_rebuilt from the raw sales workbooks (calls the Census geocoder).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$ROOT/output/logs/run_all.log"
PY="${PYTHON:-python3}"
echo "run_all started $(date -u +%FT%T)" > "$LOG"
run () { local t0=$(date +%s); ( cd "$ROOT/code/$1" && "$PY" -I "$2" >> "$LOG" 2>&1 ); \
         printf '%-48s exit=0 %5ss\n' "$1/$2" "$(( $(date +%s) - t0 ))" | tee -a "$LOG"; }
run 00_setup s00_check_env.py
if [[ "${1:-}" == "--rebuild" ]]; then
  run 01_import_clean 01_ingest_geocode.py; run 01_import_clean 02_brand_crosswalk.py; run 01_import_clean 03_build_markets.py
fi
run 02_construct_variables c01_travel_times_tribal.py
run 04_main_analysis rev00_build_cache.py
for s in 01_build_facts123.py 02_data_section_tables.py 03_fix_figures.py 04_entry_by_type.py; do run 03_descriptive "$s"; done
for s in rev02_identification_robustness.py rev02b_controls_diagnostics.py rev03_decomposition_incumbents.py \
         rev04_org_dynamics.py rev05_placebo.py rev06_structural_diagnostics.py rev07_med_incumbent_trajectories.py \
         rev08_preparedness_by_type.py rev09_chain_minus_indep.py rio01_revenue_streams_exposure.py \
         rio03_continuing_stores.py rio05_quarterly_tables.py rio06_composition_extras.py; do run 04_main_analysis "$s"; done
run 05_robustness rio02_travel_time_tribal.py
run 06_entry_model m00_model_data.py
run 06_entry_model m01_estimate_validate.py
run 07_counterfactuals m02_counterfactuals.py
run 08_figures_tables t01_main_text_tables.py
run 08_figures_tables v01_compare_tables.py
run 08_figures_tables v02_manifest.py
echo "run_all finished $(date -u +%FT%T)" | tee -a "$LOG"
