# run_all.ps1 -- reproduce every table, figure, model estimate, and counterfactual of the paper (Windows PowerShell).
# Usage (from the replication_package folder):   powershell -ExecutionPolicy Bypass -File run_all.ps1 [-Rebuild]
#   -Rebuild  also rebuilds data/intermediate/build_rebuilt from the raw sales workbooks (calls the Census geocoder
#             over the internet; geocoder results can change over time, so the archived build files are the default).
param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "output\logs\run_all.log"
"run_all started $(Get-Date -Format s)" | Out-File $log -Encoding utf8
$py = "python"; if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py" }
function Run($dir, $script) {
  $t0 = Get-Date
  Push-Location (Join-Path $root "code\$dir")
  cmd /c "$py -I $script >> `"$log`" 2>&1"
  $code = $LASTEXITCODE
  Pop-Location
  $msg = "{0,-48} exit={1} {2,5}s" -f "$dir/$script", $code, [int]((Get-Date) - $t0).TotalSeconds
  $msg | Tee-Object -FilePath $log -Append
  if ($code -ne 0) { throw "step failed: $dir/$script (see output\logs\run_all.log)" }
}
Run "00_setup" "s00_check_env.py"
if ($Rebuild) {
  Run "01_import_clean" "01_ingest_geocode.py"
  Run "01_import_clean" "02_brand_crosswalk.py"
  Run "01_import_clean" "03_build_markets.py"
}
Run "02_construct_variables" "c01_travel_times_tribal.py"      # uses the cached OSRM matrix if present
Run "04_main_analysis" "rev00_build_cache.py"
Run "03_descriptive" "01_build_facts123.py"
Run "03_descriptive" "02_data_section_tables.py"
Run "03_descriptive" "03_fix_figures.py"
Run "03_descriptive" "04_entry_by_type.py"
foreach ($s in "rev02_identification_robustness.py","rev02b_controls_diagnostics.py","rev03_decomposition_incumbents.py",
               "rev04_org_dynamics.py","rev05_placebo.py","rev06_structural_diagnostics.py","rev07_med_incumbent_trajectories.py",
               "rev08_preparedness_by_type.py","rev09_chain_minus_indep.py","rio01_revenue_streams_exposure.py",
               "rio03_continuing_stores.py","rio05_quarterly_tables.py","rio06_composition_extras.py") { Run "04_main_analysis" $s }
Run "05_robustness" "rio02_travel_time_tribal.py"
Run "06_entry_model" "m00_model_data.py"
Run "06_entry_model" "m01_estimate_validate.py"
Run "07_counterfactuals" "m02_counterfactuals.py"
Run "08_figures_tables" "t01_main_text_tables.py"
Run "08_figures_tables" "v01_compare_tables.py"
Run "08_figures_tables" "v02_manifest.py"
"run_all finished $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
