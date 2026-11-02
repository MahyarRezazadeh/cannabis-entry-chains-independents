# Replication package

**More Stores or Bigger Stores? Cross-Border Demand and Retail Entry in New Mexico's Cannabis Market**
Mahyar Rezazadeh, Simon Fraser University. Submitted to *Review of Industrial Organization*.

This package reproduces every table, figure, model estimate, and counterfactual in the paper and in Online Resource 1,
except a small set of exhibits that are reproduced from saved outputs (listed below).

## Quick start

```
pip install -r environment/requirements.txt          # Python 3.13; versions pinned
powershell -ExecutionPolicy Bypass -File run_all.ps1   # Windows
bash run_all.sh                                        # macOS / Linux
```

`run_all` runs every step in order, logs to `output/logs/run_all.log`, and stops at the first failure. On the
author's machine (Windows 11, Python 3.13.12, 8 GB RAM) the full run takes about 6 minutes; the counterfactual
simulations (300 parameter draws) take about 2 minutes. No step needs more than 2 GB of memory.

Optional flag `-Rebuild` / `--rebuild` also rebuilds the storefront, operator, and market files from the raw monthly
sales workbooks (`code/01_import_clean`). That step calls the U.S. Census Bureau batch geocoder over the internet,
whose matches can change over time, so the default run uses the archived build files in `data/intermediate/build/`
produced by the same scripts. Rebuilt files are written to `data/intermediate/build_rebuilt/` and never overwrite the
archived ones.

**Path length on Windows.** The longest path inside the package is 71 characters relative to the package root (314 files, 154 MB).
Unzip the package close to a drive root (e.g. `C:\repl\`) to stay below the 260-character Windows path limit.

## Structure

| Folder | Contents |
|---|---|
| `data/raw/nm_crop/` | 51 monthly "Sales by Licensee" workbooks, April 2022 -- July 2026 (NM Cannabis Control Division) |
| `data/raw/ref/` | SimpleMaps US ZIP database (Basic), Census state boundaries and AIANNH (tribal) boundaries, legal-date table |
| `data/raw/zbp/` | ZIP Codes Business Patterns extract for NM ZIP codes, 2018--2023 |
| `data/raw/census_blocks/` | empty; 2020 Census blocks for NM are only needed to rebuild `sf_local_population.csv` (see DATA_AVAILABILITY.md) |
| `data/intermediate/build/` | storefronts, monthly revenue, operator crosswalk, market panels (outputs of `code/01_import_clean`) |
| `data/intermediate/` | OSRM travel-time matrix (cached), tribal/travel-time ZIP file, store local populations, saved static-game outputs |
| `data/analysis/` | cached analysis panels and logs written by the analysis scripts |
| `code/lib/` | `paths.py` (all locations), `_common.py` (data builders), `_econ.py` (estimators) |
| `code/00_setup/` | environment check |
| `code/01_import_clean/` | raw workbooks -> storefronts, geocodes, operators, market panels (optional rebuild) |
| `code/02_construct_variables/` | road travel times (OSRM) and tribal-land indicator |
| `code/03_descriptive/` | summary statistics, maps, store-level figure, entry flows, appendix tables |
| `code/04_main_analysis/` | market-level gaps, decomposition, identification checks, organizational dynamics, new RIO analyses |
| `code/05_robustness/` | travel-time exposure and tribal-land robustness |
| `code/06_entry_model/` | entry-and-adjustment model: data, estimation, validation |
| `code/07_counterfactuals/` | counterfactual simulations with parameter uncertainty |
| `code/08_figures_tables/` | main-text summary tables, comparison with published tables, manifests |
| `code/legacy/` | archived scripts for the static entry game (`game/`) and Colorado/extension analyses (`ext/`); **not re-run** (they contain the original absolute paths) |
| `output/tables`, `output/figures` | everything the paper uses; `published/` subfolders hold the versions in the previous draft |
| `output/model_diagnostics`, `output/logs` | model logs, run log, table comparison, result manifest |

## Exhibit-to-code map

`output/logs/result_manifest.csv` lists every table and figure in the paper and Online Resource 1 with the producing
script. `manifest.csv` lists every file with its SHA-256 hash, role, and redistribution status.

Exhibits **reproduced from saved outputs, not re-estimated** in this package:
* Online Resource Tables OR17 (town-level static entry game; regenerated from the game's saved CSV/JSON outputs),
  OR18 (ZIP-level static game), OR19 (own-network Poisson regression), OR20--OR22 (Colorado comparison),
  and Figures OR4 (diffusion-fit figure; its point estimates are reproduced by `rev04`), OR5, OR6.
* Online Resource Table OR1 (sample definitions) is hand-assembled from counts printed by the scripts.
The archived source code for these exhibits is in `code/legacy/`.

## Random seeds

All bootstrap and simulation steps use fixed seeds: 20261023 (new analyses, model, counterfactuals),
11 (wild cluster bootstrap), and the seeds recorded at the top of each `rev*` script.

## Verification

`output/logs/table_comparison.csv` compares every regenerated table with the version in the previous draft:
25 tables are identical number for number; 12 tables are new. See `verification_report.md`.
