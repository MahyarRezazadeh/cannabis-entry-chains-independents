"""Project-relative paths for the replication package.

Every script imports its locations from here, so the package runs from any directory once it is
copied as a whole. The root can be overridden with the environment variable CANNABIS_REPL_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CANNABIS_REPL_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
CROP = DATA_RAW / "nm_crop"                      # NM Cannabis Control Division monthly sales workbooks
REF = DATA_RAW / "ref"                           # SimpleMaps ZIP database, Census state boundaries
ZBP = DATA_RAW / "zbp"                           # ZIP Codes Business Patterns extract (NM ZIPs, 2018-2023)
BLOCKS = DATA_RAW / "census_blocks"              # 2020 Census block shapefile for NM (downloaded separately)
BUILD = ROOT / "data" / "intermediate" / "build"  # outputs of code/01_import_clean
INTER = ROOT / "data" / "intermediate"
ANALYSIS = ROOT / "data" / "analysis"            # cached analysis panels
OUT = ROOT / "output"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
DIAG = OUT / "model_diagnostics"
LOGS = OUT / "logs"
for _p in (BUILD, ANALYSIS, TABLES, FIGURES, DIAG, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
