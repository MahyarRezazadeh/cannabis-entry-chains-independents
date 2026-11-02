r"""
rio02_travel_time_tribal.py -- exposure measured by road-network travel time; tribal-area robustness.
Uses data/intermediate/zip_travel_tribal.csv (code/02_construct_variables/c01_travel_times_tribal.py).
Outputs: output/tables/tab_rio_travel_tribal.tex, output/logs/rio02.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os
import numpy as np
import pandas as pd
from paths import INTER, LOGS
from _common import load_zx, load_storefronts, load_store_quarters, classify, build_zip_covariates, zip_panel, stars, ROOT
from _econ import design, ols_fit, fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

zx = load_zx(); sf, bm = load_storefronts(zx); sq, sf = load_store_quarters(sf)
meta = classify(sq, bm); cov = build_zip_covariates(zx, sf)
tt = pd.read_csv(INTER / "zip_travel_tribal.csv", dtype={"zip5": str})
cov = cov.merge(tt, on="zip5", how="left")
for m in (60, 90, 120):
    cov[f"x_tt{m}"] = np.log1p(cov[f"tt_pop_{m}"] / 1000)
cov["x_ttdecay"] = np.log1p(cov.tt_decay_60 / 1000)
cov["x_r75"] = np.log1p(cov.nlpop_75 / 1000)
cov["zbp_r"] = np.arcsinh(cov.zbp19_total_retail / zx.set_index("zip5").zip_pop.reindex(cov.zip5).to_numpy() * 1000)
cov["dist_abq_100"], cov["dist_bigcity_100"] = cov.dist_abq / 100, cov.dist_bigcity / 100
GEO = ("log_density", "dist_abq_100", "dist_bigcity_100", "zbp_r")
main = zip_panel(zx, sq, meta, cov)
allz = zip_panel(zx, sq, meta, cov, ("nonlegal", "legal", "interior"))

def cont(dd, x, y, controls=()):
    X, nm = design(dd, dynamic=False, controls=controls, treat=x)
    r = ols_fit(dd[y].to_numpy(), X, dd.county_fips.to_numpy())
    return r["b"][0], r["se"][0], pval(r["b"][0], r["se"][0], r["G"])

p(f"corr(log1p straight-line pop 75mi, log1p road pop 90min) across 368 ZIPs: "
  f"{np.corrcoef(cov.x_r75, cov.x_tt90)[0,1]:.3f}")
rows = []
for x, lab in [("x_r75", "Straight-line: residents within 75 miles (baseline)"), ("x_tt60", "Road: residents within 60 minutes"),
               ("x_tt90", "Road: residents within 90 minutes"), ("x_tt120", "Road: residents within 120 minutes"),
               ("x_ttdecay", r"Road: travel-time decay, $e^{-t/60}$")]:
    cells = []
    for dd, ctr in [(main, ()), (allz, GEO), (main[main.nonlegal == 1], GEO)]:
        for y in ("y_rpc", "y_den"):
            cells.append(cont(dd, x, y, ctr))
        cells.append(cont(dd[dd.n_stores > 0], x, "y_rps", ctr))
    rows.append((lab, cells))
    p(f"{lab:55s} " + " | ".join(f"{b:+.3f}({s:.3f}){stars(pv)}" for b, s, pv in cells))

# tribal-area robustness for the binary comparison
trib = set(tt.zip5[tt.tribal_centroid == 1])
mt = main[~main.zip5.isin(trib)]
tr_rows = []
for lab, dd, ctr in [("Baseline", main, ()), ("Drop ZIPs with centroid on tribal land", mt, ()),
                     ("Control for tribal-land centroid", main, ("tribal_centroid",))]:
    a = fit(dd, "y_rpc", dynamic=False, controls=ctr); b = fit(dd, "y_den", dynamic=False, controls=ctr)
    c = fit(dd[dd.n_stores > 0], "y_rps", dynamic=False, controls=ctr)
    tr_rows.append((lab, [(a["gap_b"][0], a["gap_se"][0], a["gap_p"][0]), (b["gap_b"][0], b["gap_se"][0], b["gap_p"][0]),
                          (c["gap_b"][0], c["gap_se"][0], c["gap_p"][0])], dd.zip5.nunique()))
    p(f"{lab:42s} ZIPs {dd.zip5.nunique()} | " + " | ".join(f"{x[0]:+.3f}({x[1]:.3f}){stars(x[2])}" for x in tr_rows[-1][1]))

def c(x):
    return f"${x[0]:+.3f}^{{{stars(x[2])}}}$".replace("^{}", ""), f"({x[1]:.3f})"
L = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{209 ZIPs, no controls} & \multicolumn{3}{c}{368 ZIPs, controls} & \multicolumn{3}{c}{92 nonlegal ZIPs, controls} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
     r"Exposure measure & Rev./res. & Density & Rev./store & Rev./res. & Density & Rev./store & Rev./res. & Density & Rev./store \\", r"\midrule"]
for lab, cells in rows:
    cc = [c(x) for x in cells]
    L += [lab + " & " + " & ".join(a for a, _ in cc) + r" \\", " & " + " & ".join(b for _, b in cc) + r" \\"]
L += [r"\midrule", r"\multicolumn{10}{@{}l}{\textit{Binary comparison and tribal lands}} \\"]
for lab, cells, n in tr_rows:
    cc = [c(x) for x in cells]
    L += [f"{lab} ({n} ZIPs) & " + " & ".join(a for a, _ in cc) + r" & & & & & & \\", " & " + " & ".join(b for _, b in cc) + r" & & & & & & \\"]
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_travel_tribal.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
open(LOGS / "rio02.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
