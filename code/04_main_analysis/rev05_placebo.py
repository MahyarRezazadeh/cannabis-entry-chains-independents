r"""
rev05_placebo.py -- non-cannabis placebo re-estimated on the paper's main comparison.

Data: Census ZIP Codes Business Patterns establishment counts, 2018-2023 (alcohol retail NAICS 445310,
grocery 4451, total retail 44-45), the same file used by the earlier placebo (Marijuana spollover/New Mexico
project/tables/nm_zip_zbp_raw.csv). Missing ZIP-year-sector cells = 0 establishments.
Differences from the earlier figure: (i) sample = the 209 nonlegal- vs always-legal-border ZIPs (earlier: border
vs all other ZIPs), (ii) county-clustered SEs (earlier: ZIP), (iii) outcome asinh(establishments per 10,000 residents),
(iv) both level gaps by year (as in Table 3) and changes relative to 2021, the last pre-legalization year.
Outputs: tab_rev_placebo.tex, fig_rev_placebo.png, output/rev05_results.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import OUT, ROOT, ZBP, stars
from _econ import ols_fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
zx = A["zx"]
zb = pd.read_csv(os.path.join(ZBP, "nm_zip_zbp_raw.csv"), dtype={"zip5": str})
years = sorted(zb.year.unique()); SECT = ["alcohol", "grocery", "total_retail"]
grid = pd.MultiIndex.from_product([zx.zip5, years, SECT], names=["zip5", "year", "sector"]).to_frame(index=False)
d = grid.merge(zb, on=["zip5", "year", "sector"], how="left").fillna({"ESTAB": 0})
d = d.merge(zx[["zip5", "zip_pop", "group", "county_fips"]], on="zip5")
d = d[d.group != "interior"].copy()
d["nonlegal"] = (d.group == "nonlegal").astype(int)
d["y"] = np.arcsinh(d.ESTAB / d.zip_pop * 10000)
d["log_pop"] = np.log(d.zip_pop)
p(f"ZIP-years with >=1 establishment by sector: {d[d.ESTAB > 0].groupby('sector').size().to_dict()}")
p(f"Placebo panel: {d.zip5.nunique()} ZIPs x {len(years)} years ({years[0]}-{years[-1]}), sectors {SECT}")

RES = {}
for s in SECT:
    ds = d[d.sector == s]
    Y = ds.year.to_numpy()
    # (a) level gaps by year (no omitted year), like Table 3
    X = np.column_stack([ds.nonlegal * (Y == t) for t in years] + [(Y == t) for t in years] + [ds.log_pop]).astype(float)
    r = ols_fit(ds.y.to_numpy(), X, ds.county_fips.to_numpy())
    lv = (r["b"][:len(years)], r["se"][:len(years)], pval(r["b"][:len(years)], r["se"][:len(years)], r["G"]))
    # (b) changes relative to 2021 (genuine pre-period exists for these sectors)
    others = [t for t in years if t != 2021]
    X2 = np.column_stack([ds.nonlegal] + [ds.nonlegal * (Y == t) for t in others] + [(Y == t) for t in years] + [ds.log_pop]).astype(float)
    r2 = ols_fit(ds.y.to_numpy(), X2, ds.county_fips.to_numpy())
    ch = (r2["b"][1:1 + len(others)], r2["se"][1:1 + len(others)], pval(r2["b"][1:1 + len(others)], r2["se"][1:1 + len(others)], r2["G"]))
    # (c) pooled post-minus-pre change: Nonlegal x Post
    ds2 = ds.assign(post=(ds.year >= 2022).astype(int))
    X3 = np.column_stack([ds2.nonlegal, ds2.nonlegal * ds2.post] + [(Y == t) for t in years] + [ds2.log_pop]).astype(float)
    r3 = ols_fit(ds2.y.to_numpy(), X3, ds2.county_fips.to_numpy())
    pp = (r3["b"][1], r3["se"][1], pval(np.array([r3["b"][1]]), np.array([r3["se"][1]]), r3["G"])[0])
    lvl = (r3["b"][0], r3["se"][0], pval(np.array([r3["b"][0]]), np.array([r3["se"][0]]), r3["G"])[0])
    RES[s] = dict(lv=lv, ch=ch, others=others, pp=pp, lvl=lvl, mean_lb=ds[ds.nonlegal == 0].y.mean())
    p(f"\n[{s}] level gap by year: " + "  ".join(f"{t}:{b:+.3f}({se:.3f}){stars(pv)}" for t, b, se, pv in zip(years, *lv)))
    p(f"  change vs 2021: " + "  ".join(f"{t}:{b:+.3f}({se:.3f}){stars(pv)}" for t, b, se, pv in zip(others, *ch)))
    p(f"  pre-period level gap (2018-21) {lvl[0]:+.3f} ({lvl[1]:.3f}) p={lvl[2]:.3f}; Nonlegal x Post(2022-23) {pp[0]:+.3f} ({pp[1]:.3f}) p={pp[2]:.3f}")

lab = {"alcohol": "Alcohol retail (NAICS 445310)", "grocery": "Grocery stores (NAICS 4451)", "total_retail": "Total retail trade (NAICS 44--45)"}
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Alcohol & Grocery & Total \\", r" & retail & stores & retail \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Gap in asinh(establishments per 10,000 residents) by year}} \\"]
for i, t in enumerate(years):
    L.append(f"{t}{' (post)' if t >= 2022 else ''} & " + " & ".join(f"${RES[s]['lv'][0][i]:+.3f}^{{{stars(RES[s]['lv'][2][i])}}}$".replace("^{}", "") for s in SECT) + r" \\")
    L.append(" & " + " & ".join(f"({RES[s]['lv'][1][i]:.3f})" for s in SECT) + r" \\")
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: Change in the gap after legalization}} \\")
L.append("Nonlegal $\\times$ Post (2022--23 vs.\\ 2018--21) & " + " & ".join(f"${RES[s]['pp'][0]:+.3f}^{{{stars(RES[s]['pp'][2])}}}$".replace("^{}", "") for s in SECT) + r" \\")
L.append(" & " + " & ".join(f"({RES[s]['pp'][1]:.3f})" for s in SECT) + r" \\")
L.append(r"\midrule")
L.append("Always-legal-border mean of outcome & " + " & ".join(f"{RES[s]['mean_lb']:.3f}" for s in SECT) + r" \\")
L.append(f"ZIP codes / years & \\multicolumn{{3}}{{c}}{{{d.zip5.nunique()} / {len(years)}}} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_placebo.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(7.2, 3.8))
off = {"alcohol": -0.15, "grocery": 0.0, "total_retail": 0.15}; col = {"alcohol": "#1f3b5c", "grocery": "#b5651d", "total_retail": "#4a7c59"}
for s in SECT:
    b, se, _ = RES[s]["lv"]
    ax.errorbar(np.array(years) + off[s], b, yerr=1.96 * se, fmt="o", ms=4, capsize=2.5, lw=1, color=col[s], label=lab[s])
ax.axhline(0, color="grey", ls="--", lw=0.8); ax.axvline(2021.5, color="grey", ls=":", lw=0.8)
ax.text(2021.55, ax.get_ylim()[1] * 0.85, "adult-use sales begin", fontsize=8, color="grey")
ax.set_ylabel("Nonlegal $-$ always-legal gap"); ax.set_xticks(years); ax.legend(fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rev_placebo.png"), dpi=600); plt.close(fig)
pickle.dump(RES, open(os.path.join(OUT, "rev05.pkl"), "wb"))
open(os.path.join(OUT, "rev05_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
