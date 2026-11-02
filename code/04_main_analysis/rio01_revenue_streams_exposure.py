r"""
rio01_revenue_streams_exposure.py -- new evidence for the RIO revision.

A. Adult-use vs. medical vs. total revenue: pooled asinh gaps, PPML ratios, exact decomposition
   (log R_s/P = log N/P + log R_s/N, county-cluster bootstrap within group, 999 draws, seed 20261023).
B. Quarterly PPML ratios for revenue and store counts (replace the invalid 100(e^b - 1) reading of
   asinh coefficients), plus quarterly level gaps in revenue per resident.
C. Exposure measurement: state-specific exposure (TX / OK / UT residents within 75 mi), alternative
   population thresholds for the binary definition, dropping the ZIPs exposed to both kinds of border,
   east-west (longitude) control, and alternative clustering (town markets).
Outputs: output/tables/tab_rio_streams.tex, tab_rio_quarterly_ppml.tex, tab_rio_exposure_alt.tex,
         output/logs/rio01.txt, data/analysis/rio01.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np
import pandas as pd
from _common import (load_zx, load_storefronts, load_store_quarters, classify, build_zip_covariates, zip_panel,
                     us_zips, hav, QS, QLAB, QMIN, NONLEGAL_STATES, stars, OUT, ROOT)
from _econ import design, ols_fit, fit, ppml, pval, wild_boot

SEED = 20261023
LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

zx = load_zx()
sf, bm = load_storefronts(zx)
sq, sf = load_store_quarters(sf)
sq["adult"] = (sq.revenue - sq.medical).clip(lower=0)
meta = classify(sq, bm)
cov = build_zip_covariates(zx, sf)

def panel(groups=("nonlegal", "legal")):
    d = zip_panel(zx, sq, meta, cov, groups)
    s = sq.groupby(["zip5", "qt"])[["medical", "adult"]].sum().reset_index()
    d = d.merge(s, on=["zip5", "qt"], how="left").fillna({"medical": 0, "adult": 0})
    d["rpc_med"], d["rpc_adult"] = d.medical / d.zip_pop, d.adult / d.zip_pop
    d["y_rpc_med"], d["y_rpc_adult"] = np.arcsinh(d.rpc_med), np.arcsinh(d.rpc_adult)
    d["y_rps_med"] = np.log((d.medical / d.n_stores).where((d.n_stores > 0) & (d.medical > 0)))
    d["y_rps_adult"] = np.log((d.adult / d.n_stores).where((d.n_stores > 0) & (d.adult > 0)))
    zcity = dict(zip(zx.zip5, zx.city))
    d["town"] = d.zip5.map(zcity)
    return d

d = panel()
assert d.zip5.nunique() == 209 and len(d) == 3553
G = d.county_fips.nunique()

def pooled(dd, y, controls=(), cluster="county_fips"):
    r = fit(dd, y, dynamic=False, controls=controls, cluster=cluster)
    return float(r["gap_b"][0]), float(r["gap_se"][0]), float(r["gap_p"][0])

# ------------------------------------------------------------------ A. revenue streams
p("A. Revenue streams (209 ZIPs, 2022Q2-2026Q2)")
tot = sq[sq.zip5.isin(d.zip5)].copy()
tot["group"] = tot.zip5.map(dict(zip(zx.zip5, zx.group)))
share_med = tot.medical.sum() / tot.revenue.sum()
p(f"  medical share of revenue: all {share_med:.3f}; NL {tot[tot.group=='nonlegal'].medical.sum()/tot[tot.group=='nonlegal'].revenue.sum():.3f}; "
  f"LB {tot[tot.group=='legal'].medical.sum()/tot[tot.group=='legal'].revenue.sum():.3f}")
streams = {"Total": ("rev", "y_rpc", "y_rps"), "Adult-use": ("adult", "y_rpc_adult", "y_rps_adult"),
           "Medical": ("medical", "y_rpc_med", "y_rps_med")}
res_streams = {}
pops = d.groupby("nonlegal").zip_pop.sum() / len(QS)      # population per quarter (time-invariant)

def decomp_totals(dd, rcol):
    g = dd.groupby("nonlegal").agg(R=(rcol, "sum"), N=("n_stores", "sum"), P=("zip_pop", "sum"))
    tot_ = np.log(g.R[1] / g.P[1]) - np.log(g.R[0] / g.P[0])
    den_ = np.log(g.N[1] / g.P[1]) - np.log(g.N[0] / g.P[0])
    return tot_, den_, tot_ - den_

rng = np.random.default_rng(SEED)
cty = {k: v.county_fips.unique() for k, v in d.groupby("nonlegal")}
by_cty = {c: g for c, g in d.groupby("county_fips")}
for name, (rcol, yr, ys) in streams.items():
    b1 = pooled(d, yr); b2 = pooled(d, ys)
    pp = ppml(d, rcol)
    t, de, rp = decomp_totals(d, rcol)
    draws = []
    for _ in range(999):
        parts = []
        for k in (0, 1):
            for c in rng.choice(cty[k], len(cty[k])):
                parts.append(by_cty[c])
        draws.append(decomp_totals(pd.concat(parts), rcol))
    draws = np.array(draws)
    ci = lambda j: np.percentile(draws[:, j], [2.5, 97.5])
    sh = draws[:, 1] / draws[:, 0]
    res_streams[name] = dict(rpc=b1, rps=b2, ppml=(float(pp[0][0]), float(pp[1][0]), float(pp[2][0])),
                             total=t, dens=de, rps_comp=rp, ci_total=ci(0), ci_rps=ci(2),
                             share=de / t, ci_share=np.percentile(sh, [2.5, 97.5]))
    p(f"  {name:9s} asinh rpc gap {b1[0]:+.3f} ({b1[1]:.3f}){stars(b1[2])} | log rps gap {b2[0]:+.3f} ({b2[1]:.3f}){stars(b2[2])}"
      f" | PPML {pp[0][0]:+.3f} ({pp[1][0]:.3f}) ratio {np.exp(pp[0][0]):.2f}"
      f" | decomposition total {t:.3f} [{ci(0)[0]:.2f},{ci(0)[1]:.2f}] = density {de:.3f} + rev/store {rp:.3f} [{ci(2)[0]:.2f},{ci(2)[1]:.2f}];"
      f" density share {de/t:.2f} [{np.percentile(sh,2.5):.2f},{np.percentile(sh,97.5):.2f}]")
# medical-incumbent stores vs. all others: is the adult-use gap driven by stores that were medical incumbents?
d_new = zip_panel(zx, sq[sq.med_incumbent == 0], meta, cov)
d_new = d_new.merge(sq[sq.med_incumbent == 0].groupby(["zip5", "qt"]).adult.sum().reset_index(), on=["zip5", "qt"], how="left").fillna({"adult": 0})
b_new = pooled(d_new, "y_rpc"); b_newd = pooled(d_new, "y_den")
p(f"  stores that were NOT medical incumbents only: rpc gap {b_new[0]:+.3f} ({b_new[1]:.3f}){stars(b_new[2])}, "
  f"density gap {b_newd[0]:+.3f} ({b_newd[1]:.3f}){stars(b_newd[2])}")
res_streams["non_incumbent"] = dict(rpc=b_new, den=b_newd)

# ------------------------------------------------------------------ B. quarterly PPML ratios and levels
p("\nB. Quarterly PPML (coefficient, ratio) and level gaps")
qp_rev = ppml(d, "rev", dynamic=True)
qp_n = ppml(d, "n_stores", dynamic=True)
lev = fit(d, "rpc", dynamic=True)
qrows = []
for i, q in enumerate(QS):
    qrows.append((QLAB[i], qp_rev[0][i], qp_rev[1][i], qp_rev[2][i], qp_n[0][i], qp_n[1][i], qp_n[2][i],
                  lev["gap_b"][i], lev["gap_se"][i], lev["gap_p"][i]))
    p(f"  {QLAB[i]} revenue PPML {qp_rev[0][i]:+.3f} ({qp_rev[1][i]:.3f}) x{np.exp(qp_rev[0][i]):.2f} | stores PPML "
      f"{qp_n[0][i]:+.3f} ({qp_n[1][i]:.3f}) x{np.exp(qp_n[0][i]):.2f} | $/resident {lev['gap_b'][i]:+.1f} ({lev['gap_se'][i]:.1f})")
lb_mean = d[d.nonlegal == 0].groupby("qt").rpc.mean()
p("  always-legal-border mean $/resident by quarter: " + ", ".join(f"{QLAB[i]} {lb_mean.loc[q]:.1f}" for i, q in enumerate(QS)))

# ------------------------------------------------------------------ C. exposure measurement
p("\nC. Exposure measurement")
z = us_zips()
lat, lon = zx.lat.to_numpy()[:, None], zx.lon.to_numpy()[:, None]
expo = zx[["zip5"]].copy()
for s in NONLEGAL_STATES:
    zz = z[z.state_id == s]
    dd_ = hav(lat, lon, zz.lat.to_numpy()[None, :], zz.lng.to_numpy()[None, :])
    expo[f"pop75_{s}"] = (zz.population.to_numpy()[None, :] * (dd_ <= 75)).sum(1)
for s in ["CO", "AZ"]:
    zz = z[z.state_id == s]
    dd_ = hav(lat, lon, zz.lat.to_numpy()[None, :], zz.lng.to_numpy()[None, :])
    expo[f"pop75_{s}"] = (zz.population.to_numpy()[None, :] * (dd_ <= 75)).sum(1)
d = d.merge(expo, on="zip5", how="left")
for s in NONLEGAL_STATES:
    d[f"e_{s}"] = np.log1p(d[f"pop75_{s}"] / 1000)
alt_rows = []
# C1. state-specific exposure, all three in one regression (209 ZIPs), and each alone
X_names = [f"e_{s}" for s in NONLEGAL_STATES]
for yv, lab in [("y_rpc", "Revenue per resident"), ("y_den", "Dispensaries per 1,000")]:
    Xd, nm = design(d, dynamic=False, controls=X_names[1:], treat=X_names[0])
    r = ols_fit(d[yv].to_numpy(), Xd, d.county_fips.to_numpy())
    idx = {X_names[0]: 0, X_names[1]: nm.index(X_names[1]), X_names[2]: nm.index(X_names[2])}
    p(f"  [{lab}] joint state-specific exposure: " + "; ".join(
        f"{s} {r['b'][idx['e_'+s]]:+.3f} ({r['se'][idx['e_'+s]]:.3f}){stars(pval(r['b'][idx['e_'+s]], r['se'][idx['e_'+s]], r['G']))}"
        for s in NONLEGAL_STATES))
    alt_rows.append((f"State-specific exposure, {lab.lower()}", {s: (r['b'][idx['e_'+s]], r['se'][idx['e_'+s]],
                     pval(r['b'][idx['e_'+s]], r['se'][idx['e_'+s]], r['G'])) for s in NONLEGAL_STATES}))
p("  ZIPs with any TX / OK / UT residents within 75 mi (209-ZIP sample): " +
  ", ".join(f"{s} {(d.groupby('zip5')[f'pop75_{s}'].first() > 0).sum()}" for s in NONLEGAL_STATES))

# C2. alternative population thresholds for the binary definition (centroid within 75 mi of the state line is
#     approximated by having that state's residents within 75 mi; the baseline 1,000 rule is reproduced exactly)
full_all = panel(("nonlegal", "legal", "interior"))
full_all = full_all.merge(expo, on="zip5", how="left")
nlpop = full_all[[f"pop75_{s}" for s in NONLEGAL_STATES]].max(axis=1)
lgpop = full_all[["pop75_CO", "pop75_AZ"]].max(axis=1)
thr_rows = []
for thr in [1000, 500, 2500, 5000, 10000]:
    nl_ = (nlpop >= thr)
    lb_ = (~nl_) & (lgpop >= thr)
    dd = full_all[nl_ | lb_].copy()
    dd["nonlegal"] = nl_[nl_ | lb_].astype(int).to_numpy()
    b1, b2, b3 = pooled(dd, "y_rpc"), pooled(dd, "y_den"), pooled(dd[dd.n_stores > 0], "y_rps")
    nz = dd.groupby("zip5").nonlegal.first()
    thr_rows.append((thr, b1, b2, b3, int(nz.sum()), int((1 - nz).sum())))
    p(f"  threshold {thr:>6,}: NL {int(nz.sum())} / LB {int((1-nz).sum())} ZIPs | rpc {b1[0]:+.3f} ({b1[1]:.3f}){stars(b1[2])}"
      f" | den {b2[0]:+.3f} ({b2[1]:.3f}){stars(b2[2])} | rps {b3[0]:+.3f} ({b3[1]:.3f}){stars(b3[2])}")
base_check = full_all[(nlpop >= 1000) | (lgpop >= 1000)].zip5.nunique()
p(f"  (check: 1,000 rule with this approximation keeps {base_check} ZIPs; the published groups use the state-line distance rule, 209 ZIPs)")

# C3. drop the ZIPs exposed to both kinds of border; longitude control; town clustering
dual = d.groupby("zip5").apply(lambda g: (g.nonlegal.iloc[0] == 1) and (max(g.pop75_CO.iloc[0], g.pop75_AZ.iloc[0]) >= 1000),
                               include_groups=False)
dual_z = dual[dual].index
p(f"  ZIPs in the nonlegal group with >=1,000 CO/AZ residents within 75 mi: {len(dual_z)}")
other_rows = []
dd = d[~d.zip5.isin(dual_z)]
other_rows.append(("Drop ZIPs exposed to both borders", pooled(dd, "y_rpc"), pooled(dd, "y_den"), pooled(dd[dd.n_stores > 0], "y_rps"),
                   dd.zip5.nunique()))
d["lon_c"] = d.lon - d.lon.mean()
other_rows.append(("Control for longitude", pooled(d, "y_rpc", ("lon_c",)), pooled(d, "y_den", ("lon_c",)),
                   pooled(d[d.n_stores > 0], "y_rps", ("lon_c",)), 209))
other_rows.append(("Cluster by town market", pooled(d, "y_rpc", cluster="town"), pooled(d, "y_den", cluster="town"),
                   pooled(d[d.n_stores > 0], "y_rps", cluster="town"), 209))
for lab, a, b, c, n in other_rows:
    p(f"  {lab:34s} ZIPs {n} | rpc {a[0]:+.3f} ({a[1]:.3f}){stars(a[2])} | den {b[0]:+.3f} ({b[1]:.3f}){stars(b[2])} | rps {c[0]:+.3f} ({c[1]:.3f}){stars(c[2])}")
p(f"  town clusters: {d.town.nunique()}")
corr_lon = np.corrcoef(d.groupby('zip5').lon.first(), d.groupby('zip5').nonlegal.first())[0, 1]
p(f"  correlation of longitude with the nonlegal indicator across ZIPs: {corr_lon:.2f}")

# ------------------------------------------------------------------ tables
def cell(b):
    return f"${b[0]:+.3f}^{{{stars(b[2])}}}$".replace("^{}", ""), f"({b[1]:.3f})"
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Total & Adult-use & Medical \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Regression gaps (209 ZIP codes)}} \\"]
for lab, key in [("Revenue per resident (asinh)", "rpc"), ("Revenue per dispensary (log)", "rps")]:
    c = [cell(res_streams[k][key]) for k in streams]
    L += [f"{lab} & {c[0][0]} & {c[1][0]} & {c[2][0]} \\\\", f" & {c[0][1]} & {c[1][1]} & {c[2][1]} \\\\"]
c = [cell(res_streams[k]["ppml"]) for k in streams]
L += [f"Revenue, Poisson coefficient & {c[0][0]} & {c[1][0]} & {c[2][0]} \\\\", f" & {c[0][1]} & {c[1][1]} & {c[2][1]} \\\\",
      "Implied revenue ratio & " + " & ".join(f"{np.exp(res_streams[k]['ppml'][0]):.2f}" for k in streams) + r" \\",
      r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: Exact decomposition of the log gap in revenue per resident, pooled}} \\",
      "Total gap & " + " & ".join(f"{res_streams[k]['total']:.3f}" for k in streams) + r" \\",
      " & " + " & ".join(f"[{res_streams[k]['ci_total'][0]:.2f}, {res_streams[k]['ci_total'][1]:.2f}]" for k in streams) + r" \\",
      "Store-density component & " + " & ".join(f"{res_streams[k]['dens']:.3f}" for k in streams) + r" \\",
      "Revenue-per-store component & " + " & ".join(f"{res_streams[k]['rps_comp']:.3f}" for k in streams) + r" \\",
      " & " + " & ".join(f"[{res_streams[k]['ci_rps'][0]:.2f}, {res_streams[k]['ci_rps'][1]:.2f}]" for k in streams) + r" \\",
      "Store-density share & " + " & ".join(f"{res_streams[k]['share']:.2f}" for k in streams) + r" \\",
      " & " + " & ".join(f"[{res_streams[k]['ci_share'][0]:.2f}, {res_streams[k]['ci_share'][1]:.2f}]" for k in streams) + r" \\",
      r"\midrule", f"Share of revenue & 1.00 & {1-share_med:.2f} & {share_med:.2f} \\\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_streams.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Revenue (Poisson)} & \multicolumn{2}{c}{Dispensaries (Poisson)} & Revenue per resident \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Quarter & Coefficient & Ratio & Coefficient & Ratio & (dollars, OLS) \\", r"\midrule"]
for (ql, br, sr, pr, bn, sn, pn, bl, sl, pl) in qrows:
    L.append(f"{ql} & ${br:+.3f}^{{{stars(pr)}}}$ ({sr:.3f}) & {np.exp(br):.2f} & ${bn:+.3f}^{{{stars(pn)}}}$ ({sn:.3f}) & {np.exp(bn):.2f} & "
             f"${bl:+.1f}^{{{stars(pl)}}}$ ({sl:.1f}) \\\\".replace("^{}", ""))
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_quarterly_ppml.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & Revenue per & Dispensaries & log revenue & ZIP codes \\", r" & resident (asinh) & per 1,000 (asinh) & per dispensary & (NL / LB) \\", r"\midrule",
     r"\multicolumn{5}{@{}l}{\textit{Panel A: Population threshold for the binary exposure definition}} \\"]
for thr, b1, b2, b3, nn, nl_ in thr_rows:
    c1, c2, c3 = cell(b1), cell(b2), cell(b3)
    L += [f"{thr:,} residents & {c1[0]} & {c2[0]} & {c3[0]} & {nn} / {nl_} \\\\", f" & {c1[1]} & {c2[1]} & {c3[1]} & \\\\"]
L.append(r"\addlinespace\multicolumn{5}{@{}l}{\textit{Panel B: Sample, controls, and clustering}} \\")
for lab, a, b, c, n in other_rows:
    c1, c2, c3 = cell(a), cell(b), cell(c)
    L += [f"{lab} & {c1[0]} & {c2[0]} & {c3[0]} & {n} \\\\", f" & {c1[1]} & {c2[1]} & {c3[1]} & \\\\"]
L.append(r"\addlinespace\multicolumn{5}{@{}l}{\textit{Panel C: State-specific exposure, $\log(1+E^s_z/1000)$, entered jointly}} \\")
for lab, dct in alt_rows:
    pass
r_rpc, r_den = alt_rows[0][1], alt_rows[1][1]
for s, nmz in [("TX", "Texas"), ("OK", "Oklahoma"), ("UT", "Utah")]:
    c1, c2 = cell(r_rpc[s]), cell(r_den[s])
    L += [f"{nmz} residents within 75 miles & {c1[0]} & {c2[0]} & & 209 \\\\", f" & {c1[1]} & {c2[1]} & & \\\\"]
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_exposure_alt.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

pickle.dump(dict(res_streams=res_streams, qrows=qrows, thr_rows=thr_rows, other_rows=other_rows, alt_rows=alt_rows,
                 share_med=share_med), open(os.path.join(OUT, "rio01.pkl"), "wb"))
from paths import LOGS
open(LOGS / "rio01.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
