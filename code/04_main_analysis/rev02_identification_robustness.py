r"""
rev02_identification_robustness.py -- referee Phase 2: covariate balance, predetermined controls, exposure
measures, geographic sensitivity, influential observations, functional form, and small-cluster / spatial inference.

Main spec (pooled version of Table 3): y_zq = beta Nonlegal_z + gamma log pop_z + quarter FE [+ X_z] + e,
209 ZIPs, county-clustered SEs. Dynamic versions use the Table 3 design (17 quarter-specific gaps).
Outputs (revisions/):
  tab_rev_balance.tex, tab_rev_robustness.tex, tab_rev_exposure.tex, fig_rev_dynamic_controls.png
  output/rev02_results.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import OUT, ROOT, QLAB, stars
from _econ import fit, wild_boot, conley, ppml, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
full = A["full"].copy()
full["a_med"] = np.arcsinh(full.n_med_incumbent / full.zip_pop * 1000)
full["a_zbp"] = np.arcsinh(full.zbp19_total_retail / full.zip_pop * 1000)
full["rpc_lvl"] = full.rpc
full["den_lvl"] = full.den
main = full[full.group != "interior"].copy()
OUTC = [("y_rpc", "Revenue per resident (asinh)"), ("y_den", "Dispensaries per 1,000 (asinh)"),
        ("y_rps", "log revenue per dispensary")]
GEO = ["log_density", "dist_abq", "dist_bigcity"]
for c in ["dist_abq", "dist_bigcity"]:
    full[c + "_100"] = full[c] / 100; main[c + "_100"] = main[c] / 100
GEO = ["log_density", "dist_abq_100", "dist_bigcity_100"]

# ------------------------------------------------------------------ 1. covariate balance (ZIP level)
z = main.groupby("zip5").first().reset_index()
z["med_pc"] = z.n_med_incumbent / z.zip_pop * 1000
z["zbp_pc"] = z.zbp19_total_retail / z.zip_pop * 1000
BAL = [("Resident population (000s)", z.zip_pop / 1000), ("log population density", z.log_density),
       ("Distance to Albuquerque (mi)", z.dist_abq), ("Distance to nearest NM city $\\geq$50k (mi)", z.dist_bigcity),
       ("Retail establishments per 1,000, 2019 (ZBP)", z.zbp_pc),
       ("Medical incumbents per 1,000, April 2022", z.med_pc),
       ("Any medical incumbent, April 2022", (z.n_med_incumbent > 0).astype(float)),
       ("Nonlegal-state residents within 75 mi (000s)", z.nlpop_75 / 1000),
       ("Legal-state residents within 75 mi (000s)", z.lgpop_75 / 1000)]
import statsmodels.formula.api as smf
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & Nonlegal- & Always-legal- & Difference & Normalized \\",
     r" & border & border & (SE) & difference \\", r"\midrule"]
p("1. Covariate balance (ZIP level, 92 vs 117)")
for lab, v in BAL:
    dd = pd.DataFrame({"v": v, "nl": z.nonlegal, "c": z.county_fips})
    m = smf.ols("v ~ nl", dd).fit(cov_type="cluster", cov_kwds={"groups": dd.c})
    a, b = v[z.nonlegal == 1], v[z.nonlegal == 0]
    nd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    L.append(f"{lab} & {a.mean():,.2f} & {b.mean():,.2f} & ${m.params.nl:+,.2f}^{{{stars(m.pvalues.nl)}}}$ ({m.bse.nl:,.2f}) & {nd:+.2f} \\\\".replace("^{}", ""))
    p(f"  {lab}: NL {a.mean():.3f} LB {b.mean():.3f} diff {m.params.nl:+.3f} (SE {m.bse.nl:.3f}, p={m.pvalues.nl:.3f}) norm {nd:+.2f}")
L += [r"\midrule", f"ZIP codes & {int(z.nonlegal.sum())} & {int((1 - z.nonlegal).sum())} & & \\\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_balance.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ 2. robustness of the pooled gap
def row(d, label, controls=(), extra=None):
    out = []
    for y, _ in OUTC:
        dd = d.dropna(subset=[y]) if y == "y_rps" else d
        r = fit(dd, y, dynamic=False, controls=controls)
        out.append((r["gap_b"][0], r["gap_se"][0], r["gap_p"][0]))
    return dict(label=label, res=out, nz=d.zip5.nunique(), nnl=d[d.nonlegal == 1].zip5.nunique(), extra=extra)

R = []
R.append(row(main, "Baseline (pooled Table 3 specification)"))
R.append(row(main, "+ geography (density, distance to ABQ and to city $\\geq$50k)", GEO))
R.append(row(main, "+ geography + 2019 retail establishments", GEO + ["a_zbp"]))
R.append(row(main, "+ geography + 2019 retail + medical incumbents", GEO + ["a_zbp", "a_med"]))
for st_, lab in [("TX", "Drop nonlegal ZIPs nearest Texas"), ("UT", "Drop nonlegal ZIPs nearest Utah")]:
    R.append(row(main[~((main.nonlegal == 1) & (main.nearest_nonlegal_state == st_))], lab))
for st_, lab in [("CO", "Drop always-legal ZIPs nearest Colorado"), ("AZ", "Drop always-legal ZIPs nearest Arizona")]:
    R.append(row(main[~((main.nonlegal == 0) & (main.nearest_legal_state == st_))], lab))
# largest markets
top = main.groupby(["group", "county_fips", "county_name" if "county_name" in main else "group"]).rev.sum() if False else None
cty = A["zx"].set_index("county_fips").county_name.to_dict()
crev = main.groupby(["nonlegal", "county_fips"]).rev.sum().sort_values()
big_nl, big_lg = crev.loc[1].index[-1], crev.loc[0].index[-1]
R.append(row(main[main.county_fips != big_nl], f"Drop largest nonlegal-border county ({cty[big_nl]})"))
R.append(row(main[main.county_fips != big_lg], f"Drop largest always-legal-border county ({cty[big_lg]})"))
zr = main.groupby("zip5").rpc.mean()
cut = zr.quantile(0.95)
R.append(row(main[~main.zip5.isin(zr[zr > cut].index)], "Drop top 5\\% of ZIPs by revenue per resident"))
# thresholds 50 / 100 miles
zx = A["zx"]
for r_ in [50, 100]:
    f2 = full.copy()
    nl = (f2.dist_nonleg_mi <= r_) & (f2[f"nlpop_{r_}"] >= 1000)
    lg = (f2.dist_legalneigh_mi <= r_) & (f2[f"lgpop_{r_}"] >= 1000) & ~nl
    f2 = f2[nl | lg].copy(); f2["nonlegal"] = nl[nl | lg].astype(int)
    R.append(row(f2, f"{r_}-mile exposure definition"))
# pooled control (adds interior)
R.append(row(full, "Pooled control (adds 159 interior ZIPs)"))

p("\n2. Pooled nonlegal-border gap, robustness")
for r in R:
    p(f"  {r['label']:<70s} ZIPs {r['nz']:>3d} (NL {r['nnl']:>2d}) " + "  ".join(
        f"{b:+.3f}({s:.3f}){stars(pv)}" for b, s, pv in r["res"]))

# county jackknife
jk = {y: [] for y, _ in OUTC}
for c in main.county_fips.unique():
    d = main[main.county_fips != c]
    for y, _ in OUTC:
        dd = d.dropna(subset=[y]) if y == "y_rps" else d
        r = fit(dd, y, dynamic=False)
        jk[y].append((r["gap_b"][0], r["gap_p"][0]))
p("\n  County jackknife (23 leave-one-county-out fits):")
for y, lab in OUTC:
    a = np.array(jk[y])
    p(f"    {lab}: min {a[:,0].min():+.3f} max {a[:,0].max():+.3f}; p<0.05 in {int((a[:,1] < .05).sum())}/23, p<0.10 in {int((a[:,1] < .10).sum())}/23")

# inference: wild cluster bootstrap and Conley
p("\n3. Inference for the pooled baseline gap")
INF = {}
for y, lab in OUTC:
    dd = main.dropna(subset=[y]) if y == "y_rps" else main
    r = fit(dd, y, dynamic=False)
    wb = wild_boot(dd, y)
    c50 = conley(dd, y, 50); c100 = conley(dd, y, 100)
    INF[y] = dict(b=r["gap_b"][0], se=r["gap_se"][0], p=r["gap_p"][0], wb=wb, c50=c50[1], c100=c100[1])
    p(f"  {lab}: b={r['gap_b'][0]:+.3f} cluster SE {r['gap_se'][0]:.3f} (p={r['gap_p'][0]:.3f}); "
      f"WCR bootstrap p={wb:.3f}; Conley SE 50mi {c50[1]:.3f}, 100mi {c100[1]:.3f}")

# functional form
p("\n4. Functional form (pooled)")
FF = {}
r = fit(main, "rpc_lvl", dynamic=False); FF["rpc_lvl"] = (r["gap_b"][0], r["gap_se"][0], r["gap_p"][0])
r = fit(main, "den_lvl", dynamic=False); FF["den_lvl"] = (r["gap_b"][0], r["gap_se"][0], r["gap_p"][0])
r = fit(main, "any", dynamic=False); FF["any"] = (r["gap_b"][0], r["gap_se"][0], r["gap_p"][0])
b, s, pv = ppml(main, "rev"); FF["ppml_rev"] = (b[0], s[0], pv[0])
b, s, pv = ppml(main, "n_stores"); FF["ppml_n"] = (b[0], s[0], pv[0])
lb = main[main.nonlegal == 0]
p(f"  Revenue per resident, levels ($/qtr): {FF['rpc_lvl'][0]:+.2f} ({FF['rpc_lvl'][1]:.2f}) p={FF['rpc_lvl'][2]:.3f}; LB mean {lb.rpc.mean():.2f}")
p(f"  Dispensaries per 1,000, levels: {FF['den_lvl'][0]:+.3f} ({FF['den_lvl'][1]:.3f}) p={FF['den_lvl'][2]:.3f}; LB mean {lb.den.mean():.3f}")
p(f"  Any dispensary (LPM): {FF['any'][0]:+.3f} ({FF['any'][1]:.3f}) p={FF['any'][2]:.3f}; LB mean {lb['any'].mean():.3f}")
p(f"  PPML revenue (log pop control): {FF['ppml_rev'][0]:+.3f} ({FF['ppml_rev'][1]:.3f}) p={FF['ppml_rev'][2]:.3f} -> x{np.exp(FF['ppml_rev'][0]):.2f}")
p(f"  PPML store count (log pop control): {FF['ppml_n'][0]:+.3f} ({FF['ppml_n'][1]:.3f}) p={FF['ppml_n'][2]:.3f} -> x{np.exp(FF['ppml_n'][0]):.2f}")

# robustness table
L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r" & Revenue per & Dispensaries & log revenue & ZIP & of which \\",
     r" & resident (asinh) & per 1,000 (asinh) & per dispensary & codes & nonlegal \\",
     r" & (1) & (2) & (3) & (4) & (5) \\", r"\midrule",
     r"\multicolumn{6}{@{}l}{\textit{Panel A: Predetermined controls}} \\"]
def trow(r):
    a = " & ".join(f"${b:+.3f}^{{{stars(pv)}}}$".replace("^{}", "") for b, s, pv in r["res"])
    s_ = " & ".join(f"({s:.3f})" for b, s, pv in r["res"])
    return [f"{r['label']} & {a} & {r['nz']} & {r['nnl']} \\\\", f" & {s_} & & \\\\"]
for i, r in enumerate(R):
    if i == 4: L.append(r"\addlinespace\multicolumn{6}{@{}l}{\textit{Panel B: Leave-one-border-state-out and leave-one-market-out}} \\")
    if i == 11: L.append(r"\addlinespace\multicolumn{6}{@{}l}{\textit{Panel C: Exposure definition and control group}} \\")
    L += trow(r)
L.append(r"\addlinespace\multicolumn{6}{@{}l}{\textit{Panel D: Inference for the baseline (row 1)}} \\")
L.append("Wild cluster bootstrap $p$-value & " + " & ".join(f"{INF[y]['wb']:.3f}" for y, _ in OUTC) + r" & & \\")
L.append("Conley SE, 50-mile cutoff & " + " & ".join(f"({INF[y]['c50']:.3f})" for y, _ in OUTC) + r" & & \\")
L.append("Conley SE, 100-mile cutoff & " + " & ".join(f"({INF[y]['c100']:.3f})" for y, _ in OUTC) + r" & & \\")
jkr = lambda y: f"[{np.array(jk[y])[:,0].min():+.2f}, {np.array(jk[y])[:,0].max():+.2f}]"
L.append("County jackknife range (23 fits) & " + " & ".join(jkr(y) for y, _ in OUTC) + r" & & \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_robustness.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ 5. exposure measures (continuous)
p("\n5. Continuous exposure measures, log(1 + E/1000); 209-ZIP sample and 368-ZIP pooled sample")
EXPO = [("nlpop_75", "Nonlegal residents within 75 mi (baseline)"), ("nlpop_50", "Nonlegal residents within 50 mi"),
        ("nlpop_100", "Nonlegal residents within 100 mi"), ("nl_decay_25", "Distance-decay, $e^{-d/25}$"),
        ("nl_decay_50", "Distance-decay, $e^{-d/50}$"),
        ("nlacc_75", "Accessible residents within 75 mi$^{a}$"), ("nlacc_decay_50", "Accessible, distance-decay $e^{-d/50}$$^{a}$")]
EX = {}
for samp, d0 in [("main", main), ("pooled", full)]:
    for e, lab in EXPO:
        d = d0.copy(); d["E"] = np.log1p(d[e] / 1000)
        res = []
        for y, _ in OUTC:
            dd = d.dropna(subset=[y]) if y == "y_rps" else d
            r = fit(dd, y, dynamic=False, treat="E")
            res.append((r["gap_b"][0], r["gap_se"][0], r["gap_p"][0]))
        # implied gap at the median nonlegal-border ZIP exposure relative to zero exposure
        med = np.log1p(d0.loc[d0.nonlegal == 1].groupby("zip5")[e].first().median() / 1000)
        EX[(samp, e)] = (res, med)
        p(f"  [{samp}] {lab:<48s} " + "  ".join(f"{b:+.3f}({s:.3f}){stars(pv)}" for b, s, pv in res) +
          f"   implied rpc gap at NL median exposure: {res[0][0]*med:+.3f}")
# horse race: accessible vs raw
d = main.copy(); d["E1"] = np.log1p(d.nlpop_75 / 1000); d["E2"] = np.log1p(d.nlacc_75 / 1000)
r = fit(d, "y_rpc", dynamic=False, treat="E2", controls=["E1"])
p(f"  Horse race (rpc): accessible {r['gap_b'][0]:+.3f} ({r['gap_se'][0]:.3f}) given raw 75-mi population")
corr = main.groupby("zip5")[["nlpop_75", "nlacc_75", "nl_decay_50"]].first().corr()
p(f"  corr raw-accessible {corr.loc['nlpop_75','nlacc_75']:.3f}, raw-decay50 {corr.loc['nlpop_75','nl_decay_50']:.3f}")
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{Main sample (209 ZIPs)} & \multicolumn{3}{c}{Pooled control (368 ZIPs)} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Exposure measure & Rev./resident & Density & Rev./store & Rev./resident & Density & Rev./store \\", r"\midrule"]
for e, lab in EXPO:
    a, b_ = [], []
    for samp in ["main", "pooled"]:
        for (bb, s, pv) in EX[(samp, e)][0]:
            a.append(f"${bb:+.3f}^{{{stars(pv)}}}$".replace("^{}", "")); b_.append(f"({s:.3f})")
    L.append(f"{lab} & " + " & ".join(a) + r" \\"); L.append(" & " + " & ".join(b_) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_exposure.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ 6. dynamic profile with controls (figure)
fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
x = np.arange(17)
for ax, (y, lab) in zip(axes, OUTC):
    dd = main.dropna(subset=[y]) if y == "y_rps" else main
    r0 = fit(dd, y); r1 = fit(dd, y, controls=GEO + ["a_zbp", "a_med"])
    ax.errorbar(x - 0.15, r0["gap_b"], yerr=1.96 * r0["gap_se"], fmt="o", ms=3.5, color="#1f3b5c", capsize=2, lw=1, label="Baseline")
    ax.errorbar(x + 0.15, r1["gap_b"], yerr=1.96 * r1["gap_se"], fmt="s", ms=3.5, color="#b5651d", capsize=2, lw=1,
                label="With predetermined controls")
    ax.axhline(0, color="grey", ls="--", lw=0.8); ax.set_title(lab, fontsize=10, loc="left"); ax.set_ylabel(r"$\hat\beta_q$")
    p(f"  dynamic [{lab}] baseline 2022Q2 {r0['gap_b'][0]:+.3f}, 2026Q2 {r0['gap_b'][-1]:+.3f}; controls 2022Q2 "
      f"{r1['gap_b'][0]:+.3f}, 2026Q2 {r1['gap_b'][-1]:+.3f}; max |diff| {np.abs(r0['gap_b']-r1['gap_b']).max():.3f}")
axes[0].legend(fontsize=8, frameon=False)
axes[-1].set_xticks(x); axes[-1].set_xticklabels(QLAB, rotation=90, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rev_dynamic_controls.png"), dpi=600); plt.close(fig)

pickle.dump(dict(R=R, INF=INF, FF=FF, EX=EX, jk=jk), open(os.path.join(OUT, "rev02.pkl"), "wb"))
open(os.path.join(OUT, "rev02_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
