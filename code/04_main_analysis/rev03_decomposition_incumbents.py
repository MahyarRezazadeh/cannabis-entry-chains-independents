r"""
rev03_decomposition_incumbents.py -- referee Phase 3.

(1) Exact accounting decomposition of the market-revenue gap. For exposure group g and quarter q,
      R_gq / P_gq = (N_gq / P_gq) x (R_gq / N_gq),
    so  log(R/P)_NL - log(R/P)_LB  =  [log(N/P)_NL - log(N/P)_LB]  +  [log(R/N)_NL - log(R/N)_LB]   (exact).
    Group totals include ZIP-quarters with no store (they add population, no stores or revenue), so zero
    markets are handled without transformation. Pooled version sums over all 17 quarters. 95% intervals from a
    county cluster bootstrap (999 draws, counties resampled within exposure group).
    This is an accounting identity, not a causal decomposition.
(2) Inference on the revenue-per-store gap (Table 3 col 3, pooled): 95% CI; one-sided test of the binding-
    constraint benchmark (H0: revenue-per-store gap >= the full revenue-per-resident gap, in log points).
(3) Incumbents vs entrants (store level): stores operating in 2022Q2 ("continuing") vs later entrants;
    within-store revenue paths (store FE) by exposure group; relation between nearby entry and incumbent revenue.
Outputs: tab_rev_decomp.tex, tab_rev_incumbents.tex, fig_rev_decomp.png, fig_rev_incumbents.png,
         output/rev03_results.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from _common import OUT, ROOT, BUILD, QS, QLAB, QMIN, stars
from _econ import fit, ols_fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
full = A["full"]; main = full[full.group != "interior"].copy()

# ------------------------------------------------------------------ (1) exact decomposition
def decomp(d):
    g = d.groupby(["nonlegal", "qt"]).agg(R=("rev", "sum"), N=("n_stores", "sum"), P=("zip_pop", "sum")).reset_index()
    out = {}
    for q in QS + ["pooled"]:
        s = g if q == "pooled" else g[g.qt == q]
        a = s.groupby("nonlegal")[["R", "N", "P"]].sum()
        tot = np.log(a.R[1] / a.P[1]) - np.log(a.R[0] / a.P[0])
        den = np.log(a.N[1] / a.P[1]) - np.log(a.N[0] / a.P[0])
        rps = np.log(a.R[1] / a.N[1]) - np.log(a.R[0] / a.N[0])
        out[q] = (tot, den, rps)
    return out

pt = decomp(main)
rng = np.random.default_rng(3)
cty = {g: main[main.nonlegal == g].county_fips.unique() for g in (0, 1)}
byc = {c: main[main.county_fips == c] for c in main.county_fips.unique()}
BS = []
for _ in range(999):
    parts = []
    for g in (0, 1):
        for i, c in enumerate(rng.choice(cty[g], len(cty[g]))):
            x = byc[c]; x = x[x.nonlegal == g].copy(); x["zip5"] = x.zip5 + f"_{i}"; parts.append(x)
    try:
        BS.append(decomp(pd.concat(parts)))
    except Exception:
        pass
p(f"(1) Exact decomposition of log(revenue per resident) gap; bootstrap draws used: {len(BS)}")
DEC = {}
for q in QS + ["pooled"]:
    arr = np.array([b[q] for b in BS if np.isfinite(b[q]).all()])
    ci = np.percentile(arr, [2.5, 97.5], axis=0)
    sh = arr[:, 1] / arr[:, 0]
    DEC[q] = dict(pt=pt[q], ci=ci, share=pt[q][1] / pt[q][0], share_ci=np.percentile(sh, [2.5, 97.5]),
                  p_rps=2 * min((arr[:, 2] <= 0).mean(), (arr[:, 2] >= 0).mean()),
                  p_den=2 * min((arr[:, 1] <= 0).mean(), (arr[:, 1] >= 0).mean()))
lab = lambda q: "Pooled" if q == "pooled" else QLAB[QS.index(q)]
for q in QS + ["pooled"]:
    t, dn, rp = DEC[q]["pt"]; ci = DEC[q]["ci"]
    p(f"  {lab(q):7s} total {t:+.3f} [{ci[0,0]:+.2f},{ci[1,0]:+.2f}] = density {dn:+.3f} [{ci[0,1]:+.2f},{ci[1,1]:+.2f}] "
      f"+ rev/store {rp:+.3f} [{ci[0,2]:+.2f},{ci[1,2]:+.2f}]  density share {DEC[q]['share']:.2f} "
      f"[{DEC[q]['share_ci'][0]:.2f},{DEC[q]['share_ci'][1]:.2f}]")
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & $\Delta\log(R/P)$ & $\Delta\log(N/P)$ & $\Delta\log(R/N)$ & Share via \\",
     r"Period & (total) & (stores) & (revenue per store) & store count \\", r"\midrule"]
for q in [QS[0], QS[4], QS[8], QS[12], QS[16], "pooled"]:
    t, dn, rp = DEC[q]["pt"]; ci = DEC[q]["ci"]
    if q == "pooled": L.append(r"\midrule")
    L.append(f"{lab(q)} & {t:+.3f} & {dn:+.3f} & {rp:+.3f} & {DEC[q]['share']:.2f} \\\\")
    L.append(f" & [{ci[0,0]:+.2f}, {ci[1,0]:+.2f}] & [{ci[0,1]:+.2f}, {ci[1,1]:+.2f}] & [{ci[0,2]:+.2f}, {ci[1,2]:+.2f}] & "
             f"[{DEC[q]['share_ci'][0]:.2f}, {DEC[q]['share_ci'][1]:.2f}] \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_decomp.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(7.5, 3.9))
x = np.arange(17)
tot = np.array([DEC[q]["pt"][0] for q in QS]); dn = np.array([DEC[q]["pt"][1] for q in QS]); rp = np.array([DEC[q]["pt"][2] for q in QS])
ax.bar(x, dn, color="#1f3b5c", label="Store count, $\\Delta\\log(N/P)$")
ax.bar(x, rp, bottom=np.where(rp >= 0, dn.clip(min=0), dn.clip(max=0) * 0), color="#b5651d", alpha=0.8, label="Revenue per store, $\\Delta\\log(R/N)$")
ax.plot(x, tot, "ko-", ms=3.5, lw=1, label="Total, $\\Delta\\log(R/P)$")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(QLAB, rotation=90, fontsize=8); ax.set_ylabel("Log points"); ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rev_decomp.png"), dpi=600); plt.close(fig)

# ------------------------------------------------------------------ (2) inference on revenue per store
p("\n(2) Revenue-per-store gap (Table 3 column 3, pooled ZIP regression)")
d3 = main.dropna(subset=["y_rps"])
r = fit(d3, "y_rps", dynamic=False); b, se, G = r["gap_b"][0], r["gap_se"][0], r["G"]
tc = stats.t.ppf(0.975, G - 1)
rr = fit(main, "y_rpc", dynamic=False)
p(f"  rev/store gap {b:+.3f} (SE {se:.3f}); 95% CI [{b - tc*se:+.3f}, {b + tc*se:+.3f}] "
  f"= [{np.exp(b - tc*se)-1:+.1%}, {np.exp(b + tc*se)-1:+.1%}] in proportional terms")
# benchmark: under a binding cap the entire log revenue-per-resident gap would show up in revenue per store.
bench = DEC["pooled"]["pt"][0]
tstat = (b - bench) / se
p(f"  binding-constraint benchmark (pooled exact log gap in revenue per resident) = {bench:+.3f}; "
  f"one-sided H0: rev/store gap >= benchmark: t = {tstat:.2f}, p = {stats.t.cdf(tstat, G - 1):.3f}")
# same test with the exact decomposition bootstrap (no regression adjustment)
arr = np.array([bb["pooled"] for bb in BS])
p(f"  exact-decomposition bootstrap: share of draws with rev/store component >= total gap: {(arr[:,2] >= arr[:,0]).mean():.3f}; "
  f"rev/store component 95% CI [{np.percentile(arr[:,2],2.5):+.3f}, {np.percentile(arr[:,2],97.5):+.3f}]")
RPS = dict(b=b, se=se, ci=(b - tc*se, b + tc*se), bench=bench, p_bench=stats.t.cdf(tstat, G - 1))

# ------------------------------------------------------------------ (3) incumbents vs entrants
p("\n(3) Incumbents (operating in 2022Q2) vs later entrants, store level")
sq = A["sq"].merge(A["meta"][["storefront_id", "faq", "lc"]], on="storefront_id")
grp = A["zx"].set_index("zip5").group
sq["group"] = sq.zip5.map(grp)
sq = sq[sq.group.isin(["nonlegal", "legal"])].copy()
sq["nonlegal"] = (sq.group == "nonlegal").astype(int)
sq["lrev"] = np.log(sq.revenue)
sq["county_fips"] = sq.zip5.map(A["zx"].set_index("zip5").county_fips)
inc = sq[sq.lc]
p(f"  stores in NL/LB ZIPs: {sq.storefront_id.nunique()} (incumbents {inc.storefront_id.nunique()}: NL "
  f"{inc[inc.nonlegal==1].storefront_id.nunique()}, LB {inc[inc.nonlegal==0].storefront_id.nunique()})")
# within-store revenue path of incumbents: log rev ~ store FE + quarter FE + sum_{q>2022Q2} NL x 1{q}
def within(d, label):
    d = d.copy()
    sid = pd.factorize(d.storefront_id)[0]
    Q = d.qt.to_numpy()
    cols = [d.nonlegal.to_numpy() * (Q == q) for q in QS[1:]]
    X = np.column_stack(cols + [(Q == q) for q in QS[1:]]).astype(float)
    # demean by store (store FE)
    def dm(v):
        s = pd.Series(v).groupby(sid).transform("mean").to_numpy(); return v - s
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])]); yd = dm(d.lrev.to_numpy())
    keep = Xd.std(0) > 0
    r = ols_fit(yd, Xd[:, keep], d.county_fips.to_numpy())
    nb = keep[:16].sum()
    b, se = np.full(16, np.nan), np.full(16, np.nan)
    b[keep[:16]] = r["b"][:nb]; se[keep[:16]] = r["se"][:nb]
    return b, se, r["G"]
b_inc, se_inc, G_inc = within(inc, "incumbents")
p("  incumbents' within-store revenue change relative to 2022Q2, NL minus LB:")
p("   " + "  ".join(f"{QLAB[i+1]}:{b_inc[i]:+.2f}({se_inc[i]:.2f})" for i in range(16)))
avg = np.nanmean(b_inc[-8:])
p(f"   mean of last 8 quarters: {avg:+.3f}; counties {G_inc}")
# entrants' revenue path by market age (quarters since own entry), NL vs LB
ent = sq[~sq.lc].copy(); ent["age"] = ent.qt - ent.faq
eg = ent[ent.age <= 12].groupby(["nonlegal", "age"]).lrev.mean().unstack(0)
p("  entrants' mean log revenue by quarters since entry (LB, NL): " +
  "  ".join(f"{a}:{eg.loc[a,0]:.2f}/{eg.loc[a,1]:.2f}" for a in eg.index[:9]))
# nearby entry and incumbent revenue: store FE + quarter FE, log(1 + dispensaries within 5 mi), x NL
ps = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), usecols=["storefront_id", "qt", "n_disp_5mi", "n_disp_10mi"])
inc2 = inc.merge(ps, on=["storefront_id", "qt"], how="left").dropna(subset=["n_disp_5mi"])
inc2["ln5"] = np.log1p(inc2.n_disp_5mi); inc2["ln5_nl"] = inc2.ln5 * inc2.nonlegal
sid = pd.factorize(inc2.storefront_id)[0]
def dm(v): return v - pd.Series(v).groupby(sid).transform("mean").to_numpy()
Qd = np.column_stack([(inc2.qt.to_numpy() == q).astype(float) for q in QS[1:]])
X = np.column_stack([inc2.ln5, inc2.ln5_nl, Qd]); Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
r = ols_fit(dm(inc2.lrev.to_numpy()), Xd, inc2.county_fips.to_numpy())
pv = pval(r["b"][:2], r["se"][:2], r["G"])
p(f"  incumbent log revenue on log(1+dispensaries within 5 mi), store+quarter FE: LB slope {r['b'][0]:+.3f} ({r['se'][0]:.3f}) "
  f"p={pv[0]:.3f}; NL difference {r['b'][1]:+.3f} ({r['se'][1]:.3f}) p={pv[1]:.3f}; store-quarters {len(inc2)}")
NEAR = dict(b=r["b"][:2], se=r["se"][:2], p=pv, n=len(inc2), stores=inc2.storefront_id.nunique())

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.errorbar(np.arange(1, 17), b_inc, yerr=1.96 * se_inc, fmt="o", ms=4, capsize=2.5, lw=1, color="#1f3b5c")
ax.axhline(0, color="grey", ls="--", lw=0.8)
ax.set_xticks(np.arange(0, 17)); ax.set_xticklabels(QLAB, rotation=90, fontsize=8)
ax.set_ylabel("NL $-$ LB change in log revenue\nrelative to 2022Q2")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rev_incumbents.png"), dpi=600); plt.close(fig)

L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r" & Coefficient & (SE) \\", r"\midrule",
     r"\multicolumn{3}{@{}l}{\textit{Panel A: Continuing stores, within-store change in log revenue relative to 2022Q2, NL $-$ LB}} \\"]
for i in [0, 3, 7, 11, 15]:
    L.append(f"{QLAB[i+1]} & ${b_inc[i]:+.3f}$ & ({se_inc[i]:.3f}) \\\\")
L.append(f"Mean, last eight quarters & ${avg:+.3f}$ & \\\\")
L.append(r"\addlinespace\multicolumn{3}{@{}l}{\textit{Panel B: Continuing stores, log revenue on local rival count (store and quarter FE)}} \\")
L.append(f"$\\log(1+\\text{{dispensaries within 5 mi}})$ & ${NEAR['b'][0]:+.3f}^{{{stars(NEAR['p'][0])}}}$ & ({NEAR['se'][0]:.3f}) \\\\".replace("^{}", ""))
L.append(f"\\quad $\\times$ Nonlegal & ${NEAR['b'][1]:+.3f}^{{{stars(NEAR['p'][1])}}}$ & ({NEAR['se'][1]:.3f}) \\\\".replace("^{}", ""))
L.append(r"\midrule")
L.append(f"Continuing stores (NL / LB) & \\multicolumn{{2}}{{c}}{{{inc[inc.nonlegal==1].storefront_id.nunique()} / {inc[inc.nonlegal==0].storefront_id.nunique()}}} \\\\")
L.append(f"Store-quarters (Panel B) & \\multicolumn{{2}}{{c}}{{{NEAR['n']:,}}} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_incumbents.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
pickle.dump(dict(DEC=DEC, RPS=RPS, b_inc=b_inc, se_inc=se_inc, NEAR=NEAR, n_inc=(inc[inc.nonlegal==1].storefront_id.nunique(), inc[inc.nonlegal==0].storefront_id.nunique())), open(os.path.join(OUT, "rev03.pkl"), "wb"))
open(os.path.join(OUT, "rev03_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
