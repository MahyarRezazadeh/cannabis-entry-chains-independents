r"""
rio03_continuing_stores.py -- do continuing dispensaries earn an initial premium in high-exposure markets, and does
it erode as rivals enter? (Model-selection diagnostic for the RIO revision.)

Continuing stores = storefronts reporting sales in 2022Q2 (304 statewide; 94 in the 209 main-sample ZIP codes).
 (a) Level at opening: 2022Q2 log revenue on exposure, log residents within 1 km, statewide and main sample.
 (b) Path: store FE + quarter FE + exposure x quarter (2022Q2 omitted) -> change in revenue relative to opening.
 (c) Entry association: log revenue on the number of other active stores within 2 / 5 km, store FE + quarter FE,
     with an exposure interaction; conditional association, not a causal business-stealing estimate.
 (d) Survival of continuing stores to 2026Q2 by exposure.
 (e) Revenue path of entrants by store age (descriptive), by exposure group.
Outputs: output/tables/tab_rio_continuing.tex, output/figures/fig_rio_continuing.png, output/logs/rio03.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import INTER, LOGS
from _common import load_zx, load_storefronts, load_store_quarters, build_zip_covariates, hav, QS, QLAB, QMIN, QMAX, stars, ROOT
from _econ import ols_fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

zx = load_zx(); sf, bm = load_storefronts(zx); sq, sf = load_store_quarters(sf)
cov = build_zip_covariates(zx, sf)
sq["adult"] = (sq.revenue - sq.medical).clip(lower=0)
grp = dict(zip(zx.zip5, zx.group)); cty = dict(zip(zx.zip5, zx.county_fips))
expo = dict(zip(cov.zip5, np.log1p(cov.nlpop_75 / 1000)))
lp = pd.read_csv(INTER / "sf_local_population.csv")
lp.columns = [c.lower() for c in lp.columns]
popcol = [c for c in lp.columns if "1" in c and "pop" in c][0]
sq = sq.merge(lp[["storefront_id", popcol]].rename(columns={popcol: "pop1km"}), on="storefront_id", how="left")
sq["group"] = sq.zip5.map(grp); sq["county"] = sq.zip5.map(cty); sq["x"] = sq.zip5.map(expo)
sq["nl"] = (sq.group == "nonlegal").astype(int)
sq["lrev"] = np.log(sq.revenue); sq["ladult"] = np.log(sq.adult.where(sq.adult > 0))
first = sq.groupby("storefront_id").qt.min()
sq["cont"] = sq.storefront_id.map(first == QMIN).astype(int)
sq["age"] = sq.qt - sq.storefront_id.map(first)

# competitors within r km, by quarter (all active storefronts statewide)
ll = sq.groupby("storefront_id")[["lat", "lon"]].first()
ids = ll.index.to_numpy()
Dkm = hav(ll.lat.to_numpy()[:, None], ll.lon.to_numpy()[:, None], ll.lat.to_numpy()[None, :], ll.lon.to_numpy()[None, :]) * 1.609
pos = {s: i for i, s in enumerate(ids)}
act = sq.pivot_table(index="storefront_id", columns="qt", values="revenue", aggfunc="sum").reindex(ids).notna().to_numpy()
for r in (2, 5):
    Wm = ((Dkm <= r) & (Dkm > 0)).astype(float)
    nb = Wm @ act                       # stores x quarters
    qi = {q: j for j, q in enumerate(sorted(sq.qt.unique()))}
    sq[f"n{r}km"] = [nb[pos[s], qi[q]] for s, q in zip(sq.storefront_id, sq.qt)]

CS = sq[sq.cont == 1].copy()
p(f"continuing stores: statewide {CS.storefront_id.nunique()}, main sample NL {CS[CS.nl==1].storefront_id.nunique()}, "
  f"LB {CS[CS.group=='legal'].storefront_id.nunique()}")

# (a) level at opening
c0 = CS[CS.qt == QMIN].copy(); c0["lpop1"] = np.log(c0.pop1km.clip(lower=1))
ra = smf.ols("lrev ~ x + lpop1", c0).fit(cov_type="cluster", cov_kwds={"groups": c0.county})
cm = c0[c0.group.isin(["nonlegal", "legal"])]
rb = smf.ols("lrev ~ nl + lpop1", cm).fit(cov_type="cluster", cov_kwds={"groups": cm.county})
p(f"(a) 2022Q2 log revenue: statewide exposure {ra.params['x']:+.3f} ({ra.bse['x']:.3f}) N={int(ra.nobs)}; "
  f"main-sample NL {rb.params['nl']:+.3f} ({rb.bse['nl']:.3f}) N={int(rb.nobs)}")

# (b) path relative to opening: store FE + quarter FE + x*quarter
def path(df, xvar, y="lrev"):
    df = df.dropna(subset=[y]).copy()
    Q = df.qt.to_numpy()
    cols, names = [], []
    for q in QS[1:]:
        cols.append(df[xvar].to_numpy() * (Q == q)); names.append(f"g{q}")
    for q in QS[1:]:
        cols.append((Q == q).astype(float)); names.append(f"q{q}")
    S = pd.get_dummies(df.storefront_id).to_numpy(float)
    X = np.column_stack(cols + [S])
    r = ols_fit(df[y].to_numpy(), X, df.county.to_numpy())
    b = np.r_[0, r["b"][:len(QS) - 1]]; se = np.r_[0, r["se"][:len(QS) - 1]]
    late = [names.index(f"g{q}") for q in QS[-9:]]
    w = np.zeros(len(r["b"])); w[late] = 1 / len(late)
    m, s = w @ r["b"], np.sqrt(w @ r["V"] @ w)
    return b, se, (m, s, pval(m, s, r["G"])), r["G"]

b_x, se_x, late_x, Gx = path(CS, "x")
b_nl, se_nl, late_nl, Gn = path(CS[CS.group.isin(["nonlegal", "legal"])], "nl")
b_xa, se_xa, late_xa, _ = path(CS, "x", "ladult")
p(f"(b) continuing-store revenue relative to 2022Q2, late mean (2024Q2-2026Q2): exposure slope {late_x[0]:+.3f} ({late_x[1]:.3f}){stars(late_x[2])} "
  f"[{Gx} counties]; NL vs LB {late_nl[0]:+.3f} ({late_nl[1]:.3f}){stars(late_nl[2])} [{Gn} counties]; adult-use exposure slope "
  f"{late_xa[0]:+.3f} ({late_xa[1]:.3f}){stars(late_xa[2])}")
p("    exposure slope by quarter: " + ", ".join(f"{QLAB[i]} {b_x[i]:+.3f}({se_x[i]:.3f})" for i in range(len(QS))))

# (c) entry association
rc = {}
for r in (2, 5):
    df = CS.copy(); df["lnb"] = np.log1p(df[f"n{r}km"])
    for spec in ("base", "inter"):
        f = "lrev ~ lnb + C(qt) + C(storefront_id)" + (" + lnb:x" if spec == "inter" else "")
        m = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": df.county})
        rc[(r, spec)] = m
        extra = f"; lnb:x {m.params['lnb:x']:+.3f} ({m.bse['lnb:x']:.3f})" if spec == "inter" else ""
        p(f"(c) continuing stores, log(1+rivals within {r} km): {m.params['lnb']:+.3f} ({m.bse['lnb']:.3f}){extra}  N={int(m.nobs)}")
df = sq.copy(); df["lnb"] = np.log1p(df["n5km"])
m_all = smf.ols("lrev ~ lnb + C(qt) + C(storefront_id) + C(age.clip(upper=8))", df).fit(cov_type="cluster", cov_kwds={"groups": df.county})
p(f"(c) all stores, store FE + quarter FE + age FE, log(1+rivals within 5 km): {m_all.params['lnb']:+.3f} ({m_all.bse['lnb']:.3f}) N={int(m_all.nobs)}")

# (d) survival
surv = CS.groupby("storefront_id").agg(last=("qt", "max"), x=("x", "first"), nl=("nl", "first"), group=("group", "first"), county=("county", "first"))
surv["alive"] = (surv["last"] == QMAX).astype(int)
rs = smf.ols("alive ~ x", surv).fit(cov_type="cluster", cov_kwds={"groups": surv.county})
p(f"(d) continuing stores alive in 2026Q2: mean {surv.alive.mean():.2f}; on exposure {rs.params['x']:+.3f} ({rs.bse['x']:.3f}); "
  f"NL {surv[surv.group=='nonlegal'].alive.mean():.2f} vs LB {surv[surv.group=='legal'].alive.mean():.2f}")

# (e) entrants' revenue by age
E = sq[(sq.cont == 0) & sq.group.isin(["nonlegal", "legal"])]
age_prof = E.groupby(["nl", E.age.clip(upper=8)]).lrev.mean().unstack(0)
p("(e) entrants' mean log revenue by store age (cols: LB, NL): " + "; ".join(f"age {a}: {r[0]:.2f}, {r[1]:.2f}" for a, r in age_prof.iterrows()))
me = smf.ols("lrev ~ C(age.clip(upper=8)) + C(qt) + C(storefront_id)", sq[sq.cont == 0]).fit()
ages = [me.params.get(f"C(age.clip(upper=8))[T.{a}]", 0.0) for a in range(1, 9)]
p("    entrants' within-store age profile (relative to first quarter): " + ", ".join(f"{a+1}: {v:+.2f}" for a, v in enumerate(ages)))

# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
xq = np.arange(len(QS))
for a, b, s, ttl in [(ax[0], b_x, se_x, "A. Slope on log exposure, 304 continuing stores"),
                     (ax[1], b_nl, se_nl, "B. Nonlegal vs. always-legal border, 94 stores")]:
    a.axhline(0, color="black", lw=0.8)
    a.errorbar(xq, b, yerr=1.96 * s, fmt="o", color="#1f4e79", ecolor="#7f9fbf", capsize=3, ms=4.5)
    a.set_xticks(xq[::2]); a.set_xticklabels(QLAB[::2], rotation=45, ha="right", fontsize=8)
    a.set_title(ttl, fontsize=10); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("Change in log revenue since 2022Q2")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rio_continuing.png"), dpi=600)

def c(b, s, pv):
    return f"${b:+.3f}^{{{stars(pv)}}}$".replace("^{}", ""), f"({s:.3f})"
rows = [("Log revenue in 2022Q2: log exposure (statewide)", (ra.params['x'], ra.bse['x'], ra.pvalues['x']), int(ra.nobs)),
        ("Log revenue in 2022Q2: nonlegal border (main sample)", (rb.params['nl'], rb.bse['nl'], rb.pvalues['nl']), int(rb.nobs)),
        ("Change since 2022Q2, late mean: log exposure", late_x, CS.storefront_id.nunique()),
        ("Change since 2022Q2, late mean: nonlegal border", late_nl, CS[CS.group.isin(['nonlegal','legal'])].storefront_id.nunique()),
        ("Change since 2022Q2, late mean: log exposure, adult-use revenue", late_xa, CS.storefront_id.nunique()),
        ("Log(1 + rivals within 2 km), store and quarter FE", (rc[(2,'base')].params['lnb'], rc[(2,'base')].bse['lnb'], rc[(2,'base')].pvalues['lnb']), int(rc[(2,'base')].nobs)),
        ("Log(1 + rivals within 5 km), store and quarter FE", (rc[(5,'base')].params['lnb'], rc[(5,'base')].bse['lnb'], rc[(5,'base')].pvalues['lnb']), int(rc[(5,'base')].nobs)),
        ("\\quad $\\times$ log exposure", (rc[(5,'inter')].params['lnb:x'], rc[(5,'inter')].bse['lnb:x'], rc[(5,'inter')].pvalues['lnb:x']), int(rc[(5,'inter')].nobs)),
        ("Survived to 2026Q2 (linear probability): log exposure", (rs.params['x'], rs.bse['x'], rs.pvalues['x']), int(rs.nobs))]
L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Outcome and regressor & Estimate & Observations \\", r"\midrule"]
for lab, est, n in rows:
    a_, b_ = c(*est)
    L += [f"{lab} & {a_} & {n:,} \\\\", f" & {b_} & \\\\"]
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_continuing.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
open(LOGS / "rio03.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
