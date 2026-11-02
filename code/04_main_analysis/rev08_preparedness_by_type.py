r"""
rev08_preparedness_by_type.py -- how prepared were chains and independents when adult-use sales began?

Preparedness is not observed; we measure revealed readiness:
  A. operator level: share of operators with a medical incumbent / an active store in 2022Q2, by lifetime type
  B. storefront level: prepared share phi_k = active stores in 2022Q2 / active stores in 2026Q2, by type and group;
     same for the nonlegal - legal-border density gap (stores per 10,000 residents)
  C. entry timing of operators, chain vs single, before and after conditioning on a medical incumbent;
     and among operators with no medical incumbent (new entrants) only
  D. model fit: gap_k(t) = b_k [1 - (1 - phi_k) exp(-lam t)] fitted by NLS to the raw per-capita density gaps
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import least_squares
from _common import load_zx, load_storefronts, load_store_quarters, QMIN, QMAX, QS, OUT

zx = load_zx()
sf, bm = load_storefronts(zx)
sq, sf = load_store_quarters(sf)
grp = dict(zip(zx.zip5, zx.group))
cty = dict(zip(zx.zip5, zx.county_fips))
pop = zx.groupby("group").zip_pop.sum()
sq["group"] = sq.zip5.map(grp)

lines = []
def p(s=""):
    print(s); lines.append(str(s))

first = sq.groupby("storefront_id").qt.min()
st = sf.set_index("storefront_id").loc[first.index, ["brand_id", "med_incumbent", "zip5", "chain_life"]].copy()
st["faq"] = first

# ---------------- A. operator level
ops = st.groupby("brand_id").agg(n_ever=("faq", "size"), n_med=("med_incumbent", "sum"), first_q=("faq", "min"),
                                 zip5=("zip5", "first"))
ops["chain"] = (ops.n_ever >= 2).astype(int)
ops["med"] = (ops.n_med > 0).astype(int)
ops["open"] = (ops.first_q == QMIN).astype(int)
ops["t_entry"] = ops.first_q - QMIN               # quarters after 2022Q2 of operator's first store
ops["year1"] = (ops.t_entry <= 3).astype(int)     # first store by 2023Q1
ops["county"] = ops.zip5.map(cty)
p("A. Operator level (statewide), by lifetime type")
p(ops.groupby("chain").agg(operators=("med", "size"), share_med=("med", "mean"), share_open=("open", "mean"),
                           mean_t_entry=("t_entry", "mean"), share_year1=("year1", "mean")).round(3).to_string())

# ---------------- B. storefront prepared share
act = sq.groupby(["qt", "group", "chain_life"]).storefront_id.nunique().unstack("chain_life").fillna(0)
act.columns = ["indep", "chain"]
p("\nB. Prepared share phi = active stores 2022Q2 / active stores 2026Q2 (lifetime type)")
for g in ["nonlegal", "legal", "interior"]:
    a0, a1 = act.loc[(QMIN, g)], act.loc[(QMAX, g)]
    p(f"  {g:9s} chain {int(a0.chain)}/{int(a1.chain)} = {a0.chain/a1.chain:.2f}   "
      f"indep {int(a0.indep)}/{int(a1.indep)} = {a0.indep/a1.indep:.2f}")
dens = {k: (act.xs("nonlegal", level="group")[k] / pop["nonlegal"] - act.xs("legal", level="group")[k] / pop["legal"]) * 1e4
        for k in ["chain", "indep"]}
p("  border gap in stores per 10,000 (NL - LB):")
for k in ["chain", "indep"]:
    p(f"    {k}: 2022Q2 {dens[k].loc[QMIN]:.3f}, 2026Q2 {dens[k].loc[QMAX]:.3f}, "
      f"share present at opening {dens[k].loc[QMIN]/dens[k].loc[QMAX]:.2f}")

# ---------------- C. entry timing regressions (operator level, SEs clustered by county of first store)
p("\nC. Operator entry timing: chain vs single (statewide operators; cluster = county of first store)")
o = ops.dropna(subset=["county"])
for y in ["t_entry", "year1"]:
    for f in [f"{y} ~ chain", f"{y} ~ chain + med"]:
        r = smf.ols(f, o).fit(cov_type="cluster", cov_kwds={"groups": o.county})
        p(f"  {f:22s} chain {r.params['chain']:+.3f} (SE {r.bse['chain']:.3f}, p {r.pvalues['chain']:.3f})"
          + (f"; med {r.params['med']:+.3f} (SE {r.bse['med']:.3f})" if "med" in f else "") + f"  N={int(r.nobs)}")
on = o[o.med == 0]
p("  Operators with no medical incumbent only:")
for y in ["t_entry", "year1"]:
    r = smf.ols(f"{y} ~ chain", on).fit(cov_type="cluster", cov_kwds={"groups": on.county})
    p(f"    {y:8s} chain {r.params['chain']:+.3f} (SE {r.bse['chain']:.3f}, p {r.pvalues['chain']:.3f}); "
      f"single mean {on[on.chain==0][y].mean():.2f}, N={int(r.nobs)}")
p(f"    share entering in 2022Q2: chain {on[on.chain==1].open.mean():.3f}, single {on[on.chain==0].open.mean():.3f}")
for g in ["nonlegal", "legal"]:
    sub = on[on.zip5.map(grp) == g]
    p(f"    first store in {g}: N chain {int(sub.chain.sum())}, single {int((1-sub.chain).sum())}; "
      f"mean t_entry chain {sub[sub.chain==1].t_entry.mean():.2f}, single {sub[sub.chain==0].t_entry.mean():.2f}")

# ---------------- D. diffusion model fit to the per-capita gap series
p("\nD. NLS fit gap_k(t) = b_k[1-(1-phi_k)exp(-lam t)], common lam")
t = np.arange(len(QS)); yc, yi = dens["chain"].reindex(QS).to_numpy(), dens["indep"].reindex(QS).to_numpy()
def res(th):
    bc, bi, pc, pi, lam = th
    f = lambda b, ph: b * (1 - (1 - ph) * np.exp(-lam * t))
    return np.r_[yc - f(bc, pc), yi - f(bi, pi)]
fit = least_squares(res, x0=[yc[-1], yi[-1], 0.5, 0.3, 0.2], bounds=([-5, -5, -1, -1, 1e-3], [10, 10, 2, 2, 5]))
bc, bi, pc, pi, lam = fit.x
p(f"  b_chain {bc:.3f}, b_indep {bi:.3f}, phi_chain {pc:.2f}, phi_indep {pi:.2f}, lam {lam:.3f}")
# block bootstrap over ZIP codes for phi
rng = np.random.default_rng(0)
zs = {g: zx[zx.group == g].zip5.to_numpy() for g in ["nonlegal", "legal"]}
zq = sq.groupby(["zip5", "qt", "chain_life"]).storefront_id.nunique().unstack("chain_life").fillna(0)
zq.columns = ["indep", "chain"]
zq = zq.reindex(pd.MultiIndex.from_product([zx.zip5, QS], names=["zip5", "qt"]), fill_value=0)
zpop = dict(zip(zx.zip5, zx.zip_pop))
draws = []
for b in range(500):
    gaps = {}
    for k in ["chain", "indep"]:
        v = []
        for g in ["nonlegal", "legal"]:
            s = rng.choice(zs[g], len(zs[g]))
            cnt = zq.loc[s, k].groupby(level="qt").sum() if False else \
                  zq[k].unstack("qt").loc[s].sum(0)
            v.append(cnt / sum(zpop[z] for z in s))
        gaps[k] = ((v[0] - v[1]) * 1e4).reindex(QS).to_numpy()
    yc, yi = gaps["chain"], gaps["indep"]
    try:
        draws.append(least_squares(res, x0=fit.x, bounds=([-5, -5, -1, -1, 1e-3], [10, 10, 2, 2, 5])).x)
    except Exception:
        pass
draws = np.array(draws)
ci = lambda j: np.percentile(draws[:, j], [2.5, 97.5])
p(f"  bootstrap (500 ZIP resamples within group): phi_chain 95% CI [{ci(2)[0]:.2f}, {ci(2)[1]:.2f}], "
  f"phi_indep [{ci(3)[0]:.2f}, {ci(3)[1]:.2f}], phi_chain - phi_indep: mean {np.mean(draws[:,2]-draws[:,3]):.2f}, "
  f"CI [{np.percentile(draws[:,2]-draws[:,3],2.5):.2f}, {np.percentile(draws[:,2]-draws[:,3],97.5):.2f}]")

with open(os.path.join(OUT, "rev08_preparedness_by_type.txt"), "w") as f:
    f.write("\n".join(lines))
