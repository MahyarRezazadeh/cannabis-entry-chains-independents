r"""
05_composition_decomp.py -- updated_draft_with_IO -- how much of the Table 3 gap is organizational supply?

Main sample and spec identical to Table 3 (tab_fact1fact2_dynamic.tex / 03_fix_figures.py):
209 ZIPs (92 nonlegal-border, 117 always-legal-border), 2022Q2-2026Q2, y ~ sum_q nonlegal x 1{q} + log pop + quarter FE,
SEs clustered by county. Operator type = lifetime brand definition used for Fact 3 (chain = brand with >= 2 NM storefronts).

  A. Composition by quarter: nonlegal gap in independent and chain store density (asinh, per 1,000 residents)
  B. Exact accounting in levels: rev per resident = indep part + chain part; density = indep + chain
  C. Gelbach (2016) decomposition of the Table 3 coefficient on asinh(revenue per resident):
       base  = Table 3 spec
       full  = base + asinh(indep stores per 1,000) + asinh(chain stores per 1,000)
       explained_q = sum_k gamma_k * delta_kq (exact); leftover = beta_full_q
     plus an alternative split by at-entry status (single / first-location / established).
     County cluster bootstrap (499 draws) for pooled shares.
Output: composition_decomp_results.txt
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

BUILD = str(_P.BUILD)
ROOT = str(_P.OUT)
EARTH_R = 3958.8
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
QLAB = [f"{(q)//4}Q{q%4+1}" for q in range(QMIN, QMAX + 1)]
LOG = []
asinh = np.arcsinh


def p(s=""):
    print(s); LOG.append(str(s))


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


def st(pv):
    return "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""


# ---------------------------------------------------------------- build (as 03_fix_figures.py)
zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})
zx["group"] = np.select([zx.border_z == 1, (zx.placebo_border_z == 1) & (zx.border_z == 0)],
                        ["nonlegal", "legal"], default="interior")
sf = pd.read_csv(os.path.join(BUILD, "storefronts.csv"), dtype={"zip5": str})
xw = pd.read_csv(os.path.join(BUILD, "brand_crosswalk.csv"))
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
sf = sf.merge(xw[["storefront_id", "brand_id"]], on="storefront_id", how="left")
sf["chain"] = sf.brand_id.map(dict(zip(bm.brand_id, (bm.n_storefronts >= 2).astype(int))))
for i in sf.index[~sf.zip5.isin(set(zx.zip5))]:
    sf.at[i, "zip5"] = zx.zip5.to_numpy()[np.argmin(hav(sf.at[i, "lat"], sf.at[i, "lon"],
                                                        zx.lat.to_numpy(), zx.lon.to_numpy()))]
sfm = pd.read_csv(os.path.join(BUILD, "sf_month_revenue.csv"))
sfm["qt"] = sfm.year * 4 + (sfm.month - 1) // 3
sq = sfm[sfm.qt.between(QMIN, QMAX)].groupby(["storefront_id", "qt"]).agg(revenue=("total", "sum")).reset_index()
sq = sq[sq.revenue > 0].merge(sf[["storefront_id", "zip5", "brand_id", "chain"]], on="storefront_id")

# at-entry status (same algorithm as 02/04)
ps = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), dtype={"zip5": str})
ps = ps[ps.qt.between(QMIN, QMAX)]
fa = ps[ps.active == 1].groupby("storefront_id").qt.min().rename("faq")
meta = ps.groupby("storefront_id").agg(brand_id=("brand_id", "first"), lc=("left_censored", "first")).reset_index()
meta = meta.merge(fa, on="storefront_id").merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
meta["lc"] = meta.lc.astype(str).isin(["True", "1", "1.0", "true"])
rows = []
for bid, gg in meta.groupby("brand_id"):
    for _, r in gg.iterrows():
        prior = ((gg.faq < r.faq) & (gg.storefront_id != r.storefront_id)).sum()
        prior_lc = ((gg.lc) & (gg.storefront_id != r.storefront_id) & (gg.faq <= r.faq)).sum()
        has = (prior > 0) or ((not r.lc) and prior_lc > 0)
        rows.append((r.storefront_id, "established" if has else ("first_loc" if r.n_storefronts >= 2 else "single")))
sq = sq.merge(pd.DataFrame(rows, columns=["storefront_id", "status"]), on="storefront_id", how="left")

grid = pd.MultiIndex.from_product([sorted(zx.zip5), range(QMIN, QMAX + 1)], names=["zip5", "qt"]).to_frame(index=False)
full = grid.merge(zx[["zip5", "zip_pop", "group", "county_fips"]], on="zip5")
agg = {"n_all": sq.assign(k=1), "n_I": sq[sq.chain == 0].assign(k=1), "n_C": sq[sq.chain == 1].assign(k=1),
       "n_S": sq[sq.status == "single"].assign(k=1), "n_F": sq[sq.status == "first_loc"].assign(k=1),
       "n_E": sq[sq.status == "established"].assign(k=1)}
for c, d in agg.items():
    full = full.merge(d.groupby(["zip5", "qt"]).agg(**{c: ("k", "sum"), "r" + c[1:]: ("revenue", "sum")}).reset_index(),
                      on=["zip5", "qt"], how="left")
full = full.fillna(0)
full = full[full.group != "interior"].copy()
full["nonlegal"] = (full.group == "nonlegal").astype(int)
full["log_pop"] = np.log(full.zip_pop.clip(lower=1))
for k in ["all", "I", "C", "S", "F", "E"]:
    full[f"den_{k}"] = full[f"n_{k}"] / full.zip_pop * 1000          # stores per 1,000
    full[f"rpc_{k}"] = full[f"r_{k}"] / full.zip_pop                  # $ per resident
    full[f"a_den_{k}"] = asinh(full[f"den_{k}"])
full["y_rpc"] = asinh(full.rpc_all)
full["y_den"] = asinh(full.den_all)
full = full.sort_values(["zip5", "qt"]).reset_index(drop=True)
assert len(full) == 3553
p(f"Main sample: {full.zip5.nunique()} ZIPs, {len(full)} ZIP-quarters, {full.county_fips.nunique()} counties")
p(f"Stores (lifetime def.): indep store-qtrs={int(full.n_I.sum())}, chain store-qtrs={int(full.n_C.sum())}")

# ---------------------------------------------------------------- design matrices (Table 3 spec)
Q = full.qt.to_numpy(); qs = list(range(QMIN, QMAX + 1))
D_int = np.column_stack([(full.nonlegal.to_numpy() * (Q == q)) for q in qs]).astype(float)   # 17 gaps
D_fe = np.column_stack([(Q == q) for q in qs]).astype(float)                                 # quarter FE (absorbs const)
X0 = np.column_stack([D_int, D_fe, full.log_pop.to_numpy()])
D_pool = np.column_stack([full.nonlegal.to_numpy(), D_fe, full.log_pop.to_numpy()]).astype(float)
cl = pd.factorize(full.county_fips)[0]


def ols_cl(y, X, idx):
    """OLS with county-clustered (CR1) SEs; returns coefs and SEs for columns idx."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    G = cl.max() + 1; n, k = X.shape
    meat = np.zeros((k, k))
    for g in range(G):
        s = X[cl == g].T @ e[cl == g]; meat += np.outer(s, s)
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    return b[idx], np.sqrt(np.diag(V))[idx], b


from scipy import stats as sst
pval = lambda b, se: 2 * (1 - sst.t.cdf(abs(b / se), df=cl.max()))

# check against Table 3
b, se, _ = ols_cl(full.y_rpc.to_numpy(), X0, range(17))
p(f"[check] Table 3 col 1 recomputed: 2022Q2 {b[0]:+.3f} ({se[0]:.3f}), 2026Q2 {b[-1]:+.3f} ({se[-1]:.3f})")

# ---------------------------------------------------------------- A. composition by quarter
p("\nA. Nonlegal - always-legal gap in store density by operator type, asinh(stores per 1,000) [Table 3 spec]")
p(f"{'q':8s}{'all':>16s}{'independent':>18s}{'chain':>18s}{'indep. share of stores NL/LB':>32s}")
bA = {}
for k in ["all", "I", "C"]:
    bA[k] = ols_cl(full[f"a_den_{k}"].to_numpy(), X0, range(17))[:2]
for i, q in enumerate(qs):
    s = full[full.qt == q]
    shnl = s.loc[s.nonlegal == 1, "n_I"].sum() / max(s.loc[s.nonlegal == 1, "n_all"].sum(), 1)
    shlg = s.loc[s.nonlegal == 0, "n_I"].sum() / max(s.loc[s.nonlegal == 0, "n_all"].sum(), 1)
    cells = "".join(f"{bA[k][0][i]:+9.3f}{st(pval(bA[k][0][i], bA[k][1][i])):3s}({bA[k][1][i]:.3f})" for k in ["all", "I", "C"])
    p(f"{QLAB[i]:8s}{cells}    {shnl:.2f} / {shlg:.2f}")

# ---------------------------------------------------------------- B. exact accounting in levels
p("\nB. Exact accounting in levels (OLS is linear: beta_total = beta_indep + beta_chain)")
for wlab, w in [("unweighted", None), ("population-weighted", full.zip_pop.to_numpy())]:
    p(f"  [{wlab}]")
    for out, lab in [("rpc", "revenue per resident ($/qtr)"), ("den", "stores per 1,000 residents")]:
        resq = {}
        for k in ["all", "I", "C"]:
            y = full[f"{out}_{k}"].to_numpy(); X = X0; Xp = D_pool
            if w is not None:
                sw = np.sqrt(w); y = y * sw; X = X0 * sw[:, None]; Xp = D_pool * sw[:, None]
            resq[k] = (ols_cl(y, X, range(17))[:2], ols_cl(y, Xp, [0])[:2])
        bp = {k: resq[k][1][0][0] for k in resq}; sp = {k: resq[k][1][1][0] for k in resq}
        p(f"    {lab}: pooled gap {bp['all']:+.3f} ({sp['all']:.3f}) = indep {bp['I']:+.3f} ({sp['I']:.3f}) "
          f"[{bp['I']/bp['all']:.0%}] + chain {bp['C']:+.3f} ({sp['C']:.3f}) [{bp['C']/bp['all']:.0%}]")
        for i in [0, 4, 8, 12, 16]:
            bt, bi, bc = (resq[k][0][0][i] for k in ["all", "I", "C"])
            p(f"      {QLAB[i]}: total {bt:+.3f} = indep {bi:+.3f} [{bi/bt:.0%}] + chain {bc:+.3f} [{bc/bt:.0%}]")

# ---------------------------------------------------------------- C. Gelbach decomposition of Table 3 col 1
def gelbach(y, Z, X=X0, nint=17):
    """Exact Gelbach split of the nint gap coefficients when adding controls Z (n x m)."""
    XF = np.column_stack([X, Z])
    bb = np.linalg.lstsq(X, y, rcond=None)[0][:nint]
    bf_all = np.linalg.lstsq(XF, y, rcond=None)[0]
    bf, gam = bf_all[:nint], bf_all[X.shape[1]:]
    delt = np.linalg.lstsq(X, Z, rcond=None)[0][:nint]          # nint x m
    contrib = delt * gam[None, :]                                # nint x m ; sum = bb - bf
    return bb, bf, contrib, gam


def boot(fn, B=499, seed=7):
    rng = np.random.default_rng(seed); G = cl.max() + 1; out = []
    groups = [np.where(cl == g)[0] for g in range(G)]
    for _ in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, G, G)])
        out.append(fn(idx))
    return np.array(out)


y = full.y_rpc.to_numpy()
SETS = {"lifetime type: indep + chain": ["a_den_I", "a_den_C"],
        "at-entry status: single + first-loc + established": ["a_den_S", "a_den_F", "a_den_E"],
        "independent density only": ["a_den_I"],
        "chain density only": ["a_den_C"],
        "total density only (no composition)": ["a_den_all"]}
p("\nC. Gelbach decomposition of the Table 3 gap in asinh(revenue per resident)")
for slab, zc in SETS.items():
    Z = full[zc].to_numpy()
    bb, bf, ct, gam = gelbach(y, Z)
    _, sef, _ = ols_cl(y, np.column_stack([X0, Z]), range(17))
    # pooled version
    bbp, bfp, ctp, _ = gelbach(y, Z, D_pool, 1)
    _, sefp, _ = ols_cl(y, np.column_stack([D_pool, Z]), [0])

    def fn(idx, Z=Z):
        b0, b1, c, _ = gelbach(y[idx], Z[idx], D_pool[idx], 1)
        return np.r_[b1[0], c[0], (b0[0] - b1[0]) / b0[0]]
    bs = boot(fn)
    p(f"\n  [{slab}]  gamma = " + ", ".join(f"{c}:{g:.3f}" for c, g in zip(zc, gam)))
    p(f"   pooled: gap {bbp[0]:+.3f} -> leftover {bfp[0]:+.3f} (SE {sefp[0]:.3f}, p={pval(bfp[0], sefp[0]):.3f}); "
      f"explained {bbp[0]-bfp[0]:+.3f} = {(bbp[0]-bfp[0])/bbp[0]:.0%} [boot 95% CI {np.percentile(bs[:, -1], 2.5):.0%}, "
      f"{np.percentile(bs[:, -1], 97.5):.0%}]")
    for j, c in enumerate(zc):
        p(f"      via {c}: {ctp[0, j]:+.3f} ({ctp[0, j]/bbp[0]:.0%} of gap; boot SE {bs[:, 1 + j].std():.3f})")
    p(f"   {'q':8s}{'gap':>8s}{'leftover':>18s}{'explained':>11s}" + "".join(f"{c:>14s}" for c in zc))
    for i in range(17):
        p(f"   {QLAB[i]:8s}{bb[i]:+8.3f}{bf[i]:+9.3f}{st(pval(bf[i], sef[i])):3s}({sef[i]:.3f}){(bb[i]-bf[i])/bb[i]:10.0%}"
          + "".join(f"{ct[i, j]:+14.3f}" for j in range(len(zc))))

# same Gelbach for density gap (Table 3 col 2) and revenue per store, lifetime type
p("\n  Note: for asinh(density) the indep/chain controls explain ~100% by construction (density = indep + chain);")
p("  the split of that gap is reported in A and B.")
d = full[full.n_all > 0].copy()
d["y_rps"] = np.log(d.r_all / d.n_all); d["sh_I"] = d.n_I / d.n_all
m0 = smf.ols("y_rps ~ nonlegal + log_pop + C(qt)", d).fit(cov_type="cluster", cov_kwds={"groups": d.county_fips})
m1 = smf.ols("y_rps ~ nonlegal + sh_I + log_pop + C(qt)", d).fit(cov_type="cluster", cov_kwds={"groups": d.county_fips})
p(f"\n  log(revenue per store), {len(d)} ZIP-qtrs with stores: gap {m0.params.nonlegal:+.3f} ({m0.bse.nonlegal:.3f}); "
  f"+ indep share of stores: {m1.params.nonlegal:+.3f} ({m1.bse.nonlegal:.3f}); indep share coef "
  f"{m1.params.sh_I:+.3f} ({m1.bse.sh_I:.3f})")
rI = (d.r_I.sum() / d.n_I.sum()); rC = (d.r_C.sum() / d.n_C.sum())
p(f"  mean quarterly revenue per store: independent ${rI:,.0f}, chain ${rC:,.0f}")

open(os.path.join(ROOT, "logs", "composition_decomp_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
