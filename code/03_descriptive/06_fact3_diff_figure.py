r"""
06_fact3_diff_figure.py -- updated_draft_with_IO -- Fact 3 event-study figure for the independent-minus-chain gap

Replicates Table 4 (tab_fact3_dynamic.tex): y_zq = sum_s beta_s Nonlegal_z x 1{q=s} + gamma log pop + quarter FE,
y = asinh(# distinct independent brands), asinh(# distinct chain brands), and their difference
asinh(I) - asinh(C) (whose coefficient equals beta_I - beta_C exactly, with the correct joint SE).
Chain = brand with >= 2 NM storefronts over the sample. SEs clustered by county.
Checks which sample reproduces the published table, then plots the difference column.
Output: fig_fact3_diff.png
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUILD = str(_P.BUILD)
ROOT = str(_P.OUT)
EARTH_R = 3958.8
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
QLAB = [f"{q//4}Q{q%4+1}" for q in range(QMIN, QMAX + 1)]


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


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
g = (sq.groupby(["zip5", "qt"]).apply(lambda s: pd.Series({
        "n_I": s.loc[s.chain == 0, "brand_id"].nunique(), "n_C": s.loc[s.chain == 1, "brand_id"].nunique()}))
     .reset_index())
grid = pd.MultiIndex.from_product([sorted(zx.zip5), range(QMIN, QMAX + 1)], names=["zip5", "qt"]).to_frame(index=False)
full = grid.merge(g, on=["zip5", "qt"], how="left").fillna(0).merge(
    zx[["zip5", "zip_pop", "group", "county_fips"]], on="zip5")
full["log_pop"] = np.log(full.zip_pop.clip(lower=1))
full["yI"] = np.arcsinh(full.n_I); full["yC"] = np.arcsinh(full.n_C); full["yD"] = full.yI - full.yC
full["nonlegal"] = (full.group == "nonlegal").astype(int)
full["legal"] = (full.group == "legal").astype(int)
for i, q in enumerate(range(QMIN, QMAX + 1)):
    full[f"nl{i}"] = full.nonlegal * (full.qt == q)
    full[f"lg{i}"] = full.legal * (full.qt == q)
NL = " + ".join(f"nl{i}" for i in range(17)); LG = " + ".join(f"lg{i}" for i in range(17))


def dyn(d, y, extra=""):
    m = smf.ols(f"{y} ~ {NL}{extra} + log_pop + C(qt)", d).fit(cov_type="cluster", cov_kwds={"groups": d.county_fips})
    return pd.DataFrame({"b": [m.params[f"nl{i}"] for i in range(17)], "se": [m.bse[f"nl{i}"] for i in range(17)],
                         "p": [m.pvalues[f"nl{i}"] for i in range(17)]})


PUB = {"yI": [0.055, 0.127, 0.204], "yC": [0.219, 0.243, 0.280], "yD": [-0.173, -0.124, -0.084]}
cands = {"main 209 (interior excluded)": (full[full.group != "interior"], ""),
         "pooled 368 (interior + legal in control, NL vs rest)": (full, ""),
         "pooled 368 with separate legal-border x quarter terms": (full, " + " + LG)}
best = None
for lab, (d, ex) in cands.items():
    r = {y: dyn(d, y, ex) for y in ["yI", "yC", "yD"]}
    err = max(abs(r[y].b.iloc[k] - PUB[y][k]) for y in PUB for k in range(3))
    print(f"[{lab}] {d.zip5.nunique()} ZIPs, {d.county_fips.nunique()} counties: first 3 qtrs "
          f"I {r['yI'].b.head(3).round(3).tolist()} C {r['yC'].b.head(3).round(3).tolist()} "
          f"D {r['yD'].b.head(3).round(3).tolist()} SE_D {r['yD'].se.head(3).round(3).tolist()}  max err {err:.4f}")
    if best is None or err < best[0]:
        best = (err, lab, d, r)
err, lab, d, r = best
print(f"\nBest match for columns 1-2: {lab} (max err {err:.4f})")

# The published difference column is a stacked regression on both types: type-specific quarter FE and
# type-specific Nonlegal x quarter terms, common log-population slope. Reproduce it, then plot the published values.
L = pd.concat([d.assign(y=d.yI, I=1), d.assign(y=d.yC, I=0)])
for i in range(17):
    L[f"dI{i}"] = L[f"nl{i}"] * L.I
m = smf.ols(f"y ~ {NL} + " + " + ".join(f"dI{i}" for i in range(17)) + " + I + log_pop + C(qt):C(I)", L).fit(
    cov_type="cluster", cov_kwds={"groups": L.county_fips})
rep = np.array([m.params[f"dI{i}"] for i in range(17)])
rows = [l for l in open(os.path.join(ROOT, "tables", "tab_fact3_dynamic.tex"), encoding="utf-8").read().splitlines()
        if l[:2] == "20" and "Q" in l[:7]]
D = pd.DataFrame([[float(c.strip().replace("*", "").replace("(", "").replace(")", "").replace("\\", ""))
                   for c in l.split("&")[5:7]] for l in rows], columns=["b", "se"])
assert len(D) == 17
print(f"stacked replication vs published difference column: max |diff| = {np.abs(rep - D.b).max():.4f}")
print(D.assign(q=QLAB).to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = np.arange(17)
ax.axhline(0, color="grey", ls="--", lw=0.8)
ax.fill_between(x, D.b - 1.96 * D.se, D.b + 1.96 * D.se, color="#1f3b5c", alpha=0.15, lw=0)
ax.errorbar(x, D.b, yerr=1.96 * D.se, fmt="o", ms=4.5, color="#1f3b5c", capsize=2.5, lw=1)
ax.set_xticks(x); ax.set_xticklabels(QLAB, rotation=90, fontsize=8)
ax.set_ylabel(r"$\hat\beta_{I,q}-\hat\beta_{C,q}$")
ax.set_xlabel("Quarter")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_fact3_diff.png"), dpi=600)
print("saved fig_fact3_diff.png")
