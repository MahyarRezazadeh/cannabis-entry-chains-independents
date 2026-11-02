import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
r"""
03_fix_figures.py -- updated_draft_with_IO -- regenerate figures/tables that used old samples or groupings

All outputs use the current 368-ZIP crosswalk, the three exposure groups (nonlegal-border /
always-legal-border / interior), the 13-storefront ZIP-snapping correction, and 2022Q2-2026Q2.

  fig_zip_dynamic_main.png   Fact 1-2 dynamic ZIP estimates, nonlegal vs always-legal-border,
                             full zero-inclusive panel (replaces the 116-ZIP fig_nm_dynamic_legalcontrol.png);
                             also re-checks tab_fact1fact2_dynamic.tex coefficient by coefficient.
  fig_disp_dynamic_main.png  dispensary-level revenue per local resident (1 km / 2 km), nonlegal-border vs
                             ALWAYS-LEGAL-border stores (replaces the border-vs-all-others
                             fig_eventstudy_nm_dispensary_revperresident_1km_vs_2km.png)
  tab_disp_static.tex        static dispensary-level estimates, same comparison (replaces the
                             appendix table tab:nm_dispensary_border_best)
  fig_nm_exposure_map.png    map of exposure groups + group means (replaces fig_nm_spatial_inequality.png)
  sf_local_population.csv    2020-block population within 1 km / 2 km of every storefront (area-apportioned)
"""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTRY = str(_P.ROOT)
BUILD = str(_P.BUILD)
STATES = str(_P.REF / "cb_2023_us_state_500k" / "cb_2023_us_state_500k.shp")
BLOCKS = str(_P.BLOCKS / "tl_2020_35_tabblock20" / "tl_2020_35_tabblock20.shp")
ROOT = str(_P.OUT)
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
QLAB = [f"{q // 4}Q{q % 4 + 1}" for q in range(QMIN, QMAX + 1)]
COL = {"nonlegal": "#b2182b", "legal": "#2166ac", "interior": "#8c8c8c"}
NAME = {"nonlegal": "Nonlegal-border", "legal": "Always-legal-border", "interior": "Interior"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def asinh(x):
    return np.arcsinh(x)


def stars(pv):
    return "^{***}" if pv < 0.01 else "^{**}" if pv < 0.05 else "^{*}" if pv < 0.10 else ""


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def dynamic(df, y, group_col, cluster, extra=""):
    d = df.dropna(subset=[y]).copy()
    cols = []
    for i, q in enumerate(range(QMIN, QMAX + 1)):
        c = f"nl_q{i}"
        d[c] = ((d.qt == q) & (d[group_col] == 1)).astype(float)
        if d.loc[d.qt == q, c].sum() > 0 and d.loc[(d.qt == q) & (d[group_col] == 0)].shape[0] > 0:
            cols.append((i, c))
    f = f"{y} ~ " + " + ".join(c for _, c in cols) + " + C(qt)" + extra
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return pd.DataFrame([(QLAB[i], m.params[c], m.bse[c], m.pvalues[c]) for i, c in cols],
                        columns=["q", "b", "se", "p"])


# ============================================================ build ZIP panel (as in 01/02)
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
g = sq.groupby(["zip5", "qt"]).agg(n_stores=("storefront_id", "nunique"), rev=("revenue", "sum")).reset_index()
grid = pd.MultiIndex.from_product([sorted(zx.zip5), range(QMIN, QMAX + 1)], names=["zip5", "qt"]).to_frame(index=False)
full = grid.merge(g, on=["zip5", "qt"], how="left").fillna(0).merge(
    zx[["zip5", "zip_pop", "group", "county_fips", "lat", "lon"]], on="zip5")
full["log_pop"] = np.log(full.zip_pop.clip(lower=1))
full["y_rpc"] = asinh(full.rev / full.zip_pop)
full["y_den"] = asinh(full.n_stores / full.zip_pop * 1000)
full["y_rps"] = np.log((full.rev / full.n_stores).where(full.n_stores > 0))
full["nonlegal"] = (full.group == "nonlegal").astype(int)
main = full[full.group != "interior"].copy()

# ============================================================ 1. ZIP dynamic figure (+ check against table)
OUTC = [("y_rpc", "Revenue per resident (asinh)"), ("y_den", "Dispensary density (asinh)"),
        ("y_rps", "log(revenue per dispensary)")]
res = {y: dynamic(main, y, "nonlegal", "county_fips", " + log_pop") for y, _ in OUTC}
tab = open(os.path.join(ROOT, "tables", "tab_fact1fact2_dynamic.tex"), encoding="utf-8").read()
rows = [l for l in tab.splitlines() if l[:4] == "2022" or l[:4] in ("2023", "2024", "2025", "2026")]
maxdiff = 0.0
for k, l in enumerate(rows):
    cells = [c.strip() for c in l.replace("\\\\", "").split("&")]
    pub = [float(cells[j].replace("*", "")) for j in (1, 3, 5)]
    new = [res[y].b.iloc[k] for y, _ in OUTC]
    maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(pub, new)))
print(f"[check] max |published - recomputed| over 17x3 dynamic ZIP coefficients: {maxdiff:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
x = np.arange(len(QLAB))
for ax, (y, lab) in zip(axes, OUTC):
    r = res[y]
    ax.errorbar(x, r.b, yerr=1.96 * r.se, fmt="o", ms=4, color="#1f3b5c", capsize=2.5, lw=1)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title(lab, fontsize=10, loc="left")
    ax.set_ylabel(r"$\hat\beta_q$")
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(QLAB, rotation=90, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_zip_dynamic_main.png"), dpi=600)
plt.close(fig)

# ============================================================ 2. block population around each storefront
lp_path = str(_P.INTER / "sf_local_population.csv")
if not os.path.exists(lp_path):
    blocks = gpd.read_file(BLOCKS, columns=["POP20"])
    blocks = blocks[blocks.POP20 > 0].to_crs("EPSG:26913")
    blocks["barea"] = blocks.geometry.area
    pts = gpd.GeoDataFrame(sf[["storefront_id"]].copy(), geometry=gpd.points_from_xy(sf.lon, sf.lat),
                           crs="EPSG:4326").to_crs("EPSG:26913")
    out = sf[["storefront_id"]].copy()
    for tag, r in [("1km", 1000.0), ("2km", 2000.0)]:
        buf = pts.copy(); buf["geometry"] = buf.geometry.buffer(r)
        inter = gpd.overlay(buf, blocks, how="intersection", keep_geom_type=True)
        inter["pop"] = inter.POP20 * (inter.geometry.area / inter.barea).clip(upper=1)
        out[f"pop_{tag}"] = out.storefront_id.map(inter.groupby("storefront_id")["pop"].sum()).fillna(0).round(1)
    out.to_csv(lp_path, index=False)
lp = pd.read_csv(lp_path)
print(f"[local pop] storefronts: {len(lp)}; with >=100 residents within 1 km: {(lp.pop_1km >= 100).sum()}; "
      f"with >=250 within 2 km: {(lp.pop_2km >= 250).sum()}")

# ============================================================ 3. dispensary panel, nonlegal vs always-legal-border stores
ps = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), dtype={"zip5": str})
ps = ps[ps.qt.between(QMIN, QMAX) & (ps.active == 1)].copy()
ps["grp"] = np.select([ps.border_sf == 1, (ps.placebo_border_sf == 1) & (ps.border_sf == 0)],
                      ["nonlegal", "legal"], default="interior")
ps = ps.merge(lp, on="storefront_id", how="left")
ps["county"] = ps.storefront_id.map(dict(zip(sf.storefront_id, sf.zip5))).map(dict(zip(zx.zip5, zx.county_fips))).astype(str)
ps["nonlegal"] = (ps.grp == "nonlegal").astype(int)
ps["brand"] = ps.brand_id.astype(str)
ps["chain"] = ps.brand_id.map(dict(zip(bm.brand_id, (bm.n_storefronts >= 2).astype(int))))
dm = ps[ps.grp != "interior"].copy()
print(f"[disp] nonlegal vs legal-border stores: {dm.storefront_id.nunique()} stores "
      f"({dm.groupby('grp').storefront_id.nunique().to_dict()}), {len(dm)} store-quarters, {dm.county.nunique()} counties")

def prep(d, popcol, floor):
    d = d[d[popcol] >= floor].copy()
    d["y"] = np.log(d.revenue / d[popcol])
    lo, hi = d.y.quantile([0.01, 0.99]); d["y"] = d.y.clip(lo, hi)
    d["log_lp"] = np.log(d[popcol])
    return d

d1, d2 = prep(dm, "pop_1km", 100), prep(dm, "pop_2km", 250)
r1 = dynamic(d1, "y", "nonlegal", "county", " + log_lp")
r2 = dynamic(d2, "y", "nonlegal", "county", " + log_lp")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = np.arange(len(QLAB))
ix1 = [QLAB.index(q) for q in r1.q]; ix2 = [QLAB.index(q) for q in r2.q]
ax.errorbar(np.array(ix1) - 0.12, r1.b, yerr=1.96 * r1.se, fmt="o", ms=4, capsize=2.5, lw=1,
            color="#1f3b5c", label="Residents within 1 km")
ax.errorbar(np.array(ix2) + 0.12, r2.b, yerr=1.96 * r2.se, fmt="s", ms=3.5, capsize=2.5, lw=1,
            color="#b2182b", label="Residents within 2 km")
ax.axhline(0, color="grey", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(QLAB, rotation=90, fontsize=8)
ax.set_ylabel(r"$\hat\beta_q$, log revenue per local resident")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_disp_dynamic_main.png"), dpi=600)
plt.close(fig)
print(f"[disp dyn 1km] quarters p<.05: {(r1.p < .05).sum()} of {len(r1)}; range {r1.b.min():+.3f} to {r1.b.max():+.3f}")
print(f"[disp dyn 2km] quarters p<.05: {(r2.p < .05).sum()} of {len(r2)}; range {r2.b.min():+.3f} to {r2.b.max():+.3f}")

# static table: log revenue, log revenue per 1 km / 2 km resident; without and with brand FE
dm["y"] = np.log(dm.revenue); d0 = dm.copy(); d0["log_lp"] = 0.0
spec = [("log(Revenue)", d0, ""), ("log(Revenue per resident within 1 km)", d1, " + log_lp"),
        ("log(Revenue per resident within 2 km)", d2, " + log_lp")]
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{No brand fixed effects} & \multicolumn{2}{c}{Brand fixed effects} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Outcome & Coef. & Obs. & Coef. & Obs. \\", r"\midrule"]
for lab, d, ex in spec:
    cells = [lab]; ses = [""]
    for fe in ["", " + C(brand)"]:
        m = smf.ols("y ~ nonlegal + C(qt)" + ex + fe, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.county})
        cells += [f"${m.params['nonlegal']:+.3f}{stars(m.pvalues['nonlegal'])}$", f"{int(m.nobs):,}"]
        ses += [f"({m.bse['nonlegal']:.3f})", ""]
        print(f"[disp static] {lab}{' brandFE' if fe else ''}: b={m.params['nonlegal']:+.3f} se={m.bse['nonlegal']:.3f} "
              f"p={m.pvalues['nonlegal']:.3f} N={int(m.nobs)}")
    L += [" & ".join(cells) + r" \\", " & ".join(ses) + r" \\"]
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_disp_static.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ============================================================ 4. exposure map + group means
st = gpd.read_file(STATES).to_crs("EPSG:4326")
zm = full.groupby(["zip5", "group", "lat", "lon"]).agg(den=("n_stores", "sum"), pop=("zip_pop", "sum")).reset_index()
zm["den"] = zm.den / zm["pop"] * 1000
fig = plt.figure(figsize=(10.5, 5.4))
ax = fig.add_axes([0.0, 0.10, 0.50, 0.82])
nb = {"TX": "#f4d6d6", "OK": "#f4d6d6", "UT": "#f4d6d6", "CO": "#d6e2f0", "AZ": "#d6e2f0"}
for ab, c in nb.items():
    st[st.STUSPS == ab].plot(ax=ax, color=c, edgecolor="white", lw=0.5)
st[st.STUSPS == "NM"].plot(ax=ax, color="#f7f7f7", edgecolor="black", lw=0.8)
for gk in ["interior", "legal", "nonlegal"]:
    s = zm[zm.group == gk]
    ax.scatter(s.lon, s.lat, s=6 + 60 * np.sqrt(s.den.clip(upper=4)), color=COL[gk], alpha=0.7,
               edgecolor="white", lw=0.3, label=NAME[gk])
for ab, (x0, y0) in {"TX": (-102.7, 33.2), "OK": (-102.6, 36.75), "UT": (-109.6, 37.4),
                     "CO": (-106.0, 37.45), "AZ": (-109.6, 34.0)}.items():
    ax.text(x0, y0, ab, fontsize=9, ha="center", color="#444444")
ax.set_xlim(-110.2, -102.4); ax.set_ylim(31.2, 37.7); ax.set_axis_off()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8.5, markerscale=0.8)
ax.set_title("(a) ZIP codes by exposure group", fontsize=10, loc="left")

ax2 = fig.add_axes([0.62, 0.16, 0.27, 0.72])
gs = full.groupby("group").agg(st=("n_stores", "sum"), rev=("rev", "sum"), pop=("zip_pop", "sum"))
order = ["nonlegal", "legal", "interior"]
den = [gs.loc[k, "st"] / gs.loc[k, "pop"] * 1000 for k in order]
rpc = [gs.loc[k, "rev"] / gs.loc[k, "pop"] for k in order]
xx = np.arange(3); w = 0.38
ax2.bar(xx - w / 2, den, w, color=[COL[k] for k in order])
ax2.set_ylabel("Dispensaries per 1,000 residents (solid)")
ax3 = ax2.twinx()
ax3.bar(xx + w / 2, rpc, w, color="white", edgecolor=[COL[k] for k in order], hatch="///")
ax3.set_ylabel("Revenue per resident, \\$/quarter (hatched)")
ax3.spines["top"].set_visible(False)
ax2.set_xticks(xx); ax2.set_xticklabels(["Nonlegal-\nborder", "Always-legal-\nborder", "Interior"], fontsize=9)
ax2.set_title("(b) Population-weighted means", fontsize=10, loc="left")
fig.savefig(os.path.join(ROOT, "figures", "fig_nm_exposure_map.png"), dpi=600)
plt.close(fig)
for k, a, b in zip(order, den, rpc):
    print(f"[map b] {k}: density={a:.3f} per 1,000, revenue per resident=${b:.1f}/qtr (pop-weighted)")
print("done")
