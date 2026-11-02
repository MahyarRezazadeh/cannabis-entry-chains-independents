r"""
04_entry_by_type.py -- updated_draft_with_IO -- entry by operator type, nonlegal- vs always-legal-border

Three tests on the main comparison (92 nonlegal-border + 117 always-legal-border ZIPs, 2022Q2-2026Q2):
  T1  established multi-store operators opening an additional storefront (all; and 2nd storefront only)
  T2  single-store operators (brand never has a 2nd NM storefront)
  T3  first locations of operators that later open a 2nd storefront
Status at entry uses the identical algorithm to 02_data_section_tables.py / 18_zip_structural_model.py.
Entries = storefront's first active quarter; storefronts operating at market opening (left-censored)
are not entries.

Specifications (all with calendar-quarter FE, SEs clustered by county):
  (a) PPML, ZIP-quarter entry count, log ZIP population control
  (b) OLS, ZIP-quarter entries per 100,000 residents
  (c) PPML, ZIP total entries over the sample (cross-section), log population
  (d) Composition: LPM on entrants, 1{type} on nonlegal, log ZIP pop, quarter FE
  (e) Stock: active storefronts of each type per 1,000 residents (ZIP-quarter OLS)
Output: entry_by_type_results.txt and tab_entry_by_type.tex
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BUILD = str(_P.BUILD)
ROOT = str(_P.OUT)
EARTH_R = 3958.8
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
LOG = []


def p(s=""):
    print(s); LOG.append(str(s))


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


def stars(pv):
    return "^{***}" if pv < 0.01 else "^{**}" if pv < 0.05 else "^{*}" if pv < 0.10 else ""


# ---------------------------------------------------------------- build
zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})
zx["group"] = np.select([zx.border_z == 1, (zx.placebo_border_z == 1) & (zx.border_z == 0)],
                        ["nonlegal", "legal"], default="interior")
sf = pd.read_csv(os.path.join(BUILD, "storefronts.csv"), dtype={"zip5": str})
xw = pd.read_csv(os.path.join(BUILD, "brand_crosswalk.csv"))
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
sf = sf.merge(xw[["storefront_id", "brand_id"]], on="storefront_id", how="left")
bad = ~sf.zip5.isin(set(zx.zip5))
for i in sf.index[bad]:
    sf.at[i, "zip5"] = zx.zip5.to_numpy()[np.argmin(hav(sf.at[i, "lat"], sf.at[i, "lon"],
                                                        zx.lat.to_numpy(), zx.lon.to_numpy()))]

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
        rows.append((r.storefront_id, "established" if has else ("first_loc" if r.n_storefronts >= 2 else "single"),
                     int(prior + (0 if r.lc else 0))))
st = pd.DataFrame(rows, columns=["storefront_id", "status", "n_prior"])
meta = meta.merge(st, on="storefront_id").merge(sf[["storefront_id", "zip5"]], on="storefront_id")
# 2nd storefront of the brand in NM (rank 2 by first active quarter; ties broken by id)
meta = meta.sort_values(["brand_id", "faq", "storefront_id"])
meta["rank"] = meta.groupby("brand_id").cumcount() + 1
meta["second"] = ((meta.status == "established") & (meta["rank"] == 2)).astype(int)
meta = meta.merge(zx[["zip5", "group"]], on="zip5")
p(f"Status (all storefronts): {meta.status.value_counts().to_dict()}")
ent = meta[~meta.lc].copy()
p(f"Entries (not operating at market opening): {len(ent)}; by status {ent.status.value_counts().to_dict()}")
p("Entries by group x status:")
p(pd.crosstab(ent.group, ent.status, margins=True).to_string())
p(f"2nd-storefront entries by group: {ent[ent.second == 1].group.value_counts().to_dict()}")

TYPES = [("established", "Established operator, additional store"),
         ("second", "  of which: operator's 2nd store"),
         ("single", "Single-store operator"),
         ("first_loc", "First location of later-expanding operator"),
         ("all", "All entries")]


def flag(df, t):
    if t == "all":
        return pd.Series(1, index=df.index)
    if t == "second":
        return df.second
    return (df.status == t).astype(int)


# ---------------------------------------------------------------- ZIP-quarter panel (main sample)
mz = zx[zx.group.isin(["nonlegal", "legal"])][["zip5", "group", "zip_pop", "county_fips"]].copy()
grid = pd.MultiIndex.from_product([mz.zip5, range(QMIN, QMAX + 1)], names=["zip5", "qt"]).to_frame(index=False)
panel = grid.merge(mz, on="zip5")
for t, _ in TYPES:
    e = ent.assign(f=flag(ent, t)).groupby(["zip5", "faq"]).f.sum().rename(f"e_{t}").reset_index().rename(columns={"faq": "qt"})
    panel = panel.merge(e, on=["zip5", "qt"], how="left")
    panel[f"e_{t}"] = panel[f"e_{t}"].fillna(0)
panel["nonlegal"] = (panel.group == "nonlegal").astype(int)
panel["lpop"] = np.log(panel.zip_pop)
panel["qtc"] = panel.qt.astype(str)
p(f"\nMain panel: {panel.zip5.nunique()} ZIPs, {len(panel)} ZIP-quarters, {panel.county_fips.nunique()} counties")

# stock: active storefronts by status in each ZIP-quarter
act = ps[ps.active == 1][["storefront_id", "qt"]].merge(meta[["storefront_id", "status", "zip5"]], on="storefront_id")
for t in ["established", "single", "first_loc"]:
    s = act[act.status == t].groupby(["zip5", "qt"]).size().rename(f"s_{t}").reset_index()
    panel = panel.merge(s, on=["zip5", "qt"], how="left")
    panel[f"s_{t}"] = panel[f"s_{t}"].fillna(0)
    panel[f"sd_{t}"] = panel[f"s_{t}"] / panel.zip_pop * 1000

res = {}


def ppml(y, d, x="nonlegal + lpop + C(qtc)"):
    m = smf.glm(f"{y} ~ {x}", data=d, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.county_fips)[0]})
    return m.params["nonlegal"], m.bse["nonlegal"], m.pvalues["nonlegal"]


def ols(y, d, x="nonlegal + lpop + C(qtc)"):
    m = smf.ols(f"{y} ~ {x}", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.county_fips)[0]})
    return m.params["nonlegal"], m.bse["nonlegal"], m.pvalues["nonlegal"]


cs = panel.groupby(["zip5", "nonlegal", "lpop", "county_fips", "zip_pop"])[[f"e_{t}" for t, _ in TYPES]].sum().reset_index()
for t, lab in TYPES:
    y = f"e_{t}"
    panel[f"r_{t}"] = panel[y] / panel.zip_pop * 1e5
    nl, lg = panel.loc[panel.nonlegal == 1, y].sum(), panel.loc[panel.nonlegal == 0, y].sum()
    rnl = nl / mz.loc[mz.group == "nonlegal", "zip_pop"].sum() * 1e5
    rlg = lg / mz.loc[mz.group == "legal", "zip_pop"].sum() * 1e5
    a = ppml(y, panel); b = ols(f"r_{t}", panel); c = ppml(y, cs, "nonlegal + lpop")
    res[t] = dict(nl=nl, lg=lg, rnl=rnl, rlg=rlg, a=a, b=b, c=c)
    p(f"\n[{t}] entries NL={int(nl)} LB={int(lg)}; per 100k residents over sample NL={rnl:.2f} LB={rlg:.2f}")
    p(f"  (a) PPML ZIP-qtr      : {a[0]:+.3f} (SE {a[1]:.3f}) p={a[2]:.3f}  IRR={np.exp(a[0]):.2f}")
    p(f"  (b) OLS per 100k      : {b[0]:+.3f} (SE {b[1]:.3f}) p={b[2]:.3f}  mean LB={panel.loc[panel.nonlegal==0, f'r_{t}'].mean():.3f}")
    p(f"  (c) PPML ZIP totals   : {c[0]:+.3f} (SE {c[1]:.3f}) p={c[2]:.3f}")

# composition among entrants in main ZIPs
ce = ent[ent.group.isin(["nonlegal", "legal"])].merge(mz[["zip5", "zip_pop", "county_fips"]], on="zip5")
ce["nonlegal"] = (ce.group == "nonlegal").astype(int); ce["lpop"] = np.log(ce.zip_pop); ce["qtc"] = ce.faq.astype(str)
p(f"\nComposition of entrants (n={len(ce)}; NL={int(ce.nonlegal.sum())}, LB={int((1-ce.nonlegal).sum())})")
for t, lab in TYPES[:4]:
    ce[f"d_{t}"] = flag(ce, t)
    d = ols(f"d_{t}", ce)
    res[t]["d"] = d
    res[t]["shnl"] = ce.loc[ce.nonlegal == 1, f"d_{t}"].mean(); res[t]["shlg"] = ce.loc[ce.nonlegal == 0, f"d_{t}"].mean()
    p(f"  share {t}: NL={res[t]['shnl']:.3f} LB={res[t]['shlg']:.3f}; LPM diff {d[0]:+.3f} (SE {d[1]:.3f}) p={d[2]:.3f}")

# stock density by status
p("\nActive storefronts per 1,000 residents by status (ZIP-quarter OLS, log pop + quarter FE):")
for t in ["established", "single", "first_loc"]:
    e = ols(f"sd_{t}", panel)
    res[t]["e"] = e
    p(f"  {t}: NL mean={panel.loc[panel.nonlegal==1, f'sd_{t}'].mean():.3f} LB mean={panel.loc[panel.nonlegal==0, f'sd_{t}'].mean():.3f}; "
      f"diff {e[0]:+.3f} (SE {e[1]:.3f}) p={e[2]:.3f}")

# robustness: drop Albuquerque-metro? (none are in main groups by construction) -> exclude 2022Q2 entries
p("\nRobustness: drop entries in the first sample quarter (2022Q2)")
pr = panel[panel.qt > QMIN]
for t, _ in TYPES:
    a = ppml(f"e_{t}", pr)
    p(f"  [{t}] PPML {a[0]:+.3f} (SE {a[1]:.3f}) p={a[2]:.3f}")

p("\nPopulation-weighted OLS, entries per 100k residents:")
for t, _ in TYPES:
    m = smf.wls(f"r_{t} ~ nonlegal + lpop + C(qtc)", data=panel, weights=panel.zip_pop).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(panel.county_fips)[0]})
    res[t]["w"] = (m.params["nonlegal"], m.bse["nonlegal"], m.pvalues["nonlegal"])
    p(f"  [{t}] {m.params['nonlegal']:+.3f} (SE {m.bse['nonlegal']:.3f}) p={m.pvalues['nonlegal']:.3f}")

p("\nStacked PPML: does the nonlegal-border effect differ by type? (base = single)")
long = pd.concat([panel.assign(y=panel[f"e_{t}"], typ=t) for t in ["single", "established", "first_loc"]])
m = smf.glm("y ~ C(typ, Treatment('single'))*nonlegal + C(typ):lpop + C(typ):C(qtc)", data=long,
            family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(long.county_fips)[0]})
for k in m.params.index:
    if "nonlegal" in k:
        p(f"  {k}: {m.params[k]:+.3f} (SE {m.bse[k]:.3f}) p={m.pvalues[k]:.3f}")

p("\nLeave out the county with the most entries in each group (PPML):")
big = panel.groupby(["group", "county_fips"]).e_all.sum().sort_values().groupby(level=0).tail(1)
for (g_, c_) in big.index:
    d = panel[panel.county_fips != c_]
    for t, _ in TYPES[:4]:
        a = ppml(f"e_{t}", d)
        p(f"  drop {g_} county {c_} [{t}]: {a[0]:+.3f} (SE {a[1]:.3f}) p={a[2]:.3f}")

open(os.path.join(ROOT, "logs", "entry_by_type_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")

# ---------------------------------------------------------------- LaTeX table
def fmt(t3, d=3):
    return f"${t3[0]:.{d}f}{stars(t3[2])}$", f"({t3[1]:.{d}f})"


L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Entries per 100k res.} & PPML & \multicolumn{2}{c}{OLS, entries per 100k} & Share of \\",
     r" & NL & LB & ZIP-qtr & unweighted & pop.-weighted & entrants \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
     r" & (1) & (2) & (3) & (4) & (5) & (6) \\", r"\midrule"]
for t, lab in TYPES:
    r = res[t]
    a1, a2 = fmt(r["a"]); b1, b2 = fmt(r["b"]); w1, w2 = fmt(r["w"])
    d1, d2 = fmt(r["d"]) if "d" in r else ("", "")
    L.append(f"{lab} & {r['rnl']:.1f} & {r['rlg']:.1f} & {a1} & {b1} & {w1} & {d1} \\\\")
    L.append(f" & [{int(r['nl'])}] & [{int(r['lg'])}] & {a2} & {b2} & {w2} & {d2} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_entry_by_type.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
