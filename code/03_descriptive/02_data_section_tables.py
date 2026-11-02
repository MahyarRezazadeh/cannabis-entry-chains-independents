import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
r"""
02_data_section_tables.py -- updated_draft_with_IO -- Data and Background section tables

Builds the descriptive tables for Section "Data and Background" from the SAME build files and
the SAME definitions as 01_build_facts123.py (which produces the Fact 1-3 estimates):
  * 368-ZIP crosswalk, nonlegal / always-legal / interior groups (92 / 117 / 159)
  * 13-storefront ZIP-snapping correction (only invalid raw ZIPs are reassigned)
  * quarters 2022Q2-2026Q2 (17), storefront active in a quarter iff revenue > 0
  * Fact-3 composition outcome = number of DISTINCT independent / chain operators (brands)
    active in the ZIP-quarter; chain = brand with >= 2 NM storefronts over the sample
  * at-entry status (used by the entry game) = identical algorithm to 18_zip_structural_model.py

Outputs (written next to the draft, \input-ready):
  tab_data_sumstats_zip.tex     ZIP-quarter panel, by exposure group, mean [SD] + NL-LB difference
  tab_data_sumstats_disp.tex    dispensary-quarter panel, by own-location exposure group
  tab_data_samples.tex          sample construction: units, observations, where used
  data_section_numbers.txt      every number quoted in the Data section text
"""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

BUILD = str(_P.BUILD)
NMP = str(_P.ZBP)
ROOT = str(_P.OUT)

ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}
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


# ------------------------------------------------------------------ build (as in 01)
zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})
zx["group"] = np.select([zx.border_z == 1, (zx.placebo_border_z == 1) & (zx.border_z == 0)],
                        ["nonlegal", "legal"], default="interior")
assert (zx.group == "nonlegal").sum() == 92 and (zx.group == "legal").sum() == 117
both = int(((zx.border_z == 1) & (zx.placebo_border_z == 1)).sum())
p(f"ZIPs qualifying on both rules (assigned nonlegal): {both}")

sf = pd.read_csv(os.path.join(BUILD, "storefronts.csv"), dtype={"zip5": str})
xw = pd.read_csv(os.path.join(BUILD, "brand_crosswalk.csv"))
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
sf = sf.merge(xw[["storefront_id", "brand_id"]], on="storefront_id", how="left")
sf["chain"] = sf["brand_id"].map(dict(zip(bm.brand_id, (bm.n_storefronts >= 2).astype(int))))
valid = set(zx.zip5)
bad = ~sf.zip5.isin(valid)
for i in sf.index[bad]:
    sf.at[i, "zip5"] = zx.zip5.to_numpy()[np.argmin(hav(sf.at[i, "lat"], sf.at[i, "lon"],
                                                        zx.lat.to_numpy(), zx.lon.to_numpy()))]
p(f"Storefronts: {len(sf)}; ZIP-snapped: {int(bad.sum())}; brands: {sf.brand_id.nunique()}; "
  f"chain brands (>=2 NM storefronts): {int(bm.loc[bm.brand_id.isin(sf.brand_id), 'n_storefronts'].ge(2).sum())}; "
  f"storefronts of chain brands: {int(sf.chain.sum())}")

sfm = pd.read_csv(os.path.join(BUILD, "sf_month_revenue.csv"))
sfm["qt"] = sfm.year * 4 + ((sfm.month - 1) // 3)
sq = (sfm[sfm.qt.between(QMIN, QMAX)].groupby(["storefront_id", "qt"])
      .agg(revenue=("total", "sum")).reset_index())
sq = sq[sq.revenue > 0].merge(sf[["storefront_id", "zip5", "brand_id", "chain", "city"]], on="storefront_id")
p(f"Storefronts ever active 2022Q2-2026Q2: {sq.storefront_id.nunique()}; active storefront-quarters: {len(sq)}")

g = (sq.groupby(["zip5", "qt"]).apply(lambda s: pd.Series({
        "n_stores": s.storefront_id.nunique(), "rev": s.revenue.sum(),
        "n_indep": s.loc[s.chain == 0, "brand_id"].nunique(),
        "n_chain": s.loc[s.chain == 1, "brand_id"].nunique()})).reset_index())
grid = pd.MultiIndex.from_product([sorted(zx.zip5), range(QMIN, QMAX + 1)], names=["zip5", "qt"]).to_frame(index=False)
full = grid.merge(g, on=["zip5", "qt"], how="left").fillna(0)
full = full.merge(zx[["zip5", "zip_pop", "group", "county_fips", "nonleg_pop_75", "dist_nonleg_mi",
                      "dist_legalneigh_mi", "legalneigh_pop_75"]], on="zip5")
full["rev_pc"] = full.rev / full.zip_pop
full["density"] = full.n_stores / full.zip_pop * 1000
full["rev_per_store"] = (full.rev / full.n_stores).where(full.n_stores > 0) / 1000
full["any"] = (full.n_stores > 0).astype(float)
full["rev_k"] = full.rev / 1000
full["nlpop_k"] = full.nonleg_pop_75 / 1000
full["lgpop_k"] = full.legalneigh_pop_75 / 1000
full["nonlegal"] = (full.group == "nonlegal").astype(float)
assert len(full) == 6256

ever = full.groupby("zip5").n_stores.max().gt(0)
p(f"ZIPs with >=1 dispensary in some quarter: {int(ever.sum())} "
  f"(by group: {full[full.zip5.isin(ever[ever].index)].groupby('group').zip5.nunique().to_dict()})")
p(f"ZIP-quarters with >=1 dispensary: {int(full['any'].sum())} "
  f"(by group: {full.groupby('group')['any'].sum().astype(int).to_dict()})")
p(f"Population: zero-pop ZIPs={int((zx.zip_pop<=0).sum())}; min pop={zx.zip_pop.min()}")

main = full[full.group.isin(["nonlegal", "legal"])]
p(f"Main sample: {main.zip5.nunique()} ZIPs, {len(main)} ZIP-quarters, {main.county_fips.nunique()} counties; "
  f"pooled: {full.county_fips.nunique()} counties")
p(f"Main sample revenue-per-store obs (ZIP-quarters with >=1 store): {int(main['any'].sum())}; "
  f"pooled: {int(full['any'].sum())}")
p(f"Share of zero ZIP-quarters: main {1-main['any'].mean():.3f}, pooled {1-full['any'].mean():.3f}")

# conditional-on-activity sample (01, Part C: quarters from ZIP's own first active quarter on, store>0)
zfq = full[full['any'] == 1].groupby("zip5").qt.min()
cond = full[full.zip5.isin(zfq.index) & (full['any'] == 1)]
cond = cond[cond.qt >= cond.zip5.map(zfq)]
p(f"Conditional sample: {cond.zip5.nunique()} ZIPs, {len(cond)} ZIP-quarters; main groups only: "
  f"{cond[cond.group!='interior'].zip5.nunique()} ZIPs, {len(cond[cond.group!='interior'])} ZIP-quarters")
mage = (full.qt - full.zip5.map(zfq))
full["mkt_age"] = mage.where(mage >= 0)

# ------------------------------------------------------------------ Table A: ZIP panel
VARS = [("A", "Revenue per resident (\\$/quarter)", "rev_pc", "{:,.1f}"),
        ("A", "Dispensaries per 1,000 residents", "density", "{:.3f}"),
        ("A", "Revenue per dispensary (\\$000/quarter)$^{a}$", "rev_per_store", "{:,.1f}"),
        ("A", "Independent operators (count)", "n_indep", "{:.2f}"),
        ("A", "Chain operators (count)", "n_chain", "{:.2f}"),
        ("B", "Dispensaries (count)", "n_stores", "{:.2f}"),
        ("B", "Any dispensary (share of ZIP-quarters)", "any", "{:.3f}"),
        ("B", "Total revenue (\\$000/quarter)", "rev_k", "{:,.1f}"),
        ("B", "Market age (quarters since first dispensary)$^{a}$", "mkt_age", "{:.1f}"),
        ("C", "Resident population", "zip_pop", "{:,.0f}"),
        ("C", "Nonlegal-state residents within 75 mi (000s)", "nlpop_k", "{:,.1f}"),
        ("C", "Legal-state residents within 75 mi (000s)", "lgpop_k", "{:,.1f}"),
        ("C", "Distance to nearest nonlegal-state line (mi)", "dist_nonleg_mi", "{:,.1f}"),
        ("C", "Distance to nearest legal-state line (mi)", "dist_legalneigh_mi", "{:,.1f}")]
PANELS = {"A": "Panel A: Outcomes", "B": "Panel B: Market activity", "C": "Panel C: Market characteristics (time-invariant)"}
GROUPS = [("nonlegal", "Nonlegal-border"), ("legal", "Always-legal-border"), ("interior", "Interior")]


def cell(x, f):
    x = x.dropna()
    return f.format(x.mean()), "[" + f.format(x.std()) + "]"


def diff(var, f):
    d = main.copy() if var not in ("mkt_age",) else full[full.group.isin(["nonlegal", "legal"])].copy()
    d = d.dropna(subset=[var])
    m = smf.ols(f"{var} ~ nonlegal", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.county_fips})
    return f.format(m.params["nonlegal"]) + stars(m.pvalues["nonlegal"]), "(" + f.format(m.bse["nonlegal"]) + ")"


main = full[full.group.isin(["nonlegal", "legal"])]
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & Nonlegal- & Always-legal- & Difference & Interior \\",
     r" & border & border & (1)$-$(2) & \\",
     r" & (1) & (2) & (3) & (4) \\", r"\midrule"]
cur = None
for pan, lab, var, f in VARS:
    if pan != cur:
        if cur is not None:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{5}{@{}l}{\textit{" + PANELS[pan] + r"}} \\")
        cur = pan
    top, bot = [lab], [""]
    for gk, _ in GROUPS[:2]:
        a, b = cell(full.loc[full.group == gk, var], f); top.append(a); bot.append(b)
    a, b = diff(var, f); top.append("$" + a.replace(",", "{,}") + "$"); bot.append(b)
    a, b = cell(full.loc[full.group == "interior", var], f); top.append(a); bot.append(b)
    L.append(" & ".join(top) + r" \\")
    L.append(" & ".join(bot) + r" \\")
L.append(r"\midrule")
cnt = lambda gk, fn: fn(full[full.group == gk])
L.append("ZIP codes & " + " & ".join([str(cnt("nonlegal", lambda d: d.zip5.nunique())),
                                       str(cnt("legal", lambda d: d.zip5.nunique())), "",
                                       str(cnt("interior", lambda d: d.zip5.nunique()))]) + r" \\")
L.append("ZIP codes with a dispensary in any quarter & " + " & ".join(
    [str(int(ever[full[full.group == gk].zip5.unique()].sum())) if gk else "" for gk in ["nonlegal", "legal", None, "interior"]]) + r" \\")
L.append("ZIP-quarters & " + " & ".join([f"{len(full[full.group=='nonlegal']):,}", f"{len(full[full.group=='legal']):,}",
                                         "", f"{len(full[full.group=='interior']):,}"]) + r" \\")
L.append("ZIP-quarters with $\\geq$1 dispensary & " + " & ".join(
    [f"{int(full.loc[full.group==gk,'any'].sum()):,}" if gk else "" for gk in ["nonlegal", "legal", None, "interior"]]) + r" \\")
L.append("Counties & " + " & ".join([str(cnt("nonlegal", lambda d: d.county_fips.nunique())),
                                      str(cnt("legal", lambda d: d.county_fips.nunique())), "",
                                      str(cnt("interior", lambda d: d.county_fips.nunique()))]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_data_sumstats_zip.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

for gk, _ in GROUPS:
    s = full[full.group == gk]
    p(f"[{gk}] rev_pc={s.rev_pc.mean():.1f} density={s.density.mean():.3f} rps={s.rev_per_store.mean():.1f} "
      f"indep={s.n_indep.mean():.2f} chain={s.n_chain.mean():.2f} any={s['any'].mean():.3f} pop={s.zip_pop.mean():,.0f} "
      f"median pop={s.zip_pop.median():,.0f}")

# ------------------------------------------------------------------ at-entry status (as in 18_)
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
st = pd.DataFrame(rows, columns=["storefront_id", "status"])
p(f"At-entry status (storefronts active 2022Q2-2026Q2): {st.status.value_counts().to_dict()}; "
  f"left-censored (operating at market opening): {int(meta.lc.sum())}")

# ------------------------------------------------------------------ Table B: dispensary panel
d = ps[ps.active == 1].copy()
d["grp"] = np.select([d.border_sf == 1, (d.placebo_border_sf == 1) & (d.border_sf == 0)],
                     ["nonlegal", "legal"], default="interior")
d = d.merge(st, on="storefront_id", how="left")
# 1-km / 2-km area-apportioned 2020-block population for every storefront (built by 03_fix_figures.py)
lp = pd.read_csv(str(_P.INTER / "sf_local_population.csv")).rename(
    columns={"pop_1km": "pop_1km_areal", "pop_2km": "pop_2km_areal"})
d = d.merge(lp, on="storefront_id", how="left")
p(f"Dispensary panel: {d.storefront_id.nunique()} storefronts, {len(d)} storefront-quarters; "
  f"1-km population matched for {d.drop_duplicates('storefront_id').pop_1km_areal.notna().mean():.3f} of storefronts")
d["rev_k"] = d.revenue / 1000
d["rev_1km"] = (d.revenue / d.pop_1km_areal).where(d.pop_1km_areal >= 100)
d["rev_2km"] = (d.revenue / d.pop_2km_areal).where(d.pop_2km_areal >= 250)
d["chainb"] = d.brand_id.map(dict(zip(bm.brand_id, (bm.n_storefronts >= 2).astype(float))))
d["estab"] = (d.status == "established").astype(float)
d["lc"] = d.left_censored.astype(str).isin(["True", "1", "1.0", "true"]).astype(float)
DV = [("Revenue (\\$000/quarter)", "rev_k", "{:,.1f}"),
      ("Revenue per resident within 1 km (\\$)$^{b}$", "rev_1km", "{:,.1f}"),
      ("Revenue per resident within 2 km (\\$)$^{b}$", "rev_2km", "{:,.1f}"),
      ("Residents within 1 km", "pop_1km_areal", "{:,.0f}"),
      ("Dispensaries within 25 mi", "n_disp_25mi", "{:,.1f}"),
      ("Dispensary age (quarters since first sale)", "age_q", "{:.1f}"),
      ("Chain operator (share)", "chainb", "{:.3f}"),
      ("Established multi-store operator at entry (share)", "estab", "{:.3f}"),
      ("Operating at market opening (share)", "lc", "{:.3f}")]
dm = d[d.grp.isin(["nonlegal", "legal"])].copy()
dm["nonlegal"] = (dm.grp == "nonlegal").astype(float)
# cluster by county of the storefront's (ZIP-snapping-corrected) ZIP, as in the ZIP panel
dm["cl"] = dm.storefront_id.map(dict(zip(sf.storefront_id, sf.zip5))).map(
    dict(zip(zx.zip5, zx.county_fips))).astype(str)
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & Nonlegal- & Always-legal- & Difference & Interior \\",
     r" & border & border & (1)$-$(2) & \\", r" & (1) & (2) & (3) & (4) \\", r"\midrule"]
for lab, var, f in DV:
    top, bot = [lab], [""]
    for gk in ["nonlegal", "legal"]:
        a, b = cell(d.loc[d.grp == gk, var], f); top.append(a); bot.append(b)
    q = dm.dropna(subset=[var])
    m = smf.ols(f"{var} ~ nonlegal", data=q).fit(cov_type="cluster", cov_kwds={"groups": q.cl})
    top.append("$" + f.format(m.params["nonlegal"]).replace(",", "{,}") + stars(m.pvalues["nonlegal"]) + "$")
    bot.append("(" + f.format(m.bse["nonlegal"]) + ")")
    a, b = cell(d.loc[d.grp == "interior", var], f); top.append(a); bot.append(b)
    L += [" & ".join(top) + r" \\", " & ".join(bot) + r" \\"]
L.append(r"\midrule")
L.append("Dispensaries & " + " & ".join([str(d[d.grp == "nonlegal"].storefront_id.nunique()),
                                          str(d[d.grp == "legal"].storefront_id.nunique()), "",
                                          str(d[d.grp == "interior"].storefront_id.nunique())]) + r" \\")
L.append("Dispensary-quarters & " + " & ".join([f"{(d.grp=='nonlegal').sum():,}", f"{(d.grp=='legal').sum():,}", "",
                                                 f"{(d.grp=='interior').sum():,}"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_data_sumstats_disp.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
p(f"Dispensary panel by group: {d.groupby('grp').storefront_id.nunique().to_dict()} storefronts; "
  f"{d.groupby('grp').size().to_dict()} storefront-quarters; brands {d.brand_id.nunique()}")
p(f"  rev_1km non-missing obs: {int(d.rev_1km.notna().sum())}, storefronts {d.loc[d.rev_1km.notna(),'storefront_id'].nunique()}")

# ------------------------------------------------------------------ town panel size (as in 01, Part D)
cs = pd.read_csv(os.path.join(BUILD, "city_cross_section.csv"))
ct = cs[(cs.is_abq == 0)].dropna(subset=["city_pop"])
ct["grp"] = np.select([ct.border_city == 1, (ct.placebo_border_city == 1) & (ct.border_city == 0)],
                      ["nonlegal", "legal"], default="interior")
p(f"Town panel (ex-ABQ): {len(ct)} towns x 17 = {len(ct)*17} town-quarters; by group {ct.grp.value_counts().to_dict()}")
n_abq = int(zx.city.str.upper().isin(ABQ).sum())
p(f"Albuquerque-metro ZIPs (all interior): {n_abq}")

p(f"Counties: conditional sample {cond.county_fips.nunique()}; dispensary panel "
  f"{d.storefront_id.map(dict(zip(sf.storefront_id, sf.zip5))).map(dict(zip(zx.zip5, zx.county_fips))).nunique()}")
p(f"Total sales 2022Q2-2026Q2: ${sq.revenue.sum()/1e9:.2f}bn")

open(os.path.join(ROOT, "logs", "data_section_numbers.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print("done")
