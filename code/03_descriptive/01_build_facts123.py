r"""
01_build_facts123.py -- updated_draft_with_IO -- CORRECTED, full-sample-as-main rebuild

REVISION (this pass): two corrections propagated through every table.

(1) ZIP-snapping fix. 13 of 1,255 storefronts carry a raw self-reported zip5 that does
    not exist in the 368-ZIP crosswalk (typos, transposed digits, PO-box codes, one
    wrong-state prefix). Left alone, these silently default to "interior" in the group
    classification (np.select's default arm) or get dropped entirely from a strict
    inner join -- neither is a geographic classification. Each of the 13 is snapped to
    its NEAREST real ZIP centroid (haversine on lat/lon), and ONLY these 13 -- every
    other storefront keeps its own recorded (valid) zip5 untouched. (An earlier attempt
    snapped every storefront to its nearest centroid; that reassigned 431/1255 storefronts,
    corrupting perfectly valid ZIPs in dense areas like Albuquerque. Do not repeat that.)

(2) Full sample is now MAIN, conditional-on-activity is now SECONDARY. The 136-ZIP
    conditional-on-activity sample can only ever see towns that already have a dispensary,
    which is exactly the extensive-margin variation the free-entry mechanism (P1) lives on.
    The full 368-ZIP zero-inclusive panel (all populated NM ZIPs x all quarters, true zeros
    filled in) is therefore now the MAIN specification throughout; the conditional-on-activity
    design (log outcomes, market-age control) is retained as a SECONDARY check -- it speaks to
    the intensive margin among markets that already have some activity, which is a real and
    useful question, just not the main one.

    The market-age control (evt) is NOT used in the full-368 main specification: for a
    zero-inclusive panel, "quarters since a ZIP's first active dispensary" is undefined for a
    ZIP that never has one, and forcing every never-active ZIP-quarter into its own bucket
    creates a near-perfect "ever had a dispensary" fixed effect that absorbs exactly the
    extensive-margin entry response the model is trying to measure (verified: doing this
    inflates R^2 to ~0.86 and flips the sign/significance of log_pop). The full-368 main
    specification therefore uses quarter FE only. The conditional-on-activity secondary
    specification keeps the market-age control, where it remains well-defined.

Groups (368-ZIP universe, unchanged from the geography-only classification):
    nonlegal-border = 92 ZIPs (border_z==1; includes the 14 ZIPs that also qualify for the
        legal-neighbor rule -- nonlegal takes priority for those, per instruction)
    always-legal-border = 117 ZIPs (placebo_border_z==1 & border_z==0)
    interior = 159 ZIPs
    368 total.

Outputs (LaTeX tables, no hand-retyped numbers) go to updated_draft_with_IO/output/.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import paths as _P
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

BUILD = str(_P.BUILD)
ROOT = str(_P.OUT)
OUT = str(_P.TABLES)
os.makedirs(OUT, exist_ok=True)

ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}
EARTH_R = 3958.8


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


def asinh(x):
    return np.log(x + np.sqrt(x ** 2 + 1))


def stars(pv):
    return "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""


def ols_cluster(df, y, rhs, cluster):
    d = df.dropna(subset=[y, cluster] + [c for c in rhs if "(" not in c]).copy()
    formula = f"{y} ~ " + " + ".join(rhs)
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    tidy = {t: (m.params[t], m.bse[t], m.pvalues[t]) for t in m.params.index}
    tidy["_N"] = int(m.nobs); tidy["_r2"] = float(m.rsquared)
    return tidy


def fmt_cell(t, key, pct=True):
    if key not in t:
        return ". & . \\\\"
    b, se, p = t[key]
    pctxt = f" ({100*(np.exp(b)-1):+.1f}\\%)" if pct else ""
    return f"{b:+.3f}{stars(p)}{pctxt} & ({se:.3f}) \\\\"


# =====================================================================================
# STEP 0 -- ZIP-snapping correction, applied ONCE, upstream of everything
# =====================================================================================
zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})
zx["group"] = np.select([zx.border_z == 1, (zx.placebo_border_z == 1) & (zx.border_z == 0)],
                         ["nonlegal", "legal"], default="interior")
assert zx.border_z.sum() == 92 and (zx.group == "legal").sum() == 117 and (zx.group == "interior").sum() == 159, \
    "368-ZIP group counts changed -- check zip_xwalk.csv before proceeding"

sf = pd.read_csv(os.path.join(BUILD, "storefronts.csv"), dtype={"zip5": str})
xw = pd.read_csv(os.path.join(BUILD, "brand_crosswalk.csv"))
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
sf = sf.merge(xw[["storefront_id", "brand_id", "brand_name", "ownership_change_flag"]], on="storefront_id", how="left")
chain_map = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))
sf["chain"] = sf["brand_id"].map(chain_map)

valid_zips = set(zx["zip5"])
zlat, zlon, zzip = zx["lat"].to_numpy(), zx["lon"].to_numpy(), zx["zip5"].to_numpy()
sf["zip5_orig"] = sf["zip5"]
bad_mask = ~sf["zip5"].isin(valid_zips)
for i in sf.index[bad_mask]:
    lat, lon = sf.at[i, "lat"], sf.at[i, "lon"]
    d = hav(lat, lon, zlat, zlon)
    sf.at[i, "zip5"] = zzip[np.argmin(d)]
n_fixed = bad_mask.sum()
print(f"ZIP-snapping correction: {n_fixed} of {len(sf)} storefronts had an invalid/unmatched raw ZIP, "
      f"each snapped to its nearest real ZIP (only these {n_fixed} touched):")
print(sf.loc[bad_mask, ["storefront_id", "zip5_orig", "zip5", "city"]].to_string(index=False))
fixlog = sf.loc[bad_mask, ["storefront_id", "zip5_orig", "zip5", "city", "name_modal", "lifetime_sales"]].copy()
fixlog.columns = ["storefront_id", "raw_zip5", "corrected_zip5", "city", "name", "lifetime_sales"]
fixlog.to_csv(str(_P.LOGS / "zip_snap_correction_log.csv"), index=False)

sf = sf.merge(zx[["zip5", "county_fips", "county_name"]].drop_duplicates("zip5"), on="zip5", how="left")

# ---------------- storefront x quarter panel (revenue), corrected zip5 ----------------
sfm = pd.read_csv(os.path.join(BUILD, "sf_month_revenue.csv"))
sfm["q"] = (sfm["month"] - 1) // 3 + 1
sfm["qt"] = sfm["year"] * 4 + (sfm["q"] - 1)
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
sq = (sfm[sfm["qt"].between(QMIN, QMAX)].groupby(["storefront_id", "qt"])
      .agg(revenue=("total", "sum")).reset_index())
sq["active"] = (sq["revenue"] > 0).astype(int)
meta_cols = ["zip5", "city", "county_fips", "county_name", "lat", "lon", "brand_id", "chain"]
sq = sq.merge(sf[["storefront_id"] + meta_cols], on="storefront_id", how="left")

# =====================================================================================
# PART A (NEW MAIN) -- full 368-ZIP x quarter zero-inclusive panel
# =====================================================================================
act = sq[sq.active == 1].copy()
g2 = (act.groupby(["zip5", "qt"])[["storefront_id", "revenue", "brand_id", "chain"]]
      .apply(lambda s: pd.Series({
          "n_stores": s["storefront_id"].nunique(),
          "total_revenue": s["revenue"].sum(),
          "n_indep_brands": s.loc[s.chain == 0, "brand_id"].nunique(),
          "n_chain_brands": s.loc[s.chain == 1, "brand_id"].nunique()}))
      .reset_index())

all_qt = sorted(sq["qt"].unique())
all_zip = sorted(zx["zip5"].unique())
grid = pd.MultiIndex.from_product([all_zip, all_qt], names=["zip5", "qt"]).to_frame(index=False)
full = grid.merge(g2, on=["zip5", "qt"], how="left")
for c in ["n_stores", "total_revenue", "n_indep_brands", "n_chain_brands"]:
    full[c] = full[c].fillna(0)
full = full.merge(zx[["zip5", "zip_pop", "border_z", "placebo_border_z", "group", "county_fips"]],
                   on="zip5", how="left")
full["log_pop"] = np.log(full["zip_pop"].clip(lower=1))
full["asinh_revenue_pc"] = asinh(full["total_revenue"] / full["zip_pop"])
full["asinh_density"] = asinh(full["n_stores"] / full["zip_pop"] * 1000)
full["asinh_indep"] = asinh(full["n_indep_brands"])
full["asinh_chain"] = asinh(full["n_chain_brands"])
full["log_revenue_per_store"] = np.log((full["total_revenue"] / full["n_stores"]).where(full["n_stores"] > 0))
full["nonlegal"] = (full["group"] == "nonlegal").astype(float)
full["legal"] = (full["group"] == "legal").astype(float)
full["county_fips"] = full["county_fips"].astype(str)

n_nl_full = full.loc[full.group == "nonlegal", "zip5"].nunique()
n_lg_full = full.loc[full.group == "legal", "zip5"].nunique()
n_in_full = full.loc[full.group == "interior", "zip5"].nunique()
print(f"\nFull 368-ZIP zero-inclusive panel (MAIN): {n_nl_full} nonlegal-border / {n_lg_full} always-legal-border / "
      f"{n_in_full} interior ZIPs, {len(full)} ZIP-quarters")

MAIN_368 = full[full.group.isin(["nonlegal", "legal"])].copy()
POOLED_368 = full.copy()

main_res, pooled_res = {}, {}
RHS_MAIN = ["nonlegal", "log_pop", "C(qt)"]
for y, label in [("asinh_revenue_pc", "revenue_pc"), ("asinh_density", "density"),
                  ("log_revenue_per_store", "rev_per_store"),
                  ("asinh_indep", "n_indep"), ("asinh_chain", "n_chain")]:
    main_res[label] = ols_cluster(MAIN_368, y, RHS_MAIN, "county_fips")
    pooled_res[label] = ols_cluster(POOLED_368, y, RHS_MAIN, "county_fips")

print("\n--- MAIN (nonlegal vs always-legal-border only), full 368-ZIP sample ---")
for label in ["revenue_pc", "density", "rev_per_store", "n_indep", "n_chain"]:
    b, se, p = main_res[label]["nonlegal"]
    print(f"  {label:14s} beta={b:+.4f}{stars(p)} se={se:.4f} N={main_res[label]['_N']}")
print("--- POOLED (nonlegal vs legal+interior pooled), full 368-ZIP sample ---")
for label in ["revenue_pc", "density", "rev_per_store", "n_indep", "n_chain"]:
    b, se, p = pooled_res[label]["nonlegal"]
    print(f"  {label:14s} beta={b:+.4f}{stars(p)} se={se:.4f} N={pooled_res[label]['_N']}")

# ---- LaTeX: Fact 1 & 2 main table (revenue per resident, density, revenue/store) ----
tab = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab += r" & \multicolumn{2}{c}{Main: nonlegal- vs.\ legal-border} & \multicolumn{2}{c}{Pooled: legal+interior pooled} \\" + "\n"
tab += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab += "Outcome & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
rows = [("Revenue per resident (asinh)", "revenue_pc"), ("Dispensary density (asinh)", "density"),
        ("log(Revenue per store)", "rev_per_store")]
for lab, key in rows:
    m, r = main_res[key], pooled_res[key]
    bm, sm, pm = m["nonlegal"]; br, sr, pr = r["nonlegal"]
    tab += f"{lab} & {bm:+.3f}{stars(pm)} & ({sm:.3f}) & {br:+.3f}{stars(pr)} & ({sr:.3f}) \\\\\n"
tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_fact1_fact2_main.tex"), "w", encoding="utf-8").write(tab)

# ---- LaTeX: Fact 3 composition table (n_indep, n_chain) ----
tab3 = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab3 += r" & \multicolumn{2}{c}{Main: nonlegal- vs.\ legal-border} & \multicolumn{2}{c}{Pooled: legal+interior pooled} \\" + "\n"
tab3 += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab3 += "Outcome (asinh) & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
rows3 = [("N independent brands", "n_indep"), ("N chain brands", "n_chain")]
for lab, key in rows3:
    m, r = main_res[key], pooled_res[key]
    bm, sm, pm = m["nonlegal"]; br, sr, pr = r["nonlegal"]
    tab3 += f"{lab} & {bm:+.3f}{stars(pm)} & ({sm:.3f}) & {br:+.3f}{stars(pr)} & ({sr:.3f}) \\\\\n"
tab3 += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_fact3_composition_main.tex"), "w", encoding="utf-8").write(tab3)

# =====================================================================================
# PART B -- Albuquerque include/exclude, composition only, on the FULL-368 pooled sample
# =====================================================================================
is_abq = full["zip5"].isin(zx.loc[zx["city"].str.upper().isin(ABQ), "zip5"])
n_abq_zips = int(full.loc[is_abq, "zip5"].nunique())
POOLED_noabq = full[~is_abq].copy()
abq_rows = {}
for y, label in [("asinh_indep", "n_indep"), ("asinh_chain", "n_chain")]:
    abq_rows[(label, "incl")] = pooled_res[label]
    abq_rows[(label, "excl")] = ols_cluster(POOLED_noabq, y, RHS_MAIN, "county_fips")
print(f"\nAlbuquerque ZIPs in full-368 sample: {n_abq_zips} (all interior -- affects only the pooled "
      f"legal+interior robustness sample, not the main nonlegal-vs-legal comparison)")
for label in ["n_indep", "n_chain"]:
    ti, te = abq_rows[(label, "incl")], abq_rows[(label, "excl")]
    print(f"  {label}: ABQ incl beta={ti['nonlegal'][0]:+.4f}  ABQ excl beta={te['nonlegal'][0]:+.4f}")

tab_abq = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab_abq += r" & \multicolumn{2}{c}{Pooled control, ABQ included} & \multicolumn{2}{c}{Pooled control, ABQ excluded} \\" + "\n"
tab_abq += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab_abq += "Outcome (asinh) & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
for lab, key in rows3:
    ti, te = abq_rows[(key, "incl")], abq_rows[(key, "excl")]
    bi, si, pi = ti["nonlegal"]; be, se_, pe = te["nonlegal"]
    tab_abq += f"{lab} & {bi:+.3f}{stars(pi)} & ({si:.3f}) & {be:+.3f}{stars(pe)} & ({se_:.3f}) \\\\\n"
tab_abq += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_fact3_abq_robustness.tex"), "w", encoding="utf-8").write(tab_abq)

# =====================================================================================
# PART C (NOW SECONDARY) -- conditional-on-activity ZIP panel, log outcomes, market-age control
# =====================================================================================
zq = act.groupby(["zip5", "qt"])[["storefront_id", "revenue", "brand_id", "chain"]].apply(
    lambda s: pd.Series({
        "n_stores": s["storefront_id"].nunique(),
        "total_revenue": s["revenue"].sum(),
        "n_indep_brands": s.loc[s.chain == 0, "brand_id"].nunique(),
        "n_chain_brands": s.loc[s.chain == 1, "brand_id"].nunique()})
).reset_index()
zq = zq.merge(zx[["zip5", "zip_pop", "border_z", "placebo_border_z", "group", "county_fips"]], on="zip5", how="left")
zq["nonlegal"] = (zq["group"] == "nonlegal").astype(float)
zq["log_pop"] = np.log(zq["zip_pop"].clip(lower=1))

_zfq = act.groupby("zip5")["qt"].min().rename("zfq")
zq = zq.merge(_zfq, on="zip5", how="left")
zq["evt"] = zq["qt"] - zq["zfq"]
zq = zq[zq["evt"] >= 0].copy()
zq["log_revenue"] = np.log(zq["total_revenue"].clip(lower=1))
zq["log_stores"] = np.log(zq["n_stores"].clip(lower=1))
zq["log_rev_per_store"] = np.log((zq["total_revenue"] / zq["n_stores"]).clip(lower=1))

n_nl_c = zq.loc[zq.group == "nonlegal", "zip5"].nunique()
n_lg_c = zq.loc[zq.group == "legal", "zip5"].nunique()
n_in_c = zq.loc[zq.group == "interior", "zip5"].nunique()
print(f"\nConditional-on-activity ZIP panel (SECONDARY, ZIP-snapping corrected): "
      f"{n_nl_c} nonlegal-border / {n_lg_c} always-legal-border / {n_in_c} interior; "
      f"{zq.zip5.nunique()} ZIPs total, {len(zq)} ZIP-quarters")

MAIN_cond = zq[zq.group.isin(["nonlegal", "legal"])].copy()
RHS_COND = ["nonlegal", "log_pop", "C(qt)", "C(evt)"]
cond_res = {}
for y, label in [("log_revenue", "revenue"), ("log_stores", "stores"), ("log_rev_per_store", "rev_per_store"),
                  ("n_indep_brands", "n_indep"), ("n_chain_brands", "n_chain")]:
    cond_res[(label, "main")] = ols_cluster(MAIN_cond, y, RHS_COND, "county_fips")
    cond_res[(label, "pooled")] = ols_cluster(zq, y, RHS_COND, "county_fips")

print("--- SECONDARY conditional-on-activity, MAIN (nonlegal vs legal only) ---")
for label in ["revenue", "stores", "rev_per_store", "n_indep", "n_chain"]:
    b, se, p = cond_res[(label, "main")]["nonlegal"]
    print(f"  {label:14s} beta={b:+.4f}{stars(p)} se={se:.4f} N={cond_res[(label,'main')]['_N']}")

tab_cond1 = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab_cond1 += r" & \multicolumn{2}{c}{Main: nonlegal- vs.\ legal-border} & \multicolumn{2}{c}{Pooled: legal+interior pooled} \\" + "\n"
tab_cond1 += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab_cond1 += "Outcome (log) & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
rows_cond = [("log(Total revenue)", "revenue"), ("log(N stores)", "stores"), ("log(Revenue per store)", "rev_per_store")]
for lab, key in rows_cond:
    m, r = cond_res[(key, "main")], cond_res[(key, "pooled")]
    bm, sm, pm = m["nonlegal"]; br, sr, pr = r["nonlegal"]
    tab_cond1 += f"{lab} & {bm:+.3f}{stars(pm)} & ({sm:.3f}) & {br:+.3f}{stars(pr)} & ({sr:.3f}) \\\\\n"
tab_cond1 += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_fact1_fact2_conditional.tex"), "w", encoding="utf-8").write(tab_cond1)

tab_cond3 = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab_cond3 += r" & \multicolumn{2}{c}{Main: nonlegal- vs.\ legal-border} & \multicolumn{2}{c}{Pooled: legal+interior pooled} \\" + "\n"
tab_cond3 += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab_cond3 += "Outcome & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
rows_cond3 = [("N independent brands", "n_indep"), ("N chain brands", "n_chain")]
for lab, key in rows_cond3:
    m, r = cond_res[(key, "main")], cond_res[(key, "pooled")]
    bm, sm, pm = m["nonlegal"]; br, sr, pr = r["nonlegal"]
    tab_cond3 += f"{lab} & {bm:+.3f}{stars(pm)} & ({sm:.3f}) & {br:+.3f}{stars(pr)} & ({sr:.3f}) \\\\\n"
tab_cond3 += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_zeroinclusive_robustness.tex"), "w", encoding="utf-8").write(tab_cond3)
# NOTE: this file is repurposed -- it now holds the CONDITIONAL-ON-ACTIVITY secondary check
# (composition columns), not the old zero-inclusive check (the zero-inclusive design is now
# the MAIN sample, reported in tab_fact3_composition_main.tex). A parallel Facts-1-2 conditional
# table is written separately as tab_fact1_fact2_conditional.tex (new file, needs a new \input
# in the draft since there was no slot for it before).

# =====================================================================================
# PART D -- town-level, now ALSO rebuilt as a full zero-inclusive panel (all towns x all
# quarters, ex-ABQ), for consistency with "full sample is main" applied at every level
# =====================================================================================
meta = sf.groupby("storefront_id").agg(city=("city", "first"), zip5=("zip5", "first")).reset_index()
meta["mkt"] = np.where(meta["city"].isin(ABQ), "ABQ", meta["city"])
act2 = sq[sq.active == 1].merge(meta[["storefront_id", "mkt"]], on="storefront_id", how="left")
tg = (act2.groupby(["mkt", "qt"])[["storefront_id", "revenue", "brand_id", "chain"]]
      .apply(lambda s: pd.Series({
          "n_stores": s["storefront_id"].nunique(),
          "total_revenue": s["revenue"].sum(),
          "n_indep_brands": s.loc[s.chain == 0, "brand_id"].nunique(),
          "n_chain_brands": s.loc[s.chain == 1, "brand_id"].nunique()}))
      .reset_index())
cs = pd.read_csv(os.path.join(BUILD, "city_cross_section.csv")).rename(columns={"metro": "mkt"})
cs_town = cs[cs["is_abq"] == 0].dropna(subset=["city_pop"]).copy()

all_qt_t = sorted(sq["qt"].unique())
all_town = sorted(cs_town["mkt"].unique())
grid_t = pd.MultiIndex.from_product([all_town, all_qt_t], names=["mkt", "qt"]).to_frame(index=False)
town = grid_t.merge(tg, on=["mkt", "qt"], how="left")
for c in ["n_stores", "total_revenue", "n_indep_brands", "n_chain_brands"]:
    town[c] = town[c].fillna(0)
town = town.merge(cs_town[["mkt", "city_pop", "border_city", "placebo_border_city"]], on="mkt", how="left")
town["group"] = np.select([town.border_city == 1, (town.placebo_border_city == 1) & (town.border_city == 0)],
                           ["nonlegal", "legal"], default="interior")
town["nonlegal"] = (town.group == "nonlegal").astype(float)
town["log_pop"] = np.log(town["city_pop"].clip(lower=1))
town["asinh_revenue_pc"] = asinh(town["total_revenue"] / town["city_pop"])
town["asinh_density"] = asinh(town["n_stores"] / town["city_pop"] * 1000)
town["asinh_indep"] = asinh(town["n_indep_brands"])
town["asinh_chain"] = asinh(town["n_chain_brands"])
town["mkt_c"] = town["mkt"]

n_nl_t = town.loc[town.group == "nonlegal", "mkt"].nunique()
n_lg_t = town.loc[town.group == "legal", "mkt"].nunique()
n_in_t = town.loc[town.group == "interior", "mkt"].nunique()
print(f"\nFull zero-inclusive town panel (ex-ABQ, MAIN-consistent): {n_nl_t} nonlegal / {n_lg_t} legal / "
      f"{n_in_t} interior towns, {len(town)} town-quarters")

TOWN_MAIN = town[town.group.isin(["nonlegal", "legal"])].copy()
RHS_TOWN = ["nonlegal", "log_pop", "C(qt)"]
town_main_res, town_pooled_res = {}, {}
for y, label in [("asinh_revenue_pc", "revenue_pc"), ("asinh_density", "density"),
                  ("asinh_indep", "n_indep"), ("asinh_chain", "n_chain")]:
    town_main_res[label] = ols_cluster(TOWN_MAIN, y, RHS_TOWN, "mkt_c")
    town_pooled_res[label] = ols_cluster(town, y, RHS_TOWN, "mkt_c")

tab_town = r"\begin{tabular}{@{}lcccc@{}}" + "\n" + r"\toprule" + "\n"
tab_town += r" & \multicolumn{2}{c}{Main: nonlegal vs.\ legal-border towns} & \multicolumn{2}{c}{Pooled: legal+interior pooled} \\" + "\n"
tab_town += r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
tab_town += "Outcome (asinh) & Coef. & (SE) & Coef. & (SE) \\\\\n" + r"\midrule" + "\n"
rows_town = [("Revenue per resident", "revenue_pc"), ("Dispensary density", "density"),
             ("N independent brands", "n_indep"), ("N chain brands", "n_chain")]
for lab, key in rows_town:
    m, r = town_main_res[key], town_pooled_res[key]
    bm, sm, pm = m["nonlegal"]; br, sr, pr = r["nonlegal"]
    tab_town += f"{lab} & {bm:+.3f}{stars(pm)} & ({sm:.3f}) & {br:+.3f}{stars(pr)} & ({sr:.3f}) \\\\\n"
tab_town += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_townlevel_robustness.tex"), "w", encoding="utf-8").write(tab_town)

# =====================================================================================
# PART E -- summary statistics table, FULL 368-ZIP sample (now the main descriptive table)
# =====================================================================================
grp_stats = []
for lab, sub in [("Full sample", full), ("Nonlegal-border", full[full.group == "nonlegal"]),
                 ("Always-legal-border", full[full.group == "legal"]), ("Interior", full[full.group == "interior"])]:
    grp_stats.append(dict(
        group=lab, n_zips=sub.zip5.nunique(), n_zipq=len(sub),
        mean_pop=sub["zip_pop"].mean(), mean_rev=sub["total_revenue"].mean(),
        mean_stores=sub["n_stores"].mean(), mean_indep=sub["n_indep_brands"].mean(),
        mean_chain=sub["n_chain_brands"].mean()))
sdf = pd.DataFrame(grp_stats)

tab_sum = r"\begin{tabular}{@{}lccccccc@{}}" + "\n" + r"\toprule" + "\n"
tab_sum += r"Sample & ZIPs & ZIP-qtrs & Pop. & Revenue & N stores & N indep. & N chain \\" + "\n"
tab_sum += r"\midrule" + "\n"
for _, r in sdf.iterrows():
    tab_sum += (f"{r['group']} & {int(r['n_zips'])} & {int(r['n_zipq'])} & {r['mean_pop']:,.0f} & "
                f"\\${r['mean_rev']:,.0f} & {r['mean_stores']:.2f} & {r['mean_indep']:.2f} & "
                f"{r['mean_chain']:.2f} \\\\\n")
tab_sum += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_summary_stats.tex"), "w", encoding="utf-8").write(tab_sum)
print("\nFull-368-ZIP summary statistics (now the main Table 1):")
print(sdf.to_string(index=False))

# also keep the conditional-136 summary stats as a labeled secondary table
grp_stats_c = []
for lab, sub in [("Full sample", zq), ("Nonlegal-border", zq[zq.group == "nonlegal"]),
                 ("Always-legal-border", zq[zq.group == "legal"]), ("Interior", zq[zq.group == "interior"])]:
    grp_stats_c.append(dict(
        group=lab, n_zips=sub.zip5.nunique(), n_zipq=len(sub),
        mean_pop=sub["zip_pop"].mean(), mean_rev=sub["total_revenue"].mean(),
        mean_stores=sub["n_stores"].mean(), mean_indep=sub["n_indep_brands"].mean(),
        mean_chain=sub["n_chain_brands"].mean()))
sdf_c = pd.DataFrame(grp_stats_c)
tab_sum_c = r"\begin{tabular}{@{}lccccccc@{}}" + "\n" + r"\toprule" + "\n"
tab_sum_c += r"Sample & ZIPs & ZIP-qtrs & Pop. & Revenue & N stores & N indep. & N chain \\" + "\n"
tab_sum_c += r"\midrule" + "\n"
for _, r in sdf_c.iterrows():
    tab_sum_c += (f"{r['group']} & {int(r['n_zips'])} & {int(r['n_zipq'])} & {r['mean_pop']:,.0f} & "
                  f"\\${r['mean_rev']:,.0f} & {r['mean_stores']:.2f} & {r['mean_indep']:.2f} & "
                  f"{r['mean_chain']:.2f} \\\\\n")
tab_sum_c += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_summary_stats_conditional.tex"), "w", encoding="utf-8").write(tab_sum_c)
print("\nConditional-on-activity (ZIP-snapping corrected) summary statistics (secondary Table):")
print(sdf_c.to_string(index=False))

print("\nDONE. Wrote 8 tables to", OUT)
