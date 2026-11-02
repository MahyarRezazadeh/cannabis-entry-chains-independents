r"""
_common.py -- shared builders for the referee revision (revisions/code).

Every object here is rebuilt from the same build files and definitions used for the manuscript's
Tables 3-4 (updated_draft_with_IO/code/01-06):
  * zip_xwalk.csv groups: nonlegal-border (92), always-legal-border (117), interior (159)
  * 13 storefronts with invalid ZIPs snapped to the nearest valid NM ZIP
  * 2022Q2-2026Q2, a storefront is active in a quarter iff its quarterly revenue > 0
  * chain (lifetime) = brand with >= 2 NM storefronts over the sample
New objects (all predetermined with respect to adult-use entry decisions, or purely geographic):
  * exposure measures built from SimpleMaps ZIP populations of TX/OK/UT (nonlegal) and CO/AZ/NV (legal)
  * covariates: log population density, distance to Albuquerque and to the nearest NM city >= 50k,
    medical incumbents per 1,000 (storefronts reporting medical sales in April 2022),
    2019 ZBP retail establishments per 1,000, nearest neighboring state
  * three organizational classifications (lifetime, at entry, predetermined at market opening)
"""
import os
import numpy as np
import pandas as pd

from paths import BUILD, REF, ZBP, ANALYSIS, TABLES, FIGURES, LOGS
BUILD, REF, ZBP = str(BUILD), str(REF), str(ZBP)
OUT = str(ANALYSIS)          # cached panels and logs of the analysis scripts
ROOT = str(TABLES.parent)    # output/ (tables and figures are written to output/tables, output/figures)
EARTH_R = 3958.8
QMIN, QMAX = 2022 * 4 + 1, 2026 * 4 + 1
QS = list(range(QMIN, QMAX + 1))
QLAB = [f"{q // 4}Q{q % 4 + 1}" for q in QS]
NONLEGAL_STATES = ["TX", "OK", "UT"]
LEGAL_STATES = ["CO", "AZ", "NV"]          # adult-use sales before April 2022 (CO 2014, NV 2017, AZ 2021)


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def tex_escape_pct(s):
    return s.replace("%", r"\%")


# ------------------------------------------------------------------ crosswalk and storefronts
def load_zx():
    zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})
    zx["group"] = np.select([zx.border_z == 1, (zx.placebo_border_z == 1) & (zx.border_z == 0)],
                            ["nonlegal", "legal"], default="interior")
    assert (zx.group == "nonlegal").sum() == 92 and (zx.group == "legal").sum() == 117
    return zx


def load_storefronts(zx):
    sf = pd.read_csv(os.path.join(BUILD, "storefronts.csv"), dtype={"zip5": str})
    xw = pd.read_csv(os.path.join(BUILD, "brand_crosswalk.csv"))
    bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
    sf = sf.merge(xw[["storefront_id", "brand_id"]], on="storefront_id", how="left")
    sf["chain_life"] = sf.brand_id.map(dict(zip(bm.brand_id, (bm.n_storefronts >= 2).astype(int))))
    for i in sf.index[~sf.zip5.isin(set(zx.zip5))]:
        sf.at[i, "zip5"] = zx.zip5.to_numpy()[np.argmin(hav(sf.at[i, "lat"], sf.at[i, "lon"],
                                                            zx.lat.to_numpy(), zx.lon.to_numpy()))]
    return sf, bm


def load_store_quarters(sf):
    sfm = pd.read_csv(os.path.join(BUILD, "sf_month_revenue.csv"))
    sfm["qt"] = sfm.year * 4 + (sfm.month - 1) // 3
    sq = (sfm[sfm.qt.between(QMIN, QMAX)].groupby(["storefront_id", "qt"])
          .agg(revenue=("total", "sum"), medical=("medical", "sum")).reset_index())
    sq = sq[sq.revenue > 0]
    # medical incumbents: storefronts reporting medical sales in April 2022, the first month of adult-use sales
    apr = sfm[(sfm.year == 2022) & (sfm.month == 4) & (sfm.medical > 0)].storefront_id.unique()
    sf = sf.copy()
    sf["med_incumbent"] = sf.storefront_id.isin(apr).astype(int)
    return sq.merge(sf[["storefront_id", "zip5", "brand_id", "chain_life", "med_incumbent", "lat", "lon"]],
                    on="storefront_id"), sf


def classify(sq, bm):
    """Three organizational classifications for every storefront active 2022Q2-2026Q2.
    status_entry: 'established' if the operator already ran another NM storefront when this one first
                  reported sales (same algorithm as the manuscript's entry game), else 'first_loc' (operator
                  later expands) or 'single' (operator never has a second storefront).
    chain_pre:    operator ran >= 2 NM storefronts in 2022Q2 (market opening) -- predetermined, no look-ahead.
    """
    fa = sq.groupby("storefront_id").qt.min().rename("faq")
    meta = sq.groupby("storefront_id").agg(brand_id=("brand_id", "first")).join(fa).reset_index()
    meta = meta.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
    meta["lc"] = meta.faq == QMIN
    rows = []
    for bid, gg in meta.groupby("brand_id"):
        for _, r in gg.iterrows():
            prior = ((gg.faq < r.faq) & (gg.storefront_id != r.storefront_id)).sum()
            prior_lc = ((gg.lc) & (gg.storefront_id != r.storefront_id) & (gg.faq <= r.faq)).sum()
            has = (prior > 0) or ((not r.lc) and prior_lc > 0)
            rows.append((r.storefront_id, "established" if has else ("first_loc" if r.n_storefronts >= 2 else "single")))
    st = pd.DataFrame(rows, columns=["storefront_id", "status_entry"])
    open_cnt = sq[sq.qt == QMIN].groupby("brand_id").storefront_id.nunique()
    meta = meta.merge(st, on="storefront_id")
    meta["chain_pre"] = (meta.brand_id.map(open_cnt).fillna(0) >= 2).astype(int)
    meta["n_open_brand"] = meta.brand_id.map(open_cnt).fillna(0)
    return meta[["storefront_id", "faq", "lc", "status_entry", "chain_pre", "n_open_brand"]]


# ------------------------------------------------------------------ exposure and covariates
def us_zips():
    z = pd.read_csv(os.path.join(REF, "uszips_simplemaps.csv"), dtype={"zip": str})
    z["zip"] = z.zip.str.zfill(5)
    return z[z.population.fillna(0) > 0]


def build_zip_covariates(zx, sf_med):
    z = us_zips()
    nl = z[z.state_id.isin(NONLEGAL_STATES)]
    lg = z[z.state_id.isin(LEGAL_STATES)]
    nm = z[z.state_id == "NM"]
    lat, lon = zx.lat.to_numpy()[:, None], zx.lon.to_numpy()[:, None]
    dnl = hav(lat, lon, nl.lat.to_numpy()[None, :], nl.lng.to_numpy()[None, :])
    dlg = hav(lat, lon, lg.lat.to_numpy()[None, :], lg.lng.to_numpy()[None, :])
    pnl, plg = nl.population.to_numpy()[None, :], lg.population.to_numpy()[None, :]
    cov = zx[["zip5"]].copy()
    for r in [25, 50, 75, 100, 150]:
        cov[f"nlpop_{r}"] = (pnl * (dnl <= r)).sum(1)
        cov[f"lgpop_{r}"] = (plg * (dlg <= r)).sum(1)
    for tau in [25, 50]:
        cov[f"nl_decay_{tau}"] = (pnl * np.exp(-dnl / tau)).sum(1)
    # accessible nonlegal population: residents whose nearest legal adult-use state is New Mexico
    import geopandas as gpd
    from shapely.geometry import Point
    stt = gpd.read_file(os.path.join(REF, "cb_2023_us_state_500k", "cb_2023_us_state_500k.shp")).to_crs("EPSG:5070")
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(nl.lng, nl.lat)], crs="EPSG:4326").to_crs("EPSG:5070")
    dist_to = {s: pts.distance(stt[stt.STUSPS == s].geometry.iloc[0]).to_numpy() for s in ["NM"] + LEGAL_STATES}
    d_alt = np.minimum.reduce([dist_to[s] for s in LEGAL_STATES])
    nm_nearest = (dist_to["NM"] < d_alt)[None, :]
    for r in [75, 100]:
        cov[f"nlacc_{r}"] = (pnl * (dnl <= r) * nm_nearest).sum(1)
    cov["nlacc_decay_50"] = (pnl * np.exp(-dnl / 50) * nm_nearest).sum(1)
    # nearest neighboring state (by distance to the nearest out-of-state populated ZIP)
    nb = z[z.state_id.isin(NONLEGAL_STATES + ["CO", "AZ"])]
    dnb = hav(lat, lon, nb.lat.to_numpy()[None, :], nb.lng.to_numpy()[None, :])
    cov["nearest_state"] = nb.state_id.to_numpy()[dnb.argmin(1)]
    cov["nearest_nonlegal_state"] = nl.state_id.to_numpy()[dnl.argmin(1)]
    cov["nearest_legal_state"] = lg.state_id.to_numpy()[dlg.argmin(1)]
    # urbanization and distance to population centres (all from NM ZIP populations, no external coordinates)
    dens = z.set_index("zip").density
    cov["log_density"] = np.log1p(cov.zip5.map(dens).fillna(0).to_numpy())
    zxa = zx.copy()
    city_pop = zxa.groupby("city").zip_pop.sum()
    big = city_pop[city_pop >= 50000].index
    cen = zxa[zxa.city.isin(big)].groupby("city").apply(
        lambda d: pd.Series({"lat": np.average(d.lat, weights=d.zip_pop), "lon": np.average(d.lon, weights=d.zip_pop)}))
    dc = hav(lat, lon, cen.lat.to_numpy()[None, :], cen.lon.to_numpy()[None, :])
    cov["dist_bigcity"] = dc.min(1)
    abq = cen.loc[[c for c in cen.index if c.upper().startswith("ALBUQUERQUE")]].iloc[0]
    cov["dist_abq"] = hav(zx.lat.to_numpy(), zx.lon.to_numpy(), abq.lat, abq.lon)
    # predetermined commercial activity: 2019 ZBP retail establishments
    zb = pd.read_csv(os.path.join(ZBP, "nm_zip_zbp_raw.csv"), dtype={"zip5": str})
    zb = zb[(zb.year == 2019)].pivot_table(index="zip5", columns="sector", values="ESTAB", aggfunc="sum").fillna(0)
    for s in ["total_retail", "alcohol", "grocery"]:
        cov[f"zbp19_{s}"] = cov.zip5.map(zb[s]).fillna(0).to_numpy() if s in zb else 0.0
    # medical incumbents (storefronts with medical sales in April 2022)
    med = sf_med[sf_med.med_incumbent == 1].groupby("zip5").size()
    cov["n_med_incumbent"] = cov.zip5.map(med).fillna(0).to_numpy()
    cov.attrs["big_cities"] = list(big)
    return cov


def zip_panel(zx, sq, meta, cov=None, groups=("nonlegal", "legal")):
    """Balanced ZIP-quarter panel with all outcomes; groups restricts to the listed exposure groups."""
    d = sq.merge(meta, on="storefront_id", how="left")
    g = d.groupby(["zip5", "qt"])
    agg = g.agg(n_stores=("storefront_id", "nunique"), rev=("revenue", "sum")).reset_index()

    def nb(mask, name):
        s = d[mask].groupby(["zip5", "qt"]).brand_id.nunique().rename(name)
        return s

    def ns(mask, name):
        return d[mask].groupby(["zip5", "qt"]).storefront_id.nunique().rename(name)

    parts = [nb(d.chain_life == 0, "nb_I_life"), nb(d.chain_life == 1, "nb_C_life"),
             ns(d.chain_life == 0, "ns_I_life"), ns(d.chain_life == 1, "ns_C_life"),
             ns(d.status_entry == "established", "ns_C_entry"), ns(d.status_entry != "established", "ns_I_entry"),
             nb(d.status_entry == "established", "nb_C_entry"), nb(d.status_entry != "established", "nb_I_entry"),
             nb(d.chain_pre == 1, "nb_C_pre"), nb(d.chain_pre == 0, "nb_I_pre"),
             ns(d.chain_pre == 1, "ns_C_pre"), ns(d.chain_pre == 0, "ns_I_pre"),
             ns(d.med_incumbent == 1, "ns_med"), ns(d.med_incumbent == 0, "ns_new")]
    grid = pd.MultiIndex.from_product([sorted(zx.zip5), QS], names=["zip5", "qt"]).to_frame(index=False)
    full = grid.merge(agg, on=["zip5", "qt"], how="left")
    for s in parts:
        full = full.merge(s.reset_index(), on=["zip5", "qt"], how="left")
    full = full.fillna(0)
    full = full.merge(zx[["zip5", "zip_pop", "group", "county_fips", "lat", "lon", "dist_nonleg_mi",
                          "dist_legalneigh_mi", "nonleg_pop_75"]], on="zip5")
    if cov is not None:
        full = full.merge(cov, on="zip5", how="left")
    full["log_pop"] = np.log(full.zip_pop.clip(lower=1))
    full["rpc"] = full.rev / full.zip_pop
    full["den"] = full.n_stores / full.zip_pop * 1000
    full["y_rpc"] = np.arcsinh(full.rpc)
    full["y_den"] = np.arcsinh(full.den)
    full["y_rps"] = np.log((full.rev / full.n_stores).where(full.n_stores > 0))
    full["nonlegal"] = (full.group == "nonlegal").astype(int)
    full["any"] = (full.n_stores > 0).astype(int)
    full = full[full.group.isin(groups)].sort_values(["zip5", "qt"]).reset_index(drop=True)
    return full


def build_all(groups=("nonlegal", "legal"), with_cov=True):
    zx = load_zx()
    sf, bm = load_storefronts(zx)
    sq, sf = load_store_quarters(sf)
    meta = classify(sq, bm)
    cov = build_zip_covariates(zx, sf) if with_cov else None
    full = zip_panel(zx, sq, meta, cov, groups)
    return dict(zx=zx, sf=sf, bm=bm, sq=sq, meta=meta, cov=cov, full=full)
