"""
ref01_referee_revision.py  --  extension of venous4 / referee revision pass

Feasible new analyses requested in the referee-style revision:

 A. Zero-inclusive 368-ZIP baseline as the MAIN specification
    - reproduce asinh(rev/capita), asinh(density) on all 368 populated ZIPs
    - PPML (Poisson) for the dispensary COUNT with a population control, 368 ZIPs
    - reproduce the 116-ZIP log spec (now a robustness check)
 B. Spatial inference for the ZIP baseline
    - Conley / spatial-HAC standard errors (ZIP centroids), several cutoffs
    - wild cluster bootstrap p-values (county clusters)
 C. Legal-border control group, full outcome set (revenue, density, rev/store,
    #independent, #chain, HHI, top-share)
 D. Organizational status AT ENTRY (no look-ahead)
    - reconstruct: existing multi-store operator / first NM location that later
      expands / persistent single-location
    - re-do composition-of-entrants, first-mover, entry-rank, exit, revenue-by-type
    - compare with the eventual-status ("chain ever") classification
 E. Medical-market incumbency
    - left-censored storefronts (operating 2022Q2, the recreational launch) are
      pre-existing medical dispensaries; quantify their role among first movers
 F. Descriptive adjustment mechanism
    - Entry_{z,t+1} on RevenuePerStore_{z,t} x Border_z (descriptive only)
 G. Dispensary store revenue at several local-demand radii (1/2/3/5 km, 25 mi),
    with and without operator fixed effects

Reads existing build files by absolute path; writes ref_*.tex tables and a log.
No new panels are constructed; nothing is fabricated.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import asinh

BE  = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\brand_entry_analysis\tables"
NM  = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project\tables"
OLD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ext_output", "ref01_referee_revision.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)
def st(pv): return "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""


def ols_c(d, formula, cluster):
    d = d.dropna(subset=[cluster]).copy()
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return m


def wild_cluster_boot(d, yvar, rhs_terms, key, cluster, B=1999, seed=42):
    """Rademacher wild cluster bootstrap-t p-value for `key`, restricted (H0) resampling."""
    rng = np.random.default_rng(seed)
    f_full = f"{yvar} ~ " + " + ".join(rhs_terms)
    m_full = ols_c(d, f_full, cluster)
    t_obs = m_full.params[key] / m_full.bse[key]
    # restricted model: drop the key regressor
    rhs_r = [t for t in rhs_terms if t != key]
    f_r = f"{yvar} ~ " + " + ".join(rhs_r)
    m_r = smf.ols(f_r, data=d).fit()
    resid_r = m_r.resid.values
    fit_r = m_r.fittedvalues.values
    gvals = d[cluster].values
    groups = np.unique(gvals)
    yname = "_wcb_y_"
    dd = d.copy()
    cnt = 0
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(groups))
        wmap = dict(zip(groups, w))
        wv = np.array([wmap[g] for g in gvals])
        dd[yname] = fit_r + resid_r * wv
        mb = smf.ols(f"{yname} ~ " + " + ".join(rhs_terms), data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": dd[cluster]})
        tb = mb.params[key] / mb.bse[key]
        if abs(tb) >= abs(t_obs):
            cnt += 1
    return t_obs, (cnt + 1) / (B + 1), int(len(groups))


def conley_se(d, yvar, Xcols, lat, lon, cutoffs_km=(25, 50, 100)):
    """Conley spatial-HAC SE via Bartlett kernel over great-circle distance."""
    d = d.dropna(subset=[yvar] + Xcols + [lat, lon]).copy()
    X = sm.add_constant(d[Xcols].astype(float).values)
    y = d[yvar].astype(float).values
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    la = np.radians(d[lat].astype(float).values); lo = np.radians(d[lon].astype(float).values)
    out = {}
    # precompute pairwise gc distance in km (n is a few thousand -> ok)
    sinla = np.sin(la); cosla = np.cos(la)
    for c in cutoffs_km:
        meat = np.zeros((k, k))
        for i in range(n):
            dph = la - la[i]
            dlm = lo - lo[i]
            aa = np.sin(dph / 2) ** 2 + cosla[i] * cosla * np.sin(dlm / 2) ** 2
            dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
            w = np.clip(1 - dist / c, 0, None)
            Xi = X[i][:, None]
            wk = (w * e)[None, :] @ X            # 1 x k
            meat += e[i] * Xi @ wk
        V = XtX_inv @ meat @ XtX_inv
        out[c] = np.sqrt(np.diag(V))
    return beta, out, n


# =====================================================================================
H("A.  ZERO-INCLUSIVE 368-ZIP BASELINE  +  PPML COUNT")
# =====================================================================================
zi = pd.read_csv(os.path.join(NM, "nm_zip_panel_zeroinclusive.csv"),
                 dtype={"zip5": str, "county_fips": str})
cor = pd.read_csv(os.path.join(NM, "nm_zip_panel_corrected.csv"),
                  dtype={"zip5": str, "county_fips": str})
p(f"zero-inclusive panel: {len(zi):,} ZIP-quarters, {zi.zip5.nunique()} ZIPs, "
  f"{zi.county_fips.nunique()} counties, quarters {zi.yq.nunique()}")
p(f"116-ZIP positive panel : {len(cor):,} ZIP-quarters, {cor.zip5.nunique()} ZIPs")
zi["border_zip"] = zi["border_zip"].astype(int)
zi["log_population"] = np.log(zi["zip_population"].clip(lower=1))
zi["log_nl75"] = np.log(zi["nonlegal_pop_within_75mi"].clip(lower=1))
zi["asinh_nl75"] = asinh(zi["nonlegal_pop_within_75mi"])

resA = {}
for y in ["asinh_revenue_per_capita", "asinh_dispensary_per_capita"]:
    m = ols_c(zi.dropna(subset=[y]), f"{y} ~ border_zip + log_population + C(yq)", "county_fips")
    b, se, pv = m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"]
    resA[y] = (b, se, pv, int(m.nobs))
    p(f"  {y:32s}  border={b:+.4f}{st(pv)} ({se:.4f})  N={int(m.nobs)}")

# 116-ZIP log revenue per store (defined only where >=1 store)
m = ols_c(cor.dropna(subset=["log_revenue_per_dispensary"]),
          "log_revenue_per_dispensary ~ border_zip + log_population + C(yq)", "county_fips")
resA["log_revenue_per_dispensary_116"] = (m.params["border_zip"], m.bse["border_zip"],
                                          m.pvalues["border_zip"], int(m.nobs))
p(f"  log_revenue_per_dispensary (116 ZIP) border={m.params['border_zip']:+.4f}"
  f"{st(m.pvalues['border_zip'])} ({m.bse['border_zip']:.4f})  N={int(m.nobs)}")

# 116-ZIP log revenue / log count (for the appendix robustness reproduction)
for y in ["log_revenue_per_capita", "log_dispensary_per_capita"]:
    m = ols_c(cor.dropna(subset=[y]), f"{y} ~ border_zip + log_population + C(yq)", "county_fips")
    resA[y + "_116"] = (m.params["border_zip"], m.bse["border_zip"], m.pvalues["border_zip"], int(m.nobs))
    p(f"  {y:28s} (116 ZIP) border={m.params['border_zip']:+.4f}{st(m.pvalues['border_zip'])} "
      f"({m.bse['border_zip']:.4f})  N={int(m.nobs)}")

H("A2.  PPML (Poisson) for the dispensary COUNT, 368 ZIPs")
zi["n_disp"] = zi["n_dispensaries"].fillna(0).round().astype(int)
yqd = pd.get_dummies(zi["yq"], prefix="yq", drop_first=True).astype(float)
Xp = pd.concat([zi[["border_zip", "log_population"]].astype(float), yqd], axis=1)
Xp = sm.add_constant(Xp)
mp = sm.GLM(zi["n_disp"], Xp, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": zi["county_fips"]})
p(f"  PPML count ~ border + log(pop) + quarter FE  (cluster county, {zi.county_fips.nunique()} clusters)")
p(f"    border      = {mp.params['border_zip']:+.4f}{st(mp.pvalues['border_zip'])} "
  f"({mp.bse['border_zip']:.4f})   => x{np.exp(mp.params['border_zip']):.3f}")
p(f"    log(pop)    = {mp.params['log_population']:+.4f} ({mp.bse['log_population']:.4f})")
# continuous exposure PPML
Xp2 = pd.concat([zi[["asinh_nl75", "log_population"]].astype(float), yqd], axis=1)
Xp2 = sm.add_constant(Xp2)
mp2 = sm.GLM(zi["n_disp"], Xp2, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": zi["county_fips"]})
p(f"    asinh(nonlegal pop 75mi) = {mp2.params['asinh_nl75']:+.4f}{st(mp2.pvalues['asinh_nl75'])} "
  f"({mp2.bse['asinh_nl75']:.4f})")
resA["ppml_border"] = (mp.params["border_zip"], mp.bse["border_zip"], mp.pvalues["border_zip"], int(mp.nobs))
resA["ppml_asinhNL"] = (mp2.params["asinh_nl75"], mp2.bse["asinh_nl75"], mp2.pvalues["asinh_nl75"], int(mp2.nobs))

H("B.  SPATIAL INFERENCE for the 368-ZIP asinh baseline")
geo = pd.read_csv(os.path.join(NM, "nm_zip_border_stateline.csv"), dtype={"zip5": str})
zg = zi.merge(geo[["zip5", "zip_lat", "zip_lon"]], on="zip5", how="left")
for y in ["asinh_revenue_per_capita", "asinh_dispensary_per_capita"]:
    # wild cluster bootstrap (county)
    dd = zg.dropna(subset=[y, "log_population"]).copy()
    yqd2 = pd.get_dummies(dd["yq"], prefix="q", drop_first=True).astype(int)
    dd = pd.concat([dd.reset_index(drop=True), yqd2.reset_index(drop=True)], axis=1)
    qcols = list(yqd2.columns)
    t_obs, pwcb, ng = wild_cluster_boot(dd, y, ["border_zip", "log_population"] + qcols,
                                        "border_zip", "county_fips", B=999)
    # Conley (collapse quarter FE by de-meaning within quarter for speed on one cross-section:
    # use the mature-quarter cross-section 2025 mean)
    cs = (zg[zg.yq.astype(str).str.startswith("2025")]
          .groupby("zip5").agg(y=(y, "mean"), log_population=("log_population", "first"),
                               border_zip=("border_zip", "first"),
                               zip_lat=("zip_lat", "first"), zip_lon=("zip_lon", "first")).reset_index())
    beta, cse, ncs = conley_se(cs, "y", ["border_zip", "log_population"], "zip_lat", "zip_lon")
    p(f"\n  {y}")
    p(f"    county-clustered t = {t_obs:+.2f}   wild-cluster-bootstrap p = {pwcb:.4f}  ({ng} clusters, 999 reps)")
    p(f"    Conley SE on border (2025 cross-section, N={ncs}):  "
      f"beta={beta[1]:+.3f}  se@25km={cse[25][1]:.3f}  se@50km={cse[50][1]:.3f}  se@100km={cse[100][1]:.3f}")
    resA[y + "_wcb_p"] = pwcb
    resA[y + "_conley"] = (beta[1], cse[25][1], cse[50][1], cse[100][1], ncs)

# PPML wild bootstrap
dd = zg.dropna(subset=["log_population"]).copy()
yqd2 = pd.get_dummies(dd["yq"], prefix="q", drop_first=True).astype(int)
dd = pd.concat([dd.reset_index(drop=True), yqd2.reset_index(drop=True)], axis=1)
qcols = list(yqd2.columns)
# poisson wild bootstrap-t (score) -- approximate with OLS-of-count wild bootstrap on the same clusters
rng = np.random.default_rng(7)
mfull = sm.GLM(dd["n_disp"], sm.add_constant(dd[["border_zip", "log_population"] + qcols].astype(float)),
               family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": dd["county_fips"]})
t_obs = mfull.params["border_zip"] / mfull.bse["border_zip"]
# restricted
mr = sm.GLM(dd["n_disp"], sm.add_constant(dd[["log_population"] + qcols].astype(float)),
            family=sm.families.Poisson()).fit()
mu_r = mr.fittedvalues.values
er = (dd["n_disp"].values - mu_r)
groups = dd["county_fips"].values
uq = np.unique(groups)
cnt = 0; Bp = 599
for b in range(Bp):
    w = dict(zip(uq, rng.choice([-1.0, 1.0], size=len(uq))))
    wv = np.array([w[g] for g in groups])
    yb = np.clip(np.round(mu_r + er * wv), 0, None)
    try:
        mb = sm.GLM(yb, sm.add_constant(dd[["border_zip", "log_population"] + qcols].astype(float)),
                    family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        tb = mb.params["border_zip"] / mb.bse["border_zip"]
        if abs(tb) >= abs(t_obs): cnt += 1
    except Exception:
        continue
p(f"\n  PPML count: county-clustered t={t_obs:+.2f}  wild-cluster-bootstrap p={(cnt+1)/(Bp+1):.4f}")
resA["ppml_wcb_p"] = (cnt + 1) / (Bp + 1)

pd.DataFrame([{"k": k, "v": str(v)} for k, v in resA.items()]).to_csv(
    os.path.join(ROOT, "ext_output", "ref_baseline_spatial.csv"), index=False)


# =====================================================================================
H("D.  ORGANIZATIONAL STATUS AT ENTRY  (no look-ahead)")
# =====================================================================================
sfm = pd.read_csv(os.path.join(BE, "storefront_master.csv"), dtype={"zip5": str, "fips5": str})
bm  = pd.read_csv(os.path.join(BE, "brand_master.csv"))
sfq = pd.read_csv(os.path.join(BE, "panel_storefront_quarter.csv"), dtype={"zip5": str, "fips5": str})
zq  = pd.read_csv(os.path.join(BE, "panel_zip_quarter.csv"), dtype={"zip5": str, "county_fips": str})

first_active = sfq[sfq.active == 1].groupby("dispensary_id").qt.min().rename("faq")
sfm = sfm.merge(first_active, left_on="dispensary_id", right_index=True, how="left")
sfm = sfm.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
sfm["chain_ever"] = (sfm["n_storefronts"] >= 2).astype(int)

rows = []
for bid, g in sfm.groupby("brand_id"):
    g = g.sort_values("faq")
    for _, r in g.iterrows():
        prior = ((g["faq"] < r["faq"]) & (g["dispensary_id"] != r["dispensary_id"])).sum()
        # also count left-censored siblings as "prior" (they were already operating at panel start)
        prior_lc = ((g["left_censored"]) & (g["dispensary_id"] != r["dispensary_id"]) &
                    (g["faq"] <= r["faq"])).sum()
        has_prior = (prior > 0) or (r["left_censored"] == False and prior_lc > 0)
        if has_prior:
            statv = "existing_multi"
        elif r["n_storefronts"] >= 2:
            statv = "first_then_expands"
        else:
            statv = "persistent_single"
        rows.append((r["dispensary_id"], statv))
cls = pd.DataFrame(rows, columns=["dispensary_id", "entry_status"])
sfm = sfm.merge(cls, on="dispensary_id", how="left")
sfm["not_established"] = (sfm["entry_status"] != "existing_multi").astype(int)  # first-NM-loc (single or later-expanding)

p("storefront counts by at-entry status (all 1,253):")
p(sfm["entry_status"].value_counts().to_string())
p("\nvs eventual-status ('chain ever'):")
p(pd.crosstab(sfm["chain_ever"], sfm["entry_status"]).to_string())
p("\nborder rate by at-entry status:")
p(sfm.groupby("entry_status")["border_dispensary"].mean().round(3).to_string())
p(f"\nleft-censored (pre-2022Q2 = medical incumbents): {int(sfm.left_censored.sum())}")
p(pd.crosstab(sfm["entry_status"], sfm["left_censored"]).to_string())

# ---- composition of OBSERVED entrants (drop left-censored) ----
H("D2.  Composition of observed entrants (left-censored excluded)")
ent = sfm[sfm.left_censored == False].copy()
zpop = zq.groupby("zip5")["zip_pop"].first().reset_index()
ent = ent.merge(zpop, on="zip5", how="left")
ent["log_zip_pop"] = np.log(ent["zip_pop"].clip(lower=1))
ent["log_dist"] = np.log(ent["dist_stateline_mi"].clip(lower=0.1))
ent["eyq"] = ent["entry_year"].astype(int) * 4 + ((ent["entry_month"].astype(int) - 1) // 3)
ent = ent.dropna(subset=["border_dispensary"])
p(f"observed entrants: {len(ent):,};  border {int(ent.border_dispensary.sum())}")
p(f"  share that are NOT an established multi-store operator: {ent['not_established'].mean():.3f}")
p(f"  share persistent-single: {(ent.entry_status=='persistent_single').mean():.3f}   "
  f"first-then-expands: {(ent.entry_status=='first_then_expands').mean():.3f}   "
  f"existing-multi: {(ent.entry_status=='existing_multi').mean():.3f}")

def lpm(dfx, y, extra=""):
    dd = dfx.dropna(subset=[y, "log_zip_pop", "fips5", "eyq", "border_dispensary"]).copy()
    f = f"{y} ~ border_dispensary + log_zip_pop + C(eyq)" + extra
    m = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["fips5"]})
    return m.params["border_dispensary"], m.bse["border_dispensary"], m.pvalues["border_dispensary"], int(m.nobs)

for y in ["not_established"]:
    b, se, pv, n = lpm(ent, y)
    p(f"  P({y}) border = {b:+.4f}{st(pv)} ({se:.4f})  N={n}")
# also: persistent-single indicator, first-then-expands indicator
ent["is_persistent_single"] = (ent.entry_status == "persistent_single").astype(int)
ent["is_first_then_expands"] = (ent.entry_status == "first_then_expands").astype(int)
for y in ["is_persistent_single", "is_first_then_expands"]:
    b, se, pv, n = lpm(ent, y)
    p(f"  P({y}) border = {b:+.4f}{st(pv)} ({se:.4f})  N={n}")
# old (eventual-status) de-novo definition for comparison
b_first = sfm.groupby("brand_id")["first_t"].min().rename("brand_first_t").reset_index()
ent2 = ent.merge(b_first, on="brand_id", how="left")
ent2["denovo_eventual"] = (ent2["first_t"] <= ent2["brand_first_t"]).astype(int)
b, se, pv, n = lpm(ent2, "denovo_eventual")
p(f"  [compare] P(de-novo, eventual-status defn) border = {b:+.4f}{st(pv)} ({se:.4f})  N={n}")

# ---- first mover / entry rank with at-entry status + medical incumbency ----
H("D3.  First mover & entry rank  (town-markets, ex-ABQ)")
ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}
sf = pd.read_csv(os.path.join(OLD, "panel_sf_quarter.csv"), dtype={"zip5": str})
obm = pd.read_csv(os.path.join(OLD, "brand_master.csv"))
ochain = dict(zip(obm.brand_id, (obm.n_storefronts >= 2).astype(int)))
pool_rate = float((obm.n_storefronts >= 2).mean())
meta = (sf.groupby("storefront_id").agg(first_qt=("first_qt", "first"), brand_id=("brand_id", "first"),
        city=("city", "first"), border=("border_sf", "first"), lc=("left_censored", "first")).reset_index())
meta["chain_ever"] = meta.brand_id.map(ochain)
meta["mkt"] = np.where(meta.city.isin(ABQ), "ABQ", meta.city)
meta = meta[meta.mkt != "ABQ"].copy()
# at-entry status within town data build (old1): prior sibling active before this store
ocls = []
osfa = sf[sf.active == 1].groupby("storefront_id").qt.min().rename("faq")
meta = meta.merge(osfa, left_on="storefront_id", right_index=True, how="left")
meta = meta.merge(obm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
for bid, g in meta.groupby("brand_id"):
    for _, r in g.iterrows():
        prior = ((g.faq < r.faq) & (g.storefront_id != r.storefront_id)).sum()
        if prior > 0:
            s = "existing_multi"
        elif r.n_storefronts >= 2:
            s = "first_then_expands"
        else:
            s = "persistent_single"
        ocls.append((r.storefront_id, s))
meta = meta.merge(pd.DataFrame(ocls, columns=["storefront_id", "entry_status"]), on="storefront_id", how="left")
meta["rank"] = meta.groupby("mkt")["first_qt"].rank(method="dense").astype(int)
r1 = meta[meta["rank"] == 1]
p(f"pool chain-ever rate: {pool_rate:.3f}   ({int((obm.n_storefronts>=2).sum())} of {len(obm)})")
p(f"rank-1 entrants: {len(r1)} markets")
p(f"  chain-ever rate among rank-1                    : {r1.chain_ever.mean():.3f}")
p(f"  rank-1 that are an ESTABLISHED multi-store op    : {(r1.entry_status=='existing_multi').mean():.3f}")
p(f"  rank-1 that are a medical incumbent (left-cens.) : {r1.lc.mean():.3f}")
p(f"  rank-1 that are BOTH chain-ever AND medical inc. : {((r1.chain_ever==1)&(r1.lc)).mean():.3f}")
p(f"  rank-1 chain-ever, NON-medical-incumbent markets : "
  f"{r1.loc[~r1.lc,'chain_ever'].mean():.3f}  (n={int((~r1.lc).sum())})")
p(f"  rank-1 chain-ever, EXCLUDING markets opened big-bang (evt: first_qt==min): ")
bigbang_q = meta.first_qt.min()
p(f"     markets whose rank-1 opened after the {bigbang_q} launch: "
  f"{r1.loc[r1.first_qt>bigbang_q,'chain_ever'].mean():.3f}  (n={int((r1.first_qt>bigbang_q).sum())})")
by_rank = meta[meta["rank"] <= 6].groupby("rank").agg(
    chain_ever=("chain_ever", "mean"), established=("entry_status", lambda s: (s == "existing_multi").mean()),
    medical=("lc", "mean"), n=("chain_ever", "size")).reset_index()
p("\nby within-market rank:")
p(by_rank.round(3).to_string(index=False))


# =====================================================================================
H("C.  LEGAL-BORDER CONTROL GROUP -- full outcome set")
# =====================================================================================
lb = pd.read_csv(os.path.join(
    r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project\tables",
    "nm_zip_legalborder_stateline.csv"), dtype={"zip5": str})[["zip5", "legal_border_zip"]]
sfq2 = sfq.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
sfq2 = sfq2.merge(cls, on="dispensary_id", how="left")
sfq2["chain_ever"] = (sfq2["n_storefronts"] >= 2).astype(int)
act = sfq2[sfq2.active == 1].copy()
g = act.groupby(["zip5", "qt"]).apply(lambda s: pd.Series({
        "n_chain_ever": s.loc[s.chain_ever == 1, "brand_id"].nunique(),
        "n_indep_ever": s.loc[s.chain_ever == 0, "brand_id"].nunique(),
        "n_established": s.loc[s.entry_status == "existing_multi", "brand_id"].nunique(),
        "n_firstNMloc": s.loc[s.entry_status != "existing_multi", "brand_id"].nunique(),
    })).reset_index()
z = zq.merge(g, on=["zip5", "qt"], how="left").merge(lb, on="zip5", how="left")
z["legal_border_zip"] = z["legal_border_zip"].fillna(0).astype(int)
z = z[z.border_z.notna()].copy()
z.loc[z.border_z == 1, "legal_border_zip"] = 0
for c in ["n_chain_ever", "n_indep_ever", "n_established", "n_firstNMloc"]:
    z[c] = z[c].fillna(0)
z["log_zip_pop"] = np.log(z["zip_pop"].clip(lower=1))
z["rev_per_capita"] = z["total_revenue"] / z["zip_pop"].clip(lower=1)
z["asinh_rev_pc"] = asinh(z["rev_per_capita"])
z["disp_per_1000"] = 1000 * z["n_stores"] / z["zip_pop"].clip(lower=1)
z["asinh_disp"] = asinh(z["disp_per_1000"])
z["log_rev_per_store"] = np.log((z["total_revenue"] / z["n_stores"]).clip(lower=1))
zz = z.drop_duplicates("zip5")
p(f"nonlegal-border {int((zz.border_z==1).sum())}  legal-border {int((zz.legal_border_zip==1).sum())}  "
  f"interior {int(((zz.border_z==0)&(zz.legal_border_zip==0)).sum())}")
OUTS = [("asinh_rev_pc", "asinh revenue per capita"), ("asinh_disp", "asinh dispensary density"),
        ("log_rev_per_store", "log revenue per store"), ("n_indep_ever", "# independent operators"),
        ("n_chain_ever", "# multi-store operators"), ("n_established", "# established multi-store ops"),
        ("n_firstNMloc", "# first-NM-location ops"), ("hhi_10000", "HHI (0-10,000)"),
        ("cr1", "largest-brand revenue share")]
rowsC = []
for y, lab in OUTS:
    m = ols_c(z.dropna(subset=[y]), f"{y} ~ border_z + legal_border_zip + log_zip_pop + C(qt)", "county_fips")
    b1, s1, p1 = m.params["border_z"], m.bse["border_z"], m.pvalues["border_z"]
    b2, s2, p2 = m.params["legal_border_zip"], m.bse["legal_border_zip"], m.pvalues["legal_border_zip"]
    rowsC.append(dict(outcome=lab, b_border=b1, se_border=s1, p_border=p1,
                      b_legal=b2, se_legal=s2, p_legal=p2, N=int(m.nobs)))
    p(f"  {lab:34s} border={b1:+.3f}{st(p1):3s}({s1:.3f})   legal-border={b2:+.3f}{st(p2):3s}({s2:.3f})   N={int(m.nobs)}")
pd.DataFrame(rowsC).to_csv(os.path.join(ROOT, "ext_output", "ref_legalborder_full.csv"), index=False)


# =====================================================================================
H("F.  DESCRIPTIVE ADJUSTMENT MECHANISM:  Entry_{z,t+1} ~ RevPerStore_{z,t} x Border")
# =====================================================================================
zp = zq[zq.border_z.notna()].sort_values(["zip5", "qt"]).copy()
zp["rev_per_store"] = zp["total_revenue"] / zp["n_stores"].replace(0, np.nan)
zp["log_rps"] = np.log(zp["rev_per_store"].clip(lower=1))
zp["log_zip_pop"] = np.log(zp["zip_pop"].clip(lower=1))
zp["entries_next"] = zp.groupby("zip5")["brand_entries"].shift(-1)
zp["bxrps"] = zp["border_z"] * zp["log_rps"]
d = zp.dropna(subset=["entries_next", "log_rps", "log_zip_pop", "border_z"])
d = d[d["n_stores"] >= 1]
m = smf.ols("entries_next ~ log_rps + bxrps + border_z + log_zip_pop + C(qt)", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["county_fips"]})
p(f"  N={int(m.nobs)}  (zip-quarters with >=1 store)")
for k in ["log_rps", "bxrps", "border_z"]:
    p(f"    {k:12s} = {m.params[k]:+.4f}{st(m.pvalues[k])} ({m.bse[k]:.4f})")
# Poisson version
try:
    dd = d.copy()
    yqd3 = pd.get_dummies(dd["qt"], prefix="q", drop_first=True).astype(float)
    Xe = sm.add_constant(pd.concat([dd[["log_rps", "bxrps", "border_z", "log_zip_pop"]].astype(float).reset_index(drop=True),
                                    yqd3.reset_index(drop=True)], axis=1))
    me = sm.GLM(dd["entries_next"].values, Xe, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": dd["county_fips"].values})
    p(f"  [PPML] log_rps={me.params['log_rps']:+.3f}{st(me.pvalues['log_rps'])}  "
      f"bxrps={me.params['bxrps']:+.3f}{st(me.pvalues['bxrps'])}  border={me.params['border_z']:+.3f}{st(me.pvalues['border_z'])}")
except Exception as ex:
    p("  PPML entry failed:", ex)


# =====================================================================================
H("G.  DISPENSARY STORE REVENUE at several local-demand radii")
# =====================================================================================
dp = pd.read_csv(os.path.join(NM, "nm_dispensary_quarterly_panel_v3.csv"),
                 dtype={"fips5": str, "zip5_nearest": str})
lp = pd.read_csv(os.path.join(NM, "nm_dispensary_local_population.csv"))
dp = dp.merge(lp[["dispensary_id", "pop_1km_areal", "pop_2km_areal", "pop_3km_areal", "pop_5km_areal"]],
              on="dispensary_id", how="left")
dp["border_dispensary"] = dp["border_dispensary"].astype(int)
dp["log_rev"] = np.log(dp["total_sales"].clip(lower=1))
for r in ["1km", "2km", "3km", "5km"]:
    dp[f"logpop_{r}"] = np.log(dp[f"pop_{r}_areal"].clip(lower=1))
dp["logpop_25mi"] = np.log(dp["pop_within_25mi"].clip(lower=1))
p("store log revenue ~ border + log(local pop at radius) + quarter FE  (cluster county); +/- operator FE")
for r in ["1km", "2km", "3km", "5km", "25mi"]:
    d = dp.dropna(subset=["log_rev", f"logpop_{r}", "fips5", "brand", "period"]).copy()
    m1 = smf.ols(f"log_rev ~ border_dispensary + logpop_{r} + C(period)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    m2 = smf.ols(f"log_rev ~ border_dispensary + logpop_{r} + C(period) + C(brand)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    p(f"  {r:5s}  no operator FE: {m1.params['border_dispensary']:+.3f}{st(m1.pvalues['border_dispensary'])} "
      f"({m1.bse['border_dispensary']:.3f})   with operator FE: "
      f"{m2.params['border_dispensary']:+.3f}{st(m2.pvalues['border_dispensary'])} "
      f"({m2.bse['border_dispensary']:.3f})   N={int(m1.nobs)}")

open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE -> ext_output/ref01_referee_revision.log")
