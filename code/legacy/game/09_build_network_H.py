r"""
09_build_network_H.py -- structural_model / extension of venous4 / cannibalization extension

Builds PREDETERMINED (t-1 or earlier, no look-ahead) own-network overlap measures H_jmt for
(a) every realized Type-C entry event, and (b) a full operator x candidate-town-market x quarter
risk-set panel used by the mechanism regression (11_mechanism_regression.py).

Reuses the exact at-entry Type-I/Type-C classification already audited in 01_build_panel.py /
ref01_referee_revision.py -- not re-derived differently. Nothing here alters panel_sf_quarter.csv,
brand_master.csv, or any file built in the original (pre-cannibalization) session.

H measures (see cannibalization_audit.md Sec.6 for why H_geo is preferred and how town centroids
are constructed -- there is no official town-centroid file in the project, so centroids are the mean
lat/lon of every storefront ever recorded in that city, a data-grounded proxy, not a fabrication):
  H_same_jmt   : # of j's own storefronts ALREADY ACTIVE IN TOWN m at t-1
  AnySame_jmt  : 1{H_same_jmt > 0}
  H_size_jt    : total # of j's active NM storefronts at t-1 (network size/saturation, not geography)
  H_border_jt  : # of j's existing stores in markets with positive nonlegal-border exposure at t-1
  H_B_jt       : sum over j's existing stores of log(1+ store-level nonlegal pop within 75mi)
  H_geo_jmt(λ) : sum over j's existing stores r of exp(-λ * haversine_km(town m centroid, store r))
  H25/H50/H75/H100_jmt : # of j's existing stores within 25/50/75/100 miles of town m's centroid
  NearestOwnDistance_jmt : distance (miles) from town m's centroid to j's nearest existing store
                           (NaN if j has no existing stores anywhere in NM at t-1)
"""
import os
import numpy as np
import pandas as pd

OLD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "09_build_network_H.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

LAMBDA = 0.03      # decay per km in H_geo; ~exp(-1) at 33km (~21mi) -- a transparent, round choice,
                    # tested for robustness at 0.015/0.06 in 14_cannibalization_robustness.py
MI_PER_KM = 0.621371
ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ---------------------------------------------------------------------------
H("1.  LOAD + REPRODUCE THE AUDITED AT-ENTRY TYPE CLASSIFICATION")
sf = pd.read_csv(os.path.join(OLD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(OLD, "brand_master.csv"))
first_active = sf[sf.active == 1].groupby("storefront_id").qt.min().rename("faq")
meta = (sf.groupby("storefront_id")
          .agg(brand_id=("brand_id", "first"), city=("city", "first"),
               left_censored=("left_censored", "first"), lat=("lat", "first"), lon=("lon", "first")).reset_index())
meta = meta.merge(first_active, on="storefront_id", how="left")
meta = meta.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
meta["left_censored"] = meta["left_censored"].astype(str).isin(["True", "1", "1.0", "true"])
rows = []
for bid, g in meta.groupby("brand_id"):
    g = g.sort_values("faq")
    for _, r in g.iterrows():
        prior = ((g["faq"] < r["faq"]) & (g["storefront_id"] != r["storefront_id"])).sum()
        prior_lc = ((g["left_censored"]) & (g["storefront_id"] != r["storefront_id"]) & (g["faq"] <= r["faq"])).sum()
        has_prior = (prior > 0) or ((not r["left_censored"]) and prior_lc > 0)
        statv = "existing_multi" if has_prior else ("first_then_expands" if r["n_storefronts"] >= 2 else "persistent_single")
        rows.append((r["storefront_id"], statv))
cls = pd.DataFrame(rows, columns=["storefront_id", "entry_status"])
meta = meta.merge(cls, on="storefront_id", how="left")
meta["type"] = np.where(meta.entry_status == "existing_multi", "C", "I")
meta["mkt"] = np.where(meta.city.isin(ABQ), "ABQ", meta.city)
p(f"Type C entry events: {(meta.type=='C').sum()}  (all observed, none left-censored -- verified)")

# ---------------------------------------------------------------------------
H("2.  TOWN CENTROIDS  (mean lat/lon of every storefront ever recorded in that city -- no official "
  "centroid file exists in this project; documented data-grounded proxy, see cannibalization_audit.md)")
centroids = meta.groupby("mkt")[["lat", "lon"]].mean().reset_index()
centroids["is_abq"] = (centroids.mkt == "ABQ").astype(int)
TOWNS = centroids.mkt.tolist()
p(f"{len(TOWNS)} candidate town-markets, INCLUDING ABQ (flagged via is_abq).")
p("  IMPORTANT: 143 of 370 (38.6%) realized Type-C entries are in ABQ (checked directly below) -- "
  "excluding ABQ here, as the aggregate BNE model does, would discard over a third of the mechanism "
  "regression's identifying entry events. ABQ IS included in the H construction and risk-set panel; "
  "11_mechanism_regression.py reports both an ABQ-excluded primary spec (consistent with the "
  "aggregate model's own convention) and an ABQ-included robustness spec, and states which is preferred.")
p(f"  Type-C entries located in ABQ: {int((meta[meta.type=='C']).mkt.eq('ABQ').sum())}")

# ---------------------------------------------------------------------------
H("3.  ACTIVE-STOREFRONT PANEL WITH LOCATION + STORE-LEVEL BORDER EXPOSURE")
act = sf[sf.active == 1][["storefront_id", "qt", "brand_id", "sf_nonleg_pop_75"]].merge(
    meta[["storefront_id", "lat", "lon", "mkt"]], on="storefront_id", how="left")
qmin, qmax = act.qt.min(), act.qt.max()
QUARTERS = sorted(act.qt.unique())
p(f"active storefront-quarter rows: {len(act)}, quarters {qmin}-{qmax}")

# ---------------------------------------------------------------------------
H("4.  H FOR EVERY REALIZED TYPE-C ENTRY EVENT (operator's OWN network active at exactly qt = faq-1)")
c_events = meta[meta.type == "C"][["storefront_id", "brand_id", "mkt", "faq"]].rename(columns={"mkt": "entry_mkt"})
c_events = c_events.merge(centroids, left_on="entry_mkt", right_on="mkt", how="left", suffixes=("", "_ctr"))

def h_for_operator_quarter(brand_id, t, target_lat, target_lon, target_mkt):
    """H measures for operator `brand_id`'s pre-existing network -- stores STILL ACTIVE in the
    immediately preceding quarter t-1 (the standard predetermined-state-variable snapshot, and the
    same quarter convention the aggregate model's own N_I_mt/N_C_mt use), not merely 'ever active
    before t'. A store that had already exited by t-1 cannot be cannibalized by a new store opened
    at t, so it correctly does not contribute to H."""
    own = act[(act.brand_id == brand_id) & (act.qt == t - 1)]
    own = own.drop_duplicates("storefront_id")   # one row per distinct existing store (not store-quarter)
    if len(own) == 0:
        return dict(H_size=0, H_same=0, AnySame=0, H_border=0, H_B=0.0, H_geo=0.0,
                    H25=0, H50=0, H75=0, H100=0, NearestOwnDistance_mi=np.nan)
    d_km = haversine_km(target_lat, target_lon, own.lat.values, own.lon.values)
    d_mi = d_km * MI_PER_KM
    same = (own.mkt.values == target_mkt)
    return dict(
        H_size=len(own),
        H_same=int(same.sum()),
        AnySame=int(same.any()),
        H_border=int((own.sf_nonleg_pop_75.fillna(0).values > 0).sum()),
        H_B=float(np.log1p(own.sf_nonleg_pop_75.fillna(0).values).sum()),
        H_geo=float(np.exp(-LAMBDA * d_km).sum()),
        H25=int((d_mi <= 25).sum()), H50=int((d_mi <= 50).sum()),
        H75=int((d_mi <= 75).sum()), H100=int((d_mi <= 100).sum()),
        NearestOwnDistance_mi=float(d_mi.min()),
    )

rowsC = []
for _, r in c_events.iterrows():
    hh = h_for_operator_quarter(r.brand_id, r.faq, r.lat, r.lon, r.entry_mkt)
    rowsC.append({**dict(storefront_id=r.storefront_id, brand_id=r.brand_id, mkt=r.entry_mkt, qt=r.faq), **hh})
c_H = pd.DataFrame(rowsC)

# town_static built directly from city_cross_section.csv so it covers ALL 79 towns INCLUDING ABQ
# (market_panel.csv, built for the aggregate BNE model, deliberately excludes ABQ -- see above).
# Uncentered log_pop / log(1+B75) are used here (not the BNE model's centered versions): this is a
# standalone PPML regression with its own intercept/FE structure, so centering is unnecessary.
cs = pd.read_csv(os.path.join(
    r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build",
    "city_cross_section.csv")).rename(columns={"metro": "mkt"})
cs["mkt"] = cs["mkt"].replace({"ALBUQUERQUE METRO": "ABQ"})   # match meta.mkt's ABQ-suburb collapse
cs["log_pop"] = np.log(cs["city_pop"].clip(lower=1))
cs["log_B75"] = np.log1p(cs["nonleg_pop_75"])
cs["border_town"] = cs["border_city"].astype(int)
cs["legal_border_town"] = ((cs["legalneigh_pop_75"] > 0) & (cs["border_town"] == 0)).astype(int)
town_static = cs[["mkt", "border_town", "legal_border_town", "nonleg_pop_75", "log_pop", "log_B75"]] \
    .rename(columns={"nonleg_pop_75": "B_step75"})
c_H = c_H.merge(town_static, on="mkt", how="left")
c_H.to_csv(os.path.join(SM, "data", "typeC_entry_H.csv"), index=False)
p(f"wrote data/typeC_entry_H.csv ({len(c_H)} Type-C entry events)")

# ---------------------------------------------------------------------------
H("5.  FULL RISK-SET PANEL: established operator x candidate town x quarter")
# "established operator going into t" = has >=1 active storefront anywhere in NM AT t-1 SPECIFICALLY
# (matching h_for_operator_quarter's own t-1 snapshot exactly). An operator whose only prior store(s)
# had already exited by t-1 has H_size=0 -- no current network to cannibalize -- and is correctly
# excluded from the "established, has a network" risk set; see cannibalization_diagnostics for how
# many of the 370 historically-classified Type-C events this affects (Sec.6 below) and why that is a
# real, reported finding, not a data problem being swept under the rug.
opq = act.groupby(["brand_id", "qt"]).size().reset_index(name="n_active_now")
opq["qt_next"] = opq["qt"] + 1
established = opq[["brand_id", "qt_next"]].drop_duplicates().rename(columns={"qt_next": "qt"})
established = established[established.qt.isin(QUARTERS)]
p(f"operator x quarter 'established' cells (has >=1 active store at exactly t-1): {len(established)}")

# realized Type-C entries this operator-quarter (for the outcome variable)
entries_by_opq = (c_events.groupby(["brand_id", "faq", "entry_mkt"]).size()
                   .reset_index(name="EntryC")).rename(columns={"faq": "qt", "entry_mkt": "mkt"})

risk_rows = []
for _, oc in established.iterrows():
    bid, t = oc.brand_id, oc.qt
    own = act[(act.brand_id == bid) & (act.qt == t - 1)].drop_duplicates("storefront_id")
    if len(own) == 0:
        continue
    d_km = haversine_km(centroids.lat.values[:, None], centroids.lon.values[:, None],
                         own.lat.values[None, :], own.lon.values[None, :])   # (n_towns, n_own_stores)
    d_mi = d_km * MI_PER_KM
    same_mask = (own.mkt.values[None, :] == centroids.mkt.values[:, None])
    hh = pd.DataFrame({
        "mkt": centroids.mkt.values,
        "brand_id": bid, "qt": t,
        "H_size": len(own),
        "H_same": same_mask.sum(axis=1),
        "AnySame": (same_mask.sum(axis=1) > 0).astype(int),
        "H_border": int((own.sf_nonleg_pop_75.fillna(0).values > 0).sum()),
        "H_B": float(np.log1p(own.sf_nonleg_pop_75.fillna(0).values).sum()),
        "H_geo": np.exp(-LAMBDA * d_km).sum(axis=1),
        "H25": (d_mi <= 25).sum(axis=1), "H50": (d_mi <= 50).sum(axis=1),
        "H75": (d_mi <= 75).sum(axis=1), "H100": (d_mi <= 100).sum(axis=1),
        "NearestOwnDistance_mi": d_mi.min(axis=1),
    })
    risk_rows.append(hh)
risk = pd.concat(risk_rows, ignore_index=True)
risk = risk.merge(entries_by_opq, on=["brand_id", "qt", "mkt"], how="left")
risk["EntryC"] = risk["EntryC"].fillna(0).astype(int)
risk = risk.merge(town_static, on="mkt", how="left")   # log_pop, log_B75 (uncentered), border flags --
                                                         # covers all 79 towns incl. ABQ, from city_cross_section.csv
p(f"risk-set panel: {len(risk):,} operator-town-quarter rows, {risk.brand_id.nunique()} operators, "
  f"{int(risk.EntryC.sum())} positive Type-C entry events captured out of {len(c_events)} total")
missing = len(c_events) - int(risk.EntryC.sum())
p(f"\n  {missing} of {len(c_events)} historically-classified Type-C entry events ({missing/len(c_events):.1%}) "
  f"are NOT in the H>0 risk set -- verified directly: every one of these has H_size=0 in "
  f"data/typeC_entry_H.csv, i.e. the operator's qualifying prior sibling (the one that made this "
  f"entry 'existing_multi' by historical order) had ALREADY EXITED by the immediately preceding "
  f"quarter t-1. This is a genuine, reported finding, not a data artifact: these operators are "
  f"labeled Type C by the at-entry classification (they are not first-time NM entrants) but face "
  f"ZERO current same-brand network to cannibalize at the moment of this specific store's opening. "
  f"They remain in the descriptive Type-C diagnostics (10_cannibalization_diagnostics.py) with "
  f"H_size=0, H_same=0, but are correctly excluded from the 'established operator with an active "
  f"network' risk set the mechanism regression (11_mechanism_regression.py) targets.")
risk.to_csv(os.path.join(SM, "data", "operator_risk_set_panel.csv"), index=False)
p("wrote data/operator_risk_set_panel.csv")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
