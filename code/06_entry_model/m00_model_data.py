r"""
m00_model_data.py -- ZIP-quarter panel for the entry-and-adjustment model (all 368 New Mexico ZIP codes).

Stock definition used by the model: a storefront is ALIVE in quarter t if its first quarter with sales <= t <= its
last quarter with sales (temporary zero-sales quarters inside that window are counted as alive). The published
reduced-form tables use ACTIVE stores (positive sales in the quarter); the two differ by temporary gaps only.
Openings: storefronts whose first quarter with sales is t > 2022Q2, split into
  F  : first location of an operator (single-store operators and first stores of later-expanding operators),
  EM : additional store of an established operator that held a medical incumbent in April 2022,
  EN : additional store of any other established operator.
Exits: a storefront exits after its last quarter with sales (not observed for stores alive in 2026Q2).
Output: data/analysis/model_panel.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import pickle
import numpy as np
import pandas as pd
from paths import ANALYSIS, INTER
from _common import load_zx, load_storefronts, load_store_quarters, classify, build_zip_covariates, QS, QMIN, QMAX

zx = load_zx(); sf, bm = load_storefronts(zx); sq, sf = load_store_quarters(sf)
meta = classify(sq, bm); cov = build_zip_covariates(zx, sf)
tt = pd.read_csv(INTER / "zip_travel_tribal.csv", dtype={"zip5": str})
cov = cov.merge(tt[["zip5", "tribal_centroid", "tt_decay_60"]], on="zip5", how="left")

st = sq.groupby("storefront_id").agg(first=("qt", "min"), last=("qt", "max"), zip5=("zip5", "first"),
                                     brand_id=("brand_id", "first"), med=("med_incumbent", "first"),
                                     rev=("revenue", "sum")).reset_index()
st = st.merge(meta[["storefront_id", "status_entry"]], on="storefront_id")
med_brands = set(st.loc[st.med == 1, "brand_id"])
st["type"] = np.where(st["first"] == QMIN, "initial",
              np.where(st.status_entry != "established", "F",
               np.where(st.brand_id.isin(med_brands), "EM", "EN")))
st["exit_q"] = np.where(st["last"] < QMAX, st["last"] + 1, np.nan)

grid = pd.MultiIndex.from_product([sorted(zx.zip5), QS], names=["zip5", "qt"]).to_frame(index=False)
alive = []
for q in QS:
    a = st[(st["first"] <= q) & (st["last"] >= q)].groupby("zip5").size().rename("N")
    alive.append(a.reset_index().assign(qt=q))
alive = pd.concat(alive)
P = grid.merge(alive, on=["zip5", "qt"], how="left").fillna({"N": 0})
for k in ("F", "EM", "EN"):
    o = st[st.type == k].groupby(["zip5", "first"]).size().rename(f"open_{k}").reset_index().rename(columns={"first": "qt"})
    P = P.merge(o, on=["zip5", "qt"], how="left")
ex = st.dropna(subset=["exit_q"]).groupby(["zip5", "exit_q"]).size().rename("exits").reset_index().rename(columns={"exit_q": "qt"})
P = P.merge(ex, on=["zip5", "qt"], how="left").fillna(0)
P["open_E"] = P.open_EM + P.open_EN
P["open_all"] = P.open_F + P.open_E
rev = sq.groupby(["zip5", "qt"]).revenue.sum().rename("R")
P = P.merge(rev.reset_index(), on=["zip5", "qt"], how="left").fillna({"R": 0})
P = P.sort_values(["zip5", "qt"])
P["N_lag"] = P.groupby("zip5").N.shift(1)
P["t"] = P.qt - QMIN                                   # quarters since adult-use sales began
P = P.merge(zx[["zip5", "zip_pop", "group", "county_fips", "lat", "lon"]], on="zip5")
P = P.merge(cov[["zip5", "nlpop_75", "lgpop_75", "log_density", "dist_abq", "dist_bigcity", "zbp19_total_retail",
                 "tribal_centroid", "n_med_incumbent", "tt_decay_60"]], on="zip5")
P["lL"] = np.log(P.zip_pop.clip(lower=1))
P["xB"] = np.log1p(P.nlpop_75 / 1000)
P["xLB"] = np.log1p(P.lgpop_75 / 1000)
P["dabq"], P["dbig"] = P.dist_abq / 100, P.dist_bigcity / 100
P["zbp"] = np.arcsinh(P.zbp19_total_retail / P.zip_pop.clip(lower=1) * 1000)
P["nonlegal"] = (P.group == "nonlegal").astype(int)

# store-quarter panel for the exit model
rows = []
for q in QS[:-1]:
    a = st[(st["first"] <= q) & (st["last"] >= q)].copy()
    a["qt"] = q; a["exit_next"] = (a["last"] == q).astype(int); a["age"] = q - a["first"]
    rows.append(a)
SQ = pd.concat(rows).merge(P[["zip5", "qt", "N", "xB", "lL", "county_fips", "t"]], on=["zip5", "qt"], how="left")

pickle.dump(dict(P=P, st=st, SQ=SQ), open(ANALYSIS / "model_panel.pkl", "wb"))
print(f"ZIP-quarters {len(P)}; openings F {int(P.open_F.sum())}, EM {int(P.open_EM.sum())}, EN {int(P.open_EN.sum())}; "
      f"initial stock {int((st.type=='initial').sum())}; exits {int(P.exits.sum())}")
print("alive vs active stores, statewide by quarter (first/last):",
      int(P[P.qt == QMIN].N.sum()), sq[sq.qt == QMIN].storefront_id.nunique(), "|",
      int(P[P.qt == QMAX].N.sum()), sq[sq.qt == QMAX].storefront_id.nunique())
