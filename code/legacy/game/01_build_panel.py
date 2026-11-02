r"""
01_build_panel.py -- structural_model / extension of venous4

Builds the market-period analysis panel for the Seim-style incomplete-information
entry/type structural model.  Market unit = New Mexico town-market (the same
unit used by ext02_structural_model.py / city_cross_section.csv), excluding the
Albuquerque metro (top-coded, not comparable -- existing project convention).
Period = quarter.

Organizational TYPE is fixed at first entry (no look-ahead), reusing the exact
at-entry classification algorithm audited and validated in
ext_code/ref01_referee_revision.py Part D:
  - "existing_multi"     : the operator already had >=1 OTHER active storefront
                            (in this brand) at this storefront's first active
                            quarter  ->  TYPE C  (chain / established multi-store)
  - "first_then_expands"
    "persistent_single"  : operator had no other active storefront yet when this
                            store opened, whether or not it expands later
                            ->  TYPE I  (independent at the entry margin)
This matches the model's assumption that org type is the choice made AT ENTRY,
not a running eventual-status label; it is why the model does not require
I->C transitions for identification (see structural_entry_model.tex, Sec. 3).

Inputs (read-only; nothing here is modified):
  old1/build/panel_sf_quarter.csv   -- storefront x quarter activity panel
  old1/build/brand_master.csv       -- brand roll-up (used only for a cross-check)
  old1/build/city_cross_section.csv -- town-level time-invariant market characteristics
  old1/build/panel_city_quarter.csv -- town x quarter revenue (kept for reference only)

Output:
  structural_model/data/market_panel.csv   -- the estimation panel
  structural_model/logs/01_build_panel.log
"""
import os
import numpy as np
import pandas as pd

OLD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "01_build_panel.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}

# ---------------------------------------------------------------------------
H("1.  LOAD")
sf = pd.read_csv(os.path.join(OLD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(OLD, "brand_master.csv"))
cs = pd.read_csv(os.path.join(OLD, "city_cross_section.csv"))
p(f"panel_sf_quarter: {len(sf):,} storefront-quarter rows, "
  f"{sf.storefront_id.nunique()} storefronts, {sf.qt.nunique()} quarters "
  f"({sf.qt.min()}-{sf.qt.max()})")
p(f"city_cross_section: {len(cs)} towns")

# ---------------------------------------------------------------------------
H("2.  AT-ENTRY ORGANIZATIONAL TYPE  (reproduces ref01_referee_revision.py Part D, "
  "applied here to the old1/build storefront panel used by the town-market unit)")
first_active = sf[sf.active == 1].groupby("storefront_id").qt.min().rename("faq")
meta = (sf.groupby("storefront_id")
          .agg(brand_id=("brand_id", "first"), city=("city", "first"),
               left_censored=("left_censored", "first"), border_sf=("border_sf", "first"))
          .reset_index())
meta = meta.merge(first_active, on="storefront_id", how="left")
meta = meta.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
meta["left_censored"] = meta["left_censored"].astype(str).isin(["True", "1", "1.0", "true"])

rows = []
for bid, g in meta.groupby("brand_id"):
    g = g.sort_values("faq")
    for _, r in g.iterrows():
        prior = ((g["faq"] < r["faq"]) & (g["storefront_id"] != r["storefront_id"])).sum()
        prior_lc = ((g["left_censored"]) & (g["storefront_id"] != r["storefront_id"]) &
                    (g["faq"] <= r["faq"])).sum()
        has_prior = (prior > 0) or ((not r["left_censored"]) and prior_lc > 0)
        if has_prior:
            statv = "existing_multi"
        elif r["n_storefronts"] >= 2:
            statv = "first_then_expands"
        else:
            statv = "persistent_single"
        rows.append((r["storefront_id"], statv))
cls = pd.DataFrame(rows, columns=["storefront_id", "entry_status"])
meta = meta.merge(cls, on="storefront_id", how="left")
meta["type"] = np.where(meta["entry_status"] == "existing_multi", "C", "I")
p(meta["entry_status"].value_counts().to_string())
p(f"\nTYPE C (established multi-store at entry): {int((meta.type=='C').sum())}")
p(f"TYPE I (independent at entry, incl. later-expanding): {int((meta.type=='I').sum())}")
p("(cf. ref01_referee_revision.py Part D on the brand_entry_analysis build: 389 / 391+473 -- "
  "same construction, different underlying storefront roster (old1 vs brand_entry_analysis), "
  "so counts are close but not identical; both are reported in structural_entry_model.tex Sec. 2.)")

# ---------------------------------------------------------------------------
H("3.  TOWN x QUARTER ACTIVE COUNTS BY TYPE")
sf2 = sf.merge(meta[["storefront_id", "type"]], on="storefront_id", how="left")
sf2["mkt"] = np.where(sf2["city"].isin(ABQ), "ABQ", sf2["city"])
act = sf2[sf2.active == 1].copy()
g = (act.groupby(["mkt", "qt"])
        .apply(lambda s: pd.Series({
            "N_I": (s["type"] == "I").sum(),
            "N_C": (s["type"] == "C").sum(),
        })).reset_index())
g["N_I"] = g["N_I"].astype(int); g["N_C"] = g["N_C"].astype(int)
g["N_total"] = g["N_I"] + g["N_C"]
p(f"{g.mkt.nunique()} distinct markets (incl. ABQ), {g.qt.nunique()} quarters, {len(g)} market-quarters")

# fill zero-store town-quarters: build the full mkt x qt grid so absence of entry is a zero, not missing
all_qt = sorted(g.qt.unique())
all_mkt = sorted(g.mkt.unique())
grid = pd.MultiIndex.from_product([all_mkt, all_qt], names=["mkt", "qt"]).to_frame(index=False)
g = grid.merge(g, on=["mkt", "qt"], how="left")
for c in ["N_I", "N_C", "N_total"]:
    g[c] = g[c].fillna(0).astype(int)

# ---------------------------------------------------------------------------
H("4.  MERGE TOWN-LEVEL MARKET CHARACTERISTICS  (time-invariant; old1's own town build)")
cs2 = cs.rename(columns={"metro": "mkt"}).copy()
panel = g.merge(cs2, on="mkt", how="left")
panel = panel[panel["is_abq"] == 0].copy()          # drop ABQ metro (top-coded, excluded like ext02)
panel = panel.dropna(subset=["city_pop"]).copy()
p(f"after dropping ABQ / unmatched towns: {panel.mkt.nunique()} towns, {len(panel)} market-quarters")

# ring-weighted smoother continuous B_mt built ONLY from the project's own 25/50/75/100-mile bands
# (declining step weights across the existing concentric rings -- no new geography invented)
panel["B_step75"] = panel["nonleg_pop_75"]                       # main definition (paper's own)
panel["B_ring"] = (panel["nonleg_pop_25"]
                    + 0.75 * (panel["nonleg_pop_50"] - panel["nonleg_pop_25"]).clip(lower=0)
                    + 0.50 * (panel["nonleg_pop_75"] - panel["nonleg_pop_50"]).clip(lower=0)
                    + 0.25 * (panel["nonleg_pop_100"] - panel["nonleg_pop_75"]).clip(lower=0))
panel["B_50"] = panel["nonleg_pop_50"]
panel["B_100"] = panel["nonleg_pop_100"]
panel["L_legal75"] = panel["legalneigh_pop_75"]                  # legal-neighbor population (placebo)
panel["log_pop"] = np.log(panel["city_pop"].clip(lower=1))
panel["log_B75"] = np.log1p(panel["B_step75"])
panel["log_Bring"] = np.log1p(panel["B_ring"])
panel["border_town"] = panel["border_city"].astype(int)
panel["legal_border_town"] = ((panel["L_legal75"] > 0) & (panel["border_town"] == 0)).astype(int)
panel["indep_share"] = np.where(panel["N_total"] > 0, panel["N_I"] / panel["N_total"].clip(lower=1), np.nan)

# ---------------------------------------------------------------------------
H("5.  MATURE-QUARTER RESTRICTION")
qmin, qmax = panel.qt.min(), panel.qt.max()
DROP_Q = sorted(panel.qt.unique())[:2]   # exclude the April-2022 big-bang launch window (first 2
                                          # quarters), matching ext05_dynamic_transition.py's own
                                          # exclusion of qtrs 0-1 as a non-equilibrium scramble period
p(f"quarters in panel: {qmin}-{qmax} ({panel.qt.nunique()} total); "
  f"dropping launch quarters {DROP_Q} as in ext05_dynamic_transition.py")
panel["mature"] = (~panel.qt.isin(DROP_Q)).astype(int)

# ---------------------------------------------------------------------------
H("5b.  CENTER THE PROFIT-INDEX COVARIATES  (numerical scaling, not economic content)")
# kappa_tau (the profit intercept) otherwise has to offset beta_tau*log_pop over log_pop's full
# range (roughly 7-12 in levels), pushing it to extreme, poorly-identified values and, in initial
# runs, into the optimizer's bounds. Centering log_pop and log(1+B) on the mature-sample mean makes
# kappa_tau interpretable as "net profitability at a representative New Mexico town" and removes
# the artificial extreme-intercept/steep-slope tradeoff. This is a rescaling of the SAME linear
# index (kappa_tau' = kappa_tau + beta_tau*mean_logpop + delta_tau*mean_logB), not a new assumption.
mat_mean_logpop = float(panel.loc[panel.mature == 1, "log_pop"].mean())
mat_mean_logB75 = float(panel.loc[panel.mature == 1, "log_B75"].mean())
mat_mean_logBring = float(panel.loc[panel.mature == 1, "log_Bring"].mean())
panel["log_pop_c"] = panel["log_pop"] - mat_mean_logpop
panel["log_B75_c"] = panel["log_B75"] - mat_mean_logB75
panel["log_Bring_c"] = panel["log_Bring"] - mat_mean_logBring
p(f"centering constants (mature sample): mean log_pop={mat_mean_logpop:.4f}  "
  f"mean log(1+B75)={mat_mean_logB75:.4f}  mean log(1+Bring)={mat_mean_logBring:.4f}")

out = os.path.join(SM, "data", "market_panel.csv")
panel.to_csv(out, index=False)
p(f"\nwrote {out}  ({len(panel)} rows, {panel.mkt.nunique()} towns, {panel.qt.nunique()} quarters)")
p(f"mature-sample size: {int(panel.mature.sum())} market-quarters, "
  f"{panel[panel.mature==1].mkt.nunique()} towns")

# quick descriptive check against the reduced-form headline facts (not a formal test, just a sanity check)
H("6.  SANITY CHECK against reduced-form facts (mature sample, means by border status)")
mm = panel[panel.mature == 1]
for lab, sub in [("interior", mm[mm.border_town == 0]), ("nonlegal-border", mm[mm.border_town == 1]),
                 ("legal-border", mm[mm.legal_border_town == 1])]:
    p(f"  {lab:16s} n={len(sub):4d}  N_total {sub.N_total.mean():5.2f}  N_I {sub.N_I.mean():5.2f}  "
      f"N_C {sub.N_C.mean():5.2f}  indep.share {sub.indep_share.mean():.3f}  "
      f"B75 {sub.B_step75.mean():>10,.0f}")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
