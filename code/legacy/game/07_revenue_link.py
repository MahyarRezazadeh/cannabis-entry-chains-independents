r"""
07_revenue_link.py -- structural_model / extension of venous4

Section 12 of the brief: determine whether the project's revenue data support
a credible, DATA-DRIVEN (not fabricated) mapping between equilibrium entry and
per-store revenue, to show how entry dissipates the initial cross-border demand
rent. This does NOT re-estimate the BNE model jointly with a demand system
(not justified by the available data -- no price/quantity/cost information);
it is a light, reduced-form supplementary check, run on the same converged-
quarter estimation sample, of the revenue-congestion relationship the paper's
Prop. 3 predicts qualitatively.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
OLD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
LOG = os.path.join(SM, "logs", "07_revenue_link.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

panel = pd.read_csv(os.path.join(SM, "data", "market_panel.csv")).drop(columns=["revenue"])
rev = pd.read_csv(os.path.join(OLD, "panel_city_quarter.csv"))
rev = rev.rename(columns={"metro": "mkt"})[["mkt", "qt", "revenue"]]
d = panel[panel.qt >= 8097].merge(rev, on=["mkt", "qt"], how="left")
d = d[(d.N_total > 0) & (d.revenue > 0)].copy()
d["log_rev"] = np.log(d["revenue"])
d["log_N"] = np.log(d["N_total"])
d["log_rev_per_store"] = d["log_rev"] - d["log_N"]
d["log_pop"] = d["log_pop"]

H("A.  Does total town revenue rise 1-for-1 with store count, or is it largely N-invariant "
  "(the 'extensive margin absorbs the demand shock' prediction, Prop. 3)?")
X = sm.add_constant(d[["log_N", "log_pop"]])
m = sm.OLS(d["log_rev"], X).fit(cov_type="HC1")
p(m.summary().tables[1].as_text())
p(f"\ncoef(log N_total) = {m.params['log_N']:.3f} (SE {m.bse['log_N']:.3f}). "
  f"A coefficient near 0 means total town revenue is close to congestion-invariant in the store "
  f"count (matching the project's existing r(N)=kappa*S/N, rho=1 finding in ext02_structural_model.py); "
  f"a coefficient near 1 would mean revenue scales linearly with entry (no per-store dissipation).")

H("B.  Revenue PER STORE against B_mt directly, controlling for population and N_total "
  "(does the demand shock show up in revenue/store once entry is held fixed?)")
d["log_B75"] = d["log_B75"]
X2 = sm.add_constant(d[["log_B75", "log_pop", "log_N"]])
m2 = sm.OLS(d["log_rev_per_store"], X2).fit(cov_type="HC1")
p(m2.summary().tables[1].as_text())
p(f"\ncoef(log(1+B75) | pop, N_total) = {m2.params['log_B75']:.3f} (SE {m2.bse['log_B75']:.3f}): "
  "the cross-border demand shock's association with per-store revenue, HOLDING the equilibrium "
  "store count fixed -- i.e., what would remain if entry did not respond. Comparing this to the "
  "raw (entry-inclusive) B-revenue-per-capita reduced form (Sec.~2, fact 2) quantifies how much of "
  "the raw demand shock is absorbed by the entry margin rather than passed through to incumbents.")

H("C.  Raw (entry-inclusive) association: log(rev per store) on B75 WITHOUT controlling for N_total")
X3 = sm.add_constant(d[["log_B75", "log_pop"]])
m3 = sm.OLS(d["log_rev_per_store"], X3).fit(cov_type="HC1")
p(f"coef(log(1+B75)), no N_total control = {m3.params['log_B75']:.3f} (SE {m3.bse['log_B75']:.3f})")
p(f"\n=> moving from (C) to (B) [i.e., adding the equilibrium store count as a control] changes the "
  f"B-revenue-per-store association from {m3.params['log_B75']:.3f} to {m2.params['log_B75']:.3f}; "
  f"the difference is the part of the per-store revenue effect that operates THROUGH the entry margin "
  f"-- entry's share of dissipating the initial demand rent.")

import json
rev_json = dict(
    coef_logN_on_revenue=float(m.params["log_N"]), se_logN_on_revenue=float(m.bse["log_N"]),
    coef_B75_on_revpstore_ctrlN=float(m2.params["log_B75"]), se_B75_on_revpstore_ctrlN=float(m2.bse["log_B75"]),
    coef_B75_on_revpstore_noctrl=float(m3.params["log_B75"]), se_B75_on_revpstore_noctrl=float(m3.bse["log_B75"]),
    n=int(len(d)),
)
with open(os.path.join(SM, "output", "revenue_diagnostics.json"), "w") as f:
    json.dump(rev_json, f, indent=2)

d.to_csv(os.path.join(SM, "output", "revenue_link_sample.csv"), index=False)
os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
