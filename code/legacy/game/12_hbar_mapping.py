r"""
12_hbar_mapping.py -- structural_model / extension of venous4 / cannibalization extension

Part 8 of the brief: map the microfoundation back to the freely-estimated delta_I - delta_C gap.

delta_I = delta ;  delta_C(Hbar) = delta - rho*Hbar  =>  delta_I - delta_C(Hbar) = rho*Hbar.

IMPORTANT: 11_mechanism_regression.py did NOT produce a defensible estimate of rho < 0 -- the
operator-level interaction phi (H x D(B)) is statistically insignificant in the preferred
(operator+quarter FE, ABQ-excluded) specification and significantly POSITIVE (wrong sign) once ABQ
is included. Per the brief's explicit instruction ("do NOT force equality if sampling error/
specification differences make them differ" / "if [rho] is weakly identified, say so"), this script
does NOT back out an estimated rho and does NOT claim the aggregate delta_I - delta_C gap is
"explained" by a validated cannibalization magnitude. It reports representative Hbar values (a
purely descriptive input) and, ONLY as an explicitly labeled illustrative/what-if calculation, the
rho that WOULD be needed to rationalize the observed gap at each Hbar -- not a claim that this rho is
the true parameter.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "12_hbar_mapping.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

c_H = pd.read_csv(os.path.join(SM, "data", "typeC_entry_H.csv"))
risk = pd.read_csv(os.path.join(SM, "data", "operator_risk_set_panel.csv"))
risk["is_abq"] = (risk["mkt"] == "ABQ").astype(int)

# freely-estimated aggregate gap, from the UNCHANGED (Option B) BNE estimator, last session:
theta = np.load(os.path.join(SM, "data", "theta_hat_restricted.npy"))
boot = np.load(os.path.join(SM, "data", "boot_restricted.npy"))
delta_I, delta_C = theta[4], theta[5]
gap = delta_I - delta_C
gap_se = float(np.nanstd(boot[:, 4] - boot[:, 5]))
p(f"Freely-estimated (unrestricted delta_I vs delta_C, aggregate BNE model, unchanged from prior "
  f"session): delta_I={delta_I:.4f}, delta_C={delta_C:.4f}, gap={gap:.4f} (bootstrap SE {gap_se:.4f})")

H("REPRESENTATIVE Hbar VALUES")
rows = []
rows.append(("Realized Type-C entrants, all",             c_H.H_same.mean(), c_H.H_geo.fillna(0).mean()))
rows.append(("Realized Type-C entrants, nonlegal-border",  c_H.loc[c_H.border_town==1,"H_same"].mean(),
             c_H.loc[c_H.border_town==1,"H_geo"].fillna(0).mean()))
rows.append(("Realized Type-C entrants, interior",         c_H.loc[(c_H.border_town==0)&(c_H.legal_border_town==0),"H_same"].mean(),
             c_H.loc[(c_H.border_town==0)&(c_H.legal_border_town==0),"H_geo"].fillna(0).mean()))
prim = risk[risk.is_abq==0]
rows.append(("Risk-set (all chain-expansion opportunities, ABQ-excl.)", prim.H_same.mean(), prim.H_geo.fillna(0).mean()))
rows.append(("Risk-set, nonlegal-border opportunities",    prim.loc[prim.border_town==1,"H_same"].mean(),
             prim.loc[prim.border_town==1,"H_geo"].fillna(0).mean()))
tab = pd.DataFrame(rows, columns=["definition", "Hbar_same", "Hbar_geo"])
p(tab.to_string(index=False))
tab.to_csv(os.path.join(SM, "output", "hbar_values.csv"), index=False)

H("ILLUSTRATIVE (NOT VALIDATED) rho THAT WOULD RATIONALIZE THE GAP")
p("Per 11_mechanism_regression.py, phi is NOT robustly negative (preferred spec: p=0.49 for H_same, "
  "p=0.21 for H_geo; ABQ-included robustness: significantly POSITIVE, p<0.0001). The calculation "
  "below is explicitly a WHAT-IF, not an estimate: 'if the entire freely-estimated gap were "
  "attributed to network-overlap attenuation at a given Hbar, rho would have to equal gap/Hbar.'")
for _, r in tab.iterrows():
    if r.Hbar_same > 0:
        p(f"  at Hbar_same={r.Hbar_same:.3f} ({r.definition}): illustrative rho = {gap/r.Hbar_same:.4f}")
p("\nThese illustrative rho values are reported ONLY to give a sense of scale (Sec.8 of the brief); "
  "they are NOT presented in the paper as an identified structural parameter, precisely because "
  "§11's direct operator-level test does not support rho>0.")

H("THEORETICAL ILLUSTRATION FIGURE (payoff slopes, NOT a re-solved equilibrium)")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Bgrid = np.linspace(0, c_H.B_step75.quantile(0.95), 50)
Dgrid = np.log1p(Bgrid)
rho_illustrative = gap / tab.loc[tab.definition == "Realized Type-C entrants, all", "Hbar_same"].values[0]
H_lo, H_mean, H_hi = 0.0, float(c_H.H_same.mean()), float(c_H.H_same.quantile(0.75))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(Bgrid / 1e3, delta_I * Dgrid, lw=2.2, color="C0", label=r"independent, $\delta_I \cdot D(B)$")
for Hval, ls, lab in [(H_lo, "--", r"chain, $H=0$"),
                        (H_mean, "-.", rf"chain, $H=\bar H$={H_mean:.1f}"),
                        (H_hi, ":", rf"chain, $H=$P75={H_hi:.1f}")]:
    slope = max(delta_I - rho_illustrative * Hval, 0)
    ax.plot(Bgrid / 1e3, slope * Dgrid, lw=2, color="C3", ls=ls, label=lab)
ax.set_xlabel("nonlegal population within 75mi (000s), $B_m$")
ax.set_ylabel(r"payoff contribution, $[\delta_I \text{ or } \delta-\rho H]\cdot D(B_m)$")
ax.set_title("Theoretical illustration only -- NOT a re-solved equilibrium\n"
             r"(uses the illustrative $\rho$ from Part 8, which the operator-level test does not confirm)",
             fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SM, "output", "fig_theory_illustration.png"), dpi=140)
plt.close(fig)
p(f"wrote fig_theory_illustration.png (illustrative rho={rho_illustrative:.4f}, NOT an estimated parameter)")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
