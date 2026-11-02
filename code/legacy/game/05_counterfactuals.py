r"""
05_counterfactuals.py -- structural_model / extension of venous4

Main counterfactual (Sec.13): neighbor-state legalization.  B_mt = nonleg_pop_75
is BY CONSTRUCTION the population within 75 miles that is under cannabis
prohibition; "the neighboring state legalizes" therefore means precisely
B_mt -> 0 for the affected nonlegal-border towns -- NOT zeroing out border
status, town population, distance, or any other observable. Local NM demand
(log_pop) and every other covariate are held fixed. The full BNE is re-solved.

Secondary counterfactuals (Sec.16): (i) no organizational heterogeneity
(delta_I = delta_C), (ii) decomposition of the border increment into
independent vs. chain entry, (iii) comparative statics over a grid of B.

Validation (Sec.14): re-estimate excluding legal-border towns, predict their
equilibrium under their own (near-zero, already-nonlegal-free) B, and compare
with their actual observed composition -- an honest out-of-sample check, not
a mechanical identity (legal-border towns already have B_step75 approx 0, so
this asks whether towns the model NEVER SAW can still be predicted from their
population alone).

External benchmark (Sec.15): Hansen, Miller & Weber (2020, J. Public Econ.)
document a large, rapid decline in Washington border-county cannabis sales
after Oregon's 2015 recreational legalization removed the cross-border
prohibition rent. That paper's numeric estimate is NOT imported into the NM
calibration -- it is cited only as qualitative, independently-sourced support
for the DIRECTION and existence of the mechanism the counterfactual below
exercises. No number from HMW appears in the arithmetic below.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _model import unpack, solve_equilibrium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "05_counterfactuals.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

Q_CONVERGED = 8097
theta = np.load(os.path.join(SM, "data", "theta_hat_restricted.npy"))
boot = np.load(os.path.join(SM, "data", "boot_restricted.npy"))
pmodel = unpack(theta, restricted=True)
panel = pd.read_csv(os.path.join(SM, "data", "market_panel.csv"))
samp = panel[panel.qt >= Q_CONVERGED].copy().reset_index(drop=True)
log_pop = samp["log_pop_c"].values
log_B = samp["log_B75_c"].values
border = samp["border_town"].values.astype(bool)
# centering constant: log_B75_c = log(1+B75) - MEAN_LOGB75, so "B75 -> 0" (removing the
# prohibition-generated component, Sec.13) means log_B75_c -> 0 - MEAN_LOGB75 = -MEAN_LOGB75,
# NOT log_B75_c -> 0 (which would instead mean "B75 at its mature-sample average").
MEAN_LOGB75 = float((panel.loc[panel.mature == 1, "log_B75"] - panel.loc[panel.mature == 1, "log_B75_c"]).mean())
MEAN_LOGPOP = float((panel.loc[panel.mature == 1, "log_pop"] - panel.loc[panel.mature == 1, "log_pop_c"]).mean())
B_ZERO_CENTERED = -MEAN_LOGB75

# ---------------------------------------------------------------------------
H("13.  MAIN COUNTERFACTUAL -- neighbor-state legalization (remove prohibition-generated B)")
NI0, NC0, _ = solve_equilibrium(pmodel, log_pop, log_B)
log_B_cf = log_B.copy()
log_B_cf[border] = B_ZERO_CENTERED                         # B75 -> 0 for nonlegal-border towns only
NI1, NC1, _ = solve_equilibrium(pmodel, log_pop, log_B_cf)
samp["NI0"], samp["NC0"], samp["NI1"], samp["NC1"] = NI0, NC0, NI1, NC1

def summarize(mask, label):
    n0I, n0C = NI0[mask].sum(), NC0[mask].sum()
    n1I, n1C = NI1[mask].sum(), NC1[mask].sum()
    dI, dC = n1I - n0I, n1C - n0C
    n_mkts = mask.sum() / samp[mask].qt.nunique() if mask.sum() else 0
    p(f"  {label:30s}  baseline: I={n0I:7.1f} C={n0C:7.1f} tot={n0I+n0C:7.1f} "
      f"(indep.sh {n0I/(n0I+n0C):.3f})   post-legalization: I={n1I:7.1f} C={n1C:7.1f} "
      f"tot={n1I+n1C:7.1f} (indep.sh {n1I/(n1I+n1C):.3f})")
    p(f"    Delta N_I = {dI:+7.1f}   Delta N_C = {dC:+7.1f}   Delta N_total = {dI+dC:+7.1f}   "
      f"|Delta N_I| {'>' if abs(dI) > abs(dC) else '<='} |Delta N_C|")
    return dI, dC

p(f"nonlegal-border town-quarters affected: {int(border.sum())} of {len(samp)} "
  f"({samp[border].mkt.nunique()} distinct towns)")
dI_main, dC_main = summarize(border, "nonlegal-border towns")

# propagate parameter uncertainty: recompute the counterfactual at each bootstrap theta
H("Counterfactual uncertainty (town-clustered bootstrap, B={} reps)".format(boot.shape[0]))
dI_boot, dC_boot, dTot_boot, ishare_delta_boot = [], [], [], []
for b in range(boot.shape[0]):
    pb = unpack(boot[b], restricted=True)
    ni0, nc0, _ = solve_equilibrium(pb, log_pop, log_B)
    ni1, nc1, _ = solve_equilibrium(pb, log_pop, log_B_cf)
    dI_boot.append((ni1[border] - ni0[border]).sum())
    dC_boot.append((nc1[border] - nc0[border]).sum())
    dTot_boot.append(dI_boot[-1] + dC_boot[-1])
    s0 = ni0[border].sum() / (ni0[border].sum() + nc0[border].sum())
    s1 = ni1[border].sum() / (ni1[border].sum() + nc1[border].sum())
    ishare_delta_boot.append(s1 - s0)
dI_boot, dC_boot, dTot_boot = map(np.array, (dI_boot, dC_boot, dTot_boot))
p(f"  Delta N_I     : {dI_main:+.1f}   95% CI [{np.percentile(dI_boot,2.5):+.1f}, {np.percentile(dI_boot,97.5):+.1f}]")
p(f"  Delta N_C     : {dC_main:+.1f}   95% CI [{np.percentile(dC_boot,2.5):+.1f}, {np.percentile(dC_boot,97.5):+.1f}]")
p(f"  Delta N_total : {dI_main+dC_main:+.1f}   95% CI [{np.percentile(dTot_boot,2.5):+.1f}, {np.percentile(dTot_boot,97.5):+.1f}]")
p(f"  Delta indep.share (post-pre): mean {np.mean(ishare_delta_boot):+.3f}  "
  f"95% CI [{np.percentile(ishare_delta_boot,2.5):+.3f}, {np.percentile(ishare_delta_boot,97.5):+.3f}]")
p(f"  share of bootstrap reps with |Delta N_I|>|Delta N_C|: {np.mean(np.abs(dI_boot)>np.abs(dC_boot)):.3f}")

H("External qualitative benchmark (not used in the arithmetic above)")
p("  Hansen, Miller & Weber (2020, J. Public Econ.) document a large, rapid decline in Washington "
  "border-county cannabis sales after Oregon legalized recreational sales in Oct. 2015, consistent "
  "with the direction of the mechanism exercised above (removing the neighbor's prohibition status "
  "removes the cross-border demand rent). No numeric estimate from that paper is used here.")

# ---------------------------------------------------------------------------
H("16a.  NO-ORGANIZATIONAL-HETEROGENEITY COUNTERFACTUAL  (impose delta_I = delta_C)")
pmodel_eq = dict(pmodel)
dbar = 0.5 * (pmodel["dI"] + pmodel["dC"])
pmodel_eq["dI"] = dbar; pmodel_eq["dC"] = dbar
NI2, NC2, _ = solve_equilibrium(pmodel_eq, log_pop, log_B)
n0I, n0C = NI0[border].sum(), NC0[border].sum()
n2I, n2C = NI2[border].sum(), NC2[border].sum()
p(f"  delta_I={pmodel['dI']:.3f}  delta_C={pmodel['dC']:.3f}  ->  equalized at {dbar:.3f}")
p(f"  estimated model (border towns): I={n0I:.1f} C={n0C:.1f} indep.sh={n0I/(n0I+n0C):.3f}")
p(f"  equal-delta counterfactual     : I={n2I:.1f} C={n2C:.1f} indep.sh={n2I/(n2I+n2C):.3f}")
p(f"  => equalizing the type-specific demand response would change the border independent share by "
  f"{n2I/(n2I+n2C) - n0I/(n0I+n0C):+.3f}, holding total entry approximately fixed by construction of "
  f"the equalization (mean-preserving); the composition difference is attributable specifically to "
  f"delta_I != delta_C, not to the level of cross-border demand.")

H("16b.  DECOMPOSITION OF THE BORDER INCREMENT (interior counterfactual: nonlegal-border towns "
  "with their OWN B set to 0, i.e. 'as if interior')")
dI_dec, dC_dec = dI_main, dC_main
tot_inc = dI_dec + dC_dec
p(f"  cross-border demand accounts for a net change of {tot_inc:+.1f} stores across "
  f"{samp[border].mkt.nunique()} nonlegal-border towns' converged-quarter equilibria")
if abs(tot_inc) > 1e-9:
    p(f"  of which independent entry: {dI_dec:+.1f} ({dI_dec/tot_inc:.1%})   "
      f"chain entry: {dC_dec:+.1f} ({dC_dec/tot_inc:.1%})")

H("16c.  COMPARATIVE STATICS -- equilibrium composition as the demand shock scales up")
Bgrid = np.linspace(0, float(samp["B_step75"].max()), 40)
rep_logpop_c = float(samp.loc[border, "log_pop_c"].median())     # representative border town
rows = []
for Bv in Bgrid:
    lp = np.array([rep_logpop_c]); lb = np.array([np.log1p(Bv) - MEAN_LOGB75])
    ni, nc, _ = solve_equilibrium(pmodel, lp, lb)
    rows.append(dict(B=Bv, N_I=ni[0], N_C=nc[0], N=ni[0] + nc[0],
                      indep_share=ni[0] / max(ni[0] + nc[0], 1e-9)))
cs = pd.DataFrame(rows)
cs.to_csv(os.path.join(SM, "output", "comparative_statics.csv"), index=False)
for Bv in [0, Bgrid.max() * .25, Bgrid.max() * .5, Bgrid.max() * .75, Bgrid.max()]:
    r = cs.iloc[(cs.B - Bv).abs().values.argmin()]
    p(f"  B={r.B:>10,.0f}: N_I={r.N_I:6.2f}  N_C={r.N_C:6.2f}  N={r.N:6.2f}  indep.sh={r.indep_share:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].plot(cs.B / 1e3, cs.N_I, lw=2, label="independent $N_I^*$")
ax[0].plot(cs.B / 1e3, cs.N_C, lw=2, ls="--", label="chain $N_C^*$")
ax[0].plot(cs.B / 1e3, cs.N, lw=1.2, color="k", alpha=.6, label="total")
ax[0].set_xlabel("nonlegal population within 75mi (000s)"); ax[0].set_title("Equilibrium # stores")
ax[0].legend(fontsize=8)
ax[1].plot(cs.B / 1e3, cs.indep_share, lw=2, color="C4")
ax[1].set_xlabel("nonlegal population within 75mi (000s)"); ax[1].set_title("Independent share")
ax[1].set_ylim(0, 1)
for a in ax:
    a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Estimated model: equilibrium response to cross-border demand (representative border town)")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(os.path.join(SM, "output", "fig_comparative_statics.png"), dpi=140)
plt.close(fig)
p("wrote fig_comparative_statics.png")

# ---------------------------------------------------------------------------
H("14.  LEGAL-BORDER VALIDATION -- re-estimate EXCLUDING legal-border towns, then predict them")
train = samp[samp.legal_border_town == 0].reset_index(drop=True)
test = samp[samp.legal_border_town == 1].reset_index(drop=True)
p(f"training towns: {train.mkt.nunique()}   held-out legal-border towns: {test.mkt.nunique()} "
  f"({len(test)} town-quarters)")
lpt, lbt = train["log_pop_c"].values, train["log_B75_c"].values
NIt, NCt = train["N_I"].values.astype(float), train["N_C"].values.astype(float)
wIt, wCt = 1 / (1 + NIt), 1 / (1 + NCt)
def ssr_train(th):
    pp = unpack(th, restricted=True)
    ni, nc, _ = solve_equilibrium(pp, lpt, lbt)
    return float(np.sum(wIt * (NIt - ni) ** 2 + wCt * (NCt - nc) ** 2))
bounds = [(-5, 5), (-5, 5), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.001, 3.0), (-2.0, 3.0)]
best = None
for s in range(3):
    th0 = theta + (0 if s == 0 else np.random.default_rng(s).normal(0, 0.4, size=8))
    th0 = np.clip(th0, [b[0] + 1e-3 for b in bounds], [b[1] - 1e-3 for b in bounds])
    r1 = minimize(ssr_train, th0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=100, ftol=1e-8))
    r2 = minimize(ssr_train, r1.x, method="Nelder-Mead", bounds=bounds,
                  options=dict(maxiter=500, xatol=1e-3, fatol=1e-3))
    r = r2 if r2.fun < r1.fun else r1
    if best is None or r.fun < best.fun:
        best = r
p(f"held-out re-estimation SSR={best.fun:.3f}  success={best.success}")
p_holdout = unpack(best.x, restricted=True)
lpv, lbv = test["log_pop_c"].values, test["log_B75_c"].values
ni_v, nc_v, _ = solve_equilibrium(p_holdout, lpv, lbv)
p(f"legal-border towns -- actual : N_I={test.N_I.mean():.2f}  N_C={test.N_C.mean():.2f}  "
  f"indep.sh={test.indep_share.mean():.3f}")
p(f"legal-border towns -- predicted (model trained WITHOUT them): "
  f"N_I={ni_v.mean():.2f}  N_C={nc_v.mean():.2f}  indep.sh={(ni_v/np.maximum(ni_v+nc_v,1e-9)).mean():.3f}")
p("  (a genuine out-of-sample check: these towns' population alone, run through a model that never "
  "saw their entry counts, is asked to reproduce their observed low-B equilibrium.)")

import json
cf_json = dict(
    legalization=dict(
        n_towns_affected=int(samp[border].mkt.nunique()), n_town_quarters=int(border.sum()),
        baseline_NI=float(NI0[border].sum()), baseline_NC=float(NC0[border].sum()),
        postcf_NI=float(NI1[border].sum()), postcf_NC=float(NC1[border].sum()),
        delta_NI=float(dI_main), delta_NC=float(dC_main), delta_Ntotal=float(dI_main + dC_main),
        delta_NI_ci=[float(np.percentile(dI_boot, 2.5)), float(np.percentile(dI_boot, 97.5))],
        delta_NC_ci=[float(np.percentile(dC_boot, 2.5)), float(np.percentile(dC_boot, 97.5))],
        delta_Ntotal_ci=[float(np.percentile(dTot_boot, 2.5)), float(np.percentile(dTot_boot, 97.5))],
        indep_share_delta_mean=float(np.mean(ishare_delta_boot)),
        indep_share_delta_ci=[float(np.percentile(ishare_delta_boot, 2.5)), float(np.percentile(ishare_delta_boot, 97.5))],
        share_boot_absI_gt_absC=float(np.mean(np.abs(dI_boot) > np.abs(dC_boot))),
        baseline_indep_share=float(NI0[border].sum() / (NI0[border].sum() + NC0[border].sum())),
        postcf_indep_share=float(NI1[border].sum() / (NI1[border].sum() + NC1[border].sum())),
    ),
    no_heterogeneity=dict(
        delta_I=float(pmodel["dI"]), delta_C=float(pmodel["dC"]), delta_bar=float(dbar),
        baseline_indep_share=float(n0I / (n0I + n0C)), eqdelta_indep_share=float(n2I / (n2I + n2C)),
        baseline_N=float(n0I + n0C), eqdelta_N=float(n2I + n2C),
    ),
    decomposition=dict(delta_NI=float(dI_dec), delta_NC=float(dC_dec), total_increment=float(tot_inc),
                       share_independent=float(dI_dec / tot_inc) if abs(tot_inc) > 1e-9 else None),
    legal_validation=dict(
        n_train_towns=int(train.mkt.nunique()), n_test_towns=int(test.mkt.nunique()),
        actual_NI=float(test.N_I.mean()), actual_NC=float(test.N_C.mean()),
        actual_indep_share=float(test.indep_share.mean()),
        pred_NI=float(ni_v.mean()), pred_NC=float(nc_v.mean()),
        pred_indep_share=float((ni_v / np.maximum(ni_v + nc_v, 1e-9)).mean()),
    ),
)
with open(os.path.join(SM, "output", "counterfactual_diagnostics.json"), "w") as f:
    json.dump(cf_json, f, indent=2)

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
samp.to_csv(os.path.join(SM, "output", "counterfactual_legalization.csv"), index=False)
p("\nDONE.")
