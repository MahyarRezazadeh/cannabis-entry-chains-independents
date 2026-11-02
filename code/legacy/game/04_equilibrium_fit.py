r"""
04_equilibrium_fit.py -- structural_model / extension of venous4

Solves the BNE equilibrium at theta_hat (preferred, restricted model) for every
market-quarter in the mature sample and compares model-predicted (N_I*, N_C*)
to observed (N_I, N_C):
  (i)  IN-SAMPLE fit on the converged-quarter estimation sample (qt>=8097);
  (ii) OUT-OF-SAMPLE / holdout fit on the earlier "transitioning" mature
       quarters (8091<=qt<8097) that were NOT used to estimate theta -- a
       genuine holdout test, and one where the model's static-equilibrium
       assumption is expected (per ext05_dynamic_transition.py's own finding
       that composition converges over ~6-8 quarters) to fit worse, which is
       itself an informative diagnostic, not swept under the rug.
Both broken out by interior / nonlegal-border / legal-border.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _model import unpack, solve_equilibrium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "04_equilibrium_fit.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

Q_CONVERGED = 8097
theta = np.load(os.path.join(SM, "data", "theta_hat_restricted.npy"))
pmodel = unpack(theta, restricted=True)
panel = pd.read_csv(os.path.join(SM, "data", "market_panel.csv"))
mat = panel[panel.mature == 1].copy()

log_pop = mat["log_pop_c"].values
log_B = mat["log_B75_c"].values
NI_star, NC_star, conv = solve_equilibrium(pmodel, log_pop, log_B)
mat["NI_hat"] = NI_star
mat["NC_hat"] = NC_star
mat["N_hat"] = NI_star + NC_star
mat["indep_share_hat"] = mat["NI_hat"] / mat["N_hat"].clip(lower=1e-6)
mat["sample_lab"] = np.where(mat.qt >= Q_CONVERGED, "in-sample (converged, qt>=8097)",
                              "holdout (transitioning, 8091<=qt<8097)")
p(f"all converged: {conv}")

def report(sub, label):
    p(f"\n-- {label}  (n={len(sub)}) --")
    for lab, m in [("all", sub.index == sub.index), ("interior", sub.border_town == 0),
                   ("nonlegal-border", sub.border_town == 1),
                   ("legal-border", sub.legal_border_town == 1)]:
        s = sub[m]
        if len(s) == 0:
            continue
        mae_tot = float(np.mean(np.abs(s.N_total - s.N_hat)))
        mae_i = float(np.mean(np.abs(s.N_I - s.NI_hat)))
        mae_c = float(np.mean(np.abs(s.N_C - s.NC_hat)))
        p(f"  {lab:16s} n={len(s):4d}  N_tot act {s.N_total.mean():5.2f} mod {s.N_hat.mean():5.2f} "
          f"(MAE {mae_tot:.2f})   N_I act {s.N_I.mean():5.2f} mod {s.NI_hat.mean():5.2f} (MAE {mae_i:.2f})  "
          f"N_C act {s.N_C.mean():5.2f} mod {s.NC_hat.mean():5.2f} (MAE {mae_c:.2f})  "
          f"indep.sh act {s.indep_share.mean():.3f} mod {s.indep_share_hat.mean():.3f}")
    if sub.N_total.std() > 0 and sub.N_hat.std() > 0:
        p(f"  corr(actual N_total, model N_total) = {np.corrcoef(sub.N_total, sub.N_hat)[0,1]:.3f}")
    if sub.N_I.std() > 0 and sub.NI_hat.std() > 0:
        p(f"  corr(actual N_I, model N_I)         = {np.corrcoef(sub.N_I, sub.NI_hat)[0,1]:.3f}")
    if sub.N_C.std() > 0 and sub.NC_hat.std() > 0:
        p(f"  corr(actual N_C, model N_C)         = {np.corrcoef(sub.N_C, sub.NC_hat)[0,1]:.3f}")

H("IN-SAMPLE FIT")
report(mat[mat.sample_lab.str.startswith("in-sample")], "converged (estimation) sample")
H("OUT-OF-SAMPLE / HOLDOUT FIT")
report(mat[mat.sample_lab.str.startswith("holdout")], "pre-convergence holdout quarters")

# ---------------------------------------------------------------------------
H("TOWN-LEVEL (mean over converged quarters) FIT -- the cross-sectional object the paper's reduced form targets")
tw = (mat[mat.qt >= Q_CONVERGED].groupby("mkt")
        .agg(N_total=("N_total", "mean"), N_hat=("N_hat", "mean"),
             N_I=("N_I", "mean"), NI_hat=("NI_hat", "mean"),
             N_C=("N_C", "mean"), NC_hat=("NC_hat", "mean"),
             indep_share=("indep_share", "mean"), indep_share_hat=("indep_share_hat", "mean"),
             border_town=("border_town", "first"), legal_border_town=("legal_border_town", "first"))
        .reset_index())
p(f"{tw.mkt.nunique()} towns")
p(f"corr(actual, model) town-mean N_total = {np.corrcoef(tw.N_total, tw.N_hat)[0,1]:.3f}")
p(f"corr(actual, model) town-mean N_I     = {np.corrcoef(tw.N_I, tw.NI_hat)[0,1]:.3f}")
p(f"corr(actual, model) town-mean N_C     = {np.corrcoef(tw.N_C, tw.NC_hat)[0,1]:.3f}")
tw.to_csv(os.path.join(SM, "output", "town_fit.csv"), index=False)

# ---------------------------------------------------------------------------
H("FIGURES")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
for b, c, lb in [(0, "C0", "interior"), (1, "C3", "nonlegal-border")]:
    s = tw[tw.border_town == b]
    ax[0].scatter(s.N_total, s.N_hat, s=26, alpha=.75, color=c, label=lb)
s = tw[tw.legal_border_town == 1]
ax[0].scatter(s.N_total, s.N_hat, s=26, alpha=.75, color="C2", marker="^", label="legal-border")
lim = [0, float(max(tw.N_total.max(), tw.N_hat.max()) + 1)]
ax[0].plot(lim, lim, "k--", lw=.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("actual mean # stores (converged qtrs)"); ax[0].set_ylabel("model equilibrium # stores")
ax[0].set_title("Town-level store count"); ax[0].legend(fontsize=8)
x = np.arange(3); w = .18
groups = [("interior", tw.border_town == 0), ("nonlegal-border", tw.border_town == 1),
          ("legal-border", tw.legal_border_town == 1)]
vals = [("indep actual", [tw[m].N_I.mean() for _, m in groups], "C0", 1.0),
        ("indep model", [tw[m].NI_hat.mean() for _, m in groups], "C0", .5),
        ("chain actual", [tw[m].N_C.mean() for _, m in groups], "C3", 1.0),
        ("chain model", [tw[m].NC_hat.mean() for _, m in groups], "C3", .5)]
for i, (lb, v, c, al) in enumerate(vals):
    ax[1].bar(x + (i - 1.5) * w, v, w, label=lb, color=c, alpha=al)
ax[1].set_xticks(x); ax[1].set_xticklabels([g[0] for g in groups])
ax[1].set_title("Composition: actual vs model equilibrium"); ax[1].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SM, "output", "fig_equilibrium_fit.png"), dpi=140)
plt.close(fig)
p("wrote fig_equilibrium_fit.png")

import json
fit_json = {}
for samp_lab, sub in [("insample", mat[mat.qt >= Q_CONVERGED]), ("holdout", mat[mat.qt < Q_CONVERGED])]:
    fit_json[samp_lab] = {}
    for lab, m in [("all", sub.index == sub.index), ("interior", sub.border_town == 0),
                   ("nonlegal_border", sub.border_town == 1), ("legal_border", sub.legal_border_town == 1)]:
        s = sub[m]
        if len(s) == 0:
            continue
        fit_json[samp_lab][lab] = dict(
            n=int(len(s)), N_total_act=float(s.N_total.mean()), N_total_mod=float(s.N_hat.mean()),
            N_I_act=float(s.N_I.mean()), N_I_mod=float(s.NI_hat.mean()),
            N_C_act=float(s.N_C.mean()), N_C_mod=float(s.NC_hat.mean()),
            indep_share_act=float(s.indep_share.mean()), indep_share_mod=float(s.indep_share_hat.mean()),
            mae_total=float(np.mean(np.abs(s.N_total - s.N_hat))))
fit_json["corr_town_N_total"] = float(np.corrcoef(tw.N_total, tw.N_hat)[0, 1])
fit_json["corr_town_N_I"] = float(np.corrcoef(tw.N_I, tw.NI_hat)[0, 1])
fit_json["corr_town_N_C"] = float(np.corrcoef(tw.N_C, tw.NC_hat)[0, 1])
with open(os.path.join(SM, "output", "fit_diagnostics.json"), "w") as f:
    json.dump(fit_json, f, indent=2)

mat.to_csv(os.path.join(SM, "output", "market_quarter_fit.csv"), index=False)
os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
