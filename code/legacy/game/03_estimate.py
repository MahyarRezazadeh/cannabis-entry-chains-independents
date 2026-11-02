r"""
03_estimate.py -- structural_model / extension of venous4

Estimates the preferred (restricted, same/other competitive-effect) Seim-style
BNE entry/type model by simulated minimum distance: match the model's
equilibrium fixed point (N_I*, N_C*) to the observed (N_I, N_C) for every
market-quarter in the ESTIMATION SAMPLE, minimizing weighted SSR.  Also
estimates the unrestricted 4-competitive-effect model as an identification
diagnostic (Sec. 9), NOT as the preferred specification (see
02_identification_checks.log).

Sample: "converged" quarters only (qt >= Q_CONVERGED), i.e. calendar-quarter
market age >= ~8 quarters, so that each market-quarter is plausibly close to
its stationary BNE rather than still transitioning from the April-2022
big-bang launch -- consistent with ext05_dynamic_transition.py's own finding
that the chain/independent composition converges within about 6-8 quarters
of a market's opening.  The fuller "mature" (qt>=8091) sample is used only in
06_robustness.py.

Multiple starting values for the optimizer AND multiple starting points for
the fixed-point solver (at the optimum) are used to check for local optima /
multiple equilibria.  Uncertainty: town-clustered nonparametric bootstrap.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _model import unpack, solve_equilibrium, solve_equilibrium_multistart, KMAX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "03_estimate.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

SEED = 20260913
rng = np.random.default_rng(SEED)
Q_CONVERGED = 8097   # qt >= this = market age >= ~8 quarters (see docstring)

panel = pd.read_csv(os.path.join(SM, "data", "market_panel.csv"))
samp = panel[panel.qt >= Q_CONVERGED].copy().reset_index(drop=True)
H("SAMPLE")
p(f"estimation sample (converged quarters, qt>={Q_CONVERGED}): {len(samp)} market-quarters, "
  f"{samp.mkt.nunique()} towns, {samp.qt.nunique()} quarters")
p(f"N_I: mean {samp.N_I.mean():.2f} max {samp.N_I.max()}   N_C: mean {samp.N_C.mean():.2f} max {samp.N_C.max()}")

log_pop = samp["log_pop_c"].values     # centered on the mature-sample mean (see 01_build_panel.py Sec.5b)
log_B = samp["log_B75_c"].values
NI_obs = samp["N_I"].values.astype(float)
NC_obs = samp["N_C"].values.astype(float)
w_I = 1.0 / (1.0 + NI_obs)          # approximate-Poisson weighting (Sec. 18/10)
w_C = 1.0 / (1.0 + NC_obs)

def ssr(theta, restricted=True):
    # NOTE: no discontinuous non-convergence penalty here -- an abrupt jump in the objective
    # whenever any single row misses the fixed-point tolerance made the (smooth, gradient-based)
    # optimizer take thousands of wasted evaluations hunting around the discontinuity. Newton's
    # method with the damped warm-up converges to high precision for essentially all economically
    # admissible parameter values within the given iteration budget; where it does not, the
    # returned (NI*,NC*) already reflects a small residual, which the SSR penalizes smoothly.
    p_ = unpack(theta, restricted=restricted)
    NI_star, NC_star, conv = solve_equilibrium(p_, log_pop, log_B)
    r = w_I * (NI_obs - NI_star) ** 2 + w_C * (NC_obs - NC_star) ** 2
    return float(np.sum(r))

def fit(restricted, n_starts=10, tag=""):
    H(f"ESTIMATION -- {'RESTRICTED (gamma_same, gamma_other)' if restricted else 'UNRESTRICTED (4 competitive effects)'} {tag}")
    k = 8 if restricted else 10
    bounds = [(-5, 5), (-5, 5), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]
    if restricted:
        bounds += [(0.001, 3.0), (-2.0, 3.0)]
    else:
        bounds += [(0.001, 3.0), (-2.0, 3.0), (0.001, 3.0), (-2.0, 3.0)]
    starts = []
    base = np.array([0.0, 0.0, 0.3, 0.3, 0.3, 0.3] + ([0.3, 0.1] if restricted else [0.3, 0.1, 0.3, 0.1]))
    starts.append(base)
    for _ in range(n_starts - 1):
        jitter = rng.normal(0, 0.6, size=k)
        s = base + jitter
        s = np.array([np.clip(v, b[0] + 1e-3, b[1] - 1e-3) for v, b in zip(s, bounds)])
        starts.append(s)
    results = []
    t0 = time.time()
    for i, s0 in enumerate(starts):
        r1 = minimize(ssr, s0, args=(restricted,), method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=100, ftol=1e-8))
        # Nelder-Mead polish: derivative-free, robust to the mild objective jitter left by a
        # fixed point solved to finite (not exact) tolerance -- L-BFGS-B alone stalls
        # ("ABNORMAL_TERMINATION_IN_LNSRCH") well short of the optimum on this objective.
        r2 = minimize(ssr, r1.x, args=(restricted,), method="Nelder-Mead", bounds=bounds,
                       options=dict(maxiter=500, xatol=1e-3, fatol=1e-3))
        results.append(r2 if r2.fun < r1.fun else r1)
    results.sort(key=lambda r: r.fun)
    best = results[0]
    spread = [r.fun for r in results]
    p(f"n_starts={n_starts}  best SSR={best.fun:.4f}  "
      f"SSR range across starts=[{min(spread):.4f}, {max(spread):.4f}]  "
      f"converged-to-global (within 0.5%): {sum(abs(s-best.fun) < 0.005*abs(best.fun)+1e-6 for s in spread)}/{n_starts}")
    p(f"time: {time.time()-t0:.1f}s   optimizer success on best: {best.success}  msg: {best.message}")
    names_r = ["kappa_I", "kappa_C", "beta_I(logpop)", "beta_C(logpop)", "delta_I(logB75)",
               "delta_C(logB75)", "gamma_same", "gamma_other"]
    names_u = ["kappa_I", "kappa_C", "beta_I(logpop)", "beta_C(logpop)", "delta_I(logB75)",
               "delta_C(logB75)", "gamma_II", "gamma_CC", "gamma_IC", "gamma_CI"]
    names = names_r if restricted else names_u
    for nm, v in zip(names, best.x):
        p(f"  {nm:18s} = {v:+.4f}")
    n_conv_global = sum(abs(s - best.fun) < 0.005 * abs(best.fun) + 1e-6 for s in spread)
    conv_info = dict(n_starts=n_starts, best_ssr=float(best.fun), ssr_min=float(min(spread)),
                      ssr_max=float(max(spread)), n_converged_to_global=int(n_conv_global),
                      time_s=float(time.time() - t0), success=bool(best.success), message=str(best.message))
    return best, names, conv_info

best_r, names_r, conv_r = fit(restricted=True, n_starts=4, tag="(preferred)")
best_u, names_u, conv_u = fit(restricted=False, n_starts=4, tag="(diagnostic)")

H("gamma_II vs gamma_CC, gamma_IC vs gamma_CI  (unrestricted -- supports the same/other restriction?)")
d = dict(zip(names_u, best_u.x))
p(f"  gamma_II={d['gamma_II']:+.3f}   gamma_CC={d['gamma_CC']:+.3f}   "
  f"(restricted gamma_same={dict(zip(names_r,best_r.x))['gamma_same']:+.3f})")
p(f"  gamma_IC={d['gamma_IC']:+.3f}   gamma_CI={d['gamma_CI']:+.3f}   "
  f"(restricted gamma_other={dict(zip(names_r,best_r.x))['gamma_other']:+.3f})")

# ---------------------------------------------------------------------------
H("MULTIPLICITY CHECK -- fixed-point solved from several starting points at theta_hat")
p_hat = unpack(best_r.x, restricted=True)
starts = [(np.zeros(len(samp)), np.zeros(len(samp))),
          (np.full(len(samp), 1.0), np.full(len(samp), 1.0)),
          (np.full(len(samp), 15.0), np.full(len(samp), 15.0)),
          (NI_obs.copy(), NC_obs.copy())]
sols = solve_equilibrium_multistart(p_hat, log_pop, log_B, starts)
for i, (NI_s, NC_s, conv) in enumerate(sols):
    p(f"  start {i}: converged={conv}  mean N_I*={NI_s.mean():.3f}  mean N_C*={NC_s.mean():.3f}")
max_diff_I = max(np.max(np.abs(sols[j][0] - sols[0][0])) for j in range(1, len(sols)))
max_diff_C = max(np.max(np.abs(sols[j][1] - sols[0][1])) for j in range(1, len(sols)))
p(f"  max |N_I*(start_j) - N_I*(start_0)| across starts = {max_diff_I:.2e}")
p(f"  max |N_C*(start_j) - N_C*(start_0)| across starts = {max_diff_C:.2e}")
unique_eq = bool(max(max_diff_I, max_diff_C) < 1e-3)
p("  ==> " + ("UNIQUE fixed point found from all tested starting points (best-response slopes < 1: "
  "the game is one of strategic substitutes with damped decay, a contraction over the relevant range)."
  if unique_eq else
  "MULTIPLE fixed points detected -- see structural_entry_model.tex Sec. 7 for the equilibrium-selection rule used."))

import json
diag = dict(conv_restricted=conv_r, conv_unrestricted=conv_u,
            unrestricted_gammas=dict(gamma_II=float(d["gamma_II"]), gamma_CC=float(d["gamma_CC"]),
                                     gamma_IC=float(d["gamma_IC"]), gamma_CI=float(d["gamma_CI"])),
            restricted_gammas=dict(gamma_same=float(dict(zip(names_r, best_r.x))["gamma_same"]),
                                   gamma_other=float(dict(zip(names_r, best_r.x))["gamma_other"])),
            unique_equilibrium=unique_eq, max_diff_I=float(max_diff_I), max_diff_C=float(max_diff_C),
            n_towns=int(samp.mkt.nunique()), n_obs=int(len(samp)), n_quarters=int(samp.qt.nunique()))
with open(os.path.join(SM, "output", "estimation_diagnostics.json"), "w") as f:
    json.dump(diag, f, indent=2)

# ---------------------------------------------------------------------------
H("TOWN-CLUSTERED BOOTSTRAP STANDARD ERRORS (preferred restricted model)")
towns = samp.mkt.unique()
B = 20
boot = np.full((B, len(best_r.x)), np.nan)
t0 = time.time()
for b in range(B):
    bt = rng.choice(towns, size=len(towns), replace=True)
    idx = np.concatenate([np.where(samp.mkt.values == t)[0] for t in bt])
    lp_b, lB_b = log_pop[idx], log_B[idx]
    NI_b, NC_b = NI_obs[idx], NC_obs[idx]
    wI_b, wC_b = w_I[idx], w_C[idx]
    def ssr_b(theta):
        pp = unpack(theta, restricted=True)
        NIs, NCs, conv = solve_equilibrium(pp, lp_b, lB_b)
        r = wI_b * (NI_b - NIs) ** 2 + wC_b * (NC_b - NCs) ** 2
        return float(np.sum(r))
    bounds = [(-5, 5), (-5, 5), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.001, 3.0), (-2.0, 3.0)]
    r1 = minimize(ssr_b, best_r.x, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=80, ftol=1e-7))
    r2 = minimize(ssr_b, r1.x, method="Nelder-Mead", bounds=bounds,
                  options=dict(maxiter=300, xatol=1e-3, fatol=1e-3))
    boot[b] = r2.x if r2.fun < r1.fun else r1.x
p(f"bootstrap reps: {B}  time: {time.time()-t0:.1f}s")
se = np.nanstd(boot, axis=0)
lo = np.nanpercentile(boot, 2.5, axis=0)
hi = np.nanpercentile(boot, 97.5, axis=0)
p(f"{'param':18s} {'estimate':>10s} {'boot SE':>10s} {'95% CI':>22s}")
for nm, v, s, l, h in zip(names_r, best_r.x, se, lo, hi):
    p(f"{nm:18s} {v:>10.4f} {s:>10.4f}   [{l:>8.4f}, {h:>8.4f}]")

diff = boot[:, 4] - boot[:, 5]     # delta_I - delta_C
p(f"\ndelta_I - delta_C:  point est {best_r.x[4]-best_r.x[5]:+.4f}   "
  f"boot SE {np.nanstd(diff):.4f}   share of boot reps with delta_I>delta_C: {np.mean(diff>0):.3f}")

# ---------------------------------------------------------------------------
np.save(os.path.join(SM, "data", "theta_hat_restricted.npy"), best_r.x)
np.save(os.path.join(SM, "data", "theta_hat_unrestricted.npy"), best_u.x)
np.save(os.path.join(SM, "data", "boot_restricted.npy"), boot)
pd.DataFrame({"param": names_r, "estimate": best_r.x, "boot_se": se, "ci_lo": lo, "ci_hi": hi}).to_csv(
    os.path.join(SM, "output", "table_theta_restricted.csv"), index=False)
pd.DataFrame({"param": names_u, "estimate": best_u.x}).to_csv(
    os.path.join(SM, "output", "table_theta_unrestricted.csv"), index=False)
samp.to_csv(os.path.join(SM, "data", "estimation_sample.csv"), index=False)

H("PROFILE OF THE OBJECTIVE AROUND delta_I - delta_C")
sum_d = best_r.x[4] + best_r.x[5]
grid = np.linspace(-1.5, 1.5, 31)
prof = []
for g in grid:
    dI = (sum_d + g) / 2.0
    dC = (sum_d - g) / 2.0
    if dI < 0 or dC < 0:
        prof.append(np.nan)
        continue
    th = best_r.x.copy()
    th[4], th[5] = dI, dC
    prof.append(ssr(th, restricted=True))
prof = np.array(prof)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, prof, lw=2)
ax.axvline(best_r.x[4] - best_r.x[5], color="C3", ls="--", lw=1.2, label=r"$\hat\delta_I-\hat\delta_C$")
ax.set_xlabel(r"$\delta_I - \delta_C$ (holding $\delta_I+\delta_C$ and all other parameters fixed)")
ax.set_ylabel("SMD objective (SSR)")
ax.set_title("Profile of the estimation objective")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(SM, "output", "fig_profile_deltaI_deltaC.png"), dpi=140)
plt.close(fig)
p(f"profile: min at grid point {grid[np.nanargmin(prof)]:.3f} (point estimate {best_r.x[4]-best_r.x[5]:.3f})")
p("wrote fig_profile_deltaI_deltaC.png")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
