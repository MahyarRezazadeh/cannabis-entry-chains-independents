r"""
_model.py -- shared model primitives for the Seim-style incomplete-information
entry/type game (extension of venous4 / structural_model).

Two ex ante heterogeneous pools of potential entrants per market-period (m,t):
  type I (independent at entry)      type C (established multi-store at entry)
Each pool has "slots" k=1,2,...,Kmax.  Slot k's profit if it enters, given the
number of active rivals of the OTHER type, N_other, and X=(1,log_pop), B=log1p(B75):

  pi_tau(k, N_other; theta) = kappa_tau + beta_tau*log_pop + delta_tau*log1p(B75)
                               - gamma_tauTau*(k-1) - gamma_tauOther*N_other

Private cost/profit shock eps_tau,k ~ iid logistic(0,1)  [scale normalized to 1,
a standard normalization, NOT an estimated parameter].  Slot k enters iff
pi_tau(k,...) + eps > 0, so P(slot k enters | N_other) = Lambda(pi_tau(k,N_other)),
Lambda = standard logistic cdf.  With a continuum of ex ante symmetric potential
entrants per type, the EXPECTED number of type-tau entrants given the rival
type's count is  E[N_tau | N_other] = sum_{k=1}^{Kmax} Lambda(pi_tau(k,N_other)).
A Bayesian-Nash equilibrium is a fixed point (N_I*, N_C*) of the two best-response
maps.  See structural_entry_model.tex Sec. 4-5 for the full derivation.
"""
import numpy as np

KMAX = 80   # matches ext02_structural_model.py's own Nmax=80 cap; Las Cruces/Santa Fe (the two
            # largest non-ABQ town-markets, observed N up to ~75) are kept in-sample, consistent
            # with the project's existing convention (ext02 excludes only the Albuquerque metro)

def logistic_cdf(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))

def unpack(theta, restricted=True):
    if restricted:
        kI, kC, bI, bC, dI, dC, g_same, g_other = theta
        return dict(kI=kI, kC=kC, bI=bI, bC=bC, dI=dI, dC=dC,
                    gII=g_same, gCC=g_same, gIC=g_other, gCI=g_other)
    kI, kC, bI, bC, dI, dC, gII, gCC, gIC, gCI = theta
    return dict(kI=kI, kC=kC, bI=bI, bC=bC, dI=dI, dC=dC, gII=gII, gCC=gCC, gIC=gIC, gCI=gCI)

def best_response(kappa, beta, delta, gamma_own, gamma_other, log_pop, log_B, N_other,
                   want_deriv=False):
    """Vectorized over markets. Returns E[N_tau | N_other] (and, if want_deriv, its
    derivative w.r.t. N_other) for every market row."""
    base = kappa + beta * log_pop + delta * log_B - gamma_other * N_other   # (n,)
    k = np.arange(1, KMAX + 1).reshape(-1, 1)                              # (Kmax,1)
    pi = base.reshape(1, -1) - gamma_own * (k - 1)                          # (Kmax, n)
    L = logistic_cdf(pi)
    R = L.sum(axis=0)                                                      # (n,)
    if not want_deriv:
        return R
    dR = (-gamma_other) * (L * (1 - L)).sum(axis=0)                        # dR/dN_other, (n,)
    return R, dR

def solve_equilibrium(p, log_pop, log_B, tol=1e-8, maxit=60, warmup=6, start=None):
    """Vectorized Newton iteration (with a short damped-fixed-point warm-up for global
    stability) to the BNE fixed point (N_I*, N_C*) of
        N_I = R_I(N_C),  N_C = R_C(N_I).
    Each market's 2-equation system is independent of every other market's, so the
    n-market problem reduces to n independent 2x2 Newton steps, solved in closed form
    and applied elementwise -- this is what makes Newton fast here (quadratic local
    convergence) instead of the ~100s-of-iterations damped fixed-point map.
    p: dict from unpack().  log_pop, log_B: arrays (n,).  start: optional (NI0,NC0)."""
    n = len(log_pop)
    if start is None:
        NI = np.full(n, 2.0); NC = np.full(n, 1.0)
    else:
        NI, NC = start[0].copy(), start[1].copy()
    NI = np.clip(NI, 0, KMAX); NC = np.clip(NC, 0, KMAX)
    # short damped warm-up: globally stable, gets us into Newton's basin of attraction
    for _ in range(warmup):
        RI = best_response(p["kI"], p["bI"], p["dI"], p["gII"], p["gIC"], log_pop, log_B, NC)
        RC = best_response(p["kC"], p["bC"], p["dC"], p["gCC"], p["gCI"], log_pop, log_B, NI)
        NI = 0.5 * NI + 0.5 * RI
        NC = 0.5 * NC + 0.5 * RC
    for _ in range(maxit):
        RI, dRI = best_response(p["kI"], p["bI"], p["dI"], p["gII"], p["gIC"], log_pop, log_B, NC, True)
        RC, dRC = best_response(p["kC"], p["bC"], p["dC"], p["gCC"], p["gCI"], log_pop, log_B, NI, True)
        F1 = NI - RI            # = 0 at the fixed point
        F2 = NC - RC
        a = -dRI                # dF1/dN_C
        b = -dRC                # dF2/dN_I
        det = 1.0 - a * b
        det = np.where(np.abs(det) < 1e-8, np.sign(det + 1e-12) * 1e-8, det)
        dNI = (-F1 + a * F2) / det
        dNC = (b * F1 - F2) / det
        # step-size damping keeps iterates in the economically meaningful region
        step = np.minimum(1.0, 2.0 / (1e-9 + np.maximum(np.abs(dNI), np.abs(dNC))))
        NI = np.clip(NI + step * dNI, 0, KMAX)
        NC = np.clip(NC + step * dNC, 0, KMAX)
        if max(np.max(np.abs(F1)), np.max(np.abs(F2))) < tol:
            return NI, NC, True
    RI = best_response(p["kI"], p["bI"], p["dI"], p["gII"], p["gIC"], log_pop, log_B, NC)
    RC = best_response(p["kC"], p["bC"], p["dC"], p["gCC"], p["gCI"], log_pop, log_B, NI)
    conv = max(np.max(np.abs(NI - RI)), np.max(np.abs(NC - RC))) < 1e-4
    return NI, NC, conv

def solve_equilibrium_multistart(p, log_pop, log_B, starts):
    """Runs solve_equilibrium from several starting points; returns the list of
    fixed points found (for a multiplicity check) and the modal/first-converged one."""
    sols = []
    for s in starts:
        NI, NC, conv = solve_equilibrium(p, log_pop, log_B, start=s)
        sols.append((NI, NC, conv))
    return sols
