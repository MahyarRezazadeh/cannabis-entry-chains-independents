r"""
06_robustness.py -- structural_model / extension of venous4

Tests whether delta_I > delta_C (and the sign/order of magnitude of the
legalization counterfactual) survive:
  (R1) an alternative continuous cross-border-demand definition (ring-weighted
       decay across the project's own 25/50/75/100-mile bands, instead of the
       flat 75-mile count);
  (R2) dropping the two large secondary cities (Las Cruces, Santa Fe) that are
       far bigger than the rest of the town-market distribution;
  (R3) the "eventual" (whole-panel, look-ahead) chain/independent definition
       instead of the at-entry (no-look-ahead) one, to see whether the
       organizational-heterogeneity result depends on the entry-status fix
       documented in ref01_referee_revision.py;
  (R4) the fuller mature sample (qt>=8091, includes the not-yet-fully-converged
       quarters) instead of the converged-only estimation sample.
Each uses fewer optimizer starts than the main run (this is a robustness
sweep, not the preferred estimate) but the same two-stage L-BFGS-B +
Nelder-Mead procedure.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _model import unpack, solve_equilibrium

OLD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "06_robustness.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

BOUNDS = [(-5, 5), (-5, 5), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0), (0.001, 3.0), (-2.0, 3.0)]
NAMES = ["kappa_I", "kappa_C", "beta_I", "beta_C", "delta_I", "delta_C", "gamma_same", "gamma_other"]
theta_main = np.load(os.path.join(SM, "data", "theta_hat_restricted.npy"))

def refit(log_pop, log_B, NI_obs, NC_obs, n_starts=3, seed=1):
    rng = np.random.default_rng(seed)
    wI = 1 / (1 + NI_obs); wC = 1 / (1 + NC_obs)
    def ssr(theta):
        pp = unpack(theta, restricted=True)
        ni, nc, _ = solve_equilibrium(pp, log_pop, log_B)
        return float(np.sum(wI * (NI_obs - ni) ** 2 + wC * (NC_obs - nc) ** 2))
    best = None
    for s in range(n_starts):
        th0 = theta_main if s == 0 else theta_main + rng.normal(0, 0.5, size=8)
        th0 = np.clip(th0, [b[0] + 1e-3 for b in BOUNDS], [b[1] - 1e-3 for b in BOUNDS])
        r1 = minimize(ssr, th0, method="L-BFGS-B", bounds=BOUNDS, options=dict(maxiter=100, ftol=1e-8))
        r2 = minimize(ssr, r1.x, method="Nelder-Mead", bounds=BOUNDS,
                       options=dict(maxiter=500, xatol=1e-3, fatol=1e-3))
        r = r2 if r2.fun < r1.fun else r1
        if best is None or r.fun < best.fun:
            best = r
    return best

def report(tag, res):
    d = dict(zip(NAMES, res.x))
    p(f"  [{tag}]  SSR={res.fun:.2f}  delta_I={d['delta_I']:+.3f}  delta_C={d['delta_C']:+.3f}  "
      f"(delta_I-delta_C={d['delta_I']-d['delta_C']:+.3f})  gamma_same={d['gamma_same']:+.3f}  "
      f"gamma_other={d['gamma_other']:+.3f}")
    return d

panel = pd.read_csv(os.path.join(SM, "data", "market_panel.csv"))
base = panel[panel.qt >= 8097].copy()
d0 = dict(zip(NAMES, theta_main))
H("BASELINE (preferred spec, for comparison)")
report("baseline qt>=8097, B75, at-entry type", type("R", (), {"x": theta_main, "fun": np.nan})())

# ---------------------------------------------------------------------------
H("R1.  ALTERNATIVE CROSS-BORDER-DEMAND DEFINITION -- ring-weighted 25/50/75/100mi decay")
lp = base["log_pop_c"].values; lb = base["log_Bring_c"].values
NI, NC = base["N_I"].values.astype(float), base["N_C"].values.astype(float)
res1 = refit(lp, lb, NI, NC, n_starts=3, seed=11)
report("R1 ring-weighted B", res1)

# ---------------------------------------------------------------------------
H("R2.  DROP LARGE SECONDARY CITIES (Las Cruces, Santa Fe)")
sub = base[~base.mkt.isin(["LAS CRUCES", "SANTA FE"])].copy()
p(f"remaining towns: {sub.mkt.nunique()} of {base.mkt.nunique()}")
lp2 = sub["log_pop_c"].values; lb2 = sub["log_B75_c"].values
NI2, NC2 = sub["N_I"].values.astype(float), sub["N_C"].values.astype(float)
res2 = refit(lp2, lb2, NI2, NC2, n_starts=3, seed=12)
report("R2 drop LC/SF", res2)

# ---------------------------------------------------------------------------
H("R3.  ALTERNATIVE CHAIN DEFINITION -- eventual ('chain ever', whole-panel look-ahead) status")
sf = pd.read_csv(os.path.join(OLD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(OLD, "brand_master.csv"))
ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}
sf2 = sf.merge(bm[["brand_id", "n_storefronts"]], on="brand_id", how="left")
sf2["type_ever"] = np.where(sf2["n_storefronts"] >= 2, "C", "I")
sf2["mkt"] = np.where(sf2["city"].isin(ABQ), "ABQ", sf2["city"])
act = sf2[sf2.active == 1]
g = (act.groupby(["mkt", "qt"]).apply(lambda s: pd.Series({
        "N_I_ev": (s["type_ever"] == "I").sum(), "N_C_ev": (s["type_ever"] == "C").sum()}))
     .reset_index())
r3panel = base.merge(g, on=["mkt", "qt"], how="left")
r3panel[["N_I_ev", "N_C_ev"]] = r3panel[["N_I_ev", "N_C_ev"]].fillna(0)
lp3 = r3panel["log_pop_c"].values; lb3 = r3panel["log_B75_c"].values
NI3, NC3 = r3panel["N_I_ev"].values.astype(float), r3panel["N_C_ev"].values.astype(float)
res3 = refit(lp3, lb3, NI3, NC3, n_starts=3, seed=13)
report("R3 eventual chain-ever definition", res3)
p(f"  (cf. at-entry definition: corr(N_I at-entry, N_I eventual)={np.corrcoef(base.N_I, r3panel.N_I_ev)[0,1]:.3f}, "
  f"corr(N_C at-entry, N_C eventual)={np.corrcoef(base.N_C, r3panel.N_C_ev)[0,1]:.3f})")

# ---------------------------------------------------------------------------
H("R4.  FULLER MATURE SAMPLE (qt>=8091, includes not-yet-fully-converged quarters)")
full = panel[panel.mature == 1].copy()
lp4 = full["log_pop_c"].values; lb4 = full["log_B75_c"].values
NI4, NC4 = full["N_I"].values.astype(float), full["N_C"].values.astype(float)
res4 = refit(lp4, lb4, NI4, NC4, n_starts=3, seed=14)
report("R4 full mature sample", res4)

# ---------------------------------------------------------------------------
H("SUMMARY -- stability of delta_I > delta_C across specifications")
rows = [("baseline", d0)]
for tag, r in [("R1 ring-B", res1), ("R2 drop LC/SF", res2), ("R3 eventual chain", res3), ("R4 full mature", res4)]:
    rows.append((tag, dict(zip(NAMES, r.x))))
p(f"{'spec':20s} {'delta_I':>9s} {'delta_C':>9s} {'delta_I-delta_C':>16s}")
for tag, d in rows:
    p(f"{tag:20s} {d['delta_I']:>9.3f} {d['delta_C']:>9.3f} {d['delta_I']-d['delta_C']:>16.3f}")
n_confirm = sum(1 for _, d in rows if d['delta_I'] > d['delta_C'])
p(f"\ndelta_I > delta_C holds in {n_confirm} of {len(rows)} specifications.")

pd.DataFrame([{"spec": t, **d} for t, d in rows]).to_csv(
    os.path.join(SM, "output", "table_robustness.csv"), index=False)
os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
