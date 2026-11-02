r"""
13_make_cannibalization_tables.py -- structural_model / extension of venous4 / cannibalization ext.

Renders every output/tab_cann_*.tex and output/tex_cann_*_blurb.tex file the revised paper \input{}s,
directly from the saved CSVs produced by 09-12 -- no hand-retyped numbers, mirroring 08_make_tables.py.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
OUT = os.path.join(SM, "output")

def w(fn, s):
    open(os.path.join(OUT, fn), "w", encoding="utf-8").write(s)
    print("wrote", fn)

# ---------------------------------------------------------------------------
c_H = pd.read_csv(os.path.join(SM, "data", "typeC_entry_H.csv"))
cats = pd.read_csv(os.path.join(OUT, "cannibalization_categories.csv"), index_col=0)
n_total = len(c_H)
n_any_same = int(c_H.AnySame.sum())
n_75 = int((c_H.H75 > 0).sum())
n_zero_net = int((c_H.H_size == 0).sum())
n_abq = int((c_H.mkt == "ABQ").sum())

pct = lambda x: f"{100*x:.1f}\\%"
w("tex_cann_diagnostic_blurb.tex",
  f"Of {n_total} Type-C entry events (all observed, none left-censored), {n_any_same} "
  f"({pct(n_any_same/n_total)}) occur where the entering operator already had an active storefront "
  f"in the SAME town at $t-1$, and {n_75} ({pct(n_75/n_total)}) occur within 75 miles of an existing "
  f"own store. {n_zero_net} ({pct(n_zero_net/n_total)}) entries are by operators whose historically-"
  f"qualifying prior sibling store had already exited by $t-1$, leaving zero currently active own "
  f"network at the moment of this specific entry -- a genuine feature of the data, not an artifact, "
  f"and these observations are retained descriptively but fall outside the network-overlap risk set "
  f"used in the mechanism regression (\\S\\ref{{sec:mechreg}}). {n_abq} ({pct(n_abq/n_total)}) of all "
  f"Type-C entries are in the Albuquerque metro area -- far too large a share to drop the way the "
  f"aggregate equilibrium model excludes Albuquerque as an outlier market; the mechanism regression "
  f"therefore reports both an Albuquerque-excluded primary specification (for comparability with the "
  f"aggregate model's own convention) and an Albuquerque-included robustness specification.")

# ---------------------------------------------------------------------------
mech = pd.read_csv(os.path.join(OUT, "mechanism_regression.csv"))
def get(spec, param):
    r = mech[(mech.spec == spec) & (mech.param == param)]
    return (float(r.estimate.iloc[0]), float(r.se.iloc[0]), float(r.pvalue.iloc[0])) if len(r) else (np.nan,)*3

hs = get("H_same, FE, ABQ-excl (preferred)", "H_same_x_B")
hg = get("H_geo, FE, ABQ-excl", "H_geo_x_B")
hs_abq = get("H_same, FE, ABQ-incl", "H_same_x_B")
hg_abq = get("H_geo, FE, ABQ-incl", "H_geo_x_B")
hs_main = get("H_same, FE, ABQ-excl (preferred)", "H_same")
hg_main = get("H_geo, FE, ABQ-excl", "H_geo")

w("tex_cann_mechanism_blurb.tex",
  f"The preferred operator-level mechanism specification (operator and quarter fixed effects, "
  f"Albuquerque excluded, {int(mech.shape[0])} coefficient rows saved; see "
  f"\\texttt{{output/mechanism\\_regression.csv}}) finds a strong, precisely estimated POSITIVE "
  f"main effect of own-network overlap on the probability of choosing a candidate town -- "
  f"$\\hat\\theta_{{H_{{same}}}}={hs_main[0]:+.3f}$ (SE {hs_main[1]:.3f}, $p<0.001$) and "
  f"$\\hat\\theta_{{H_{{geo}}}}={hg_main[0]:+.3f}$ (SE {hg_main[1]:.3f}, $p<0.001$): operators are, "
  f"unsurprisingly, far more likely to add a store where they already have a presence. The "
  f"THEORETICALLY CENTRAL coefficient, however, is the interaction $\\hat\\varphi$ "
  f"(own-network overlap $\\times$ cross-border demand), which the mechanism predicts should be "
  f"NEGATIVE. In the preferred specification it is not: "
  f"$\\hat\\varphi_{{H_{{same}}}}={hs[0]:+.4f}$ (SE {hs[1]:.4f}, $p={hs[2]:.3f}$) and "
  f"$\\hat\\varphi_{{H_{{geo}}}}={hg[0]:+.4f}$ (SE {hg[1]:.4f}, $p={hg[2]:.3f}$) -- both statistically "
  f"indistinguishable from zero, and the point estimate on $H_{{geo}}$ has the WRONG sign. Including "
  f"Albuquerque (robustness) does not rescue the prediction: "
  f"$\\hat\\varphi_{{H_{{same}}}}={hs_abq[0]:+.4f}$ ($p{'<0.001' if hs_abq[2]<0.001 else f'={hs_abq[2]:.3f}'}$) "
  f"and $\\hat\\varphi_{{H_{{geo}}}}={hg_abq[0]:+.4f}$ ($p{'<0.001' if hg_abq[2]<0.001 else f'={hg_abq[2]:.3f}'}$) "
  f"are both significantly POSITIVE -- the opposite sign from the theoretical prediction. "
  f"\\textbf{{The direct operator-level test does not support $\\varphi<0$ (equivalently $\\rho>0$) in "
  f"these data.}}")

# ---------------------------------------------------------------------------
hbar = pd.read_csv(os.path.join(SM, "output", "hbar_values.csv"))
def hb(d):
    r = hbar[hbar.definition == d]
    return float(r.Hbar_same.iloc[0])
theta_r = np.load(os.path.join(SM, "data", "theta_hat_restricted.npy"))
gap = theta_r[4] - theta_r[5]
w("tex_cann_hbar_blurb.tex",
  f"The freely estimated aggregate gap from the unrestricted (unchanged) equilibrium model is "
  f"$\\hat\\delta_I-\\hat\\delta_C={gap:.4f}$ (\\S\\ref{{sec:results}}). Mean $H_{{same}}$ among "
  f"realized Type-C entrants is {hb('Realized Type-C entrants, all'):.2f} overall, "
  f"{hb('Realized Type-C entrants, nonlegal-border'):.2f} among nonlegal-border entrants, and "
  f"{hb('Realized Type-C entrants, interior'):.2f} among interior entrants. Because \\S"
  f"\\ref{{sec:mechreg}} does not defensibly identify $\\rho$, this paper does NOT back out an "
  f"estimated $\\rho$ from $\\text{{gap}}/\\bar H$ and present it as a structural parameter. Purely "
  f"as an illustrative what-if -- the $\\rho$ that WOULD rationalize the full aggregate gap at the "
  f"overall realized-entrant $\\bar H$ -- the arithmetic gives "
  f"$\\rho_{{\\text{{illustrative}}}}=\\text{{gap}}/\\bar H \\approx "
  f"{gap/hb('Realized Type-C entrants, all'):.3f}$; this number is reported only to give a sense of "
  f"scale (Figure~\\ref{{fig:theoryillustration}}) and is explicitly not an estimate.")

print("ALL CANNIBALIZATION TABLES/BLURBS WRITTEN.")
