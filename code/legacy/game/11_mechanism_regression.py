r"""
11_mechanism_regression.py -- structural_model / extension of venous4 / cannibalization extension

Part 7 of the brief: operator-level mechanism test. Does the attractiveness of cross-border demand
to an established operator's NEXT store fall as that operator's own pre-existing network overlaps
more with the candidate town? Prediction: phi < 0 in

  E[EntryC_jmt | .] = exp{ alpha_j + alpha_t + beta*log(L_mt) + delta_B*D(B_m)
                            + theta*H_jmt + phi*(H_jmt x D(B_m)) }

Estimated by PPML (Poisson pseudo-max-likelihood). Three tiers, per the brief's own fallback
instruction ("also estimate simpler versions ... if the risk set cannot support the saturated FE
structure, clearly labeled"):
  (0) pooled, no FE, operator-clustered SE            -- simplest baseline
  (1) operator FE + quarter FE (additive), preferred  -- see cannibalization_audit.md Sec.4 for why
      the full operator x quarter interaction (6,137 mostly-singleton levels) is not attempted
  Both run for H_same (transparent) and H_geo (preferred continuous measure), and for the ABQ-excluded
  primary sample and an ABQ-included robustness sample (see cannibalization_audit.md / 09's log for
  why ABQ, 38.6% of Type-C events, is not dropped by default here the way the aggregate BNE model
  drops it).
"""
import os, sys, time, math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ppml_fe import build_sparse_design, poisson_irls_sparse, cluster_robust_se

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "11_mechanism_regression.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

risk = pd.read_csv(os.path.join(SM, "data", "operator_risk_set_panel.csv"))
risk = risk.dropna(subset=["log_pop", "log_B75"]).copy()
risk["is_abq"] = (risk["mkt"] == "ABQ").astype(int)
risk["H_same_x_B"] = risk["H_same"] * risk["log_B75"]
risk["H_geo_x_B"] = risk["H_geo"].fillna(0) * risk["log_B75"]
risk["AnySame_x_B"] = risk["AnySame"] * risk["log_B75"]
p(f"full risk-set (all towns incl. ABQ): {len(risk):,} rows, {int(risk.EntryC.sum())} entry events")

results_summary = []

def run_pooled(df, rhs, label):
    f = f"EntryC ~ {rhs}"
    m = smf.glm(f, data=df, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": df["brand_id"]})
    p(f"  [{label}] pooled PPML, operator-clustered SE, N={int(m.nobs)}")
    for term in rhs.split(" + "):
        if term in m.params.index:
            p(f"    {term:16s} = {m.params[term]:+.4f}  SE {m.bse[term]:.4f}  p={m.pvalues[term]:.4f}")
    return m

def run_fe(df, cont_cols, label, cluster_col="brand_id"):
    t0 = time.time()
    X, names = build_sparse_design(df, fe_cols=["brand_id", "qt"], cont_cols=cont_cols,
                                    drop_one_level_of="qt")
    y = df["EntryC"].values.astype(float)
    beta, mu, n_iter, delta = poisson_irls_sparse(X, y)
    se, V = cluster_robust_se(X, y, beta, mu, df[cluster_col].values)
    p(f"  [{label}] operator+quarter FE PPML: N={len(df):,}  k={X.shape[1]}  "
      f"converged in {n_iter} iters (delta={delta:.2e})  time={time.time()-t0:.1f}s")
    out = {}
    for c in cont_cols:
        idx = names.index(c)
        z = beta[idx] / se[idx] if se[idx] > 0 else np.nan
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2)))) if np.isfinite(z) else np.nan
        p(f"    {c:16s} = {beta[idx]:+.4f}  SE {se[idx]:.4f}  z={z:+.2f}  p={pval:.4f}")
        out[c] = (beta[idx], se[idx], pval)
    return out

# ---------------------------------------------------------------------------
H("A.  PRIMARY SAMPLE -- ABQ EXCLUDED (consistent with the aggregate BNE model's own convention)")
prim = risk[risk.is_abq == 0].copy()
p(f"N={len(prim):,}, entry events={int(prim.EntryC.sum())}, operators={prim.brand_id.nunique()}")

H("A1.  H_same specification (transparent same-town count)")
m_pool_same = run_pooled(prim, "log_pop + log_B75 + H_same + H_same_x_B", "pooled, H_same")
fe_same = run_fe(prim, ["log_pop", "log_B75", "H_same", "H_same_x_B"], "FE, H_same")

H("A2.  H_geo specification (preferred continuous distance-weighted measure)")
m_pool_geo = run_pooled(prim.assign(H_geo=prim.H_geo.fillna(0)),
                         "log_pop + log_B75 + H_geo + H_geo_x_B", "pooled, H_geo")
fe_geo = run_fe(prim.assign(H_geo=prim.H_geo.fillna(0)),
                 ["log_pop", "log_B75", "H_geo", "H_geo_x_B"], "FE, H_geo")

H("A3.  AnySame (binary) specification")
m_pool_any = run_pooled(prim, "log_pop + log_B75 + AnySame + AnySame_x_B", "pooled, AnySame")
fe_any = run_fe(prim, ["log_pop", "log_B75", "AnySame", "AnySame_x_B"], "FE, AnySame")

# ---------------------------------------------------------------------------
H("B.  ROBUSTNESS -- ABQ INCLUDED (38.6% of realized Type-C events; see audit)")
allT = risk.copy()
fe_same_abq = run_fe(allT, ["log_pop", "log_B75", "H_same", "H_same_x_B"], "FE incl.ABQ, H_same")
fe_geo_abq = run_fe(allT.assign(H_geo=allT.H_geo.fillna(0)),
                     ["log_pop", "log_B75", "H_geo", "H_geo_x_B"], "FE incl.ABQ, H_geo")

# ---------------------------------------------------------------------------
H("C.  VERDICT")
phi_same = fe_same["H_same_x_B"]
phi_geo = fe_geo["H_geo_x_B"]
p(f"Preferred spec (operator+quarter FE, ABQ excluded):")
p(f"  phi (H_same x D(B))  = {phi_same[0]:+.4f}  SE {phi_same[1]:.4f}  p={phi_same[2]:.4f}   "
  f"{'CONFIRMS phi<0' if phi_same[0]<0 and phi_same[2]<0.10 else 'DOES NOT confirm phi<0 at 10%'}")
p(f"  phi (H_geo x D(B))   = {phi_geo[0]:+.4f}  SE {phi_geo[1]:.4f}  p={phi_geo[2]:.4f}   "
  f"{'CONFIRMS phi<0' if phi_geo[0]<0 and phi_geo[2]<0.10 else 'DOES NOT confirm phi<0 at 10%'}")
p(f"Robustness (ABQ included):")
p(f"  phi (H_same x D(B))  = {fe_same_abq['H_same_x_B'][0]:+.4f}  p={fe_same_abq['H_same_x_B'][2]:.4f}")
p(f"  phi (H_geo x D(B))   = {fe_geo_abq['H_geo_x_B'][0]:+.4f}  p={fe_geo_abq['H_geo_x_B'][2]:.4f}")

# ---------------------------------------------------------------------------
H("SAVE")
rows = []
for lab, dd in [("H_same, FE, ABQ-excl (preferred)", fe_same), ("H_geo, FE, ABQ-excl", fe_geo),
                ("AnySame, FE, ABQ-excl", fe_any), ("H_same, FE, ABQ-incl", fe_same_abq),
                ("H_geo, FE, ABQ-incl", fe_geo_abq)]:
    for k, (b, se, pv) in dd.items():
        rows.append(dict(spec=lab, param=k, estimate=b, se=se, pvalue=pv))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(SM, "output", "mechanism_regression.csv"), index=False)

def stars(pv):
    return "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""

def mk_tex(dd_list, labels, fname, caption):
    params = ["log_pop", "log_B75", list(dd_list[0].keys())[2], list(dd_list[0].keys())[3]]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(dd_list) + r"@{}}", r"\toprule",
             " & " + " & ".join(labels) + r" \\", r"\midrule"]
    plab = {"log_pop": r"$\log L_{mt}$", "log_B75": r"$D(B_m)$",
            "H_same": r"$H_{same,jmt}$", "H_same_x_B": r"$H_{same}\times D(B)$",
            "H_geo": r"$H_{geo,jmt}$", "H_geo_x_B": r"$H_{geo}\times D(B)$",
            "AnySame": r"$\mathbb{1}\{H_{same}>0\}$", "AnySame_x_B": r"$\mathbb{1}\{H_{same}>0\}\times D(B)$"}
    for par in params:
        row1 = plab.get(par, par) + " & " + " & ".join(
            f"{dd.get(par,(np.nan,np.nan,np.nan))[0]:+.3f}{stars(dd.get(par,(0,0,1))[2])}" if par in dd else ""
            for dd in dd_list) + r" \\"
        row2 = " & " + " & ".join(
            f"({dd.get(par,(np.nan,np.nan,np.nan))[1]:.3f})" if par in dd else "" for dd in dd_list) + r" \\"
        lines.append(row1); lines.append(row2)
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SM, "output", fname), "w", encoding="utf-8").write("\n".join(lines))

mk_tex([fe_same, fe_geo, fe_any], ["$H_{same}$", "$H_{geo}$", "AnySame"],
       "tab_mechanism_primary.tex", "Operator-level mechanism regression, primary (ABQ-excluded)")
mk_tex([fe_same_abq, fe_geo_abq], ["$H_{same}$", "$H_{geo}$"],
       "tab_mechanism_robustABQ.tex", "Operator-level mechanism regression, ABQ included")
p("wrote mechanism_regression.csv, tab_mechanism_primary.tex, tab_mechanism_robustABQ.tex")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
