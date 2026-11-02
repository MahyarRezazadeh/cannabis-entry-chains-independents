r"""
rev06_structural_diagnostics.py -- referee Phase 6. Collects the structural model's saved diagnostics
(no re-estimation: the estimator in ../../structural_model takes 3-15 minutes per fit and the saved outputs are
the authoritative estimates) and adds one new diagnostic:
  residual correlation between independent and chain operator counts after partialling out log population,
  exposure, and quarter FE -- the variation that would identify cross-type competition.
Outputs: tab_rev_structural.tex, output/rev06_results.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, json, pickle
import numpy as np, pandas as pd
from _common import OUT, ROOT
from _econ import design

from paths import INTER as _INTER; SM = str(_INTER / "game_out")   # saved outputs of code/legacy/game (not re-estimated)
LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

rob = pd.read_csv(os.path.join(SM, "table_robustness.csv"))
th = pd.read_csv(os.path.join(SM, "table_theta_restricted.csv")).set_index("param")
un = pd.read_csv(os.path.join(SM, "table_theta_unrestricted.csv")).set_index("param")
cf = json.load(open(os.path.join(SM, "counterfactual_diagnostics.json")))
fit = json.load(open(os.path.join(SM, "fit_diagnostics.json")))
est = json.load(open(os.path.join(SM, "estimation_diagnostics.json")))
p("Town-level restricted estimates (bootstrap SE, 95% CI):")
for k, r in th.iterrows():
    p(f"  {k}: {r.estimate:+.4f} ({r.boot_se:.4f}) [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]")
p("Robustness specs (delta_I, delta_C, gamma_same, gamma_other):")
for _, r in rob.iterrows():
    p(f"  {r.spec}: {r.delta_I:.4f} {r.delta_C:.4f} {r.gamma_same:.4f} {r.gamma_other:+.4f}")
L0 = cf["legalization"]
p(f"Counterfactual: dNI {L0['delta_NI']:.1f} {L0['delta_NI_ci']}, dNC {L0['delta_NC']:.1f} {L0['delta_NC_ci']}; "
  f"independent share of increment {cf['decomposition']['share_independent']:.3f}; baseline independent share "
  f"{L0['baseline_indep_share']:.3f}; change in independent share {L0['indep_share_delta_mean']:+.3f} {L0['indep_share_delta_ci']}")

# new diagnostic: residual correlation (ZIP panel, main sample) and town-level if available
A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
m = A["full"][A["full"].group != "interior"].copy()
m["E"] = np.log1p(m.nlpop_75 / 1000)
res = {}
for t in ["I", "C"]:
    X, _ = design(m, dynamic=False, treat="E")
    y = np.arcsinh(m[f"ns_{t}_entry"]).to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]; res[t] = y - X @ b
rc = np.corrcoef(res["I"], res["C"])[0, 1]
act = m.n_stores > 0
rc_act = np.corrcoef(res["I"][act], res["C"][act])[0, 1]
p(f"Residual corr(asinh N_I, asinh N_C | log pop, exposure, quarter FE), at-entry store counts, 209 ZIPs: {rc:.3f} "
  f"(ZIP-quarters with any store: {rc_act:.3f})")

L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"Specification & $\hat\delta_I$ & $\hat\delta_C$ & $\hat\gamma_{\text{same}}$ & $\hat\gamma_{\text{other}}$ \\", r"\midrule"]
names = {"baseline": "Baseline (town, converged quarters)", "R1 ring-B": "Ring-weighted exposure", "R2 drop LC/SF": "Drop Las Cruces and Santa Fe",
         "R3 eventual chain": "Lifetime chain definition", "R4 full mature": "Fuller mature sample"}
for _, r in rob.iterrows():
    L.append(f"{names[r.spec]} & {r.delta_I:.3f} & {r.delta_C:.3f} & {r.gamma_same:.3f} & {r.gamma_other:+.3f} \\\\")
L.append(f"Baseline 95\\% bootstrap interval & [{th.loc['delta_I(logB75)','ci_lo']:.3f}, {th.loc['delta_I(logB75)','ci_hi']:.3f}] & "
         f"[{th.loc['delta_C(logB75)','ci_lo']:.3f}, {th.loc['delta_C(logB75)','ci_hi']:.3f}] & "
         f"[{th.loc['gamma_same','ci_lo']:.3f}, {th.loc['gamma_same','ci_hi']:.3f}] & [{th.loc['gamma_other','ci_lo']:+.3f}, {th.loc['gamma_other','ci_hi']:+.3f}] \\\\")
L.append(r"\addlinespace\multicolumn{5}{@{}l}{\textit{Unrestricted competition (diagnostic; optimizer hit its iteration limit)}} \\")
L.append(f"$\\gamma_{{II}}, \\gamma_{{CC}}, \\gamma_{{IC}}, \\gamma_{{CI}}$ & \\multicolumn{{4}}{{c}}{{{un.loc['gamma_II','estimate']:.3f}, {un.loc['gamma_CC','estimate']:.3f}, {un.loc['gamma_IC','estimate']:.3f}, {un.loc['gamma_CI','estimate']:+.3f}}} \\\\")
L.append(r"\addlinespace\multicolumn{5}{@{}l}{\textit{Fit and counterfactual (baseline)}} \\")
fi = fit["insample"]
L.append(f"Mean stores per town-quarter, actual / model & \\multicolumn{{4}}{{c}}{{{fi['all']['N_total_act']:.2f} / {fi['all']['N_total_mod']:.2f}}} \\\\")
L.append(f"Independent share, actual / model & \\multicolumn{{4}}{{c}}{{{fi['all']['indep_share_act']:.3f} / {fi['all']['indep_share_mod']:.3f}}} \\\\")
lv = cf["legal_validation"]
L.append(f"Held-out legal-border towns: independent share, actual / predicted & \\multicolumn{{4}}{{c}}{{{lv['actual_indep_share']:.3f} / {lv['pred_indep_share']:.3f}}} \\\\")
L.append(f"Legalization counterfactual: independent share of lost stores & \\multicolumn{{4}}{{c}}{{{cf['decomposition']['share_independent']:.3f}}} \\\\")
L.append(f"\\quad baseline independent share of stores & \\multicolumn{{4}}{{c}}{{{L0['baseline_indep_share']:.3f}}} \\\\")
L.append(f"\\quad change in independent share (95\\% interval) & \\multicolumn{{4}}{{c}}{{{L0['indep_share_delta_mean']:+.3f} [{L0['indep_share_delta_ci'][0]:+.3f}, {L0['indep_share_delta_ci'][1]:+.3f}]}} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_structural.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
pickle.dump(dict(rc=rc, rc_act=rc_act), open(os.path.join(OUT, "rev06.pkl"), "wb"))
open(os.path.join(OUT, "rev06_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
