r"""
rio06_composition_extras.py -- three results cited in Section 6 and the appendix.
 (1) Additional chain and independent operators per 100 nonlegal-border ZIP codes in 2022Q2: average over the 92
     ZIP codes of sinh(fitted) minus sinh(fitted with the 2022Q2 nonlegal coefficient set to zero), times 100;
     1,999 county-cluster bootstrap draws (seed 20261023).
 (2) Conditional-on-activity composition: log(1+N) of chain and independent operators on ZIP-quarters with at least one
     dispensary, with log population, quarter FE, and market-age FE (quarters since the ZIP's first dispensary).
 (3) Diffusion-model table (weighted NLS fit of the quarterly coefficients; reproduces the point estimates of the
     earlier appendix table from rev04_org_dynamics.py and reports the standard errors computed there).
Outputs: output/tables/tab_rio_conditional_comp.tex, tab_rio_diffusion.tex, output/logs/rio06.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np
import pandas as pd
from paths import LOGS
from _common import build_all, QS, QLAB, QMIN, stars, OUT, ROOT
from _econ import design, ols_fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))
d = build_all(with_cov=False)["full"]
X, names = design(d); j = names.index(f"g{QMIN}")
Y = {k: np.arcsinh(d[f"nb_{k}_life"].to_numpy()) for k in "CI"}
cl = d.county_fips.to_numpy(); cs = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in cs}
def stat(rows):
    Xr = X[rows]; m = (d.nonlegal.to_numpy()[rows] == 1) & (d.qt.to_numpy()[rows] == QMIN); out = {}
    for k in "CI":
        b = np.linalg.lstsq(Xr, Y[k][rows], rcond=None)[0]
        X0 = Xr[m].copy(); X0[:, j] = 0
        out[k] = 100 * (np.sinh(Xr[m] @ b).mean() - np.sinh(X0 @ b).mean())
        out[k + "_fit"], out[k + "_cf"] = np.sinh(Xr[m] @ b).mean(), np.sinh(X0 @ b).mean()
    out["C-I"] = out["C"] - out["I"]
    return out
pt = stat(np.arange(len(d)))
rng = np.random.default_rng(20261023); B = []
for _ in range(1999):
    B.append(stat(np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])))
per100 = {}
for k in ("C", "I", "C-I"):
    v = np.array([b[k] for b in B])
    per100[k] = (pt[k], v.std(), np.percentile(v, 2.5), np.percentile(v, 97.5))
    p(f"(1) additional {k} operators per 100 NL ZIPs, 2022Q2: {pt[k]:.1f} (boot SE {v.std():.1f}) 95% CI [{np.percentile(v,2.5):.1f}, {np.percentile(v,97.5):.1f}]")
p(f"    fitted vs counterfactual operators per NL ZIP: chain {pt['C_fit']:.3f} vs {pt['C_cf']:.3f}; independent {pt['I_fit']:.3f} vs {pt['I_cf']:.3f}")

# (2) conditional on activity
fq = d[d.n_stores > 0].groupby("zip5").qt.min(); d["age"] = d.qt - d.zip5.map(fq)
act = d[d.n_stores > 0].reset_index(drop=True)
Xa, na = design(act)
A = pd.get_dummies(act.age.astype(int), prefix="a", drop_first=True).to_numpy(float)
Xa = np.column_stack([Xa, A])
def win(r, qs):
    w = np.zeros(len(r["b"])); w[[na.index(f"g{q}") for q in qs]] = 1 / len(qs)
    m = w @ r["b"]; s = np.sqrt(w @ r["V"] @ w); return m, s, pval(m, s, r["G"])
rows = []
for lab, y in [("Chain operators", np.log1p(act.nb_C_life)), ("Independent operators", np.log1p(act.nb_I_life)),
               ("Chain $-$ independent", np.log1p(act.nb_C_life) - np.log1p(act.nb_I_life))]:
    r = ols_fit(y.to_numpy(), Xa, act.county_fips.to_numpy())
    jj = na.index(f"g{QMIN}")
    q0 = (r["b"][jj], r["se"][jj], pval(r["b"][jj], r["se"][jj], r["G"]))
    e, l = win(r, QS[:4]), win(r, QS[-9:])
    rows.append((lab, q0, e, l))
    p(f"(2) {lab:24s} 2022Q2 {q0[0]:+.3f} ({q0[1]:.3f}){stars(q0[2])} | early {e[0]:+.3f} ({e[1]:.3f}){stars(e[2])} | late {l[0]:+.3f} ({l[1]:.3f}){stars(l[2])}")
n_nl0 = int(((act.qt == QMIN) & (act.nonlegal == 1)).sum()); n_lb0 = int(((act.qt == QMIN) & (act.nonlegal == 0)).sum())
p(f"    active ZIP-quarters {len(act)} ({act.zip5.nunique()} ZIPs); active in 2022Q2: NL {n_nl0}, LB {n_lb0}")
def c(x):
    return f"${x[0]:+.3f}^{{{stars(x[2])}}}$".replace("^{}", ""), f"({x[1]:.3f})"
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & 2022Q2 & 2022Q2--2023Q1 & 2024Q2--2026Q2 \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Additional operators per 100 nonlegal-border ZIP codes, 2022Q2 (full sample)}} \\"]
for k, lab in [("C", "Chain operators"), ("I", "Independent operators"), ("C-I", "Chain $-$ independent")]:
    v = per100[k]; L.append(f"{lab} & {v[0]:.1f} [{v[2]:.1f}, {v[3]:.1f}] & & \\\\")
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: ZIP-quarters with at least one dispensary, $\log(1+N)$, market-age fixed effects}} \\")
for lab, q0, e, l in rows:
    a, b_, cc = c(q0), c(e), c(l)
    L += [f"{lab} & {a[0]} & {b_[0]} & {cc[0]} \\\\", f" & {a[1]} & {b_[1]} & {cc[1]} \\\\"]
L += [r"\midrule", f"Active ZIP-quarters (ZIP codes) & \\multicolumn{{3}}{{l}}{{{len(act)} ({act.zip5.nunique()}); active in 2022Q2: {n_nl0} nonlegal-border, {n_lb0} always-legal-border}} \\\\",
      r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_conditional_comp.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# (3) diffusion table from rev04 output
DIF = pickle.load(open(os.path.join(OUT, "rev04.pkl"), "rb"))["DIF"]
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Parameter & Lifetime & At market opening & Status at entry \\", r"\midrule"]
labs = [r"Long-run response, $\beta^\infty$", r"Adjustment rate per quarter, $\lambda$", r"Initial share, independents, $\phi_I$", r"Initial share, chains, $\phi_C$"]
for i, lab in enumerate(labs):
    L += [lab + " & " + " & ".join(f"{DIF[k]['x'][i]:.3f}" for k in ("life", "pre", "entry")) + r" \\",
          " & " + " & ".join(f"({DIF[k]['se'][i]:.3f})" for k in ("life", "pre", "entry")) + r" \\"]
L += [r"$\phi_C-\phi_I$ & " + " & ".join(f"{DIF[k]['dphi']:+.3f}" for k in ("life", "pre", "entry")) + r" \\",
      " & " + " & ".join(f"({DIF[k]['sed']:.3f})" for k in ("life", "pre", "entry")) + r" \\",
      r"Half-life (quarters), $\ln 2/\lambda$ & " + " & ".join(f"{np.log(2)/DIF[k]['x'][1]:.1f}" for k in ("life", "pre", "entry")) + r" \\",
      r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rio_diffusion.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
for k in ("life", "pre", "entry"):
    p(f"(3) diffusion [{k}] " + ", ".join(f"{v:.3f}({s:.3f})" for v, s in zip(DIF[k]['x'], DIF[k]['se'])) + f"; dphi {DIF[k]['dphi']:+.3f} ({DIF[k]['sed']:.3f})")
open(LOGS / "rio06.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
