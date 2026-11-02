r"""
rio05_quarterly_tables.py -- regenerate the two quarterly tables of the main text from the rebuilt panel:
  tab_fact1fact2_dynamic.tex : quarterly gaps in asinh revenue per resident, asinh dispensary density, log revenue per store
  tab_fact3_dynamic.tex      : quarterly gaps in asinh independent and chain operators (lifetime classification) and
                               their difference from a stacked regression with type-specific quarter FE and a common
                               log-population slope (county-clustered).
The script prints the largest absolute difference from the archived published versions.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, re
import numpy as np
import pandas as pd
from _common import build_all, QS, QLAB, stars, ROOT
from _econ import fit, ols_fit, pval

d = build_all(with_cov=False)["full"]
def c3(b, s, pv):
    return f"{b:+.3f}{stars(pv)}", f"({s:.3f})"

res = {y: fit(d if y != "y_rps" else d[d.n_stores > 0], y) for y in ("y_rpc", "y_den", "y_rps")}
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r"Quarter & \multicolumn{2}{c}{Revenue per resident (asinh)} & \multicolumn{2}{c}{Dispensary density (asinh)} & \multicolumn{2}{c}{log(Revenue per store)} \\",
     r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}", r"& Coef. & (SE) & Coef. & (SE) & Coef. & (SE) \\", r"\midrule"]
for i, q in enumerate(QLAB):
    cells = []
    for y in ("y_rpc", "y_den", "y_rps"):
        r = res[y]; cells += list(c3(r["gap_b"][i], r["gap_se"][i], r["gap_p"][i]))
    L.append(f"{q} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_fact1fact2_dynamic.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# Table 4
d["yI"], d["yC"] = np.arcsinh(d.nb_I_life), np.arcsinh(d.nb_C_life)
rI, rC = fit(d, "yI"), fit(d, "yC")
n = len(d); Q = d.qt.to_numpy(); NL = d.nonlegal.to_numpy()
cols = []
for tau in (0, 1):                                   # 0 = independent block, 1 = chain block
    for q in QS:
        cols.append(np.r_[NL * (Q == q) * (tau == 0), NL * (Q == q) * (tau == 1)])
for tau in (0, 1):
    for q in QS:
        cols.append(np.r_[(Q == q) * (tau == 0), (Q == q) * (tau == 1)].astype(float))
cols.append(np.r_[d.log_pop.to_numpy(), d.log_pop.to_numpy()])
X = np.column_stack(cols).astype(float)
y = np.r_[d.yI.to_numpy(), d.yC.to_numpy()]
cl = np.r_[d.county_fips.to_numpy(), d.county_fips.to_numpy()]
r = ols_fit(y, X, cl)
K = len(QS)
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r"Quarter & \multicolumn{2}{c}{N independent brands (1)} & \multicolumn{2}{c}{N chain brands (2)} & \multicolumn{2}{c}{Indep.\ $-$ chain (3)}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"& Coef. & (SE) & Coef. & (SE) & Coef. & (SE) \\", r"\midrule"]
for i, q in enumerate(QLAB):
    w = np.zeros(X.shape[1]); w[i] = 1; w[K + i] = -1
    dif, se = w @ r["b"], np.sqrt(w @ r["V"] @ w)
    cells = list(c3(rI["gap_b"][i], rI["gap_se"][i], rI["gap_p"][i])) + list(c3(rC["gap_b"][i], rC["gap_se"][i], rC["gap_p"][i])) + \
            list(c3(dif, se, pval(dif, se, r["G"])))
    L.append(f"{q} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_fact3_dynamic.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

NUM = re.compile(r"[-+]?\d+\.\d+")
for f in ("tab_fact1fact2_dynamic.tex", "tab_fact3_dynamic.tex"):
    pub = open(os.path.join(ROOT, "tables", "published", f), encoding="utf-8").read()
    new = open(os.path.join(ROOT, "tables", f), encoding="utf-8").read()
    a = [float(x) for l in pub.splitlines() if l[:2] == "20" for x in NUM.findall(l)]
    b = [float(x) for l in new.splitlines() if l[:2] == "20" for x in NUM.findall(l)]
    print(f, "tokens", len(a), len(b), "max |published - rebuilt| =", max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else "count mismatch")
