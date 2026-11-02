r"""
rev09_chain_minus_indep.py -- (1) chain-minus-independent nonlegal-border gap by quarter, (2) chain share check.

Types (lifetime): chain = operator with >= 2 NM storefronts at any point in 2022Q2-2026Q2 (includes medical
incumbents and single-store operators that later expand); independent = operator with exactly one NM storefront
throughout the sample.

Figure, panel A (equation 3): stacked regression of asinh(N^tau_zq) with Nonlegal x quarter x type interactions,
  type-specific quarter fixed effects and type-specific log-population slopes; no quarter omitted. Identical to
  regressing D_zq = asinh(N^C) - asinh(N^I) on Nonlegal x quarter, quarter FE and log population.
Figure, panel B: adds ZIP-by-type fixed effects (persistent market attractiveness by type), so the Nonlegal x type
  level is absorbed and 2022Q2 is the omitted quarter: each coefficient is the change in the chain-minus-independent
  gap since market opening. Equivalent to D_zq on ZIP FE, quarter FE and Nonlegal x quarter (q > 2022Q2).
Table: chain share of storefronts as the outcome, ZIP-quarters with >= 1 store, three classifications.
County-clustered (23) standard errors; 95% intervals in the figure.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import OUT, QS, QMIN, QLAB, ROOT, stars
from _econ import design, ols_fit, pval

c = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
d = c["full"]
d = d[d.group.isin(["nonlegal", "legal"])].copy().reset_index(drop=True)
d["nonlegal"] = (d.group == "nonlegal").astype(int)
assert d.zip5.nunique() == 209 and len(d) == 3553
d["D"] = np.arcsinh(d.nb_C_life) - np.arcsinh(d.nb_I_life)

lines = []
def p(s=""):
    print(s); lines.append(str(s))

# ---------------- panel A: equation 3, no omitted quarter
X, names = design(d, dynamic=True)
rA = ols_fit(d.D, X, d.county_fips)
bA = np.array([rA["b"][names.index(f"g{q}")] for q in QS])
sA = np.array([rA["se"][names.index(f"g{q}")] for q in QS])
# check: stacked regression with type-specific FE and slopes gives the same differences
yC, yI = np.arcsinh(d.nb_C_life), np.arcsinh(d.nb_I_life)
bC = ols_fit(yC, X, d.county_fips)["b"]; bI = ols_fit(yI, X, d.county_fips)["b"]
assert np.allclose(bA, [bC[names.index(f"g{q}")] - bI[names.index(f"g{q}")] for q in QS])

def window(r, nm, qs):
    idx = [nm.index(f"g{q}") for q in qs]
    w = np.zeros(len(r["b"])); w[idx] = 1 / len(idx)
    m = w @ r["b"]; se = np.sqrt(w @ r["V"] @ w)
    return m, se, pval(m, se, r["G"])

early, late = QS[:4], QS[-9:]
p("Panel A: chain - independent nonlegal-border gap (eq. 3, type-specific quarter FE and pop slopes)")
for q, b, s in zip(QS, bA, sA):
    p(f"  {q//4}Q{q%4+1}: {b:+.3f} ({s:.3f}){stars(pval(b, s, rA['G']))}")
for lab, qs in [("early 2022Q2-2023Q1", early), ("late 2024Q2-2026Q2", late)]:
    m, se, pv = window(rA, names, qs); p(f"  mean {lab}: {m:+.3f} ({se:.3f}) p={pv:.3f}")

# ---------------- panel B: ZIP-by-type FE, 2022Q2 omitted
Q = d.qt.to_numpy()
cols, nmB = [], []
for q in QS[1:]:
    cols.append(d.nonlegal.to_numpy() * (Q == q)); nmB.append(f"g{q}")
for q in QS[1:]:
    cols.append((Q == q).astype(float)); nmB.append(f"fe{q}")
Z = pd.get_dummies(d.zip5).to_numpy(float)
XB = np.column_stack(cols + [Z])
rB = ols_fit(d.D, XB, d.county_fips)
bB = np.r_[0, [rB["b"][nmB.index(f"g{q}")] for q in QS[1:]]]
sB = np.r_[0, [rB["se"][nmB.index(f"g{q}")] for q in QS[1:]]]
p("\nPanel B: with ZIP-by-type FE, change since 2022Q2")
for q, b, s in zip(QS, bB, sB):
    p(f"  {q//4}Q{q%4+1}: {b:+.3f} ({s:.3f}){stars(pval(b, s, rB['G'])) if s > 0 else ''}")
mB, seB, pvB = window(rB, nmB, late); p(f"  mean late 2024Q2-2026Q2: {mB:+.3f} ({seB:.3f}) p={pvB:.3f}")
mB1, seB1, pvB1 = window(rB, nmB, early[1:]); p(f"  mean 2022Q3-2023Q1: {mB1:+.3f} ({seB1:.3f}) p={pvB1:.3f}")

# ---------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(len(QS))
for a, b, s, ttl in [(ax[0], bA, sA, "A. Chain $-$ independent gap (equation 3)"),
                     (ax[1], bB, sB, "B. Adding ZIP-by-type fixed effects (2022Q2 = 0)")]:
    a.axhline(0, color="black", lw=0.8)
    a.errorbar(x, b, yerr=1.96 * s, fmt="o", color="#1f4e79", ecolor="#7f9fbf", capsize=3, ms=5)
    a.set_xticks(x[::2]); a.set_xticklabels(QLAB[::2], rotation=45, ha="right", fontsize=8)
    a.set_title(ttl, fontsize=10); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel(r"$\hat\beta_{C,q}-\hat\beta_{I,q}$ (asinh operators)")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_rev_chain_minus_indep.png"), dpi=600)

# ---------------- chain share table
d["n_any"] = d.n_stores
pos = d[d.n_any > 0].copy()
p(f"\nChain share sample: {len(pos)} ZIP-quarters with >=1 store "
  f"(NL {int(pos.nonlegal.sum())}, LB {int((1 - pos.nonlegal).sum())}); ZIPs NL {pos[pos.nonlegal==1].zip5.nunique()}, "
  f"LB {pos[pos.nonlegal==0].zip5.nunique()}")
rows = []
for k, lab in [("life", "Lifetime"), ("pre", "At market opening"), ("entry", "Status at entry")]:
    pos[f"sh_{k}"] = pos[f"ns_C_{k}"] / pos.n_any
    cells = [lab]
    for w in [None, pos.n_any]:
        Xs, ns_ = design(pos, dynamic=False)
        r = ols_fit(pos[f"sh_{k}"], Xs, pos.county_fips, w=w)
        j = ns_.index("gap"); b, s = r["b"][j], r["se"][j]
        cells.append(f"{b:.3f}{stars(pval(b, s, r['G']))}"); cells.append(f"({s:.3f})")
        p(f"  {lab:18s} {'weighted' if w is not None else 'unweighted':10s} {b:+.3f} ({s:.3f})")
    Xd, nd = design(pos, dynamic=True)
    r = ols_fit(pos[f"sh_{k}"], Xd, pos.county_fips, w=pos.n_any)
    m, se, _ = window(r, nd, early); m2, se2, _ = window(r, nd, late)
    cells += [f"{m:.3f}", f"({se:.3f})", f"{m2:.3f}", f"({se2:.3f})"]
    cells.append(f"{pos[pos.nonlegal == 0][f'sh_{k}'].mean():.2f}")
    p(f"  {lab:18s} weighted early {m:+.3f} ({se:.3f}), late {m2:+.3f} ({se2:.3f})")
    rows.append(cells)

agg = d.groupby(["qt", "nonlegal"])[["ns_C_life", "n_stores"]].sum()
agg = (agg.ns_C_life / agg.n_stores).unstack()
p("  aggregate lifetime chain share NL/LB: " +
  ", ".join(f"{q//4}Q{q%4+1} {agg.loc[q,1]:.2f}/{agg.loc[q,0]:.2f}" for q in [QS[0], QS[4], QS[8], QS[12], QS[-1]]))

tex = [r"\begin{tabular}{lccccc}", r"\toprule",
       r" & \multicolumn{2}{c}{Pooled gap} & \multicolumn{2}{c}{Store-weighted window means} & Always-legal \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"Classification & Unweighted & Store-weighted & 2022Q2--2023Q1 & 2024Q2--2026Q2 & mean share \\",
       r"\midrule"]
for c_ in rows:
    tex.append(f"{c_[0]} & {c_[1]} & {c_[3]} & {c_[5]} & {c_[7]} & {c_[9]} \\\\")
    tex.append(f" & {c_[2]} & {c_[4]} & {c_[6]} & {c_[8]} & \\\\[2pt]")
tex += [r"\midrule", f"ZIP-quarters & \\multicolumn{{5}}{{l}}{{{len(pos)} ({int(pos.nonlegal.sum())} nonlegal border, "
        f"{int((1-pos.nonlegal).sum())} always-legal border)}} \\\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_chainshare.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
open(os.path.join(OUT, "rev09_chain_minus_indep.txt"), "w").write("\n".join(lines))
