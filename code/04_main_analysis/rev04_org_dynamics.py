r"""
rev04_org_dynamics.py -- referee Phase 4: chain vs independent dynamics without look-ahead.

Classifications (operator counts per ZIP-quarter, asinh, Table 4 design):
  life   chain = operator with >= 2 NM storefronts at any time in the sample   (manuscript Table 4; look-ahead)
  pre    chain = operator running >= 2 NM storefronts in 2022Q2               (predetermined at market opening)
  entry  a storefront counts as chain if its operator already ran another NM storefront when it first sold
Unified specification: y_diff = asinh(N_I) - asinh(N_C) on the Table 3 design. Because both type regressions
share the same regressors, its coefficients equal beta_I,q - beta_C,q exactly, with the joint clustered SE.
Early window = 2022Q2-2023Q1 (4 quarters); late window = 2024Q2-2026Q2 (9 quarters).
Also: excluding storefronts already operating in 2022Q2 (incumbents), preparedness proxies, diffusion refit.
Outputs: tab_rev_orgdyn.tex, fig_rev_orgdyn.png, tab_rev_prepared.tex, output/rev04_results.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from _common import OUT, ROOT, QS, QLAB, QMIN, stars
from _econ import fit, ols_fit, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
full = A["full"]; main = full[full.group != "interior"].copy()
sq = A["sq"].merge(A["meta"], on="storefront_id")

# counts excluding storefronts operating in 2022Q2 (entrants only)
d = sq[~sq.lc]
for cls, col in [("life", "chain_life"), ("pre", "chain_pre")]:
    for t, v in [("I", 0), ("C", 1)]:
        s = d[d[col] == v].groupby(["zip5", "qt"]).brand_id.nunique().rename(f"nbx_{t}_{cls}").reset_index()
        main = main.merge(s, on=["zip5", "qt"], how="left"); main[f"nbx_{t}_{cls}"] = main[f"nbx_{t}_{cls}"].fillna(0)
for t, m in [("I", d.status_entry != "established"), ("C", d.status_entry == "established")]:
    s = d[m].groupby(["zip5", "qt"]).brand_id.nunique().rename(f"nbx_{t}_entry").reset_index()
    main = main.merge(s, on=["zip5", "qt"], how="left"); main[f"nbx_{t}_entry"] = main[f"nbx_{t}_entry"].fillna(0)

EARLY, LATE = list(range(0, 4)), list(range(8, 17))
SPECS = [("life", "nb_{}_life", "Lifetime"), ("pre", "nb_{}_pre", "Predetermined at opening"),
         ("entry", "nb_{}_entry", "Status at entry"),
         ("life_x", "nbx_{}_life", "Lifetime, excl. stores open in 2022Q2"),
         ("pre_x", "nbx_{}_pre", "Predetermined, excl. stores open in 2022Q2"),
         ("entry_x", "nbx_{}_entry", "At entry, excl. stores open in 2022Q2")]

def lincomb(r, idx):
    w = np.zeros(len(r["b"])); w[idx] = 1 / len(idx)
    b = w @ r["b"]; se = np.sqrt(w @ r["V"] @ w)
    return b, se, 2 * (1 - stats.t.cdf(abs(b / se), r["G"] - 1))

RES = {}
p("Unified test: beta_I,q - beta_C,q (asinh operator counts), 209 ZIPs, county-clustered")
for key, tmpl, lab in SPECS:
    yI, yC = tmpl.format("I"), tmpl.format("C")
    main[f"aI_{key}"] = np.arcsinh(main[yI]); main[f"aC_{key}"] = np.arcsinh(main[yC])
    main[f"dif_{key}"] = main[f"aI_{key}"] - main[f"aC_{key}"]
    rI, rC, rD = fit(main, f"aI_{key}"), fit(main, f"aC_{key}"), fit(main, f"dif_{key}")
    gi = [i for i, n in enumerate(rD["names"]) if n.startswith("g")]
    e = lincomb(rD, [gi[i] for i in EARLY]); l = lincomb(rD, [gi[i] for i in LATE])
    # equivalence (TOST) for the late average with margin = |early average| (the size of the initial chain lead)
    m = abs(e[0]); tc = stats.t.ppf(0.95, rD["G"] - 1)
    tost_p = max(1 - stats.t.cdf((l[0] + m) / l[1], rD["G"] - 1), stats.t.cdf((l[0] - m) / l[1], rD["G"] - 1))
    RES[key] = dict(lab=lab, I=rI, C=rC, D=rD, early=e, late=l, tost_p=tost_p, margin=m,
                    n_sig_early=int((rD["gap_p"][:4] < .05).sum()), n_sig_late=int((rD["gap_p"][8:] < .05).sum()),
                    meanI=main[yI].mean(), meanC=main[yC].mean())
    p(f"  [{lab}] 2022Q2: I {rI['gap_b'][0]:+.3f}({rI['gap_se'][0]:.3f}) C {rC['gap_b'][0]:+.3f}({rC['gap_se'][0]:.3f}) "
      f"I-C {rD['gap_b'][0]:+.3f}({rD['gap_se'][0]:.3f}){stars(rD['gap_p'][0])} | early avg {e[0]:+.3f}({e[1]:.3f}) p={e[2]:.3f} | "
      f"late avg {l[0]:+.3f}({l[1]:.3f}) p={l[2]:.3f}, 90% CI [{l[0]-tc*l[1]:+.3f},{l[0]+tc*l[1]:+.3f}], TOST p (margin {m:.3f}) = {tost_p:.3f} "
      f"| sig early {RES[key]['n_sig_early']}/4, late {RES[key]['n_sig_late']}/9")

# predicted counts (interpretable units): mean predicted operators of each type, NL with vs without gap
p("\nInterpretable magnitudes, lifetime classification (mean operators per NL ZIP-quarter, actual vs. NL indicator switched off)")
for t in ["I", "C"]:
    r = RES["life"][t]; X = r["X"]; nl = main.nonlegal.to_numpy() == 1
    yhat = X @ r["b"]
    Xo = X.copy(); Xo[:, [i for i, n in enumerate(r["names"]) if n.startswith("g")]] = 0
    yo = Xo @ r["b"]
    for qi, qq in [(0, "2022Q2"), (16, "2026Q2")]:
        msk = nl & (main.qt.to_numpy() == QS[qi])
        p(f"  {t} {qq}: sinh(fitted) {np.sinh(yhat[msk]).mean():.3f} vs counterfactual {np.sinh(yo[msk]).mean():.3f} "
          f"(actual mean {main.loc[msk, 'nb_'+t+'_life'].mean():.3f})")

# figure: I-C difference by classification
fig, ax = plt.subplots(figsize=(7.5, 4.0))
x = np.arange(17)
for k, (key, col, mk) in enumerate([("life", "#1f3b5c", "o"), ("pre", "#b5651d", "s"), ("pre_x", "#4a7c59", "^")]):
    r = RES[key]["D"]
    ax.errorbar(x + (k - 1) * 0.22, r["gap_b"], yerr=1.96 * r["gap_se"], fmt=mk, ms=3.5, capsize=2, lw=0.9, color=col, label=RES[key]["lab"])
ax.axhline(0, color="grey", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(QLAB, rotation=90, fontsize=8)
ax.set_ylabel(r"$\hat\beta_{I,q}-\hat\beta_{C,q}$"); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "fig_rev_orgdyn.png"), dpi=600); plt.close(fig)

L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{2022Q2} & Early mean & Late mean & TOST & Late quarters \\",
     r"\cmidrule(lr){2-3}",
     r"Classification & Indep. & Chain & $I-C$ & $I-C$ & $p$-value & with $p<.05$ \\", r"\midrule"]
for key, _, lab in SPECS:
    R = RES[key]
    if key == "life_x": L.append(r"\addlinespace\multicolumn{7}{@{}l}{\textit{Excluding storefronts already operating in 2022Q2}} \\")
    c = lambda b, se, pv: (f"${b:+.3f}^{{{stars(pv)}}}$".replace("^{}", ""), f"({se:.3f})")
    a1 = c(R["I"]["gap_b"][0], R["I"]["gap_se"][0], R["I"]["gap_p"][0]); a2 = c(R["C"]["gap_b"][0], R["C"]["gap_se"][0], R["C"]["gap_p"][0])
    a3 = c(*R["early"]); a4 = c(*R["late"])
    L.append(f"{lab.replace(', excl. stores open in 2022Q2','')} & {a1[0]} & {a2[0]} & {a3[0]} & {a4[0]} & {R['tost_p']:.3f} & {R['n_sig_late']}/9 \\\\")
    L.append(f" & {a1[1]} & {a2[1]} & {a3[1]} & {a4[1]} & & \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_orgdyn.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ preparedness proxies
p("\nPreparedness proxies (predetermined): operator storefronts in 2022Q2, medical-incumbent operator")
st = A["sf"][["storefront_id", "zip5", "brand_id", "med_incumbent"]].merge(A["meta"], on="storefront_id")
grp = A["zx"].set_index("zip5").group; st["group"] = st.zip5.map(grp)
med_op = st.groupby("brand_id").med_incumbent.max()
st["med_operator"] = st.brand_id.map(med_op)
st["exp_open"] = np.log1p(st.n_open_brand)
st["county_fips"] = st.zip5.map(A["zx"].set_index("zip5").county_fips)
new = st[~st.lc & st.group.isin(["nonlegal", "legal"])].copy()
new["nonlegal"] = (new.group == "nonlegal").astype(int)
new["entry_q"] = new.faq - QMIN          # quarters after market opening
new["early"] = (new.entry_q <= 3).astype(int)
import statsmodels.formula.api as smf
PREP = {}
for y, lab in [("entry_q", "Quarters from opening to entry"), ("early", "Entered within first year (LPM)")]:
    m = smf.ols(f"{y} ~ exp_open + med_operator + nonlegal + exp_open:nonlegal", new).fit(
        cov_type="cluster", cov_kwds={"groups": new.county_fips})
    PREP[y] = m
    p(f"  {lab}: " + "  ".join(f"{k} {m.params[k]:+.3f}({m.bse[k]:.3f}){stars(m.pvalues[k])}" for k in m.params.index if k != "Intercept")
      + f"  N={int(m.nobs)}, mean {new[y].mean():.2f}")
# share of the stock in 2022Q2 that is medical-incumbent, by classification and group
s0 = sq[sq.qt == QMIN].merge(A["sf"][["storefront_id"]], on="storefront_id")
s0["group"] = s0.zip5.map(grp)
for g in ["nonlegal", "legal"]:
    x = s0[s0.group == g]
    p(f"  2022Q2 stock in {g} ZIPs: {len(x)} storefronts; chain(life) {int(x.chain_life.sum())}, of which medical incumbents "
      f"{int(x[(x.chain_life==1)].med_incumbent.sum())}; independent(life) {int((1-x.chain_life).sum())}, medical incumbents "
      f"{int(x[(x.chain_life==0)].med_incumbent.sum())}")
first_loc_early = st[(~st.lc) & (st.status_entry == "first_loc")]
p(f"  first-location storefronts (operator expands later) entering after opening: {len(first_loc_early)}; "
  f"share whose operator's 2nd store opened later than this one: by construction 100%")

L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r" & Quarters from & Entered within \\", r" & opening to entry & first year \\", r"\midrule"]
names = {"exp_open": "$\\log(1+\\text{operator storefronts in 2022Q2})$", "med_operator": "Operator was a medical incumbent",
         "nonlegal": "Nonlegal-border ZIP", "exp_open:nonlegal": "$\\log(1+\\text{storefronts in 2022Q2})\\times$ Nonlegal"}
for k, nm in names.items():
    L.append(nm + " & " + " & ".join(f"${PREP[y].params[k]:+.3f}^{{{stars(PREP[y].pvalues[k])}}}$".replace("^{}", "") for y in ["entry_q", "early"]) + r" \\")
    L.append(" & " + " & ".join(f"({PREP[y].bse[k]:.3f})" for y in ["entry_q", "early"]) + r" \\")
L.append(r"\midrule")
L.append("Mean of outcome & " + " & ".join(f"{new[y].mean():.2f}" for y in ["entry_q", "early"]) + r" \\")
L.append(f"Storefronts (entered after 2022Q2) & \\multicolumn{{2}}{{c}}{{{len(new)}}} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_prepared.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ diffusion refit (common lambda, common beta_inf)
p("\nDiffusion model refit: beta_k(t) = b_inf [1 - (1 - phi_k) exp(-lambda t)], WNLS on quarter coefficients")
DIF = {}
for key in ["life", "pre", "entry"]:
    bI, sI = RES[key]["I"]["gap_b"], RES[key]["I"]["gap_se"]; bC, sC = RES[key]["C"]["gap_b"], RES[key]["C"]["gap_se"]
    t = np.arange(17.0)
    def resid(th):
        binf, lam, phI, phC = th
        fI = binf * (1 - (1 - phI) * np.exp(-lam * t)); fC = binf * (1 - (1 - phC) * np.exp(-lam * t))
        return np.r_[(bI - fI) / sI, (bC - fC) / sC]
    sol = optimize.least_squares(resid, x0=[0.35, 0.4, 0.1, 0.5], bounds=([0, 0.001, -2, -2], [3, 10, 2, 2]))
    J = sol.jac; cov = np.linalg.pinv(J.T @ J) * (sol.fun @ sol.fun) / (34 - 4)
    se = np.sqrt(np.diag(cov))
    dphi = sol.x[3] - sol.x[2]; sed = np.sqrt(cov[2, 2] + cov[3, 3] - 2 * cov[2, 3])
    DIF[key] = dict(x=sol.x, se=se, dphi=dphi, sed=sed)
    p(f"  [{RES[key]['lab']}] b_inf {sol.x[0]:.3f}({se[0]:.3f}) lambda {sol.x[1]:.3f}({se[1]:.3f}) phi_I {sol.x[2]:.3f}({se[2]:.3f}) "
      f"phi_C {sol.x[3]:.3f}({se[3]:.3f}); phi_C - phi_I {dphi:+.3f} ({sed:.3f})")
pickle.dump(dict(RES={k: {kk: vv for kk, vv in v.items() if kk not in ("I", "C", "D")} |
                      {"I": (v["I"]["gap_b"], v["I"]["gap_se"], v["I"]["gap_p"]), "C": (v["C"]["gap_b"], v["C"]["gap_se"], v["C"]["gap_p"]),
                       "D": (v["D"]["gap_b"], v["D"]["gap_se"], v["D"]["gap_p"])} for k, v in RES.items()},
                 PREP={y: (m.params.to_dict(), m.bse.to_dict(), m.pvalues.to_dict(), int(m.nobs)) for y, m in PREP.items()}, DIF=DIF),
            open(os.path.join(OUT, "rev04.pkl"), "wb"))
open(os.path.join(OUT, "rev04_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
