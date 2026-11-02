r"""
rev02b_controls_diagnostics.py -- why do geographic controls attenuate the pooled gap?
  (i)   one control at a time
  (ii)  overlap: distribution of distance to Albuquerque by group
  (iii) do the controls predict outcomes among UNEXPOSED ZIPs (always-legal-border + interior)?
        If a control has no predictive power where there is no nonlegal exposure, conditioning on it in the
        NL-vs-LB regression mainly absorbs exposure variation (it is collinear with Texas proximity).
  (iv)  imputation estimator: fit outcome ~ controls + log pop + quarter FE on unexposed ZIPs only,
        predict nonlegal-border ZIPs, average the gap; restricted to the region of common support.
  (v)   dose-response WITHIN nonlegal-border ZIPs (exposure intensity), with and without controls.
Output: output/rev02b_results.txt, tab_rev_controls.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
from _common import OUT, ROOT, stars
from _econ import fit, ols_fit, design, pval

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = pickle.load(open(os.path.join(OUT, "cache.pkl"), "rb"))
full = A["full"].copy()
full["a_med"] = np.arcsinh(full.n_med_incumbent / full.zip_pop * 1000)
full["a_zbp"] = np.arcsinh(full.zbp19_total_retail / full.zip_pop * 1000)
full["dist_abq_100"] = full.dist_abq / 100; full["dist_bigcity_100"] = full.dist_bigcity / 100
main = full[full.group != "interior"].copy()
OUTC = [("y_rpc", "rev/resident"), ("y_den", "density"), ("y_rps", "rev/store")]
CTRL = [("log_density", "log density"), ("dist_abq_100", "distance to ABQ"), ("dist_bigcity_100", "distance to city >=50k"),
        ("a_zbp", "2019 retail estab."), ("a_med", "medical incumbents")]

p("(i) One control at a time, pooled gap")
ONE = {}
for c, lab in CTRL:
    res = []
    for y, _ in OUTC:
        dd = main.dropna(subset=[y]) if y == "y_rps" else main
        r = fit(dd, y, dynamic=False, controls=[c])
        res.append((r["gap_b"][0], r["gap_se"][0], r["gap_p"][0], r["b"][-1], r["se"][-1]))
    ONE[c] = res
    p(f"  + {lab:<24s} " + "  ".join(f"gap {b:+.3f}({s:.3f}){stars(pv)} [ctrl {cb:+.3f}({cs:.3f})]" for b, s, pv, cb, cs in res))

p("\n(ii) Overlap in distance to Albuquerque (ZIP level)")
z = full.groupby("zip5").first()
for g in ["nonlegal", "legal", "interior"]:
    s = z[z.group == g].dist_abq
    p(f"  {g:<9s} min {s.min():6.1f} p25 {s.quantile(.25):6.1f} median {s.median():6.1f} p75 {s.quantile(.75):6.1f} max {s.max():6.1f}")
nl_d = z[z.group == "nonlegal"].dist_abq; lg_d = z[z.group == "legal"].dist_abq
lo, hi = max(nl_d.min(), lg_d.min()), min(nl_d.max(), lg_d.max())
p(f"  common support (NL vs LB): [{lo:.0f}, {hi:.0f}] mi; NL ZIPs inside: {int(nl_d.between(lo, hi).sum())}/92; "
  f"LB inside: {int(lg_d.between(lo, hi).sum())}/117")

p("\n(iii) Do the controls predict outcomes among unexposed ZIPs (always-legal + interior, 276 ZIPs)?")
un = full[full.nonlegal == 0].copy()
un["dummy"] = 0
PRED = {}
for y, lab in OUTC:
    dd = un.dropna(subset=[y]) if y == "y_rps" else un
    X, names = design(dd, dynamic=False, controls=[c for c, _ in CTRL], treat="dummy")
    X = X[:, 1:]; names = names[1:]
    r = ols_fit(dd[y].to_numpy(), X, dd.county_fips.to_numpy())
    pv = pval(r["b"], r["se"], r["G"])
    PRED[y] = {n: (r["b"][i], r["se"][i], pv[i]) for i, n in enumerate(names) if not n.startswith("fe")}
    p(f"  [{lab}] " + "  ".join(f"{n}: {r['b'][i]:+.3f}({r['se'][i]:.3f}){stars(pv[i])}" for i, n in enumerate(names) if not n.startswith("fe")))

p("\n(iv) Imputation estimator: outcome model fit on unexposed ZIPs, predicted for nonlegal-border ZIPs")
def impute(y, controls, support=None, B=499, seed=5):
    d = full.dropna(subset=[y]) if y == "y_rps" else full
    if support is not None:
        d = d[(d.nonlegal == 0) | d.dist_abq.between(*support)]
    d = d.copy(); d["dummy"] = 0
    X, names = design(d, dynamic=False, controls=controls, treat="dummy"); X = X[:, 1:]
    yv = d[y].to_numpy(); tr = d.nonlegal.to_numpy() == 1; cl = d.county_fips.to_numpy()
    def est(idx):
        Xi, yi, ti = X[idx], yv[idx], tr[idx]
        b = np.linalg.lstsq(Xi[~ti], yi[~ti], rcond=None)[0]
        return (yi[ti] - Xi[ti] @ b).mean()
    point = est(np.arange(len(d)))
    rng = np.random.default_rng(seed); cls = np.unique(cl); groups = {c: np.where(cl == c)[0] for c in cls}
    bs = []
    for _ in range(B):
        idx = np.concatenate([groups[c] for c in rng.choice(cls, len(cls))])
        if tr[idx].sum() == 0 or (~tr[idx]).sum() < X.shape[1]: continue
        bs.append(est(idx))
    bs = np.array(bs)
    return point, bs.std(), np.mean(np.sign(bs) != np.sign(point)) * 2
IMP = {}
for lab, ctr, sup in [("log pop only (unexposed = LB + interior)", [], None),
                      ("+ geography", ["log_density", "dist_abq_100", "dist_bigcity_100"], None),
                      ("+ geography + 2019 retail", ["log_density", "dist_abq_100", "dist_bigcity_100", "a_zbp"], None),
                      ("+ geography + 2019 retail, common support in distance to ABQ", ["log_density", "dist_abq_100", "dist_bigcity_100", "a_zbp"], (lo, hi))]:
    res = [impute(y, ctr, sup) for y, _ in OUTC]
    IMP[lab] = res
    p(f"  {lab:<62s} " + "  ".join(f"{b:+.3f}(boot SE {s:.3f}, p~{pv:.2f})" for b, s, pv in res))

p("\n(v) Dose-response within the 92 nonlegal-border ZIPs: log(1 + nonlegal residents within 75 mi / 1000)")
nl = full[full.nonlegal == 1].copy(); nl["E"] = np.log1p(nl.nlpop_75 / 1000)
DOSE = {}
for lab, ctr in [("no controls", []), ("+ geography + 2019 retail", ["log_density", "dist_abq_100", "dist_bigcity_100", "a_zbp"])]:
    res = []
    for y, _ in OUTC:
        dd = nl.dropna(subset=[y]) if y == "y_rps" else nl
        r = fit(dd, y, dynamic=False, controls=ctr, treat="E")
        res.append((r["gap_b"][0], r["gap_se"][0], r["gap_p"][0]))
    DOSE[lab] = res
    p(f"  {lab:<28s} " + "  ".join(f"{b:+.3f}({s:.3f}){stars(pv)}" for b, s, pv in res) + f"  [{nl.county_fips.nunique()} counties]")
zz = nl.groupby("zip5").first()
p(f"  exposure range among NL ZIPs: {zz.nlpop_75.min():,.0f} to {zz.nlpop_75.max():,.0f}; UT/OK-nearest NL ZIPs median "
  f"{zz[zz.nearest_nonlegal_state != 'TX'].nlpop_75.median():,.0f} vs TX-nearest median {zz[zz.nearest_nonlegal_state == 'TX'].nlpop_75.median():,.0f}")

# continuous exposure in the full 368-ZIP sample with controls
p("\n(vi) Continuous exposure, 368 ZIPs, with controls")
f2 = full.copy(); f2["E"] = np.log1p(f2.nlpop_75 / 1000)
CONT = {}
for lab, ctr in [("no controls", []), ("+ geography + 2019 retail", ["log_density", "dist_abq_100", "dist_bigcity_100", "a_zbp"]),
                 ("+ geography + 2019 retail + medical incumbents", ["log_density", "dist_abq_100", "dist_bigcity_100", "a_zbp", "a_med"])]:
    res = []
    for y, _ in OUTC:
        dd = f2.dropna(subset=[y]) if y == "y_rps" else f2
        r = fit(dd, y, dynamic=False, controls=ctr, treat="E")
        res.append((r["gap_b"][0], r["gap_se"][0], r["gap_p"][0]))
    CONT[lab] = res
    p(f"  {lab:<48s} " + "  ".join(f"{b:+.3f}({s:.3f}){stars(pv)}" for b, s, pv in res))

# table
def cell(t): return f"${t[0]:+.3f}^{{{stars(t[2])}}}$".replace("^{}", ""), f"({t[1]:.3f})"
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     r" & Revenue per & Dispensaries & log revenue \\", r" & resident (asinh) & per 1,000 (asinh) & per dispensary \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Nonlegal-border gap, one predetermined control at a time (209 ZIPs)}} \\"]
for c, lab in CTRL:
    a = [cell(t[:3]) for t in ONE[c]]
    L.append(f"+ {lab} & " + " & ".join(x[0] for x in a) + r" \\"); L.append(" & " + " & ".join(x[1] for x in a) + r" \\")
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: Imputation estimator (outcome model fit on 276 unexposed ZIPs)}} \\")
for lab, res in IMP.items():
    L.append(f"{lab} & " + " & ".join(f"${b:+.3f}$" for b, s, pv in res) + r" \\")
    L.append(" & " + " & ".join(f"({s:.3f})" for b, s, pv in res) + r" \\")
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel C: Exposure intensity, $\log(1+E_z/1000)$}} \\")
for lab, res in DOSE.items():
    a = [cell(t) for t in res]
    L.append(f"Within 92 nonlegal-border ZIPs, {lab} & " + " & ".join(x[0] for x in a) + r" \\"); L.append(" & " + " & ".join(x[1] for x in a) + r" \\")
for lab, res in CONT.items():
    a = [cell(t) for t in res]
    L.append(f"All 368 ZIPs, {lab} & " + " & ".join(x[0] for x in a) + r" \\"); L.append(" & " + " & ".join(x[1] for x in a) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tab_rev_controls.tex"), "w", encoding="utf-8").write("\n".join(L).replace("city >=50k", "city $\\geq$50k") + "\n")
pickle.dump(dict(ONE=ONE, PRED=PRED, IMP=IMP, DOSE=DOSE, CONT=CONT, support=(lo, hi)), open(os.path.join(OUT, "rev02b.pkl"), "wb"))
open(os.path.join(OUT, "rev02b_results.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
