"""
ext02_structural_model.py  --  extension of venous4

A free-entry model of cannabis retail entry with two organizational types:
  I  independent  (single-outlet operator; ignores the business-stealing externality)
  F  franchise / chain (multi-outlet operator; internalizes cannibalization of its
     own nearby stores; per-store cost advantage phi_F = psi * phi_I, psi <= 1).

Calibrated on the cross-section of ~78 New Mexico town-markets at their mature
(2024-2025 mean) configuration (old1 build `city_cross_section.csv`), using the
paper's effective-market-size device  S_m = pop_m + lambda * NL_m.

Solver is closed-form + a single monotone sweep -> guaranteed to terminate.

Outputs (project root): ext_tab_calibration.tex, ext_tab_comparative_statics.tex,
  ext_tab_counterfactual.tex, ext_fig_comparative_statics.png, ext_fig_model_fit.png,
  ext_output/ext02_structural_model.log
"""
import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CS = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build\city_cross_section.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ext_output", "ext02_structural_model.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

d = pd.read_csv(CS)
d = d.dropna(subset=["city_pop", "border_city"]).copy()
for c in ["N_stores", "N_brands", "N_indep", "N_chain", "revenue", "nonleg_pop_75"]:
    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
d["border"] = d["border_city"].astype(int)
d = d[d["is_abq"] == 0].reset_index(drop=True)
p(f"town-markets (excl. ABQ metro): {len(d)}  ({int(d['border'].sum())} border / "
  f"{int((d['border']==0).sum())} interior)")
p(f"mean resident pop: border {d.loc[d.border==1,'city_pop'].mean():,.0f} / "
  f"interior {d.loc[d.border==0,'city_pop'].mean():,.0f}")
p(f"mean nonlegal pop within 75mi (border towns): {d.loc[d.border==1,'nonleg_pop_75'].mean():,.0f}")
p(f"mean stores: border {d.loc[d.border==1,'N_stores'].mean():.2f} / interior "
  f"{d.loc[d.border==0,'N_stores'].mean():.2f}")

# ============================================================ 1. lambda
H("1.  MARKET-SIZE WEIGHT lambda:  N = exp(b0 + a*log(pop + lambda*NL75))   [NLS]")
def _m(X, b0, a, lam):
    pop, nl = X
    return np.exp(b0 + a * np.log(np.clip(pop + lam * nl, 1, None)))
popt, pcov = curve_fit(_m, (d["city_pop"].values, d["nonleg_pop_75"].values),
                       d["N_stores"].values, p0=[-3, 0.5, 0.05],
                       bounds=([-15, 0, 0], [5, 3, 5]), maxfev=20000)
se = np.sqrt(np.diag(pcov)); b0_hat, a_hat, lam = popt
p(f"  b0={b0_hat:.3f}({se[0]:.3f})  a={a_hat:.3f}({se[1]:.3f})  lambda={lam:.4f}({se[2]:.4f})")
d["S"] = d["city_pop"] + lam * d["nonleg_pop_75"]
d["logS"] = np.log(d["S"].clip(lower=1))

# ============================================================ 2. revenue technology
H("2.  REVENUE / CONGESTION:  per-store revenue  r(N) = kappa * S / N^rho")
# Bresnahan-Reiss thresholds on this cross-section (old1 log): interior S_1~5,147, S_2~10,547,
# S_3~12,923  =>  S_N proportional to N^rho with rho ~ 1  =>  per-store revenue r(N) ~ kappa S / N.
BR_thr = {1: 5147.0, 2: 10547.0, 3: 12923.0}
lnN = np.log(list(BR_thr.keys())); lnS = np.log(list(BR_thr.values()))
rho_br = float(np.polyfit(lnN, lnS, 1)[0])
dr = d[(d["N_stores"] >= 1) & (d["revenue"] > 0)].copy()
dr["logrps"] = np.log(dr["revenue"] / dr["N_stores"]); dr["logN"] = np.log(dr["N_stores"])
m2 = sm.OLS(dr["logrps"], sm.add_constant(dr[["logS", "logN"]])).fit(cov_type="HC1")
p(m2.summary().tables[1].as_text())
p(f"  cross-sectional -coef(logN) = {-m2.params['logN']:.2f} (n.s., small sample, N mechanically "
  f"correlated with revenue); Bresnahan-Reiss threshold slope rho = {rho_br:.2f}")
rho = 1.0
p(f"  rho imposed at {rho:.2f}  (total town revenue independent of N; the paper's extensive-margin result)")
one = d[(d["N_stores"].round() == 1) & (d["revenue"] > 0)]
kappa = float((one["revenue"] / one["S"]).median())
p(f"  kappa = median(revenue / S | N=1) = {kappa:,.1f}   (n={len(one)} one-store towns)")

def r_of(N, S):
    N = np.asarray(N, float)
    return kappa * S / np.power(np.maximum(N, 1e-9), rho)

# ============================================================ 3. entry costs
H("3.  ENTRY / OPERATING COST  phi_I (independent),  phi_F = psi * phi_I (chain)")
MEAN_POP = float(d["city_pop"].mean())
NL_BORDER = float(d.loc[d.border == 1, "nonleg_pop_75"].mean())
POP_INT = float(d.loc[d.border == 0, "city_pop"].mean())
POP_BOR = float(d.loc[d.border == 1, "city_pop"].mean())
CHAIN_SIZE = 4.0

# phi_I is the independent free-entry cost.  With rho=1 the total store count is
# N*(S) = floor(kappa*S/phi_I).  Calibrate phi_I so the model reproduces the TOTAL number of
# stores across the 78 town-markets (a single clean free-entry moment).
N_total_data = float(d["N_stores"].round().sum())
def total_N(phi):
    return sum(int(np.floor(kappa * s / phi)) if kappa * s >= phi else 0 for s in d["S"])
lo, hi = kappa * d["S"].min() * 0.2, kappa * d["S"].max()
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if total_N(mid) > N_total_data:
        lo = mid
    else:
        hi = mid
phi_I = 0.5 * (lo + hi)
S1_thr = phi_I / kappa          # effective market size that supports the 1st store
p(f"  total stores: data {N_total_data:.0f}  model {total_N(phi_I)}   ->  phi_I = {phi_I:,.0f}")
p(f"  implied thresholds:  1st store at S = {S1_thr:,.0f};  2nd at S = {2*S1_thr:,.0f}  "
  f"(Bresnahan-Reiss non-parametric medians: {1768:,}, {5414:,} -- rho=1 compresses the spacing)")

Nmax = 80
CANNIB = 0.25      # effective own-store cannibalization  (updated after calibration)
def equilibrium(S, phi_I, cannib=None, cap=None):
    """Free-entry equilibrium (N_F, N_I).

    Total count.  With rho = 1 the independent free-entry margin sets the total:
        N* = floor(kappa * S / phi_I)          (0 if the first store cannot break even).

    Composition.  Slots are ranked by proximity to demand; a chain fills slot k (its own
    k-th town store) only while its marginal willingness exceeds an independent's.  The
    chain's marginal own store yields gross revenue r(k) but takes a fraction `cannib` of a
    full outlet's sales from EACH of its k-1 existing town stores, so the chain values slot k
    at  r(k) * (1 - cannib*(k-1)).  An independent, which does not internalize that loss,
    values any slot at phi_I.  Hence
        n_F = largest k <= N*  with  r(k) * (1 - cannib*(k-1)) >= phi_I ,
    and independents take the remaining N* - n_F slots.  Because the discount 1 - cannib*(k-1)
    reaches zero at k = 1 + 1/cannib, n_F is bounded by 1 + 1/cannib regardless of market
    size:  chains stop adding own stores while r is still well above cost, and every further
    slot a larger (border) market supports goes to an independent.
    """
    if cannib is None:
        cannib = CANNIB
    if kappa * S < phi_I:
        return 0, 0
    Nstar = int(np.floor(kappa * S / phi_I + 1e-9))
    Nstar = max(1, min(Nstar, Nmax))
    if cap is not None:
        Nstar = min(Nstar, int(cap))
    nF = 0
    for k in range(1, Nstar + 1):
        if r_of(k, S) * (1.0 - cannib * (k - 1)) >= phi_I:
            nF = k
        else:
            break
    return nF, Nstar - nF

# two data targets for the chain block: chain share of stores, interior and border town-groups
csh_int_data = float(d.loc[d.border == 0, "N_chain"].sum() / d.loc[d.border == 0, "N_stores"].sum())
csh_bor_data = float(d.loc[d.border == 1, "N_chain"].sum() / d.loc[d.border == 1, "N_stores"].sum())
p(f"  data chain share of stores:  interior {csh_int_data:.3f}   border {csh_bor_data:.3f}")

# town-level Poisson border coefficients (validation, not targeted)
def pois_border(y):
    dd = d.copy(); dd["yr"] = dd[y].round().astype(int)
    mm = sm.GLM(dd["yr"], sm.add_constant(dd[["logS", "border"]]),
                family=sm.families.Poisson()).fit(cov_type="HC1")
    return float(mm.params["border"])
bI_data, bF_data = pois_border("N_indep"), pois_border("N_chain")
p(f"  town Poisson border coef (control logS, validation):  N_indep {bI_data:+.3f}   N_chain {bF_data:+.3f}")

# The paper's headline franchise/independent fact is the COMPOSITION OF THE BORDER INCREMENT:
# border ZIPs have +1.26 independent brands but +0.63 (n.s.) chain brands -- the extra stores a
# border market supports are about 2/3 independent.  Calibrate the single chain-block parameter
# c (own-store cannibalization) so the model's border-town increment (baseline minus the
# no-cross-border counterfactual) has that same independent share.
INC_ISHARE_TARGET = 1.26 / (1.26 + 0.63)
p(f"  target: independent share of the border store increment = {INC_ISHARE_TARGET:.3f}  "
  f"(ZIP reduced form: +1.26 independent vs +0.63 chain brands)")
def border_increment(c):
    Fb = Ib = Fp = Ip = 0.0
    for _, row in d[d.border == 1].iterrows():
        f, i = equilibrium(row["S"], phi_I, cannib=c)
        fp, ip = equilibrium(row["city_pop"], phi_I, cannib=c)
        Fb += f; Ib += i; Fp += fp; Ip += ip
    dF, dI = Fb - Fp, Ib - Ip
    return dI / (dF + dI) if (dF + dI) > 0 else np.nan, (Fb, Ib, Fp, Ip)
def shares(c):
    F = I = Fb = Ib = 0.0
    for _, row in d.iterrows():
        f, i = equilibrium(row["S"], phi_I, cannib=c)
        if row["border"] == 1:
            Fb += f; Ib += i
        else:
            F += f; I += i
    return (F / (F + I) if F + I else 0), (Fb / (Fb + Ib) if Fb + Ib else 0)
grid_c = np.round(np.linspace(0.05, 0.70, 66), 3)
_cand = [(c, border_increment(c)[0]) for c in grid_c]
cannib_c = float(min(_cand, key=lambda t: abs((t[1] if np.isfinite(t[1]) else 0) - INC_ISHARE_TARGET))[0])
CANNIB = cannib_c
inc_ish, _ = border_increment(cannib_c)
si, sb = shares(cannib_c)
p(f"  calibrated  c (own-store cannibalization fraction) = {cannib_c:.3f}   "
  f"=> chain stores per town bounded by 1 + 1/c = {1 + 1/cannib_c:.1f}")
p(f"  model independent share of the border increment = {inc_ish:.3f}  (target {INC_ISHARE_TARGET:.3f})")
p(f"  non-targeted -- chain share of stores:  interior {si:.3f} (data {csh_int_data:.3f})   "
  f"border {sb:.3f} (data {csh_bor_data:.3f})")

# ============================================================ 4. fit (non-targeted)
H("4.  MODEL FIT  (non-targeted moments)")
fi = d.apply(lambda r_: equilibrium(r_["S"], phi_I), axis=1, result_type="expand")
d["nF_hat"], d["nI_hat"] = fi[0], fi[1]
d["N_hat"] = d["nF_hat"] + d["nI_hat"]
for lab, m in [("all", d.index.to_series() > -1), ("interior", d.border == 0), ("border", d.border == 1)]:
    s = d[m]
    p(f"  {lab:9s}  N act {s.N_stores.mean():5.2f} / mod {s.N_hat.mean():5.2f}    "
      f"indep act {s.N_indep.mean():5.2f} / mod {s.nI_hat.mean():5.2f}    "
      f"chain act {s.N_chain.mean():5.2f} / mod {s.nF_hat.mean():5.2f}")
corrN = float(np.corrcoef(d.N_stores, d.N_hat)[0, 1])
p(f"  corr(actual N, model N) = {corrN:.3f}")
d["ish_act"] = d.N_indep / d.N_stores.clip(lower=1)
d["ish_hat"] = d.nI_hat / d.N_hat.clip(lower=1)
for lab, m in [("interior", d.border == 0), ("border", d.border == 1)]:
    s = d[m]
    p(f"  indep. store share {lab:9s}:  actual {s.ish_act.mean():.3f}   model {s.ish_hat.mean():.3f}")
# per-store revenue premium (model)
def rps(pop, nl):
    S = pop + lam * nl
    n = sum(equilibrium(S, phi_I))
    return float(r_of(n, S)) if n else np.nan
r_int, r_bor = rps(MEAN_POP, 0), rps(MEAN_POP, NL_BORDER)
p(f"  per-store revenue  interior {r_int:,.0f}  border {r_bor:,.0f}  log gap {np.log(r_bor/r_int):+.3f}"
  f"   (data +0.228, n.s.)")

# ============================================================ 5. comparative statics
REP_POP = 15000.0
H(f"5.  COMPARATIVE STATICS  -- representative town (pop {REP_POP:,.0f}), vary cross-border population")
NLg = np.linspace(0, 800_000, 41)
rows = []
for nl in NLg:
    S = REP_POP + lam * nl
    nF, nI = equilibrium(S, phi_I)
    N = nF + nI
    g = max(1, round(nF / CHAIN_SIZE)) if nF else 0
    sh = ([nF / g / N] * g if g else []) + ([1 / N] * nI if N else [])
    hhi = 10000 * float(np.sum(np.square(sh))) if N else np.nan
    rows.append(dict(NL=nl, S=S, N=N, nF=nF, nI=nI,
                     r_store=float(r_of(N, S)) if N else np.nan, hhi=hhi,
                     indep_share=nI / N if N else np.nan))
cs = pd.DataFrame(rows)
cs.to_csv(os.path.join(ROOT, "ext_output", "ext_comparative_statics.csv"), index=False)
for nl in [0, 100_000, 300_000, 500_000, 800_000]:
    r = cs.iloc[(cs.NL - nl).abs().values.argmin()]
    p(f"  NL={nl:>8,.0f}:  N={r.N:.0f} (chain {r.nF:.0f}, indep {r.nI:.0f})  r/store={r.r_store:,.0f}  "
      f"HHI={r.hhi:,.0f}  indep sh={r.indep_share:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].plot(cs.NL / 1e3, cs.nI, lw=2, label=r"independent $N_I^*$")
ax[0].plot(cs.NL / 1e3, cs.nF, lw=2, ls="--", label=r"chain $N_F^*$")
ax[0].plot(cs.NL / 1e3, cs.N, lw=1.3, color="k", alpha=.6, label=r"total $N^*$")
ax[0].set_title("Equilibrium number of stores"); ax[0].legend(fontsize=8)
ax[1].plot(cs.NL / 1e3, cs.r_store / 1e3, lw=2, color="C3")
ax[1].set_title("Revenue per store ($000/yr)"); ax[1].set_ylim(bottom=0)
ax[2].plot(cs.NL / 1e3, cs.hhi, lw=2, color="C4")
ax[2].set_title("Concentration (HHI, 0-10000)")
for a in ax:
    a.set_xlabel("nonlegal pop. within 75 mi (000s)"); a.grid(alpha=.25)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Model comparative statics: a cross-border demand increase under free entry", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(os.path.join(ROOT, "ext_fig_comparative_statics.png"), dpi=140); plt.close(fig)
p("wrote ext_fig_comparative_statics.png")

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for b, c, lb in [(0, "C0", "interior"), (1, "C3", "border")]:
    s = d[d.border == b]
    ax[0].scatter(s.N_stores, s.N_hat, s=22, alpha=.7, color=c, label=lb)
lim = [0, min(30, d.N_stores.max() + 2)]
ax[0].plot(lim, lim, "k--", lw=.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("actual # stores"); ax[0].set_ylabel("model # stores")
ax[0].set_title(f"Store count (corr {corrN:.2f})"); ax[0].legend(fontsize=8)
x = np.arange(2); w = .18
vals = [("indep actual", [d[d.border==0].N_indep.mean(), d[d.border==1].N_indep.mean()], "C0", 1.0),
        ("indep model",  [d[d.border==0].nI_hat.mean(),  d[d.border==1].nI_hat.mean()],  "C0", .5),
        ("chain actual",  [d[d.border==0].N_chain.mean(), d[d.border==1].N_chain.mean()], "C3", 1.0),
        ("chain model",   [d[d.border==0].nF_hat.mean(),  d[d.border==1].nF_hat.mean()],  "C3", .5)]
for i, (lb, v, c, al) in enumerate(vals):
    ax[1].bar(x + (i - 1.5) * w, v, w, label=lb, color=c, alpha=al)
ax[1].set_xticks(x); ax[1].set_xticklabels(["interior", "border"])
ax[1].set_title("Composition: actual vs model"); ax[1].legend(fontsize=7)
for a in ax:
    a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "ext_fig_model_fit.png"), dpi=140); plt.close(fig)
p("wrote ext_fig_model_fit.png")

# ============================================================ 6. counterfactuals
H("6.  COUNTERFACTUALS  (28 border town-markets)")
bt = d[d.border == 1].copy()
def tot(Svec, **kw):
    F = I = 0.0
    for S in Svec:
        f, i = equilibrium(S, phi_I, **kw); F += f; I += i
    return F, I
base_F, base_I = tot(bt.S.values)
nl0_F, nl0_I = tot(bt.city_pop.values)                       # (1) no cross-border demand
nc_F, nc_I = tot(bt.S.values, cannib=1e-9)                   # (2) chains ignore cannibalization
capF = capI = 0.0; rf, rc = [], []
for _, row in bt.iterrows():
    cap = sum(equilibrium(row.city_pop, phi_I))              # (3) cap at interior-implied count
    f, i = equilibrium(row.S, phi_I)
    fc, ic = equilibrium(row.S, phi_I, cap=cap)
    capF += fc; capI += ic
    if f + i: rf.append(float(r_of(f + i, row.S)))
    if fc + ic: rc.append(float(r_of(fc + ic, row.S)))
r_free, r_cap = float(np.mean(rf)), float(np.mean(rc))
def L(lbl, F, I):
    N = F + I
    p(f"  {lbl:42s} N={N:6.1f}  chain={F:6.1f}  indep={I:6.1f}  indep share={I/N if N else 0:.2f}")
L("baseline (free entry + cross-border)", base_F, base_I)
L("(1) no cross-border demand", nl0_F, nl0_I)
L("(2) chains ignore cannibalization", nc_F, nc_I)
L("(3) license cap at interior-implied N", capF, capI)
extra = (base_F + base_I) - (nl0_F + nl0_I)
p(f"\n  cross-border demand -> +{extra:.0f} border stores "
  f"({extra/(base_F+base_I):.0%} of the border stock); of the increment, "
  f"chain +{base_F-nl0_F:.0f} vs independent +{base_I-nl0_I:.0f}  "
  f"({(base_I-nl0_I)/max(extra,1e-9):.0%} independent)")
p(f"  (2) if chains did not internalize cannibalization the market would be "
  f"{nc_F/(nc_F+nc_I):.0%} chain instead of {base_F/(base_F+base_I):.0%}")
p(f"  (3) license cap: mean per-store revenue {r_free:,.0f} (free) -> {r_cap:,.0f} (capped), "
  f"log gap {np.log(r_cap/r_free):+.3f}  [extensive -> intensive margin, as in NM vs CO]")

# ---------------- LaTeX tables
def w(fn, s):
    open(os.path.join(ROOT, fn), "w", encoding="utf-8").write(s); p("wrote " + fn)

w("ext_tab_calibration.tex", r"""\begin{table}[htbp]
\centering
\caption{Calibration of the two-type free-entry model}
\label{tab:ext_calibration}
\footnotesize
\begin{tabular}{@{}l p{0.60\textwidth} r@{}}
\toprule
Parameter & Meaning and source & Value \\
\midrule
\multicolumn{3}{@{}l}{\emph{Estimated on the town-market cross-section}}\\
$\lambda$  & value of a nonlegal-state resident within 75\,mi relative to a local resident (NLS) & """ + f"{lam:.4f}" + r""" \\
$a$        & elasticity of the store count to $\log S_m$ (NLS) & """ + f"{a_hat:.2f}" + r""" \\
$\rho$     & congestion in per-store revenue, $r(N)=\kappa S/N^{\rho}$ ($\rho{=}1$ imposed) & """ + f"{rho:.2f}" + r""" \\
$\kappa$   & revenue per unit of effective market size (\$/yr) & """ + f"{kappa:,.0f}" + r""" \\
\midrule
\multicolumn{3}{@{}l}{\emph{Calibrated}}\\
$\phi_I$   & independent per-store operating cost; set so the summed model store count matches the data & """ + f"\\${phi_I/1000:,.0f}k/yr" + r""" \\
$c$        & sales a chain's new co-located store takes from each of its own town stores, as a fraction of a full outlet; set to the independent share of the border store increment & """ + f"{cannib_c:.2f}" + r""" \\
\midrule
\multicolumn{3}{@{}l}{\emph{Assumed}}\\
 & entry order: chains fill the demand-central slots, independents the rest & --- \\
 & storefronts per chain operator (used only to form an HHI) & """ + f"{CHAIN_SIZE:.0f}" + r""" \\
\bottomrule
\end{tabular}
\begin{minipage}{0.96\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} """ + f"{len(d)}" + r""" New Mexico town-markets (excluding the Albuquerque metro), mean
2024--2025 configuration. $S_m=\text{pop}_m+\lambda NL_m$; $r(N)=\kappa S_m/N$. $\phi_I$ is set by
bisection so the summed model store count equals the summed observed count (449). $c$ is chosen so the
model's border store increment (baseline minus the no-cross-border-demand counterfactual) has the same
independent share as the ZIP reduced form (""" + f"{INC_ISHARE_TARGET:.2f}" + r"""); with $r(N){=}\kappa
S/N$ this bounds chain stores per town at $1+1/c\approx""" + f"{1 + 1/cannib_c:.0f}" + r"""$. The chain
share of stores, independent counts, the per-type border response, and revenue per store are left free
and reported in the text as model fit.
\end{minipage}
\end{table}
""")

sh = cs.iloc[[0, 10, 20, 30, 40]]
w("ext_tab_comparative_statics.tex", r"""\begin{table}[htbp]
\centering
\caption{Model comparative statics: raising cross-border demand under free entry}
\label{tab:ext_compstat}
\footnotesize
\begin{tabular}{rrrrrrr}
\toprule
Nonlegal pop. & Effective mkt. & \multicolumn{3}{c}{Equilibrium stores} & Revenue & HHI \\
\cmidrule(lr){3-5}
within 75\,mi & size $S$ & total & chain & indep. & per store & (0--10{,}000)\\
\midrule
""" + "\n".join(
    f"{r.NL:,.0f} & {r.S:,.0f} & {r.N:.0f} & {r.nF:.0f} & {r.nI:.0f} & \\${r.r_store/1e3:,.0f}k & {r.hhi:,.0f} \\\\"
    for r in sh.itertuples()) + r"""
\bottomrule
\end{tabular}
\begin{minipage}{0.88\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Representative town, resident population """ + f"{REP_POP:,.0f}" + r""".
A cross-border demand increase expands the store count almost entirely through \emph{independent} entry,
leaves revenue per store nearly flat, and lowers concentration -- matching the reduced-form border
premium (Table~\ref{tab:nm_baseline}, Table~\ref{tab:ext_composition}).
\end{minipage}
\end{table}
""")

w("ext_tab_counterfactual.tex", r"""\begin{table}[htbp]
\centering
\caption{Counterfactuals on the 28 border town-markets}
\label{tab:ext_counterfactual}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Scenario & Total stores & Chain & Independent & Indep.\ share \\
\midrule
""" +
f"Baseline (free entry, cross-border demand) & {base_F+base_I:,.0f} & {base_F:,.0f} & {base_I:,.0f} & {base_I/(base_F+base_I):.2f} \\\\\n" +
f"(1) No cross-border demand & {nl0_F+nl0_I:,.0f} & {nl0_F:,.0f} & {nl0_I:,.0f} & {nl0_I/(nl0_F+nl0_I):.2f} \\\\\n" +
f"(2) Chains ignore cannibalization & {nc_F+nc_I:,.0f} & {nc_F:,.0f} & {nc_I:,.0f} & {nc_I/(nc_F+nc_I):.2f} \\\\\n" +
f"(3) License cap at interior-implied $N$ & {capF+capI:,.0f} & {capF:,.0f} & {capI:,.0f} & {capI/(capF+capI):.2f} \\\\\n" +
r"""\bottomrule
\end{tabular}
\begin{minipage}{0.93\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Equilibrium store counts summed over the 28 border town-markets under the calibration in
Table~\ref{tab:ext_calibration}. (1) sets the nonlegal-state population to zero: cross-border demand
accounts for """ + f"{extra:.0f}" + r""" of the """ + f"{base_F+base_I:.0f}" + r""" border stores, split
$+""" + f"{base_F-nl0_F:.0f}" + r"""$ chain and $+""" + f"{base_I-nl0_I:.0f}" + r"""$ independent. (2)
makes chains ignore the business-stealing externality at their own stores, so they behave like
independents: the border market becomes """ + f"{100*nc_F/(nc_F+nc_I):.0f}" + r"""\% chain rather than """
+ f"{100*base_F/(base_F+base_I):.0f}" + r"""\%. (3) caps each border town at the store count its resident
population alone would support: mean per-store revenue rises to """
+ f"{r_cap/r_free:.2f}" + r"""$\times$ its free-entry level, redirecting the cross-border
premium from entry to incumbent revenue -- the same direction as the New Mexico versus Colorado
comparison in Section~\ref{sec:discussion}.
\end{minipage}
\end{table}
""")

open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
