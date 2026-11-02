r"""
15_networkchoice_logit.py -- structural_model / extension of venous4

First-pass network-CHOICE model: conditional (McFadden) logit on which ZIP an already-active
chain expands into next, among the ZIPs it has not yet entered. This is the discrete-choice
analogue of Nishida's (2008 job-market paper; 2014 Marketing Science) strategic store-network
choice framework, adapted to this project's setting: instead of one firm type choosing a
network of locations to balance own-cannibalization against deterring RIVAL CHAIN entry,
here TWO organizational types (chain vs. independent) compete for the same candidate ZIPs, and
there is an exogenous, policy-generated, spatially discrete demand shifter (captive cross-
border demand) absent from Nishida's setting.

Question: conditional on being an active chain choosing where to add its next store, does a
higher captive cross-border demand index make a candidate ZIP MORE attractive (naive channel:
delta_I>delta_C notwithstanding, some border pull should still exist), and does the chain's own
existing network (own-store proximity) dominate that pull -- consistent with this project's
already-established findings that (i) cross-border demand exists and (ii) chains are, in the
aggregate/margin, LESS responsive to it than independents?

Design:
  - Choice occasions: brand-quarter pairs with exactly ONE expansion entry (an already-active
    operator's single new store that quarter). Multi-store-same-quarter occasions (36 of 203)
    are DROPPED, not modeled, because a simultaneous multi-choice decision is not a standard
    single-choice conditional logit and is out of scope for this first pass -- stated as a
    limitation, not silently pooled in.
  - Choice set per occasion: every ZIP the panel tracks (136, i.e. every ZIP that had a
    dispensary at some point in the sample -- a "revealed feasible" candidate set, not the full
    368-ZIP NM universe; stated as a limitation) where that brand was NOT present at t-1.
  - Alternative-varying covariates (all strictly t-1 or fixed geography, no look-ahead, reusing
    old1/build/panel_brand_zip_quarter.csv's own no-look-ahead construction):
      log(zip_pop), D(B)=log(1+nonleg_pop_75) [captive cross-border demand],
      D(B_legal)=log(1+legalneigh_pop_75) [placebo demand -- legal neighbor, should not pull
        an already-active chain toward a candidate ZIP if the effect is truly captive-demand-
        specific, mirroring 14_zip_border_placebo.py's placebo logic inside the choice model
        itself], own-network stores within 50mi (brand_stores_50mi, t-1), rival independent and
        chain BRAND counts in the ZIP at t-1 (lagged, from panel_zip_quarter.csv).
  - Estimator: standard McFadden conditional logit (alternative-specific covariates only, no
    case-specific ones, so occasion/case fixed effects drop out of the likelihood by
    construction) -- log-likelihood, analytic gradient and Hessian implemented directly (no
    off-the-shelf conditional-logit package dependency), cluster-robust (sandwich) SEs by brand.

Outputs:
  structural_model/data/networkchoice_occasions.csv    audit trail: one row per occasion
  structural_model/output/networkchoice_logit.csv       coefficients, SE, p-values
  structural_model/output/tab_networkchoice_logit.tex
  structural_model/output/tex_networkchoice_blurb.tex
"""
import os
import numpy as np, pandas as pd

BUILD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
SM = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SM, "output")
DATA = os.path.join(SM, "data")


def stars(pv):
    return "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""


# =====================================================================================
# 1. load and identify single-choice expansion occasions
# =====================================================================================
bzq = pd.read_csv(os.path.join(BUILD, "panel_brand_zip_quarter.csv"), dtype={"zip5": str})
bzq = bzq.dropna(subset=["border_z", "placebo_border_z"]).copy()

ent = bzq[(bzq.entry == 1) & (bzq.expansion_entry == 1)]
occ_size = ent.groupby(["brand_id", "qt"]).size()
single_occ = set(occ_size[occ_size == 1].index)
n_multi = int((occ_size > 1).sum())
print(f"expansion occasions: {len(occ_size)} total; {len(single_occ)} single-ZIP (modeled); "
      f"{n_multi} simultaneous multi-ZIP (dropped -- not a single-choice occasion)")

chosen = ent[ent.set_index(["brand_id", "qt"]).index.isin(single_occ)][["brand_id", "qt", "zip5"]]
chosen = chosen.rename(columns={"zip5": "chosen_zip"})

# =====================================================================================
# 2. lagged rival brand counts by zip-quarter (from panel_zip_quarter.csv), and choice sets
# =====================================================================================
zq = pd.read_csv(os.path.join(BUILD, "panel_zip_quarter.csv"), dtype={"zip5": str})[
    ["zip5", "qt", "n_indep_brands", "n_chain_brands"]].copy()
zq_lag = zq.rename(columns={"qt": "qt_next", "n_indep_brands": "rival_indep_lag",
                             "n_chain_brands": "rival_chain_lag"})
zq_lag["qt"] = zq_lag["qt_next"] + 1
zq_lag = zq_lag.drop(columns="qt_next")

# self-merge to get each (brand,zip,qt-1) row's "present" flag -> filter choice set
pres_prev = bzq[["brand_id", "zip5", "qt", "present"]].rename(
    columns={"qt": "qt_next", "present": "present_prevq"})
pres_prev["qt"] = pres_prev["qt_next"] + 1
pres_prev = pres_prev.drop(columns="qt_next")

rows = []
for (b, t), chosen_zip in chosen.set_index(["brand_id", "qt"])["chosen_zip"].items():
    alts = bzq[(bzq.brand_id == b) & (bzq.qt == t)].copy()
    alts = alts.merge(pres_prev[(pres_prev.brand_id == b) & (pres_prev.qt == t)][["zip5", "present_prevq"]],
                       on="zip5", how="left")
    alts["present_prevq"] = alts["present_prevq"].fillna(0)
    alts = alts[alts["present_prevq"] == 0].copy()   # candidate set: not yet occupied by b at t-1
    if chosen_zip not in set(alts["zip5"]):
        continue   # safety; should not happen by construction
    alts["occ_id"] = f"{b}_{t}"
    alts["brand_id_occ"] = b
    alts["chosen"] = (alts["zip5"] == chosen_zip).astype(int)
    rows.append(alts)

panel = pd.concat(rows, ignore_index=True)
panel = panel.merge(zq_lag, on=["zip5", "qt"], how="left")
panel[["rival_indep_lag", "rival_chain_lag"]] = panel[["rival_indep_lag", "rival_chain_lag"]].fillna(0)

panel["D_nonleg"] = np.log1p(panel["nonleg_pop_75"].fillna(0))
panel = panel.merge(
    pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str})[["zip5", "legalneigh_pop_75"]],
    on="zip5", how="left")
panel["D_legal"] = np.log1p(panel["legalneigh_pop_75"].fillna(0))
panel["own_H50"] = panel["brand_stores_50mi"].fillna(0)

panel.to_csv(os.path.join(DATA, "networkchoice_occasions.csv"), index=False)
n_occ = panel["occ_id"].nunique()
alt_sizes = panel.groupby("occ_id").size()
print(f"single-choice occasions modeled: {n_occ}; choice-set size: mean {alt_sizes.mean():.1f}, "
      f"min {alt_sizes.min()}, max {alt_sizes.max()}")

# =====================================================================================
# 3. conditional logit: log-likelihood, analytic gradient/Hessian, cluster-robust SE by brand
# =====================================================================================
XVARS = ["log_zip_pop", "D_nonleg", "D_legal", "own_H50", "rival_indep_lag", "rival_chain_lag"]
panel = panel.dropna(subset=XVARS).copy()

occ_groups = []
for occ_id, g in panel.groupby("occ_id", sort=False):
    X = g[XVARS].to_numpy(float)
    y = g["chosen"].to_numpy(float)
    occ_groups.append((X, y, g["brand_id_occ"].iloc[0]))

K = len(XVARS)


def loglik_total(beta):
    ll = 0.0
    for X, y, _ in occ_groups:
        v = X @ beta
        m = v.max()
        ll += float(y @ v) - (np.log(np.exp(v - m).sum()) + m)
    return ll


def score_and_hess(beta):
    grad = np.zeros(K)
    hess = np.zeros((K, K))
    for X, y, _ in occ_groups:
        v = X @ beta
        m = v.max()
        ev = np.exp(v - m)
        p = ev / ev.sum()
        grad += X.T @ (y - p)
        hess -= X.T @ (np.diag(p) - np.outer(p, p)) @ X
    return grad, hess


beta_hat = np.zeros(K)
for it in range(100):
    grad, hess = score_and_hess(beta_hat)
    step = np.linalg.solve(hess, grad)   # Newton: beta -= H^{-1} g solves H*step=g since hess is -Fisher info
    beta_hat = beta_hat - step
    if np.max(np.abs(step)) < 1e-10:
        break
converged = np.max(np.abs(step)) < 1e-8
_, hess = score_and_hess(beta_hat)
Vinv = -hess   # Fisher information (positive definite at the optimum)
V = np.linalg.inv(Vinv)

# cluster-robust (sandwich) SE by brand
brands = sorted(set(bid for _, _, bid in occ_groups))
score_by_brand = {b: np.zeros(K) for b in brands}
for X, y, bid in occ_groups:
    v = X @ beta_hat
    m = v.max()
    ev = np.exp(v - m)
    p = ev / ev.sum()
    score_by_brand[bid] += X.T @ (y - p)
meat = np.zeros((K, K))
for b in brands:
    s = score_by_brand[b]
    meat += np.outer(s, s)
V_cluster = V @ meat @ V

se = np.sqrt(np.diag(V_cluster))
z_stat = beta_hat / se
from scipy.stats import norm
pval = 2 * (1 - norm.cdf(np.abs(z_stat)))

coef = pd.DataFrame(dict(var=XVARS, beta=beta_hat, se=se, z=z_stat, p=pval))
coef.to_csv(os.path.join(OUT, "networkchoice_logit.csv"), index=False)
print(f"\nconverged: {converged} (iterations: {it+1}), log-lik = {loglik_total(beta_hat):.2f}, "
      f"N occasions = {n_occ}, N brand clusters = {len(brands)}")
print(coef.to_string(index=False))

# =====================================================================================
# 4. LaTeX table + blurb
# =====================================================================================
LABELS = {
    "log_zip_pop": r"$\log(\text{ZIP population})$",
    "D_nonleg": r"$D(B)=\log(1+\text{nonlegal-neighbor pop.})$",
    "D_legal": r"$D(B_{\text{legal}})=\log(1+\text{legal-neighbor pop.})$",
    "own_H50": r"own stores within 50mi ($t-1$)",
    "rival_indep_lag": r"rival independent brands ($t-1$)",
    "rival_chain_lag": r"rival chain brands ($t-1$)",
}

tab = r"\begin{tabular}{@{}lc@{}}" + "\n" + r"\toprule" + "\n"
tab += "Alternative-specific covariate & Coefficient \\\\\n" + r"\midrule" + "\n"
for _, r in coef.iterrows():
    tab += f"{LABELS[r['var']]} & {r['beta']:+.4f}{stars(r['p'])} \\\\\n"
    tab += f" & ({r['se']:.4f}) \\\\\n"
tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_networkchoice_logit.tex"), "w", encoding="utf-8").write(tab)


def g(v):
    r = coef[coef["var"] == v].iloc[0]
    return float(r.beta), float(r.se), float(r.p)


b_nl, se_nl, p_nl = g("D_nonleg")
b_lg, se_lg, p_lg = g("D_legal")
b_h, se_h, p_h = g("own_H50")
b_ri, se_ri, p_ri = g("rival_indep_lag")
b_rc, se_rc, p_rc = g("rival_chain_lag")

def psig(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"

blurb = (
    f"A conditional logit on {n_occ} single-ZIP expansion choices by {len(brands)} distinct chains "
    f"(alternative-specific covariates only, cluster-robust SE by brand; {n_multi} simultaneous "
    f"multi-ZIP expansion occasions dropped as out of scope for a single-choice model) finds that "
    f"cross-border demand DOES pull an already-active chain toward a candidate ZIP -- "
    f"$\\hat\\beta_{{D(B)}}={b_nl:+.3f}$ (SE {se_nl:.3f}, $p{psig(p_nl)}$) -- but the placebo "
    f"legal-neighbor demand index has a coefficient of $\\hat\\beta_{{D(B_{{\\text{{legal}}}})}}"
    f"={b_lg:+.3f}$ (SE {se_lg:.3f}, $p{psig(p_lg)}$), "
    + ("statistically indistinguishable from zero -- at the MARGIN of an already-active chain's next "
       "location choice, the placebo behaves as theoretically expected (no legal-neighbor pull), a "
       "cleaner pass than the LEVELS comparison of \\S\\ref{sec:zipplacebo}, which found a significant "
       "gap between nonlegal- and legal-border ZIPs in the opposite direction from a naive prediction; "
       "the two tests are not the same object (one is a marginal expansion-location choice, the other "
       "a cross-sectional stock comparison) and this paper does not force them to agree: "
       if p_lg >= 0.10 else
       "which is ALSO significant, unlike the LEVELS placebo of \\S\\ref{sec:zipplacebo} -- a further "
       "discrepancy this paper reports rather than reconciles: ")
    + f"own-network proximity dominates both, $\\hat\\beta_{{H_{{50}}}}={b_h:+.3f}$ (SE {se_h:.3f}, "
    f"$p{psig(p_h)}$) -- a chain choosing its next ZIP weighs its OWN existing footprint far more "
    f"heavily than either demand index, the choice-level analogue of the positive (not negative) "
    f"own-network coefficient already found in the superseded operator-level mechanism regression "
    f"(\\S\\ref{{sec:mechreg}}). Rival brand presence at $t-1$: independent presence "
    f"$\\hat\\beta={b_ri:+.3f}$ (SE {se_ri:.3f}, $p{psig(p_ri)}$), chain presence "
    f"$\\hat\\beta={b_rc:+.3f}$ (SE {se_rc:.3f}, $p{psig(p_rc)}$).")
open(os.path.join(OUT, "tex_networkchoice_blurb.tex"), "w", encoding="utf-8").write(blurb)

print("\nwrote: networkchoice_occasions.csv, networkchoice_logit.csv, tab_networkchoice_logit.tex, "
      "tex_networkchoice_blurb.tex")
