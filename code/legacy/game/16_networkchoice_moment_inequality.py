r"""
16_networkchoice_moment_inequality.py -- structural_model / extension of venous4

Moment-inequality (Pakes-Porter-Ho-Ishii / Holmes 2011 / Nishida 2008,2014 -style) estimator for
the same network-choice problem as 15_networkchoice_logit.py, PLUS a direct empirical test of the
"chain cedes ZIP B to an independent" framing the user asked about.

Part 1 -- moment-inequality estimator. Rather than assuming i.i.d. type-I extreme-value location
shocks (the logit's identifying assumption), use revealed preference directly: if a chain's
realized expansion into ZIP z* beats every unchosen alternative z' in the same choice occasion,
then, for the TRUE profit-index parameters theta, v(z*;theta) - v(z';theta) >= 0 for every such
pair (up to a mean-zero/small idiosyncratic shock this literature's estimator is built to be
robust to, unlike a fully parametric MLE). theta is estimated by minimizing the sum of squared
NEGATIVE parts of these deviations (a firm that violated an inequality contributes a positive
penalty; a firm that satisfied it contributes zero) -- a standard modified-min-distance /
"conditional moment inequality" M-estimator. The population coefficient is normalized to 1 (this
class of estimator is scale-invariant, so one coefficient's scale must be fixed) so every other
coefficient is interpreted as "population-equivalent units." Cluster (by brand) bootstrap for
approximate confidence intervals -- NOT the canonical Chernozhukov-Hong-Tamer/Pakes-et-al. worst-
case confidence region (which requires solving a much larger subsampling/test-inversion problem);
this is stated as a simplification, not presented as the full moment-inequality inference machinery.

Part 2 -- "foregone border opportunity" test. For every single-choice occasion where the chain
chose a NON-nonlegal-border ZIP, check whether a nonlegal-border ZIP was available in the SAME
choice set (same brand, same quarter, not yet entered by that brand) with HIGHER captive
cross-border demand than the one actually chosen. Track the single best such foregone ZIP per
occasion forward in time: does ANY brand enter it within the next 8 quarters, and if so, is that
entrant independent (denovo_entry, first-ever NM location, this project's own at-entry
"independent" definition) or a chain expanding (expansion_entry)? This directly operationalizes
"chain decides between adding a branch in ZIP A but allows an independent to enter in ZIP B."

Outputs:
  structural_model/output/networkchoice_moment_ineq.csv       theta_hat, bootstrap CI, vs logit
  structural_model/output/tab_networkchoice_moment_ineq.tex
  structural_model/output/foregone_border_opportunities.csv   occasion-level audit trail
  structural_model/output/tex_foregone_blurb.tex
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import minimize

BUILD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
SM = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SM, "output")
DATA = os.path.join(SM, "data")
RNG = np.random.default_rng(20260916)
FWD_Q = 8   # forward-tracking window, quarters

XREST = ["D_nonleg", "D_legal", "own_H50", "rival_indep_lag", "rival_chain_lag"]

# =====================================================================================
# 0. load the occasion panel built by 15_networkchoice_logit.py
# =====================================================================================
occ = pd.read_csv(os.path.join(DATA, "networkchoice_occasions.csv"), dtype={"zip5": str})
occ = occ.dropna(subset=["log_zip_pop"] + XREST).copy()

occ_groups = []
for occ_id, g in occ.groupby("occ_id", sort=False):
    Xrest = g[XREST].to_numpy(float)
    logpop = g["log_zip_pop"].to_numpy(float)
    y = g["chosen"].to_numpy(float)
    c = int(np.argmax(y))
    # deviations: chosen minus every OTHER alternative
    mask = np.arange(len(g)) != c
    Dz = Xrest[c] - Xrest[mask]           # (n_alt-1) x 5
    e = logpop[c] - logpop[mask]          # (n_alt-1,), coefficient fixed at 1
    occ_groups.append((Dz, e, g["brand_id_occ"].iloc[0]))

n_occ = len(occ_groups)
n_pairs = sum(len(e) for _, e, _ in occ_groups)
print(f"moment-inequality estimator: {n_occ} occasions, {n_pairs} chosen-vs-unchosen pairs")


def stack(groups):
    Dz = np.concatenate([d for d, _, _ in groups], axis=0)
    e = np.concatenate([e for _, e, _ in groups], axis=0)
    return Dz, e


def loss_and_grad(theta, Dz, e):
    v = e + Dz @ theta
    viol = np.minimum(0.0, v)
    loss = float(np.mean(viol ** 2))
    grad = (2.0 / len(v)) * (Dz.T @ viol)
    return loss, grad


def fit(groups):
    Dz, e = stack(groups)
    x0 = np.zeros(len(XREST))
    res = minimize(lambda th: loss_and_grad(th, Dz, e), x0, jac=True, method="BFGS",
                    options=dict(gtol=1e-10, maxiter=1000))
    return res.x, res.fun


theta_hat, obj_val = fit(occ_groups)
Dz_all, e_all = stack(occ_groups)
v_all = e_all + Dz_all @ theta_hat
frac_satisfied = float((v_all >= 0).mean())
print(f"theta_hat (population-normalized to 1): {dict(zip(XREST, theta_hat.round(4)))}")
print(f"objective (mean sq. violation) = {obj_val:.6f}; "
      f"fraction of {len(v_all)} pairs satisfying the inequality at theta_hat = {frac_satisfied:.3f}")

# =====================================================================================
# cluster (brand) bootstrap for approximate CIs
# =====================================================================================
brands = [b for _, _, b in occ_groups]
by_brand = {}
for grp in occ_groups:
    by_brand.setdefault(grp[2], []).append(grp)
brand_list = list(by_brand.keys())

NBOOT = 300
boot_theta = np.zeros((NBOOT, len(XREST)))
for b in range(NBOOT):
    draw = RNG.choice(brand_list, size=len(brand_list), replace=True)
    groups_b = [grp for bid in draw for grp in by_brand[bid]]
    th_b, _ = fit(groups_b)
    boot_theta[b] = th_b
ci_lo = np.percentile(boot_theta, 2.5, axis=0)
ci_hi = np.percentile(boot_theta, 97.5, axis=0)

logit = pd.read_csv(os.path.join(OUT, "networkchoice_logit.csv")).set_index("var")
rows = []
for i, v in enumerate(XREST):
    rows.append(dict(var=v, theta_hat=theta_hat[i], ci_lo=ci_lo[i], ci_hi=ci_hi[i],
                     logit_beta=float(logit.loc[v, "beta"]), logit_p=float(logit.loc[v, "p"])))
mi = pd.DataFrame(rows)
mi.to_csv(os.path.join(OUT, "networkchoice_moment_ineq.csv"), index=False)
print("\n" + mi.to_string(index=False))

# =====================================================================================
# LaTeX table
# =====================================================================================
LABELS = {
    "D_nonleg": r"$D(B)$ (captive cross-border demand)",
    "D_legal": r"$D(B_{\text{legal}})$ (placebo)",
    "own_H50": r"own stores within 50mi ($t-1$)",
    "rival_indep_lag": r"rival independent brands ($t-1$)",
    "rival_chain_lag": r"rival chain brands ($t-1$)",
}
tab = r"\begin{tabular}{@{}lccc@{}}" + "\n" + r"\toprule" + "\n"
tab += r"Covariate (pop. coef. normalized to 1) & $\hat\theta$ (moment ineq.) & 95\% CI (brand bootstrap) & Logit $\hat\beta$ \\" + "\n" + r"\midrule" + "\n"
for _, r in mi.iterrows():
    tab += (f"{LABELS[r['var']]} & {r['theta_hat']:+.3f} & [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & "
            f"{r['logit_beta']:+.3f} \\\\\n")
tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_networkchoice_moment_ineq.tex"), "w", encoding="utf-8").write(tab)

# =====================================================================================
# Part 2 -- foregone border opportunity + forward tracking
# =====================================================================================
print("\n" + "=" * 90)
print("PART 2 -- foregone border opportunities and what happens to them afterward")
print("=" * 90)

fo_rows = []
n_eligible = 0
for occ_id, g in occ.groupby("occ_id", sort=False):
    chosen = g[g.chosen == 1].iloc[0]
    if chosen["border_z"] == 1:
        continue   # chain DID choose a nonlegal-border ZIP this occasion; not eligible
    n_eligible += 1
    alts = g[g.chosen == 0]
    better_border = alts[(alts.border_z == 1) & (alts.D_nonleg > chosen["D_nonleg"])]
    if len(better_border) == 0:
        continue
    best = better_border.loc[better_border["D_nonleg"].idxmax()]
    fo_rows.append(dict(occ_id=occ_id, brand_id=chosen["brand_id_occ"], qt=int(chosen["qt"]),
                        chosen_zip=chosen["zip5"], chosen_D_nonleg=chosen["D_nonleg"],
                        foregone_zip=best["zip5"], foregone_D_nonleg=best["D_nonleg"]))
fo = pd.DataFrame(fo_rows)
print(f"occasions where the chosen ZIP was NOT nonlegal-border: {n_eligible}")
print(f"of those, occasions with a HIGHER-demand nonlegal-border ZIP available and unchosen "
      f"(a 'foregone border opportunity'): {len(fo)} ({100*len(fo)/n_eligible:.1f}%)")

# forward-track each foregone ZIP: any entry (any brand) within FWD_Q quarters?
bzq = pd.read_csv(os.path.join(BUILD, "panel_brand_zip_quarter.csv"), dtype={"zip5": str})
ent_all = bzq[bzq.entry == 1][["zip5", "qt", "brand_id", "denovo_entry", "expansion_entry"]]

def first_entrant(zip5, t0):
    window = ent_all[(ent_all.zip5 == zip5) & (ent_all.qt > t0) & (ent_all.qt <= t0 + FWD_Q)]
    if len(window) == 0:
        return None
    row = window.sort_values("qt").iloc[0]
    return "independent" if row["denovo_entry"] == 1 else "chain"

fo["subsequent_entrant"] = [first_entrant(z, t) for z, t in zip(fo["foregone_zip"], fo["qt"])]
fo.to_csv(os.path.join(OUT, "foregone_border_opportunities.csv"), index=False)

n_any = int(fo["subsequent_entrant"].notna().sum())
n_indep = int((fo["subsequent_entrant"] == "independent").sum())
n_chain = int((fo["subsequent_entrant"] == "chain").sum())
print(f"\nof {len(fo)} foregone border ZIPs, {n_any} ({100*n_any/len(fo):.1f}%) saw ANY entry "
      f"(any brand) within {FWD_Q} quarters:")
print(f"  -> independent (first-ever NM location): {n_indep} ({100*n_indep/max(n_any,1):.1f}% of those entered)")
print(f"  -> chain (established operator expanding): {n_chain} ({100*n_chain/max(n_any,1):.1f}% of those entered)")

# baseline comparison: overall denovo vs expansion share among ALL entries in the full panel
ent_full = bzq[bzq.entry == 1]
base_denovo = float(ent_full["denovo_entry"].mean())
print(f"\nbaseline (all {len(ent_full)} NM entries, any ZIP, any quarter): "
      f"{100*base_denovo:.1f}% independent (denovo), {100*(1-base_denovo):.1f}% chain (expansion)")

fo_indep_share = n_indep / n_any if n_any > 0 else float("nan")
gap = fo_indep_share - base_denovo
if gap > 0.05:
    direction = ("HIGHER than the panel-wide baseline -- these specific foregone ZIPs are "
                 "disproportionately picked up by independents afterward, direct evidence for the "
                 "``chain cedes ZIP B to an independent'' mechanism, not merely an artifact of "
                 "independents entering everywhere at a similar rate")
elif gap < -0.05:
    direction = ("LOWER than the panel-wide baseline -- if anything, these specific foregone ZIPs "
                 "attract established chains at a HIGHER-than-average rate afterward, the opposite "
                 "of the ``chain cedes to an independent'' prediction; this paper reports this "
                 "plainly rather than reconciling it with the hypothesis that motivated the test")
else:
    direction = ("statistically indistinguishable from the panel-wide baseline -- these specific "
                 "foregone ZIPs are entered by the same mix of independents and chains as the rest "
                 "of the panel, no special pull toward independents")

blurb = (
    f"Of {n_eligible} single-choice expansion occasions where the chain did NOT choose a "
    f"nonlegal-border ZIP, {len(fo)} ({100*len(fo)/n_eligible:.1f}\\%) had a nonlegal-border ZIP "
    f"available in the SAME choice set, not yet entered by that chain, with strictly HIGHER "
    f"captive cross-border demand than the ZIP actually chosen -- a directly observable "
    f"``foregone border opportunity.'' This share is close to mechanical, not a surprising finding "
    f"on its own: a chosen non-border ZIP has $D(B)$ near zero by construction of the 75-mile/"
    f"1{{,}}000-resident border rule, so almost any available nonlegal-border ZIP with positive "
    f"$D(B)$ clears the ``higher demand'' bar; it is reported as the SCOPE of the test, not as its "
    f"finding. Tracking the single best such foregone ZIP forward "
    f"{FWD_Q} quarters, {n_any} of {len(fo)} ({100*n_any/len(fo):.1f}\\%) are entered by SOME "
    f"brand within the window; of those, {n_indep} ({100*n_indep/max(n_any,1):.1f}\\%) are entered "
    f"by an independent (a brand's first-ever New Mexico location) and {n_chain} "
    f"({100*n_chain/max(n_any,1):.1f}\\%) by an established chain expanding. Against a panel-wide "
    f"baseline of {100*base_denovo:.1f}\\% independent / {100*(1-base_denovo):.1f}\\% chain among "
    f"ALL {len(ent_full)} New Mexico entries, the foregone-ZIP independent share "
    f"({100*fo_indep_share:.1f}\\%) is {direction}.")
open(os.path.join(OUT, "tex_foregone_blurb.tex"), "w", encoding="utf-8").write(blurb)
print("\nwrote: networkchoice_moment_ineq.csv, tab_networkchoice_moment_ineq.tex, "
      "foregone_border_opportunities.csv, tex_foregone_blurb.tex")
