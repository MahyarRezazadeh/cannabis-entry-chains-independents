r"""
19_ownnetwork_borderonly_test.py -- structural_model / extension of venous4

Sharper version of the operator-level own-network cannibalization test (Sec.11.4's phi test),
requested as a follow-up: instead of pooling all 89 town-markets with a continuous demand index
D(B_m) (which includes many near-zero-B interior markets that dilute the contrast), restrict the
risk set to ONLY nonlegal-border ZIPs (treatment) vs. legal-border ZIPs (control) -- the same
captive-vs-placebo contrast that sharpened the composition-level test in 17_zip_composition_
controlled.py. True interior is dropped entirely here (this is not the same design as 17's level
regression, which needs interior as the reference; this one needs only two demand regimes to
form a clean treatment/control contrast for the INTERACTION term).

Theory being tested (same Proposition 1 as before): pi^C_jmt = kappa_C + beta_C*log(pop) +
[delta - rho*H_jmt]*D(B_m) - competition + eps. rho>0 implies an operator's existing network
overlap (H) should suppress its own entry probability MORE where cross-border demand is
genuinely captive (nonlegal-border) than where it isn't (legal-border) -- because cannibalizing
a real demand pool costs something; cannibalizing a market with no captive demand to begin with
doesn't. The sharp test is the sign of phi in:

    E[entry_jzt] = exp{ alpha_j + alpha_t + beta*log(zip_pop) + theta*H_jzt
                         + phi*(H_jzt x Nonlegal_z) }

restricted to candidate (brand, zip, quarter) risk-set rows where zip in {nonlegal-border,
legal-border} only, and the brand was NOT YET present in that zip as of t-1 (still eligible to
enter). Theory predicts phi < 0.

Data: reuses old1/build/panel_brand_zip_quarter.csv (already brand x zip x quarter, no
look-ahead, already used for the network-choice models earlier this session) and
old1/build/zip_xwalk.csv for the nonlegal-border/legal-border classification. Nothing here is
built fresh from raw storefront data; H (own-network overlap) reuses brand_stores_50mi, already
computed in panel_brand_zip_quarter.csv from each brand's own store coordinates at t-1.

Outputs:
  structural_model/output/tab_ownnetwork_borderonly.tex
  structural_model/output/tex_ownnetwork_borderonly_blurb.tex
"""
import os
import numpy as np, pandas as pd

BUILD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
SM = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SM, "output")


def stars(pv):
    return "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""


# =====================================================================================
# 1. classify ZIPs, restrict to nonlegal-border / legal-border only
# =====================================================================================
zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str})[["zip5", "border_z", "placebo_border_z"]]
zx.loc[zx["border_z"] == 1, "placebo_border_z"] = 0
zx["group"] = np.select([zx["border_z"] == 1, zx["placebo_border_z"] == 1],
                         ["nonlegal_border", "legal_border"], default="interior")
keep = zx[zx["group"] != "interior"][["zip5", "group"]]

bzq = pd.read_csv(os.path.join(BUILD, "panel_brand_zip_quarter.csv"), dtype={"zip5": str})
bzq = bzq.merge(keep, on="zip5", how="inner")   # drop interior ZIPs entirely
print(f"ZIPs kept: {bzq.zip5.nunique()} ({keep.group.value_counts().to_dict()} in the full ZIP universe; "
      f"only those with a dispensary appear in panel_brand_zip_quarter.csv)")

# =====================================================================================
# 2. risk set: candidate rows where the brand was NOT YET present in that ZIP at t-1
#    (present_prevq==0), restricted to established operators (brand_in_nm_prevq==1, i.e. the
#    operator already has an active NM network elsewhere -- same population as the original
#    Type-C mechanism test; a first-time entrant has no network to cannibalize by construction)
# =====================================================================================
pres_prev = bzq[["brand_id", "zip5", "qt", "present"]].rename(
    columns={"qt": "qt_next", "present": "present_prevq"})
pres_prev["qt"] = pres_prev["qt_next"] + 1
pres_prev = pres_prev.drop(columns="qt_next")

risk = bzq.merge(pres_prev, on=["brand_id", "zip5", "qt"], how="left")
risk["present_prevq"] = risk["present_prevq"].fillna(0)
risk = risk[(risk["present_prevq"] == 0) & (risk["brand_in_nm_prevq"] == 1)].copy()
print(f"risk-set rows (established-operator candidates, not yet present): {len(risk)}")
print(f"entries captured in this risk set: {int(risk['entry'].sum())} "
      f"(of which expansion_entry: {int(risk['expansion_entry'].sum())})")

risk["nonlegal"] = (risk["group"] == "nonlegal_border").astype(float)
risk["H"] = risk["brand_stores_50mi"].fillna(0)
risk["H_x_nonlegal"] = risk["H"] * risk["nonlegal"]
risk["log_zip_pop"] = np.log(risk["zip_pop"].clip(lower=1))

n_brands = risk.brand_id.nunique()
print(f"distinct established operators in this risk set: {n_brands}")
print(risk.groupby("group")["entry"].agg(["sum", "count", "mean"]))

# =====================================================================================
# 3. sparse two-way FE Poisson IRLS (reuse the same hand-rolled estimator as 11_mechanism_
#    regression.py / _ppml_fe.py -- this panel is far smaller, but the same estimator is used
#    for consistency and because statsmodels' dense GLM does not scale to many FE levels)
# =====================================================================================
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def ppml_fe(df, y, xvars, fe_vars, cluster_var):
    d = df.dropna(subset=[y] + xvars + fe_vars + [cluster_var]).copy()
    n = len(d)
    X = d[xvars].to_numpy(float)
    fe_blocks = []
    fe_names = []
    for fv in fe_vars:
        cats = pd.Categorical(d[fv])
        n_levels = len(cats.categories)
        rows_idx = np.arange(n)
        cols_idx = cats.codes
        # drop the first level of each FE to avoid the dummy trap (intercept absorbs it)
        mask = cols_idx > 0
        block = sp.csr_matrix((np.ones(mask.sum()), (rows_idx[mask], cols_idx[mask] - 1)), shape=(n, n_levels - 1))
        fe_blocks.append(block)
        fe_names.extend([f"{fv}={c}" for c in cats.categories[1:]])
    Xd = sp.csr_matrix(np.column_stack([np.ones(n), X]))
    Xfull = sp.hstack([Xd] + fe_blocks).tocsr()
    K = Xfull.shape[1]
    beta = np.zeros(K)
    y_arr = d[y].to_numpy(float)
    for it in range(50):
        eta = Xfull @ beta
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        W = mu
        z = eta + (y_arr - mu) / np.maximum(mu, 1e-10)
        Wd = sp.diags(W)
        XtWX = (Xfull.T @ Wd @ Xfull).tocsc()
        XtWz = Xfull.T @ (W * z)
        XtWX = XtWX + sp.eye(K) * 1e-8
        beta_new = spsolve(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    # cluster-robust sandwich SE
    eta = np.clip(Xfull @ beta, -30, 30)
    mu = np.exp(eta)
    W = mu
    Wd = sp.diags(W)
    XtWX = (Xfull.T @ Wd @ Xfull).tocsc() + sp.eye(K) * 1e-8
    Ainv = sp.linalg.inv(XtWX) if K < 2000 else None
    score = Xfull.multiply((y_arr - mu)[:, None]).tocsr()
    clusters = d[cluster_var].values
    meat = np.zeros((K, K))
    for c in pd.unique(clusters):
        idx = clusters == c
        s_c = np.asarray(score[idx].sum(axis=0)).flatten()
        meat += np.outer(s_c, s_c)
    Ainv_dense = Ainv.toarray() if Ainv is not None else np.linalg.inv(XtWX.toarray())
    V = Ainv_dense @ meat @ Ainv_dense
    se = np.sqrt(np.diag(V))
    names = ["const"] + xvars + fe_names
    return dict(beta=beta[:len(xvars) + 1], se=se[:len(xvars) + 1], names=["const"] + xvars,
                n=n, n_clusters=d[cluster_var].nunique())


xvars = ["log_zip_pop", "H", "nonlegal", "H_x_nonlegal"]
res = ppml_fe(risk, "entry", xvars, fe_vars=["brand_id", "qt"], cluster_var="brand_id")

from scipy.stats import norm
print("\n" + "=" * 90)
print("PPML, operator + quarter FE, cluster-robust SE by operator, ZIP restricted to border-only")
print("=" * 90)
rows_out = []
for nm, b, s in zip(res["names"], res["beta"], res["se"]):
    z = b / s if s > 0 else np.nan
    pv = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    print(f"  {nm:16s} {b:+.4f}{stars(pv):<3} ({s:.4f})  p={pv:.4f}")
    rows_out.append(dict(var=nm, coef=b, se=s, p=pv))
coef_df = pd.DataFrame(rows_out).set_index("var")
print(f"\nN = {res['n']}, operators = {res['n_clusters']}")

phi = coef_df.loc["H_x_nonlegal"]
print(f"\nphi (H x Nonlegal) = {phi['coef']:+.4f}  (SE {phi['se']:.4f}, p={phi['p']:.4f})  "
      f"-- theory predicts phi < 0")
print("VERDICT: " + ("CONSISTENT with own-network cannibalization (phi<0, and " +
      ("significant)" if phi['p'] < 0.10 else "but not significant)")
      if phi['coef'] < 0 else
      "NOT consistent with own-network cannibalization (phi>=0, wrong sign)"))

# =====================================================================================
# outputs
# =====================================================================================
tab = r"\begin{tabular}{@{}lc@{}}" + "\n" + r"\toprule" + "\n"
tab += "Covariate & Coefficient \\\\\n" + r"\midrule" + "\n"
labels = {"log_zip_pop": r"$\log(\text{ZIP pop.})$", "H": r"$H$ (own stores within 50mi, $t-1$)",
          "nonlegal": r"Nonlegal-border", "H_x_nonlegal": r"$H \times$ Nonlegal-border"}
for nm in xvars:
    r = coef_df.loc[nm]
    tab += f"{labels[nm]} & {r['coef']:+.4f}{stars(r['p'])} \\\\\n & ({r['se']:.4f}) \\\\\n"
tab += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(OUT, "tab_ownnetwork_borderonly.tex"), "w", encoding="utf-8").write(tab)

def psig(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"

blurb = (
    f"Restricting the operator-level risk set to nonlegal-border vs.\\ legal-border ZIPs only "
    f"(interior dropped; {res['n']} established-operator candidate rows, {res['n_clusters']} "
    f"operators, operator+quarter FE PPML, cluster-robust SE by operator) sharpens the demand "
    f"contrast relative to the original pooled test (\\S\\ref{{sec:mechreg}}), which used a "
    f"continuous demand index across all 89 town-markets including many near-zero-demand interior "
    f"markets. The theoretically central coefficient is "
    f"$\\hat\\varphi_{{H\\times\\text{{Nonlegal}}}}={phi['coef']:+.4f}$ (SE {phi['se']:.4f}, "
    f"$p{psig(phi['p'])}$), predicted negative by own-network internalization. "
    + ("This IS the predicted sign" if phi['coef'] < 0 else "This is NOT the predicted sign")
    + (", and statistically significant" if phi['p'] < 0.10 else ", but not statistically significant")
    + ".")
open(os.path.join(OUT, "tex_ownnetwork_borderonly_blurb.tex"), "w", encoding="utf-8").write(blurb)
print("\nwrote tab_ownnetwork_borderonly.tex, tex_ownnetwork_borderonly_blurb.tex")
