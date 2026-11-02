"""
ext06_eventstudy_count_diff.py  --  extension of venous4

Event-study design: the (independent - chain) store-COUNT differential over market age
(quarters since a town's first dispensary), border vs interior.

This complements ext05's chain-SHARE transition (a descriptive ratio) with a formal
event-study regression on the LEVEL differential N_indep - N_chain, town fixed effects,
event-time dummies interacted with border, clustered SE by town.

Same universe/definitions as ext05 (re-derived here, not re-used, to keep this script
self-contained): NM CROP census, town-markets ex-Albuquerque-metro, chain = operator with
>=2 NM storefronts lifetime (brand_master.csv), border = border_sf (town-level border flag
carried on the storefront record), market age evt = qt - (town's first active quarter).

Outputs (project root):
  ext_tab_eventstudy_count_diff.tex   border-interior gap in (N_indep-N_chain) by market age
  ext_fig_eventstudy_count_diff.png   (A) raw group means, data; (B) event-study regression coef.
  ext_output/ext06_eventstudy_count_diff.log
  ext_output/ext_eventstudy_count_diff.csv   (town-quarter panel: N_indep, N_chain, diff, evt, border)
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, stars

BUILD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ext_output")
LOG = os.path.join(OUT, "ext06_eventstudy_count_diff.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}
EVMAX = 14

# =====================================================================================
# PART A -- build the town-quarter (evt x border) panel of N_indep, N_chain
# =====================================================================================
H("PART A -- build event-time panel: N_independent, N_chain, diff by town-quarter market age")

sf = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
chain = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))
p(f"operator pool: {len(bm)} operators, {int((bm['n_storefronts']>=2).sum())} multi-store (chain)")

meta = (sf.groupby("storefront_id")
          .agg(first_qt=("first_qt", "first"), brand_id=("brand_id", "first"),
               city=("city", "first"), border=("border_sf", "first"))
          .reset_index())
meta["mkt"] = np.where(meta["city"].isin(ABQ), "ABQ METRO", meta["city"])
meta = meta[meta["mkt"] != "ABQ METRO"].copy()
meta["mfq"] = meta.groupby("mkt")["first_qt"].transform("min")

act = (sf[sf["active"] == 1][["storefront_id", "qt", "brand_id"]]
       .merge(meta[["storefront_id", "mkt", "border", "mfq"]], on="storefront_id", how="inner"))
act["chain"] = act["brand_id"].map(chain)
act["evt"] = (act["qt"] - act["mfq"]).astype(int)
act = act[act["evt"].between(0, EVMAX)].copy()

panel = (act.groupby(["mkt", "border", "evt"])
            .agg(N_indep=("chain", lambda s: int((s == 0).sum())),
                 N_chain=("chain", lambda s: int((s == 1).sum())))
            .reset_index())
panel["diff"] = panel["N_indep"] - panel["N_chain"]
n_towns = panel["mkt"].nunique()
n_border = panel.loc[panel.border == 1, "mkt"].nunique()
p(f"town-quarter cells: {len(panel)}  ({n_towns} towns, {n_border} border / {n_towns - n_border} interior), "
  f"market age 0-{EVMAX}")
panel.to_csv(os.path.join(OUT, "ext_eventstudy_count_diff.csv"), index=False)

# =====================================================================================
# PART B -- descriptive: mean(diff) by evt x border, cluster-robust (town) SE of the mean
# =====================================================================================
H("PART B -- data: mean(N_indep - N_chain) by market age, interior vs border")

def group_mean_se(sub, ycol="diff"):
    """cluster-robust SE of a simple group mean: town-level means, then SE of the mean of means,
    scaled to reflect town-level clustering (matches the regression's clustering unit)."""
    tm = sub.groupby("mkt")[ycol].mean()
    n = len(tm)
    return float(tm.mean()), float(tm.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, n

rows = []
for e in range(EVMAX + 1):
    for b, lab in [(0, "interior"), (1, "border")]:
        sub = panel[(panel.evt == e) & (panel.border == b)]
        if len(sub) == 0:
            continue
        m, se, n = group_mean_se(sub)
        rows.append(dict(evt=e, border=b, mean_diff=m, se=se, n_towns=n))
desc = pd.DataFrame(rows)
piv_m = desc.pivot(index="evt", columns="border", values="mean_diff")
piv_se = desc.pivot(index="evt", columns="border", values="se")
p("  evt   interior mean(diff)   border mean(diff)   gap(border-interior)")
for e in range(EVMAX + 1):
    if e not in piv_m.index:
        continue
    i, b = piv_m.loc[e, 0], piv_m.loc[e, 1]
    p(f"  {e:3d}      {i:+7.2f}                {b:+7.2f}              {b-i:+7.2f}")

# =====================================================================================
# PART C -- event-study regression: town FE + event-time dummies x border, clustered SE (town)
#   diff_mt = alpha_m + sum_e theta_e * 1[evt=e] + sum_e delta_e * 1[evt=e]*border_m + eps_mt
#   delta_e is the border-interior gap in (N_indep - N_chain) at market age e, net of town FE
#   and the common age profile theta_e. Clustered by town (mkt).
# =====================================================================================
H("PART C -- event-study regression: diff ~ town FE + C(evt) + C(evt)*border, clustered by town")

reg = panel.copy()
be_cols = []
for e in range(EVMAX + 1):
    col = f"be_{e}"
    reg[col] = ((reg["evt"] == e) & (reg["border"] == 1)).astype(float)
    be_cols.append(col)

rhs = ["C(mkt)", "C(evt)"] + be_cols
_, tidy = ols_cluster(reg, "diff", rhs, "mkt")
p(f"  N = {tidy['_N']}   R2 = {tidy['_r2']:.3f}   dep. var. mean = {tidy['_ymean']:+.3f}")
p("\n  evt   delta_e (border-interior gap, town FE)   SE       95% CI")
coef_rows = []
for e in range(EVMAX + 1):
    key = f"be_{e}"
    if key in tidy:
        b, se, pv = tidy[key]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        p(f"  {e:3d}   {b:+8.3f}{stars(pv):<3}                          ({se:.3f})  [{lo:+.3f}, {hi:+.3f}]")
        coef_rows.append(dict(evt=e, delta=b, se=se, p=pv, lo=lo, hi=hi))
coef = pd.DataFrame(coef_rows).set_index("evt")

# =====================================================================================
# Fig: (A) raw data means, interior vs border, with cluster-robust CI
#      (B) event-study regression coefficients (border-interior gap), town FE, with 95% CI
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

axA = axes[0]
for b, c, lab in [(0, "C0", "interior towns"), (1, "C3", "border towns")]:
    x = piv_m.index.values
    y = piv_m[b].values
    se = piv_se[b].values
    axA.plot(x, y, "-o", color=c, lw=2, ms=4, label=lab)
    axA.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=c, alpha=.15)
axA.axhline(0, color="gray", lw=0.8, ls=":")
axA.set_xlabel("market age (quarters since the town's first dispensary)")
axA.set_ylabel("mean(N independent $-$ N chain), stores")
axA.set_title("(A) Independent $-$ chain store count, by market age", fontsize=10)
axA.legend(frameon=False, fontsize=8.5)
axA.grid(alpha=.25); axA.spines[["top", "right"]].set_visible(False)

axB = axes[1]
x = coef.index.values
y = coef["delta"].values
lo, hi = coef["lo"].values, coef["hi"].values
axB.plot(x, y, "-o", color="#5b1a1a", lw=2, ms=4)
axB.fill_between(x, lo, hi, color="#5b1a1a", alpha=.15)
axB.axhline(0, color="gray", lw=0.8, ls=":")
axB.axvspan(-0.5, 1.5, color="gray", alpha=.10)
axB.text(0.5, min(lo) - 0.4, "big-bang\nscramble", ha="center", va="bottom", fontsize=6.8, color="gray")
axB.set_xlabel("market age (quarters since the town's first dispensary)")
axB.set_ylabel(r"$\delta_e$: border $-$ interior gap in (N indep $-$ N chain)")
axB.set_title("(B) Event-study coefficients (town FE, clustered SE)", fontsize=10)
axB.grid(alpha=.25); axB.spines[["top", "right"]].set_visible(False)

fig.suptitle("The independent$-$chain composition gap between border and interior towns opens up with market age",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93])
fig_path = os.path.join(ROOT, "ext_fig_eventstudy_count_diff.png")
fig.savefig(fig_path, dpi=140)
plt.close(fig)
p(f"\nwrote {fig_path}")

# =====================================================================================
# LaTeX table
# =====================================================================================
def wtex(fn, s):
    full = os.path.join(ROOT, fn)
    open(full, "w", encoding="utf-8").write(s)
    p("wrote " + fn)
    return full

sel = [0, 2, 4, 6, 8, 10, 12, 14]
def rowline(lbl, series, fmt_="{:+.2f}"):
    cells = " & ".join(fmt_.format(series.loc[e]) for e in sel)
    return f"{lbl} & " + cells + r" \\" + "\n"

tex = r"""\begin{table}[htbp]
\centering
\caption{Independent $-$ chain store count differential, by market age}
\label{tab:ext_eventstudy_count_diff}
\footnotesize
\begin{tabular}{lcccccccc}
\toprule
Market age (quarters) & """ + " & ".join(str(e) for e in sel) + r""" \\
\midrule
\multicolumn{9}{@{}l}{\emph{Mean(N independent $-$ N chain), data}}\\
""" + \
rowline(r"\quad interior towns", piv_m[0]) + \
rowline(r"\quad border towns", piv_m[1]) + \
r"""\midrule
\multicolumn{9}{@{}l}{\emph{Event-study regression, town FE, clustered by town}}\\
""" + \
rowline(r"\quad $\delta_e$ (border $-$ interior gap)", coef["delta"]) + \
rowline(r"\quad SE", coef["se"], "({:.2f})") + \
r"""\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} New Mexico town-markets excluding the Albuquerque metro, NM CROP census 2022Q2--2026Q2.
Market age is quarters since a town's first active dispensary. A storefront is a \emph{chain} unit if its
parent operator runs two or more storefronts in New Mexico over the panel; \emph{independent} otherwise.
$\delta_e$ is the coefficient on (market age $=e$) $\times$ border in a regression of
(N independent $-$ N chain) on town fixed effects, a full set of market-age dummies, and their interaction
with border status; standard errors clustered by town. $N=""" + str(tidy["_N"]) + r"""$, $R^2=""" + \
f"{tidy['_r2']:.2f}" + r"""$.
\end{minipage}
\end{table}
"""
wtex("ext_tab_eventstudy_count_diff.tex", tex)

open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
p(f"Figure : {fig_path}")
p(f"Table  : {os.path.join(ROOT, 'ext_tab_eventstudy_count_diff.tex')}")
p(f"Panel  : {os.path.join(OUT, 'ext_eventstudy_count_diff.csv')}")
p(f"Log    : {LOG}")
