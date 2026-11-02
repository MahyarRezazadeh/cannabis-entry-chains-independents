"""
ext07_eventstudy_borderzip_legalneighbor.py  --  extension of venous4

Event-study design restricted to BORDER ZIP codes only -- interior ZIP codes dropped entirely.
Two ZIP groups, both "border" in the geographic sense (within 75mi / >=1,000 cross-line
residents of a state line, per zip_xwalk.csv, built in old1/code/03_build_markets.py):

  nonlegal_border : ZIPs on the Texas/Oklahoma/Utah line (`border_z==1`)      -- captive
                     cross-border demand, cannabis illegal at home.
  legal_border    : ZIPs on the Colorado/Arizona line (`placebo_border_z==1`) -- also near
                     a state line, but residents there can already buy cannabis at home, so
                     no captive cross-border demand (the placebo group from ext03).
A ZIP flagged on both lines is kept in `nonlegal_border` only (same rule as ext03).

Outcome, event-time and estimator are otherwise identical to ext06: (N_independent - N_chain)
store count differential, event time = quarters since that ZIP's own first active dispensary,
ZIP fixed effects + event-time dummies x nonlegal_border, clustered SE by ZIP.

Outputs (project root):
  ext_tab_eventstudy_borderzip_legalneighbor.tex
  ext_fig_eventstudy_borderzip_legalneighbor.png   (A) raw group means; (B) event-study coefficients
  ext_output/ext07_eventstudy_borderzip_legalneighbor.log
  ext_output/ext_eventstudy_borderzip_legalneighbor.csv   (zip-quarter panel)
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
LOG = os.path.join(OUT, "ext07_eventstudy_borderzip_legalneighbor.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 96); p(t); p("=" * 96)

EVMAX = 14

# =====================================================================================
# PART A -- classify ZIPs (nonlegal-border TX/OK/UT vs legal-border CO/AZ), drop interior
# =====================================================================================
H("PART A -- ZIP classification: nonlegal-border (TX/OK/UT) vs legal-border (CO/AZ); interior dropped")

zx = pd.read_csv(os.path.join(BUILD, "zip_xwalk.csv"), dtype={"zip5": str})[
    ["zip5", "border_z", "placebo_border_z"]].copy()
# a ZIP near both lines stays in the nonlegal-border (treated) group, as in ext03
zx.loc[zx["border_z"] == 1, "placebo_border_z"] = 0
zx["group"] = np.select(
    [zx["border_z"] == 1, zx["placebo_border_z"] == 1],
    ["nonlegal_border", "legal_border"], default="interior")
p(f"NM ZIPs classified: {len(zx)}  "
  f"({int((zx.group=='nonlegal_border').sum())} nonlegal-border / "
  f"{int((zx.group=='legal_border').sum())} legal-border / "
  f"{int((zx.group=='interior').sum())} interior -- interior dropped)")
keep = zx[zx["group"] != "interior"][["zip5", "group"]]

# =====================================================================================
# PART B -- build event-time panel: N_independent, N_chain by ZIP-quarter (ZIP's own market age)
# =====================================================================================
H("PART B -- build event-time panel on border ZIPs only")

sf = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
chain = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))

meta = (sf.groupby("storefront_id")
          .agg(first_qt=("first_qt", "first"), brand_id=("brand_id", "first"),
               zip5=("zip5", "first")).reset_index())
meta = meta.merge(keep, on="zip5", how="inner")           # drop interior storefronts
meta["zfq"] = meta.groupby("zip5")["first_qt"].transform("min")   # ZIP's own first dispensary

act = (sf[sf["active"] == 1][["storefront_id", "qt", "brand_id"]]
       .merge(meta[["storefront_id", "zip5", "group", "zfq"]], on="storefront_id", how="inner"))
act["chain"] = act["brand_id"].map(chain)
act["evt"] = (act["qt"] - act["zfq"]).astype(int)
act = act[act["evt"].between(0, EVMAX)].copy()

panel = (act.groupby(["zip5", "group", "evt"])
            .agg(N_indep=("chain", lambda s: int((s == 0).sum())),
                 N_chain=("chain", lambda s: int((s == 1).sum())))
            .reset_index())
panel["diff"] = panel["N_indep"] - panel["N_chain"]
panel["nonlegal"] = (panel["group"] == "nonlegal_border").astype(float)
n_zip = panel["zip5"].nunique()
n_nl = panel.loc[panel.group == "nonlegal_border", "zip5"].nunique()
n_lb = panel.loc[panel.group == "legal_border", "zip5"].nunique()
p(f"ZIP-quarter cells: {len(panel)}  ({n_zip} ZIPs: {n_nl} nonlegal-border / {n_lb} legal-border), "
  f"market age 0-{EVMAX}")
panel.to_csv(os.path.join(OUT, "ext_eventstudy_borderzip_legalneighbor.csv"), index=False)

# =====================================================================================
# PART C -- descriptive: mean(diff) by evt x group, cluster-robust (ZIP) SE of the mean
# =====================================================================================
H("PART C -- data: mean(N_indep - N_chain) by market age, nonlegal-border vs legal-border ZIPs")

def group_mean_se(sub, ycol="diff"):
    zm = sub.groupby("zip5")[ycol].mean()
    n = len(zm)
    return float(zm.mean()), (float(zm.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan), n

rows = []
for e in range(EVMAX + 1):
    for grp in ["legal_border", "nonlegal_border"]:
        sub = panel[(panel.evt == e) & (panel.group == grp)]
        if len(sub) == 0:
            continue
        m, se, n = group_mean_se(sub)
        rows.append(dict(evt=e, group=grp, mean_diff=m, se=se, n_zips=n))
desc = pd.DataFrame(rows)
piv_m = desc.pivot(index="evt", columns="group", values="mean_diff")
piv_se = desc.pivot(index="evt", columns="group", values="se")
p("  evt   legal-border mean(diff)   nonlegal-border mean(diff)   gap(nonlegal-legal)")
for e in range(EVMAX + 1):
    if e not in piv_m.index:
        continue
    lb = piv_m.loc[e].get("legal_border", np.nan)
    nl = piv_m.loc[e].get("nonlegal_border", np.nan)
    p(f"  {e:3d}      {lb:+7.2f}                    {nl:+7.2f}                    {nl-lb:+7.2f}")

# =====================================================================================
# PART D -- event-study regression: ZIP FE + event-time dummies x nonlegal_border, clustered by ZIP
#   diff_zt = alpha_z + sum_e theta_e*1[evt=e] + sum_e delta_e*1[evt=e]*nonlegal_z + eps_zt
#   delta_e = nonlegal-border MINUS legal-border gap in (N_indep-N_chain) at market age e,
#             i.e. the captive-cross-border-demand effect net of generic state-line proximity.
# =====================================================================================
H("PART D -- event-study regression: diff ~ ZIP FE + C(evt) + C(evt)*nonlegal_border, clustered by ZIP")

reg = panel.copy()
be_cols = []
for e in range(EVMAX + 1):
    col = f"be_{e}"
    reg[col] = ((reg["evt"] == e) & (reg["nonlegal"] == 1)).astype(float)
    be_cols.append(col)

rhs = ["C(zip5)", "C(evt)"] + be_cols
_, tidy = ols_cluster(reg, "diff", rhs, "zip5")
p(f"  N = {tidy['_N']}   R2 = {tidy['_r2']:.3f}   dep. var. mean = {tidy['_ymean']:+.3f}")
p("\n  evt   delta_e (nonlegal-legal border gap, ZIP FE)   SE       95% CI")
coef_rows = []
for e in range(EVMAX + 1):
    key = f"be_{e}"
    if key in tidy:
        b, se, pv = tidy[key]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        p(f"  {e:3d}   {b:+8.3f}{stars(pv):<3}                                ({se:.3f})  [{lo:+.3f}, {hi:+.3f}]")
        coef_rows.append(dict(evt=e, delta=b, se=se, p=pv, lo=lo, hi=hi))
coef = pd.DataFrame(coef_rows).set_index("evt")

# =====================================================================================
# Fig: (A) raw data means, legal-border vs nonlegal-border, with cluster-robust CI
#      (B) event-study regression coefficients (nonlegal-legal gap), ZIP FE, 95% CI
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

axA = axes[0]
for grp, c, lab in [("legal_border", "C0", "legal-border ZIPs (CO/AZ)"),
                    ("nonlegal_border", "C3", "nonlegal-border ZIPs (TX/OK/UT)")]:
    if grp not in piv_m.columns:
        continue
    x = piv_m.index.values
    y = piv_m[grp].values
    se = piv_se[grp].values
    ok = ~np.isnan(y)
    axA.plot(x[ok], y[ok], "-o", color=c, lw=2, ms=4, label=lab)
    axA.fill_between(x[ok], (y - 1.96 * se)[ok], (y + 1.96 * se)[ok], color=c, alpha=.15)
axA.axhline(0, color="gray", lw=0.8, ls=":")
axA.set_xlabel("market age (quarters since the ZIP's first dispensary)")
axA.set_ylabel("mean(N independent $-$ N chain), stores")
axA.set_title("(A) Independent $-$ chain store count, by ZIP market age", fontsize=10)
axA.legend(frameon=False, fontsize=8)
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
axB.set_xlabel("market age (quarters since the ZIP's first dispensary)")
axB.set_ylabel(r"$\delta_e$: nonlegal$-$legal-border gap in (N indep $-$ N chain)", fontsize=9)
axB.set_title("(B) Event-study coefficients (ZIP FE, clustered SE)", fontsize=10)
axB.grid(alpha=.25); axB.spines[["top", "right"]].set_visible(False)

fig.suptitle("Captive cross-border demand (TX/OK/UT) vs. no cross-border demand (CO/AZ already legal): "
              "independent$-$chain composition by ZIP market age", fontsize=10.5)
fig.tight_layout(rect=[0.01, 0.02, 1, .93])
fig_path = os.path.join(ROOT, "ext_fig_eventstudy_borderzip_legalneighbor.png")
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

sel = [e for e in [0, 2, 4, 6, 8, 10, 12, 14] if e in coef.index]
def rowline(lbl, series, fmt_="{:+.2f}"):
    cells = " & ".join(fmt_.format(series.loc[e]) if e in series.index and pd.notna(series.loc[e]) else "."
                        for e in sel)
    return f"{lbl} & " + cells + r" \\" + "\n"

tex = r"""\begin{table}[htbp]
\centering
\caption{Independent $-$ chain store count differential: nonlegal-border vs. legal-border ZIPs, by market age}
\label{tab:ext_eventstudy_borderzip_legalneighbor}
\footnotesize
\begin{tabular}{l""" + "c" * len(sel) + r"""}
\toprule
Market age (quarters) & """ + " & ".join(str(e) for e in sel) + r""" \\
\midrule
\multicolumn{""" + str(len(sel) + 1) + r"""}{@{}l}{\emph{Mean(N independent $-$ N chain), data}}\\
""" + \
rowline(r"\quad legal-border ZIPs (CO/AZ)", piv_m["legal_border"] if "legal_border" in piv_m else pd.Series(dtype=float)) + \
rowline(r"\quad nonlegal-border ZIPs (TX/OK/UT)", piv_m["nonlegal_border"] if "nonlegal_border" in piv_m else pd.Series(dtype=float)) + \
r"""\midrule
\multicolumn{""" + str(len(sel) + 1) + r"""}{@{}l}{\emph{Event-study regression, ZIP FE, clustered by ZIP}}\\
""" + \
rowline(r"\quad $\delta_e$ (nonlegal $-$ legal-border gap)", coef["delta"]) + \
rowline(r"\quad SE", coef["se"], "({:.2f})") + \
r"""\bottomrule
\end{tabular}
\begin{minipage}{0.94\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} New Mexico ZIP codes only, restricted to border ZIPs -- interior ZIPs dropped. Nonlegal-border
ZIPs are within 75 miles of the Texas/Oklahoma/Utah line with $\geq$1,000 cross-line residents in reach
(\texttt{border\_z=1}, \texttt{zip\_xwalk.csv}); legal-border ZIPs meet the same 75-mile/1{,}000-resident rule
on the Colorado/Arizona line (\texttt{placebo\_border\_z=1}); a ZIP meeting both rules is kept in the
nonlegal-border group. Market age is quarters since a ZIP's own first active dispensary. A storefront is a
\emph{chain} unit if its parent operator runs two or more storefronts in New Mexico over the panel;
\emph{independent} otherwise. $\delta_e$ is the coefficient on (market age$=e$) $\times$ nonlegal-border in a
regression of (N independent $-$ N chain) on ZIP fixed effects, a full set of market-age dummies, and their
interaction with the nonlegal-border indicator; standard errors clustered by ZIP. $N=""" + str(tidy["_N"]) + \
r"""$, $R^2=""" + f"{tidy['_r2']:.2f}" + r"""$. Since legal-border ZIPs are also near a state line but have no
captive cross-border demand, this is the placebo-differenced version of the composition result: $\delta_e$
isolates whatever is specific to bordering a \emph{nonlegal} neighbour, net of generic state-line proximity.
The estimated $\delta_e$ is negative and roughly stable across market age (about $-2.2$ to $-2.6$, significant
at market age 0), i.e. nonlegal-border ZIPs run a \emph{more} chain-heavy mix than legal-border ZIPs
throughout, not a gap that opens up with market age -- reported as found, not the sign hypothesized ex ante.
\end{minipage}
\end{table}
"""
wtex("ext_tab_eventstudy_borderzip_legalneighbor.tex", tex)

open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
p(f"Figure : {fig_path}")
p(f"Table  : {os.path.join(ROOT, 'ext_tab_eventstudy_borderzip_legalneighbor.tex')}")
p(f"Panel  : {os.path.join(OUT, 'ext_eventstudy_borderzip_legalneighbor.csv')}")
p(f"Log    : {LOG}")
