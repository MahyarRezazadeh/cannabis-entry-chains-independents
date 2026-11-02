"""
ext04_regime_fiscal.py  --  extension of venous4

(1) State-specific border premium, New Mexico (free entry) vs. Colorado (locally
    restricted): the empirical analogue of the model's cap counterfactual.  Under free
    entry the border premium is extensive (dispensary density); under local restriction it
    is intensive (revenue per store).

(2) Fiscal counterfactual: how much cannabis revenue -- and state tax -- New Mexico's border
    counties would forgo under a Colorado-style restricted-entry regime.

County-quarter panel `master_quarterly_panel_no_wa.csv` (the base project's cross-state file);
Colorado restricted to 2022Q2 onward to match New Mexico's window (draft Section A.4).

Outputs: ext_tab_regime_split.tex, ext_tab_fiscal.tex ; ext_output/ext04_regime_fiscal.log
"""
import os, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import stars

NM = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project\tables"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ext_output", "ext04_regime_fiscal.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _l.append(s)
pct = lambda b: f"{(np.exp(b) - 1) * 100:+.1f}\\%"

df = pd.read_csv(os.path.join(NM, "master_quarterly_panel_no_wa.csv"), dtype={"fips5": str})
b = pd.read_csv(os.path.join(NM, "county_border_stateline.csv"), dtype={"fips5": str})
df = df[df.state_abbr.isin(["NM", "CO"])].merge(b[["fips5", "border_county"]], on="fips5", how="left")
df = df.dropna(subset=["border_county"]).copy()
df["border_county"] = df["border_county"].astype(int)
# CO restricted to 2022Q2 onward; both states cut at 2025Q2 (draft Section A.4)
df = df[(df.year < 2025) | ((df.year == 2025) & (df.quarter <= 2))]
df = df[(df.state_abbr != "CO") | (df.year > 2022) | ((df.year == 2022) & (df.quarter >= 2))].copy()
df["period"] = df["yq"].astype(str)

OUTC = [("log_revenue_per_capita", "Revenue per capita"),
        ("log_dispensary_per_1000", "Dispensary density"),
        ("log_revenue_per_store", "Revenue per store")]

# ---------------------------------------------------------------- (1) state-specific premium
p("=" * 88)
p("(1)  STATE-SPECIFIC BORDER PREMIUM  (y ~ border_county + C(period), cluster county)")
p("=" * 88)
res = {}
for st in ["NM", "CO"]:
    d0 = df[df.state_abbr == st]
    for y, lab in OUTC:
        d = d0.dropna(subset=[y]).copy()
        m = smf.ols(f"{y} ~ border_county + C(period)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        bb, se, pv = m.params["border_county"], m.bse["border_county"], m.pvalues["border_county"]
        res[(st, lab)] = (bb, se, pv, int(m.nobs), d.fips5.nunique(),
                          int(d.drop_duplicates("fips5").border_county.sum()))
        p(f"  {st}  {lab:22s}  b={bb:+.3f} ({se:.3f})  {pct(bb):>8s}  p={pv:.3f}  "
          f"n={int(m.nobs)}  {d.fips5.nunique()} counties ({res[(st,lab)][5]} border)")

with open(os.path.join(ROOT, "ext_tab_regime_split.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{State-specific border premium: free entry (New Mexico) vs.\ local restriction (Colorado)}
\label{tab:ext_regime_split}
\footnotesize
\begin{tabular}{lcc}
\toprule
Outcome & New Mexico (free entry) & Colorado (locally restricted) \\
\midrule
""")
    for y, lab in OUTC:
        nb = res[("NM", lab)]; cb = res[("CO", lab)]
        fh.write(f"{lab} & {nb[0]:+.3f}{stars(nb[2])} ({nb[1]:.3f}) & {cb[0]:+.3f}{stars(cb[2])} ({cb[1]:.3f}) \\\\\n")
    nb = res[("NM", "Revenue per capita")]; cb = res[("CO", "Revenue per capita")]
    fh.write(r"\midrule" + "\n")
    fh.write(f"Counties (of which border) & {nb[4]} ({nb[5]}) & {cb[4]} ({cb[5]}) \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.90\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} County-quarter panel; each cell is the coefficient on a fixed border-county
indicator from a separate within-state regression that also includes quarter fixed effects, with
county-clustered standard errors. New Mexico is 2022Q2--2025Q2; Colorado is restricted to the same
window. Border counties are within 75 miles of a nonlegal-state line (Texas/Oklahoma/Utah for New
Mexico; Utah/Oklahoma/Kansas/Nebraska/Wyoming for Colorado). Under New Mexico's free entry the
border premium is on the \emph{extensive} margin (dispensary density); under Colorado's local
restriction it is on the \emph{intensive} margin (revenue per store) -- the pattern the model's
license-cap counterfactual (Table~\ref{tab:ext_counterfactual}) predicts. $^{*}p<0.10$,
$^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_regime_split.tex")

# ---------------------------------------------------------------- (2) fiscal counterfactual
p("\n" + "=" * 88)
p("(2)  FISCAL COUNTERFACTUAL  --  New Mexico border counties under a Colorado-style cap")
p("=" * 88)
beta_rpc_nm = res[("NM", "Revenue per capita")][0]         # NM border rev-per-capita premium
beta_rpc_co = res[("CO", "Revenue per capita")][0]         # CO's (~0)
# density-interaction theta from the base draft (Table A.6 / build_counterfactual_co2022_density.py)
theta_density = -0.932

nm24 = df[(df.state_abbr == "NM") & (df.year == 2024)]
nm_border_2024 = float(nm24.loc[nm24.border_county == 1, "total_sales"].sum())
nm_border_counties = int(nm24.loc[nm24.border_county == 1, "fips5"].nunique())
p(f"  New Mexico 2024 border-county cannabis sales: ${nm_border_2024/1e6:,.1f} M  "
  f"({nm_border_counties} counties)")
p(f"  NM border revenue-per-capita premium: b={beta_rpc_nm:+.3f}  ({pct(beta_rpc_nm)})")
p(f"  CO border revenue-per-capita premium: b={beta_rpc_co:+.3f}  ({pct(beta_rpc_co)}, n.s.)")

# a CO-style cap removes the NM border revenue premium (CO's is ~0):
forgone_rpc = nm_border_2024 * (1 - np.exp(-beta_rpc_nm))
forgone_density = nm_border_2024 * (1 - np.exp(theta_density))   # using the significant density interaction
p(f"\n  Forgone border-county cannabis sales under a Colorado-style cap:")
p(f"    via the revenue-per-capita premium : ${forgone_rpc/1e6:,.0f} M/yr  "
  f"(sales fall to ${nm_border_2024*np.exp(-beta_rpc_nm)/1e6:,.0f} M)")
p(f"    via the density interaction theta  : ${forgone_density/1e6:,.0f} M/yr")

# New Mexico tax rates (2024): 12% Cannabis Excise Tax + gross receipts (~7.6% combined state+local avg)
EXCISE = 0.12
GRT = 0.076
for lbl, forg in [("revenue-per-capita", forgone_rpc), ("density interaction", forgone_density)]:
    p(f"  [{lbl}] forgone tax:  state excise (12%) ${forg*EXCISE/1e6:,.1f} M/yr ;  "
      f"excise + gross receipts (~19.6%) ${forg*(EXCISE+GRT)/1e6:,.1f} M/yr")

with open(os.path.join(ROOT, "ext_tab_fiscal.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Fiscal cost of a restricted-entry regime: New Mexico border counties}
\label{tab:ext_fiscal}
\footnotesize
\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
New Mexico border-county cannabis sales, 2024 (""" + f"{nm_border_counties}" + r""" counties) & """
+ f"\\${nm_border_2024/1e6:,.0f} M" + r""" \\
Border revenue-per-capita premium (free entry) & """ + f"{pct(beta_rpc_nm)}" + r""" \\
Same premium under Colorado's restricted regime & """ + f"{pct(beta_rpc_co)}" + r""" (n.s.) \\
\midrule
Forgone border-county sales under a Colorado-style cap & """ + f"\\${forgone_rpc/1e6:,.0f} M/yr" + r""" \\
\quad forgone state cannabis excise tax (12\%) & """ + f"\\${forgone_rpc*EXCISE/1e6:,.0f} M/yr" + r""" \\
\quad forgone excise + gross-receipts tax ($\approx$19.6\%) & """ + f"\\${forgone_rpc*(EXCISE+GRT)/1e6:,.0f} M/yr" + r""" \\
\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} New Mexico's 10 border counties earned \$""" + f"{nm_border_2024/1e6:,.0f}" + r""" million in
cannabis sales in 2024. Their revenue-per-capita border premium is """ + f"{pct(beta_rpc_nm)}" + r"""
(Table~\ref{tab:ext_regime_split}); Colorado's restricted-entry border counties show no such premium
(""" + f"{pct(beta_rpc_co)}" + r""", not significant). A regime that removed the premium would forgo
$\text{sales}\times(1-e^{-\hat\beta})\approx$ \$""" + f"{forgone_rpc/1e6:,.0f}" + r""" million per year in
border-county sales, and the corresponding state tax at New Mexico's 2024 rates. This is the mirror
image of the Colorado benchmark in Table~\ref{tab:counterfactual} (\$499 million of dispensary revenue
Colorado's border counties would have generated under free entry). Both are illustrative: the
underlying cross-state comparison rests on two states and is only marginally significant.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_fiscal.tex")
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
