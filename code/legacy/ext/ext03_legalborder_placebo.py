"""
ext03_legalborder_placebo.py  --  extension of venous4

Legal-neighbour placebo for the franchise/independent composition result (Section 5.3).

The main comparison is border ZIP codes (within 75 mi of the Texas/Oklahoma/Utah line,
>=1,000 nonlegal-state residents in reach) vs. all other New Mexico ZIP codes.  Here we add a
THIRD group: "legal-border" ZIP codes -- same 75-mile / 1,000-resident rule applied to the
Colorado/Arizona line, where consumers on the other side can already buy cannabis at home, so
there is no captive cross-border demand.  Classification file
`nm_zip_legalborder_stateline.csv` is produced by the base project's
`analyze_nm_zip_dynamic_legalcontrol.py`.

Horse race: regress each composition outcome on BOTH  border_z  and  legal_border_z  (interior
= omitted).  If the composition effect is driven by captive cross-border demand rather than by
being near any state line, border_z should be large and significant while legal_border_z is
near zero.

Output: ext_tab_legalborder_placebo.tex ; ext_output/ext03_legalborder_placebo.log
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, stars, asinh

BE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\brand_entry_analysis\tables"
LB = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project\tables\nm_zip_legalborder_stateline.csv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ext_output", "ext03_legalborder_placebo.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _l.append(s)

zq = pd.read_csv(os.path.join(BE, "panel_zip_quarter.csv"), dtype={"zip5": str, "county_fips": str})
sfq = pd.read_csv(os.path.join(BE, "panel_storefront_quarter.csv"), dtype={"zip5": str, "fips5": str})
bm = pd.read_csv(os.path.join(BE, "brand_master.csv"))
lb = pd.read_csv(LB, dtype={"zip5": str})[["zip5", "legal_border_zip"]]

bm["chain"] = (bm["n_storefronts"] >= 2).astype(int)
sfq = sfq.merge(bm[["brand_id", "chain"]], on="brand_id", how="left")
act = sfq[sfq["active"] == 1]
g = (act.groupby(["zip5", "qt"])
        .apply(lambda s: pd.Series({
            "n_chain_brands": s.loc[s.chain == 1, "brand_id"].nunique(),
            "n_indep_brands": s.loc[s.chain == 0, "brand_id"].nunique()}), include_groups=False)
        .reset_index())
z = zq.merge(g, on=["zip5", "qt"], how="left").merge(lb, on="zip5", how="left")
z["legal_border_zip"] = z["legal_border_zip"].fillna(0).astype(int)
z = z[z["border_z"].notna()].copy()
# a ZIP near BOTH a nonlegal and a legal line stays in the treated (nonlegal-border) group
z.loc[z["border_z"] == 1, "legal_border_zip"] = 0
for c in ["n_chain_brands", "n_indep_brands"]:
    z[c] = z[c].fillna(0)
z["log_zip_pop"] = np.log(z["zip_pop"].clip(lower=1))
z["log_nbrands"] = np.log(z["n_brands"].clip(lower=1))
z["log_revenue"] = np.log(z["total_revenue"].clip(lower=1))
z["log_rev_per_store"] = np.log((z["total_revenue"] / z["n_stores"]).clip(lower=1))

zz = z.drop_duplicates("zip5")
p(f"ZIP-quarter N = {len(z):,}   ZIPs = {z.zip5.nunique()}")
p(f"  nonlegal-border ZIPs : {int((zz.border_z==1).sum())}")
p(f"  legal-border ZIPs    : {int((zz.legal_border_zip==1).sum())}")
p(f"  interior ZIPs        : {int(((zz.border_z==0)&(zz.legal_border_zip==0)).sum())}")

OUT = {
    "log_nbrands": "Log(number of brands)",
    "n_indep_brands": "Number of independent brands",
    "n_chain_brands": "Number of chain brands",
    "hhi_10000": "HHI (0--10{,}000)",
    "cr1": "Largest-brand revenue share",
    "entropy": "Brand entropy",
    "log_revenue": "Log(total revenue)",
    "log_rev_per_store": "Log(revenue per store)",
}
p("\nHORSE RACE  (each outcome ~ border_z + legal_border_z + log pop + quarter FE, cluster county)")
p(f"{'outcome':32s} {'border_z':>18s} {'legal_border_z':>18s}   N")
rows = []
for y, lab in OUT.items():
    _, t = ols_cluster(z, y, ["border_z", "legal_border_zip", "log_zip_pop", "C(qt)"], "county_fips")
    b1, s1, p1 = t["border_z"]; b2, s2, p2 = t["legal_border_zip"]
    p(f"{lab:32s} {b1:>10.3f}{stars(p1):<3}({s1:.3f}) {b2:>10.3f}{stars(p2):<3}({s2:.3f})   {t['_N']}")
    rows.append(dict(outcome=lab, b_border=b1, se_border=s1, p_border=p1,
                     b_legal=b2, se_legal=s2, p_legal=p2, N=t["_N"]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "ext_output", "ext_legalborder_placebo.csv"), index=False)

# restricted sample: keep only border + legal-border ZIPs (drop true interior), border_z vs legal-border
p("\nRESTRICTED to border ZIPs only (nonlegal-border vs legal-border; interior dropped)")
r = z[(z.border_z == 1) | (z.legal_border_zip == 1)].copy()
rr = r.drop_duplicates("zip5")
p(f"  {int((rr.border_z==1).sum())} nonlegal-border vs {int((rr.legal_border_zip==1).sum())} legal-border ZIPs")
rows2 = []
for y, lab in OUT.items():
    _, t = ols_cluster(r, y, ["border_z", "log_zip_pop", "C(qt)"], "county_fips")
    b, s, pv = t["border_z"]
    p(f"  {lab:32s} border_z = {b:>10.3f}{stars(pv):<3} ({s:.3f})   N={t['_N']}")
    rows2.append(dict(outcome=lab, b=b, se=s, p=pv, N=t["_N"]))
df2 = pd.DataFrame(rows2)

st = lambda pv: stars(pv)
with open(os.path.join(ROOT, "ext_tab_legalborder_placebo.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Legal-neighbour placebo for the composition result}
\label{tab:ext_legalborder}
\footnotesize
\begin{tabular}{lccc}
\toprule
 & \multicolumn{2}{c}{Horse race (interior omitted)} & Border ZIPs only \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}
Outcome & $\text{Border}_z$ (TX/OK/UT) & Legal-border$_z$ (CO/AZ) & $\text{Border}_z$ \\
\midrule
""")
    for (_, a), (_, b) in zip(df.iterrows(), df2.iterrows()):
        fh.write(f"{a['outcome']} & {a['b_border']:.3f}{st(a['p_border'])} ({a['se_border']:.3f}) & "
                 f"{a['b_legal']:.3f}{st(a['p_legal'])} ({a['se_legal']:.3f}) & "
                 f"{b['b']:.3f}{st(b['p'])} ({b['se']:.3f}) \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} ZIP-code $\times$ quarter panel, New Mexico, 2022Q2--2026Q2. Columns 1--2: each row is
one regression of the outcome on the nonlegal-border indicator, the legal-border indicator (CO/AZ line,
same 75-mile / 1{,}000-resident rule), $\log$(ZIP population), and quarter fixed effects, with the
interior ZIP codes omitted. Column 3 drops the interior ZIP codes and compares nonlegal-border with
legal-border ZIP codes directly ("""
+ f"{int((rr.border_z==1).sum())}" + r""" vs """ + f"{int((rr.legal_border_zip==1).sum())}" + r""" ZIP
codes). Standard errors clustered at the county level. A composition effect driven by captive
cross-border demand -- rather than by generic proximity to a state line -- should load on
$\text{Border}_z$ and not on the legal-border indicator. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("\nwrote ext_tab_legalborder_placebo.tex")
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
