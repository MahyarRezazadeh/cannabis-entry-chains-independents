"""
ext01_reduced_form_bytype.py  --  extension of venous4

Franchise/chain vs independent supply response at the New Mexico cannabis border.
Reuses the documented brand crosswalk + border classification from
`brand_entry_analysis/` (2022Q2-2026Q2 NM CROP census; 1,230 storefronts,
653 parent brands, 180 multi-location).  A "chain" storefront/brand is one whose
parent operator runs >=2 storefronts in New Mexico over the panel; an
"independent" is a single-storefront operator.

Outputs (project root):
  ext_tab_composition_bytype.tex   ZIP x quarter: # independent vs # chain brands, shares, HHI
  ext_tab_entry_bytype.tex         storefront entrants: P(de-novo brand) at the border
  ext_tab_sorting.tex              brand cross-section: border share by brand size
  ext_tab_exit_bytype.tex          storefront exit: border x chain interaction
  ext_output/ext01_reduced_form_bytype.log
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, poisson_hc1, stars, fmt, col, asinh

BE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana\brand_entry_analysis\tables"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ext_output", "ext01_reduced_form_bytype.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

# ------------------------------------------------------------------ load
zq  = pd.read_csv(os.path.join(BE, "panel_zip_quarter.csv"), dtype={"zip5": str, "county_fips": str})
sfq = pd.read_csv(os.path.join(BE, "panel_storefront_quarter.csv"), dtype={"zip5": str, "fips5": str})
bm  = pd.read_csv(os.path.join(BE, "brand_master.csv"))
sfm = pd.read_csv(os.path.join(BE, "storefront_master.csv"), dtype={"zip5": str, "fips5": str})

bm["chain"] = (bm["n_storefronts"] >= 2).astype(int)
sfq = sfq.merge(bm[["brand_id", "n_storefronts", "chain"]], on="brand_id", how="left")
p(f"storefront-quarter rows: {len(sfq):,};  brands: {bm.shape[0]};  chains: {int(bm['chain'].sum())};  "
  f"independents: {int((bm['chain']==0).sum())}")

# ================================================================ §1  ZIP x quarter composition by type
H("§1  ZIP x QUARTER MARKET STRUCTURE, INDEPENDENT vs CHAIN  (border_z, +log pop, quarter FE, cluster county)")
act = sfq[sfq["active"] == 1].copy()
g = (act.groupby(["zip5", "qt"])
        .agg(n_chain_sf=("chain", "sum"),
             n_indep_sf=("chain", lambda s: int((s == 0).sum())),
             n_chain_brands=("brand_id", lambda s: act.loc[s.index].loc[act["chain"] == 1, "brand_id"].nunique()),
             ).reset_index())
# robust rebuild of brand counts by type
def _bc(sub):
    return pd.Series({
        "n_chain_brands": sub.loc[sub["chain"] == 1, "brand_id"].nunique(),
        "n_indep_brands": sub.loc[sub["chain"] == 0, "brand_id"].nunique(),
        "n_chain_sf": int((sub["chain"] == 1).sum()),
        "n_indep_sf": int((sub["chain"] == 0).sum()),
    })
g = act.groupby(["zip5", "qt"]).apply(_bc).reset_index()
z = zq.merge(g, on=["zip5", "qt"], how="left")
for c in ["n_chain_brands", "n_indep_brands", "n_chain_sf", "n_indep_sf"]:
    z[c] = z[c].fillna(0)
z = z[z["border_z"].notna()].copy()
z["log_zip_pop"] = np.log(z["zip_pop"].clip(lower=1))
z["log_nbrands"] = np.log(z["n_brands"].clip(lower=1))
z["log_revenue"] = np.log(z["total_revenue"].clip(lower=1))
z["log_rev_per_store"] = np.log((z["total_revenue"] / z["n_stores"]).clip(lower=1))
z["indep_brand_share"] = z["n_indep_brands"] / z["n_brands"].clip(lower=1)
z["asinh_nl75"] = asinh(z["nonleg_pop_75"])
p(f"ZIP-quarter sample (>=1 active store): N = {len(z):,}, ZIPs = {z['zip5'].nunique()}, "
  f"border ZIP-quarters = {int((z['border_z']==1).sum())}")

OUT = {
    "log_nbrands":        ("Log(number of brands)",        "log"),
    "n_indep_brands":     ("Number of independent brands", "lvl"),
    "n_chain_brands":     ("Number of chain brands",       "lvl"),
    "indep_brand_share":  ("Independent share of brands",  "lvl"),
    "indep_store_share":  ("Independent share of stores",  "lvl"),
    "hhi_10000":          ("HHI (0--10{,}000)",            "lvl"),
    "cr1":                ("Largest-brand revenue share",  "lvl"),
    "entropy":            ("Brand entropy",                "lvl"),
    "log_revenue":        ("Log(total revenue)",           "log"),
    "log_rev_per_store":  ("Log(revenue per store)",       "log"),
}
rows = []
for y, (lab, kind) in OUT.items():
    _, b = ols_cluster(z, y, ["border_z", "log_zip_pop", "C(qt)"], "county_fips")
    _, d = ols_cluster(z, y, ["log_dist_stateline", "log_zip_pop", "C(qt)"], "county_fips")
    _, x = ols_cluster(z, y, ["asinh_nl75", "log_zip_pop", "C(qt)"], "county_fips")
    p(f"  {lab:32s}  mean={b['_ymean']:>9.3f}  border={fmt(b,'border_z'):>12s}  "
      f"logDist={fmt(d,'log_dist_stateline'):>12s}  asinhNL={fmt(x,'asinh_nl75'):>12s}   N={b['_N']}")
    rows.append(dict(outcome=lab, mean=b["_ymean"],
                     b_border=b["border_z"][0], se_border=b["border_z"][1], p_border=b["border_z"][2],
                     b_logdist=d["log_dist_stateline"][0], p_logdist=d["log_dist_stateline"][2],
                     b_asinh=x["asinh_nl75"][0], p_asinh=x["asinh_nl75"][2], N=b["_N"]))
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(ROOT, "ext_output", "ext_tab_composition_bytype.csv"), index=False)

# LaTeX table
def st(pv): return stars(pv)
with open(os.path.join(ROOT, "ext_tab_composition_bytype.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Market structure at the border, independent versus chain dispensaries}
\label{tab:ext_composition}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Outcome & Mean & $\text{Border}_z$ & $\log$ distance & $\mathrm{asinh}$ nonlegal \\
 &  & $+\log$ pop & to line $+\log$ pop & pop.\ 75\,mi $+\log$ pop \\
\midrule
""")
    for _, r in comp.iterrows():
        fh.write(f"{r['outcome']} & {r['mean']:,.2f} & "
                 f"{r['b_border']:.3f}{st(r['p_border'])} ({r['se_border']:.3f}) & "
                 f"{r['b_logdist']:.3f}{st(r['p_logdist'])} & "
                 f"{r['b_asinh']:.3f}{st(r['p_asinh'])} \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} ZIP-code $\times$ quarter panel of New Mexico ZIP codes with at least one operating
dispensary, 2022Q2--2026Q2 (""" + f"$N={comp['N'].iloc[0]:,}$" + r"""). Each cell is the coefficient on the
named cross-border proxy from a separate regression that also includes $\log$(ZIP population) and quarter
fixed effects; standard errors clustered at the county level. A dispensary is a \emph{chain} unit if its
parent operator runs two or more storefronts in New Mexico over the panel, and \emph{independent}
otherwise. $\log$ distance and $\mathrm{asinh}$ nonlegal-population columns are separate regressions
(both are border proxies and collinear with the indicator). $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_composition_bytype.tex")

# Poisson of the equilibrium count by type (ZIP mature cross-section)
H("§1b  POISSON: equilibrium # of independent vs chain brands per ZIP (mature 2024Q1-2026Q2 mean)")
mat = z[z["qt"] >= 8097]  # 2024Q1+
zc = (mat.groupby("zip5").agg(n_brands=("n_brands", "mean"), n_indep=("n_indep_brands", "mean"),
      n_chain=("n_chain_brands", "mean"), zip_pop=("zip_pop", "first"),
      border_z=("border_z", "first"), nonleg_pop_75=("nonleg_pop_75", "first")).reset_index())
zc["log_zip_pop"] = np.log(zc["zip_pop"].clip(lower=1))
zc["log_nl75"] = np.log(zc["nonleg_pop_75"].clip(lower=1))
for y in ["n_brands", "n_indep", "n_chain"]:
    zc[y + "_r"] = zc[y].round().astype(int)
    a = poisson_hc1(zc, y + "_r", ["log_zip_pop", "border_z"])
    aa = poisson_hc1(zc, y + "_r", ["log_zip_pop", "log_nl75"])
    p(f"  {y:10s}  border={a['border_z'][0]:+.3f}{stars(a['border_z'][2])}  "
      f"log_nl75={aa['log_nl75'][0]:+.3f}{stars(aa['log_nl75'][2])}   N={a['_N']}")

# ================================================================ §2  entrant composition
H("§2  STOREFRONT ENTRANTS: is a border entrant a de-novo brand or a chain adding a location?")
# de-novo = this storefront is the first NM storefront (earliest first_active_qt) of its parent brand
b_first = sfm.groupby("brand_id")["first_t"].min().rename("brand_first_t").reset_index()
ent = sfm.merge(b_first, on="brand_id", how="left")
ent = ent[ent["left_censored"] == False].copy()          # observed entrants only
ent["denovo"] = (ent["first_t"] <= ent["brand_first_t"]).astype(int)
ent = ent.merge(bm[["brand_id", "chain"]], on="brand_id", how="left")
ent["log_localpop"] = np.log(ent.get("nonlegal_pop_within_75mi", pd.Series(index=ent.index)).fillna(0).clip(lower=1))
zpop = zq.groupby("zip5")["zip_pop"].first().reset_index()
ent = ent.merge(zpop, on="zip5", how="left")
ent["log_zip_pop"] = np.log(ent["zip_pop"].clip(lower=1))
ent["log_dist_stateline"] = np.log(ent["dist_stateline_mi"].clip(lower=0.1))
ent = ent.dropna(subset=["border_dispensary"])
ent["entry_yq"] = ent["entry_year"].astype(int) * 4 + ((ent["entry_month"].astype(int) - 1) // 3)
p(f"observed storefront entrants: {len(ent):,}   de-novo share: {ent['denovo'].mean():.3f}   "
  f"border entrants: {int(ent['border_dispensary'].sum())}")
_, e1 = ols_cluster(ent, "denovo", ["border_dispensary", "log_zip_pop", "C(entry_yq)"], "fips5")
_, e2 = ols_cluster(ent, "denovo", ["border_dispensary", "log_zip_pop", "log_dist_stateline", "C(entry_yq)"], "fips5")
p(col(e1, ["border_dispensary", "log_zip_pop"]))
p("  --- + log distance to line ---")
p(col(e2, ["border_dispensary", "log_zip_pop", "log_dist_stateline"]))
b_dn, s_dn, p_dn = e1["border_dispensary"]

with open(os.path.join(ROOT, "ext_tab_entry_bytype.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Border exposure and the type of storefront entrant}
\label{tab:ext_entry}
\footnotesize
\begin{tabular}{lcc}
\toprule
 & (1) & (2) \\
Dependent variable: entrant is a \emph{de-novo brand} & \multicolumn{2}{c}{(first New Mexico store of its operator)} \\
\midrule
""")
    fh.write(f"$\\text{{Border}}_d$ & {e1['border_dispensary'][0]:.3f}{st(e1['border_dispensary'][2])} & "
             f"{e2['border_dispensary'][0]:.3f}{st(e2['border_dispensary'][2])} \\\\\n")
    fh.write(f" & ({e1['border_dispensary'][1]:.3f}) & ({e2['border_dispensary'][1]:.3f}) \\\\\n")
    fh.write(f"$\\log$(ZIP population) & {e1['log_zip_pop'][0]:.3f}{st(e1['log_zip_pop'][2])} & "
             f"{e2['log_zip_pop'][0]:.3f}{st(e2['log_zip_pop'][2])} \\\\\n")
    fh.write(f"$\\log$ distance to nonlegal line & & {e2['log_dist_stateline'][0]:.3f}{st(e2['log_dist_stateline'][2])} \\\\\n")
    fh.write(r"\midrule" + "\n")
    fh.write(f"Entry-year-quarter fixed effects & Yes & Yes \\\\\n")
    fh.write(f"Observations & {e1['_N']:,} & {e2['_N']:,} \\\\\n")
    fh.write(f"Mean of dependent variable & {ent['denovo'].mean():.3f} & {ent['denovo'].mean():.3f} \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.86\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Linear probability models on the """ + f"{e1['_N']:,}" + r""" storefront entries observed after
2022Q2 (left-censored storefronts operating at the panel start are excluded). The dependent variable equals
one if the entering storefront is the first New Mexico location of its parent operator and zero if an
existing multi-store operator is adding a location. Standard errors clustered at the county level.
$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_entry_bytype.tex")

# ================================================================ §3  brand sorting
H("§3  BRAND SORTING: does a brand's border share fall with its size?")
bs = bm.copy()
bs["log_n_sf"] = np.log(bs["n_storefronts"].clip(lower=1))
base_rate = sfm["border_dispensary"].mean()
p(f"overall storefront border rate = {base_rate:.3f}")
_, so_all = ols_cluster(bs, "border_share", ["log_n_sf"], "brand_id")
multi = bs[bs["n_storefronts"] >= 2].copy()
_, so_mul = ols_cluster(multi, "border_share", ["log_n_sf"], "brand_id")
p("  all brands:        " + fmt(so_all, "log_n_sf") + f"  (N={so_all['_N']})")
p("  multi-location:     " + fmt(so_mul, "log_n_sf") + f"  (N={so_mul['_N']})")
multi["grp"] = np.where(multi["n_storefronts"] >= 6, "large (6+)",
                np.where(multi["n_storefronts"] >= 3, "mid (3-5)", "small (2)"))
tab = multi.groupby("grp")["border_share"].agg(["mean", "count"])
sng = bs[bs["n_storefronts"] == 1]["border_share"].mean()
p(tab.to_string())
p(f"  single-storefront independents: mean border share = {sng:.3f}  (n={int((bs['n_storefronts']==1).sum())})")

with open(os.path.join(ROOT, "ext_tab_sorting.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Brand sorting: the border share of a brand's stores falls with brand size}
\label{tab:ext_sorting}
\footnotesize
\begin{tabular}{lcc}
\toprule
 & All brands & Multi-location brands \\
\midrule
""")
    fh.write(f"$\\log$(\\# New Mexico storefronts) & {so_all['log_n_sf'][0]:.3f}{st(so_all['log_n_sf'][2])} & "
             f"{so_mul['log_n_sf'][0]:.3f}{st(so_mul['log_n_sf'][2])} \\\\\n")
    fh.write(f" & ({so_all['log_n_sf'][1]:.3f}) & ({so_mul['log_n_sf'][1]:.3f}) \\\\\n")
    fh.write(f"Observations (brands) & {so_all['_N']:,} & {so_mul['_N']:,} \\\\\n")
    fh.write(r"\midrule" + "\n")
    fh.write(r"\multicolumn{3}{l}{\emph{Mean border share by brand size (multi-location brands):}} \\" + "\n")
    for gname in ["small (2)", "mid (3-5)", "large (6+)"]:
        if gname in tab.index:
            fh.write(f"\\quad {gname} stores & {tab.loc[gname,'mean']:.3f} & (n={int(tab.loc[gname,'count'])}) \\\\\n")
    fh.write(f"\\quad single-storefront independents & {sng:.3f} & (n={int((bs['n_storefronts']==1).sum())}) \\\\\n")
    fh.write(r"\midrule" + "\n")
    fh.write(f"\\quad overall storefront border rate & {base_rate:.3f} & \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.82\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Brand cross-section. The dependent variable is the share of a brand's New Mexico
storefronts that are border-exposed. Robust standard errors in parentheses. A brand whose border share
exceeds the overall storefront border rate is over-represented at the border. $^{*}p<0.10$,
$^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_sorting.tex")

# ================================================================ §4  exit by type
H("§4  STOREFRONT EXIT: border x chain interaction")
e = sfq[sfq["in_exit_sample"] == 1].copy()
e["log_age"] = np.log(e["age_q"] + 1)
e["log_rev"] = np.log(e["revenue"].clip(lower=1))
e["log_localpop"] = np.log(e["sf_localpop_25"].clip(lower=1))
e["clu"] = e["fips5"].fillna(e["zip5"])
e = e.dropna(subset=["border_dispensary", "n_disp_25mi", "chain"])
e["border_x_chain"] = e["border_dispensary"] * e["chain"]
p(f"exit sample: {len(e):,} active storefront-quarters; baseline Pr(exit) = {e['exit_next'].mean():.4f}")
_, q1 = ols_cluster(e, "exit_next", ["border_dispensary", "n_disp_25mi", "log_age", "log_rev", "log_localpop", "C(qt)"], "clu")
_, q2 = ols_cluster(e, "exit_next", ["border_dispensary", "border_x_chain", "chain", "n_disp_25mi",
                                     "log_age", "log_rev", "log_localpop", "C(qt)"], "clu")
mature = e[e["qt"] >= 8097]
_, q3 = ols_cluster(mature, "exit_next", ["border_dispensary", "border_x_chain", "chain", "n_disp_25mi",
                                          "log_age", "log_rev", "log_localpop", "C(qt)"], "clu")
p(col(q1, ["border_dispensary"]))
p("  --- + border x chain (full) ---")
p(col(q2, ["border_dispensary", "border_x_chain", "chain"]))
p("  --- + border x chain (mature 2024Q1+) ---")
p(col(q3, ["border_dispensary", "border_x_chain", "chain"]))

# also: revenue by type (does a border chain store earn more?)
H("§4b  DISPENSARY REVENUE: border x chain")
rr = sfq[(sfq["active"] == 1) & (sfq["revenue"] > 0)].copy()
rr["log_rev"] = np.log(rr["revenue"])
rr["log_localpop"] = np.log(rr["sf_localpop_25"].clip(lower=1))
rr["clu"] = rr["fips5"].fillna(rr["zip5"])
rr["border_x_chain"] = rr["border_dispensary"] * rr["chain"]
rr = rr.dropna(subset=["border_dispensary", "chain"])
_, v1 = ols_cluster(rr, "log_rev", ["border_dispensary", "border_x_chain", "chain", "log_localpop", "C(qt)"], "clu")
p(col(v1, ["border_dispensary", "border_x_chain", "chain", "log_localpop"]))

with open(os.path.join(ROOT, "ext_tab_exit_bytype.tex"), "w", encoding="utf-8") as fh:
    fh.write(r"""\begin{table}[htbp]
\centering
\caption{Storefront exit and dispensary revenue: chain versus independent at the border}
\label{tab:ext_exit}
\footnotesize
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{3}{c}{Exit next quarter (LPM)} & Log revenue \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
 & (1) & (2) full & (3) mature & (4) \\
\midrule
""")
    def row(name, key):
        cells = []
        for t in (q1, q2, q3, v1):
            cells.append(f"{t[key][0]:.3f}{st(t[key][2])}" if key in t else "")
        return f"{name} & " + " & ".join(cells) + r" \\" + "\n"
    fh.write(row(r"$\text{Border}_d$", "border_dispensary"))
    fh.write(f" & ({q1['border_dispensary'][1]:.3f}) & ({q2['border_dispensary'][1]:.3f}) & "
             f"({q3['border_dispensary'][1]:.3f}) & ({v1['border_dispensary'][1]:.3f}) \\\\\n")
    fh.write(row(r"$\text{Border}_d \times \text{Chain}_d$", "border_x_chain"))
    fh.write(row(r"$\text{Chain}_d$", "chain"))
    fh.write(r"\midrule" + "\n")
    fh.write(f"Controls (competition, age, revenue, local pop.) & Yes & Yes & Yes & Yes \\\\\n")
    fh.write(f"Quarter fixed effects & Yes & Yes & Yes & Yes \\\\\n")
    fh.write(f"Observations & {q1['_N']:,} & {q2['_N']:,} & {q3['_N']:,} & {v1['_N']:,} \\\\\n")
    fh.write(f"Baseline / mean & {e['exit_next'].mean():.3f} & {e['exit_next'].mean():.3f} & "
             f"{mature['exit_next'].mean():.3f} & {rr['log_rev'].mean():.2f} \\\\\n")
    fh.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.94\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Columns (1)--(3): linear probability models for storefront exit in the next quarter,
active New Mexico storefront-quarters, 2022Q2--2026Q2. Column (3) restricts to 2024Q1 onward. Column (4):
$\log$ quarterly dispensary revenue. All columns include $\log$(age), $\log$(own revenue) [exit models
only], $\log$(local population), dispensaries within 25 miles, and quarter fixed effects; standard errors
clustered on the nearest county. $\text{Chain}_d$ equals one if the operator runs two or more New Mexico
storefronts. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
p("wrote ext_tab_exit_bytype.tex")

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_l))
p("\nDONE. log -> ext_output/ext01_reduced_form_bytype.log")
