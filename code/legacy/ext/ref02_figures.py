"""
ref02_figures.py -- regenerate the two dynamic border-premium figures with the
clean, no-reference-quarter parametrization requested in the revision:

    y_zt = sum_q  beta_q ( Border_z x 1[quarter=q] )  +  gamma log(pop_z)  +  delta_q  +  e_zt

so each beta_q IS the quarter-q border-interior conditional difference in levels
(no omitted/differenced reference quarter). County-clustered 95% CIs.

Outputs (overwrite, same filenames the manuscript already \includegraphics):
  fig_nm_dynamic_borderpremium.png
  fig_nm_dynamic_legalcontrol.png
"""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NM = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\New Mexico project\tables"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTCOMES = [("log_revenue_per_capita", "Revenue per capita"),
            ("log_dispensary_per_capita", "Dispensary density"),
            ("log_revenue_per_dispensary", "Revenue per dispensary")]


def per_quarter_levels(d, yvar, treat="border_zip", clu="county_fips"):
    d = d.dropna(subset=[yvar, "log_population", clu]).copy()
    d["q"] = d["yq"].astype(str)
    # no main treat term, no intercept collinearity: interact treat with every quarter dummy
    f = f"{yvar} ~ C(q) + {treat}:C(q) + log_population"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[clu]})
    rows = []
    for q in sorted(d["q"].unique()):
        term = f"{treat}:C(q)[T.{q}]" if f"{treat}:C(q)[T.{q}]" in m.params.index else f"{treat}:C(q)[{q}]"
        if term not in m.params.index:
            # first level may be dropped; recover via C(q) baseline handling -> fit w/o intercept
            continue
        rows.append((q, m.params[term], m.bse[term], m.pvalues[term]))
    return pd.DataFrame(rows, columns=["quarter", "coef", "se", "p"])


def per_quarter_levels_noint(d, yvar, treat="border_zip", clu="county_fips"):
    d = d.dropna(subset=[yvar, "log_population", clu]).copy()
    d["q"] = d["yq"].astype(str)
    f = f"{yvar} ~ 0 + C(q) + {treat}:C(q) + log_population"
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[clu]})
    rows = []
    for q in sorted(d["q"].unique()):
        cands = [t for t in m.params.index if t.startswith(f"{treat}:C(q)") and f"{q}" in t]
        if not cands:
            continue
        t = cands[0]
        rows.append((q, m.params[t], m.bse[t], m.pvalues[t]))
    return pd.DataFrame(rows, columns=["quarter", "coef", "se", "p"])


def panel_fig(getdf, fname, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (yvar, lab) in zip(axes, OUTCOMES):
        ev = getdf(yvar)
        x = np.arange(len(ev))
        ax.errorbar(x, ev["coef"], yerr=1.96 * ev["se"], fmt="o-", ms=4, lw=1.1,
                    capsize=3, color="#2b6cb0")
        ax.axhline(0, color="gray", lw=.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(ev["quarter"], rotation=90, fontsize=7)
        ax.set_title(lab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Border$-$interior difference (log points)")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ROOT, fname)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---- main dynamic border premium (116-ZIP positive panel, as in the base figure) ----
cor = pd.read_csv(os.path.join(NM, "nm_zip_panel_corrected.csv"), dtype={"zip5": str, "county_fips": str})
panel_fig(lambda y: per_quarter_levels_noint(cor, y),
          "ref_fig_nm_dynamic_borderpremium.png",
          "Border$-$interior difference by quarter, New Mexico. Each point is the quarter-specific "
          "conditional\ndifference between border and interior ZIP codes (levels; no differenced reference "
          "quarter). County-clustered 95% CIs.\nThese are not event-study, pre-trend, or treatment-effect "
          "estimates.")

# ---- legal-border control version ----
lb = pd.read_csv(os.path.join(NM, "nm_zip_legalborder_stateline.csv"), dtype={"zip5": str})[["zip5", "legal_border_zip"]]
cl = cor.merge(lb, on="zip5", how="left")
cl["legal_border_zip"] = cl["legal_border_zip"].fillna(0).astype(int)
keep = (cl.border_zip == 1) | ((cl.legal_border_zip == 1) & (cl.border_zip == 0))
cl = cl[keep].copy()
panel_fig(lambda y: per_quarter_levels_noint(cl, y),
          "ref_fig_nm_dynamic_legalcontrol.png",
          "Border$-$interior difference by quarter, legal-border control group. Treated: border ZIP codes "
          "(near TX/OK/UT).\nControl: New Mexico ZIP codes near the CO/AZ line under the same 75-mile / "
          "1,000-resident rule; interior ZIP codes dropped.\nQuarter-specific conditional differences in "
          "levels; county-clustered 95% CIs.")
print("done")
