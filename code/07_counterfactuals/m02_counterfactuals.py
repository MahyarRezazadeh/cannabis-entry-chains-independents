r"""
m02_counterfactuals.py -- counterfactual simulations of the entry-and-adjustment model.

Two versions of the entry block bracket the unidentified effect of the existing stock on later openings:
  A  preferred specification S3 (pooled, with the Wooldridge initial-condition proxy): later openings rise with the
     existing stock (gamma > 0; persistent attractiveness is only partly absorbed by the proxy),
  B  S3 with the stock coefficient fixed at the within-ZIP (ZIP fixed effects) Poisson estimate (gamma < 0,
     biased toward crowd-out in short panels); remaining parameters re-estimated with that offset.
Scenarios, each simulated from 2022Q2 to 2026Q2 and compared with the model's own baseline path:
  CF1a  no inherited medical outlets: medical-incumbent storefronts removed from the 2022Q2 stock;
  CF1b  CF1a and expansions by medical-incumbent operators follow the timing of other established operators
        (same total volume as in the baseline, redistributed over time);
  CF2   shorter entry delays: the opening schedule runs one (or two) quarters ahead;
  CF3   hypothetical cap on stores per ZIP code, c stores per 1,000 residents (at least one), existing stores
        grandfathered, c in {0.10, 0.19, 0.30} -- an assumption-dependent scenario, not an estimate of any real regime.
Uncertainty: 300 draws of all parameters from their estimated sampling distributions (seed 20261023); intervals are
the 5th and 95th percentiles of the scenario-minus-baseline differences.
Outputs: output/tables/tab_rio_counterfactuals.tex, output/figures/fig_rio_counterfactuals.png,
         output/model_diagnostics/m02.txt, data/analysis/counterfactuals.pkl
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "06_entry_model"))
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import ANALYSIS, DIAG, TABLES, FIGURES
import m01_estimate_validate as M

SEED, NDRAW = 20261023, 300
LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

P, E, SQ, ZIPS, Zc = M.P, M.E, M.SQ, M.ZIPS, M.Zc
grp = M.grp
QMIN, QMAX, QS = M.QMIN, M.QMAX, M.QS
estA = M.estimate(E, SQ, P, "S3")

# version B: stock coefficient fixed at the within-ZIP estimate
ezf = E[E.groupby("zip5").open_all.transform("sum") > 0]
gfe = {}
for k, ycol in [("F", "open_F"), ("E", "open_E")]:
    Zd = pd.get_dummies(ezf.zip5).to_numpy(float)
    Xf = np.column_stack([ezf.lNlag, ezf.logt, ezf.t, Zd])
    gfe[k] = sm.GLM(ezf[ycol].to_numpy(float), Xf, family=sm.families.Poisson()).fit(maxiter=300).params[0]
estB = {"entry": {}, "exit": estA["exit"], "rev": estA["rev"], "spec": "S3-FE"}
cols = M.SPECS["S3"]
j = cols.index("lNlag")
for k in M.TYPES:
    g = gfe["F"] if k == "F" else gfe["E"]
    keep = [c for c in cols if c != "lNlag"]
    X = M.X_of(E, keep)
    off = g * E.lNlag.to_numpy()
    m = sm.GLM(E[f"open_{k}"].to_numpy(float), X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(E.county_fips)[0]}, maxiter=300)
    b = np.insert(m.params, 1 + j, g)
    V = np.insert(np.insert(m.cov_params(), 1 + j, 0.0, axis=0), 1 + j, 0.0, axis=1)
    estB["entry"][k] = dict(b=b, V=V, names=["const"] + cols + M.TPROF)
p(f"within-ZIP stock coefficients used in version B: F {gfe['F']:+.3f}, expansions {gfe['E']:+.3f}")

N0 = M.N_obs[QMIN].to_numpy()
med0 = M.st[(M.st.type == "initial") & (M.st.med == 1)].groupby("zip5").size().reindex(ZIPS).fillna(0).to_numpy()
pop = Zc.zip_pop.clip(lower=1).to_numpy()
masks = {"all": np.ones(len(ZIPS), bool), "nonlegal": grp == "nonlegal", "legal": grp == "legal", "interior": grp == "interior"}

def summarize(sim, N0_):
    piv = sim.pivot(index="zip5", columns="qt", values="N").loc[ZIPS]
    Rp = sim.pivot(index="zip5", columns="qt", values="R").loc[ZIPS]
    Nfull = np.column_stack([N0_, piv.to_numpy()])
    out = {}
    for g, mk in masks.items():
        out[f"storeq_{g}"] = float(Nfull[mk].sum())
        out[f"rev_{g}"] = float(Rp.to_numpy()[mk].sum())
        out[f"avail_early_{g}"] = float((Nfull[mk][:, :4] >= 0.5).mean())
    out["storeq_early_all"] = float(Nfull[:, :4].sum())                       # 2022Q2-2023Q1
    out["rev_early_all"] = float(Rp.to_numpy()[:, :3].sum())                   # 2022Q3-2023Q1
    out["storeq_early_nonlegal"] = float(Nfull[masks["nonlegal"]][:, :4].sum())
    out["openings"] = float(sim[["open_F", "open_EM", "open_EN"]].sum().sum())
    out["open_F"] = float(sim.open_F.sum()); out["open_E"] = float((sim.open_EM + sim.open_EN).sum())
    out["rev_per_store"] = out["rev_all"] / max(out["storeq_all"] - N0_.sum(), 1e-9)
    # NL vs LB decomposition of the revenue-per-resident gap (pooled 2022Q3-2026Q2)
    nl, lb = masks["nonlegal"], masks["legal"]
    Rn, Rl = Rp.to_numpy()[nl].sum(), Rp.to_numpy()[lb].sum()
    Nn, Nl = piv.to_numpy()[nl].sum(), piv.to_numpy()[lb].sum()
    Pn, Pl = pop[nl].sum(), pop[lb].sum()
    tot = np.log(Rn / Pn) - np.log(Rl / Pl); den = np.log(Nn / Pn) - np.log(Nl / Pl)
    out["gap_total"], out["gap_density_share"] = tot, den / tot
    return out

def scenarios(est):
    base = M.simulate(est, QMIN, QMAX, N0)
    res = {"baseline": summarize(base, N0)}
    N0a = np.clip(N0 - med0, 0, None)
    res["CF1a"] = summarize(M.simulate(est, QMIN, QMAX, N0a), N0a)
    # CF1b: medical-incumbent operators' expansions take the timing of other established operators
    tot_em = base.open_EM.sum(); tot_en = base.open_EN.sum()
    cfac = tot_em / max(tot_en, 1e-9)
    res["CF1b"] = summarize(M.simulate(est, QMIN, QMAX, N0a, em_cf=lambda t, N, lam: lam["EN"] * cfac), N0a)
    res["CF2_1"] = summarize(M.simulate(est, QMIN, QMAX, N0, shift=1), N0)
    res["CF2_2"] = summarize(M.simulate(est, QMIN, QMAX, N0, shift=2), N0)
    for c in (0.10, 0.19, 0.30):
        cap = np.maximum(1.0, np.maximum(np.floor(c * pop / 1000), N0))      # grandfather existing stores
        res[f"CF3_{c:.2f}"] = summarize(M.simulate(est, QMIN, QMAX, N0, cap=cap), N0)
    return res

out = {}
for ver, est in [("A", estA), ("B", estB)]:
    point = scenarios(est)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(NDRAW):
        draws.append(scenarios(M.draw(est, rng)))
    out[ver] = dict(point=point, draws=draws)
    b = point["baseline"]
    p(f"\n[version {ver}] baseline: store-quarters {b['storeq_all']:.0f}, openings {b['openings']:.0f}, revenue ${b['rev_all']/1e9:.2f}bn, "
      f"NL/LB gap {b['gap_total']:.2f}, density share {b['gap_density_share']:.2f}")
    for s in [k for k in point if k != "baseline"]:
        d_ = np.array([[dr[s][m] - dr["baseline"][m] for m in ("storeq_all", "rev_all", "openings", "avail_early_all", "storeq_early_all")] for dr in draws])
        pt = [point[s][m] - b[m] for m in ("storeq_all", "rev_all", "openings", "avail_early_all", "storeq_early_all")]
        lo, hi = np.percentile(d_, 5, axis=0), np.percentile(d_, 95, axis=0)
        p(f"  {s:9s} d store-quarters {pt[0]:+8.0f} [{lo[0]:+.0f},{hi[0]:+.0f}] ({pt[0]/b['storeq_all']:+.1%}) | d revenue ${pt[1]/1e6:+8.0f}m "
          f"[{lo[1]/1e6:+.0f},{hi[1]/1e6:+.0f}] ({pt[1]/b['rev_all']:+.1%}) | d openings {pt[2]:+6.0f} [{lo[2]:+.0f},{hi[2]:+.0f}] | "
          f"d early availability {pt[3]:+.3f} | d first-year store-quarters {pt[4]:+.0f} [{lo[4]:+.0f},{hi[4]:+.0f}] ({pt[4]/b['storeq_early_all']:+.1%}) | rev/store {point[s]['rev_per_store']/b['rev_per_store']-1:+.1%} | density share {point[s]['gap_density_share']:.2f}")

pickle.dump(out, open(ANALYSIS / "counterfactuals.pkl", "wb"))

# ------------------------------------------------------------------ table
labels = [("CF1a", "No inherited medical outlets"), ("CF1b", "No inherited outlets and no timing advantage"),
          ("CF2_1", "Entry schedule one quarter faster"), ("CF2_2", "Entry schedule two quarters faster"),
          ("CF3_0.30", "Cap: 0.30 stores per 1,000 residents"), ("CF3_0.19", "Cap: 0.19 stores per 1,000 residents"),
          ("CF3_0.10", "Cap: 0.10 stores per 1,000 residents")]
def pct_cell(ver, s, m):
    b = out[ver]["point"]["baseline"][m]; pt = out[ver]["point"][s][m] / b - 1
    d_ = np.array([dr[s][m] / dr["baseline"][m] - 1 for dr in out[ver]["draws"]])
    lo, hi = np.percentile(d_, [5, 95])
    return f"{100*pt:+.1f}", f"[{100*lo:+.1f}, {100*hi:+.1f}]"
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{A: stock raises later openings} & \multicolumn{3}{c}{B: stock lowers later openings} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Scenario (percent change vs.\ model baseline) & Store-quarters & Revenue & Rev./store & Store-quarters & Revenue & Rev./store \\", r"\midrule"]
for s, lab in labels:
    cells = []
    for ver in ("A", "B"):
        for m in ("storeq_all", "rev_all", "rev_per_store"):
            cells.append(pct_cell(ver, s, m))
    L += [lab + " & " + " & ".join(c[0] for c in cells) + r" \\", " & " + " & ".join(c[1] for c in cells) + r" \\"]
L += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{First year only (store-quarters 2022Q2--2023Q1; revenue 2022Q3--2023Q1)}} \\"]
for s, lab in labels[:4]:
    cells = []
    for ver in ("A", "B"):
        for m in ("storeq_early_all", "rev_early_all"):
            cells.append(pct_cell(ver, s, m))
        cells.append(("", ""))
    L += [lab + " & " + " & ".join(c[0] for c in cells) + r" \\", " & " + " & ".join(c[1] for c in cells) + r" \\"]
L += [r"\midrule"]
for ver in ("A", "B"):
    b = out[ver]["point"]["baseline"]
    L.append(f"Baseline ({ver}): store-quarters {b['storeq_all']:,.0f}; revenue \\${b['rev_all']/1e9:.2f} billion; "
             f"density share of the NL--LB gap {b['gap_density_share']:.2f} & & & & & & \\\\".replace("& & & & & & \\\\", r"\\"))
L += [r"\bottomrule", r"\end{tabular}"]
# simpler, robust row construction for the baseline lines
L = [l if not l.startswith("Baseline") else l.split(" & ")[0] for l in L]
L = [ (l + r" \\") if l.startswith("Baseline") and not l.endswith(r"\\") else l for l in L]
L = [ (r"\multicolumn{7}{@{}l}{" + l[:-3] + r"} \\") if l.startswith("Baseline") else l for l in L]
open(TABLES / "tab_rio_counterfactuals.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ figure: baseline vs CF paths (version A and B)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for a, ver, ttl in [(ax[0], "A", "A. Existing stock raises later openings"), (ax[1], "B", "B. Existing stock lowers later openings")]:
    est = estA if ver == "A" else estB
    N0a = np.clip(N0 - med0, 0, None)
    paths = {"Baseline": M.simulate(est, QMIN, QMAX, N0), "No inherited medical outlets": M.simulate(est, QMIN, QMAX, N0a),
             "Schedule one quarter faster": M.simulate(est, QMIN, QMAX, N0, shift=1),
             "Cap 0.19 per 1,000": M.simulate(est, QMIN, QMAX, N0, cap=np.maximum(1.0, np.maximum(np.floor(0.19 * pop / 1000), N0)))}
    sty = {"Baseline": ("k", "-"), "No inherited medical outlets": ("#c0504d", "--"), "Schedule one quarter faster": ("#1f4e79", "-."),
           "Cap 0.19 per 1,000": ("#9bbb59", ":")}
    for lab, sim in paths.items():
        start = N0a.sum() if lab == "No inherited medical outlets" else N0.sum()
        y = np.r_[start, sim.groupby("qt").N.sum().loc[QS[1:]].to_numpy()]
        a.plot(range(len(QS)), y, color=sty[lab][0], ls=sty[lab][1], label=lab)
    a.set_xticks(range(0, len(QS), 2)); a.set_xticklabels(M.QLAB[::2], rotation=45, ha="right", fontsize=8)
    a.set_title(ttl, fontsize=10); a.grid(alpha=0.3)
ax[0].set_ylabel("Operating dispensaries, New Mexico"); ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(FIGURES / "fig_rio_counterfactuals.png", dpi=600)
open(DIAG / "m02.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
