r"""
m01_estimate_validate.py -- estimate and validate the entry-and-adjustment model.

Blocks (all estimated on the 368-ZIP panel built by m00_model_data.py):
 (1) Entry: for each opening type k in {F, EM, EN}, Poisson pseudo-ML
        E[open_k,mt] = exp(kappa_k + beta_k lL_m + delta_k xB_m + mu_k xLB_m + psi_k'W_m + phi1_k log t + phi2_k t
                           - gamma_k log(1 + N_m,t-1)),   t = 1..16 (2022Q3-2026Q2)
     W = log density, distance to Albuquerque and to the nearest large city, 2019 retail establishments, tribal centroid.
 (2) Exit: per-store logit for leaving after quarter t on log(1+N_mt), xB, lL, t, log t.
 (3) Revenue: log R_mt = xi_m + a_t + sigma log N_mt on ZIP-quarters with N > 0 (ZIP and quarter FE);
     xi_m for ZIPs that never host a store is predicted from (lL, xB, xLB, W).
Specifications compared: S0 (time profile + population), S1 (+ exposure and W), S2 (+ saturation, preferred).
Validation: in-sample fit (full sample) and out-of-sample prediction of 2024Q3-2026Q2 from parameters estimated on
2022Q3-2024Q2, starting from the observed 2024Q2 stock.
Parameter uncertainty: 300 draws from the estimated sampling distribution of each block (seed 20261023).
Outputs: data/analysis/model_estimates.pkl, output/tables/tab_rio_model_estimates.tex, tab_rio_model_fit.tex,
         output/figures/fig_rio_model_fit.png, output/model_diagnostics/m01.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import pickle, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import ANALYSIS, DIAG, TABLES, FIGURES
from _common import QS, QLAB, QMIN, QMAX, stars
from _econ import ols_fit

SEED = 20261023
LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

D = pickle.load(open(ANALYSIS / "model_panel.pkl", "rb"))
P, st, SQ = D["P"].copy(), D["st"], D["SQ"].copy()
P["lNlag"] = np.log1p(P.N_lag)
P["logt"] = np.log(P.t.clip(lower=1))
P["lN0"] = np.log1p(P.zip5.map(P[P.qt == QMIN].set_index("zip5").N))   # Wooldridge (2005) initial-condition proxy
W = ["log_density", "dabq", "dbig", "zbp", "tribal_centroid"]
SPECS = {"S0": ["lL"], "S1": ["lL", "xB", "xLB"] + W, "S2": ["lL", "xB", "xLB"] + W + ["lNlag"],
         "S3": ["lL", "xB", "xLB"] + W + ["lNlag", "lN0"]}
TYPES = ("F", "EM", "EN")
TPROF = ["logt", "t"]
E = P[P.t >= 1].copy()

def X_of(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols + TPROF])

def fit_poisson(df, y, cols):
    X = X_of(df, cols)
    m = sm.GLM(df[y].to_numpy(float), X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.county_fips)[0]}, maxiter=300)
    return dict(b=m.params, V=m.cov_params(), names=["const"] + cols + TPROF, ll=m.llf, n=len(df))

def fit_exit(sq):
    sq = sq.assign(lN=np.log1p(sq.N), logt=np.log(sq.t.clip(lower=1)))
    cols = ["lN", "xB", "lL", "logt"]
    X = np.column_stack([np.ones(len(sq))] + [sq[c].to_numpy(float) for c in cols])
    m = sm.GLM(sq.exit_next.to_numpy(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(sq.county_fips)[0]})
    return dict(b=m.params, V=m.cov_params(), names=["const"] + cols, tmax=int(sq.t.max()))

def fit_revenue(df):
    d = df[(df.N > 0) & (df.R > 0)].copy()
    zs = sorted(d.zip5.unique()); qs = sorted(d.qt.unique())
    Z = (d.zip5.to_numpy()[:, None] == np.array(zs)[None, :]).astype(float)
    Qd = (d.qt.to_numpy()[:, None] == np.array(qs[1:])[None, :]).astype(float)
    X = np.column_stack([np.log(d.N.to_numpy(float)), Qd, Z])
    r = ols_fit(np.log(d.R.to_numpy(float)), X, d.county_fips.to_numpy())
    sigma, se = r["b"][0], r["se"][0]
    a_t = dict(zip(qs, np.r_[0.0, r["b"][1:len(qs)]]))
    xi = dict(zip(zs, r["b"][len(qs):]))
    resid_var = float(np.var(r["e"]))
    # project xi on ZIP covariates to predict xi for ZIPs without stores
    zc = df.groupby("zip5")[["lL", "xB", "xLB"] + W].first()
    have = zc.loc[zs]
    Xp = np.column_stack([np.ones(len(have))] + [have[c].to_numpy(float) for c in have.columns])
    g = np.linalg.lstsq(Xp, np.array([xi[z] for z in zs]), rcond=None)[0]
    Xa = np.column_stack([np.ones(len(zc))] + [zc[c].to_numpy(float) for c in zc.columns])
    xi_all = dict(zip(zc.index, Xa @ g))
    xi_all.update(xi)
    return dict(sigma=sigma, se=se, a_t=a_t, xi=xi_all, smear=np.exp(resid_var / 2), n=len(d), G=r["G"])

def estimate(Eset, SQset, Pset, spec="S2"):
    ent = {k: fit_poisson(Eset, f"open_{k}", SPECS[spec]) for k in TYPES}
    return dict(entry=ent, exit=fit_exit(SQset), rev=fit_revenue(Pset), spec=spec)

# ------------------------------------------------------------------ simulation (expected-value recursion)
ZIPS = sorted(P.zip5.unique())
Zc = P.groupby("zip5").first().loc[ZIPS]

def simulate(est, start_q, end_q, N0, ent_override=None, cap=None, shift=0, em_cf=None):
    """Expected-value path of stores, openings, exits and revenue for every ZIP from start_q+1 to end_q."""
    spec_cols = est["entry"]["F"]["names"][1:-2]
    out = []
    N = N0.astype(float).copy()
    for q in range(start_q + 1, end_q + 1):
        t = q - QMIN
        lam = {}
        for k in TYPES:
            e = (ent_override or {}).get(k, est["entry"][k])
            b = e["b"]
            def rate(tt):
                base = b[0] + sum(b[1 + i] * (np.log1p(N) if c == "lNlag" else Zc[c].to_numpy(float)) for i, c in enumerate(spec_cols))
                return np.exp(base + b[-2] * np.log(max(tt, 1)) + b[-1] * tt)
            if shift:
                lam[k] = rate(t + shift) if t > 1 else sum(rate(s) for s in range(1, 2 + shift))
            else:
                lam[k] = rate(t)
        if em_cf is not None:
            lam["EM"] = em_cf(t, N, lam)
        opens = sum(lam.values())
        if cap is not None:
            room = np.clip(cap - N, 0, None)
            scale = np.where(opens > 0, np.minimum(1.0, room / np.maximum(opens, 1e-12)), 0.0)
            lam = {k: v * scale for k, v in lam.items()}
            opens = sum(lam.values())
        xb = est["exit"]["b"]
        te = min(t - 1, est["exit"].get("tmax", 99))
        lin = xb[0] + xb[1] * np.log1p(N) + xb[2] * Zc.xB.to_numpy() + xb[3] * Zc.lL.to_numpy() + xb[4] * np.log(max(te, 1))
        ex = N / (1 + np.exp(-lin))
        N = N + opens - ex
        rv = est["rev"]
        a_t = rv["a_t"].get(q, np.mean(list(rv["a_t"].values())))
        xi = np.array([rv["xi"][z] for z in ZIPS])
        R = np.where(N > 0.05, np.exp(xi + a_t + rv["sigma"] * np.log(np.maximum(N, 1e-9))) * rv["smear"], 0.0)
        out.append(pd.DataFrame({"zip5": ZIPS, "qt": q, "N": N, "R": R, "open_F": lam["F"], "open_EM": lam["EM"],
                                 "open_EN": lam["EN"], "exits": ex}))
    return pd.concat(out)

def draw(est, rng):
    """One draw of all parameters from their estimated sampling distributions."""
    new = {"entry": {}, "spec": est["spec"]}
    for k, e in est["entry"].items():
        new["entry"][k] = dict(e, b=rng.multivariate_normal(e["b"], e["V"]))
    new["exit"] = dict(est["exit"], b=rng.multivariate_normal(est["exit"]["b"], est["exit"]["V"]))
    rv = dict(est["rev"]); rv["sigma"] = rng.normal(est["rev"]["sigma"], est["rev"]["se"])
    new["rev"] = rv
    return new

N_obs = P.pivot(index="zip5", columns="qt", values="N").loc[ZIPS]
grp = Zc.group.to_numpy()

if __name__ == "__main__":
    # ------------------------------------------------------------------ full-sample estimates
    est = {s: estimate(E, SQ, P, s) for s in SPECS}
    for s in SPECS:
        p(f"[{s}] log-likelihood: " + ", ".join(f"{k} {est[s]['entry'][k]['ll']:.1f}" for k in TYPES))
    e2 = est["S3"]
    for k in TYPES:
        b, V, nm = e2["entry"][k]["b"], e2["entry"][k]["V"], e2["entry"][k]["names"]
        se = np.sqrt(np.diag(V))
        p(f"[S3 entry {k}] " + "; ".join(f"{n} {b[i]:+.3f} ({se[i]:.3f})" for i, n in enumerate(nm)))
    xb, xs = e2["exit"]["b"], np.sqrt(np.diag(e2["exit"]["V"]))
    p("[exit logit] " + "; ".join(f"{n} {xb[i]:+.3f} ({xs[i]:.3f})" for i, n in enumerate(e2["exit"]["names"])))
    p(f"[revenue] sigma = {e2['rev']['sigma']:.3f} (SE {e2['rev']['se']:.3f}); N = {e2['rev']['n']} ZIP-quarters, {e2['rev']['G']} counties")
    # ZIP fixed-effects check on saturation (within-ZIP variation only), ZIPs with at least one opening
    ezf = E[E.groupby("zip5").open_all.transform("sum") > 0].copy()
    for k in ("F", "E"):
        y = ezf[f"open_{k}"].to_numpy(float)
        Zd = pd.get_dummies(ezf.zip5).to_numpy(float)
        Xf = np.column_stack([ezf.lNlag, ezf.logt, ezf.t, Zd])
        try:
            mf = sm.GLM(y, Xf, family=sm.families.Poisson()).fit(cov_type="cluster",
                                                                  cov_kwds={"groups": pd.factorize(ezf.county_fips)[0]}, maxiter=300)
            p(f"[ZIP FE Poisson, {k}] log(1+N_lag) {mf.params[0]:+.3f} ({mf.bse[0]:.3f}); ZIPs {ezf.zip5.nunique()}")
        except Exception as ex_:
            p(f"[ZIP FE Poisson, {k}] failed: {ex_}")

    # ------------------------------------------------------------------ in-sample fit (start from observed 2022Q2)
    sims_in = {s: simulate(est[s], QMIN, QMAX, N_obs[QMIN].to_numpy()) for s in SPECS}
    # ------------------------------------------------------------------ out-of-sample: estimate on t <= 8, predict t = 9..16
    QCUT = QMIN + 8
    est_oos = {s: estimate(E[E.t <= 8], SQ[SQ.qt <= QCUT - 1], P[P.qt <= QCUT], s) for s in SPECS}
    sims_oos = {s: simulate(est_oos[s], QCUT, QMAX, N_obs[QCUT].to_numpy()) for s in SPECS}
    obs = P.copy()

    def gtot(df, col):
        g = df.assign(group=df.zip5.map(dict(zip(ZIPS, grp))))
        return g.groupby(["group", "qt"])[col].sum().unstack(0)

    fit_rows = []
    for s in SPECS:
        so = sims_oos[s]; si = sims_in[s]
        o_late = obs[obs.qt > QCUT]
        # ZIP-level RMSE of the 2026Q2 stock, group totals, openings, revenue, exposure gradient
        n_pred = so[so.qt == QMAX].set_index("zip5").N.loc[ZIPS].to_numpy(); n_true = N_obs[QMAX].to_numpy()
        rmse = np.sqrt(np.mean((n_pred - n_true) ** 2))
        go_p, go_o = gtot(so, "open_F") + gtot(so, "open_EM") + gtot(so, "open_EN"), gtot(o_late, "open_all")
        gN_p, gN_o = gtot(so[so.qt == QMAX], "N"), gtot(obs[obs.qt == QMAX], "N")
        gR_p, gR_o = gtot(so, "R"), gtot(o_late, "R")
        # exposure gradient in simulated vs observed 2026Q2 store density (asinh per 1,000), 368 ZIPs, with log pop
        def grad(nv):
            y = np.arcsinh(nv / Zc.zip_pop.clip(lower=1).to_numpy() * 1000)
            X = np.column_stack([np.ones(len(y)), Zc.xB, Zc.lL])
            return np.linalg.lstsq(X, y, rcond=None)[0][1]
        row = dict(spec=s, rmse=rmse, open_pred=float(go_p.sum().sum()), open_obs=float(go_o.sum().sum()),
                   N_pred={g: float(gN_p[g].iloc[0]) for g in gN_p}, N_obs={g: float(gN_o[g].iloc[0]) for g in gN_o},
                   R_ratio_pred=float((gR_p["nonlegal"].sum() / Zc[grp == "nonlegal"].zip_pop.sum()) / (gR_p["legal"].sum() / Zc[grp == "legal"].zip_pop.sum())),
                   R_ratio_obs=float((gR_o["nonlegal"].sum() / Zc[grp == "nonlegal"].zip_pop.sum()) / (gR_o["legal"].sum() / Zc[grp == "legal"].zip_pop.sum())),
                   grad_pred=grad(n_pred), grad_obs=grad(n_true),
                   rmse_in=np.sqrt(np.mean((si[si.qt == QMAX].set_index("zip5").N.loc[ZIPS].to_numpy() - n_true) ** 2)))
        fit_rows.append(row)
        p(f"[fit {s}] OOS 2026Q2 ZIP RMSE {rmse:.2f} (in-sample path {row['rmse_in']:.2f}); OOS openings pred {row['open_pred']:.0f} vs obs {row['open_obs']:.0f}; "
          f"2026Q2 stock pred " + ", ".join(f"{g} {row['N_pred'][g]:.0f}/{row['N_obs'][g]:.0f}" for g in row["N_pred"]) +
          f"; NL/LB revenue-per-resident ratio pred {row['R_ratio_pred']:.2f} vs obs {row['R_ratio_obs']:.2f}; density gradient pred {row['grad_pred']:.3f} vs obs {row['grad_obs']:.3f}")
    # composition of openings and EM timing, in-sample S3
    si = sims_in["S3"]
    comp = si.groupby(si.zip5.map(dict(zip(ZIPS, grp))))[["open_F", "open_EM", "open_EN"]].sum()
    comp_o = obs[obs.t >= 1].groupby("group")[["open_F", "open_EM", "open_EN"]].sum()
    p("[composition, in-sample S3] predicted shares F/EM/EN by group: " + "; ".join(
        f"{g} {comp.loc[g,'open_F']/comp.loc[g].sum():.2f}/{comp.loc[g,'open_EM']/comp.loc[g].sum():.2f}/{comp.loc[g,'open_EN']/comp.loc[g].sum():.2f}"
        for g in comp.index) + " | observed " + "; ".join(
        f"{g} {comp_o.loc[g,'open_F']/comp_o.loc[g].sum():.2f}/{comp_o.loc[g,'open_EM']/comp_o.loc[g].sum():.2f}/{comp_o.loc[g,'open_EN']/comp_o.loc[g].sum():.2f}"
        for g in comp_o.index))
    em_q_p = si.groupby("qt").open_EM.sum(); em_q_o = obs[obs.t >= 1].groupby("qt").open_EM.sum()
    en_q_p = si.groupby("qt").open_EN.sum(); en_q_o = obs[obs.t >= 1].groupby("qt").open_EN.sum()
    def mean_t(s_):
        w = s_.to_numpy(); return float((w * (s_.index.to_numpy() - QMIN)).sum() / w.sum())
    p(f"[timing] mean quarter of EM openings: pred {mean_t(em_q_p):.2f}, obs {mean_t(em_q_o):.2f}; EN: pred {mean_t(en_q_p):.2f}, obs {mean_t(en_q_o):.2f}")
    p(f"[continuing-store check] implied elasticity of revenue per store to the store count: sigma - 1 = {e2['rev']['sigma']-1:+.3f}; "
      f"continuing-store elasticity to rivals within 2/5 km (rio03): -0.24 / -0.32")

    # ------------------------------------------------------------------ figure: observed vs predicted
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for j, g in enumerate(["nonlegal", "legal", "interior"]):
        a = ax[j]
        o = obs[obs.group == g].groupby("qt").N.sum()
        a.plot(range(len(QS)), o.loc[QS].to_numpy(), "k-o", ms=3, label="Observed")
        si_g = si[si.zip5.map(dict(zip(ZIPS, grp))) == g].groupby("qt").N.sum()
        a.plot(range(1, len(QS)), si_g.loc[QS[1:]].to_numpy(), "-", color="#1f4e79", label="Model, in-sample")
        so = sims_oos["S3"]; so_g = so[so.zip5.map(dict(zip(ZIPS, grp))) == g].groupby("qt").N.sum()
        a.plot(range(9, len(QS)), so_g.loc[QS[9:]].to_numpy(), "--", color="#c0504d", label="Model, out-of-sample")
        a.axvline(8, color="grey", lw=0.7, ls=":")
        a.set_title({"nonlegal": "A. Nonlegal-border ZIPs", "legal": "B. Always-legal-border ZIPs", "interior": "C. Interior ZIPs"}[g], fontsize=10)
        a.set_xticks(range(0, len(QS), 2)); a.set_xticklabels(QLAB[::2], rotation=45, ha="right", fontsize=8); a.grid(alpha=0.3)
    ax[0].set_ylabel("Operating dispensaries"); ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_rio_model_fit.png", dpi=600)

    # ------------------------------------------------------------------ tables
    def c(b, s):
        pv = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(b / s)))
        return f"${b:+.3f}^{{{stars(pv)}}}$".replace("^{}", ""), f"({s:.3f})"
    lab = {"const": "Constant", "lL": "Log resident population", "xB": r"Log nonlegal-state residents within 75 mi, $\log(1+B/1000)$",
           "xLB": "Log legal-state residents within 75 mi", "log_density": "Log population density", "dabq": "Distance to Albuquerque (100 mi)",
           "dbig": "Distance to nearest city $\\geq$50k (100 mi)", "zbp": "Retail establishments per 1,000, 2019 (asinh)",
           "tribal_centroid": "Centroid on tribal land", "lNlag": "Log(1 + operating stores), previous quarter", "lN0": "Log(1 + stores in 2022Q2), initial-condition proxy", "logt": "Log quarters since opening", "t": "Quarters since opening"}
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & \multicolumn{3}{c}{Openings (Poisson)} \\", r"\cmidrule(lr){2-4}",
         r" & First location & Expansion, medical & Expansion, other \\", r" & of an operator & incumbent operator & operator \\", r"\midrule"]
    nm = e2["entry"]["F"]["names"]
    for i, n in enumerate(nm):
        if n == "const":
            continue
        cc = [c(e2["entry"][k]["b"][i], np.sqrt(e2["entry"][k]["V"][i, i])) for k in TYPES]
        L += [f"{lab[n]} & " + " & ".join(x[0] for x in cc) + r" \\", " & " + " & ".join(x[1] for x in cc) + r" \\"]
    L += [r"\midrule", "Openings & " + " & ".join(f"{int(E[f'open_{k}'].sum())}" for k in TYPES) + r" \\",
          f"ZIP-quarters & \\multicolumn{{3}}{{c}}{{{len(E):,} (368 ZIPs, 2022Q3--2026Q2)}} \\\\", r"\midrule",
          r"\multicolumn{4}{@{}l}{\textit{Exit (logit, store-quarters) and revenue}} \\"]
    for i, n in enumerate(e2["exit"]["names"]):
        if n == "const":
            continue
        a_, b_ = c(e2["exit"]["b"][i], np.sqrt(e2["exit"]["V"][i, i]))
        nn = {"lN": "Exit: log(1 + operating stores in ZIP)", "xB": "Exit: log nonlegal-state residents", "lL": "Exit: log population",
              "logt": "Exit: log quarters since opening"}[n]
        L += [f"{nn} & {a_} & & \\\\", f" & {b_} & & \\\\"]
    a_, b_ = c(e2["rev"]["sigma"], e2["rev"]["se"])
    L += [f"Revenue: elasticity of market revenue to the store count, $\\sigma$ & {a_} & & \\\\", f" & {b_} & & \\\\",
          r"\bottomrule", r"\end{tabular}"]
    open(TABLES / "tab_rio_model_estimates.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & S0: population & S1: + exposure & S2: + lagged & S3: + initial \\", r" & and time profile & and covariates & stock & stock (preferred) \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\textit{Out-of-sample: estimated on 2022Q3--2024Q2, predicting 2024Q3--2026Q2}} \\"]
    f = {r["spec"]: r for r in fit_rows}
    L += ["RMSE of ZIP-level stock in 2026Q2 & " + " & ".join(f"{f[s]['rmse']:.2f}" for s in SPECS) + r" \\",
          f"Openings, 2024Q3--2026Q2 (observed {f['S2']['open_obs']:.0f}) & " + " & ".join(f"{f[s]['open_pred']:.0f}" for s in SPECS) + r" \\"]
    for g, gl in [("nonlegal", "nonlegal border"), ("legal", "always-legal border"), ("interior", "interior")]:
        L.append(f"Stores in 2026Q2, {gl} (observed {f['S2']['N_obs'][g]:.0f}) & " + " & ".join(f"{f[s]['N_pred'][g]:.0f}" for s in SPECS) + r" \\")
    L += [f"Nonlegal/always-legal revenue per resident (observed {f['S2']['R_ratio_obs']:.2f}) & " + " & ".join(f"{f[s]['R_ratio_pred']:.2f}" for s in SPECS) + r" \\",
          f"Exposure gradient of 2026Q2 store density (observed {f['S2']['grad_obs']:.3f}) & " + " & ".join(f"{f[s]['grad_pred']:.3f}" for s in SPECS) + r" \\",
          r"\addlinespace\multicolumn{5}{@{}l}{\textit{In-sample path from the observed 2022Q2 stock}} \\",
          "RMSE of ZIP-level stock in 2026Q2 & " + " & ".join(f"{f[s]['rmse_in']:.2f}" for s in SPECS) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    open(TABLES / "tab_rio_model_fit.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

    pickle.dump(dict(est=est, est_oos=est_oos, fit_rows=fit_rows), open(ANALYSIS / "model_estimates.pkl", "wb"))
    open(DIAG / "m01.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
