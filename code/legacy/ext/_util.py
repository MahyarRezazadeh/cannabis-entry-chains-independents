"""shared helpers for the franchise/independent extension (self-contained)."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

asinh = lambda x: np.log(x + np.sqrt(np.asarray(x, float) ** 2 + 1))


def ols_cluster(df, y, rhs, cluster):
    """OLS with cluster-robust SE. rhs may include C(...) terms.
    Returns (result, tidy dict {term: (coef, se, p)} + _N,_r2,_ymean)."""
    d = df.copy()
    base = [y] + [c for c in rhs if "(" not in c] + (
        [cluster] if isinstance(cluster, str) else list(cluster))
    d = d.dropna(subset=[c for c in dict.fromkeys(base) if c in d.columns])
    formula = f"{y} ~ " + " + ".join(rhs) if rhs else f"{y} ~ 1"
    m = smf.ols(formula, data=d)
    if isinstance(cluster, str):
        res = m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    else:
        res = m.fit(cov_type="cluster", cov_kwds={"groups": d[list(cluster)]})
    tidy = {t: (res.params[t], res.bse[t], res.pvalues[t]) for t in res.params.index}
    tidy["_N"] = int(res.nobs)
    tidy["_r2"] = float(res.rsquared)
    tidy["_ymean"] = float(d[y].mean())
    return res, tidy


def poisson_hc1(df, y, rhs):
    d = df.dropna(subset=[y] + rhs).copy()
    m = sm.GLM(d[y], sm.add_constant(d[rhs]), family=sm.families.Poisson()).fit(cov_type="HC1")
    return {t: (m.params[t], m.bse[t], m.pvalues[t]) for t in m.params.index} | {"_N": int(m.nobs)}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(t, k):
    if k not in t:
        return "."
    b, se, p = t[k]
    return f"{b:.4f}{stars(p)}"


def col(tidy, terms):
    out = []
    for t in terms:
        if t in tidy:
            b, se, pv = tidy[t]
            out.append(f"  {t:34s} {b:>12.5f}{stars(pv):<3} ({se:.5f})")
        else:
            out.append(f"  {t:34s} {'.':>12s}")
    for k in ("_N", "_r2", "_ymean"):
        if k in tidy:
            out.append(f"  {k:34s} {tidy[k]:>12.4f}")
    return "\n".join(out)
