"""shared helpers -- entry game project (self-contained, no external project deps)"""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

R_EARTH_MI = 3958.8
def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R_EARTH_MI * 2 * np.arcsin(np.sqrt(a))

asinh = lambda x: np.log(x + np.sqrt(np.asarray(x, float)**2 + 1))

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def ols_cluster(df, y, rhs, cluster, absorb=None, weights=None):
    d = df.copy()
    need = [y] + list(rhs) + ([cluster] if isinstance(cluster, str) else list(cluster)) \
           + (absorb or []) + ([weights] if weights else [])
    d = d.dropna(subset=[c for c in dict.fromkeys(need) if c in d.columns])
    if absorb:
        num = [y] + [c for c in rhs if pd.api.types.is_numeric_dtype(d[c])]
        for gcol in absorb:
            d[num] = d[num] - d.groupby(gcol)[num].transform("mean")
        d[num] = d[num] + df[num].mean()
    f = f"{y} ~ " + " + ".join(rhs) if rhs else f"{y} ~ 1"
    kw = {"weights": d[weights]} if weights else {}
    m = smf.ols(f, data=d, **kw)
    res = m.fit(cov_type="cluster",
                cov_kwds={"groups": d[cluster] if isinstance(cluster, str) else d[list(cluster)]})
    tidy = {t: (res.params[t], res.bse[t], res.pvalues[t]) for t in res.params.index}
    tidy.update(_N=int(res.nobs), _r2=float(res.rsquared), _ymean=float(d[y].mean()))
    return res, tidy

def panel_ols(df, y, rhs, entity, time, cluster_col=None, entity_effects=True, time_effects=True):
    from linearmodels.panel import PanelOLS
    need = [y] + list(rhs) + [entity, time] + ([cluster_col] if cluster_col else [])
    d = df.dropna(subset=[c for c in dict.fromkeys(need)]).copy()
    d["_e"] = d[entity].astype("category").cat.codes
    d["_t"] = d[time].astype("category").cat.codes
    dd = d.set_index(["_e", "_t"])
    mod = PanelOLS(dd[y], dd[list(rhs)], entity_effects=entity_effects, time_effects=time_effects)
    if cluster_col and cluster_col != entity:
        res = mod.fit(cov_type="clustered", clusters=dd[cluster_col].astype("category").cat.codes)
    else:
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    tidy = {t: (res.params[t], res.std_errors[t], res.pvalues[t]) for t in res.params.index}
    tidy.update(_N=int(res.nobs), _r2=float(res.rsquared), _ymean=float(dd[y].mean()))
    return res, tidy

def fmt(tidy, terms):
    out = []
    for t in terms:
        if t in tidy:
            b, se, pv = tidy[t]
            out.append(f"  {t:34s} {b:>13.5f}{stars(pv):<3} ({se:.5f})")
        else:
            out.append(f"  {t:34s} {'.':>13s}")
    out += [f"  {'N':34s} {tidy['_N']:>13d}", f"  {'R2':34s} {tidy['_r2']:>13.4f}",
            f"  {'mean(y)':34s} {tidy['_ymean']:>13.5f}"]
    return "\n".join(out)

def diff_means(df, group, cols):
    from scipy import stats
    g0, g1 = df[df[group] == 0], df[df[group] == 1]
    rows = []
    for c in cols:
        a, b = g1[c].dropna(), g0[c].dropna()
        if len(a) < 2 or len(b) < 2:
            rows.append((c, a.mean(), b.mean(), np.nan, np.nan)); continue
        _, pv = stats.ttest_ind(a, b, equal_var=False)
        rows.append((c, a.mean(), b.mean(), a.mean() - b.mean(), pv))
    return pd.DataFrame(rows, columns=["var", "border_mean", "interior_mean", "diff", "p"])
