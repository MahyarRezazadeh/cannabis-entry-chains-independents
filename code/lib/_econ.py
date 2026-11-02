r"""
_econ.py -- estimation helpers for the referee revision.
  ols_fit       OLS with county-clustered CR1 standard errors (same small-sample correction as statsmodels)
  design        Table-3 design: quarter-specific Nonlegal x quarter gaps (no omitted quarter) or one pooled gap,
                + log population + quarter FE (+ optional controls)
  wild_boot     wild cluster restricted (WCR) bootstrap-t p-value, Webb six-point weights
  conley        Conley (1999) spatial HAC: Bartlett kernel in distance across ZIP codes, arbitrary serial
                correlation within ZIP code
  ppml          Poisson pseudo-ML with county-clustered SEs
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from _common import QS, hav


def design(d, dynamic=True, controls=(), treat="nonlegal"):
    Q = d.qt.to_numpy()
    cols, names = [], []
    if dynamic:
        for q in QS:
            cols.append(d[treat].to_numpy() * (Q == q)); names.append(f"g{q}")
    else:
        cols.append(d[treat].to_numpy().astype(float)); names.append("gap")
    for q in QS:
        cols.append((Q == q).astype(float)); names.append(f"fe{q}")
    cols.append(d.log_pop.to_numpy()); names.append("log_pop")
    for c in controls:
        cols.append(d[c].to_numpy().astype(float)); names.append(c)
    X = np.column_stack(cols).astype(float)
    keep = X.std(0) > 0
    keep[[i for i, n in enumerate(names) if n.startswith("fe")]] = True
    return X, names


def ols_fit(y, X, cl, w=None):
    y = np.asarray(y, float); X = np.asarray(X, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, cl = y[ok], X[ok], np.asarray(cl)[ok]
    if w is not None:
        sw = np.sqrt(np.asarray(w, float)[ok]); y = y * sw; X = X * sw[:, None]
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    codes, cl = np.unique(cl, return_inverse=True)
    G = len(codes); n, k = X.shape
    S = np.zeros((G, k))
    np.add.at(S, cl, X * e[:, None])
    c = G / (G - 1) * (n - 1) / (n - np.linalg.matrix_rank(X))
    V = c * XtXi @ (S.T @ S) @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return dict(b=b, se=se, V=V, e=e, X=X, y=y, cl=cl, G=G, n=n, XtXi=XtXi, ok=ok)


def pval(b, se, G):
    return 2 * (1 - stats.t.cdf(np.abs(b / se), df=G - 1))


def fit(d, y, dynamic=True, controls=(), cluster="county_fips", w=None, treat="nonlegal"):
    X, names = design(d, dynamic, controls, treat)
    r = ols_fit(d[y].to_numpy(), X, d[cluster].to_numpy(), None if w is None else d[w].to_numpy())
    idx = [i for i, n in enumerate(names) if n.startswith("g")]
    r["names"] = names
    r["gap_b"], r["gap_se"] = r["b"][idx], r["se"][idx]
    r["gap_p"] = pval(r["gap_b"], r["gap_se"], r["G"])
    return r


WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def wild_boot(d, y, controls=(), cluster="county_fips", B=1999, seed=11, treat="nonlegal"):
    """WCR bootstrap-t p-value for the pooled gap coefficient (H0: gap = 0)."""
    X, names = design(d, False, controls, treat)
    yy = d[y].to_numpy().astype(float)
    ok = np.isfinite(yy) & np.isfinite(X).all(1)
    X, yy = X[ok], yy[ok]
    cl = np.unique(d[cluster].to_numpy()[ok], return_inverse=True)[1]
    G = cl.max() + 1
    full = ols_fit(yy, X, cl)
    t0 = full["b"][0] / full["se"][0]
    Xr = X[:, 1:]
    br = np.linalg.lstsq(Xr, yy, rcond=None)[0]
    fr, er = Xr @ br, yy - Xr @ br
    rng = np.random.default_rng(seed)
    XtXi = np.linalg.pinv(X.T @ X)
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - np.linalg.matrix_rank(X))
    ts = np.empty(B)
    for bb in range(B):
        wv = WEBB[rng.integers(0, 6, G)][cl]
        ys = fr + er * wv
        b = XtXi @ X.T @ ys
        e = ys - X @ b
        S = np.zeros((G, k)); np.add.at(S, cl, X * e[:, None])
        v00 = c * (XtXi @ (S.T @ S) @ XtXi)[0, 0]
        ts[bb] = b[0] / np.sqrt(v00)
    return float((np.abs(ts) >= abs(t0)).mean())


def conley(d, y, cutoff_mi, controls=(), dynamic=False, treat="nonlegal"):
    """Conley SE for the pooled gap; scores summed within ZIP over quarters (serial correlation allowed)."""
    X, names = design(d, dynamic, controls, treat)
    yy = d[y].to_numpy().astype(float)
    ok = np.isfinite(yy) & np.isfinite(X).all(1)
    dd = d[ok]
    X, yy = X[ok], yy[ok]
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ yy
    e = yy - X @ b
    zi, z = np.unique(dd.zip5.to_numpy(), return_inverse=True)
    S = np.zeros((len(zi), X.shape[1])); np.add.at(S, z, X * e[:, None])
    ll = dd.groupby("zip5")[["lat", "lon"]].first().loc[zi]
    D = hav(ll.lat.to_numpy()[:, None], ll.lon.to_numpy()[:, None], ll.lat.to_numpy()[None, :], ll.lon.to_numpy()[None, :])
    K = np.clip(1 - D / cutoff_mi, 0, None)
    V = XtXi @ (S.T @ K @ S) @ XtXi
    return b[0], float(np.sqrt(V[0, 0]))


def ppml(d, y, controls=(), dynamic=False, offset=None, cluster="county_fips", treat="nonlegal"):
    X, names = design(d, dynamic, controls, treat)
    yy = d[y].to_numpy().astype(float)
    off = None if offset is None else d[offset].to_numpy()
    m = sm.GLM(yy, X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]}, maxiter=200)
    idx = [i for i, n in enumerate(names) if n.startswith("g")]
    return m.params[idx], m.bse[idx], m.pvalues[idx]


def fmt(b, se, p, d=3):
    from _common import stars
    return f"${b:+.{d}f}^{{{stars(p)}}}$".replace("^{}", ""), f"({se:.{d}f})"
