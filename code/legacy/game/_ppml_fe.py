r"""
_ppml_fe.py -- sparse two-way fixed-effects Poisson (PPML) via IRLS.

statsmodels' GLM does not accept a scipy-sparse design matrix for Poisson IRLS at the scale needed
here (operator FE alone is 600+ levels; operator x quarter is 5,000+), and dense dummy-encoding a
500k-row design matrix is not memory/time-feasible on this machine. This module implements Poisson
IRLS directly on a sparse design matrix: high-dimensional fixed effects are sparse dummy blocks,
continuous regressors are dense columns, weighted normal equations X'WX / X'Wz are formed via sparse
matrix products (X'WX ends up a small dense matrix -- #regressors x #regressors -- which is solved
directly each iteration). Standard errors are cluster-robust (sandwich) by a user-specified cluster
variable, computed via the same sparse machinery.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def build_sparse_design(df, fe_cols, cont_cols, drop_one_level_of=None):
    """fe_cols: list of column names, each turned into a full dummy block (no reference level dropped)
    EXCEPT the column named in drop_one_level_of, which drops its first level (for identification --
    the sum of any one FE block's dummies equals 1 in every row, so with >=2 FE blocks one level must
    be dropped to avoid rank deficiency; continuous columns are included as-is, no intercept added)."""
    blocks = []
    names = []
    for c in fe_cols:
        cats = pd.Categorical(df[c])
        codes = cats.codes
        cats_list = list(cats.categories)
        if c == drop_one_level_of:
            keep = codes != 0
            codes_kept = np.where(keep, codes - 1, -1)
            n = len(cats_list) - 1
            row = np.arange(len(df))[codes_kept >= 0]
            col = codes_kept[codes_kept >= 0]
            block = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(df), n))
            names += [f"{c}={v}" for v in cats_list[1:]]
        else:
            n = len(cats_list)
            row = np.arange(len(df))
            col = codes
            block = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(df), n))
            names += [f"{c}={v}" for v in cats_list]
        blocks.append(block)
    for c in cont_cols:
        blocks.append(sp.csr_matrix(df[[c]].values.astype(float)))
        names.append(c)
    X = sp.hstack(blocks, format="csr")
    return X, names


def poisson_irls_sparse(X, y, max_iter=50, tol=1e-8):
    n, k = X.shape
    y = np.asarray(y, dtype=float)
    mu = np.clip(y + 0.1, 0.1, None)
    beta = np.zeros(k)
    Xd = X  # csr
    for it in range(max_iter):
        eta = np.asarray(Xd @ beta).ravel()
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-8, 1e8)
        W = mu  # Poisson IRLS weight = mu (canonical link)
        z = eta + (y - mu) / mu
        sw = np.sqrt(W)
        Xw = Xd.multiply(sw[:, None]).tocsr()
        XtWX = (Xw.T @ Xw).toarray()
        XtWz = np.asarray(Xw.T @ (sw * z)).ravel()
        XtWX += 1e-10 * np.eye(k)   # tiny ridge for numerical stability only
        beta_new = np.linalg.solve(XtWX, XtWz)
        d = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if d < tol:
            break
    eta = np.clip(np.asarray(Xd @ beta).ravel(), -30, 30)
    mu = np.exp(eta)
    return beta, mu, it + 1, d


def cluster_robust_se(X, y, beta, mu, cluster):
    """Sandwich SEs clustered on `cluster` (array of length n). Bread = (X'WX)^-1 at convergence
    (W = mu, canonical Poisson link); meat = sum over clusters of (X_g' u_g)(X_g' u_g)', u_g = score
    contributions (y-mu) for cluster g (rows)."""
    n, k = X.shape
    W = mu
    sw = np.sqrt(W)
    Xw = X.multiply(sw[:, None]).tocsr()
    XtWX = (Xw.T @ Xw).toarray() + 1e-10 * np.eye(k)
    bread = np.linalg.inv(XtWX)
    resid = (y - mu)
    # score contribution per row: X_i * resid_i  (since d loglik/d beta = sum X_i (y_i - mu_i))
    Xr = X.multiply(resid[:, None]).tocsr()
    clusters = pd.unique(cluster)
    meat = np.zeros((k, k))
    cl_arr = np.asarray(cluster)
    for g in clusters:
        idx = np.where(cl_arr == g)[0]
        sg = np.asarray(Xr[idx].sum(axis=0)).ravel()
        meat += np.outer(sg, sg)
    V = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return se, V
