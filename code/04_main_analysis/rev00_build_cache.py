r"""
rev00_build_cache.py -- builds and caches the revision panels; verifies that the rebuilt panel reproduces
the published Table 3 (tab_fact1fact2_dynamic.tex) and Table 4 columns 1-2 (tab_fact3_dynamic.tex).
Outputs: output/cache.pkl, output/rev00_checks.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os, pickle
import numpy as np, pandas as pd
from _common import build_all, OUT, ROOT, QLAB
from _econ import fit

LOG = []
def p(s=""):
    print(s); LOG.append(str(s))

A = build_all(groups=("nonlegal", "legal", "interior"))
full = A["full"]
main = full[full.group != "interior"].copy()
p(f"main: {main.zip5.nunique()} ZIPs, {len(main)} ZIP-qtrs, {main.county_fips.nunique()} counties; "
  f"pooled: {full.zip5.nunique()} ZIPs")

def published(fname, cols):
    rows = [l for l in open(os.path.join(ROOT, "tables", "published", fname), encoding="utf-8").read().splitlines() if l[:2] == "20" and "Q" in l[:7]]
    return np.array([[float(l.split("&")[j].replace("*", "").strip()) for j in cols] for l in rows])

pub3 = published("tab_fact1fact2_dynamic.tex", (1, 3, 5))
new3 = np.column_stack([fit(main, y)["gap_b"] for y in ["y_rpc", "y_den", "y_rps"]])
p(f"[check] Table 3: max |published - rebuilt| = {np.abs(pub3 - new3).max():.4f}")
main["a_nbI"] = np.arcsinh(main.nb_I_life); main["a_nbC"] = np.arcsinh(main.nb_C_life)
pub4 = published("tab_fact3_dynamic.tex", (1, 3))
new4 = np.column_stack([fit(main, y)["gap_b"] for y in ["a_nbI", "a_nbC"]])
p(f"[check] Table 4 cols 1-2: max |published - rebuilt| = {np.abs(pub4 - new4).max():.4f}")

cov = A["cov"]
p(f"[check] own nonlegal pop within 75 mi vs zip_xwalk.nonleg_pop_75: corr = "
  f"{np.corrcoef(cov.nlpop_75, A['zx'].nonleg_pop_75)[0,1]:.3f}; ratio of means = "
  f"{cov.nlpop_75.mean() / A['zx'].nonleg_pop_75.mean():.3f}")
p(f"big NM cities (>=50k, summed ZIP pop): {cov.attrs['big_cities']}")
p(f"medical incumbents (storefronts with medical sales in April 2022): {int(A['sf'].med_incumbent.sum())}")
p(f"storefronts active in 2022Q2: {int(A['meta'].lc.sum())}; status_entry: {A['meta'].status_entry.value_counts().to_dict()}; "
  f"chain_pre storefronts: {int(A['meta'].chain_pre.sum())}")
z = full.groupby("zip5").first()
for g in ["nonlegal", "legal", "interior"]:
    s = z[z.group == g]
    p(f"[{g}] n={len(s)} nearest_nonlegal={s.nearest_nonlegal_state.value_counts().to_dict()} "
      f"nearest_state={s.nearest_state.value_counts().to_dict()}")
pickle.dump(A, open(os.path.join(OUT, "cache.pkl"), "wb"))
open(os.path.join(OUT, "rev00_checks.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
