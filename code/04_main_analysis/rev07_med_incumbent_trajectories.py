r"""
rev07_med_incumbent_trajectories.py -- do medical incumbents (storefronts reporting medical sales in
April 2022) belong to operators that are/become chains, or do they stay independent?

Operator-level trajectories, 2022Q2-2026Q2:
  stores_open = operator's active NM storefronts in 2022Q2
  stores_max  = operator's distinct NM storefronts ever active in the sample
  type at opening: multi (>=2 in 2022Q2) / single (1 in 2022Q2)
  later: expanded (added >=1 storefront after 2022Q2) / no new storefronts
Reported statewide and for operators with a medical incumbent in the 209 border ZIPs.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import os
import pandas as pd
from _common import load_zx, load_storefronts, load_store_quarters, QMIN, OUT

zx = load_zx()
sf, bm = load_storefronts(zx)
sq, sf = load_store_quarters(sf)
grp = dict(zip(zx.zip5, zx.group))
sq["group"] = sq.zip5.map(grp)

first = sq.groupby("storefront_id").qt.min()
st = sf.set_index("storefront_id").loc[first.index, ["brand_id", "med_incumbent", "zip5"]].copy()
st["faq"] = first
st["group"] = st.zip5.map(grp)

ops = st.groupby("brand_id").agg(
    stores_open=("faq", lambda s: (s == QMIN).sum()),
    stores_max=("faq", "size"),
    stores_added=("faq", lambda s: (s > QMIN).sum()),
    n_med=("med_incumbent", "sum"),
    first_q=("faq", "min"))
ops["med_op"] = ops.n_med > 0
ops["open_type"] = pd.cut(ops.stores_open, [-1, 0, 1, 1e9], labels=["not open", "single", "multi"])
ops["expanded"] = ops.stores_added > 0
ops["chain_life"] = ops.stores_max >= 2

lines = []
def p(s=""):
    print(s); lines.append(s)

def table(o, label):
    p(f"\n=== {label}: {len(o)} operators, {int(o.n_med.sum())} medical-incumbent storefronts ===")
    t = o.groupby(["open_type", "expanded"], observed=True).agg(
        operators=("n_med", "size"), med_storefronts=("n_med", "sum"),
        stores_open=("stores_open", "sum"), stores_added=("stores_added", "sum"))
    p(t.to_string())
    p(f"lifetime chain (>=2 storefronts ever): {o.chain_life.sum()} of {len(o)} operators "
      f"({o.chain_life.mean():.0%}); holding {o[o.chain_life].n_med.sum()} of {o.n_med.sum()} "
      f"med storefronts ({o[o.chain_life].n_med.sum()/o.n_med.sum():.0%})")
    sing = o[o.open_type == "single"]
    if len(sing):
        p(f"single at opening: {len(sing)}; later expanded (became chain): {sing.expanded.sum()} "
          f"({sing.expanded.mean():.0%}); stayed single: {(~sing.expanded).sum()}")
    mult = o[o.open_type == "multi"]
    if len(mult):
        p(f"multi at opening: {len(mult)}; later added stores: {mult.expanded.sum()} "
          f"({mult.expanded.mean():.0%}); stores added by them: {int(mult.stores_added.sum())}")
    p("distribution of storefronts ever (operators): " +
      o.stores_max.value_counts().sort_index().to_dict().__repr__())

med = ops[ops.med_op]
table(med, "Statewide operators with >=1 medical incumbent")

# operators with a medical incumbent located in the border sample
bord_ids = st[(st.med_incumbent == 1) & st.group.isin(["nonlegal", "legal"])].brand_id.unique()
table(med.loc[bord_ids], "Operators with a medical incumbent in the 209 border ZIPs")
for g in ["nonlegal", "legal"]:
    ids = st[(st.med_incumbent == 1) & (st.group == g)].brand_id.unique()
    table(med.loc[ids], f"Operators with a medical incumbent in {g}-border ZIPs")

# comparison: operators with no medical incumbent (all entered after opening or adult-use-only)
new = ops[~ops.med_op]
p(f"\n=== Comparison: {len(new)} operators with no medical incumbent ===")
p(f"lifetime chain: {new.chain_life.sum()} ({new.chain_life.mean():.0%}); "
  f"mean storefronts ever {new.stores_max.mean():.2f} vs {med.stores_max.mean():.2f} for med operators")
p(f"share of operators that ever add a storefront after their first: "
  f"med {(med.stores_max > med.stores_open.clip(lower=1)).mean():.0%}")

# where did post-opening expansion of medical operators go?
added = st[(st.faq > QMIN) & st.brand_id.isin(med.index)]
p(f"\nPost-2022Q2 storefronts opened by medical operators: {len(added)}; by group: "
  + added.group.value_counts().to_dict().__repr__())

with open(os.path.join(OUT, "rev07_med_incumbent_trajectories.txt"), "w") as f:
    f.write("\n".join(lines))
