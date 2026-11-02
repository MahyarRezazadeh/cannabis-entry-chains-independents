r"""
10_cannibalization_diagnostics.py -- structural_model / extension of venous4 / cannibalization ext.

Part 3 of the brief: summarize each Type-C entrant's OWN network immediately before entry, BEFORE
touching the structural model. This is the diagnostic that decides whether own-network overlap is
common enough among Type-C entrants to support a cannibalization mechanism story at all.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "structural_model")
LOG = os.path.join(SM, "logs", "10_cannibalization_diagnostics.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

d = pd.read_csv(os.path.join(SM, "data", "typeC_entry_H.csv"))
p(f"Type-C entry events: {len(d)}")

H("1.  OVERALL DISTRIBUTION")
p(f"H_same        : mean {d.H_same.mean():.3f}  median {d.H_same.median():.1f}  "
  f"share=0: {(d.H_same==0).mean():.3f}  max {d.H_same.max():.0f}")
p(f"AnySame=1 share: {d.AnySame.mean():.3f}")
p(f"H_size        : mean {d.H_size.mean():.3f}  median {d.H_size.median():.1f}")
p(f"H_border      : mean {d.H_border.mean():.3f}  median {d.H_border.median():.1f}")
p(f"H_B           : mean {d.H_B.mean():.3f}  median {d.H_B.median():.3f}")
gv = d.H_geo.dropna()
p(f"H_geo (n={len(gv)}, NaN where H_size=0): mean {gv.mean():.3f}  median {gv.median():.3f}")
p(f"NearestOwnDistance_mi (n={d.NearestOwnDistance_mi.notna().sum()}): "
  f"mean {d.NearestOwnDistance_mi.mean():.1f}  median {d.NearestOwnDistance_mi.median():.1f}")
for r in [25, 50, 75, 100]:
    p(f"share with an own store within {r}mi: {(d[f'H{r}']>0).mean():.3f}")

H("2.  BY BORDER STATUS OF THE CANDIDATE (ENTRY) MARKET")
for lab, m in [("nonlegal-border", d.border_town == 1), ("legal-border", d.legal_border_town == 1),
               ("interior", (d.border_town == 0) & (d.legal_border_town == 0))]:
    s = d[m]
    if len(s) == 0:
        continue
    p(f"  {lab:16s} n={len(s):3d}  H_same mean {s.H_same.mean():.2f}  AnySame {s.AnySame.mean():.3f}  "
      f"H_size mean {s.H_size.mean():.2f}  within25mi {(s.H25>0).mean():.3f}  "
      f"within75mi {(s.H75>0).mean():.3f}")

H("3.  CATEGORIZATION OF EACH TYPE-C ENTRY")
def categorize(r):
    if r.H_same > 0:
        return "1. same-town existing store"
    if r.H75 > 0:  # nearby (<=75mi, the project's own preferred exposure radius) but not same town
        return "2. nearby (<=75mi) but different town"
    if r.H_size > 0:
        return "3. elsewhere in NM only (>75mi)"
    return "4. no active network at entry (qualifying sibling since exited)"
d["category"] = d.apply(categorize, axis=1)
cat_tab = d["category"].value_counts().sort_index()
p(cat_tab.to_string())
p("\nas share of all 370:")
p((cat_tab / len(d)).round(3).to_string())

H("4.  IS OWN-NETWORK OVERLAP COMMON ENOUGH TO SUPPORT A CANNIBALIZATION STORY?")
p(f"AnySame (same-town existing store) = {d.AnySame.mean():.1%} of Type-C entries.")
p(f"Own store within 75mi = {(d.H75>0).mean():.1%} of Type-C entries.")
verdict = ("SUBSTANTIAL -- a majority of Type-C entrants have same-town or nearby own-network overlap"
           if d.AnySame.mean() > 0.4 else
           "MODEST -- same-town overlap is a minority but not negligible" if d.AnySame.mean() > 0.15 else
           "SPARSE -- same-town overlap is rare; geographic cannibalization cannot be the primary story")
p(f"==> {verdict}")

# ---------------------------------------------------------------------------
H("TABLES / FIGURES")
tab = pd.DataFrame({
    "stat": ["N Type-C entries", r"Mean $H_{same}$", r"Median $H_{same}$", "Share AnySame=1",
             r"Mean $H_{size}$", r"Median $H_{size}$", r"Mean $H_{border}$", r"Median $H_{border}$",
             r"Mean $H_B$", r"Median $H_B$", "Mean NearestOwnDistance (mi)", "Median NearestOwnDistance (mi)",
             r"Share w/ own store $\leq$25mi", r"Share w/ own store $\leq$50mi",
             r"Share w/ own store $\leq$75mi", r"Share w/ own store $\leq$100mi"],
    "all": [len(d), d.H_same.mean(), d.H_same.median(), d.AnySame.mean(),
            d.H_size.mean(), d.H_size.median(), d.H_border.mean(), d.H_border.median(),
            d.H_B.mean(), d.H_B.median(), d.NearestOwnDistance_mi.mean(), d.NearestOwnDistance_mi.median(),
            (d.H25>0).mean(), (d.H50>0).mean(), (d.H75>0).mean(), (d.H100>0).mean()],
})
for lab, m in [("nonlegal_border", d.border_town == 1), ("interior", (d.border_town == 0) & (d.legal_border_town==0))]:
    s = d[m]
    tab[lab] = [len(s), s.H_same.mean(), s.H_same.median(), s.AnySame.mean(),
                s.H_size.mean(), s.H_size.median(), s.H_border.mean(), s.H_border.median(),
                s.H_B.mean(), s.H_B.median(), s.NearestOwnDistance_mi.mean(), s.NearestOwnDistance_mi.median(),
                (s.H25>0).mean(), (s.H50>0).mean(), (s.H75>0).mean(), (s.H100>0).mean()]
tab.to_csv(os.path.join(SM, "output", "cannibalization_diagnostics.csv"), index=False)

def fmt(v, i):
    return f"{v:,.0f}" if i == 0 else f"{v:.3f}"
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Statistic & All & Nonlegal-border & Interior \\", r"\midrule"]
for i, row in tab.iterrows():
    lines.append(f"{row['stat']} & {fmt(row['all'],i)} & {fmt(row['nonlegal_border'],i)} & {fmt(row['interior'],i)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(SM, "output", "cannibalization_diagnostics.tex"), "w", encoding="utf-8").write("\n".join(lines))
p("wrote cannibalization_diagnostics.csv/.tex")

cat_tab.to_frame("n").assign(share=lambda x: x.n/len(d)).to_csv(
    os.path.join(SM, "output", "cannibalization_categories.csv"))
lines2 = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Category & N & Share \\", r"\midrule"]
for cat, n in cat_tab.items():
    lines2.append(f"{cat} & {n} & {n/len(d):.3f} \\\\")
lines2 += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(SM, "output", "cannibalization_categories.tex"), "w", encoding="utf-8").write("\n".join(lines2))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(d.H_same.clip(upper=8), bins=np.arange(-0.5, 9.5, 1), color="C0", edgecolor="white")
ax[0].set_xlabel(r"$H_{same}$ (own storefronts already active in the SAME town, capped at 8)")
ax[0].set_ylabel("# Type-C entry events")
ax[0].set_title(r"Distribution of $H_{same}$ among Type-C entrants")
gv2 = d.H_geo.fillna(0)
ax[1].hist(gv2, bins=25, color="C3", edgecolor="white")
ax[1].set_xlabel(r"$H_{geo}$ (distance-weighted own-network overlap)")
ax[1].set_title(r"Distribution of $H_{geo}$ among Type-C entrants")
for a in ax:
    a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SM, "output", "fig_H_distribution.png"), dpi=140)
plt.close(fig)
p("wrote fig_H_distribution.png")

fig, ax = plt.subplots(figsize=(8.5, 4))
order_full = ["1. same-town existing store", "2. nearby (<=75mi) but different town",
              "3. elsewhere in NM only (>75mi)", "4. no active network at entry (qualifying sibling since exited)"]
order = ["1. same-town existing store", "2. nearby (<=75mi) but different town",
         "3. elsewhere in NM only (>75mi)", "4. no active network (sibling since exited)"]
vals = [cat_tab.get(c, 0) for c in order_full]
ax.barh(range(len(order)), vals, color=["C0", "C1", "C2", "C7"])
ax.set_yticks(range(len(order))); ax.set_yticklabels([c[3:] for c in order], fontsize=9)
ax.set_xlabel("# Type-C entry events")
ax.set_title("Nearest own store, at the time of entry", fontsize=11)
ax.grid(alpha=.25, axis="x"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SM, "output", "fig_H_categories.png"), dpi=140)
plt.close(fig)
p("wrote fig_H_categories.png")

os.makedirs(os.path.join(SM, "logs"), exist_ok=True)
open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.")
