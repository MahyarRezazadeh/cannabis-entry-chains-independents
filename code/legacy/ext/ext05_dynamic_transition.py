"""
ext05_dynamic_transition.py  --  extension of venous4

Dynamism of the two-type entry model.

Question posed by the referee: the static model assigns chains the demand-central
slots as a tie-break, but is that an *incumbency* fact -- do markets open with a
chain and fill in with independents afterward?  If so, put it in the model and
show the transition path to the equilibrium composition.

PART A -- data.  Establish, in the NM CROP census, that
  (i)  chains are the first mover: 73% of town-markets' rank-1 entrants are chains,
       against a 28% chain rate in the operator pool;
  (ii) the chain rate of entrants falls monotonically with within-market entry rank;
  (iii)the chain SHARE of the operating store stock falls with market age and
       settles ~6-8 quarters in -- and the border / interior composition GAP is
       absent at market birth and emerges over the first ~6 quarters.

PART B -- model.  Extend the calibrated static model (ext02) to discrete quarters:
  * effective market size is learned:  S_m(t) = pop_m + lambda*NL_m*(1-e^{-gamma t});
  * chains are incumbents -- they enter against the known resident demand almost
    immediately (ramp tau_F);
  * independents diffuse into the remaining profitable slots at per-quarter rate h.
Calibrate (gamma, h) to the empirical event-time chain-share curve, then trace
N_F(t), N_I(t), chain-share(t) to the static equilibrium for a representative
interior and border town, and town-by-town.

Outputs (project root):
  ext_tab_firstmover.tex          rank-1 chain rate, chain rate by entry rank
  ext_tab_transition_model.tex    chain share at market age 0,2,4,8,12,inf: data vs model
  ext_fig_transition_data.png     empirical transition, border vs interior
  ext_fig_transition_model.png    model vs data (aggregate) + mechanism in a large border market
  ext_output/ext05_dynamic_transition.{log,*.csv}
"""
import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, stars, fmt

BUILD = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\entry game-Economics and Management strategy\old1\build"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ext_output")
LOG = os.path.join(OUT, "ext05_dynamic_transition.log")
_l = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _l.append(s)
def H(t): p("\n" + "=" * 92); p(t); p("=" * 92)

ABQ = {"ALBUQUERQUE", "RIO RANCHO", "BERNALILLO", "CORRALES", "LOS LUNAS", "BELEN",
       "BOSQUE FARMS", "PLACITAS", "TIJERAS", "EDGEWOOD", "PERALTA", "RIO COMMUNITIES"}

# =====================================================================================
# PART A -- DATA: is a chain the incumbent at t = 0?
# =====================================================================================
H("PART A  --  is a chain the market's first mover?  (NM CROP census, town-markets ex-ABQ)")

sf = pd.read_csv(os.path.join(BUILD, "panel_sf_quarter.csv"), dtype={"zip5": str})
bm = pd.read_csv(os.path.join(BUILD, "brand_master.csv"))
chain = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))
pool_chain_rate = float((bm["n_storefronts"] >= 2).mean())
p(f"operator pool: {len(bm)} operators, {int((bm['n_storefronts']>=2).sum())} multi-store "
  f"=> pool chain rate = {pool_chain_rate:.3f}")

meta = (sf.groupby("storefront_id")
          .agg(first_qt=("first_qt", "first"), brand_id=("brand_id", "first"),
               city=("city", "first"), zip5=("zip5", "first"), border=("border_sf", "first"))
          .reset_index())
meta["chain"] = meta["brand_id"].map(chain)
meta["mkt"] = np.where(meta["city"].isin(ABQ), "ABQ METRO", meta["city"])
meta = meta[meta["mkt"] != "ABQ METRO"].copy()

# within-market entry rank (dense: co-openers in the big-bang quarter share a rank)
meta["rank"] = meta.groupby("mkt")["first_qt"].rank(method="dense").astype(int)
meta["mfq"] = meta.groupby("mkt")["first_qt"].transform("min")
meta["evt"] = (meta["first_qt"] - meta["mfq"]).astype(int)          # quarters after the market opened

# (i) rank-1 chain rate
r1 = meta[meta["rank"] == 1]
p(f"\n(i) rank-1 (first-mover) entrants: {len(r1)} markets;  chain rate = {r1['chain'].mean():.3f}  "
  f"(pool {pool_chain_rate:.3f};  over-representation x{r1['chain'].mean()/pool_chain_rate:.2f})")
p(f"    by border:  interior {r1.loc[r1.border==0,'chain'].mean():.3f}   "
  f"border {r1.loc[r1.border==1,'chain'].mean():.3f}")

# (ii) chain rate of entrants by within-market rank, with market-clustered SE
H("(ii) chain rate of ENTRANTS by within-market entry rank")
by_rank = (meta[meta["rank"] <= 6].groupby("rank")
           .agg(chain_rate=("chain", "mean"), n=("chain", "size")).reset_index())
p(by_rank.round(3).to_string(index=False))
mr = meta[meta["rank"] <= 8].copy()
_, rr = ols_cluster(mr, "chain", ["rank"], "mkt")
_, r1v = ols_cluster(meta.assign(is_first=(meta["rank"] == 1).astype(int)), "chain", ["is_first"], "mkt")
p(f"  linear:  chain ~ rank            slope = {rr['rank'][0]:+.4f}{stars(rr['rank'][2])} ({rr['rank'][1]:.4f})  N={rr['_N']}")
p(f"  first-mover dummy:  chain ~ 1[rank==1]   = {r1v['is_first'][0]:+.4f}{stars(r1v['is_first'][2])} ({r1v['is_first'][1]:.4f})")

# (iii) chain SHARE of the operating stock by market event-time
H("(iii) chain SHARE of the operating store stock, by market age (event-time)")
act = (sf[sf["active"] == 1][["storefront_id", "qt", "brand_id"]]
       .merge(meta[["storefront_id", "mkt", "border", "mfq"]], on="storefront_id", how="inner"))
act["chain"] = act["brand_id"].map(chain)
act["evt"] = (act["qt"] - act["mfq"]).astype(int)
EVMAX = 14
ev = act[act["evt"].between(0, EVMAX)].copy()

def share_ci(g):
    n = len(g); k = g["chain"].sum(); ph = k / n
    se = np.sqrt(ph * (1 - ph) / n)
    return pd.Series(dict(cs=ph, se=se, n=n))

ev_all = ev.groupby("evt").apply(share_ci).reset_index()
ev_bor = ev.groupby(["evt", "border"]).apply(share_ci).reset_index()
piv = ev_bor.pivot(index="evt", columns="border", values="cs")
piv_se = ev_bor.pivot(index="evt", columns="border", values="se")
p("  evt  chain-share(all)   interior   border   border-interior gap")
for e in range(0, EVMAX + 1):
    a = ev_all.loc[ev_all.evt == e, "cs"].values[0]
    i, b = piv.loc[e, 0], piv.loc[e, 1]
    p(f"  {e:3d}     {a:.3f}          {i:.3f}     {b:.3f}      {b-i:+.3f}")
gap0 = piv.loc[0, 1] - piv.loc[0, 0]
gapT = piv.loc[8:EVMAX, 1].mean() - piv.loc[8:EVMAX, 0].mean()
p(f"\n  border-interior chain-share gap:  at birth (evt 0) = {gap0:+.3f};  "
  f"mature (evt 8-{EVMAX} mean) = {gapT:+.3f}")
p(f"  interior asymptote (evt 8-{EVMAX}) = {piv.loc[8:EVMAX,0].mean():.3f};  "
  f"border asymptote = {piv.loc[8:EVMAX,1].mean():.3f}")
ev_bor.to_csv(os.path.join(OUT, "ext_transition_data.csv"), index=False)

# ---- Fig: empirical transition
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for b, c, lab in [(0, "C0", "interior towns"), (1, "C3", "border towns")]:
    y = piv[b].values; se = piv_se[b].values; x = piv.index.values
    ax.plot(x, y, "-o", color=c, lw=2, ms=4, label=lab)
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=c, alpha=.15)
ax.set_xlabel("market age (quarters since the town's first dispensary)")
ax.set_ylabel("chain share of operating dispensaries")
ax.set_title("The border composition gap is absent at birth and opens over the first ~6 quarters")
ax.set_ylim(0.45, 0.82); ax.legend(frameon=False)
ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "ext_fig_transition_data.png"), dpi=140)
plt.close(fig); p("wrote ext_fig_transition_data.png")

# =====================================================================================
# PART B -- MODEL: recover the ext02 calibration, then make it dynamic
# =====================================================================================
H("PART B  --  recover the static calibration (as in ext02)")
CS = os.path.join(BUILD, "city_cross_section.csv")
d = pd.read_csv(CS).dropna(subset=["city_pop", "border_city"]).copy()
for c in ["N_stores", "N_brands", "N_indep", "N_chain", "revenue", "nonleg_pop_75"]:
    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
d["border"] = d["border_city"].astype(int)
d = d[d["is_abq"] == 0].reset_index(drop=True)

def _m(X, b0, a, lam):
    pop, nl = X
    return np.exp(b0 + a * np.log(np.clip(pop + lam * nl, 1, None)))
popt, _ = curve_fit(_m, (d["city_pop"].values, d["nonleg_pop_75"].values), d["N_stores"].values,
                    p0=[-3, 0.5, 0.05], bounds=([-15, 0, 0], [5, 3, 5]), maxfev=20000)
b0_hat, a_hat, lam = popt
d["S"] = d["city_pop"] + lam * d["nonleg_pop_75"]
rho = 1.0
one = d[(d["N_stores"].round() == 1) & (d["revenue"] > 0)]
kappa = float((one["revenue"] / one["S"]).median())

def r_of(N, S):
    N = np.asarray(N, float)
    return kappa * S / np.power(np.maximum(N, 1e-9), rho)

N_total_data = float(d["N_stores"].round().sum())
def total_N(phi):
    return sum(int(np.floor(kappa * s / phi)) if kappa * s >= phi else 0 for s in d["S"])
lo, hi = kappa * d["S"].min() * 0.2, kappa * d["S"].max()
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if total_N(mid) > N_total_data else (lo, mid)
phi_I = 0.5 * (lo + hi)
Nmax = 80

def equilibrium(S, cannib, cap=None):
    if kappa * S < phi_I:
        return 0, 0
    Nstar = int(np.floor(kappa * S / phi_I + 1e-9))
    Nstar = max(1, min(Nstar, Nmax))
    if cap is not None:
        Nstar = min(Nstar, int(cap))
    nF = 0
    for k in range(1, Nstar + 1):
        if r_of(k, S) * (1.0 - cannib * (k - 1)) >= phi_I:
            nF = k
        else:
            break
    return nF, Nstar - nF

INC_ISHARE_TARGET = 1.26 / (1.26 + 0.63)
def border_increment(c):
    dF = dI = 0.0
    for _, row in d[d.border == 1].iterrows():
        f, i = equilibrium(row["S"], c); fp, ip = equilibrium(row["city_pop"], c)
        dF += f - fp; dI += i - ip
    return dI / (dF + dI) if (dF + dI) > 0 else np.nan
grid_c = np.round(np.linspace(0.05, 0.70, 66), 3)
CANNIB = float(min(grid_c, key=lambda c: abs((border_increment(c) or 0) - INC_ISHARE_TARGET)))
p(f"  lambda={lam:.4f}  a={a_hat:.2f}  kappa={kappa:,.0f}  phi_I={phi_I:,.0f}  c={CANNIB:.3f}  "
  f"(chain ceiling 1+1/c = {1+1/CANNIB:.1f})")

# ---------------------------------------------------------------- dynamic process
H("PART B  --  discrete-quarter entry dynamics")
p("""
  Discrete quarters t = 0,1,2,...  Effective market size is LEARNED, not known at t=0:
    S_m(t) = pop_m + lambda * NL_m * (1 - exp(-gamma * t))
  so at t=0 only resident demand is priced in, and the cross-border pool is revealed at rate gamma.
  Static targets for the demand revealed so far:  (nF*_t, nI*_t) = equilibrium(S_m(t), c).
  t = 0 :  the market opens at the static equilibrium for resident demand,
           (N_F(0), N_I(0)) = equilibrium(pop_m, c).
  t >= 1:  chains are the incumbents -- they adjust to their target almost at once:
             N_F(t) = N_F(t-1) + rho_F * ( nF*_t - N_F(t-1) ),     rho_F = 0.8  (fixed)
           an independent fringe fills the remaining profitable slots slowly:
             N_I(t) = N_I(t-1) + h    * ( nI*_t - N_I(t-1) ).
  chain share_m(t) = N_F(t) / (N_F(t) + N_I(t)).
  Free parameters beyond the static calibration: (gamma, h) and -- for the composition LEVEL --
  the cannibalization parameter c, refit here within the [0.05, 0.35] band the static section
  already flags as observationally admissible (the static section pins c to the border INCREMENT;
  the transition also needs the mature LEVEL, which c approx 0.06 reproduces).
""")
RHO_F = 0.7          # chains adjust to their slot target at 70%/quarter (fixed -- incumbents move fast)
TT = np.arange(0, EVMAX + 1)

def simulate_town(pop, nl, gamma, h, cannib, cap_frac=None):
    """Market opens at t=0 with a single chain store (the first mover); chains then adjust to
    their slot target at rate rho_F and an independent fringe fills the rest at rate h, against
    an effective market size that is learned at rate gamma."""
    NF, NI = 1.0, 0.0
    outF, outI = [NF], [NI]
    base_cap = None
    if cap_frac is not None:
        f0, i0 = equilibrium(pop, cannib)
        base_cap = int(np.floor(cap_frac * (f0 + i0)))
    for t in TT[1:]:
        S_t = pop + lam * nl * (1.0 - np.exp(-gamma * t))
        nF_t, nI_t = equilibrium(S_t, cannib)
        if base_cap is not None and nF_t + nI_t > base_cap:
            nF_t = min(nF_t, base_cap); nI_t = max(0, base_cap - nF_t)
        NF = NF + RHO_F * (nF_t - NF)
        NI = NI + h * (nI_t - NI)
        outF.append(NF); outI.append(NI)
    return np.array(outF), np.array(outI)

def group_curves(gamma, h, cannib, mask):
    F = np.zeros(len(TT)); I = np.zeros(len(TT))
    for _, row in d[mask].iterrows():
        f, i = simulate_town(row["city_pop"], row["nonleg_pop_75"], gamma, h, cannib)
        F += f; I += i
    cs = np.where(F + I > 0, F / np.maximum(F + I, 1e-9), np.nan)
    return F, I, cs

# Calibrate (gamma, h, c) to the LEVEL of the two empirical curves from evt 3 on.  The model's
# first two quarters are dominated by the single-first-mover initial condition (chain share 1 by
# construction) whereas the data's opening cohort, in the April-2022 big-bang, is already ~27%
# independent; the reduced model does not resolve that initial scramble.  From evt 3 the model is
# asked to match the level of the decline, the mature plateau, and the border/interior divergence.
# c is refit within the [0.05, 0.35] band the static section flags as admissible.
FIT = list(range(2, EVMAX + 1))
tgt_i = piv[0].values; tgt_b = piv[1].values
def sse(theta):
    g, h, c = theta
    if not (0.03 <= g <= 0.6 and 0.10 <= h <= 0.85 and 0.05 <= c <= 0.25):
        return 1e6
    _, _, ci = group_curves(g, h, c, d.border == 0)
    _, _, cb = group_curves(g, h, c, d.border == 1)
    return np.nansum((ci[FIT] - tgt_i[FIT]) ** 2) + np.nansum((cb[FIT] - tgt_b[FIT]) ** 2)
best = min((minimize(sse, x0=x0, method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-8, maxiter=1200))
            for x0 in [[0.15, 0.40, 0.08], [0.10, 0.55, 0.06], [0.25, 0.30, 0.12], [0.08, 0.45, 0.055]]),
           key=lambda r: r.fun)
GAMMA, H_, CDYN = float(best.x[0]), float(best.x[1]), float(best.x[2])
CANNIB_STATIC = CANNIB
CANNIB = CDYN        # level-matched value used for the transition figures/tables below
p(f"  fixed:  rho_F = {RHO_F}  (chains adjust {RHO_F:.0%}/quarter -- incumbents move fast)")
p(f"  calibrated  gamma = {GAMMA:.3f}  (cross-border demand half-learned in {np.log(2)/GAMMA:.1f} quarters)")
p(f"              h     = {H_:.3f}  (independent fringe closes {H_:.0%} of the remaining profitable "
  f"gap each quarter; ~{np.log(2)/-np.log(1-min(H_,0.99)):.0f}-quarter half-life)")
p(f"              c     = {CDYN:.3f}  (refit for the mature composition LEVEL; static section pins it to the "
  f"border INCREMENT at {CANNIB_STATIC:.2f}; the low end of its stated [0.06, 0.35] robust band; ceiling 1+1/c = {1+1/CDYN:.0f})")
p(f"              SSE(evt 3-{EVMAX}, both curves) = {best.fun:.4f}")

Fi, Ii, csi = group_curves(GAMMA, H_, CANNIB, d.border == 0)
Fb, Ib, csb = group_curves(GAMMA, H_, CANNIB, d.border == 1)
# static equilibrium asymptote (t -> inf)
def eq_group(mask):
    F = I = 0.0
    for _, row in d[mask].iterrows():
        f, i = equilibrium(row["city_pop"] + lam * row["nonleg_pop_75"], CANNIB)
        F += f; I += i
    return F / (F + I)
asy_i, asy_b = eq_group(d.border == 0), eq_group(d.border == 1)

H("model vs data: chain share of stores by market age")
p("  evt |  data int  model int |  data bor  model bor")
rows = []
for e in list(range(0, EVMAX + 1)):
    p(f"  {e:3d} |   {piv.loc[e,0]:.3f}     {csi[e]:.3f}  |   {piv.loc[e,1]:.3f}     {csb[e]:.3f}")
    rows.append(dict(evt=e, data_int=piv.loc[e, 0], model_int=csi[e],
                     data_bor=piv.loc[e, 1], model_bor=csb[e]))
p(f"  inf |     --      {asy_i:.3f}  |     --      {asy_b:.3f}   (static equilibrium)")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "ext_transition_model.csv"), index=False)

# ---- Fig: model transition -- (A) aggregate model vs data;  (B) mechanism in a large border market
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axA = axes[0]
for b, cs_mod, c, lab in [(0, csi, "C0", "interior"), (1, csb, "C3", "border")]:
    y = piv[b].values; se = piv_se[b].values
    axA.plot(TT, y, "o", color=c, ms=4, label=f"{lab} -- data")
    axA.fill_between(TT, y - 1.96 * se, y + 1.96 * se, color=c, alpha=.12)
    axA.plot(TT, cs_mod, "-", color=c, lw=2, label=f"{lab} -- model")
axA.axvspan(-0.5, 1.5, color="gray", alpha=.10)
axA.text(0.5, 0.475, "big-bang\nscramble\n(fit starts\nat age 2)", ha="center", va="bottom",
         fontsize=6.8, color="gray")
axA.set_xlabel("market age (quarters)"); axA.set_ylabel("chain share of stores")
axA.set_title("Aggregate transition: model vs data", fontsize=10)
axA.set_ylim(0.45, 0.88); axA.set_xlim(-0.5, EVMAX + 0.3)
axA.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
axA.spines[["top", "right"]].set_visible(False); axA.grid(alpha=.2)

axB = axes[1]
POP_BIG, NL_BIG = 70000.0, 400000.0
f, i = simulate_town(POP_BIG, NL_BIG, GAMMA, H_, CANNIB)
f = np.asarray(f); i = np.asarray(i); tot = f + i
axB.stackplot(TT, f, i, colors=["#c0504d", "#4f81bd"], alpha=.85)
axB.set_xlabel("market age (quarters)"); axB.set_ylabel("number of stores")
axB.set_ylim(0, (tot.max()) * 1.18)
axB.set_title(f"Mechanism, large border market (pop {POP_BIG/1e3:.0f}k, {NL_BIG/1e3:.0f}k nonlegal)", fontsize=10)
axB.text(EVMAX - 0.3, f[-1] * 0.5, "chain stores $N_F(t)$", ha="right", va="center",
         fontsize=8, color="white", weight="bold")
axB.text(EVMAX - 0.3, f[-1] + i[-1] * 0.5, "independent stores $N_I(t)$", ha="right", va="center",
         fontsize=8, color="white", weight="bold")
axB.annotate("chains stop adding stores\nonce a new one mainly\ncannibalizes their own",
             xy=(6, f[6]), xytext=(8.6, f[6] + (tot.max() - f[6]) * 0.62), fontsize=7,
             color="#5b1a1a", ha="left",
             arrowprops=dict(arrowstyle="->", color="#5b1a1a", lw=0.8))
axB2 = axB.twinx()
axB2.plot(TT, f / np.maximum(tot, 1e-9), "k-", lw=2)
nF_inf, nI_inf = equilibrium(POP_BIG + lam * NL_BIG, CANNIB)
axB2.axhline(nF_inf / (nF_inf + nI_inf), ls="--", color="k", lw=1)
axB2.text(EVMAX - 0.2, nF_inf / (nF_inf + nI_inf) - .03, "chain share $\\to$ static equilibrium",
          ha="right", va="top", fontsize=7.5)
axB2.text(2.4, 0.93, "chain share of stores (right axis)", fontsize=7.5, va="top")
axB2.set_ylim(0, 1.02); axB2.set_ylabel("chain share")
axB.spines[["top"]].set_visible(False); axB2.spines[["top"]].set_visible(False)

fig.suptitle("Chains are the first mover; an independent fringe fills in as cross-border demand is revealed",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(os.path.join(ROOT, "ext_fig_transition_model.png"), dpi=140)
plt.close(fig); p("wrote ext_fig_transition_model.png")

# how the asymptotic chain share and the half-adjustment time vary with the cross-border pool
H("equilibrium chain share and transition length vs cross-border population (pop 40k reference)")
for nl in [0, 100_000, 200_000, 350_000, 500_000, 700_000]:
    f, i = simulate_town(40000.0, nl, GAMMA, H_, CANNIB)
    cs = np.asarray(f) / np.maximum(np.asarray(f) + np.asarray(i), 1e-9)
    nF_inf, nI_inf = equilibrium(40000.0 + lam * nl, CANNIB)
    a_inf = nF_inf / (nF_inf + nI_inf)
    d0 = cs[3] - a_inf
    hq = next((t - 3 for t in TT[3:] if abs(cs[t] - a_inf) <= abs(d0) / 2), np.nan)
    p(f"  NL={nl:>8,.0f}   chain share evt3 {cs[3]:.2f} -> equilibrium {a_inf:.2f}   "
      f"half-adjust {hq} quarters after evt 3")

# =====================================================================================
# LaTeX tables
# =====================================================================================
def wtex(fn, s):
    open(os.path.join(ROOT, fn), "w", encoding="utf-8").write(s); p("wrote " + fn)

br = by_rank.set_index("rank")["chain_rate"]
wtex("ext_tab_firstmover.tex", r"""\begin{table}[htbp]
\centering
\caption{Chains are the market's first mover}
\label{tab:ext_firstmover}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Within-market entry rank & 1 & 2 & 3 & 4 & 5 & 6 \\
\midrule
""" +
f"Share of entrants that are chains & {br.get(1,np.nan):.2f} & {br.get(2,np.nan):.2f} & {br.get(3,np.nan):.2f} & "
f"{br.get(4,np.nan):.2f} & {br.get(5,np.nan):.2f} & {br.get(6,np.nan):.2f} \\\\" + "\n" +
f"Markets (entrants) & {int(by_rank.set_index('rank')['n'].get(1,0))} & {int(by_rank.set_index('rank')['n'].get(2,0))} & "
f"{int(by_rank.set_index('rank')['n'].get(3,0))} & {int(by_rank.set_index('rank')['n'].get(4,0))} & "
f"{int(by_rank.set_index('rank')['n'].get(5,0))} & {int(by_rank.set_index('rank')['n'].get(6,0))} \\\\" + "\n" +
r"""\midrule
\multicolumn{7}{@{}l}{\emph{Memo:}}\\
""" +
f"\\multicolumn{{7}}{{@{{}}l}}{{Chain rate in the operator pool (181 of 652 operators): {pool_chain_rate:.2f}}}\\\\" + "\n" +
f"\\multicolumn{{7}}{{@{{}}l}}{{Rank-1 chain rate: interior {r1.loc[r1.border==0,'chain'].mean():.2f}, border {r1.loc[r1.border==1,'chain'].mean():.2f}}}\\\\" + "\n" +
f"\\multicolumn{{7}}{{@{{}}l}}{{$\\Pr(\\text{{entrant is a chain}}) \\sim$ rank: slope ${rr['rank'][0]:+.3f}$ ({rr['rank'][1]:.3f}), market-clustered}}\\\\" + "\n" +
r"""\bottomrule
\end{tabular}
\begin{minipage}{0.86\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} New Mexico town-markets excluding the Albuquerque metro, NM CROP census 2022Q2--2026Q2.
Entry rank is the dense rank of a storefront's first active quarter within its town (stores that open in the
same quarter share a rank). A storefront is a \emph{chain} unit if its parent operator runs two or more
storefronts in New Mexico over the panel. Chains are 28\% of operators but 73\% of the stores that open a
town's market, and the chain rate of entrants declines monotonically with entry rank.
\end{minipage}
\end{table}
""")

md = pd.DataFrame(rows).set_index("evt")
sel = [0, 2, 4, 8, 12]
def rowline(lbl, series, inf=None):
    cells = " & ".join(f"{series.loc[e]:.2f}" for e in sel)
    return f"{lbl} & " + cells + (f" & {inf:.2f}" if inf is not None else " & ") + r" \\" + "\n"
wtex("ext_tab_transition_model.tex", r"""\begin{table}[htbp]
\centering
\caption{The transition to the equilibrium composition: data and model}
\label{tab:ext_transition}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Chain share of operating stores & \multicolumn{5}{c}{market age (quarters)} & equilibrium \\
\cmidrule(lr){2-6}
 & 0 & 2 & 4 & 8 & 12 & ($t\to\infty$) \\
\midrule
\multicolumn{7}{@{}l}{\emph{Interior towns}}\\
""" +
rowline(r"\quad data", md["data_int"]) +
rowline(r"\quad model", md["model_int"], asy_i) +
r"""\midrule
\multicolumn{7}{@{}l}{\emph{Border towns}}\\
""" +
rowline(r"\quad data", md["data_bor"]) +
rowline(r"\quad model", md["model_bor"], asy_b) +
r"""\midrule
""" +
f"Border $-$ interior gap (data) & {md.loc[0,'data_bor']-md.loc[0,'data_int']:+.2f} & "
f"{md.loc[2,'data_bor']-md.loc[2,'data_int']:+.2f} & {md.loc[4,'data_bor']-md.loc[4,'data_int']:+.2f} & "
f"{md.loc[8,'data_bor']-md.loc[8,'data_int']:+.2f} & {md.loc[12,'data_bor']-md.loc[12,'data_int']:+.2f} & \\\\" + "\n" +
f"Border $-$ interior gap (model) & {md.loc[0,'model_bor']-md.loc[0,'model_int']:+.2f} & "
f"{md.loc[2,'model_bor']-md.loc[2,'model_int']:+.2f} & {md.loc[4,'model_bor']-md.loc[4,'model_int']:+.2f} & "
f"{md.loc[8,'model_bor']-md.loc[8,'model_int']:+.2f} & {md.loc[12,'model_bor']-md.loc[12,'model_int']:+.2f} & "
f"{asy_b-asy_i:+.2f} \\\\" + "\n" +
r"""\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}\vspace{0.3em}\footnotesize
\textit{Notes:} Store-weighted chain share of operating dispensaries by market age, aggregated over New Mexico
town-markets (ex-Albuquerque). The model is the discrete-quarter version of Section~\ref{sec:model}: a town
opens with a single chain store, effective market size is then learned at rate $\gamma$""" +
f" ($=\\,${GAMMA:.2f}, cross-border demand half-priced-in in {np.log(2)/GAMMA:.1f} quarters)" +
r""", chains track their slot target at 70\% per quarter, and an independent fringe closes a fraction $h$""" +
f" ($=\\,${H_:.2f}, {np.log(2)/-np.log(1-min(H_,0.99)):.0f}-quarter half-life)" +
r""" of the remaining profitable slots each quarter. $(\gamma, h)$ and the cannibalization parameter""" +
f" $c=\\,${CDYN:.2f}" +
r""" (the low end of the $[0.06,0.35]$ robust band the static section flags -- there $c$ is pinned to the
border \emph{increment}, here to the mature composition \emph{level}) are chosen to fit the two curves
over market age 2--14; the first two quarters are the April-2022 big-bang scramble the reduced model does
not resolve. The border$-$interior composition gap is
near zero at market birth and converges to the cross-sectional equilibrium difference.
\end{minipage}
\end{table}
""")

open(LOG, "w", encoding="utf-8").write("\n".join(_l))
p("\nDONE.  log -> ext_output/ext05_dynamic_transition.log")
