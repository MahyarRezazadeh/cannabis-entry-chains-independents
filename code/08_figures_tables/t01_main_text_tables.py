r"""
t01_main_text_tables.py -- assemble two compact main-text tables from the outputs of rev02 / rev02b:
  tab_rio_controls_imputation.tex : Panel A = predetermined controls (rows 1-4 of tab_rev_robustness.tex),
                                    Panel B = imputation estimator from the 276 unexposed ZIPs (rev02b, section iv)
  tab_rio_intensity.tex           : exposure intensity within the nonlegal group and across 368 ZIPs (rev02b, v-vi)
Run after code/04_main_analysis/rev02*.py.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import re
from scipy import stats
from paths import TABLES, ANALYSIS

def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
def cell(b, s, p=None):
    if p is None:
        p = 2 * (1 - stats.norm.cdf(abs(b / s)))
    return f"${b:+.3f}^{{{star(p)}}}$".replace("^{}", ""), f"({s:.3f})"

rob = (TABLES / "tab_rev_robustness.tex").read_text(encoding="utf-8").splitlines()
log = (ANALYSIS / "rev02b_results.txt").read_text(encoding="utf-8").splitlines()

# Panel A: first four specification rows (coefficient line + SE line) of tab_rev_robustness
panelA = []
i = next(k for k, l in enumerate(rob) if l.startswith("Baseline"))
for k in range(4):
    lab_line, se_line = rob[i + 2 * k], rob[i + 2 * k + 1]
    lab = lab_line.split("&")[0].strip()
    coefs = [c.strip() for c in lab_line.split("&")[1:4]]
    ses = [c.strip() for c in se_line.split("&")[1:4]]
    panelA.append((lab, coefs, ses))
# Panel B: imputation lines
imp = []
sec = log.index(next(l for l in log if l.startswith("(iv)")))
for l in log[sec + 1:sec + 5]:
    lab = re.split(r"\s{2,}", l.strip())[0]
    vals = re.findall(r"([+-]\d\.\d+)\(boot SE (\d\.\d+), p~(\d\.\d+)\)", l)
    imp.append((lab, [(float(b), float(s), float(p)) for b, s, p in vals]))
lab_imp = {"log pop only (unexposed = LB + interior)": "Log population only", "+ geography": "+ geography",
           "+ geography + 2019 retail": "+ geography + 2019 retail",
           "+ geography + 2019 retail, common support in distance to ABQ": "+ geography + 2019 retail, common support"}
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Revenue per & Dispensaries & log revenue \\",
     r" & resident (asinh) & per 1,000 (asinh) & per dispensary \\", r" & (1) & (2) & (3) \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Regression with predetermined controls (209 ZIP codes, 92 nonlegal border)}} \\"]
for lab, coefs, ses in panelA:
    lab = lab.replace(" (density, distance to ABQ and to city $\\geq$50k)", "").replace(" establishments", "")
    L += [f"{lab} & " + " & ".join(coefs) + r" \\", " & " + " & ".join(ses) + r" \\"]
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: Imputation from the 276 unexposed ZIP codes}} \\")
for lab, vals in imp:
    cc = [cell(b, s, p) for b, s, p in vals]
    L += [f"{lab_imp.get(lab, lab)} & " + " & ".join(c[0] for c in cc) + r" \\", " & " + " & ".join(c[1] for c in cc) + r" \\"]
L += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_rio_controls_imputation.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

# exposure intensity
def grab(prefix_sec, labels):
    s = log.index(next(l for l in log if l.startswith(prefix_sec)))
    out = []
    for l in log[s + 1:s + 1 + len(labels)]:
        vals = re.findall(r"([+-]\d\.\d+)\((\d\.\d+)\)(\**)", l)
        out.append([(float(b), float(se)) for b, se, _ in vals[:3]])
    return out
vin = grab("(v)", ["no controls", "+ geography + 2019 retail"])
vall = grab("(vi)", ["no controls", "+ geo", "+ geo + med"])
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r" & Revenue per & Dispensaries & log revenue \\",
     r" & resident (asinh) & per 1,000 (asinh) & per dispensary \\", r" & (1) & (2) & (3) \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{Panel A: Within the 92 nonlegal-border ZIP codes (13 county clusters)}} \\"]
for lab, vals in zip(["No controls", "+ geography + 2019 retail"], vin):
    cc = [cell(b, s, 2 * (1 - stats.t.cdf(abs(b / s), 12))) for b, s in vals]
    L += [f"{lab} & " + " & ".join(c[0] for c in cc) + r" \\", " & " + " & ".join(c[1] for c in cc) + r" \\"]
L.append(r"\addlinespace\multicolumn{4}{@{}l}{\textit{Panel B: All 368 New Mexico ZIP codes (33 county clusters)}} \\")
for lab, vals in zip(["No controls", "+ geography + 2019 retail", "+ geography + 2019 retail + medical incumbents"], vall):
    cc = [cell(b, s, 2 * (1 - stats.t.cdf(abs(b / s), 32))) for b, s in vals]
    L += [f"{lab} & " + " & ".join(c[0] for c in cc) + r" \\", " & " + " & ".join(c[1] for c in cc) + r" \\"]
L += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_rio_intensity.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print((TABLES / "tab_rio_controls_imputation.tex").read_text()); print((TABLES / "tab_rio_intensity.tex").read_text())
