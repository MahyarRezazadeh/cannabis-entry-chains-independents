r"""
v01_compare_tables.py -- compare every regenerated table in output/tables with the archived version of the same
file in output/tables/published (the tables in the manuscript before this revision). Numbers are extracted as
tokens in order; the report lists, for each table, the number of numeric tokens, the number that differ, and the
largest absolute difference. Writes output/logs/table_comparison.csv.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import re, csv
from paths import TABLES, LOGS

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
def nums(path):
    s = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    for t in NUM.findall(s):
        try:
            out.append(float(t.replace(",", "")))
        except ValueError:
            pass
    return out

rows = []
for new in sorted(TABLES.glob("*.tex")):
    old = TABLES / "published" / new.name
    if not old.exists():
        rows.append([new.name, "new table (no published counterpart)", "", "", ""])
        continue
    a, b = nums(old), nums(new)
    if len(a) != len(b):
        rows.append([new.name, "token count differs", len(a), len(b), ""])
        continue
    diffs = [abs(x - y) for x, y in zip(a, b)]
    nd = sum(d > 1e-9 for d in diffs)
    rows.append([new.name, "identical" if nd == 0 else "differs", len(a), nd, max(diffs) if diffs else 0])
with open(LOGS / "table_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["table", "status", "tokens_published", "tokens_differing_or_new", "max_abs_diff"]); w.writerows(rows)
for r in rows:
    print(" | ".join(str(x) for x in r))
