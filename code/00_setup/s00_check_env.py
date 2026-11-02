"""Check the Python version and the installed package versions against environment/requirements.txt."""
import sys, pathlib, importlib
root = pathlib.Path(__file__).resolve().parents[2]
req = [l.split("==") for l in (root / "environment" / "requirements.txt").read_text().splitlines()
       if l and not l.startswith("#")]
print("Python", sys.version.split()[0], "(paper: 3.13.12)")
bad = 0
for name, ver in req:
    mod = {"pyyaml": "yaml"}.get(name, name)
    try:
        v = importlib.import_module(mod).__version__
    except Exception as e:
        print(f"  MISSING {name}=={ver}"); bad += 1; continue
    flag = "" if v == ver else f"   (paper used {ver})"
    print(f"  {name:12s} {v}{flag}")
print("environment check finished;", "all packages present" if not bad else f"{bad} package(s) missing")
sys.exit(1 if bad else 0)
