import json, csv, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "leaderboard" / "runs"
OUT  = ROOT / "leaderboard" / "results.csv"

# Gather all keys (columns)
records = []
for path in RUNS.glob("*.json"):
    records.append(json.loads(path.read_text()))

# Derive header (union of keys, keep common ones first)
fixed = ["Model", "Provider", "Notes", "Overall"]
dynamic = sorted({k for r in records for k in r.keys() if k not in fixed})
header = fixed + dynamic

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in records:
        w.writerow(r)

print(f"Wrote {OUT}")
