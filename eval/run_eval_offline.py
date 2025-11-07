import argparse, json, pathlib, time
from statistics import mean

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "captcha_data"

# --- You implement this ---
def solve_with_model(model_name: str, item: dict) -> str:
    """
    Returns the predicted answer string.
    Implement your model here (LLM call, vision+LLM, rule-based, etc).
    `item` likely has fields like 'image', 'prompt', 'answer' (adjust to your JSON).
    """
    # Example pseudo-logic (replace with real model calls):
    # return call_openai_vision(model_name, item["image"], item.get("prompt",""))
    raise NotImplementedError("Wire up your solver here")

def eval_type(model_name: str, type_dir: pathlib.Path) -> float:
    gt = json.loads((type_dir / "ground_truth.json").read_text())
    correct, total = 0, 0
    for item in gt:
        try:
            pred = solve_with_model(model_name, item).strip().lower()
            ans  = str(item["answer"]).strip().lower()
            correct += (pred == ans)
            total += 1
        except Exception:
            total += 1
    return (correct / total) if total else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "leaderboard" / "runs" / f"run_{int(time.time())}.json"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    results = {}
    for type_dir in sorted([p for p in DATA.iterdir() if p.is_dir()]):
        if not (type_dir / "ground_truth.json").exists():
            continue
        acc = eval_type(args.model, type_dir)
        results[type_dir.name] = acc

    results["Overall"] = mean(results.values()) if results else 0.0
    results["Model"] = args.model
    results["Provider"] = "custom"
    results["Notes"] = args.notes

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()
