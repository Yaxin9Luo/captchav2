import argparse, requests, time, json
from statistics import mean

def solve_with_model(model_name: str, puzzle: dict) -> str:
    # TODO: call your model; for LLMs, feed puzzle["prompt"] and (optionally) the image URL.
    raise NotImplementedError

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:7860")
    ap.add_argument("--per_type", type=int, default=50, help="how many items per type")
    ap.add_argument("--out", default=f"leaderboard/runs/run_{int(time.time())}.json")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    types = requests.get(f"{args.base_url}/api/puzzle_types").json()
    per_type_acc = {}
    for t in types:
        correct = total = 0
        for _ in range(args.per_type):
            item = requests.get(f"{args.base_url}/api/get_puzzle", params={"type": t}).json()
            pred = solve_with_model(args.model, item)
            resp = requests.post(f"{args.base_url}/api/check_answer", json={"type": t, "id": item["id"], "answer": pred}).json()
            correct += bool(resp.get("correct"))
            total += 1
        per_type_acc[t] = (correct / total) if total else 0.0

    results = {
        "Model": args.model, "Provider": "custom", "Notes": args.notes,
        **per_type_acc, "Overall": mean(per_type_acc.values()) if per_type_acc else 0.0
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
