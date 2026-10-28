"""Execution accuracy: run predicted and gold SQL, compare the results.

Pass one or more prediction files to get a side-by-side comparison and a
pairwise breakdown (which examples one setup gets right and the other wrong).

Usage:
    python -m text2sql.evaluate predictions/base_1.5b.jsonl predictions/ft_1.5b.jsonl predictions/claude-opus-5-5.jsonl \
        --db-dir /kaggle/input/spider/spider/database
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from text2sql.sql_utils import ExecutionError, execute_sqlite, order_matters, results_match

# Outcome of one example:
#   correct      results match the gold query
#   wrong_result query ran but returned different rows
#   exec_error   query failed (syntax error, unknown column, ...)
#   timeout      query exceeded --timeout
#   empty        no SQL could be extracted from the model output
#   gold_error   the gold query itself failed; excluded from accuracy


def evaluate_file(path: Path, db_dir: Path, timeout: float, gold_cache: dict) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]
    results = []
    for i, p in enumerate(preds, 1):
        db_path = db_dir / p["db_id"] / f"{p['db_id']}.sqlite"
        key = (p["db_id"], p["gold"])
        if key not in gold_cache:
            try:
                gold_cache[key] = execute_sqlite(db_path, p["gold"], timeout)
            except ExecutionError as e:
                gold_cache[key] = e
        gold = gold_cache[key]

        detail = ""
        if isinstance(gold, ExecutionError):
            outcome, detail = "gold_error", str(gold)
        elif not p["pred"]:
            outcome = "empty"
        else:
            try:
                rows = execute_sqlite(db_path, p["pred"], timeout)
                outcome = "correct" if results_match(rows, gold, order_matters(p["gold"])) else "wrong_result"
            except ExecutionError as e:
                outcome, detail = ("timeout" if e.kind == "timeout" else "exec_error"), str(e)
        results.append({"id": p["id"], "db_id": p["db_id"], "question": p["question"],
                        "gold": p["gold"], "pred": p["pred"], "outcome": outcome, "detail": detail})
        if i % 200 == 0:
            print(f"  {path.name}: {i}/{len(preds)}", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    counts = Counter(r["outcome"] for r in results)
    scored = len(results) - counts["gold_error"]
    executable = counts["correct"] + counts["wrong_result"]
    return {
        "n": len(results),
        "scored": scored,
        "execution_accuracy": counts["correct"] / scored if scored else 0.0,
        "valid_sql_rate": executable / scored if scored else 0.0,
        "outcomes": dict(counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--db-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold_cache: dict = {}
    all_results: dict[str, dict[str, dict]] = {}
    summaries: dict[str, dict] = {}
    for path in args.predictions:
        results = evaluate_file(path, args.db_dir, args.timeout, gold_cache)
        name = path.stem
        all_results[name] = {r["id"]: r for r in results}
        summaries[name] = summarize(results)
        with (args.out_dir / f"{name}.details.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    width = max(len(n) for n in summaries) + 2
    print(f"\n{'setup':<{width}} {'n':>5} {'EX acc':>8} {'valid':>8}  outcomes")
    for name, s in summaries.items():
        print(f"{name:<{width}} {s['scored']:>5} {s['execution_accuracy']:>8.1%} {s['valid_sql_rate']:>8.1%}  {s['outcomes']}")

    if len(all_results) > 1:
        print("\nPairwise (on shared, scorable examples):")
        for a, b in combinations(all_results, 2):
            ids = [i for i in all_results[a] if i in all_results[b]
                   and all_results[a][i]["outcome"] != "gold_error"]
            ca = {i for i in ids if all_results[a][i]["outcome"] == "correct"}
            cb = {i for i in ids if all_results[b][i]["outcome"] == "correct"}
            print(f"  {a} vs {b}: both={len(ca & cb)}  only {a}={len(ca - cb)}  "
                  f"only {b}={len(cb - ca)}  neither={len(ids) - len(ca | cb)}")
    print(f"\nDetails written to {args.out_dir}/")


if __name__ == "__main__":
    main()
