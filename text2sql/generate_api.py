"""Generate SQL predictions with a Claude model through the Anthropic API.

Uses the exact same prompts as the local models (from dev.jsonl). Re-running
with the same --out resumes: examples that already succeeded are skipped.

Requires ANTHROPIC_API_KEY (on Kaggle, add it under Add-ons -> Secrets).

Usage:
    python -m text2sql.generate_api --out predictions/claude-opus-5-5.jsonl
    python -m text2sql.generate_api --limit 50 --out predictions/claude_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from text2sql.generate import load_rows
from text2sql.sql_utils import extract_sql


def call_model(client: anthropic.Anthropic, args, row: dict) -> dict:
    system = next(m["content"] for m in row["prompt"] if m["role"] == "system")
    messages = [m for m in row["prompt"] if m["role"] != "system"]
    record = {"id": row["id"], "db_id": row["db_id"], "question": row["question"], "gold": row["query"],
              "model": args.model}
    kwargs = dict(
        model=args.model,
        max_tokens=16000,
        system=system,
        messages=messages,
        output_config={"effort": args.effort},
    )
    try:
        if args.no_fallback:
            response = client.messages.create(**kwargs)
        else:
            # On a safety decline the API re-runs the request on a fallback model.
            # `served_by` records which model actually answered.
            response = client.beta.messages.create(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
    except anthropic.APIStatusError as e:
        return {**record, "pred": "", "raw": "", "error": f"{e.status_code}: {e.message}"}
    except anthropic.APIConnectionError as e:
        return {**record, "pred": "", "raw": "", "error": f"connection: {e}"}

    if response.stop_reason == "refusal":
        return {**record, "pred": "", "raw": "", "served_by": response.model, "error": "refusal"}
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        **record,
        "pred": extract_sql(text),
        "raw": text,
        "served_by": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-fallback", action="store_true", help="disable server-side refusal fallback")
    args = ap.parse_args()

    rows = load_rows(args.data, args.limit)

    done: dict[str, dict] = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("error"):
                    done[rec["id"]] = rec
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")

    client = anthropic.Anthropic(max_retries=5)
    results = dict(done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(call_model, client, args, r) for r in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results[rec["id"]] = rec
            if n % 50 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in results:
                f.write(json.dumps(results[r["id"]], ensure_ascii=False) + "\n")

    errors = sum(1 for rec in results.values() if rec.get("error"))
    in_tok = sum(rec.get("input_tokens", 0) for rec in results.values())
    out_tok = sum(rec.get("output_tokens", 0) for rec in results.values())
    print(f"Wrote {args.out} | errors: {errors} (re-run to retry) | tokens in/out: {in_tok}/{out_tok}")


if __name__ == "__main__":
    main()
