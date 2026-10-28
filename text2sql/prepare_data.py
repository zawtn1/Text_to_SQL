"""Convert Spider into prompt/completion JSONL files.

Expected layout of --spider-dir (the official Spider 1.0 release):
    train_spider.json  train_others.json  dev.json  tables.json  database/<db_id>/<db_id>.sqlite

Outputs in --out-dir:
    train.jsonl  training examples
    val.jsonl    held-out slice of train, for eval loss during training
    dev.jsonl    Spider dev set, used as the test set for all three setups

Usage:
    python -m text2sql.prepare_data --spider-dir /kaggle/input/spider/spider --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from text2sql.prompts import build_messages
from text2sql.schema import get_sqlite_schema_text
from text2sql.sql_utils import normalize_sql


def load_split(spider_dir: Path, files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        path = spider_dir / f
        if not path.exists():
            print(f"  skipping missing {path}")
            continue
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def to_records(rows: list[dict], db_dir: Path, split: str, sample_rows: int, dialect: str) -> list[dict]:
    records, missing = [], set()
    for i, ex in enumerate(rows):
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            missing.add(ex["db_id"])
            continue
        schema = get_sqlite_schema_text(str(db_path), sample_rows)
        query = normalize_sql(ex["query"])
        records.append({
            "id": f"{split}_{i}",
            "db_id": ex["db_id"],
            "question": ex["question"],
            "query": query,
            "prompt": build_messages(schema, ex["question"], dialect),
            "completion": [{"role": "assistant", "content": query}],
        })
    if missing:
        print(f"  {split}: skipped examples from {len(missing)} missing databases: {sorted(missing)[:5]}...")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>6} -> {path}")


def report_lengths(tokenizer_name: str, records: list[dict], max_length: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = sorted(
        len(tok(tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False),
                add_special_tokens=False)["input_ids"])
        for r in records
    )
    n = len(lengths)
    over = sum(l > max_length for l in lengths)
    print(
        f"  tokens: median={lengths[n // 2]} p95={lengths[int(n * 0.95)]} max={lengths[-1]} "
        f"| {over} examples ({over / n:.1%}) exceed max_length={max_length}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spider-dir", type=Path, required=True)
    ap.add_argument("--db-dir", type=Path, help="defaults to <spider-dir>/database")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--sample-rows", type=int, default=3, help="example rows per table in the schema (0 = none)")
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--dialect", default="SQLite")
    ap.add_argument("--no-train-others", action="store_true", help="exclude train_others.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", help="if set, report token length stats with this tokenizer")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    db_dir = args.db_dir or args.spider_dir / "database"
    train_files = ["train_spider.json"] + ([] if args.no_train_others else ["train_others.json"])

    print("Building train...")
    train = to_records(load_split(args.spider_dir, train_files), db_dir, "train", args.sample_rows, args.dialect)
    print("Building dev...")
    dev = to_records(load_split(args.spider_dir, ["dev.json"]), db_dir, "dev", args.sample_rows, args.dialect)

    random.Random(args.seed).shuffle(train)
    val, train = train[: args.val_size], train[args.val_size:]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    write_jsonl(args.out_dir / "dev.jsonl", dev)

    if args.tokenizer:
        report_lengths(args.tokenizer, train, args.max_length)


if __name__ == "__main__":
    main()
