# Text-to-SQL

Fine-tune a small open model (Qwen2.5-Coder) with QLoRA to turn natural-language questions into SQL, then measure how it compares to the base model and a large API model by **execution accuracy**.

Roadmap: **training + evaluation** (this code) -> serving with vLLM + FastAPI -> web UI on a demo PostgreSQL database -> deployment.

## Layout

```
text2sql/
  schema.py         read a SQLite schema -> CREATE TABLE text (+ example rows)
  prompts.py        the one prompt template used everywhere
  sql_utils.py      extract SQL from model output, read-only execution with timeout, result comparison
  prepare_data.py   Spider -> train/val/dev JSONL (prompt/completion format)
  train.py          QLoRA SFT with TRL (loss on the SQL only)
  generate.py       predictions from a local model, base or + LoRA adapter (transformers or vLLM)
  generate_api.py   predictions from Claude via the Anthropic API (same prompts)
  evaluate.py       execution accuracy + side-by-side and pairwise comparison
  merge_adapter.py  merge LoRA into the base model for serving
configs/            training configs (1.5B for fast runs, 7B for the final run)
kaggle/             notebook that runs the whole pipeline on Kaggle
```

All scripts run from the repo root as modules: `python -m text2sql.<script> --help`.

## Pipeline

```bash
# 1. Data: Spider 1.0 (train_spider.json, train_others.json, dev.json, database/)
python -m text2sql.prepare_data --spider-dir path/to/spider --tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct

# 2. Train (smoke test first)
python -m text2sql.train --config configs/qwen2.5-coder-1.5b.yaml --max-steps 5 --limit 64
python -m text2sql.train --config configs/qwen2.5-coder-1.5b.yaml

# 3. Predict on Spider dev with each setup
python -m text2sql.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out predictions/base.jsonl
python -m text2sql.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter outputs/.../final_adapter --out predictions/finetuned.jsonl
python -m text2sql.generate_api --out predictions/claude-opus-5-5.jsonl

# 4. Compare
python -m text2sql.evaluate predictions/*.jsonl --db-dir path/to/spider/database
```

## Design choices

- **Same prompt for every setup.** Prompts are built once in `prepare_data` and stored in the JSONL, so base, fine-tuned and API models all see identical input.
- **Schema as `CREATE TABLE` + 3 example rows.** Example rows help the model match literal values (e.g. `'France'` vs `'FR'`). Use `--sample-rows 0` to test whether that holds (an ablation).
- **Greedy decoding** for reproducible, comparable numbers.
- **Execution accuracy**: rows compared as a multiset, or as an ordered list when the gold query has `ORDER BY`. Column order must match. This is close to, but slightly stricter than, the official Spider metric. Use the [official test-suite evaluation](https://github.com/taoyds/test-suite-sql-eval) for numbers you want to compare against published results.
- **Outcome categories** (`correct`, `wrong_result`, `exec_error`, `timeout`, `empty`) show *how* each setup fails, not just how often.

## Kaggle notes

- T4 GPUs have no bf16 support; `train.py` detects this and uses fp16.
- Sessions stop after 12 hours. Set `hub_model_id` in the config (and an `HF_TOKEN` secret) to push checkpoints to the Hub, then use `--resume` in a new session.
- The 1.5B config runs comfortably on one T4. Use it to debug the pipeline and run ablations, then do one final 7B run.

## Ablation ideas

Base vs fine-tuned vs API is the main comparison. Other questions worth testing: example rows vs no example rows, LoRA rank (8/16/64), 1 vs 2 vs 3 epochs, 1.5B vs 7B, and Spider-only vs Spider + BIRD training data.
