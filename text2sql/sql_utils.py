"""SQL extraction from model output, safe execution, and result comparison."""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from text2sql.schema import connect_readonly

_FENCE_RE = re.compile(r"```(?:sql|sqlite|postgresql)?\s*(.*?)(?:```|$)", re.S | re.I)
_START_RE = re.compile(r"\b(SELECT|WITH)\b", re.I)
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.I)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace (Spider gold queries contain runs of spaces)."""
    return " ".join(sql.split())


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of a model response.

    Handles bare SQL, ```sql fenced blocks, and leading prose such as
    "Here is the query: SELECT ...".
    """
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    m = _START_RE.search(text)
    if m and m.start() > 0:
        text = text[m.start():]
    text = text.strip()
    if ";" in text:
        text = text.split(";", 1)[0]
    return normalize_sql(text)


class ExecutionError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "timeout" | "error"


def execute_sqlite(db_path: str | Path, sql: str, timeout: float = 30.0) -> list[tuple]:
    """Run `sql` against a read-only connection, aborting after `timeout` seconds."""
    conn = connect_readonly(db_path)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise ExecutionError("timeout", f"query exceeded {timeout}s") from e
        raise ExecutionError("error", str(e)) from e
    except (sqlite3.Error, ValueError) as e:
        raise ExecutionError("error", str(e)) from e
    finally:
        conn.close()


def _norm_value(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def order_matters(gold_sql: str) -> bool:
    return bool(_ORDER_BY_RE.search(gold_sql))


def results_match(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    """Execution-accuracy comparison.

    Rows are compared as a multiset unless the gold query has ORDER BY, in
    which case order must match too. Column order must match exactly.
    """
    p = [tuple(_norm_value(v) for v in r) for r in pred_rows]
    g = [tuple(_norm_value(v) for v in r) for r in gold_rows]
    if ordered:
        return p == g
    return Counter(p) == Counter(g)
