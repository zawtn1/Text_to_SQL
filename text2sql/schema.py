"""Read a database schema and render it as CREATE TABLE text for prompts.

Extraction (`read_sqlite_schema`) is kept separate from rendering
(`render_schema`) so the serving app can reuse the renderer with a
PostgreSQL reader and the model sees the same format it was trained on.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    sample_rows: list[tuple] = field(default_factory=list)


def _quote(name: str) -> str:
    return name if name.isidentifier() else '"' + name.replace('"', '""') + '"'


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    # Some Spider databases contain invalid UTF-8; don't crash on them.
    conn.text_factory = lambda b: b.decode(errors="replace")
    return conn


def read_sqlite_schema(db_path: str | Path, sample_rows: int = 0) -> list[Table]:
    conn = connect_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        tables = []
        for name in names:
            q = '"' + name.replace('"', '""') + '"'
            columns = [
                Column(name=r[1], type=(r[2] or "").upper(), primary_key=r[5] > 0)
                for r in conn.execute(f"PRAGMA table_info({q})")
            ]
            fks = [
                ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4] or "")
                for r in conn.execute(f"PRAGMA foreign_key_list({q})")
            ]
            rows: list[tuple] = []
            if sample_rows > 0:
                try:
                    rows = conn.execute(f"SELECT * FROM {q} LIMIT {int(sample_rows)}").fetchall()
                except sqlite3.Error:
                    rows = []
            tables.append(Table(name, columns, fks, rows))
        return tables
    finally:
        conn.close()


def _fmt_value(v, max_len: int) -> str:
    s = "NULL" if v is None else str(v)
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def render_schema(tables: list[Table], max_value_len: int = 40) -> str:
    blocks = []
    for t in tables:
        pk_cols = [c.name for c in t.columns if c.primary_key]
        lines = []
        for c in t.columns:
            line = f"  {_quote(c.name)}" + (f" {c.type}" if c.type else "")
            if len(pk_cols) == 1 and c.primary_key:
                line += " PRIMARY KEY"
            lines.append(line)
        if len(pk_cols) > 1:
            lines.append(f"  PRIMARY KEY ({', '.join(_quote(c) for c in pk_cols)})")
        for fk in t.foreign_keys:
            ref = f"{_quote(fk.ref_table)}({_quote(fk.ref_column)})" if fk.ref_column else _quote(fk.ref_table)
            lines.append(f"  FOREIGN KEY ({_quote(fk.column)}) REFERENCES {ref}")
        block = f"CREATE TABLE {_quote(t.name)} (\n" + ",\n".join(lines) + "\n);"
        if t.sample_rows:
            header = " | ".join(c.name for c in t.columns)
            body = "\n".join(" | ".join(_fmt_value(v, max_value_len) for v in row) for row in t.sample_rows)
            block += f"\n/* {len(t.sample_rows)} example rows from {t.name}:\n{header}\n{body}\n*/"
        blocks.append(block)
    return "\n\n".join(blocks)


@lru_cache(maxsize=512)
def get_sqlite_schema_text(db_path: str, sample_rows: int = 0) -> str:
    return render_schema(read_sqlite_schema(db_path, sample_rows))
