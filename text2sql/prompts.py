"""The single prompt format shared by training, evaluation and serving.

Every setup (base model, fine-tuned model, API model) must see exactly the
same prompt, otherwise the comparison measures prompt differences instead of
model differences.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct {dialect} queries. "
    "Given a database schema and a question, respond with a single SQL query and nothing else."
)

USER_TEMPLATE = (
    "Database schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Write a {dialect} query that answers the question."
)


def build_messages(schema: str, question: str, dialect: str = "SQLite") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(dialect=dialect)},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=question, dialect=dialect)},
    ]
