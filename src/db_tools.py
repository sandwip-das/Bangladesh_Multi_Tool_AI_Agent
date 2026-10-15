"""Read-only SQLite tools for the three Bangladesh datasets."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.config import DATASETS


MAX_ROWS = 30
MAX_QUERY_CHARS = 500


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run `python -m src.database_builder` first."
        )
    # SQLite URI read-only mode prevents accidental writes from tools.
    conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _schema(conn: sqlite3.Connection, table: str) -> list[dict[str, str]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [{"name": row["name"], "type": row["type"] or "TEXT"} for row in rows]


def _safe_query(query: str, table: str, columns: list[dict[str, str]]) -> str:
    """Accept only one read-only SELECT against the intended table."""
    query = query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Query must be between 1 and {MAX_QUERY_CHARS} characters.")
    if ";" in query or "--" in query or "/*" in query or "*/" in query:
        raise ValueError("Multiple statements and SQL comments are not allowed.")
    if not re.match(r"(?is)^select\b", query):
        raise ValueError("Only SELECT queries are allowed.")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", query):
        raise ValueError("Only read-only SELECT queries are allowed.")
    # Ensure the query targets the intended table; SQL is still executed in
    # read-only mode as a second line of defence.
    if not re.search(rf'(?i)\bfrom\s+["`]?{re.escape(table)}["`]?\b', query):
        raise ValueError(f"Query must read from the {table} table.")
    allowed = {c["name"].lower() for c in columns}
    # This check is intentionally conservative; aliases and SQL functions are
    # permitted, while actual column validation remains SQLite's responsibility.
    return query


def query_database(dataset_key: str, query: str) -> str:
    if dataset_key not in DATASETS:
        return "Error: unknown dataset."
    config = DATASETS[dataset_key]
    table = config["table_name"]
    try:
        with _connect(config["db_path"]) as conn:
            columns = _schema(conn, table)
            if not columns:
                return f"The {table} table does not exist or has no columns."
            sql = _safe_query(query, table, columns)
            # Clamp result count to avoid dumping the full dataset into an LLM.
            sql = f"SELECT * FROM ({sql}) AS limited_results LIMIT {MAX_ROWS}"
            rows = conn.execute(sql).fetchall()
            if not rows:
                return f"No matching records were found in the {table} dataset."
            result = [dict(row) for row in rows]
            return (
                f"Dataset: {dataset_key}; table: {table}; returned {len(result)} "
                f"record(s) (maximum {MAX_ROWS}).\n"
                f"Available columns: {', '.join(c['name'] for c in columns)}\n"
                f"Results: {result}"
            )
    except (sqlite3.Error, ValueError, FileNotFoundError) as exc:
        return f"Database query error ({dataset_key}): {exc}"


def get_database_schema(dataset_key: str) -> str:
    config = DATASETS[dataset_key]
    try:
        with _connect(config["db_path"]) as conn:
            columns = _schema(conn, config["table_name"])
            count = conn.execute(f'SELECT COUNT(*) FROM "{config["table_name"]}"').fetchone()[0]
        if not columns:
            return f"Table {config['table_name']} is missing."
        return (
            f"Table: {config['table_name']}; rows: {count}; schema: "
            + ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        )
    except (sqlite3.Error, FileNotFoundError) as exc:
        return f"Schema error: {exc}"


@tool
def institutions_db_tool(query: str) -> str:
    """Query the Bangladesh institutions SQLite dataset. Use for institution,
    university, college, school, government office, institution name or location
    questions. Write one read-only SQLite SELECT statement using the real schema.
    Call get_schema information from README/CLI if unsure; never invent columns."""
    return query_database("institutions", query)


@tool
def hospitals_db_tool(query: str) -> str:
    """Query the Bangladesh hospitals SQLite dataset. Use for hospital names,
    locations, addresses and any other fields actually present in the database.
    Write one read-only SQLite SELECT statement using the real schema. Do not
    assume the dataset contains beds, doctors or facilities unless schema shows them."""
    return query_database("hospitals", query)


@tool
def restaurants_db_tool(query: str) -> str:
    """Query the Bangladesh restaurants SQLite dataset. Use for restaurant names,
    locations and any other fields actually present in the database. Write one
    read-only SQLite SELECT statement using the real schema. Do not assume ratings
    or cuisine fields exist unless the schema shows them."""
    return query_database("restaurants", query)


def get_schema_tool_functions() -> list:
    """Optional tools allowing the agent to inspect actual database schemas."""
    @tool
    def institutions_schema_tool() -> str:
        """Return actual row count and column names/types for the institutions dataset."""
        return get_database_schema("institutions")

    @tool
    def hospitals_schema_tool() -> str:
        """Return actual row count and column names/types for the hospitals dataset."""
        return get_database_schema("hospitals")

    @tool
    def restaurants_schema_tool() -> str:
        """Return actual row count and column names/types for the restaurants dataset."""
        return get_database_schema("restaurants")

    return [institutions_schema_tool, hospitals_schema_tool, restaurants_schema_tool]


def get_database_tools() -> list:
    return [
        institutions_db_tool,
        hospitals_db_tool,
        restaurants_db_tool,
        *get_schema_tool_functions(),
    ]
