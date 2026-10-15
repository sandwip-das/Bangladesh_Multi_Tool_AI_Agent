"""Download the specified Hugging Face CSV data and build SQLite databases.

This script deliberately inspects the repository's real CSV files and their
headers. It does not create made-up fields such as beds, doctors, or ratings.
Original source columns are retained (with safe SQL column names).
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.config import DATASETS, DATA_DIR, DATABASE_DIR


def safe_identifier(value: str) -> str:
    """Return a conservative SQLite identifier from an arbitrary column name."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "unnamed_column"
    if value[0].isdigit():
        value = f"col_{value}"
    return value


def make_unique_columns(columns: Iterable[str]) -> list[str]:
    """Normalize names and resolve collisions without dropping any source field."""
    seen: dict[str, int] = {}
    output = []
    for col in columns:
        base = safe_identifier(col)
        count = seen.get(base, 0)
        seen[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count + 1}")
    return output


def infer_sqlite_type(series: pd.Series, column_name: str = "") -> str:
    """Infer a conservative SQLite type from observed values.

    Identifier-like columns remain TEXT to preserve leading zeroes and avoid
    treating phone numbers, postal codes, EIINs and other codes as quantities.
    """
    name = safe_identifier(column_name)
    identifier_tokens = ("id", "code", "phone", "mobile", "telephone", "postal", "postcode", "zip", "eiin")
    if any(token in name.split("_") for token in identifier_tokens):
        return "TEXT"
    non_null = series.dropna()
    if non_null.empty:
        return "TEXT"
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"

    # CSV columns commonly arrive as object/string; infer only when every
    # non-empty value can be parsed consistently as a number.
    as_text = non_null.astype(str).str.strip()
    numeric = pd.to_numeric(as_text.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().all():
        if (numeric % 1 == 0).all():
            return "INTEGER"
        return "REAL"
    return "TEXT"


def find_csv_files(repo_id: str) -> list[str]:
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    csvs = [
        f for f in repo_files
        if f.lower().endswith(".csv") and "/." not in f and not Path(f).name.startswith(".")
    ]
    if not csvs:
        raise FileNotFoundError(
            f"No CSV file found in Hugging Face dataset {repo_id}. "
            "Inspect the repository files and adapt the loader if it uses another format."
        )
    # Prefer a single main CSV over split fragments, but keep all CSVs available
    # for inspection. The caller combines compatible CSVs below.
    return csvs


def download_csvs(repo_id: str, dataset_key: str) -> list[Path]:
    destination = DATA_DIR / dataset_key
    destination.mkdir(parents=True, exist_ok=True)
    local_files = []
    for repo_file in find_csv_files(repo_id):
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_file,
            local_dir=str(destination),
        )
        local_files.append(Path(local))
    return local_files


def read_dataset_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        # Keep source data as strings initially so identifiers and leading
        # zeroes are not accidentally converted to numbers.
        frame = pd.read_csv(path, dtype="string", low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV data was loaded.")
    # Union of columns is handled by pandas; absent values remain null.
    return pd.concat(frames, ignore_index=True, sort=False)


def build_database(df: pd.DataFrame, db_path: Path, table_name: str) -> dict:
    if df.empty:
        raise ValueError(f"Refusing to build empty table: {table_name}")

    df = df.copy()
    original_columns = list(df.columns)
    df.columns = make_unique_columns(df.columns)
    column_types = {col: infer_sqlite_type(df[col], col) for col in df.columns}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        definitions = ", ".join(
            f'"{col}" {column_types[col]}' for col in df.columns
        )
        conn.execute(f'CREATE TABLE "{table_name}" ({definitions})')
        # Convert pandas missing values to Python None; preserve values as strings
        # unless a column was safely inferred numeric.
        records = []
        for row in df.itertuples(index=False, name=None):
            converted = []
            for col, value in zip(df.columns, row):
                if pd.isna(value):
                    converted.append(None)
                elif column_types[col] == "INTEGER":
                    try:
                        converted.append(int(float(str(value).replace(",", ""))))
                    except (ValueError, TypeError):
                        converted.append(None)
                elif column_types[col] == "REAL":
                    try:
                        converted.append(float(str(value).replace(",", "")))
                    except (ValueError, TypeError):
                        converted.append(None)
                else:
                    converted.append(str(value))
            records.append(tuple(converted))

        placeholders = ", ".join("?" for _ in df.columns)
        quoted_cols = ", ".join(f'"{c}"' for c in df.columns)
        conn.executemany(
            f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})',
            records,
        )
        conn.execute(
            f'CREATE INDEX "idx_{table_name}_rowid" ON "{table_name}" (rowid)'
        ) if False else None
        conn.commit()

    return {
        "database": str(db_path),
        "table": table_name,
        "rows": len(df),
        "columns": [
            {"source_name": source, "sqlite_name": normalized,
             "sqlite_type": column_types[normalized]}
            for source, normalized in zip(original_columns, df.columns)
        ],
    }


def build_one(dataset_key: str) -> dict:
    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    config = DATASETS[dataset_key]
    print(f"\n[{dataset_key}] Downloading: {config['repo_id']}")
    paths = download_csvs(config["repo_id"], dataset_key)
    print("CSV files found:")
    for path in paths:
        print(f"  - {path}")
    df = read_dataset_csvs(paths)
    result = build_database(df, config["db_path"], config["table_name"])
    print(f"Built {result['database']} ({result['rows']} rows)")
    print("Detected schema:")
    for column in result["columns"]:
        print(
            f"  {column['source_name']!r} -> {column['sqlite_name']} "
            f"({column['sqlite_type']})"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to build (default: all)",
    )
    args = parser.parse_args()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    selected = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    reports = [build_one(key) for key in selected]
    print("\nBuild complete.")
    for report in reports:
        print(f"- {report['table']}: {report['rows']} rows, {len(report['columns'])} columns")


if __name__ == "__main__":
    main()
