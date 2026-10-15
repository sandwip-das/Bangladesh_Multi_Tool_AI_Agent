import sqlite3
from pathlib import Path

import pytest

from src import db_tools
from src.config import DATASETS


@pytest.fixture
def sample_hospitals_db(tmp_path, monkeypatch):
    db = tmp_path / "hospitals.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE hospitals (name TEXT, district TEXT, beds INTEGER)")
        conn.executemany(
            "INSERT INTO hospitals VALUES (?, ?, ?)",
            [("Hospital A", "Dhaka", 100), ("Hospital B", "Sylhet", 50)],
        )
    config = dict(DATASETS["hospitals"])
    config["db_path"] = db
    monkeypatch.setitem(DATASETS, "hospitals", config)
    return db


def test_query_database_returns_records(sample_hospitals_db):
    result = db_tools.query_database(
        "hospitals",
        "SELECT name, district FROM hospitals WHERE district = 'Dhaka'",
    )
    assert "Hospital A" in result
    assert "Dhaka" in result


def test_query_database_rejects_write(sample_hospitals_db):
    result = db_tools.query_database("hospitals", "DELETE FROM hospitals")
    assert "Only SELECT" in result


def test_query_database_requires_correct_table(sample_hospitals_db):
    result = db_tools.query_database("hospitals", "SELECT name FROM institutions")
    assert "must read from the hospitals table" in result


def test_schema_reports_actual_columns(sample_hospitals_db):
    result = db_tools.get_database_schema("hospitals")
    assert "name (TEXT)" in result
    assert "beds (INTEGER)" in result
    assert "rows: 2" in result
