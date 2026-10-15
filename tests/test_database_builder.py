import sqlite3
import pandas as pd

from src.database_builder import safe_identifier, make_unique_columns, infer_sqlite_type, build_database


def test_safe_identifier():
    assert safe_identifier("Hospital Name") == "hospital_name"
    assert safe_identifier("123 District") == "col_123_district"


def test_make_unique_columns():
    assert make_unique_columns(["Name", "name", "Name "]) == ["name", "name_2", "name_3"]


def test_infer_sqlite_type():
    assert infer_sqlite_type(pd.Series(["1", "2", "3"], dtype="string")) == "INTEGER"
    assert infer_sqlite_type(pd.Series(["1.2", "2.4"], dtype="string")) == "REAL"
    assert infer_sqlite_type(pd.Series(["Dhaka", "Sylhet"], dtype="string")) == "TEXT"
    assert infer_sqlite_type(pd.Series([None, None], dtype="string")) == "TEXT"
    assert infer_sqlite_type(pd.Series(["0012", "0013"], dtype="string"), "postal_code") == "TEXT"


def test_build_database_preserves_columns_and_rows(tmp_path):
    df = pd.DataFrame({
        "Hospital Name": ["Example Hospital", "Sylhet Clinic"],
        "District": ["Dhaka", "Sylhet"],
        "Beds": ["100", "not available"],
    })
    db = tmp_path / "test.db"
    report = build_database(df, db, "hospitals")
    assert report["rows"] == 2
    assert [c["sqlite_name"] for c in report["columns"]] == ["hospital_name", "district", "beds"]
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT hospital_name, district, beds FROM hospitals ORDER BY rowid").fetchall()
    assert rows == [("Example Hospital", "Dhaka", "100"), ("Sylhet Clinic", "Sylhet", "not available")]
