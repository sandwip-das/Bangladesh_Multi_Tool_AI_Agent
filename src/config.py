"""Project paths and dataset configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATABASE_DIR = ROOT_DIR / "databases"

DATASETS = {
    "institutions": {
        "repo_id": "Mahadih534/Institutional-Information-of-Bangladesh",
        "table_name": "institutions",
        "db_path": DATABASE_DIR / "institutions.db",
    },
    "hospitals": {
        "repo_id": "Mahadih534/all-bangladeshi-hospitals",
        "table_name": "hospitals",
        "db_path": DATABASE_DIR / "hospitals.db",
    },
    "restaurants": {
        "repo_id": "Mahadih534/Bangladeshi-Restaurant-Data",
        "table_name": "restaurants",
        "db_path": DATABASE_DIR / "restaurants.db",
    },
}

for directory in (DATA_DIR, DATABASE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
