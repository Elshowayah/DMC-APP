import csv, os, hashlib
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def read_csv(name: str) -> list[dict]:
    p = _path(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]

def write_csv(name: str, rows: list[dict]) -> None:
    p = _path(name)
    fieldnames = rows[0].keys() if rows else []
    # if file exists and empty write headers; else create with headers we infer
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def append_csv(name: str, row: dict) -> None:
    p = _path(name)
    exists = os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists or os.stat(p).st_size == 0:
            writer.writeheader()
        writer.writerow(row)

def next_id(rows: list[dict]) -> int:
    if not rows:
        return 1
    return max(int(r.get("id", 0) or 0) for r in rows) + 1

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def norm_email(e: str | None) -> str:
    return (e or "").strip().lower()
