import csv
import os
from datetime import datetime
from typing import Optional, List, Dict

# Resolve the /data folder at the project root:
# .../org_attendance/src/utils/__init__.py  -> up 3 -> .../org_attendance/data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)

def _path(name: str) -> str:
    """Return absolute path to a file in the /data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)

def read_csv(name: str) -> List[Dict]:
    """
    Read a CSV from /data into a list of dicts.
    If the file doesn't exist, return an empty list.
    """
    p = _path(name)
    if not os.path.exists(p) or os.stat(p).st_size == 0:
        return []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]

def write_csv(name: str, rows: List[Dict]) -> None:
    """
    Write a list of dicts to /data/<name>, replacing the file.
    Writes headers inferred from the first row (or nothing if empty).
    """
    p = _path(name)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

def append_csv(name: str, row: Dict) -> None:
    """
    Append a single dict row to /data/<name>. Creates the file with headers if needed.
    """
    p = _path(name)
    exists = os.path.exists(p) and os.path.getsize(p) > 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def next_id(rows: List[Dict]) -> int:
    """Return the next integer ID given existing rows that have an 'id' key."""
    if not rows:
        return 1
    return max(int(r.get("id", 0) or 0) for r in rows) + 1

def now_iso() -> str:
    """UTC timestamp in ISO format (seconds precision)."""
    return datetime.utcnow().isoformat(timespec="seconds")

def norm_email(e: Optional[str]) -> str:
    """Normalize emails to lowercase, strip whitespace; empty string if None."""
    return (e or "").strip().lower()
