# db.py
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text

# --- Load DATABASE_URL from (priority):
# 1) Streamlit secrets (if available)
# 2) .env.local or .env (via python-dotenv)
# 3) Environment variable
DATABASE_URL = None

# 1) Try Streamlit secrets (optional)
try:
    import streamlit as st  # only present when running Streamlit
    DATABASE_URL = st.secrets.get("DATABASE_URL")  # type: ignore[attr-defined]
except Exception:
    pass

# 2) Try .env.local / .env
if not DATABASE_URL:
    try:
        from dotenv import load_dotenv  # pip install python-dotenv
        # prefer .env.local if it exists; otherwise load default .env
        if os.path.exists(".env.local"):
            load_dotenv(".env.local")
        else:
            load_dotenv()
    except Exception:
        pass
    DATABASE_URL = os.getenv("DATABASE_URL")

# 3) Validate
if not DATABASE_URL or "://" not in DATABASE_URL:
    raise SystemExit(
        "DATABASE_URL is missing or invalid.\n"
        "Set it in .env.local, environment, or Streamlit Secrets.\n\n"
        "Example:\n  DATABASE_URL=postgresql+psycopg://dmc:dmc_password@localhost:5432/dmc_db"
    )

# Create the Engine once
ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)

# ---------- EVENTS ----------
def list_events(limit: int = 200) -> List[Dict[str, Any]]:
    with ENGINE.begin() as c:
        rows = c.execute(text("""
            SELECT id, name, event_date, location, created_at
            FROM events
            ORDER BY event_date DESC, created_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]

def create_event(ev: Dict[str, Any]) -> None:
    with ENGINE.begin() as c:
        c.execute(text("""
            INSERT INTO events (id, name, event_date, location)
            VALUES (:id, :name, :event_date, :location)
            ON CONFLICT (id) DO UPDATE
            SET name=EXCLUDED.name, event_date=EXCLUDED.event_date, location=EXCLUDED.location
        """), ev)

# ---------- MEMBERS ----------
def upsert_member(m: Dict[str, Any]) -> None:
    m.setdefault("created_at", None)
    m.setdefault("updated_at", datetime.utcnow())
    with ENGINE.begin() as c:
        c.execute(text("""
            INSERT INTO members (id, first_name, last_name, classification, major, v_number,
                                 student_email, personal_email, created_at, updated_at)
            VALUES (:id, :first_name, :last_name, :classification, :major, :v_number,
                    :student_email, :personal_email, COALESCE(:created_at, NOW()), :updated_at)
            ON CONFLICT (id) DO UPDATE SET
                first_name=EXCLUDED.first_name,
                last_name =EXCLUDED.last_name,
                classification=EXCLUDED.classification,
                major=EXCLUDED.major,
                v_number=EXCLUDED.v_number,
                student_email=EXCLUDED.student_email,
                personal_email=EXCLUDED.personal_email,
                updated_at=EXCLUDED.updated_at
        """), m)

def find_member_by_v(v_number: str) -> Optional[Dict[str, Any]]:
    with ENGINE.begin() as c:
        r = c.execute(text("""
            SELECT * FROM members WHERE v_number = :v LIMIT 1
        """), {"v": v_number}).mappings().first()
        return dict(r) if r else None

# ---------- ATTENDANCE ----------
def check_in(event_id: str, member_id: str, method: Optional[str] = None) -> None:
    with ENGINE.begin() as c:
        c.execute(text("""
            INSERT INTO attendance (event_id, member_id, method)
            VALUES (:event_id, :member_id, :method)
        """), {"event_id": event_id, "member_id": member_id, "method": method})

def latest_checkins(limit: int = 200) -> List[Dict[str, Any]]:
    with ENGINE.begin() as c:
        rows = c.execute(text("""
            SELECT a.event_id, e.name AS event_name, a.member_id, m.first_name, m.last_name,
                   a.checked_in_at, a.method
            FROM attendance a
            JOIN events e  ON e.id = a.event_id
            JOIN members m ON m.id = a.member_id
            ORDER BY a.checked_in_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]

# Handy local test: `python db.py`
if __name__ == "__main__":
    from sqlalchemy import text as _text
    with ENGINE.begin() as conn:
        print("Ping:", conn.execute(_text("SELECT 1")).scalar())


