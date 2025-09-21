# db.py — robust single source of truth for SQLAlchemy Engine
# Works with psycopg2 or psycopg (v3). Chooses the installed driver automatically.

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL, make_url

# ---------------------------
# Build DATABASE_URL safely
# ---------------------------
def _build_database_url() -> str:
    """
    Load a Postgres URL from:
      1) Streamlit secrets (DATABASE_URL)
      2) .env(.local)/env (DATABASE_URL)
      3) Individual PG* parts (PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT)

    Then:
      - normalize postgres:// → postgresql://
      - pick an installed DBAPI (psycopg2 or psycopg)
      - add `sslmode=require` for common managed hosts if not set
      - add connect_timeout=10 if not set
    """
    # 1) Streamlit secrets
    url = None
    try:
        import streamlit as st  # type: ignore
        url = st.secrets.get("DATABASE_URL")  # type: ignore[attr-defined]
    except Exception:
        pass

    # 2) .env.local / .env / env var
    if not url:
        try:
            from dotenv import load_dotenv  # pip install python-dotenv
            if os.path.exists(".env.local"):
                load_dotenv(".env.local")
            else:
                load_dotenv()
        except Exception:
            pass
        url = os.getenv("DATABASE_URL") or os.getenv("PG_URL")

    # 3) Compose from PG* if still missing (handles special chars in password)
    if not url:
        host = os.getenv("PGHOST")
        user = os.getenv("PGUSER")
        pwd  = os.getenv("PGPASSWORD")
        db   = os.getenv("PGDATABASE")
        port = int(os.getenv("PGPORT", "5432"))
        if all([host, user, pwd, db]):
            # Driver is chosen below after we detect installed packages
            url = str(URL.create(
                drivername="postgresql",  # driver suffix added later
                username=user,
                password=pwd,
                host=host,
                port=port,
                database=db,
            ))
        else:
            raise SystemExit(
                "DATABASE_URL is missing.\n"
                "Set DATABASE_URL (or PGHOST/PGUSER/PGPASSWORD/PGDATABASE[/PGPORT]) "
                "in Streamlit Secrets, environment, or .env/.env.local.\n\n"
                "Example:\n  DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname"
            )

    # Normalize legacy scheme
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Decide the DBAPI driver we have
    have_psycopg2 = False
    have_psycopg3 = False
    try:
        import psycopg2  # noqa
        have_psycopg2 = True
    except Exception:
        pass
    try:
        import psycopg  # noqa
        have_psycopg3 = True
    except Exception:
        pass

    driver = "postgresql+psycopg2" if have_psycopg2 else "postgresql+psycopg"

    parsed = make_url(url)

    # If no explicit driver given, or just "postgresql", apply the chosen driver
    base_driver = parsed.drivername.split("+")[0] if parsed.drivername else "postgresql"
    if parsed.drivername == base_driver:
        parsed = URL.create(
            drivername=driver,
            username=parsed.username,
            password=parsed.password,
            host=parsed.host,
            port=parsed.port or 5432,
            database=parsed.database,
            query=dict(parsed.query) if parsed.query else {},
        )

    # Enforce SSL on common managed hosts (unless explicitly set)
    q = dict(parsed.query)
    host = (parsed.host or "").lower()
    needs_ssl = any(x in host for x in ["neon.tech", "supabase.co", "render.com", "rds", "aws", "azure", "gcp"])
    if needs_ssl and "sslmode" not in {k.lower(): v for k, v in q.items()}:
        q["sslmode"] = "require"

    # Faster failures on bad networks
    q.setdefault("connect_timeout", "10")

    parsed = parsed.set(query=q)
    return str(parsed)

DATABASE_URL = _build_database_url()

# ---------------------------
# Create the Engine
# ---------------------------
ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # drop dead connections automatically
    pool_recycle=300,     # recycle every 5 minutes (helps on serverless)
    future=True,
)

# Optional: quick connectivity assertion for logs (call from apps if needed)
def assert_db_connects() -> bool:
    try:
        with ENGINE.connect() as c:
            c.execute(text("SELECT 1")).scalar_one()
        return True
    except Exception:
        logging.exception("DB connectivity check failed")
        raise

# ---------------------------
# Events
# ---------------------------
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
            SET name=EXCLUDED.name,
                event_date=EXCLUDED.event_date,
                location=EXCLUDED.location
        """), ev)

# ---------------------------
# Members
# ---------------------------
def upsert_member(m: Dict[str, Any]) -> None:
    """
    Upsert by primary key (id). Admin/Check-in ensure non-null id:
      - Prefer provided V-number as id, else a generated m_<uuid>.
    """
    m.setdefault("created_at", None)               # DB default NOW() if None
    m.setdefault("updated_at", datetime.utcnow())  # always refresh updated_at
    with ENGINE.begin() as c:
        c.execute(text("""
            INSERT INTO members (id, first_name, last_name, classification, major, v_number,
                                 student_email, personal_email, created_at, updated_at)
            VALUES (:id, :first_name, :last_name, :classification, :major, :v_number,
                    :student_email, :personal_email,
                    COALESCE(:created_at, NOW()), :updated_at)
            ON CONFLICT (id) DO UPDATE SET
                first_name      = EXCLUDED.first_name,
                last_name       = EXCLUDED.last_name,
                classification  = EXCLUDED.classification,
                major           = EXCLUDED.major,
                v_number        = EXCLUDED.v_number,
                student_email   = EXCLUDED.student_email,
                personal_email  = EXCLUDED.personal_email,
                updated_at      = EXCLUDED.updated_at
        """), m)

def find_member_by_v(v_number: str) -> Optional[Dict[str, Any]]:
    with ENGINE.begin() as c:
        r = c.execute(text("""
            SELECT *
            FROM members
            WHERE v_number = :v
            LIMIT 1
        """), {"v": v_number}).mappings().first()
    return dict(r) if r else None

# ---------------------------
# Attendance
# ---------------------------
def check_in(event_id: str, member_id: str, method: Optional[str] = None) -> None:
    """
    Bare insert into attendance. (Your checkin.py handles duplicate detection
    before calling this, so we don't do ON CONFLICT here.)
    """
    with ENGINE.begin() as c:
        c.execute(text("""
            INSERT INTO attendance (event_id, member_id, method)
            VALUES (:event_id, :member_id, :method)
        """), {"event_id": event_id, "member_id": member_id, "method": method})

def latest_checkins(limit: int = 200) -> List[Dict[str, Any]]:
    with ENGINE.begin() as c:
        rows = c.execute(text("""
            SELECT a.event_id,
                   e.name        AS event_name,
                   a.member_id,
                   m.first_name,
                   m.last_name,
                   a.checked_in_at,
                   a.method
            FROM attendance a
            JOIN events  e ON e.id = a.event_id
            JOIN members m ON m.id = a.member_id
            ORDER BY a.checked_in_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return [dict(r) for r in rows]

# Handy local test: `python db.py`
if __name__ == "__main__":
    from sqlalchemy import text as _text
    try:
        with ENGINE.begin() as conn:
            print("Ping:", conn.execute(_text("SELECT 1")).scalar())
    except Exception:
        logging.exception("Local DB ping failed")
        raise



