# db.py — single source of truth for SQLAlchemy Engine
# Reads .streamlit/secrets.toml [postgres] first, then env vars.
# Auto-selects psycopg2 or psycopg (v3). Safe to use special chars in passwords.

from __future__ import annotations
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
from sqlalchemy.engine import make_url

# ---------------------------
# Config loaders
# ---------------------------
def _load_cfg() -> Dict[str, Any]:
    """Prefer Streamlit secrets, then env vars."""
    cfg = {}
    # Streamlit secrets
    try:
        import streamlit as st  # type: ignore
        if "postgres" in st.secrets:
            p = st.secrets["postgres"]
            cfg = {
                "host": p.get("host"),
                "port": int(p.get("port", 5432)),
                "user": p.get("user"),
                "password": p.get("password"),
                "database": p.get("database"),
                "sslmode": p.get("sslmode", None),
            }
    except Exception:
        pass

    # Env fallback
    cfg.setdefault("host", os.getenv("PGHOST"))
    cfg.setdefault("port", int(os.getenv("PGPORT", "5432")))
    cfg.setdefault("user", os.getenv("PGUSER"))
    cfg.setdefault("password", os.getenv("PGPASSWORD"))
    cfg.setdefault("database", os.getenv("PGDATABASE"))
    cfg.setdefault("sslmode", os.getenv("PGSSLMODE"))

    return cfg

def _choose_driver() -> str:
    """Return 'postgresql+psycopg2' if installed, else 'postgresql+psycopg'."""
    try:
        import psycopg2  # noqa: F401
        return "postgresql+psycopg2"
    except Exception:
        return "postgresql+psycopg"

def _build_sqlalchemy_url() -> str:
    """
    Build a SQLAlchemy URL safely from secrets/env parts (preferred),
    or DATABASE_URL if provided. Adds connect_timeout=10 and sslmode when needed.
    """
    # If a full DATABASE_URL is provided, honor it
    full = os.getenv("DATABASE_URL") or os.getenv("PG_URL")
    if not full:
        # Also support Streamlit secret DATABASE_URL if someone set it
        try:
            import streamlit as st  # type: ignore
            full = st.secrets.get("DATABASE_URL")  # type: ignore[attr-defined]
        except Exception:
            pass

    if full:
        if full.startswith("postgres://"):
            full = full.replace("postgres://", "postgresql://", 1)
        # Ensure a DBAPI suffix is present
        parsed = make_url(full)
        base = parsed.drivername.split("+")[0]
        driver = _choose_driver()
        if parsed.drivername == base:  # e.g. just "postgresql"
            parsed = parsed.set(drivername=driver)
        q = dict(parsed.query or {})
        q.setdefault("connect_timeout", "10")
        # Add sslmode=require for common managed hosts if absent
        host = (parsed.host or "").lower()
        if any(h in host for h in ["neon.tech", "supabase.co", "render.com", "rds", "aws", "azure", "gcp"]):
            if "sslmode" not in {k.lower(): v for k, v in q.items()}:
                q["sslmode"] = "require"
        return str(parsed.set(query=q))

    # Otherwise compose from parts (secrets/env)
    cfg = _load_cfg()
    missing = [k for k in ("host", "user", "password", "database") if not cfg.get(k)]
    if missing:
        raise SystemExit(
            "Database configuration incomplete. Provide either DATABASE_URL, or "
            "[postgres] host/port/user/password/database in .streamlit/secrets.toml, "
            "or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE in the environment."
        )

    driver = _choose_driver()
    q: Dict[str, Any] = {"connect_timeout": "10"}
    if cfg.get("sslmode"):
        q["sslmode"] = cfg["sslmode"]

    url = URL.create(
        drivername=driver,
        username=str(cfg["user"]),
        password=str(cfg["password"]),
        host=str(cfg["host"]),
        port=int(cfg.get("port", 5432)),
        database=str(cfg["database"]),
        query=q,
    )
    return str(url)

# ---------------------------
# Create the Engine
# ---------------------------
DATABASE_URL = _build_sqlalchemy_url()

ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # drop dead connections automatically
    pool_recycle=300,     # recycle every 5 minutes (nice for dev/serverless)
    future=True,
)

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
    m.setdefault("created_at", None)               # DB default NOW() if None
    m.setdefault("updated_at", datetime.utcnow())  # refresh updated_at
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
    try:
        with ENGINE.begin() as conn:
            print("Ping:", conn.execute(text("SELECT 1")).scalar())
    except Exception:
        logging.exception("Local DB ping failed")
        raise



