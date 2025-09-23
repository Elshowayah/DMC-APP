# db.py — single source of truth for SQLAlchemy Engine (Streamlit-first)
from __future__ import annotations
import os, logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

# ---------------------------
# Config loaders
# ---------------------------

def _choose_driver() -> str:
    """Return 'postgresql+psycopg2' if installed, else 'postgresql+psycopg'."""
    try:
        import psycopg2  # noqa: F401
        return "postgresql+psycopg2"
    except Exception:
        return "postgresql+psycopg"

def _secrets_cfg() -> Optional[Dict[str, Any]]:
    """Load from Streamlit secrets if available."""
    try:
        import streamlit as st  # type: ignore
        if "postgres" in st.secrets:
            p = st.secrets["postgres"]
            return {
                "host": p.get("host"),
                "port": int(p.get("port", 5432)),
                "user": p.get("user"),
                "password": p.get("password"),
                "database": p.get("database"),
                "sslmode": p.get("sslmode", None),
            }
    except Exception:
        pass
    return None

def _env_parts_cfg() -> Dict[str, Any]:
    """Load from PG* env vars."""
    return {
        "host": os.getenv("PGHOST"),
        "port": int(os.getenv("PGPORT", "5432")),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
        "database": os.getenv("PGDATABASE"),
        "sslmode": os.getenv("PGSSLMODE"),
    }

def _build_from_parts(cfg: Dict[str, Any]) -> str:
    missing = [k for k in ("host", "user", "password", "database") if not cfg.get(k)]
    if missing:
        raise SystemExit(
            "Database configuration incomplete. Provide either Streamlit [postgres] secrets, "
            "a full DATABASE_URL, or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE env vars."
        )
    driver = _choose_driver()
    q: Dict[str, Any] = {"connect_timeout": "10"}
    if cfg.get("sslmode"):
        q["sslmode"] = cfg["sslmode"]
    # Force SSL for common managed hosts if not explicitly set
    host = (cfg.get("host") or "").lower()
    if ("sslmode" not in q) and any(h in host for h in
        ["neon.tech", "supabase.co", "render.com", "railway.app", "elephantsql", "rds", "azure", "gcp", "aws"]
    ):
        q["sslmode"] = "require"

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

def _build_sqlalchemy_url() -> str:
    """
    ORDER OF PRECEDENCE (fixes your localhost issue):
      1) Streamlit secrets [postgres]  ← preferred on Streamlit Cloud
      2) DATABASE_URL (or PG_URL)
      3) PG* env vars
    """
    # 1) Prefer Streamlit secrets on Cloud
    scfg = _secrets_cfg()
    if scfg:
        return _build_from_parts(scfg)

    # 2) Full URL from env (e.g., Heroku-style)
    full = os.getenv("DATABASE_URL") or os.getenv("PG_URL")
    if full:
        if full.startswith("postgres://"):
            full = full.replace("postgres://", "postgresql://", 1)
        parsed = make_url(full)
        base = parsed.drivername.split("+")[0]
        driver = _choose_driver()
        if parsed.drivername == base:  # e.g. "postgresql"
            parsed = parsed.set(drivername=driver)
        q = dict(parsed.query or {})
        q.setdefault("connect_timeout", "10")
        host = (parsed.host or "").lower()
        if any(h in host for h in ["neon.tech", "supabase.co", "render.com", "railway.app", "elephantsql", "rds", "azure", "gcp", "aws"]):
            if "sslmode" not in {k.lower(): v for k, v in q.items()}:
                q["sslmode"] = "require"
        return str(parsed.set(query=q))

    # 3) Compose from PG* env pieces
    return _build_from_parts(_env_parts_cfg())

# ---------------------------
# Create the Engine
# ---------------------------
DATABASE_URL = _build_sqlalchemy_url()

ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # drop dead connections automatically
    pool_recycle=300,     # recycle every 5 minutes (nice for serverless)
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

if __name__ == "__main__":
    try:
        with ENGINE.begin() as conn:
            print("Ping:", conn.execute(text("SELECT 1")).scalar())
    except Exception:
        logging.exception("Local/Cloud DB ping failed")
        raise




