# db.py — Postgres connector + DAL for DMC app
from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

# ---------------------------------
# Resolve DATABASE_URL
# ---------------------------------
def _get_db_url() -> str:
    """
    Resolve Postgres URL from Streamlit secrets or environment (.env.local optional).
    Example (Neon):
      postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require&channel_binding=require
    """
    url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv  # optional for local dev
            load_dotenv(".env.local")
            url = os.getenv("DATABASE_URL")
        except Exception:
            pass
    if not url:
        raise RuntimeError(
            "DATABASE_URL not found. Set it in .streamlit/secrets.toml or .env.local."
        )
    return url

# ---------------------------------
# Engine (module global)
# ---------------------------------
ENGINE: Engine = create_engine(
    _get_db_url(),
    pool_pre_ping=True,   # avoids stale connections
    pool_recycle=300,     # recycle after 5 minutes idle
    future=True,          # SQLAlchemy 2.x semantics
)

# ---------------------------------
# One-time self-healing for new columns (safe NOOP if already present)
# ---------------------------------
def _ensure_member_flags_columns() -> None:
    """
    Make sure members.linkedin_yes and members.updated_resume_yes exist.
    This is idempotent and safe to run at import.
    """
    sql = text(
        """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='members'
              AND column_name='linkedin_yes'
          ) THEN
            ALTER TABLE members ADD COLUMN linkedin_yes BOOLEAN DEFAULT FALSE;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='members'
              AND column_name='updated_resume_yes'
          ) THEN
            ALTER TABLE members ADD COLUMN updated_resume_yes BOOLEAN DEFAULT FALSE;
          END IF;
        END
        $$;
        """
    )
    try:
        with ENGINE.begin() as c:
            c.execute(sql)
    except Exception:
        # Non-fatal: if your DB user can't run DDL here, just make sure init.sql ran.
        pass

# Run the guard once at import
_ensure_member_flags_columns()

# ---------------------------------
# Utilities
# ---------------------------------
def assert_db_connects() -> bool:
    """Ping database; raise on failure."""
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1"))
    return True

def _bool_or_none(v: Any):
    """
    Coerce common truthy/falsey inputs to bool, else return None.
    Accepts: True/False, 'yes'/'no', 'y'/'n', '1'/'0', 1/0, 'true'/'false'.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "1"):
        return True
    if s in ("n", "no", "false", "0"):
        return False
    return None

# Optional: handy caption for UI debug
def dsn_caption() -> str:
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        return f"DB → host={u.host or '<none>'} db={u.database or '<none>'} user={u.username or '<none>'}"
    except Exception as e:
        return f"DB → (unavailable: {type(e).__name__})"

# ---------------------------------
# DAL functions your app imports
# ---------------------------------
def create_event(payload: Dict[str, Any]) -> None:
    """
    Insert or update an event by id.
    Required keys in payload: id, name, event_date (YYYY-MM-DD or date), location (nullable).
    """
    sql = text(
        """
        INSERT INTO events (id, name, event_date, location)
        VALUES (:id, :name, :event_date, :location)
        ON CONFLICT (id) DO UPDATE SET
          name       = EXCLUDED.name,
          event_date = EXCLUDED.event_date,
          location   = EXCLUDED.location
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, payload)

def upsert_member(payload: Dict[str, Any]) -> None:
    """
    Insert or update a member.

    Expected keys:
      id (str), first_name (str), last_name (str)
      classification (str|None), major (str|None), student_email (str|None)
      linkedin_yes (bool|str|int|None)
      updated_resume_yes (bool|str|int|None)
      created_at (datetime|None)  # optional; defaults to NOW() if None
    """
    # Normalize the two flags so UI "Yes/No" works even if passed as strings
    payload = dict(payload)  # shallow copy (don't mutate caller's dict)
    payload["linkedin_yes"] = _bool_or_none(payload.get("linkedin_yes"))
    payload["updated_resume_yes"] = _bool_or_none(payload.get("updated_resume_yes"))

    sql = text(
        """
        INSERT INTO members (
          id, first_name, last_name, classification, major, student_email,
          linkedin_yes, updated_resume_yes,
          created_at
        ) VALUES (
          :id, :first_name, :last_name, :classification, :major, :student_email,
          :linkedin_yes, :updated_resume_yes,
          COALESCE(:created_at, NOW())
        )
        ON CONFLICT (id) DO UPDATE SET
          first_name         = EXCLUDED.first_name,
          last_name          = EXCLUDED.last_name,
          classification     = EXCLUDED.classification,
          major              = EXCLUDED.major,
          student_email      = EXCLUDED.student_email,
          linkedin_yes       = EXCLUDED.linkedin_yes,
          updated_resume_yes = EXCLUDED.updated_resume_yes,
          updated_at         = NOW()
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, payload)








