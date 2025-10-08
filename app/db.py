# db.py — Postgres connector + DAL for DMC app
from __future__ import annotations

import os
from typing import Any, Dict, Optional

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
    pool_recycle=300,     # recycle after ~5 minutes idle
    future=True,          # SQLAlchemy 2.x semantics
)

# ---------------------------------
# One-time self-healing for columns (idempotent)
# ---------------------------------
def _ensure_member_columns() -> None:
    """
    Ensure members has linkedin_yes, updated_resume_yes, had_internship columns.
    - We keep resume/LinkedIn for history but you can hide them in the UI.
    - had_internship is nullable so forms can start 'blank'.
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
            ALTER TABLE members ADD COLUMN linkedin_yes BOOLEAN;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='members'
              AND column_name='updated_resume_yes'
          ) THEN
            ALTER TABLE members ADD COLUMN updated_resume_yes BOOLEAN;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='members'
              AND column_name='had_internship'
          ) THEN
            ALTER TABLE members ADD COLUMN had_internship BOOLEAN;
          END IF;
        END
        $$;
        """
    )
    try:
        with ENGINE.begin() as c:
            c.execute(sql)
    except Exception:
        # Non-fatal: if your DB user can't run DDL here, make sure init.sql ran.
        pass

# Run the guard once at import
_ensure_member_columns()

# ---------------------------------
# Utilities
# ---------------------------------
def assert_db_connects() -> bool:
    """Ping database; raise on failure."""
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1"))
    return True

def _bool_or_none(v: Any) -> Optional[bool]:
    """
    Coerce common truthy/falsey inputs to bool; return None if unknown/blank.
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
    Required keys: id, name, event_date (YYYY-MM-DD or date), location (nullable).
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
    Insert/update a member. Safe with omitted optional fields:
    - If you pass None for a field, existing DB value is preserved via COALESCE.
    - Had internship can be True/False; keep as None to leave blank.
    Expected keys (some may be None): id, first_name, last_name, classification,
    major, student_email, linkedin_yes, updated_resume_yes, had_internship, created_at.
    """
    # Normalize booleans (allow omitted)
    p = dict(payload)  # don't mutate caller dict
    p.setdefault("linkedin_yes", None)
    p.setdefault("updated_resume_yes", None)
    p.setdefault("had_internship", None)

    p["linkedin_yes"] = _bool_or_none(p.get("linkedin_yes"))
    p["updated_resume_yes"] = _bool_or_none(p.get("updated_resume_yes"))
    p["had_internship"] = _bool_or_none(p.get("had_internship"))

    sql = text(
        """
        INSERT INTO members (
          id, first_name, last_name, classification, major, student_email,
          linkedin_yes, updated_resume_yes, had_internship, created_at, updated_at
        ) VALUES (
          :id, :first_name, :last_name, :classification, :major, :student_email,
          :linkedin_yes, :updated_resume_yes, :had_internship,
          COALESCE(:created_at, NOW()), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
          first_name         = COALESCE(EXCLUDED.first_name, members.first_name),
          last_name          = COALESCE(EXCLUDED.last_name, members.last_name),
          classification     = COALESCE(EXCLUDED.classification, members.classification),
          major              = COALESCE(EXCLUDED.major, members.major),
          student_email      = COALESCE(EXCLUDED.student_email, members.student_email),
          linkedin_yes       = COALESCE(EXCLUDED.linkedin_yes, members.linkedin_yes),
          updated_resume_yes = COALESCE(EXCLUDED.updated_resume_yes, members.updated_resume_yes),
          had_internship     = COALESCE(EXCLUDED.had_internship, members.had_internship),
          updated_at         = NOW()
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, p)
