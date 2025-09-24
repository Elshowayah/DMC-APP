# db.py — Neon/Postgres connector + DAL for DMC app
from __future__ import annotations

import os
from typing import Dict, Any

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url


# -----------------------------
# Connection resolution
# -----------------------------
def _get_db_url() -> str:
    """
    Resolve the Postgres connection string from Streamlit secrets or .env.local

    Expected formats:
      - In secrets: st.secrets["DATABASE_URL"]
      - In env:     os.environ["DATABASE_URL"]

    Example (Neon):
      postgresql://USER:PASSWORD@HOST/dbname?sslmode=require&channel_binding=require
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


# -----------------------------
# Engine (module-global)
# -----------------------------
ENGINE: Engine = create_engine(
    _get_db_url(),
    pool_pre_ping=True,   # ping before checkout; helps with stale connections
    pool_recycle=300,     # recycle after 5 min to avoid long-idle issues
    future=True,          # SQLAlchemy 2.0 style
)


# -----------------------------
# Utilities
# -----------------------------
def assert_db_connects() -> bool:
    """Ping database; raise on failure. Returns True if OK."""
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1"))
    return True


def _bool_or_none(v: Any):
    """
    Coerce common truthy/falsey inputs to bool, else return None.
    Accepts: True/False, 'yes'/'no', 'y'/'n', '1'/'0', 1/0, 'true'/'false'
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


# -----------------------------
# Data Access Layer (DAL)
# -----------------------------
def create_event(payload: Dict[str, Any]) -> None:
    """
    Insert or update an event by id.

    Required keys in payload:
      - id (str)
      - name (str)
      - event_date (YYYY-MM-DD string or date)
      - location (str | None)
    """
    sql = text(
        """
        INSERT INTO events (id, name, event_date, location)
        VALUES (:id, :name, :event_date, :location)
        ON CONFLICT (id) DO UPDATE SET
          name = EXCLUDED.name,
          event_date = EXCLUDED.event_date,
          location = EXCLUDED.location
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, payload)


def upsert_member(payload: Dict[str, Any]) -> None:
    """
    Insert or update a member.

    Expected keys (strings unless noted):
      - id (str)                                REQUIRED
      - first_name (str)                         REQUIRED
      - last_name  (str)                         REQUIRED
      - classification (str | None)
      - major          (str | None)
      - student_email  (str | None, UNIQUE)
      - linkedin_yes       (bool | str | int | None)   ← NEW
      - updated_resume_yes (bool | str | int | None)   ← NEW
      - created_at     (datetime | None)  (optional; defaults to NOW() if None)

    Notes:
      • Booleans are coerced via _bool_or_none to handle "Yes"/"No" strings.
      • On conflict by id, the record is updated and updated_at is set to NOW().
    """
    # Coerce/normalize optional booleans
    payload = dict(payload)  # copy so we don't mutate caller's dict
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


# (Optional) Handy caption for your sidebar/debugging
def dsn_caption() -> str:
    """Return a short 'host/db/user' caption of the current DB target (for UI)."""
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        user = u.username or "<none>"
        host = u.host or "<none>"
        dbn = u.database or "<none>"
        return f"DB target → host={host} db={dbn} user={user}"
    except Exception as e:
        return f"DB target → (unavailable: {type(e).__name__})"







