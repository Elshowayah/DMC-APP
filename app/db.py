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
    Resolve Postgres URL from Streamlit secrets or env (.env.local optional).
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
# Normalizers
# ---------------------------------
_HOODIE_CHOICES = ("small", "medium", "large", "xl", "2xl")

def _normalize_hoodie_size(val: Optional[str]) -> str:
    """
    Normalize user inputs to one of: small, medium, large, xl, 2xl.
    Defaults to 'medium' if unknown/blank.
    """
    v = (val or "").strip().lower().replace(" ", "")
    mapping = {
        # small
        "s": "small", "sm": "small", "small": "small",
        # medium
        "m": "medium", "med": "medium", "medium": "medium",
        # large
        "l": "large", "lg": "large", "large": "large",
        # xl
        "xl": "xl", "xlarge": "xl", "extralarge": "xl",
        # 2xl
        "2x": "2xl", "2xl": "2xl", "xxl": "2xl", "doublexl": "2xl",
    }
    out = mapping.get(v, v)
    return out if out in _HoodieChoicesSet else "medium"

_HoodieChoicesSet = set(_HOODIE_CHOICES)

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

# ---------------------------------
# One-time self-healing for columns (idempotent)
# ---------------------------------
def _ensure_member_columns() -> None:
    """
    Ensure members has the columns we rely on.
    - linkedin_yes, updated_resume_yes (BOOLEAN NOT NULL DEFAULT FALSE)
    - had_internship (BOOLEAN NULL)
    - personal_email, v_number (TEXT)
    - created_at, updated_at (TIMESTAMPTZ with defaults)
    - hoodie_size (TEXT NOT NULL DEFAULT 'medium' + CHECK constraint)
    Installs/ensures an updated_at trigger if possible.
    """
    ddl = text(
        """
        DO $$
        BEGIN
          -- columns
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='linkedin_yes'
          ) THEN
            ALTER TABLE members ADD COLUMN linkedin_yes BOOLEAN NOT NULL DEFAULT FALSE;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='updated_resume_yes'
          ) THEN
            ALTER TABLE members ADD COLUMN updated_resume_yes BOOLEAN NOT NULL DEFAULT FALSE;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='had_internship'
          ) THEN
            ALTER TABLE members ADD COLUMN had_internship BOOLEAN;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='personal_email'
          ) THEN
            ALTER TABLE members ADD COLUMN personal_email TEXT;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='v_number'
          ) THEN
            ALTER TABLE members ADD COLUMN v_number TEXT;
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='created_at'
          ) THEN
            ALTER TABLE members ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
          END IF;

          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='updated_at'
          ) THEN
            ALTER TABLE members ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
          END IF;

          -- hoodie_size column
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='members' AND column_name='hoodie_size'
          ) THEN
            ALTER TABLE members ADD COLUMN hoodie_size TEXT;
          END IF;

          -- normalize/backfill hoodie_size values
          UPDATE members
             SET hoodie_size = LOWER(REPLACE(hoodie_size,' ',''))
           WHERE hoodie_size IS NOT NULL;

          UPDATE members SET hoodie_size = 'xl'
           WHERE hoodie_size IN ('xlarge','extra large','xl ');
          UPDATE members SET hoodie_size = '2xl'
           WHERE hoodie_size IN ('xxl','2x','doublexl');

          UPDATE members
             SET hoodie_size = 'medium'
           WHERE hoodie_size IS NULL
              OR hoodie_size NOT IN ('small','medium','large','xl','2xl');

          -- default + not null + check constraint
          ALTER TABLE members
            ALTER COLUMN hoodie_size SET DEFAULT 'medium',
            ALTER COLUMN hoodie_size SET NOT NULL;

          IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname='members_hoodie_size_chk'
              AND conrelid='members'::regclass
          ) THEN
            ALTER TABLE members
              ADD CONSTRAINT members_hoodie_size_chk
              CHECK (hoodie_size IN ('small','medium','large','xl','2xl'));
          END IF;

          -- updated_at trigger (function)
          IF NOT EXISTS (
            SELECT 1 FROM pg_proc WHERE proname = 'tg_members_set_updated_at'
          ) THEN
            CREATE OR REPLACE FUNCTION tg_members_set_updated_at()
            RETURNS TRIGGER AS $fn$
            BEGIN
              NEW.updated_at := NOW();
              RETURN NEW;
            END
            $fn$ LANGUAGE plpgsql;
          END IF;

          -- updated_at trigger (hook)
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger WHERE tgname = 'trg_members_set_updated_at'
          ) THEN
            CREATE TRIGGER trg_members_set_updated_at
            BEFORE UPDATE ON members
            FOR EACH ROW
            EXECUTE FUNCTION tg_members_set_updated_at();
          END IF;
        END
        $$;
        """
    )
    try:
        with ENGINE.begin() as c:
            c.execute(ddl)
    except Exception:
        # Non-fatal if DB user cannot run DDL; assume init.sql already handled schema.
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

def dsn_caption() -> str:
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        return f"DB → host={u.host or '<none>'} db={u.database or '<none>'} user={u.username or '<none>'}"
    except Exception as e:
        return f"DB → (unavailable: {type(e).__name__})"

# ---------------------------------
# Fetchers (for prefill / history)
# ---------------------------------
def get_member(member_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a member row by id, with booleans and core fields, or None if missing.
    """
    sql = text(
        """
        SELECT id, first_name, last_name, classification, major,
               student_email, personal_email, v_number,
               linkedin_yes, updated_resume_yes, had_internship,
               hoodie_size,
               created_at, updated_at
        FROM members
        WHERE id = :id
        """
    )
    with ENGINE.connect() as c:
        row = c.execute(sql, {"id": member_id}).mappings().first()
        return dict(row) if row else None

def get_member_by_email(student_email: str) -> Optional[Dict[str, Any]]:
    sql = text(
        """
        SELECT id, first_name, last_name, classification, major,
               student_email, personal_email, v_number,
               linkedin_yes, updated_resume_yes, had_internship,
               hoodie_size,
               created_at, updated_at
        FROM members
        WHERE student_email = :student_email
        """
    )
    with ENGINE.connect() as c:
        row = c.execute(sql, {"student_email": student_email}).mappings().first()
        return dict(row) if row else None

def last_attendance(member_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent attendance record for a member (if any).
    """
    sql = text(
        """
        SELECT a.event_id, a.checked_in_at, a.method,
               e.name AS event_name, e.event_date, e.location
        FROM attendance a
        LEFT JOIN events e ON e.id = a.event_id
        WHERE a.member_id = :member_id
        ORDER BY a.checked_in_at DESC
        LIMIT 1
        """
    )
    with ENGINE.connect() as c:
        row = c.execute(sql, {"member_id": member_id}).mappings().first()
        return dict(row) if row else None

# ---------------------------------
# Event create/update
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

# ---------------------------------
# Member upsert (preserves existing values when incoming is blank)
# ---------------------------------
def upsert_member(payload: Dict[str, Any]) -> None:
    """
    Insert/update a member. Safe with omitted optional fields:
    - If you pass None for a field, existing DB value is preserved via COALESCE.
    - had_internship can be True/False; keep as None to leave blank.
    - hoodie_size will be normalized to small|medium|large|xl|2xl (default 'medium').
    """
    p = dict(payload)  # don't mutate caller dict

    # Optional text fields
    p.setdefault("v_number", None)
    p.setdefault("personal_email", None)
    # Normalize hoodie size (text)
    p["hoodie_size"] = _normalize_hoodie_size(p.get("hoodie_size"))

    # Optional boolean fields
    p.setdefault("linkedin_yes", None)
    p.setdefault("updated_resume_yes", None)
    p.setdefault("had_internship", None)

    # Normalize booleans
    p["linkedin_yes"]       = _bool_or_none(p.get("linkedin_yes"))
    p["updated_resume_yes"] = _bool_or_none(p.get("updated_resume_yes"))
    p["had_internship"]     = _bool_or_none(p.get("had_internship"))

    sql = text(
        """
        INSERT INTO members (
          id, first_name, last_name, classification, major,
          student_email, v_number, personal_email,
          linkedin_yes, updated_resume_yes, had_internship,
          hoodie_size,
          created_at, updated_at
        ) VALUES (
          :id, :first_name, :last_name, :classification, :major,
          :student_email, :v_number, :personal_email,
          :linkedin_yes, :updated_resume_yes, :had_internship,
          COALESCE(:hoodie_size, 'medium'),
          COALESCE(:created_at, NOW()), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
          first_name         = COALESCE(EXCLUDED.first_name,         members.first_name),
          last_name          = COALESCE(EXCLUDED.last_name,          members.last_name),
          classification     = COALESCE(EXCLUDED.classification,     members.classification),
          major              = COALESCE(EXCLUDED.major,              members.major),
          student_email      = COALESCE(EXCLUDED.student_email,      members.student_email),
          v_number           = COALESCE(EXCLUDED.v_number,           members.v_number),
          personal_email     = COALESCE(EXCLUDED.personal_email,     members.personal_email),
          linkedin_yes       = COALESCE(EXCLUDED.linkedin_yes,       members.linkedin_yes),
          updated_resume_yes = COALESCE(EXCLUDED.updated_resume_yes, members.updated_resume_yes),
          had_internship     = COALESCE(EXCLUDED.had_internship,     members.had_internship),
          hoodie_size        = COALESCE(EXCLUDED.hoodie_size,        members.hoodie_size),
          updated_at         = NOW()
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, p)


# Check-in flow: persist member fields + attendance, and return prefill
def check_in(event_id: str, member_payload: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
    """
    1) Upserts member (including linkedin_yes, updated_resume_yes, had_internship, hoodie_size).
    2) Inserts an attendance record.
    3) Returns the current member profile (to prefill the UI next time).

    Required member_payload keys:
      - id, first_name, last_name
      - classification (nullable), major (nullable), student_email (nullable)
      - linkedin_yes (bool/str/int), updated_resume_yes (bool/str/int), had_internship (bool/str/int or None)
      - hoodie_size (optional text; normalized)
    """
    # Persist the latest member fields
    upsert_member(member_payload)

    # Insert attendance
    sql_attend = text(
        """
        INSERT INTO attendance (event_id, member_id, method)
        VALUES (:event_id, :member_id, :method)
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql_attend, {"event_id": event_id, "member_id": member_payload["id"], "method": method})

    # Return fresh profile (includes past values to prefill UI)
    profile = get_member(member_payload["id"]) or {}
    profile["_last_attendance"] = last_attendance(member_payload["id"])
    return profile

# ---------------------------------
# Prefill helper: unify lookup by id or student_email
# ---------------------------------
def prefill_for_member(member_id: Optional[str] = None, student_email: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the member's saved fields to pre-populate the form.
    - Prefer id if provided, else look up by student_email.
    Returns a dict with linkedin_yes, updated_resume_yes, had_internship, hoodie_size, etc., or None.
    """
    row: Optional[Dict[str, Any]] = None
    if member_id:
        row = get_member(member_id)
    elif student_email:
        row = get_member_by_email(student_email)

    if not row:
        return None

    row["_last_attendance"] = last_attendance(row["id"])
    return row
