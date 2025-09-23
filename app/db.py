# db.py — Neon/Postgres connector + DAL for DMC app
from __future__ import annotations
import os
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from typing import Dict


def _get_db_url() -> str:
    """
    Resolve the Postgres connection string from Streamlit secrets or .env.local
    """
    url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env.local")
            url = os.getenv("DATABASE_URL")
        except Exception:
            pass
    if not url:
        raise RuntimeError(
            "DATABASE_URL not found. Set it in .streamlit/secrets.toml or .env.local."
        )
    return url


# ---- Create SQLAlchemy Engine for Neon ----
ENGINE: Engine = create_engine(
    _get_db_url(),
    pool_pre_ping=True,
    pool_recycle=300,
    future=True,
)


def assert_db_connects() -> bool:
    """Ping database; raise on failure."""
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1"))
    return True


# ---------- DAL functions ----------

def create_event(payload: Dict) -> None:
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


def upsert_member(payload: Dict) -> None:
    """
    Insert or update a member.
    Expected keys: id, first_name, last_name, classification, major, student_email, created_at (optional).
    """
    sql = text(
        """
        INSERT INTO members (
          id, first_name, last_name, classification, major, student_email, created_at
        ) VALUES (
          :id, :first_name, :last_name, :classification, :major, :student_email,
          COALESCE(:created_at, NOW())
        )
        ON CONFLICT (id) DO UPDATE SET
          first_name     = EXCLUDED.first_name,
          last_name      = EXCLUDED.last_name,
          classification = EXCLUDED.classification,
          major          = EXCLUDED.major,
          student_email  = EXCLUDED.student_email,
          updated_at     = NOW()
        """
    )
    with ENGINE.begin() as c:
        c.execute(sql, payload)






