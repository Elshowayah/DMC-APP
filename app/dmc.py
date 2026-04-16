# =============================
# dmc.py — Streamlit App (Check-In & Admin)
# =============================
from __future__ import annotations

from uuid import uuid4
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine.url import make_url


st.set_page_config(page_title="DMC Check-In", layout="wide", page_icon="✦")

# DMC black & gold — typography + component polish (extends [theme] in .streamlit/config.toml)
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
<style>
    :root {
        --dmc-black: #0a0a0a;
        --dmc-surface: #141414;
        --dmc-elevated: #1c1c1c;
        --dmc-gold: #D4AF37;
        --dmc-gold-dim: #9a7b2c;
        --dmc-cream: #F4F1EA;
        --dmc-muted: #a8a39a;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: "DM Sans", system-ui, sans-serif !important;
        color: var(--dmc-cream);
    }

    .stApp {
        background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(212, 175, 55, 0.08), transparent 55%),
                    linear-gradient(180deg, var(--dmc-black) 0%, #0e0e0e 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(10, 10, 10, 0.92);
        border-bottom: 1px solid rgba(212, 175, 55, 0.22);
    }

    [data-testid="stToolbar"] { visibility: visible; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(195deg, #050505 0%, #121212 45%, #0a0a0a 100%) !important;
        border-right: 2px solid rgba(212, 175, 55, 0.35);
        box-shadow: 4px 0 24px rgba(0,0,0,0.35);
    }

    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--dmc-cream) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="radio"] label {
        font-weight: 500;
    }

    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: min(1200px, 100%);
    }

    /* Hero */
    .dmc-hero-wrap {
        text-align: center;
        padding: 0.5rem 1rem 1.75rem;
        margin-bottom: 0.25rem;
    }
    .dmc-hero {
        display: inline-block;
        padding: 1.35rem 2.5rem 1.5rem;
        background: linear-gradient(145deg, var(--dmc-elevated) 0%, var(--dmc-surface) 100%);
        border: 1px solid rgba(212, 175, 55, 0.45);
        border-radius: 14px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.45),
                    inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative;
    }
    .dmc-hero::after {
        content: "";
        position: absolute;
        left: 12%;
        right: 12%;
        bottom: 0;
        height: 3px;
        border-radius: 3px;
        background: linear-gradient(90deg, transparent, var(--dmc-gold), var(--dmc-gold-dim), transparent);
        opacity: 0.9;
    }
    .dmc-hero-badge {
        font-family: "DM Sans", sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: var(--dmc-gold);
        margin-bottom: 0.35rem;
    }
    .dmc-hero-title {
        font-family: "Cormorant Garamond", Georgia, serif;
        font-size: clamp(2.1rem, 4.5vw, 2.85rem);
        font-weight: 700;
        color: var(--dmc-cream);
        line-height: 1.1;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .dmc-hero-title span { color: var(--dmc-gold); }
    .dmc-hero-sub {
        font-size: 0.95rem;
        color: var(--dmc-muted);
        margin: 0.65rem 0 0;
        font-weight: 400;
    }

    /* Headings in main */
    .main h1, .main h2, .main h3 {
        font-family: "Cormorant Garamond", Georgia, serif !important;
        color: var(--dmc-cream) !important;
        font-weight: 700 !important;
    }
    .main h2, .main h3 {
        border-left: 3px solid var(--dmc-gold);
        padding-left: 0.65rem;
        margin-top: 1.25rem;
    }

    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212,175,55,0.4), transparent);
        margin: 1.25rem 0;
    }

    /* Inputs & selects (Streamlit / Base Web) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px !important;
        border-color: rgba(212, 175, 55, 0.35) !important;
        background-color: var(--dmc-elevated) !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(212, 175, 55, 0.65) !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #e4c04e 0%, var(--dmc-gold) 100%) !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        filter: brightness(1.05);
    }

    .stButton > button[kind="secondary"] {
        border-radius: 10px !important;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
        color: var(--dmc-gold) !important;
        background: transparent !important;
        font-weight: 600 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(212, 175, 55, 0.45) !important;
        font-weight: 600 !important;
    }

    /* Alerts */
    div[data-testid="stNotification"], .stAlert {
        border-radius: 12px !important;
        border-left-width: 4px !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Radio pills in main */
    [data-baseweb="radio"] { gap: 0.35rem; }

    /* Caption / small text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--dmc-muted) !important;
    }

    /* Expander */
    details {
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 12px !important;
        background: var(--dmc-surface) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dmc-hero-wrap">
      <div class="dmc-hero">
        <div class="dmc-hero-badge">Events · Members · Attendance</div>
        <h1 class="dmc-hero-title"><span>DMC</span> Check-In</h1>
        <p class="dmc-hero-sub">Sign in fast at the door — DMC black &amp; gold.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---- from db.py ----
from db import (
    ENGINE,
    assert_db_connects,
    create_event as db_create_event,
    upsert_member as db_upsert_member,
)

# ---------------------------------
# Page + constants
# ---------------------------------
CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]
SEARCH_KEY = "checkin_search"                 
SEARCH_NONCE_KEY = "checkin_search_nonce"     # forces widget to rebuild with a new key

# ---------------------------------
# Helpers
# ---------------------------------
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return
    s = s.strip()
    return s or None

def yes_no_required(label: str, key: str, default=None):
    """
    Tri-state yes/no selector:
      - default can be True, False, or None (None = unselected)
      - returns True/False/None
    """
    options = ["— choose —", "Yes", "No"]
    if default is True:
        idx = 1
    elif default is False:
        idx = 2
    else:
        idx = 0
    choice = st.selectbox(label, options, index=idx, key=key)
    if choice == "Yes":
        return True
    if choice == "No":
        return False
    return None

# Allowed hoodie sizes (store as lowercase codes)
HOODIE_CHOICES = ["none", "small", "medium", "large", "xl", "2xl"]

def normalize_hoodie_size(val: Optional[str]) -> str:
    v = (val or "").strip().lower().replace(" ", "")
    mapping = {
        # none
        "n": "none",
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
    canon = mapping.get(v, v)
    return canon if canon in HOODIE_CHOICES else "none"


def normalize_classification(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    mapping = {"freshmen": "freshman", "sophmore": "sophomore", "jr": "junior", "sr": "senior", "alum": "alumni"}
    return v if v in CLASS_CHOICES else mapping.get(v, "freshman")


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = [ch if ch.isalnum() else "_" for ch in s]
    slug = "_".join("".join(out).split("_"))
    return slug.strip("_") or f"event_{uuid4().hex[:8]}"

def clear_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

def _dsn_caption() -> str:
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        return f"DB → host={u.host or '<none>'} db={u.database or '<none>'} user={u.username or '<none>'}"
    except Exception as e:
        return f"DB → (unavailable: {type(e).__name__})"

def flash(kind: str, msg: str) -> None:
    st.session_state["_flash"] = {"kind": kind, "msg": msg}
    st.session_state["_post_action"] = True

def show_flash_once():
    f = st.session_state.pop("_flash", None)
    if not f: return
    kind = f.get("kind", "success")
    msg = f.get("msg", "")
    if   kind == "success": st.success(msg)
    elif kind == "warning": st.warning(msg)
    elif kind == "error":   st.error(msg)
    else:                   st.info(msg)
    try:
        st.toast(msg)
    except Exception:
        pass

def _unique_nonempty_sorted(series: Optional[pd.Series]) -> List[str]:
    if series is None or series.empty:
        return []
    s = series.astype("string").fillna("").str.strip()
    vals = {v for v in s.tolist() if v and v.lower() not in ("nan", "none")}
    return sorted(vals)

def request_checkin_reset():
    st.session_state["_reset_checkin"] = True
    st.session_state[SEARCH_NONCE_KEY] = st.session_state.get(SEARCH_NONCE_KEY, 0) + 1

# ---------------------------------
# Cached queries
# ---------------------------------
@st.cache_data(ttl=5, show_spinner=False)
def list_events(limit: int = 300) -> pd.DataFrame:
    sql = """
        SELECT id, name, event_date, location
        FROM events
        ORDER BY event_date DESC, name
        LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=10, show_spinner=False)
def find_member(q: str, limit: int = 200) -> pd.DataFrame:
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=[
            "id","first_name","last_name","classification","major",
            "student_email","had_internship","linkedin_yes","updated_resume_yes","hoodie_size",
        ])
    pat = f"%{q}%"
    sql = """
        SELECT
          id,
          first_name,
          last_name,
          classification,
          major,
          student_email,
          had_internship,
          linkedin_yes,
          updated_resume_yes,
          hoodie_size
        FROM members
        WHERE
          COALESCE(first_name,'')    ILIKE :pat OR
          COALESCE(last_name,'')     ILIKE :pat OR
          COALESCE(student_email,'') ILIKE :pat
        ORDER BY last_name NULLS LAST, first_name NULLS LAST
        LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"pat": pat, "limit": limit}).mappings().all()
    return pd.DataFrame(rows)


@st.cache_data(ttl=10, show_spinner=False)
def get_member_by_id(member_id: str) -> Optional[pd.Series]:
    sql = """
        SELECT id, first_name, last_name, classification, major, student_email,
               had_internship, linkedin_yes, updated_resume_yes, hoodie_size, created_at, updated_at
        FROM members
        WHERE id = :id
        LIMIT 1
    """
    with ENGINE.begin() as c:
        row = c.execute(text(sql), {"id": member_id}).mappings().first()
    if not row:
        return None
    return pd.Series(row)

def check_in(event_id: str, member_id: str, method: str = "manual") -> Dict:
    with ENGINE.begin() as c:
        ev = c.execute(
            text("SELECT id, name, event_date, location FROM events WHERE id = :id"),
            {"id": event_id},
        ).mappings().first()
        if not ev: raise ValueError("Event not found.")

        mem = c.execute(
            text("""
                SELECT id, first_name, last_name, classification, major, student_email
                FROM members WHERE id = :id
            """),
            {"id": member_id},
        ).mappings().first()
        if not mem: raise ValueError("Member not found.")

        dup = c.execute(
            text("""
                SELECT event_id, member_id, checked_in_at, method
                FROM attendance
                WHERE event_id = :e AND member_id = :m
                ORDER BY checked_in_at DESC
                LIMIT 1
            """),
            {"e": event_id, "m": member_id},
        ).mappings().first()

        base = {
            "event_name": ev["name"],
            "event_date": ev["event_date"],
            "event_location": ev["location"],
            "member_name": f"{mem['first_name']} {mem['last_name']}".strip(),
            "member_classification": mem["classification"],
            "member_student_email": mem["student_email"],
        }

        if dup:
            return {**base, **dup, "duplicate": True}

        ins = c.execute(
            text("""
                INSERT INTO attendance (event_id, member_id, checked_in_at, method)
                VALUES (:e, :m, NOW(), :method)
                RETURNING event_id, member_id, checked_in_at, method
            """),
            {"e": event_id, "m": member_id, "method": method},
        ).mappings().first()

        return {**base, **ins, "duplicate": False}

@st.cache_data(ttl=5, show_spinner=False)
def load_databrowser(limit: int = 2000) -> pd.DataFrame:
    sql = """
        SELECT
          a.event_id,
          e.name       AS event_name,
          e.event_date AS event_date,
          e.location   AS event_location,
          a.member_id,
          m.first_name, m.last_name, m.classification, m.major,
          m.student_email, m.had_internship, m.linkedin_yes, m.updated_resume_yes, hoodie_size,
          a.checked_in_at, a.method
        FROM attendance a
        JOIN events  e ON e.id = a.event_id
        JOIN members m ON m.id = a.member_id
        ORDER BY a.checked_in_at DESC
        LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    if not rows:
        return pd.DataFrame(columns=[
            "event_id","event_name","event_date","event_location",
            "member_id","first_name","last_name","classification","major",
            "student_email","had_internship","linkedin_yes","updated_resume_yes", "hoodie_size", 
            "checked_in_at","method","member_name"
        ])
    df = pd.DataFrame(rows)
    df["member_name"] = (df.get("first_name","").fillna("") + " " + df.get("last_name","").fillna("")).str.strip()
    return df

@st.cache_data(ttl=10, show_spinner=False)
def load_members_table(limit: int = 5000) -> pd.DataFrame:
    sql = """
        SELECT id, first_name, last_name, classification, major,
               student_email, had_internship, linkedin_yes, updated_resume_yes, hoodie_size,
               created_at, updated_at
        FROM members
        ORDER BY COALESCE(updated_at, created_at) DESC NULLS LAST
        LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=10, show_spinner=False)
def load_events_index(limit: int = 2000) -> pd.DataFrame:
    sql = """
        SELECT
          e.id, e.name, e.event_date, e.location,
          (SELECT COUNT(*) FROM attendance a WHERE a.event_id = e.id) AS attendee_count
        FROM events e
        ORDER BY e.event_date DESC
        LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=10, show_spinner=False)
def load_event_attendees(event_id: str) -> pd.DataFrame:
    sql = """
        SELECT
          a.event_id, e.name AS event_name, e.event_date, e.location,
          a.member_id,
          m.first_name, m.last_name, m.classification, m.major,
          m.student_email, m.had_internship, m.linkedin_yes, m.updated_resume_yes, hoodie_size,
          a.checked_in_at, a.method
        FROM attendance a
        JOIN members m ON m.id = a.member_id
        JOIN events  e ON e.id = a.event_id
        WHERE a.event_id = :eid
        ORDER BY a.checked_in_at DESC
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"eid": event_id}).mappings().all()
    return pd.DataFrame(rows)


# ---------------------------------
# Executive reports (admin analytics)
# ---------------------------------
def _exec_event_date_filter(start_d: Optional[str], end_d: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    params: Dict[str, Any] = {}
    if start_d:
        parts.append("e.event_date >= CAST(:start_date AS DATE)")
        params["start_date"] = start_d
    if end_d:
        parts.append("e.event_date <= CAST(:end_date AS DATE)")
        params["end_date"] = end_d
    if not parts:
        return "", params
    return " AND " + " AND ".join(parts), params


def _exec_chart_layout(fig: go.Figure, title: str) -> go.Figure:
    cream = "#F4F1EA"
    gold = "#D4AF37"
    grid = "rgba(212,175,55,0.12)"
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=cream)),
        font=dict(family="DM Sans, Arial, sans-serif", color=cream, size=13),
        plot_bgcolor="#121212",
        paper_bgcolor="#141414",
        margin=dict(t=56, b=56, l=8, r=8),
        colorway=[gold, "#e8c547", "#8a7028", "#c9a227", "#5c4d1f"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=cream),
            bgcolor="rgba(20,20,20,0.85)",
            bordercolor=gold,
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
        tickfont=dict(color="#a8a39a"),
        title_font=dict(color=cream),
    )
    fig.update_yaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
        tickfont=dict(color="#a8a39a"),
        title_font=dict(color=cream),
    )
    return fig


@st.cache_data(ttl=60, show_spinner=False)
def exec_kpis(start_d: Optional[str], end_d: Optional[str]) -> pd.Series:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        SELECT
          COUNT(DISTINCT e.id) AS events_with_checkins,
          COUNT(*) AS total_check_ins,
          COUNT(DISTINCT a.member_id) AS unique_members
        FROM attendance a
        JOIN events e ON e.id = a.event_id
        WHERE 1=1 {clause}
    """
    with ENGINE.begin() as c:
        row = c.execute(text(sql), params).mappings().first()
    if not row:
        return pd.Series({"events_with_checkins": 0, "total_check_ins": 0, "unique_members": 0})
    return pd.Series(dict(row))


@st.cache_data(ttl=60, show_spinner=False)
def exec_returning_member_pct(start_d: Optional[str], end_d: Optional[str]) -> float:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        WITH per_member AS (
            SELECT a.member_id, COUNT(DISTINCT a.event_id) AS n_events
            FROM attendance a
            JOIN events e ON e.id = a.event_id
            WHERE 1=1 {clause}
            GROUP BY a.member_id
        )
        SELECT
          CASE WHEN COUNT(*) = 0 THEN 0::float
               ELSE (100.0 * SUM(CASE WHEN n_events >= 2 THEN 1 ELSE 0 END) / COUNT(*))::float
          END AS pct_returning
        FROM per_member
    """
    with ENGINE.begin() as c:
        row = c.execute(text(sql), params).mappings().first()
    if not row or row["pct_returning"] is None:
        return 0.0
    return float(row["pct_returning"])


@st.cache_data(ttl=60, show_spinner=False)
def exec_event_rollups(start_d: Optional[str], end_d: Optional[str]) -> pd.DataFrame:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        SELECT
          e.id,
          e.name,
          e.event_date,
          e.location,
          COUNT(*) AS check_ins,
          COUNT(DISTINCT a.member_id) AS unique_members
        FROM events e
        JOIN attendance a ON a.event_id = e.id
        WHERE 1=1 {clause}
        GROUP BY e.id, e.name, e.event_date, e.location
        ORDER BY e.event_date DESC, check_ins DESC
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame(
            columns=["id", "name", "event_date", "location", "check_ins", "unique_members"]
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def exec_monthly_trend(start_d: Optional[str], end_d: Optional[str]) -> pd.DataFrame:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        SELECT
          DATE_TRUNC('month', e.event_date)::date AS month_start,
          COUNT(*) AS check_ins,
          COUNT(DISTINCT a.member_id) AS unique_members
        FROM attendance a
        JOIN events e ON e.id = a.event_id
        WHERE 1=1 {clause}
        GROUP BY 1
        ORDER BY 1
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["month_start", "check_ins", "unique_members"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def exec_classification_mix(start_d: Optional[str], end_d: Optional[str]) -> pd.DataFrame:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        SELECT
          COALESCE(NULLIF(TRIM(m.classification), ''), '(not set)') AS classification,
          COUNT(*) AS check_ins
        FROM attendance a
        JOIN events e ON e.id = a.event_id
        JOIN members m ON m.id = a.member_id
        WHERE 1=1 {clause}
        GROUP BY 1
        ORDER BY check_ins DESC
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["classification", "check_ins"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def exec_method_mix(start_d: Optional[str], end_d: Optional[str]) -> pd.DataFrame:
    clause, params = _exec_event_date_filter(start_d, end_d)
    sql = f"""
        SELECT
          COALESCE(NULLIF(TRIM(a.method), ''), '(unknown)') AS method,
          COUNT(*) AS check_ins
        FROM attendance a
        JOIN events e ON e.id = a.event_id
        WHERE 1=1 {clause}
        GROUP BY 1
        ORDER BY check_ins DESC
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["method", "check_ins"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def exec_top_majors(start_d: Optional[str], end_d: Optional[str], limit: int = 12) -> pd.DataFrame:
    clause, params = _exec_event_date_filter(start_d, end_d)
    params = {**params, "lim": int(limit)}
    sql = f"""
        SELECT
          COALESCE(NULLIF(TRIM(m.major), ''), '(blank / undecided)') AS major,
          COUNT(DISTINCT a.member_id) AS unique_members,
          COUNT(*) AS check_ins
        FROM attendance a
        JOIN events e ON e.id = a.event_id
        JOIN members m ON m.id = a.member_id
        WHERE 1=1 {clause}
        GROUP BY 1
        ORDER BY unique_members DESC, check_ins DESC
        LIMIT :lim
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["major", "unique_members", "check_ins"])
    return pd.DataFrame(rows)


def delete_event(event_id: str) -> None:
    with ENGINE.begin() as c:
        c.execute(text("DELETE FROM attendance WHERE event_id = :eid"), {"eid": event_id})
        c.execute(text("DELETE FROM events WHERE id = :eid"), {"eid": event_id})

# ---------------------------------
# NAV + refresh + flash
# ---------------------------------
with st.sidebar:
    section = st.radio("Section", ["Check-In", "Admin"], index=0)
    st.caption(_dsn_caption())
    if st.button("Refresh"):
        clear_cache()
        st.session_state["_just_refreshed"] = True
        st.rerun()

# show any queued flash banner
show_flash_once()

if st.session_state.pop("_just_refreshed", False):
    st.stop()

# ---------------------------------
# CHECK-IN (PUBLIC)
# ---------------------------------
if section == "Check-In":
    if SEARCH_NONCE_KEY not in st.session_state:
        st.session_state[SEARCH_NONCE_KEY] = 0

    # DB connectivity
    try:
        assert_db_connects()
    except Exception as e:
        col1, col2 = st.columns([1, 1])
        with col1: st.error(f"Database connectivity failed: {type(e).__name__}: {e}")
        with col2:
            if st.button("↻ Reconnect (clear cache)"):
                clear_cache(); st.session_state["_just_refreshed"] = True; st.rerun()
        st.stop()

    # Events
    try:
        ev_df = list_events()
    except Exception as e:
        st.error(f"Could not load events: {type(e).__name__}: {e}")
        st.stop()

    if ev_df is None or ev_df.empty:
        st.warning("No events yet. Ask an admin to create one in the Admin Console.")
        st.stop()

    ev_df = ev_df.copy()
    ev_df["label"] = ev_df.apply(
        lambda r: f"{r['id']} — {r.get('name','(no name)')} ({str(r.get('event_date',''))}) @ {r.get('location','')}",
        axis=1,
    )
    choice = st.selectbox("Select Event", ev_df["label"].tolist())
    try:
        current_event_id = ev_df.loc[ev_df["label"] == choice, "id"].iloc[0]
    except Exception:
        st.error("Could not resolve selected event id.")
        st.stop()

    # ==================================================
    # Existing Member — print all matches (min 3 chars)
    # ==================================================
    st.divider()
    st.subheader("Existing Member — Search, Edit, and Check-In")

    sel_key = "checkin_selected_member_id"

    # One-shot UI reset requested by previous run? Do this BEFORE widgets are created.
    if st.session_state.pop("_reset_checkin", False):
        st.session_state.pop(sel_key, None)
        st.session_state.pop(SEARCH_KEY, None)

    # Build a key that changes when nonce changes ⇒ brand-new text_input instance (clears value)
    search_widget_key = f"{SEARCH_KEY}:{st.session_state.get(SEARCH_NONCE_KEY, 0)}"
    q = st.text_input(
        "Search by name or student email (min 3 characters)",
        placeholder="Start typing…",
        key=search_widget_key,
    ).strip()

    hits = pd.DataFrame()
    if 0 < len(q) < 3:
        st.info("Type at least 3 characters to search.")
    elif len(q) >= 3:
        try:
            hits = find_member(q, limit=200)
        except Exception as e:
            st.error(f"Search failed: {type(e).__name__}: {e}")
            hits = pd.DataFrame()

    if hits is not None and not hits.empty:
        st.caption(f"Found {len(hits)} matching member(s).")
        hits = hits.fillna("")
        for _, row in hits.iterrows():
            mid   = str(row.get("id","")).strip()
            fn    = (row.get("first_name") or "").strip()
            ln    = (row.get("last_name") or "").strip()
            email = (row.get("student_email") or "").strip() or "no email"
            klass = (row.get("classification") or "").strip().title() or "—"
            major = (row.get("major") or "").strip()
            meta  = " • ".join([p for p in [email, klass, major] if p])

            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{fn} {ln}**  \n{meta}  \n`{mid}`")
            with c2:
                if st.button("Select", key=f"pick_{mid}"):
                    st.session_state[sel_key] = mid
                    st.rerun()
    elif len(q) >= 3:
        st.info("No members matched your search.")

    # Selected member edit + check-in
    selected_id = st.session_state.get(sel_key)
    if selected_id:
        sel = get_member_by_id(selected_id)
        if sel is None:
            st.warning("Selected member not found anymore.")
            st.session_state.pop(sel_key, None)
        else:
            st.markdown("---")
            st.markdown("### Edit & Check-In — Selected Member")

            mid = str(sel.get("id", "")).strip()
            email_disp = sel.get("student_email") or "no email"
            klass = (sel.get("classification") or "freshman").lower()
            try:
                class_idx = CLASS_CHOICES.index(klass)
            except ValueError:
                class_idx = 0

            st.markdown(
                f"**Selected:** {sel.get('first_name','')} {sel.get('last_name','')} • {email_disp} • {(klass or '').title()}  \nID: `{mid}`"
            )

            # ------- FORM (keep submit button inside) -------
            with st.form(f"ex_edit_{mid}", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    fn = st.text_input("First name", value=sel.get("first_name", "") or "")
                    major = st.text_input("Major", value=sel.get("major", "") or "")
                    se = st.text_input("Student email", value=sel.get("student_email", "") or "")
                with c2:
                    ln = st.text_input("Last name", value=sel.get("last_name", "") or "")
                    cl = st.selectbox("Classification", CLASS_CHOICES, index=class_idx)

                    # --- Prefills (use prior saved answers) ---
                    prev_had      = sel.get("had_internship", None)  # None|True|False
                    prev_linkedin = bool(sel.get("linkedin_yes", False))
                    prev_resume   = bool(sel.get("updated_resume_yes", False))
                    

                    # yes/no with tri-state prefill for internship
                    had_internship = yes_no_required(
                        "Had an internship before?",
                        key=f"had_{mid}",
                        default=prev_had,
                    )

                    # --- Hoodie size (prefill from DB; default medium)
                    prev_size = normalize_hoodie_size(sel.get("hoodie_size") or "none")
                    size_labels = ["none", "Small", "Medium", "Large", "XL", "2XL"]
                    try:
                        size_idx = HOODIE_CHOICES.index(prev_size)
                    except ValueError:
                        size_idx = 1  # "medium"

                    hoodie_size_label = st.selectbox(
                    "Hoodie size",
                    size_labels,
                    index=size_idx,
                    key=f"hoodie_{mid}",
                    )

                    hoodie_size = normalize_hoodie_size(hoodie_size_label)


                    # simple checkboxes for the others (persist True/False)
                    linkedin_yes = st.checkbox(
                        "Has LinkedIn (on file)?",
                        value=prev_linkedin,
                        key=f"linkedin_{mid}",
                    )
                    updated_resume_yes = st.checkbox(
                        "Updated resume (on file)?",
                        value=prev_resume,
                        key=f"resume_{mid}",
                    )

                submit_existing = st.form_submit_button("Save & Check-In ✅")

            # ------- Submit handler (persist values + optional attendance) -------
            if submit_existing:
                # Save member fields (persists booleans)
                db_upsert_member({
                    "id": mid,
                    "first_name": fn.strip(),
                    "last_name": ln.strip(),
                    "classification": normalize_classification(cl),
                    "major": _norm(major),
                    "student_email": _norm(se),
                    "had_internship": had_internship,            # True/False/None
                    "linkedin_yes": linkedin_yes,                # True/False
                    "updated_resume_yes": updated_resume_yes,    # True/False
                    "hoodie_size": hoodie_size,
                    "created_at": None,
                    
                })
                # Check-in record
                _ = check_in(current_event_id, mid, method="manual")

                flash("success", "Saved & checked in.")
                request_checkin_reset()
                st.rerun()


            if submit_existing:
                if had_internship is None:
                    st.error("Please select Yes or No for the internship question.")
                else:
                    try:
                        db_upsert_member({
                            "id": mid,
                            "first_name": fn.strip(),
                            "last_name": ln.strip(),
                            "classification": normalize_classification(cl),
                            "major": _norm(major),
                            "student_email": _norm(se),
                            "had_internship": had_internship,
                            "linkedin_yes": linkedin_yes,                 
                            "updated_resume_yes": updated_resume_yes,
                            "hoodie_size": hoodie_size,
                            "created_at": None,
                        })
                        res = check_in(current_event_id, mid, method="verify")
                        if res.get("duplicate"):
                            flash("info", f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                        else:
                            flash("success", f"Success — {res['member_name']} signed in ✔")
                        request_checkin_reset()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Check-in failed: {type(e).__name__}: {e}")

    # Register new attendee (internship required; starts blank)
    st.divider()
    st.subheader("Register New Attendee (and Check-In)")

    with st.form("register_and_checkin", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            r_fn = st.text_input("First name*")
            r_major = st.text_input("Major")
            r_se = st.text_input("Student email*")
            size_labels = ["none", "Small", "Medium", "Large", "XL", "2XL"]
            r_hoodie_label = st.selectbox("Hoodie size", size_labels, index=1, key="reg_hoodie")

        with c2:
            r_ln = st.text_input("Last name*")
            r_cl = st.selectbox("Classification*", CLASS_CHOICES, index=0, key="reg_class")
            r_had = yes_no_required("Had an internship before?", key="had_register")
            r_linkedin_yes       = st.checkbox("Has LinkedIn (on file)?", value=False, key="reg_linkedin")
            r_updated_resume_yes = st.checkbox("Updated resume (on file)?", value=False, key="reg_resume")
            

        submit_new = st.form_submit_button("Create Member & Check-In ✅")

    if submit_new:
      # Safe defaults even if the checkboxes' widgets weren't rendered for some reason
        r_hoodie_size = normalize_hoodie_size(st.session_state.get("reg_hoodie", r_hoodie_label if "r_hoodie_label" in locals() else "Medium"))
        r_linkedin_yes       = bool(st.session_state.get("reg_linkedin", r_linkedin_yes if "r_linkedin_yes" in locals() else False))
        r_updated_resume_yes = bool(st.session_state.get("reg_resume", r_updated_resume_yes if "r_updated_resume_yes" in locals() else False))
       


        missing = []
        if not (r_fn or "").strip(): missing.append("first name")
        if not (r_ln or "").strip(): missing.append("last name")
        if not (r_se or "").strip(): missing.append("email")
        if not (r_cl or "").strip(): missing.append("classification")
        if r_had is None:             missing.append("had an internship (Yes/No)")
        if missing:
            st.error("Please fill the required fields: " + ", ".join(missing))
        else:
            try:
                member_id = f"m_{uuid4().hex}"
                db_upsert_member(
                    {
                        "id": member_id,
                        "first_name": r_fn.strip(),
                        "last_name": r_ln.strip(),
                        "classification": normalize_classification(r_cl),
                        "major": _norm(r_major),
                        "student_email": _norm(r_se),
                         "had_internship": r_had,
                        "linkedin_yes": r_linkedin_yes,                 
                        "updated_resume_yes": r_updated_resume_yes,
                        "hoodie_size": r_hoodie_size,
                        "created_at": None,
                    }
                )
                res = check_in(current_event_id, member_id, method="register")
                if res.get("duplicate"):
                    flash("info", f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                else:
                    flash("success", f"🎉 Congrats — successfully registered {res['member_name']} and checked in.")
                request_checkin_reset()
                st.rerun()
            except Exception as e:
                st.error(f"Check-in failed: {type(e).__name__}: {e}")

    if st.session_state.pop("_post_action", False):
        st.stop()

   

# ---------------------------------
# ADMIN (PASSWORD-PROTECTED)
# ---------------------------------
else:
    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:
        with st.form("admin_login"):
            pw = st.text_input("Admin Password", type="password")
            submit = st.form_submit_button("Sign in")
        if submit:
            try:
                sec = st.secrets.get("security") or {}
                admin_pw = sec.get("admin_password") or st.secrets.get("admin_password")
                if pw == admin_pw and admin_pw:
                    st.session_state.admin_ok = True
                    st.session_state["_just_refreshed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            except Exception:
                st.error("Admin password is not set. Define admin_password in secrets.")
        st.stop()

    # DB caption
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        st.caption(f"DB: {u.host}/{u.database}")
    except Exception as e:
        st.caption(f"DB not ready: {e}")

    with st.sidebar:
        mode = st.radio(
            "Admin Mode",
            [
                "Executive Reports",
                "Add Member",
                "Create Event",
                "Data Browser (DB)",
                "Delete Row (DB)",
                "Tables (DB)",
                "Points Leaderboard"
            ],
        )
        st.checkbox("Show DB counts", value=False, key="adm_counts")

   

    # ---------- EXECUTIVE REPORTS ----------
    if mode == "Executive Reports":
        st.subheader("Executive reports — events & attendance")
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(
            f"Figures are built automatically from live check-ins (filtered by **event date**). "
            f"Generated {generated} local time. Use sidebar **Refresh** to pull the latest data."
        )

        c1, c2 = st.columns(2)
        with c1:
            exec_start = st.date_input("From (event date)", value=None, key="exec_start")
        with c2:
            exec_end = st.date_input("To (event date)", value=None, key="exec_end")

        if exec_start and exec_end and exec_start > exec_end:
            st.error("Start date must be on or before end date.")
        else:
            start_s = exec_start.isoformat() if exec_start else None
            end_s = exec_end.isoformat() if exec_end else None
            try:
                kpi = exec_kpis(start_s, end_s)
                ret_pct = exec_returning_member_pct(start_s, end_s)
                ev_df = exec_event_rollups(start_s, end_s)
                mo_df = exec_monthly_trend(start_s, end_s)
                cl_df = exec_classification_mix(start_s, end_s)
                meth_df = exec_method_mix(start_s, end_s)
                maj_df = exec_top_majors(start_s, end_s, limit=12)
            except Exception as e:
                st.error(f"Could not load analytics: {type(e).__name__}: {e}")
            else:
                total_ci = int(kpi.get("total_check_ins", 0) or 0)
                n_ev = int(kpi.get("events_with_checkins", 0) or 0)
                uniq = int(kpi.get("unique_members", 0) or 0)
                avg = (total_ci / n_ev) if n_ev else 0.0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Events w/ check-ins", f"{n_ev:,}")
                m2.metric("Total check-ins", f"{total_ci:,}")
                m3.metric("Unique members", f"{uniq:,}")
                m4.metric("Avg check-ins / event", f"{avg:.1f}")
                m5.metric("Members at 2+ events", f"{ret_pct:.1f}%")

                st.divider()
                left, right = st.columns(2)

                with left:
                    if ev_df is not None and not ev_df.empty:
                        top = ev_df.nlargest(15, "check_ins").sort_values("check_ins", ascending=True).copy()
                        top["label"] = (
                            top["name"].astype(str).str.slice(0, 40)
                            + " ("
                            + top["event_date"].astype(str)
                            + ")"
                        )
                        fig_ev = px.bar(
                            top,
                            x="check_ins",
                            y="label",
                            orientation="h",
                            labels={"check_ins": "Check-ins", "label": "Event"},
                            text="check_ins",
                        )
                        fig_ev.update_traces(textposition="outside", marker_color="#D4AF37")
                        fig_ev.update_layout(yaxis=dict(title="", categoryorder="total ascending"))
                        st.plotly_chart(
                            _exec_chart_layout(fig_ev, "Top events by check-in volume"),
                            use_container_width=True,
                        )
                    else:
                        st.info("No check-ins in this date range — bar chart skipped.")

                with right:
                    if mo_df is not None and not mo_df.empty:
                        mo_plot = mo_df.copy()
                        mo_plot["month_label"] = pd.to_datetime(mo_plot["month_start"]).dt.strftime("%b %Y")
                        fig_mo = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_mo.add_trace(
                            go.Scatter(
                                x=mo_plot["month_label"],
                                y=mo_plot["check_ins"],
                                mode="lines+markers",
                                name="Check-ins",
                                line=dict(color="#D4AF37", width=3),
                                marker=dict(size=10, color="#1a1a1a"),
                            ),
                            secondary_y=False,
                        )
                        fig_mo.add_trace(
                            go.Bar(
                                x=mo_plot["month_label"],
                                y=mo_plot["unique_members"],
                                name="Unique members",
                                marker_color="rgba(26,26,26,0.35)",
                            ),
                            secondary_y=True,
                        )
                        fig_mo.update_yaxes(title_text="Check-ins", secondary_y=False)
                        fig_mo.update_yaxes(title_text="Unique members", secondary_y=True)
                        st.plotly_chart(
                            _exec_chart_layout(fig_mo, "Momentum — monthly check-ins & unique members"),
                            use_container_width=True,
                        )
                    else:
                        st.info("No monthly trend for this range — chart skipped.")

                st.divider()
                c_a, c_b, c_c = st.columns(3)

                with c_a:
                    if cl_df is not None and not cl_df.empty:
                        fig_cl = px.pie(
                            cl_df,
                            names="classification",
                            values="check_ins",
                            hole=0.45,
                            color_discrete_sequence=[
                                "#D4AF37", "#e8c547", "#b8962e", "#8a7028",
                                "#c9a227", "#5c4d1f", "#a89030", "#6b5a24",
                            ],
                        )
                        fig_cl.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(
                            _exec_chart_layout(fig_cl, "Audience — classification mix"),
                            use_container_width=True,
                        )
                    else:
                        st.caption("No classification breakdown.")

                with c_b:
                    if meth_df is not None and not meth_df.empty:
                        fig_me = px.bar(
                            meth_df,
                            x="method",
                            y="check_ins",
                            labels={"method": "Check-in method", "check_ins": "Check-ins"},
                            text="check_ins",
                        )
                        fig_me.update_traces(marker_color="#1a1a1a", textposition="outside")
                        st.plotly_chart(
                            _exec_chart_layout(fig_me, "How people checked in"),
                            use_container_width=True,
                        )
                    else:
                        st.caption("No method breakdown.")

                with c_c:
                    if maj_df is not None and not maj_df.empty:
                        mj = maj_df.sort_values("unique_members", ascending=True).copy()
                        fig_mj = px.bar(
                            mj,
                            x="unique_members",
                            y="major",
                            orientation="h",
                            labels={"unique_members": "Unique members", "major": "Major"},
                            text="unique_members",
                        )
                        fig_mj.update_traces(textposition="outside", marker_color="#D4AF37")
                        fig_mj.update_layout(yaxis=dict(title="", categoryorder="total ascending"))
                        st.plotly_chart(
                            _exec_chart_layout(fig_mj, "Top majors (unique members)"),
                            use_container_width=True,
                        )
                    else:
                        st.caption("No major breakdown.")

                st.divider()
                st.markdown("**Underlying tables (for slides or follow-up)**")
                d1, d2 = st.columns(2)
                with d1:
                    if ev_df.empty:
                        st.caption("No event-level rows for this filter.")
                    else:
                        st.dataframe(
                            ev_df.sort_values(["event_date", "name"], ascending=[False, True]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.download_button(
                        "Download event summary (CSV)",
                        ev_df.to_csv(index=False).encode("utf-8"),
                        file_name="executive_event_summary.csv",
                        mime="text/csv",
                    )
                with d2:
                    detail = mo_df.copy()
                    if not detail.empty:
                        detail["month"] = pd.to_datetime(detail["month_start"]).dt.strftime("%Y-%m")
                    if detail.empty:
                        st.caption("No monthly rows for this filter.")
                    else:
                        st.dataframe(detail, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download monthly trend (CSV)",
                        mo_df.to_csv(index=False).encode("utf-8"),
                        file_name="executive_monthly_trend.csv",
                        mime="text/csv",
                    )

                summary_one = pd.DataFrame(
                    [
                        {
                            "report_generated": generated,
                            "filter_event_date_from": start_s or "",
                            "filter_event_date_to": end_s or "",
                            "events_with_checkins": n_ev,
                            "total_check_ins": total_ci,
                            "unique_members": uniq,
                            "avg_checkins_per_event": round(avg, 2),
                            "pct_members_at_two_plus_events": round(ret_pct, 2),
                        }
                    ]
                )
                st.download_button(
                    "Download executive KPI summary (CSV)",
                    summary_one.to_csv(index=False).encode("utf-8"),
                    file_name="executive_kpi_summary.csv",
                    mime="text/csv",
                )

    # ---------- ADD MEMBER ----------
    elif mode == "Add Member":
        st.subheader("Add a Member (writes to DB)")
        with st.form("add_member", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                fn = st.text_input("First name")
                major = st.text_input("Major")
                se = st.text_input("Student email")
            with c2:
                ln = st.text_input("Last name")
                cl = st.selectbox("Classification", CLASS_CHOICES, index=0)
                had = yes_no_required("Had an internship before?", key="had_admin_add")
            submit = st.form_submit_button("Save")

        if submit:
            missing = []
            if not (fn or "").strip(): missing.append("first name")
            if not (ln or "").strip(): missing.append("last name")
            if had is None:            missing.append("had an internship (Yes/No)")
            if missing:
                st.error("Please fill: " + ", ".join(missing))
            else:
                member_id = f"m_{uuid4().hex}"
                try:
                    db_upsert_member(
                        {
                            "id": member_id,
                            "first_name": fn.strip(),
                            "last_name": ln.strip(),
                            "classification": normalize_classification(cl),
                            "major": _norm(major),
                            "student_email": _norm(se),
                            "had_internship": had,
                            "created_at": None,
                        }
                    )
                    st.success(f"Saved {fn} {ln} (id {member_id}).")
                    try: st.toast(f"Saved {fn} {ln}")
                    except Exception: pass
                    clear_cache()
                except Exception as e:
                    st.error(f"Save failed: {e}")

    # ---------- CREATE EVENT ----------
    elif mode == "Create Event":
        st.subheader("Create a new event (writes to DB)")
        with st.form("new_event"):
            name = st.text_input("Event name")
            dt = st.text_input("Event date (YYYY-MM-DD)", value=str(date.today()))
            loc = st.text_input("Location")
            submit = st.form_submit_button("Create")
        if submit:
            if not name.strip():
                st.error("Event name required.")
            else:
                event_id = f"{_slug(name)}_{dt.strip() or str(date.today())}"
                try:
                    db_create_event(
                        {
                            "id": event_id,
                            "name": name.strip(),
                            "event_date": dt.strip() or str(date.today()),
                            "location": loc.strip(),
                        }
                    )
                    st.success(f"Created event: {name} ({dt})  → id={event_id}")
                    try: st.toast(f"Created event {name}")
                    except Exception: pass
                    clear_cache()
                except Exception as e:
                    st.error(f"Create failed: {e}")

     # ---------- DATA BROWSER ----------
    elif mode == "Data Browser (DB)":
        st.subheader("Data Browser (live from Postgres)")
        if st.button("Refresh"):
            clear_cache()
        try:
            df = load_databrowser(2000)
        except Exception as e:
            st.error(f"Failed to load data browser: {e}")
            df = pd.DataFrame()

        if df.empty:
            st.info("No check-ins yet.")
        else:
            majors = _unique_nonempty_sorted(df.get("major"))

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: ev_name = st.text_input("Filter by event name contains")
            with c2: klass = st.multiselect("Filter by classification", CLASS_CHOICES, default=[])
            with c3: start_date = st.date_input("Start date", value=None)
            with c4: end_date = st.date_input("End date", value=None)
            with c5: selected_majors = st.multiselect("Filter by major", options=majors, default=[])

            work = df.copy()
            if ev_name:
                work = work[work["event_name"].astype(str).str.contains(ev_name, case=False, na=False)]
            if klass:
                work = work[work["classification"].isin(klass)]
            if selected_majors:
                work = work[work["major"].astype("string").fillna("").str.strip().isin(set(selected_majors))]
            if start_date:
                work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date >= start_date]
            if end_date:
                work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date <= end_date]

            q2 = st.text_input("Search name/email", placeholder="Search attendance…").strip().lower()
            if q2:
                fields = []
                for col in ["member_name","first_name","last_name","student_email","event_name"]:
                    if col in work.columns:
                        fields.append(work[col].astype(str).str.lower().str.contains(q2, na=False))
                if fields:
                    mask = fields[0]
                    for f in fields[1:]:
                        mask |= f
                    work = work[mask]

            st.caption(f"Showing {len(work)} of {len(df)} rows")
            show_cols = [
                "event_name","event_date","event_location",
                "member_name","classification","major",
                "had_internship","linkedin_yes","updated_resume_yes","hoodie_size",
                "checked_in_at","method",
            ]
            show_cols = [c for c in show_cols if c in work.columns]
            st.dataframe(
                work.sort_values("checked_in_at", ascending=False)[show_cols],
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "📥 Download filtered view (CSV)",
                work.to_csv(index=False).encode("utf-8"),
                file_name="databrowser_filtered.csv",
                mime="text/csv",
            )

    
    # ---------- DELETE ROW (DB) ----------
    elif mode == "Delete Row (DB)":
        st.sidebar.warning("⚠️ Deletions are permanent. Double-check before confirming.")
        st.subheader("Delete a single row from Postgres")

        TABLES = {
            "members": {
                "key_cols": ["id"],
                "preview_cols": [
                    "id","first_name","last_name","classification","major",
                    "student_email","had_internship","linkedin_yes","updated_resume_yes","hoodie_size","created_at","updated_at"
                ],
                "query": """
                    SELECT id, first_name, last_name, classification, major,
                           student_email, had_internship, inkedin_yes, updated_resume_yes, hoodie_size,
                           created_at, updated_at
                    FROM members
                    WHERE
                      (:q = '' OR
                       COALESCE(first_name,'')    ILIKE :pat OR
                       COALESCE(last_name,'')     ILIKE :pat OR
                       COALESCE(student_email,'') ILIKE :pat OR
                       COALESCE(id,'')            ILIKE :pat)
                    ORDER BY COALESCE(updated_at, created_at) DESC NULLS LAST
                    LIMIT :limit
                """,
                "delete_sql": "DELETE FROM members WHERE id = :id",
            },
            "events": {
                "key_cols": ["id"],
                "preview_cols": ["id","name","event_date","location","created_at"],
                "query": """
                    SELECT id, name, event_date, location, created_at
                    FROM events
                    WHERE
                      (:q = '' OR
                       COALESCE(name,'') ILIKE :pat OR
                       COALESCE(location,'') ILIKE :pat OR
                       COALESCE(id,'') ILIKE :pat)
                    ORDER BY event_date DESC, name
                    LIMIT :limit
                """,
                "delete_sql": "DELETE FROM events WHERE id = :id",
            },
            "attendance": {
                "key_cols": ["event_id","member_id","checked_in_at"],
                "preview_cols": ["event_id","member_id","checked_in_at","method"],
                "query": """
                    SELECT event_id, member_id, checked_in_at, method
                    FROM attendance
                    WHERE
                      (:q = '' OR
                       COALESCE(event_id,'') ILIKE :pat OR
                       COALESCE(member_id,'') ILIKE :pat)
                    ORDER BY checked_in_at DESC
                    LIMIT :limit
                """,
                "delete_sql": """
                    DELETE FROM attendance
                    WHERE event_id = :event_id
                      AND member_id = :member_id
                      AND checked_in_at = :checked_in_at
                """,
            },
        }

        tab = st.selectbox("Choose table", list(TABLES.keys()), index=0)
        cfg = TABLES[tab]
        q = st.text_input("Search (optional)", placeholder="Name, email, id…").strip()
        limit = st.number_input("Max results", 1, 2000, 200)

        try:
            with ENGINE.begin() as c:
                rows = c.execute(
                    text(cfg["query"]),
                    {"q": q, "pat": f"%{q}%", "limit": int(limit)},
                ).mappings().all()
            df = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Search failed: {e}")
            df = pd.DataFrame()

        if df.empty:
            st.info("No rows found for current search.")
        else:
            def _label(row: pd.Series) -> str:
                if tab == "members":
                    return f"{row.get('id','')} — {row.get('first_name','')} {row.get('last_name','')} ({row.get('major','')})"
                if tab == "events":
                    return f"{row.get('id','')} — {row.get('name','')} [{row.get('event_date','')}] @ {row.get('location','')}"
                if tab == "attendance":
                    return f"{row.get('event_id','')} / {row.get('member_id','')} @ {row.get('checked_in_at','')}"
                return "row"

            df = df.fillna("")
            opts: List[str] = []
            keymap: Dict[str, Tuple[str, ...]] = {}
            for _, r in df.iterrows():
                lbl = _label(r)
                key = tuple(str(r[k]) for k in cfg["key_cols"])
                keymap[lbl] = key
                opts.append(lbl)

            pick = st.selectbox("Select row to delete", opts)

            if pick:
                key_vals = keymap[pick]
                mask = pd.Series([True] * len(df))
                for k, v in zip(cfg["key_cols"], key_vals):
                    mask &= (df[k].astype(str) == str(v))
                preview = df.loc[mask, cfg["preview_cols"]] if set(cfg["preview_cols"]).issubset(df.columns) else df.loc[mask]
                st.subheader("Row preview")
                st.dataframe(preview, use_container_width=True, hide_index=True)

                st.warning("⚠️ This action is permanent and cannot be undone.")
                # Single-click delete (no extra confirmation)
                if st.button("Delete row", type="primary"):
                    try:
                        with ENGINE.begin() as c:
                            if tab in ("members", "events"):
                                c.execute(text(cfg["delete_sql"]), {"id": key_vals[0]})
                            else:
                                c.execute(
                                    text(cfg["delete_sql"]),
                                    {
                                        "event_id": key_vals[0],
                                        "member_id": key_vals[1],
                                        "checked_in_at": key_vals[2],
                                    },
                                )
                        st.success(f"Deleted from {tab}: {pick}")
                        try: st.toast(f"Deleted from {tab}")
                        except Exception: pass
                        clear_cache()
                        st.session_state["_just_refreshed"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    elif mode == "Points Leaderboard":
        POINTS_RULES = [
        ("e.name ILIKE '%GBM%'",                 15, "GBM"),
        ("e.name ILIKE '%LinkedIn%'",            25, "LinkedIn Workshop"),
        ("e.name ILIKE '%Internship Workshop%'", 25, "Internship Workshop"),
    ]

        st.subheader("🏆 DMC Points Leaderboard")
        pw = st.text_input("Enter Points Board password", type="password", key="pts_pwd")

        # Try to read password from secrets (both locations)
        sec = st.secrets.get("security") or {}
        board_pw = sec.get("POINTS_BOARD_PASSWORD") or st.secrets.get("POINTS_BOARD_PASSWORD")

        if not pw:
            st.info("Enter the points-board password to view the leaderboard.")
        elif pw.strip() != str(board_pw).strip():
            st.error("Incorrect password.")

        else:
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start date (optional)", value=None, key="pts_start")
            end_date   = c2.date_input("End date (optional)",   value=None, key="pts_end")

            case_sql = " ".join([f"WHEN {pred} THEN {pts}" for (pred, pts, _) in POINTS_RULES])

            params, where = {}, []
            if start_date:
                where.append("e.event_date >= :start_date"); params["start_date"] = str(start_date)
            if end_date:
                where.append("e.event_date <= :end_date");   params["end_date"]   = str(end_date)
            where_clause = ("WHERE " + " AND ".join(where)) if where else ""

            q = text(f"""
                WITH pts AS (
                    SELECT
                        a.member_id,
                        CASE
                            {case_sql}
                            ELSE 0
                        END AS pts
                    FROM attendance a
                    JOIN events e ON e.id = a.event_id
                    {where_clause}
                )
                SELECT
                    m.id AS member_id,
                    m.first_name,
                    m.last_name,
                    COALESCE(SUM(p.pts), 0) AS total_points
                FROM members m
                LEFT JOIN pts p ON p.member_id = m.id
                GROUP BY m.id, m.first_name, m.last_name
                HAVING COALESCE(SUM(p.pts), 0) > 0
                ORDER BY total_points DESC, m.last_name ASC, m.first_name ASC
                LIMIT 200
            """)

            try:
                with ENGINE.begin() as c:
                    rows = c.execute(q, params).mappings().all()
                if not rows:
                    st.warning("No points yet. (No matching attendance for the selected window.)")
                else:
                    df = pd.DataFrame(rows)
                    df["Member"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.strip()
                    df = df[["Member", "total_points"]].rename(columns={"total_points": "Points"})
                    st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to load leaderboard: {e}")

            with st.expander("Scoring rules (current)"):
                st.markdown("\n".join([f"- **{lbl}**: {pts} pts" for (_, pts, lbl) in POINTS_RULES]))

    elif mode == "Tables (DB)":
        st.subheader("📋 Members & Events (from Postgres)")
        tabs = st.tabs(["Members", "Events & Attendees", "All Check-Ins (joined)"])

        with tabs[0]:
            st.caption("Raw members table with quick filters. Export as CSV below.")
            if st.button("Refresh members"):
                clear_cache()
            try:
                mdf = load_members_table()
            except Exception as e:
                st.error(f"Failed to load members: {e}")
                mdf = pd.DataFrame()

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                klass = st.multiselect("Classification", CLASS_CHOICES, default=[])
            with c2:
                major_q = st.text_input("Major contains", "")
            with c3:
                name_q = st.text_input("Name/Email contains", "")

            work = mdf.copy()
            if not work.empty:
                if klass:
                    work = work[work["classification"].isin(klass)]
                if major_q:
                    work = work[
                        work["major"].astype("string").fillna("").str.contains(major_q, case=False, na=False)
                    ]
                if name_q:
                    pat = name_q.strip().lower()
                    cols = ["first_name", "last_name", "student_email"]
                    mask = None
                    for col in cols:
                        if col in work.columns:
                            colmask = work[col].astype(str).str.lower().str.contains(pat, na=False)
                            mask = colmask if mask is None else (mask | colmask)
                    if mask is not None:
                        work = work[mask]

            st.caption(f"{len(work)} of {len(mdf)} members")
            st.dataframe(work, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download members (CSV)",
                work.to_csv(index=False).encode("utf-8"),
                file_name="members_export.csv",
                mime="text/csv",
            )

        with tabs[1]:
            st.caption("Pick an event to see everyone who attended with their member info.")
            if st.button("Refresh events/attendees"):
                clear_cache()

            try:
                ev = load_events_index()
            except Exception as e:
                st.error(f"Failed to load events index: {e}")
                ev = pd.DataFrame()

            if ev.empty:
                st.info("No events yet.")
            else:
                left, right = st.columns([1, 2])
                with left:
                    st.write("**Events**")
                    st.dataframe(ev, use_container_width=True, hide_index=True)
                    ev = ev.copy()
                    ev["label"] = ev["id"] + " — " + ev["name"] + " (" + ev["event_date"].astype(str) + ")"
                    pick = st.selectbox("Select event", ev["label"].tolist())
                    event_id = ev.loc[ev["label"] == pick, "id"].iloc[0]

                with right:
                    try:
                        adf = load_event_attendees(event_id)
                    except Exception as e:
                        st.error(f"Failed to load attendees: {e}")
                        adf = pd.DataFrame()
                    st.write(f"**Attendees for:** {pick}")
                    if adf.empty:
                        st.info("No check-ins for this event yet.")
                    else:
                        show_cols = [
                            "checked_in_at","method","member_id",
                            "first_name","last_name","classification","major",
                            "student_email","had_internship","linkedin_yes","updated_resume_yes","hoodie_size",
                        ]
                        show_cols = [c for c in show_cols if c in adf.columns]
                        st.dataframe(adf[show_cols], use_container_width=True, hide_index=True)
                        st.download_button(
                            "📥 Download attendees (CSV)",
                            adf.to_csv(index=False).encode("utf-8"),
                            file_name=f"{event_id}_attendees.csv",
                            mime="text/csv",
                        )

        with tabs[2]:
            st.caption("Full joined view across events + attendance + members.")
            if st.button("Refresh joined view"):
                clear_cache()
            try:
                df = load_databrowser(5000)
            except Exception as e:
                st.error(f"Failed to load joined view: {e}")
                df = pd.DataFrame()
            if df.empty:
                st.info("No check-ins yet.")
            else:
                show_cols = [
                    "event_name","event_date","event_location",
                    "first_name","last_name","classification","major",
                    "student_email","had_internship", "linkedin_yes","updated_resume_yes","hoodie_size",
                    "checked_in_at","method",
                ]
                show_cols = [c for c in show_cols if c in df.columns]
                st.dataframe(
                    df[show_cols].sort_values("checked_in_at", ascending=False),
                    use_container_width=True, hide_index=True,
                )
                st.download_button(
                    "📥 Download all check-ins (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="all_checkins_joined.csv",
                    mime="text/csv",
                )


    