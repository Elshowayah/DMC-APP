# checkin.py — ✅ Event Check-In (DB-backed, member-facing)
# Python 3.9+

from __future__ import annotations

from uuid import uuid4
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine.url import make_url

# ---- Single source of truth: same ENGINE & helpers as admin.py
from db import (
    ENGINE,
    upsert_member as db_upsert_member,   # same helper Admin uses
)

# =========================
# Page / constants
# =========================
st.set_page_config(page_title="✅ Event Check-In", page_icon="✅", layout="wide")
st.title("✅ Event Check-In (DB-backed)")

CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# =========================
# Diagnostics (confirm both apps hit the SAME DB)
# =========================
try:
    u = make_url(ENGINE.url)  # type: ignore[arg-type]
    st.caption(f"DB target → host={u.host} db={u.database} user={u.username}")
    # Lightweight connectivity probe
    with ENGINE.connect() as c:
        c.execute(text("SELECT 1")).scalar_one()
except Exception as e:
    st.error(f"Database connectivity failed: {type(e).__name__}: {e}")
    st.stop()

# =========================
# Helpers
# =========================
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 or None

def normalize_classification(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    mapping = {
        "freshmen": "freshman",
        "sophmore": "sophomore",
        "jr": "junior",
        "sr": "senior",
        "alum": "alumni",
    }
    return v if v in CLASS_CHOICES else mapping.get(v, "freshman")

def split_name(full_name: str) -> Tuple[str, str]:
    full_name = (full_name or "").strip()
    if not full_name:
        return "", ""
    parts = full_name.split()
    return (parts[0], "" ) if len(parts) == 1 else (" ".join(parts[:-1]), parts[-1])

# =========================
# DB queries
# =========================
@st.cache_data(ttl=5, show_spinner=False)
def list_events(limit: int = 300) -> pd.DataFrame:
    sql = """
      SELECT id, name, event_date, location
      FROM events
      ORDER BY event_date DESC, name
      LIMIT :limit
    """
    try:
        with ENGINE.begin() as c:
            rows = c.execute(text(sql), {"limit": limit}).mappings().all()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Could not load events: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["id","name","event_date","location"])

@st.cache_data(ttl=10, show_spinner=False)
def find_member(q: str, limit: int = 100) -> pd.DataFrame:
    """
    Case-insensitive search over first/last name and emails in DB.
    Uses COALESCE to avoid NULL issues and ILIKE for case-insensitive match.
    """
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=[
            "id","first_name","last_name","classification","major","v_number","student_email","personal_email"
        ])
    pat = f"%{q}%"
    sql = """
      SELECT id, first_name, last_name, classification, major, v_number, student_email, personal_email
      FROM members
      WHERE
        COALESCE(first_name,'')     ILIKE :pat OR
        COALESCE(last_name,'')      ILIKE :pat OR
        COALESCE(student_email,'')  ILIKE :pat OR
        COALESCE(personal_email,'') ILIKE :pat
      ORDER BY last_name NULLS LAST, first_name NULLS LAST
      LIMIT :limit
    """
    try:
        with ENGINE.begin() as c:
            rows = c.execute(text(sql), {"pat": pat, "limit": limit}).mappings().all()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Member search failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["id","first_name","last_name","classification","major","v_number","student_email","personal_email"])

def check_in(event_id: str, member_id: str, method: str = "manual") -> Dict:
    """
    Insert into attendance if not already present; return joined info.
    Duplicate == same (event_id, member_id) already recorded.
    """
    with ENGINE.begin() as c:
        ev = c.execute(
            text("SELECT id, name, event_date, location FROM events WHERE id = :id"),
            {"id": event_id},
        ).mappings().first()
        if not ev:
            raise ValueError("Event not found.")

        mem = c.execute(
            text("""
                SELECT id, first_name, last_name, classification, major,
                       student_email, personal_email, v_number
                FROM members WHERE id = :id
            """),
            {"id": member_id},
        ).mappings().first()
        if not mem:
            raise ValueError("Member not found.")

        # Check for prior check-in
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

        if dup:
            return {
                "event_id": dup["event_id"],
                "member_id": dup["member_id"],
                "checked_in_at": dup["checked_in_at"],
                "method": dup["method"],
                "event_name": ev["name"],
                "event_date": ev["event_date"],
                "event_location": ev["location"],
                "member_name": f"{mem['first_name']} {mem['last_name']}".strip(),
                "member_classification": mem["classification"],
                "member_student_email": mem["student_email"],
                "member_personal_email": mem["personal_email"],
                "duplicate": True,
            }

        # Fresh insert
        ins = c.execute(
            text("""
                INSERT INTO attendance (event_id, member_id, checked_in_at, method)
                VALUES (:e, :m, NOW(), :method)
                RETURNING event_id, member_id, checked_in_at, method
            """),
            {"e": event_id, "m": member_id, "method": method},
        ).mappings().first()

        return {
            "event_id": ins["event_id"],
            "member_id": ins["member_id"],
            "checked_in_at": ins["checked_in_at"],
            "method": ins["method"],
            "event_name": ev["name"],
            "event_date": ev["event_date"],
            "event_location": ev["location"],
            "member_name": f"{mem['first_name']} {mem['last_name']}".strip(),
            "member_classification": mem["classification"],
            "member_student_email": mem["student_email"],
            "member_personal_email": mem["personal_email"],
            "duplicate": False,
        }

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.caption("Member-facing check-in (DB-backed)")
    if st.button("Refresh"):
        st.cache_data.clear()

# =========================
# Event selection
# =========================
ev_df = list_events()
if ev_df.empty:
    st.warning("No events yet. Ask an admin to create one in the Admin Console.")
    st.stop()

# Make a robust, human-readable label (unique even if names repeat)
ev_df["label"] = ev_df.apply(
    lambda r: f"{r['id']} — {r['name']} ({str(r['event_date'])}) @ {r['location'] or ''}".strip(),
    axis=1
)
choice = st.selectbox("Select Event", ev_df["label"].tolist())
current_event_id = ev_df.loc[ev_df["label"] == choice, "id"].iloc[0]

# =========================
# Existing Member — search/edit/check-in (DB)
# =========================
st.divider()
st.subheader("Existing Member — Search, Edit, and Check-In")

if "existing_hits" not in st.session_state:
    st.session_state.existing_hits = pd.DataFrame()

with st.form("existing_search_form", clear_on_submit=False):
    q = st.text_input("Search by email or name", placeholder="Type name or email…").strip()
    do_search = st.form_submit_button("Find Member 🔎")

if do_search:
    st.session_state.existing_hits = find_member(q)

hits = st.session_state.existing_hits
if do_search and (hits is None or hits.empty):
    st.info("No members matched your search. Try a different name or email.")

if hits is not None and not hits.empty:
    for _, h in hits.iterrows():
        mid = str(h.get("id", "")).strip()
        email_disp = h.get("student_email") or h.get("personal_email") or "no email"
        klass = (h.get("classification") or "freshman").lower()
        try:
            class_idx = CLASS_CHOICES.index(klass)
        except ValueError:
            class_idx = 0

        st.markdown(
            f"**{h.get('first_name','')} {h.get('last_name','')}** • {email_disp} • {(klass or '').title()}  \n"
            f"ID: `{mid}`"
        )

        with st.form(f"ex_edit_{mid}", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                fn   = st.text_input("First name", value=h.get("first_name","") or "")
                major = st.text_input("Major", value=h.get("major","") or "")
                vnum = st.text_input("V-number", value=h.get("v_number","") or "")
                se   = st.text_input("Student email", value=h.get("student_email","") or "")
            with c2:
                ln = st.text_input("Last name", value=h.get("last_name","") or "")
                cl = st.selectbox("Classification", CLASS_CHOICES, index=class_idx)
                pe = st.text_input("Personal email", value=h.get("personal_email","") or "")
            submit_existing = st.form_submit_button("Save & Check-In ✅")

        if submit_existing:
            try:
                # Update in DB (same helper as Admin)
                payload = {
                    "id": mid,
                    "first_name": fn.strip(),
                    "last_name": ln.strip(),
                    "classification": normalize_classification(cl),
                    "major": _norm(major),
                    "v_number": _norm(vnum),
                    "student_email": _norm(se),
                    "personal_email": _norm(pe),
                    "created_at": None,  # DB default
                }
                db_upsert_member(payload)

                res = check_in(current_event_id, mid, method="verify")
                st.session_state.existing_hits = pd.DataFrame()
                if res.get("duplicate"):
                    st.info(
                        f"{res['member_name']} was already checked in for "
                        f"{res.get('event_name','this event')} at {res['checked_in_at']}."
                    )
                else:
                    st.success(f"✅ Checked in {res['member_name']}!")
                st.rerun()
            except Exception as e:
                st.error(f"Check-in failed: {type(e).__name__}: {e}")

# =========================
# Register New Attendee — create in DB, then check-in
# =========================
st.divider()
st.subheader("Register New Attendee (and Check-In)")

with st.form("register_and_checkin", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        r_fn = st.text_input("First name", value="")
        r_major = st.text_input("Major", value="")
        r_v  = st.text_input("V-number", value="")
        r_se = st.text_input("Student email", value="")
    with c2:
        r_ln = st.text_input("Last name", value="")
        r_cl = st.selectbox("Classification", CLASS_CHOICES, index=0, key="reg_class")
        r_pe = st.text_input("Personal email", value="")
    submit_new = st.form_submit_button("Create Member & Check-In ✅")

if submit_new:
    try:
        # Prefer provided V-number; else generate a stable id
        member_id = (r_v or "").strip() or f"m_{uuid4().hex}"

        db_upsert_member({
            "id": member_id,
            "first_name": r_fn.strip(),
            "last_name": r_ln.strip(),
            "classification": normalize_classification(r_cl),
            "major": _norm(r_major),
            "v_number": _norm(r_v),
            "student_email": _norm(r_se),
            "personal_email": _norm(r_pe),
            "created_at": None,  # DB default
        })

        res = check_in(current_event_id, member_id, method="register")
        if res.get("duplicate"):
            st.info(
                f"{res['member_name']} was already checked in for "
                f"{res.get('event_name','this event')} at {res['checked_in_at']}."
            )
        else:
            st.success(f"Created & checked in {res['member_name']} to {res.get('event_name','event')}!")
        st.session_state.existing_hits = pd.DataFrame()
        st.rerun()
    except Exception as e:
        st.error(f"Check-in failed: {type(e).__name__}: {e}")









