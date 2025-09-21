# checkin.py — ✅ Event Check-In (DB-backed, member-facing)
# Python 3.9+

from __future__ import annotations
import os
from uuid import uuid4
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine.url import make_url

# ---- DB helpers
from db import (
    ENGINE,
    upsert_member as db_upsert_member,   # reuse the same helper admin.py uses
)

# =========================
# Page / constants
# =========================
st.set_page_config(page_title="✅ Event Check-In", page_icon="✅", layout="wide")
st.title("✅ Event Check-In (DB-backed)")

CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# Debug banner to ensure this app points at the SAME DB as admin.py
try:
    u = make_url(ENGINE.url)  # type: ignore[arg-type]
    st.caption(f"DB: {u.host}/{u.database}")
except Exception as e:
    st.caption(f"DB not ready: {e}")

# =========================
# Normalization helpers
# =========================
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 or None

def normalize_classification(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    mapping = {"freshmen": "freshman", "sophmore": "sophomore", "jr": "junior", "sr": "senior"}
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
      ORDER BY event_date DESC
      LIMIT :limit
    """
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=10, show_spinner=False)
def find_member(q: str, limit: int = 100) -> pd.DataFrame:
    """
    Case-insensitive search over first/last name and emails.
    """
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=[
            "id","first_name","last_name","classification","major","v_number","student_email","personal_email"
        ])
    sql = """
      SELECT id, first_name, last_name, classification, major, v_number, student_email, personal_email
      FROM members
      WHERE
        LOWER(first_name)    LIKE LOWER(:pat) OR
        LOWER(last_name)     LIKE LOWER(:pat) OR
        LOWER(student_email) LIKE LOWER(:pat) OR
        LOWER(personal_email)LIKE LOWER(:pat)
      ORDER BY last_name, first_name
      LIMIT :limit
    """
    pat = f"%{q}%"
    with ENGINE.begin() as c:
        rows = c.execute(text(sql), {"pat": pat, "limit": limit}).mappings().all()
    return pd.DataFrame(rows)

def check_in(event_id: str, member_id: str, method: str = "manual") -> Dict:
    """
    Insert attendance if not already present; return a joined view of the check-in.
    Duplicate means same (event_id, member_id) already exists.
    """
    # Ensure event & member exist
    with ENGINE.begin() as c:
        ev = c.execute(text("SELECT id, name, event_date, location FROM events WHERE id = :id"),
                       {"id": event_id}).mappings().first()
        if not ev:
            raise ValueError("Event not found.")

        mem = c.execute(text("""
            SELECT id, first_name, last_name, classification, major,
                   student_email, personal_email, v_number
            FROM members WHERE id = :id
        """), {"id": member_id}).mappings().first()
        if not mem:
            raise ValueError("Member not found.")

        # Duplicate?
        dup = c.execute(text("""
            SELECT event_id, member_id, checked_in_at, method
            FROM attendance
            WHERE event_id = :e AND member_id = :m
            ORDER BY checked_in_at DESC
            LIMIT 1
        """), {"e": event_id, "m": member_id}).mappings().first()

        if dup:
            # Return existing (duplicate)
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

        # Insert new check-in
        ins = c.execute(text("""
            INSERT INTO attendance (event_id, member_id, checked_in_at, method)
            VALUES (:e, :m, NOW(), :method)
            RETURNING event_id, member_id, checked_in_at, method
        """), {"e": event_id, "m": member_id, "method": method}).mappings().first()

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
# UI — pick event
# =========================
with st.sidebar:
    st.caption("Member-facing check-in (DB-backed)")
    if st.button("Refresh"):
        st.cache_data.clear()

ev_df = list_events()
if ev_df.empty:
    st.warning("No events yet. Ask an admin to create one in the Admin Console.")
    st.stop()

ev_df["label"] = ev_df["id"] + " — " + ev_df["name"] + " (" + ev_df["event_date"].astype(str) + ")"
choice = st.selectbox("Select Event", ev_df["label"].tolist())
current_event_id = ev_df.loc[ev_df["label"] == choice, "id"].iloc[0]

# =========================
# Existing Member — search/edit/check-in
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

        st.markdown(f"**{h.get('first_name','')} {h.get('last_name','')}** • {email_disp} • {(klass or '').title()}  \nID: `{mid}`")

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
                # Update member via DB helper (id-based upsert)
                payload = {
                    "id": mid,
                    "first_name": fn.strip(),
                    "last_name": ln.strip(),
                    "classification": normalize_classification(cl),
                    "major": _norm(major),
                    "v_number": _norm(vnum),
                    "student_email": _norm(se),
                    "personal_email": _norm(pe),
                    "created_at": None,  # let DB default
                }
                db_upsert_member(payload)

                res = check_in(current_event_id, mid, method="verify")
                # Clear search/results then notify
                st.session_state.existing_hits = pd.DataFrame()
                if res.get("duplicate"):
                    st.info(f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                else:
                    st.success(f"✅ Checked in {res['member_name']}!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Check-in failed: {e}")

# =========================
# Register New Attendee
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
        # Generate a stable id (prefer V-number; else UUID)
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
            st.info(f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
        else:
            st.success(f"Created & checked in {res['member_name']} to {res.get('event_name','event')}!")
        # Clear search/results for next person:
        st.session_state.existing_hits = pd.DataFrame()
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Check-in failed: {e}")






