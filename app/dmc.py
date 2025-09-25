# =============================
# dmc.py — Streamlit App (Check-In + Admin)
# =============================
from __future__ import annotations

from uuid import uuid4
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine.url import make_url

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
st.set_page_config(page_title="DMC Check-In & Admin", page_icon="🎟️", layout="wide")
st.title("🎟️ DMC — Check-In & Admin")
CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# ---------------------------------
# Helpers
# ---------------------------------
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s2 = s.strip()
    return s2 or None

def normalize_classification(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    mapping = {"freshmen":"freshman","sophmore":"sophomore","jr":"junior","sr":"senior","alum":"alumni"}
    return v if v in CLASS_CHOICES else mapping.get(v, "freshman")

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = [ch if ch.isalnum() else "_" for ch in s]
    slug = "_".join("".join(out).split("_"))
    return slug.strip("_") or f"event_{uuid4().hex[:8]}"

def clear_cache():
    try: st.cache_data.clear()
    except Exception: pass

def yn_to_bool(v: str) -> bool:
    return (v or "").strip().lower() in ("y","yes","true","1","✅","✔","ok")

def _dsn_caption() -> str:
    try:
        u = make_url(ENGINE.url)  # type: ignore[arg-type]
        return f"DB → host={u.host or '<none>'} db={u.database or '<none>'} user={u.username or '<none>'}"
    except Exception as e:
        return f"DB → (unavailable: {type(e).__name__})"

def flash(kind: str, msg: str) -> None:
    """Queue a cross-rerun banner and toast."""
    st.session_state["_flash"] = {"kind": kind, "msg": msg}
    st.session_state["_post_action"] = True

def show_flash_once():
    """Render one queued banner+toast, then forget it."""
    f = st.session_state.pop("_flash", None)
    if not f: return
    kind = f.get("kind", "success")
    msg = f.get("msg", "")
    # banner
    if   kind == "success": st.success(msg)
    elif kind == "warning": st.warning(msg)
    elif kind == "error":   st.error(msg)
    else:                   st.info(msg)
    # toast (best-effort for newer Streamlit)
    try:
        st.toast(msg)
    except Exception:
        pass

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
            "student_email","linkedin_yes","updated_resume_yes"
        ])
    pat = f"%{q}%"
    sql = """
        SELECT id, first_name, last_name, classification, major, student_email,
               linkedin_yes, updated_resume_yes
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
               linkedin_yes, updated_resume_yes, created_at, updated_at
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
          m.student_email, m.linkedin_yes, m.updated_resume_yes,
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
            "student_email","linkedin_yes","updated_resume_yes",
            "checked_in_at","method","member_name"
        ])
    df = pd.DataFrame(rows)
    df["member_name"] = (df.get("first_name","").fillna("") + " " + df.get("last_name","").fillna("")).str.strip()
    return df

@st.cache_data(ttl=10, show_spinner=False)
def load_members_table(limit: int = 5000) -> pd.DataFrame:
    sql = """
        SELECT id, first_name, last_name, classification, major,
               student_email, linkedin_yes, updated_resume_yes,
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
          m.student_email, m.linkedin_yes, m.updated_resume_yes,
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
    q = st.text_input("Search by name or student email (min 3 characters)", placeholder="Start typing…").strip()

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

            with st.form(f"ex_edit_{mid}", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    fn = st.text_input("First name", value=sel.get("first_name", "") or "")
                    major = st.text_input("Major", value=sel.get("major", "") or "")
                    se = st.text_input("Student email", value=sel.get("student_email", "") or "")
                    li_choice = st.selectbox(
                        "LinkedIn profile?",
                        ["No", "Yes"],
                        index=1 if bool(sel.get("linkedin_yes")) else 0,
                    )
                with c2:
                    ln = st.text_input("Last name", value=sel.get("last_name", "") or "")
                    cl = st.selectbox("Classification", CLASS_CHOICES, index=class_idx)
                    resume_choice = st.selectbox(
                        "Do you have an UPDATED resume?",
                        ["No", "Yes"],
                        index=1 if bool(sel.get("updated_resume_yes")) else 0,
                    )
                submit_existing = st.form_submit_button("Save & Check-In ✅")

            if submit_existing:
                try:
                    # Save edits (no strict requireds here)
                    db_upsert_member({
                        "id": mid,
                        "first_name": fn.strip(),
                        "last_name": ln.strip(),
                        "classification": normalize_classification(cl),
                        "major": _norm(major),
                        "student_email": _norm(se),
                        "linkedin_yes": yn_to_bool(li_choice),
                        "updated_resume_yes": yn_to_bool(resume_choice),
                        "created_at": None,
                    })
                    res = check_in(current_event_id, mid, method="verify")
                    if res.get("duplicate"):
                        flash("info", f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                    else:
                        flash("success", f"Success — {res['member_name']} signed in ✔")
                    st.rerun()
                except Exception as e:
                    st.error(f"Check-in failed: {type(e).__name__}: {e}")

    # Register new attendee (strict requireds)
    st.divider()
    st.subheader("Register New Attendee (and Check-In)")

    with st.form("register_and_checkin", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            r_fn = st.text_input("First name*")
            r_major = st.text_input("Major")
            r_se = st.text_input("Student email*")
            r_li = st.selectbox("LinkedIn profile?", ["No", "Yes"], index=0)
        with c2:
            r_ln = st.text_input("Last name*")
            r_cl = st.selectbox("Classification*", CLASS_CHOICES, index=0, key="reg_class")
            r_resume = st.selectbox("Do you have an UPDATED resume?", ["No", "Yes"], index=0)
        submit_new = st.form_submit_button("Create Member & Check-In ✅")

    if submit_new:
        # required: first, last, email, classification
        missing = []
        if not r_fn.strip(): missing.append("first name")
        if not r_ln.strip(): missing.append("last name")
        if not r_se.strip(): missing.append("email")
        if not (r_cl or "").strip(): missing.append("classification")
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
                        "linkedin_yes": yn_to_bool(r_li),
                        "updated_resume_yes": yn_to_bool(r_resume),
                        "created_at": None,
                    }
                )
                res = check_in(current_event_id, member_id, method="register")
                if res.get("duplicate"):
                    flash("info",
                          f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                else:
                    flash("success",
                          f"🎉 Congrats — successfully registered {res['member_name']} and checked in.")
                st.rerun()
            except Exception as e:
                st.error(f"Check-in failed: {type(e).__name__}: {e}")

    if st.session_state.pop("_post_action", False):
        st.stop()

# ---------------------------------
# ADMIN (PASSWORD-PROTECTED)
# ---------------------------------
else:
    # Auth gate
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
                "Data Browser (DB)",
                "Add Member",
                "Create Event",
                "Import Members (to DB)",
                "Data Map (Visuals)",
                "Delete Row (DB)",   # safe delete
                "Tables (DB)",
            ],
        )
        st.checkbox("Show DB counts", value=False, key="adm_counts")

    # ---------- DATA BROWSER ----------
    if mode == "Data Browser (DB)":
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
            majors = sorted([m for m in df["major"].fillna("").map(str).map(str.strip).unique() if m])
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: ev_name = st.text_input("Filter by event name contains")
            with c2: klass = st.multiselect("Filter by classification", CLASS_CHOICES, default=[])
            with c3: start_date = st.date_input("Start date", value=None)
            with c4: end_date = st.date_input("End date", value=None)
            with c5: selected_majors = st.multiselect("Filter by major", options=majors, default=[])

            work = df.copy()
            if ev_name: work = work[work["event_name"].astype(str).str.contains(ev_name, case=False, na=False)]
            if klass: work = work[work["classification"].isin(klass)]
            if selected_majors: work = work[work["major"].isin(selected_majors)]
            if start_date: work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date >= start_date]
            if end_date: work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date <= end_date]

            q2 = st.text_input("Search name/email", placeholder="Search attendance…").strip().lower()
            if q2:
                fields = []
                for col in ["member_name","first_name","last_name","student_email","event_name"]:
                    if col in work.columns:
                        fields.append(work[col].astype(str).str.lower().str.contains(q2, na=False))
                if fields:
                    mask = fields[0]
                    for f in fields[1:]: mask |= f
                    work = work[mask]

            st.caption(f"Showing {len(work)} of {len(df)} rows")
            show_cols = [
                "event_name","event_date","event_location",
                "member_name","classification","major",
                "linkedin_yes","updated_resume_yes",
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

    # ---------- ADD MEMBER ----------
    elif mode == "Add Member":
        st.subheader("Add a Member (writes to DB)")
        with st.form("add_member", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                fn = st.text_input("First name")
                major = st.text_input("Major")
                se = st.text_input("Student email")
                li = st.selectbox("LinkedIn profile?", ["No", "Yes"], index=0)
            with c2:
                ln = st.text_input("Last name")
                cl = st.selectbox("Classification", CLASS_CHOICES, index=0)
                resume = st.selectbox("Do you have an UPDATED resume?", ["No", "Yes"], index=0)
            submit = st.form_submit_button("Save")

        if submit:
            if not fn.strip() or not ln.strip():
                st.error("First and last name are required.")
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
                            "linkedin_yes": yn_to_bool(li),
                            "updated_resume_yes": yn_to_bool(resume),
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

    # ---------- IMPORT MEMBERS ----------
    elif mode == "Import Members (to DB)":
        st.subheader("Import members from a published CSV/Google Sheet (writes to DB)")
        st.caption("If using Google Sheets, publish the sheet and use the link ending with `/pub?output=csv`.")

        url = st.text_input("Public CSV URL")
        if st.button("Preview"):
            try:
                u = (url or "").strip()
                if not u: raise ValueError("Empty URL.")
                if u.endswith("/pubhtml"): u = u[:-8] + "?output=csv"
                if "output=csv" not in u and "googleapis.com" not in u:
                    u += "&output=csv" if "pub?" in u else "?output=csv"
                df_raw = pd.read_csv(u, dtype=str).fillna("")
                if df_raw.shape[1] == 1:
                    raise ValueError("Only one column detected — likely not the CSV export of the correct tab/range.")
                st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
                st.dataframe(df_raw.head(20), use_container_width=True)
                st.session_state.import_rows = df_raw.to_dict(orient="records")
            except Exception as e:
                st.error(f"Preview failed: {e}")

        if "import_rows" in st.session_state and st.session_state.import_rows:
            st.subheader("Mapped / Normalized preview")

            def _norm_row(r: Dict) -> Dict:
                fn = (r.get("first_name") or r.get("First name") or r.get("First") or "").strip()
                ln = (r.get("last_name") or r.get("Last name") or r.get("Last") or "").strip()
                if not fn and not ln:
                    nm = (r.get("name") or r.get("Name") or "").strip()
                    if nm:
                        parts = nm.split()
                        fn, ln = (parts[0], "") if len(parts) == 1 else (" ".join(parts[:-1]), parts[-1])
                li_raw = r.get("linkedin_yes") or r.get("LinkedIn") or r.get("LinkedIn profile?")
                resume_raw = r.get("updated_resume_yes") or r.get("UPDATED resume") or r.get("Do you have an UPDATED resume?")
                return {
                    "id": f"m_{uuid4().hex}",
                    "first_name": fn,
                    "last_name": ln,
                    "classification": normalize_classification(r.get("classification") or r.get("Classification")),
                    "major": _norm(r.get("major") or r.get("Major")),
                    "student_email": _norm(r.get("student_email") or r.get("Email")),
                    "linkedin_yes": yn_to_bool(str(li_raw)) if li_raw is not None else False,
                    "updated_resume_yes": yn_to_bool(str(resume_raw)) if resume_raw is not None else False,
                    "created_at": None,
                }

            norm = [_norm_row(r) for r in st.session_state.import_rows]
            df_norm = pd.DataFrame(norm)
            st.dataframe(df_norm.head(30), use_container_width=True)

            if st.button("Import into DB"):
                ok = 0; fail = 0
                for row in norm:
                    try:
                        db_upsert_member(row); ok += 1
                    except Exception as e:
                        fail += 1
                        st.warning(f"Row failed ({row.get('first_name','')} {row.get('last_name','')}): {e}")
                st.success(f"Import complete. OK: {ok}, Failed: {fail}")
                clear_cache()

    # ---------- DATA MAP / VISUALS ----------
    elif mode == "Data Map (Visuals)":
        st.subheader("📈 Data Flow")
        try:
            u = make_url(ENGINE.url)  # type: ignore[arg-type]
            db_label = f"Postgres ({u.host}/{u.database})"
        except Exception:
            db_label = "Postgres (db.py)"
        st.graphviz_chart(
            r"""
            digraph G {
              rankdir=LR;
              node [shape=box, style=rounded, fontsize=10];
              UI     [label="Streamlit UI\n(Check-In / Admin)"];
              LOGIC  [label="App Logic\n(create_event / upsert_member / check-in)"];
              DB     [label="%s", shape=cylinder];
              VIEW   [label="Data Browser\n(DataFrame + filters)"];
              UI  -> LOGIC; LOGIC -> DB; DB -> VIEW;
            }
            """ % db_label
        )

        st.subheader("🗺️ ER Diagram (live from Postgres)")
        def _pg_tables_and_fks():
            tables_q = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public'
                ORDER BY table_name
            """
            fks_q = """
                SELECT
                    tc.table_name      AS from_table,
                    kcu.column_name    AS from_column,
                    ccu.table_name     AS to_table,
                    ccu.column_name     AS to_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                     ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema    = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                     ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema    = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                ORDER BY from_table, to_table, from_column
            """
            with ENGINE.begin() as c:
                tables = [r[0] for r in c.execute(text(tables_q)).all()]
                fks = [dict(zip(["from_table","from_column","to_table","to_column"], r))
                       for r in c.execute(text(fks_q)).all()]
            return tables, fks

        def _graphviz_er(tables: List[str], fks: List[dict]) -> str:
            lines = ["digraph ER {","  rankdir=LR;","  node [shape=box, style=rounded, fontsize=10];"]
            for t in tables: lines.append(f'  "{t}";')
            for fk in fks:
                ft, fc, tt, tc = fk["from_table"], fk["from_column"], fk["to_table"], fk["to_column"]
                lines.append(f'  "{ft}" -> "{tt}" [label="{fc} → {tc}", fontsize=9];')
            lines.append("}")
            return "\n".join(lines)

        try:
            tables, fks = _pg_tables_and_fks()
            if not tables:
                st.info("No tables found in schema 'public'.")
            else:
                st.graphviz_chart(_graphviz_er(tables, fks))
        except Exception as e:
            st.error(f"Could not build ER diagram: {e}")

    # ---------- DELETE ROW (DB, safe) ----------
    elif mode == "Delete Row (DB)":
        st.sidebar.warning("⚠️ Deletions are permanent. Double-check before confirming.")
        st.subheader("Delete a single row from Postgres")

        TABLES = {
            "members": {
                "key_cols": ["id"],
                "preview_cols": [
                    "id","first_name","last_name","classification","major",
                    "student_email","linkedin_yes","updated_resume_yes","created_at","updated_at"
                ],
                "query": """
                    SELECT id, first_name, last_name, classification, major,
                           student_email, linkedin_yes, updated_resume_yes,
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

                st.warning("This action is **permanent** and cannot be undone.")
                with st.form("delete_row_form"):
                    confirm_chk = st.checkbox("I understand this will permanently delete the selected row.")
                    token = st.text_input("Type DELETE to confirm:", value="", placeholder="DELETE")
                    delete_btn = st.form_submit_button(
                        "Delete row", type="primary",
                        disabled=not (confirm_chk and token.strip() == "DELETE"),
                    )

                if delete_btn:
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

    # ---------- TABLES (DB) ----------
    else:
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
            with c1: klass = st.multiselect("Classification", CLASS_CHOICES, default=[])
            with c2: major_q = st.text_input("Major contains", "")
            with c3: name_q = st.text_input("Name/Email contains", "")

            work = mdf.copy()
            if not work.empty:
                if klass: work = work[work["classification"].isin(klass)]
                if major_q: work = work[work["major"].astype(str).str.contains(major_q, case=False, na=False)]
                if name_q:
                    pat = name_q.strip().lower()
                    cols = ["first_name", "last_name", "student_email"]
                    mask = None
                    for col in cols:
                        if col in work.columns:
                            colmask = work[col].astype(str).str.lower().str.contains(pat, na=False)
                            mask = colmask if mask is None else (mask | colmask)
                    if mask is not None: work = work[mask]

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
                            "student_email","linkedin_yes","updated_resume_yes",
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
                    "student_email","linkedin_yes","updated_resume_yes",
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
