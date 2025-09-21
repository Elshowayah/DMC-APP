# admin.py — Admin Console (DB-backed)
# Python 3.9+

from __future__ import annotations
import os
from uuid import uuid4
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine.url import make_url

# ---- DB helpers
from db import (
    ENGINE,
    create_event as db_create_event,
    upsert_member as db_upsert_member,
)

# =========================
# Page / constants
# =========================
st.set_page_config(page_title="👩‍💼 Admin Console", page_icon="📊", layout="wide")
st.title("👩‍💼 Admin Console (DB-backed)")

CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# Debug banner to ensure this app is pointed at the same DB as check-in
try:
    u = make_url(ENGINE.url)  # type: ignore[arg-type]
    st.caption(f"DB: {u.host}/{u.database}")
except Exception as e:
    st.caption(f"DB not ready: {e}")

# =========================
# Helpers
# =========================
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    slug = "_".join("".join(out).split("_"))
    return slug.strip("_") or f"event_{uuid4().hex[:8]}"

def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 or None

def normalize_classification(val: Optional[str]) -> str:
    v = (val or "").strip().lower()
    mapping = {"freshmen": "freshman", "sophmore": "sophomore", "jr": "junior", "sr": "senior"}
    return v if v in CLASS_CHOICES else mapping.get(v, "freshman")

@st.cache_data(ttl=5, show_spinner=False)
def load_databrowser(limit: int = 2000) -> pd.DataFrame:
    """
    Live joined view from Postgres. Always returns a DataFrame (never None).
    """
    sql = """
        SELECT
          a.event_id,
          e.name       AS event_name,
          e.event_date AS event_date,
          e.location   AS event_location,
          a.member_id,
          m.first_name,
          m.last_name,
          m.classification,
          m.major,
          m.student_email,
          m.personal_email,
          m.v_number,
          a.checked_in_at,
          a.method
        FROM attendance a
        JOIN events  e ON e.id = a.event_id
        JOIN members m ON m.id = a.member_id
        ORDER BY a.checked_in_at DESC
        LIMIT :limit
    """
    try:
        with ENGINE.begin() as c:
            rows = c.execute(text(sql), {"limit": limit}).mappings().all()
    except Exception as e:
        st.error(f"DB query failed: {e}")
        return pd.DataFrame()  # never None

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id","event_name","event_date","event_location",
                "member_id","first_name","last_name","classification","major",
                "student_email","personal_email","v_number",
                "checked_in_at","method","member_name"
            ]
        )

    df = pd.DataFrame(rows)
    df["member_name"] = (df.get("first_name", "").fillna("") + " " +
                         df.get("last_name", "").fillna("")).str.strip()
    return df

def clear_cache():
    st.cache_data.clear()

# =========================
# Visualization helpers
# =========================
def _pg_tables_and_fks():
    """Return (tables: [str], fks: List[dict]) from Postgres information_schema."""
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
    """Make a small Graphviz ER diagram (as DOT string) from tables + FKs."""
    lines = [
        "digraph ER {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded, fontsize=10];",
    ]
    for t in tables:
        lines.append(f'  "{t}";')
    for fk in fks:
        ft, fc, tt, tc = fk["from_table"], fk["from_column"], fk["to_table"], fk["to_column"]
        lines.append(f'  "{ft}" -> "{tt}" [label="{fc} → {tc}", fontsize=9];')
    lines.append("}")
    return "\n".join(lines)

def _graphviz_flow(db_label: str = "Postgres (db.py)") -> str:
    """Simple data-flow sketch."""
    return r"""
digraph G {
  rankdir=LR;
  node [shape=box, style=rounded, fontsize=10];

  UI     [label="Streamlit UI\n(admin.py / checkin.py)"];
  LOGIC  [label="App Logic\n(create_event / upsert_member / check-in)"];
  DB     [label="%s", shape=cylinder];
  VIEW   [label="Data Browser\n(DataFrame + filters)"];

  UI  -> LOGIC;
  LOGIC -> DB;
  DB -> VIEW;
}
""" % db_label

# =========================
# Tables (DB) helpers
# =========================
@st.cache_data(ttl=10, show_spinner=False)
def load_members_table(limit: int = 5000) -> pd.DataFrame:
    sql = """
      SELECT id, first_name, last_name, classification, major,
             v_number, student_email, personal_email,
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
        a.event_id,
        e.name        AS event_name,
        e.event_date  AS event_date,
        e.location    AS event_location,
        a.member_id,
        m.first_name, m.last_name, m.classification, m.major,
        m.v_number, m.student_email, m.personal_email,
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

# =========================
# UI
# =========================
with st.sidebar:
    mode = st.radio(
        "Mode",
        ["Data Browser (DB)", "Add Member", "Create Event", "Import Members (to DB)", "Data Map (Visuals)", "Tables (DB)"],
    )
    show_debug = st.checkbox("Show DB counts", value=False)

# ---------- DATA BROWSER ----------
if mode == "Data Browser (DB)":
    st.subheader("Data Browser (live from Postgres)")
    if st.button("Refresh"):
        clear_cache()
    df = load_databrowser(2000)

    if df.empty:
        st.info("No check-ins yet.")
    else:
        # Filters
        majors = sorted([m for m in df["major"].fillna("").map(str).map(str.strip).unique() if m])
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            ev_name = st.text_input("Filter by event name contains")
        with c2:
            klass = st.multiselect("Filter by classification", CLASS_CHOICES, default=[])
        with c3:
            start_date = st.date_input("Start date", value=None)
        with c4:
            end_date = st.date_input("End date", value=None)
        with c5:
            selected_majors = st.multiselect("Filter by major", options=majors, default=[])

        work = df.copy()
        if ev_name:
            work = work[work["event_name"].astype(str).str.contains(ev_name, case=False, na=False)]
        if klass:
            work = work[work["classification"].isin(klass)]
        if selected_majors:
            work = work[work["major"].isin(selected_majors)]
        if start_date:
            work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date >= start_date]
        if end_date:
            work = work[pd.to_datetime(work["event_date"], errors="coerce").dt.date <= end_date]

        q = st.text_input("Search name/email", placeholder="Search attendance…").strip().lower()
        if q:
            fields = []
            for col in ["member_name","first_name","last_name","student_email","personal_email","event_name"]:
                if col in work.columns:
                    fields.append(work[col].astype(str).str.lower().str.contains(q, na=False))
            if fields:
                mask = fields[0]
                for f in fields[1:]:
                    mask |= f
                work = work[mask]

        st.caption(f"Showing {len(work)} of {len(df)} rows")
        show_cols = [
            "event_name","event_date","event_location",
            "member_name","classification","major",
            "checked_in_at","method"
        ]
        show_cols = [c for c in show_cols if c in work.columns]
        st.dataframe(work.sort_values("checked_in_at", ascending=False)[show_cols],
                     use_container_width=True, hide_index=True)
        st.download_button("📥 Download filtered view (CSV)",
                           work.to_csv(index=False).encode("utf-8"),
                           file_name="databrowser_filtered.csv",
                           mime="text/csv")

# ---------- ADD MEMBER ----------
elif mode == "Add Member":
    st.subheader("Add a Member (writes to DB)")
    with st.form("add_member"):
        c1, c2 = st.columns(2)
        with c1:
            fn = st.text_input("First name")
            major = st.text_input("Major")
            v  = st.text_input("V-number (optional, used as ID if provided)")
            se = st.text_input("Student email")
        with c2:
            ln = st.text_input("Last name")
            cl = st.selectbox("Classification", CLASS_CHOICES, index=0)
            pe = st.text_input("Personal email")
        submit = st.form_submit_button("Save")

    if submit:
        if not fn.strip() or not ln.strip():
            st.error("First and last name are required.")
        else:
            # Guarantee a non-null primary key even if V-number is blank
            member_id = v.strip() if v.strip() else f"m_{uuid4().hex}"
            try:
                db_upsert_member({
                    "id": member_id,
                    "first_name": fn.strip(),
                    "last_name": ln.strip(),
                    "classification": normalize_classification(cl),
                    "major": _norm(major),
                    "v_number": _norm(v),
                    "student_email": _norm(se),
                    "personal_email": _norm(pe),
                    "created_at": None,  # db default NOW()
                })
                st.success(f"Saved {fn} {ln} (id {member_id}).")
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
                db_create_event({
                    "id": event_id,
                    "name": name.strip(),
                    "event_date": dt.strip() or str(date.today()),
                    "location": loc.strip(),
                })
                st.success(f"Created event: {name} ({dt})  → id={event_id}")
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
            if not u:
                raise ValueError("Empty URL.")
            if u.endswith("/pubhtml"):
                u = u[:-8] + "?output=csv"
            if "output=csv" not in u and "googleapis.com" not in u:
                if "pub?" in u:
                    u += "&output=csv"
                else:
                    u += "?output=csv"
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
                    if len(parts) == 1:
                        fn, ln = parts[0], ""
                    else:
                        fn, ln = " ".join(parts[:-1]), parts[-1]
            return {
                "id": (r.get("v_number") or r.get("V-number") or "").strip() or f"m_{uuid4().hex}",
                "first_name": fn,
                "last_name": ln,
                "classification": normalize_classification(r.get("classification") or r.get("Classification")),
                "major": _norm(r.get("major") or r.get("Major")),
                "v_number": _norm(r.get("v_number") or r.get("V-number")),
                "student_email": _norm(r.get("student_email") or r.get("Email")),
                "personal_email": _norm(r.get("personal_email") or r.get("Personal Email")),
                "created_at": None,
            }

        norm = [_norm_row(r) for r in st.session_state.import_rows]
        df_norm = pd.DataFrame(norm)
        st.dataframe(df_norm.head(30), use_container_width=True)

        if st.button("Import into DB"):
            ok = 0
            fail = 0
            for row in norm:
                try:
                    db_upsert_member(row)
                    ok += 1
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
    st.graphviz_chart(_graphviz_flow(db_label))

    st.subheader("🗺️ ER Diagram (live from Postgres)")
    try:
        tables, fks = _pg_tables_and_fks()
        if not tables:
            st.info("No tables found in schema 'public'.")
        else:
            dot = _graphviz_er(tables, fks)
            st.graphviz_chart(dot)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Tables**")
                st.write(tables)
            with c2:
                st.write("**Foreign Keys**")
                if fks:
                    st.dataframe(pd.DataFrame(fks), use_container_width=True, hide_index=True)
                else:
                    st.caption("No foreign keys detected.")
    except Exception as e:
        st.error(f"Could not build ER diagram: {e}")

# ---------- TABLES (DB) ----------
elif mode == "Tables (DB)":
    st.subheader("📋 Members & Events (from Postgres)")
    tabs = st.tabs(["Members", "Events & Attendees", "All Check-Ins (joined)"])

    # ===== Members tab =====
    with tabs[0]:
        st.caption("Raw members table with quick filters. Export as CSV below.")
        if st.button("Refresh members"):
            st.cache_data.clear()
        mdf = load_members_table()

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            klass = st.multiselect("Classification", CLASS_CHOICES, default=[])
        with c2:
            major_q = st.text_input("Major contains", "")
        with c3:
            name_q = st.text_input("Name/Email contains", "")

        work = mdf.copy()
        if klass:
            work = work[work["classification"].isin(klass)]
        if major_q:
            work = work[work["major"].astype(str).str.contains(major_q, case=False, na=False)]
        if name_q:
            pat = name_q.strip().lower()
            cols = ["first_name","last_name","student_email","personal_email","v_number"]
            mask = False
            for col in cols:
                if col in work.columns:
                    colmask = work[col].astype(str).str.lower().str.contains(pat, na=False)
                    mask = colmask if isinstance(mask, bool) else (mask | colmask)
            work = work[mask] if not isinstance(mask, bool) else work

        st.caption(f"{len(work)} of {len(mdf)} members")
        st.dataframe(work, use_container_width=True, hide_index=True)
        st.download_button("📥 Download members (CSV)",
                           work.to_csv(index=False).encode("utf-8"),
                           file_name="members_export.csv",
                           mime="text/csv")

    # ===== Events & Attendees tab =====
    with tabs[1]:
        st.caption("Pick an event to see everyone who attended with their member info.")
        if st.button("Refresh events/attendees"):
            st.cache_data.clear()

        ev = load_events_index()
        if ev.empty:
            st.info("No events yet.")
        else:
            left, right = st.columns([1,2])
            with left:
                st.write("**Events**")
                st.dataframe(ev, use_container_width=True, hide_index=True)
                ev["label"] = ev["id"] + " — " + ev["name"] + " (" + ev["event_date"].astype(str) + ")"
                pick = st.selectbox("Select event", ev["label"].tolist())
                event_id = ev.loc[ev["label"] == pick, "id"].iloc[0]

            with right:
                adf = load_event_attendees(event_id)
                st.write(f"**Attendees for:** {pick}")
                if adf.empty:
                    st.info("No check-ins for this event yet.")
                else:
                    show_cols = [
                        "checked_in_at","method",
                        "member_id","first_name","last_name","classification","major",
                        "student_email","personal_email","v_number"
                    ]
                    show_cols = [c for c in show_cols if c in adf.columns]
                    st.dataframe(adf[show_cols], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download attendees (CSV)",
                                       adf.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{event_id}_attendees.csv",
                                       mime="text/csv")

    # ===== All Check-Ins (joined) tab =====
    with tabs[2]:
        st.caption("Full joined view across events + attendance + members (same as Data Browser, but flat table).")
        if st.button("Refresh joined view"):
            st.cache_data.clear()
        df = load_databrowser(5000)
        if df.empty:
            st.info("No check-ins yet.")
        else:
            show_cols = [
                "event_name","event_date","event_location",
                "first_name","last_name","classification","major",
                "student_email","personal_email","v_number",
                "checked_in_at","method"
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols].sort_values("checked_in_at", ascending=False),
                         use_container_width=True, hide_index=True)
            st.download_button("📥 Download all check-ins (CSV)",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name="all_checkins_joined.csv",
                               mime="text/csv")

# ---------- Debug counts ----------
if show_debug:
    try:
        with ENGINE.begin() as c:
            counts = {
                "members": c.execute(text("SELECT COUNT(*) FROM members")).scalar(),
                "events": c.execute(text("SELECT COUNT(*) FROM events")).scalar(),
                "attendance": c.execute(text("SELECT COUNT(*) FROM attendance")).scalar(),
            }
        st.write({"counts": counts})
    except Exception as e:
        st.write({"counts_error": str(e)})


