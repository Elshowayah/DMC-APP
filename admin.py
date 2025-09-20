# admin.py
# 👩‍💼 Admin Console: Data Browser, Members, Events, Imports (Python 3.9+)

import os
import re
from pathlib import Path
from datetime import date, datetime
from typing import List, Tuple, Dict, Optional, Union

import pandas as pd
from db import list_events, create_event, upsert_member, latest_checkins
import streamlit as st


# =========================
# Configuration
# =========================
DATA_DIR = Path(os.getenv("UNIVERSAL_DATA_DIR", "data"))
MEMBERS_CSV = DATA_DIR / "members.csv"
EVENTS_CSV = DATA_DIR / "events.csv"
ATTEND_CSV = DATA_DIR / "attendance.csv"
DATABROWSER_CSV = DATA_DIR / "databrowser.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Canonical schemas
MEMBER_COLS = [
    "id", "first_name", "last_name", "classification", "major",
    "v_number", "student_email", "personal_email", "created_at", "updated_at",
]
EVENT_COLS = ["id", "name", "event_date", "location", "created_at"]
ATTEND_COLS = ["event_id", "member_id", "checked_in_at", "method"]

DATABROWSER_COLS = [
    "event_id", "event_name", "event_date", "event_location",
    "member_id", "member_name", "member_first_name", "member_last_name",
    "member_classification", "member_major", "member_student_email", "member_personal_email",
    "member_v_number", "checked_in_at", "method",
]

CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# =========================
# Utilities (CSV I/O)
# =========================
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def _load_csv(path: Path, cols: List[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str).fillna("")
    return _ensure_cols(df, cols)

def _save_csv(path: Path, df: pd.DataFrame, cols: List[str]) -> None:
    _ensure_cols(df.fillna(""), cols).to_csv(path, index=False)

def load_members() -> pd.DataFrame:
    return _load_csv(MEMBERS_CSV, MEMBER_COLS)

def save_members(df: pd.DataFrame) -> None:
    _save_csv(MEMBERS_CSV, df, MEMBER_COLS)

def load_events() -> pd.DataFrame:
    return _load_csv(EVENTS_CSV, EVENT_COLS)

def save_events(df: pd.DataFrame) -> None:
    _save_csv(EVENTS_CSV, df, EVENT_COLS)

def load_attendance() -> pd.DataFrame:
    return _load_csv(ATTEND_CSV, ATTEND_COLS)

def save_attendance(df: pd.DataFrame) -> None:
    _save_csv(ATTEND_CSV, df, ATTEND_COLS)

def load_databrowser() -> pd.DataFrame:
    if not DATABROWSER_CSV.exists() or DATABROWSER_CSV.stat().st_size == 0:
        pd.DataFrame(columns=DATABROWSER_COLS).to_csv(DATABROWSER_CSV, index=False)
    df = pd.read_csv(DATABROWSER_CSV, dtype=str).fillna("")
    return _ensure_cols(df, DATABROWSER_COLS)

def save_databrowser(df: pd.DataFrame) -> None:
    _save_csv(DATABROWSER_CSV, df, DATABROWSER_COLS)

# =========================
# Normalization helpers & domain functions
# =========================
def split_name(full_name: str) -> Tuple[str, str]:
    full_name = (full_name or "").strip()
    if not full_name:
        return "", ""
    parts = full_name.split()
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]

def normalize_classification(val: str) -> str:
    v = (val or "").strip().lower()
    if v in CLASS_CHOICES:
        return v
    mapping = {"freshmen": "freshman", "sophmore": "sophomore", "jr": "junior", "sr": "senior"}
    return mapping.get(v, "freshman")

def normalize_member_payload(row: Dict) -> Dict:
    fn = str(row.get("first_name", "")).strip()
    ln = str(row.get("last_name", "")).strip()
    if not fn and not ln:
        name = str(row.get("name", "") or row.get("Name", "")).strip()
        if name:
            fn, ln = split_name(name)

    classification = normalize_classification(row.get("classification", row.get("Classification", "")))
    major = str(row.get("major", row.get("Major", "")) or "").strip()
    vnum = str(row.get("v_number", row.get("V-number", "")) or "").strip()
    se = str(row.get("student_email", row.get("Email", "")) or "").strip().lower()
    pe = str(row.get("personal_email", row.get("Personal Email", "")) or "").strip().lower()

    return {
        "first_name": fn,
        "last_name": ln,
        "classification": classification,
        "major": major,
        "v_number": vnum,
        "student_email": se,
        "personal_email": pe,
    }

def migrate_members_file() -> None:
    df = load_members()
    if df.empty:
        return

    changed = False
    for c in ["id","first_name","last_name","student_email","personal_email","classification","major","v_number","created_at","updated_at"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    no_id = (df["id"].str.len() == 0) | (df["id"].str.lower().isin(["nan","none"]))
    if no_id.any():
        nxt = next_id_from(df)
        for i in df[no_id].index:
            df.at[i, "id"] = str(nxt)
            nxt += 1
            changed = True

    if "student_email" in df.columns:
        ne = df["student_email"].str.lower()
        if not ne.equals(df["student_email"]):
            df["student_email"] = ne; changed = True
    if "personal_email" in df.columns:
        ne = df["personal_email"].str.lower()
        if not ne.equals(df["personal_email"]):
            df["personal_email"] = ne; changed = True
    if "classification" in df.columns:
        nc = df["classification"].apply(normalize_classification)
        if not nc.equals(df["classification"]):
            df["classification"] = nc; changed = True

    if changed:
        now = _now_iso()
        if "updated_at" in df.columns:
            df.loc[:, "updated_at"] = df["updated_at"].where(df["updated_at"].str.len() > 0, now)
        save_members(df)

def next_id_from(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    try:
        return int(pd.to_numeric(df["id"], errors="coerce").fillna(0).max()) + 1
    except Exception:
        return 1

def list_members() -> List[Dict]:
    return load_members().to_dict(orient="records")

def list_events() -> List[Dict]:
    return load_events().sort_values("event_date", ascending=False).to_dict(orient="records")

def upsert_member(payload: Dict) -> Dict:
    df = load_members()
    p = normalize_member_payload(payload)

    key_mask = pd.Series([False] * len(df))
    if p["student_email"]:
        key_mask |= (df["student_email"].str.lower() == p["student_email"])
    if p["personal_email"]:
        key_mask |= (df["personal_email"].str.lower() == p["personal_email"])
    if p["first_name"] and p["last_name"] and p["v_number"]:
        key_mask |= (
            (df["first_name"].str.lower() == p["first_name"].lower()) &
            (df["last_name"].str.lower() == p["last_name"].lower()) &
            (df["v_number"].str.lower() == p["v_number"].lower())
        )

    now = _now_iso()
    if key_mask.any():
        idx = df[key_mask].index[0]
        if not str(df.at[idx, "id"]).strip():
            df.at[idx, "id"] = str(next_id_from(df))
        for k, v in p.items():
            if v:
                df.at[idx, k] = v
        df.at[idx, "updated_at"] = now
        saved = df.loc[idx].to_dict()
    else:
        new_id = next_id_from(df)
        new_row = {
            "id": str(new_id),
            **{k: p.get(k, "") for k in ["first_name","last_name","classification","major","v_number","student_email","personal_email"]},
            "created_at": now,
            "updated_at": now,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        saved = new_row

    save_members(df)
    saved["id"] = str(saved.get("id", "")).strip()
    return saved

def create_event(name: str, event_date: str, location: str) -> Dict:
    df = load_events()
    eid = next_id_from(df)
    row = {
        "id": str(eid),
        "name": (name or "").strip(),
        "event_date": (event_date or str(date.today())).strip(),
        "location": (location or "").strip(),
        "created_at": _now_iso(),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_events(df)
    return row

def rebuild_databrowser_from_sources() -> pd.DataFrame:
    migrate_members_file()
    a = load_attendance().copy()
    m = load_members().copy()
    e = load_events().copy()

    for df in (a, m, e):
        for col in ("id", "member_id", "event_id"):
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("").str.strip()

    m_ren = m.rename(columns={
        "id": "member_id",
        "first_name": "member_first_name",
        "last_name": "member_last_name",
        "classification": "member_classification",
        "major": "member_major",
        "student_email": "member_student_email",
        "personal_email": "member_personal_email",
        "v_number": "member_v_number",
    })
    e_ren = e.rename(columns={
        "id": "event_id",
        "name": "event_name",
        "event_date": "event_date",
        "location": "event_location",
    })

    merged = a.merge(m_ren, on="member_id", how="left").merge(e_ren, on="event_id", how="left")
    merged["member_first_name"] = merged.get("member_first_name", "").fillna("").astype(str).str.strip()
    merged["member_last_name"]  = merged.get("member_last_name",  "").fillna("").astype(str).str.strip()
    merged["member_name"] = (merged["member_first_name"] + " " + merged["member_last_name"]).str.strip()

    for c in DATABROWSER_COLS:
        if c not in merged.columns:
            merged[c] = ""

    flat = merged[DATABROWSER_COLS].sort_values(["event_date", "checked_in_at"], ascending=[False, False])
    save_databrowser(flat)
    return flat

# =========================
# Google Sheet CSV fetch (for Import Members)
# =========================
CSV_URL_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/e/.+/pub\?output=csv(?:&gid=\d+)?$")

def validate_gsheets_csv_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("Empty URL.")
    if url.endswith("/pubhtml"):
        url = url[:-8] + "?output=csv"
    if "output=csv" not in url:
        if "pub?" in url:
            url += "&output=csv"
        else:
            url += "?output=csv"
    if not CSV_URL_RE.match(url):
        raise ValueError("Link must be a *published* Google Sheet ending with `/pub?output=csv` (optionally `&gid=`).")
    return url

def fetch_public_csv(url: str) -> pd.DataFrame:
    url = validate_gsheets_csv_url(url)
    df = pd.read_csv(url, dtype=str).fillna("")
    if df.shape[1] == 1:
        raise ValueError("Only one column detected — likely not the CSV export of the correct tab/range.")
    return df

# =========================
# Streamlit UI (Admin only)
# =========================
st.set_page_config(page_title="👩‍💼 Admin Console", page_icon="📊", layout="wide")
st.title("👩‍💼 Admin Console")

with st.sidebar:
    mode = st.radio("Mode", [
        "Data Browser (CSV)",
        "Rebuild Data Browser",
        "Add Member",
        "Create Event",
        "Import Members",
    ])
    DEBUG = st.checkbox("Show debug info", value=False)

# ---------- DATA BROWSER ----------
if mode == "Data Browser (CSV)":
    st.subheader("Data Browser (from databrowser.csv)")
    df = load_databrowser()
    if df.empty:
        st.info("databrowser.csv is empty. Use **Rebuild Data Browser** to generate it.")
    else:
        # Build a clean list of unique majors (skip blanks)
        if "member_major" in df.columns:
            major_options = (
                pd.Series(df["member_major"], dtype=str)
                .fillna("")
                .map(lambda x: x.strip())
            )
            major_options = sorted([m for m in major_options.unique() if m])
        else:
            major_options = []

        # Added a 5th column for Major filter
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
            selected_majors = st.multiselect(
                "Filter by major",
                options=major_options,
                default=[],
                help="Show only rows where the member’s major matches your selection."
            )

        work = df.copy()

        # Event name contains
        if ev_name:
            work = work[work["event_name"].str.contains(ev_name, case=False, na=False)]

        # Classification filter
        if klass:
            work = work[work["member_classification"].isin(klass)]

        # Major filter
        if selected_majors:
            work = work[work["member_major"].isin(selected_majors)]

        # Date range filters
        if start_date:
            work = work[work["event_date"] >= start_date.isoformat()]
        if end_date:
            work = work[work["event_date"] <= end_date.isoformat()]

        # Free-text search
        q = st.text_input("Search name/email in data", placeholder="Search attendance…").strip().lower()
        if q:
            mask = (
                work["member_name"].str.lower().str.contains(q, na=False) |
                work["member_first_name"].str.lower().str.contains(q, na=False) |
                work["member_last_name"].str.lower().str.contains(q, na=False) |
                work["member_student_email"].str.lower().str.contains(q, na=False) |
                work["member_personal_email"].str.lower().str.contains(q, na=False)
            )
            work = work[mask]

        st.caption(f"Showing {len(work)} of {len(df)} rows")
        st.dataframe(
            work.sort_values(["event_date","checked_in_at"], ascending=[False, False]),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "📥 Download filtered view (CSV)",
            work.to_csv(index=False).encode("utf-8"),
            file_name="databrowser_filtered.csv",
            mime="text/csv",
        )


# ---------- REBUILD ----------
elif mode == "Rebuild Data Browser":
    st.subheader("Rebuild databrowser.csv from members/events/attendance")
    if st.button("Build now"):
        flat = rebuild_databrowser_from_sources()
        st.success(f"Rebuilt databrowser.csv with {len(flat)} rows → {DATABROWSER_CSV}")
        st.download_button(
            "Download databrowser.csv",
            flat.to_csv(index=False).encode("utf-8"),
            file_name="databrowser.csv",
            mime="text/csv",
        )

# ---------- ADD MEMBER ----------
elif mode == "Add Member":
    st.subheader("Add a Member")
    with st.form("add_member"):
        c1, c2 = st.columns(2)
        with c1:
            fn = st.text_input("First name")
            major = st.text_input("Major")
            v  = st.text_input("V-number")
            se = st.text_input("Student email")
        with c2:
            ln = st.text_input("Last name")
            cl = st.selectbox("Classification", CLASS_CHOICES, index=0)
            pe = st.text_input("Personal email")
        if st.form_submit_button("Save"):
            saved = upsert_member({
                "first_name": fn, "last_name": ln, "classification": cl,
                "major": major, "v_number": v, "student_email": se, "personal_email": pe
            })
            st.success(f"Saved {saved['first_name']} {saved['last_name']} (id {saved['id']}).")
            rebuild_databrowser_from_sources()

    st.divider()
    st.subheader("Members")
    df_m = load_members()
    st.dataframe(df_m, use_container_width=True, hide_index=True)
    st.download_button("Download members CSV", df_m.to_csv(index=False).encode("utf-8"),
                       file_name="members.csv", mime="text/csv")

# ---------- CREATE EVENT ----------
elif mode == "Create Event":
    st.subheader("Create a new event")
    with st.form("new_event"):
        name = st.text_input("Event name")
        dt = st.text_input("Event date (YYYY-MM-DD)", value=str(date.today()))
        loc = st.text_input("Location")
        if st.form_submit_button("Create"):
            ev = create_event(name, dt, loc)
            st.success(f"Created event {ev['id']}: {ev['name']} ({ev['event_date']})")
            rebuild_databrowser_from_sources()

    st.divider()
    st.subheader("All Events")
    df_e = load_events().sort_values("event_date", ascending=False)
    st.dataframe(df_e, use_container_width=True, hide_index=True)
    st.download_button("Download events CSV", df_e.to_csv(index=False).encode("utf-8"),
                       file_name="events.csv", mime="text/csv")

# ---------- IMPORT MEMBERS ----------
elif mode == "Import Members":
    st.subheader("Import members from a published Google Sheet (CSV)")
    st.caption("Your link must end with **/pub?output=csv** (optionally **&gid=...**).")
    url = st.text_input("Public CSV URL")
    if st.button("Preview"):
        try:
            df_raw = fetch_public_csv(url)
            st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.session_state.import_rows = df_raw.to_dict(orient="records")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    if "import_rows" in st.session_state and st.session_state.import_rows:
        st.subheader("Mapped / Normalized preview")
        norm = [normalize_member_payload(r) for r in st.session_state.import_rows]
        df_norm = pd.DataFrame(norm)
        st.dataframe(df_norm.head(30), use_container_width=True)
        if st.button("Import into universal CSV"):
            before = load_members()
            before_ids = set(before["id"].tolist()) if not before.empty else set()
            for row in norm:
                _ = upsert_member(row)
            after = load_members()
            after_ids = set(after["id"].tolist()) if not after.empty else set()
            added = len(after_ids - before_ids)
            total = len(norm)
            updated = max(total - added, 0)
            st.success(f"Imported {total} rows → {added} new, {updated} updated.")
            rebuild_databrowser_from_sources()
