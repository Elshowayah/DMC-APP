# checkin.py
# ✅ Event Check-In (member-facing; Python 3.9+)
# - Existing Member: search → immediate editable form(s) → Save & Check-In
# - New Member: register → Check-In
# - Forms clear on submit; search/results clear on successful check-in
# - Guaranteed write to databrowser.csv for every successful check-in (new or duplicate)
# - Idempotent upsert by (event_id, member_id, checked_in_at)

import os
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

# Flattened export schema (Data Browser)
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

# ---------- Guaranteed upsert into databrowser ----------
def upsert_databrowser_from_ids(event_id: str, member_id: str, checked_in_at: str, method: str) -> None:
    """
    Build the flat row from source CSVs and upsert it into databrowser.csv.
    Idempotent on (event_id, member_id, checked_in_at).
    """
    db = load_databrowser()
    m = load_members()
    e = load_events()

    event_id = str(event_id).strip()
    member_id = str(member_id).strip()
    checked_in_at = str(checked_in_at).strip()
    method = (method or "").strip() or "verify"

    e_row_df = e[e["id"].astype(str) == event_id]
    e_row = e_row_df.iloc[0].to_dict() if not e_row_df.empty else {}
    m_row_df = m[m["id"].astype(str) == member_id]
    m_row = m_row_df.iloc[0].to_dict() if not m_row_df.empty else {}

    def _s(d: Dict, k: str, default: str = "") -> str:
        v = d.get(k, default)
        return "" if v is None else str(v).strip()

    row = {
        "event_id":              event_id,
        "event_name":            _s(e_row,  "name"),
        "event_date":            _s(e_row,  "event_date"),
        "event_location":        _s(e_row,  "location"),
        "member_id":             member_id,
        "member_first_name":     _s(m_row, "first_name"),
        "member_last_name":      _s(m_row, "last_name"),
        "member_classification": _s(m_row, "classification"),
        "member_major":          _s(m_row, "major"),
        "member_student_email":  _s(m_row, "student_email").lower(),
        "member_personal_email": _s(m_row, "personal_email").lower(),
        "member_v_number":       _s(m_row, "v_number"),
        "checked_in_at":         checked_in_at,
        "method":                method,
    }
    row["member_name"] = f"{row['member_first_name']} {row['member_last_name']}".strip()

    for c in DATABROWSER_COLS:
        if c not in db.columns:
            db[c] = ""

    dup_mask = (
        (db["event_id"] == row["event_id"]) &
        (db["member_id"] == row["member_id"]) &
        (db["checked_in_at"] == row["checked_in_at"])
    )
    if dup_mask.any():
        return

    db = pd.concat([db[DATABROWSER_COLS], pd.DataFrame([row])[DATABROWSER_COLS]], ignore_index=True)
    save_databrowser(db)

# =========================
# Normalization helpers
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

# =========================
# Domain functions
# =========================
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

def list_events() -> List[Dict]:
    return load_events().sort_values("event_date", ascending=False).to_dict(orient="records")

def find_member(query: str) -> List[Dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    df = load_members()
    mask = (
        df["first_name"].str.lower().str.contains(q, na=False) |
        df["last_name"].str.lower().str.contains(q, na=False) |
        df["student_email"].str.lower().str.contains(q, na=False) |
        df["personal_email"].str.lower().str.contains(q, na=False)
    )
    return df[mask].to_dict(orient="records")

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

def check_in(event_id: Union[int, str], member_id: Union[int, str], method: str = "manual") -> Optional[Dict]:
    """
    Returns a dict with checked_in_at and duplicate flag.
    """
    event_id = str(event_id).strip()
    member_id = str(member_id).strip()

    e = load_events()
    if e.empty:
        return None
    e["id"] = e["id"].astype(str)
    if event_id not in set(e["id"]):
        return None
    event_row = e[e["id"] == event_id].iloc[0].to_dict()

    m = load_members()
    if m.empty or not member_id:
        return None
    m["id"] = m["id"].astype(str)
    if member_id not in set(m["id"]):
        return None
    mem_row = m[m["id"] == member_id].iloc[0].to_dict()

    a = load_attendance()
    if not a.empty:
        a["event_id"] = a["event_id"].astype(str)
        a["member_id"] = a["member_id"].astype(str)
        dup_mask = (a["event_id"] == event_id) & (a["member_id"] == member_id)
        if dup_mask.any():
            existing = a[dup_mask].iloc[-1].to_dict()
            return {
                **existing,
                "event_name": event_row.get("name", ""),
                "event_date": event_row.get("event_date", ""),
                "event_location": event_row.get("location", ""),
                "member_name": f"{mem_row.get('first_name','')} {mem_row.get('last_name','')}".strip(),
                "member_classification": mem_row.get("classification",""),
                "member_student_email": mem_row.get("student_email",""),
                "member_personal_email": mem_row.get("personal_email",""),
                "duplicate": True,
            }

    row = {"event_id": event_id, "member_id": member_id, "checked_in_at": _now_iso(), "method": method}
    a = pd.concat([a, pd.DataFrame([row])], ignore_index=True)
    save_attendance(a)

    return {
        **row,
        "event_name": event_row.get("name", ""),
        "event_date": event_row.get("event_date", ""),
        "event_location": event_row.get("location", ""),
        "member_name": f"{mem_row.get('first_name','')} {mem_row.get('last_name','')}".strip(),
        "member_classification": mem_row.get("classification",""),
        "member_student_email": mem_row.get("student_email",""),
        "member_personal_email": mem_row.get("personal_email",""),
        "duplicate": False,
    }

def rebuild_databrowser_from_sources() -> pd.DataFrame:
    migrate_members_file()
    a = load_attendance().copy()
    m = load_members().copy()
    e = load_events().copy()

    # Normalize ID types
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

    # Member display name
    merged["member_first_name"] = merged.get("member_first_name", "").fillna("").astype(str).str.strip()
    merged["member_last_name"]  = merged.get("member_last_name",  "").fillna("").astype(str).str.strip()
    merged["member_name"] = (merged["member_first_name"] + " " + merged["member_last_name"]).str.strip()

    # Ensure all columns exist
    for c in DATABROWSER_COLS:
        if c not in merged.columns:
            merged[c] = ""

    flat = merged[DATABROWSER_COLS].sort_values(["event_date", "checked_in_at"], ascending=[False, False])
    save_databrowser(flat)
    return flat

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="✅ Event Check-In", page_icon="✅", layout="wide")
st.title("✅ Event Check-In")

with st.sidebar:
    st.caption("Member-facing check-in")
    DEBUG = st.checkbox("Show debug info", value=False)

migrate_members_file()

events = list_events()
if not events:
    st.warning("No events yet. Ask an admin to create one.")
    st.stop()

# Event selection
ev_label_to_id = {f"{e['id']} — {e['name']} ({e['event_date']})": e["id"] for e in events}
ev_choice = st.selectbox("Select Event", list(ev_label_to_id.keys()))
current_event_id = ev_label_to_id[ev_choice]

# ---------------- Existing Member: instant editable forms ----------------
st.divider()
st.subheader("Existing Member — Search, Edit, and Check-In")

# Initialize keys BEFORE widgets render
if "existing_search" not in st.session_state:
    st.session_state["existing_search"] = ""
if "existing_hits" not in st.session_state:
    st.session_state["existing_hits"] = []

with st.form("existing_search_form", clear_on_submit=False):
    q = st.text_input("Search by email or name", key="existing_search", placeholder="Type name or email…").strip()
    do_search = st.form_submit_button("Find Member 🔎")

if do_search:
    hits = find_member(q)
    if any(not str(h.get("id", "")).strip() for h in hits):
        migrate_members_file()
        hits = find_member(q)
    st.session_state["existing_hits"] = hits

hits = st.session_state.get("existing_hits", [])

if do_search and not hits:
    st.info("No members matched your search. Try a different name or email.")

for h in hits:
    mid = str(h.get("id", "")).strip()
    email_disp = h.get("student_email") or h.get("personal_email") or "no email"
    klass = (h.get("classification") or "freshman").lower()
    try:
        class_idx = CLASS_CHOICES.index(klass)
    except ValueError:
        class_idx = 0

    st.markdown(f"**{h.get('first_name','')} {h.get('last_name','')}** • {email_disp} • {(klass or '').title()}  \nID: `{mid}`")

    # Per-member form; clear itself on submit
    with st.form(f"ex_edit_{mid}", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            fn   = st.text_input("First name", value=h.get("first_name",""))
            major = st.text_input("Major", value=h.get("major",""))
            vnum = st.text_input("V-number", value=h.get("v_number",""))
            se   = st.text_input("Student email", value=h.get("student_email",""))
        with c2:
            ln = st.text_input("Last name", value=h.get("last_name",""))
            cl = st.selectbox("Classification", CLASS_CHOICES, index=class_idx)
            pe = st.text_input("Personal email", value=h.get("personal_email",""))
        submit_existing = st.form_submit_button("Save & Check-In ✅")

    if submit_existing:
        try:
            saved = upsert_member({
                "first_name": fn, "last_name": ln, "classification": cl,
                "major": major, "v_number": vnum, "student_email": se, "personal_email": pe,
            })
            member_id = str(saved.get("id", "")).strip() or mid
            res = check_in(current_event_id, member_id, method="verify")
            if res is None:
                st.error("Check-in failed (missing event/member or invalid ID).")
            else:
                upsert_databrowser_from_ids(
                    event_id=str(current_event_id),
                    member_id=str(member_id),
                    checked_in_at=str(res["checked_in_at"]),
                    method=str(res.get("method", "verify")),
                )
                rebuild_databrowser_from_sources()

                # ✅ Clear search + results (avoid assigning to widget key; just remove and rerun)
                st.session_state.pop("existing_hits", None)
                st.session_state.pop("existing_search", None)
                if res.get("duplicate"):
                    st.info(f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
                else:
                    st.success(f"✅ Checked in {res['member_name']}!")
                st.rerun()
        except Exception as e:
            st.error(f"Check-in failed: {e}")
            if DEBUG:
                st.exception(e)

# ---------------- Register New Attendee ----------------
st.divider()
st.subheader("Register New Attendee (and check-in)")

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
    saved = upsert_member({
        "first_name": r_fn, "last_name": r_ln, "classification": r_cl,
        "major": r_major, "v_number": r_v, "student_email": r_se, "personal_email": r_pe
    })
    res = check_in(current_event_id, saved["id"], method="register")
    if res is None:
        st.error("Check-in failed (missing event/member or invalid ID).")
    else:
        upsert_databrowser_from_ids(
            event_id=str(current_event_id),
            member_id=str(saved["id"]),
            checked_in_at=str(res["checked_in_at"]),
            method=str(res.get("method", "register")),
        )
        rebuild_databrowser_from_sources()
        if res.get("duplicate"):
            st.info(f"{res['member_name']} was already checked in for {res.get('event_name','this event')} at {res['checked_in_at']}.")
        else:
            st.success(f"Created & checked in {res['member_name']} to {res.get('event_name','event')}!")
        # No manual field resets needed; form cleared itself. Also clear search/results for next person:
        st.session_state.pop("existing_hits", None)
        st.session_state.pop("existing_search", None)
        st.rerun()





