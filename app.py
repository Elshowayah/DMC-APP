# app.py
import os
import re
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import streamlit as st

# =========================
# Configuration
# =========================
DATA_DIR = Path(os.getenv("UNIVERSAL_DATA_DIR", "data"))
MEMBERS_CSV = DATA_DIR / "members.csv"
EVENTS_CSV = DATA_DIR / "events.csv"
ATTEND_CSV = DATA_DIR / "attendance.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Column definitions
MEMBER_COLS = [
    "id", "first_name", "last_name", "classification",
    "v_number", "student_email", "personal_email", "created_at", "updated_at"
]
EVENT_COLS = ["id", "name", "event_date", "location", "created_at"]
ATTEND_COLS = ["event_id", "member_id", "checked_in_at", "method"]

CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

# =========================
# Utilities (CSV I/O)
# =========================
def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        df = pd.DataFrame(columns=cols)
        df.to_csv(path, index=False)
        return df.copy()
    df = pd.read_csv(path, dtype=str).fillna("")
    # Ensure missing columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def _save_csv(path: Path, df: pd.DataFrame, cols: list[str]):
    df = df.fillna("")
    # Enforce columns & order
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(path, index=False)

def load_members() -> pd.DataFrame:
    return _load_csv(MEMBERS_CSV, MEMBER_COLS)

def save_members(df: pd.DataFrame):
    _save_csv(MEMBERS_CSV, df, MEMBER_COLS)

def load_events() -> pd.DataFrame:
    return _load_csv(EVENTS_CSV, EVENT_COLS)

def save_events(df: pd.DataFrame):
    _save_csv(EVENTS_CSV, df, EVENT_COLS)

def load_attendance() -> pd.DataFrame:
    return _load_csv(ATTEND_CSV, ATTEND_COLS)

def save_attendance(df: pd.DataFrame):
    _save_csv(ATTEND_CSV, df, ATTEND_COLS)

# =========================
# Normalization helpers
# =========================
def split_name(full_name: str) -> tuple[str, str]:
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

def normalize_member_payload(row: dict) -> dict:
    fn = str(row.get("first_name", "")).strip()
    ln = str(row.get("last_name", "")).strip()

    if not fn and not ln:
        name = str(row.get("name", "") or row.get("Name", "")).strip()
        if name:
            fn, ln = split_name(name)

    classification = normalize_classification(row.get("classification", row.get("Classification", "")))
    vnum = str(row.get("v_number", row.get("V-number", "")) or "").strip()
    se = str(row.get("student_email", row.get("Email", "")) or "").strip().lower()
    pe = str(row.get("personal_email", row.get("Personal Email", "")) or "").strip().lower()

    return {
        "first_name": fn,
        "last_name": ln,
        "classification": classification,
        "v_number": vnum,
        "student_email": se,
        "personal_email": pe,
    }

# =========================
# Domain functions
# =========================
def next_id_from(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    try:
        return int(pd.to_numeric(df["id"], errors="coerce").fillna(0).max()) + 1
    except Exception:
        return 1

def list_members() -> list[dict]:
    return load_members().to_dict(orient="records")

def list_events() -> list[dict]:
    return load_events().sort_values("event_date", ascending=False).to_dict(orient="records")

def list_attendance() -> list[dict]:
    return load_attendance().sort_values("checked_in_at", ascending=False).to_dict(orient="records")

def find_member(query: str) -> list[dict]:
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

def upsert_member(payload: dict) -> dict:
    """
    Upsert by student_email (preferred) else personal_email else (first+last+vnum).
    Returns the saved row as dict.
    """
    df = load_members()
    p = normalize_member_payload(payload)

    # Identify match
    key_mask = pd.Series([False]*len(df))
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
        for k, v in p.items():
            if v:  # only overwrite with non-empty
                df.at[idx, k] = v
        df.at[idx, "updated_at"] = now
        saved = df.loc[idx].to_dict()
    else:
        new_id = next_id_from(df)
        new_row = {
            "id": str(new_id),
            **{k: p.get(k, "") for k in ["first_name","last_name","classification","v_number","student_email","personal_email"]},
            "created_at": now,
            "updated_at": now,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        saved = new_row

    save_members(df)
    return saved

def create_event(name: str, event_date: str, location: str) -> dict:
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

def check_in(event_id: int | str, member_id: int | str, method: str = "manual") -> bool:
    a = load_attendance()
    event_id = str(event_id)
    member_id = str(member_id)

    dup = (a["event_id"] == event_id) & (a["member_id"] == member_id)
    if dup.any():
        return False

    row = {
        "event_id": event_id,
        "member_id": member_id,
        "checked_in_at": _now_iso(),
        "method": method,
    }
    a = pd.concat([a, pd.DataFrame([row])], ignore_index=True)
    save_attendance(a)
    return True

# =========================
# Google Sheet CSV fetch
# =========================
CSV_URL_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/e/.+/pub\?output=csv(&gid=\d+)?$")

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
# Streamlit UI
# =========================
st.set_page_config(page_title="Org Attendance", page_icon="✅", layout="wide")
st.title("✅ Org Event Check-In")

mode = st.sidebar.radio("Mode", ["Check-In", "Add Member", "Create Event", "Import Members", "Data Browser"])

# ---------- CHECK-IN ----------
if mode == "Check-In":
    events = list_events()
    if not events:
        st.warning("No events yet. Create one in **Create Event**.")
    else:
        ev_label_to_id = {f"{e['id']} — {e['name']} ({e['event_date']})": e["id"] for e in events}
        ev_choice = st.selectbox("Select Event", list(ev_label_to_id.keys()))
        current_event_id = ev_label_to_id[ev_choice]

        st.subheader("Find Member")
        q = st.text_input("Search by email or name").strip()
        if st.button("Search"):
            hits = find_member(q)
            if not hits:
                st.warning("No match found. You can register them below and check in immediately.")
            else:
                st.success(f"Found {len(hits)} match(es). Expand a row to verify & check in.")
                for m in hits:
                    with st.expander(f"{m['first_name']} {m['last_name']} — {m.get('student_email') or m.get('personal_email')} ({m['classification']})"):
                        with st.form(f"verify_{m['id']}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                fn = st.text_input("First name", value=m["first_name"])
                                vnum = st.text_input("V-number", value=m["v_number"])
                                se = st.text_input("Student email", value=m["student_email"])
                            with c2:
                                ln = st.text_input("Last name", value=m["last_name"])
                                cl = st.selectbox("Classification", CLASS_CHOICES, index=CLASS_CHOICES.index(m["classification"] or "freshman"))
                                pe = st.text_input("Personal email", value=m["personal_email"])

                            submitted = st.form_submit_button("Save updates & Check-In ✅")
                            if submitted:
                                # Save updates to universal CSV
                                saved = upsert_member({
                                    "first_name": fn, "last_name": ln,
                                    "classification": cl, "v_number": vnum,
                                    "student_email": se, "personal_email": pe
                                })
                                # Use saved ID (in case merge/update changed identity fields)
                                member_id = saved["id"]
                                ok = check_in(current_event_id, member_id, method="verify")
                                if ok:
                                    st.success(f"Checked in {saved['first_name']} {saved['last_name']}.")
                                else:
                                    st.info("Already checked in for this event.")

        st.divider()
        st.subheader("Register New Attendee (and check-in)")
        with st.form("register_and_checkin"):
            c1, c2 = st.columns(2)
            with c1:
                r_fn = st.text_input("First name", value="")
                r_v  = st.text_input("V-number", value="")
                r_se = st.text_input("Student email", value="")
            with c2:
                r_ln = st.text_input("Last name", value="")
                r_cl = st.selectbox("Classification", CLASS_CHOICES, index=0, key="reg_class")
                r_pe = st.text_input("Personal email", value="")
            if st.form_submit_button("Create Member & Check-In ✅"):
                saved = upsert_member({
                    "first_name": r_fn, "last_name": r_ln, "classification": r_cl,
                    "v_number": r_v, "student_email": r_se, "personal_email": r_pe
                })
                ok = check_in(current_event_id, saved["id"], method="register")
                if ok:
                    st.success(f"Created and checked in {saved['first_name']} {saved['last_name']}.")
                else:
                    st.info("Member exists and was already checked in.")

        st.divider()
        st.subheader("Attendance")
        a = pd.DataFrame(list_attendance())
        if a.empty:
            st.info("No one has checked in yet.")
        else:
            m = pd.DataFrame(list_members())
            merged = a.merge(m, left_on="member_id", right_on="id", how="left")
            merged["member_name"] = (merged["first_name"].fillna("") + " " + merged["last_name"].fillna("")).str.strip()
            view = merged[merged["event_id"] == str(current_event_id)][
                ["checked_in_at", "member_id", "member_name", "student_email", "classification", "method"]
            ].sort_values("checked_in_at", ascending=False)
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download this event’s attendance (CSV)",
                view.to_csv(index=False).encode("utf-8"),
                file_name=f"event_{current_event_id}_attendance.csv",
                mime="text/csv"
            )

# ---------- ADD MEMBER ----------
elif mode == "Add Member":
    st.subheader("Add a Member")
    with st.form("add_member"):
        c1, c2 = st.columns(2)
        with c1:
            fn = st.text_input("First name")
            v  = st.text_input("V-number")
            se = st.text_input("Student email")
        with c2:
            ln = st.text_input("Last name")
            cl = st.selectbox("Classification", CLASS_CHOICES, index=0)
            pe = st.text_input("Personal email")
        if st.form_submit_button("Save"):
            saved = upsert_member({
                "first_name": fn, "last_name": ln, "classification": cl,
                "v_number": v, "student_email": se, "personal_email": pe
            })
            st.success(f"Saved {saved['first_name']} {saved['last_name']} (id {saved['id']}).")

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
            added, updated = 0, 0
            before_ids = set(load_members()["id"].tolist())
            for row in norm:
                saved = upsert_member(row)
            after_ids = set(load_members()["id"].tolist())
            added = len(after_ids - before_ids)
            total = len(norm)
            updated = max(total - added, 0)
            st.success(f"Imported {total} rows → {added} new, {updated} updated.")

# ---------- DATA BROWSER ----------
elif mode == "Data Browser":
    st.subheader("Members")
    df_m = load_members()
    st.dataframe(df_m, use_container_width=True, hide_index=True)
    st.download_button("Download members CSV", df_m.to_csv(index=False).encode("utf-8"),
                       file_name="members.csv", mime="text/csv")

    st.divider()
    st.subheader("Events")
    df_e = load_events()
    st.dataframe(df_e, use_container_width=True, hide_index=True)
    st.download_button("Download events CSV", df_e.to_csv(index=False).encode("utf-8"),
                       file_name="events.csv", mime="text/csv")

    st.divider()
    st.subheader("Attendance")
    df_a = load_attendance()
    st.dataframe(df_a, use_container_width=True, hide_index=True)
    st.download_button("Download attendance CSV", df_a.to_csv(index=False).encode("utf-8"),
                       file_name="attendance.csv", mime="text/csv")








