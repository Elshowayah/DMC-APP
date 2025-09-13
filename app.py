# app.py
import re
from datetime import date
import streamlit as st
import pandas as pd

# ---- Domain services (your modules) ----
from src.members.manage_members import (
    list_members, find_member, upsert_member, update_classification, graduate_member
)
from src.events.manage_events import list_events, create_event
from src.attendance.manage_attendance import check_in, list_attendance

# ---------------------------------------
# Streamlit Setup
# ---------------------------------------
st.set_page_config(page_title="Org Attendance", page_icon="✅", layout="wide")
st.title("✅ Org Event Check-In")

# ---------------------------------------
# Helpers
# ---------------------------------------
CLASS_CHOICES = ["freshman", "sophomore", "junior", "senior", "alumni"]

def members_df() -> pd.DataFrame:
    try:
        return pd.DataFrame(list_members())
    except Exception:
        return pd.DataFrame()

def events_df() -> pd.DataFrame:
    try:
        return pd.DataFrame(list_events())
    except Exception:
        return pd.DataFrame()

def attendance_df() -> pd.DataFrame:
    try:
        return pd.DataFrame(list_attendance())
    except Exception:
        return pd.DataFrame()

def attendance_for_event_df(event_id: int) -> pd.DataFrame:
    a = attendance_df()
    if a.empty:
        return a
    a = a[a["event_id"].astype(int) == int(event_id)]
    m = members_df()
    if m.empty:
        return a
    merged = a.merge(m, left_on="member_id", right_on="id", how="left")
    merged["member_name"] = (
        merged["first_name"].fillna("").astype(str).str.strip()
        + " "
        + merged["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    cols = ["checked_in_at","member_id","member_name","primary_email","classification","method"]
    present = [c for c in cols if c in merged.columns]
    return merged[present].sort_values("checked_in_at", ascending=False)

# --- URL validation to avoid /pubhtml mistakes
CSV_URL_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/e/.+/pub\?output=csv(&gid=\d+)?$")

def validate_gsheets_csv_url(url: str) -> str:
    """
    Ensure we only accept the proper 'publish to web' CSV endpoint:
    .../pub?output=csv[&gid=...]
    If the user pasted /pubhtml, convert it.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Empty URL.")
    if url.endswith("/pubhtml"):
        url = url[:-8] + "?output=csv"
    if "output=csv" not in url:
        # Attempt to force CSV
        if "pub?" in url:
            url += "&output=csv"
        else:
            url += "?output=csv"
    if not CSV_URL_RE.match(url):
        raise ValueError(
            "URL must be a *published* Google Sheet CSV link ending in "
            "`/pub?output=csv` (optionally with `&gid=`). "
            "Open your sheet → File → Share → Publish to web → CSV."
        )
    return url

def fetch_public_csv(url: str) -> pd.DataFrame:
    """
    Read a published Google Sheet as CSV safely.
    """
    url = validate_gsheets_csv_url(url)
    # pandas can read this directly
    df = pd.read_csv(url, dtype=str).fillna("")
    # Guard against accidental single-column imports (classic /pubhtml scrape bug)
    if df.shape[1] == 1 and df.columns[0].lower() in {"classification","status","active","freshman"}:
        raise ValueError(
            "Only one column detected. This usually means the link is not the CSV export. "
            "Make sure it ends with `/pub?output=csv` and points to the correct tab (check gid)."
        )
    return df

def split_name(full_name: str) -> tuple[str,str]:
    full_name = (full_name or "").strip()
    if not full_name:
        return "", ""
    parts = full_name.split()
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]

def normalize_classification(val: str) -> str:
    v = (val or "").strip().lower()
    if v in {"freshman","sophomore","junior","senior","alumni"}:
        return v
    # Common typos/cases
    mapping = {
        "freshmen": "freshman",
        "sophmore": "sophomore",
        "jr": "junior",
        "sr": "senior",
    }
    return mapping.get(v, "freshman")

def normalize_member_row(row: dict) -> dict:
    """
    Accepts flexible incoming keys and returns a standard payload for upsert_member:
      first_name, last_name, classification, v_number, student_email, personal_email
    Handles cases:
      - 'Name' column present
      - Different header spellings
      - Email casing
    """
    # Flexible header options
    fn = row.get("first_name") or row.get("First name") or row.get("First Name") or ""
    ln = row.get("last_name")  or row.get("Last name")  or row.get("Last Name")  or ""
    name = row.get("name") or row.get("Name") or ""
    if (not fn) and name:
        fn, ln = split_name(name)

    classification = (
        row.get("classification")
        or row.get("Classification")
        or row.get("class")
        or row.get("Class")
        or ""
    )
    vnum = row.get("v_number") or row.get("V-number") or row.get("V Number") or row.get("VNumber") or ""
    stud_email = (
        row.get("student_email")
        or row.get("Student email")
        or row.get("student Email")
        or row.get("Student Email")
        or row.get("Email (student)")
        or row.get("Email")
        or ""
    )
    pers_email = (
        row.get("personal_email")
        or row.get("Personal email")
        or row.get("Personal Email")
        or row.get("Alt Email")
        or ""
    )

    stud_email = stud_email.strip().lower()
    pers_email = pers_email.strip().lower()
    classification = normalize_classification(classification)

    return {
        "first_name": str(fn).strip(),
        "last_name":  str(ln).strip(),
        "classification": classification,
        "v_number": str(vnum).strip(),
        "student_email": stud_email,
        "personal_email": pers_email,
    }

def member_search_ui():
    q = st.text_input("Search by email or name").strip()
    if st.button("Search"):
        hits = find_member(q)
        if not hits:
            st.warning("No match. Register first.")
        else:
            for m in hits:
                st.markdown(
                    f"- **{m.get('first_name','')} {m.get('last_name','')}** "
                    f"(<{m.get('primary_email','') or m.get('student_email','')}>), "
                    f"{m.get('classification','')}"
                )
                if st.button(f"Check in {m['id']}", key=f"ci{m['id']}"):
                    res = check_in(st.session_state.current_event_id, int(m["id"]))
                    st.success("Checked in!" if res else "Already checked in ✅")

# ---------------------------------------
# Sidebar Navigation
# ---------------------------------------
mode = st.sidebar.radio("Mode", ["Check-In", "Add Member", "Create Event", "Import Members"])

# ---------------------------------------
# Check-In
# ---------------------------------------
if mode == "Check-In":
    evs = list_events()
    if evs:
        ev_map = {f"{e['id']} — {e['name']} ({e['event_date']})": int(e["id"]) for e in evs}
        choice = st.selectbox("Select Event", list(ev_map.keys()))
        st.session_state.current_event_id = ev_map[choice]
        st.success(f"Event #{st.session_state.current_event_id} selected.")
        st.divider()

        # Search & check-in
        st.subheader("Find member")
        member_search_ui()

        # Attendance table + export
        st.subheader("Attendance")
        df_att = attendance_for_event_df(st.session_state.current_event_id)
        if df_att.empty:
            st.info("No one has checked in yet.")
        else:
            st.dataframe(df_att, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Attendance CSV",
                df_att.to_csv(index=False).encode("utf-8"),
                file_name=f"event_{st.session_state.current_event_id}_attendance.csv",
                mime="text/csv"
            )
    else:
        st.warning("No events yet. Create one in **Create Event**.")

# ---------------------------------------
# Add Member
# ---------------------------------------
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
        submitted = st.form_submit_button("Save")
        if submitted:
            row = upsert_member(fn, ln, cl, v, se, pe)
            st.success(f"Saved {row.get('first_name','')} {row.get('last_name','')}")

    st.divider()
    st.subheader("Members")
    df_m = members_df()
    if df_m.empty:
        st.info("No members yet.")
    else:
        st.dataframe(df_m, use_container_width=True)

# ---------------------------------------
# Create Event
# ---------------------------------------
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
    df_e = events_df()
    if df_e.empty:
        st.info("No events yet.")
    else:
        st.dataframe(df_e.sort_values("event_date", ascending=False), use_container_width=True)

# ---------------------------------------
# Import Members
# ---------------------------------------
elif mode == "Import Members":
    st.subheader("Import members from a published Google Sheet (CSV)")
    st.caption("Your link must end with **/pub?output=csv** (optionally **&gid=...**).")

    url = st.text_input("Public CSV URL")
    if st.button("Preview CSV"):
        try:
            df_raw = fetch_public_csv(url)
            st.success(f"Loaded {df_raw.shape[0]} rows · {df_raw.shape[1]} columns")
            st.dataframe(df_raw.head(25), use_container_width=True)
            st.session_state.import_preview = df_raw.to_dict(orient="records")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    if "import_preview" in st.session_state and st.session_state.import_preview:
        st.divider()
        st.subheader("Column Mapping")
        sample = st.session_state.import_preview[0]
        headers = list(sample.keys())

        # Try to auto-guess common headers
        def guess(*cands):
            for c in cands:
                if c in headers:
                    return c
            return ""

        col_name = st.selectbox("Full name column (if you have one)", [""] + headers, index=("Name" in headers) and headers.index("Name")+1 or 0)
        col_fn   = st.selectbox("First name column", [""] + headers, index=("first_name" in headers) and headers.index("first_name")+1 or 0)
        col_ln   = st.selectbox("Last name column",  [""] + headers, index=("last_name"  in headers) and headers.index("last_name")+1  or 0)
        col_cl   = st.selectbox("Classification column", [""] + headers, index=("Classification" in headers) and headers.index("Classification")+1 or 0)
        col_v    = st.selectbox("V-number column", [""] + headers, index=("V-number" in headers) and headers.index("V-number")+1 or 0)
        col_se   = st.selectbox("Student email column", [""] + headers, index=("Email" in headers) and headers.index("Email")+1 or 0)
        col_pe   = st.selectbox("Personal email column", [""] + headers, index=("Personal Email" in headers) and headers.index("Personal Email")+1 or 0)

        # Build a mapped list of dict rows
        mapped_rows = []
        for r in st.session_state.import_preview:
            # create a minimal flexible row for the normalizer
            fr = {}
            if col_name: fr["Name"] = r.get(col_name, "")
            if col_fn:   fr["first_name"] = r.get(col_fn, "")
            if col_ln:   fr["last_name"]  = r.get(col_ln, "")
            if col_cl:   fr["Classification"] = r.get(col_cl, "")
            if col_v:    fr["V-number"]   = r.get(col_v, "")
            if col_se:   fr["Email"]      = r.get(col_se, "")
            if col_pe:   fr["Personal Email"] = r.get(col_pe, "")

            mapped_rows.append(normalize_member_row(fr))

        st.subheader("Normalized Preview")
        df_norm = pd.DataFrame(mapped_rows)
        st.dataframe(df_norm.head(25), use_container_width=True)

        if st.button("Import"):
            try:
                added, updated = 0, 0
                before_set = set()
                # Cache current IDs to judge added vs updated (best-effort if your backend doesn't return flags)
                try:
                    for m in list_members():
                        before_set.add(m.get("id"))
                except Exception:
                    pass

                for row in mapped_rows:
                    upsert_member(
                        row["first_name"], row["last_name"], row["classification"],
                        row["v_number"], row["student_email"], row["personal_email"]
                    )

                after_set = set()
                try:
                    for m in list_members():
                        after_set.add(m.get("id"))
                except Exception:
                    pass

                # Heuristic on added/updated
                if after_set and before_set:
                    added = len(after_set - before_set)
                    updated = len(mapped_rows) - added
                else:
                    # Fallback if we can't diff
                    updated = 0
                    added = len(mapped_rows)

                st.success(f"Imported {added} new, {updated} updated")
            except Exception as e:
                st.error(f"Import failed: {e}")







