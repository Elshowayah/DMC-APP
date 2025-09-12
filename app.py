import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta

from src.members.manage_members import (
    list_members, find_member, upsert_member,
    update_classification, graduate_member, get_member_by_id
)
from src.events.manage_events import list_events, create_event
from src.attendance.manage_attendance import check_in, list_attendance
from src.utils.sheets import fetch_public_csv

# --- Streamlit setup ---
st.set_page_config(page_title="Org Attendance", page_icon="✅", layout="centered")
st.title("✅ Org Event Check-In")

# --- helpers ---
def members_df(): return pd.DataFrame(list_members())
def events_df(): return pd.DataFrame(list_events())
def attendance_df(): return pd.DataFrame(list_attendance())

def attendance_for_event_df(event_id: int) -> pd.DataFrame:
    a = attendance_df()
    if a.empty: return a
    a = a[a["event_id"].astype(int) == int(event_id)].copy()
    m = members_df()
    if m.empty: return a
    merged = a.merge(
        m[["id","first_name","last_name","primary_email","classification"]],
        left_on="member_id", right_on="id", how="left"
    )
    merged["member_name"] = (merged["first_name"].fillna("") + " " + merged["last_name"].fillna("")).str.strip()
    return merged[["checked_in_at","member_id","member_name","primary_email","classification","method","notes"]]

# --- sidebar nav ---
mode = st.sidebar.radio("Mode", ["Check-In", "Add Member", "Admin"])

# --- event picker ---
if mode == "Check-In":
    evs = list_events()
    if evs:
        ev_map = {f"{e['id']} - {e['name']} ({e['event_date']})": int(e["id"]) for e in evs}
        event_choice = st.selectbox("Select Event", list(ev_map.keys()))
        current_event_id = ev_map[event_choice]
    else:
        st.warning("No events yet. Create one in Admin.")
        current_event_id = None
else:
    current_event_id = None

# =========================
# CHECK-IN
# =========================
if mode == "Check-In" and current_event_id:
    st.subheader("Search by Email or Name")
    q = st.text_input("Type your email or name").strip()
    if st.button("Search"):
        hits = find_member(q)
        if not hits:
            st.warning("No match. Please register first.")
        elif len(hits) == 1:
            m = hits[0]
            st.success(f"Found: {m['first_name']} {m['last_name']} ({m['primary_email']})")

            # annual verify
            needs_verify = not m.get("last_verified_at") or (
                (datetime.utcnow() - datetime.fromisoformat(m["last_verified_at"])) > timedelta(days=365)
                if m.get("last_verified_at") else True
            )
            if needs_verify:
                with st.expander("Verify your info"):
                    new_email = st.text_input("Preferred email", value=m.get("primary_email",""))
                    new_class = st.selectbox("Classification", ["freshman","sophomore","junior","senior","alumni"])
                    if st.button("Save verification"):
                        update_classification(int(m["id"]), new_class)
                        upsert_member(m["first_name"], m["last_name"], new_class,
                                      m.get("v_number"), m.get("student_email"), new_email)
                        st.success("Info updated.")

            if st.button("Check Me In"):
                res = check_in(current_event_id, int(m["id"]))
                st.success("Checked in!" if res else "Already checked in ✅")

        else:
            st.info("Multiple matches found.")
            for m in hits:
                if st.button(f"Check in {m['first_name']} {m['last_name']} ({m['primary_email']})"):
                    res = check_in(current_event_id, int(m["id"]))
                    st.success("Checked in!" if res else "Already checked in ✅")

    st.divider()
    st.subheader("Event Attendance")
    ev_att = attendance_for_event_df(current_event_id)
    if ev_att.empty:
        st.caption("Nobody checked in yet.")
    else:
        st.dataframe(ev_att, use_container_width=True)
        st.download_button(
            "Download CSV",
            ev_att.to_csv(index=False).encode("utf-8"),
            file_name=f"event_{current_event_id}_attendance.csv",
            mime="text/csv"
        )

# =========================
# ADD MEMBER
# =========================
elif mode == "Add Member":
    st.subheader("Register")
    with st.form("new_member"):
        fn = st.text_input("First name")
        ln = st.text_input("Last name")
        cl = st.selectbox("Classification", ["freshman","sophomore","junior","senior","alumni"])
        v  = st.text_input("V-number (optional)")
        se = st.text_input("Student email (optional)")
        pe = st.text_input("Personal email (optional)")
        submitted = st.form_submit_button("Save")
    if submitted:
        row = upsert_member(fn, ln, cl, v, se, pe)
        st.success(f"Saved: {row['first_name']} {row['last_name']}")

# =========================
# ADMIN
# =========================
elif mode == "Admin":
    tab_events, tab_members, tab_import = st.tabs(["Events", "Members", "Import"])

    with tab_events:
        st.subheader("Create Event")
        with st.form("create_event"):
            name = st.text_input("Event name")
            dt = st.text_input("Event date", value=str(date.today()))
            loc = st.text_input("Location")
            ok = st.form_submit_button("Create")
        if ok:
            ev = create_event(name, dt, loc)
            st.success(f"Created event {ev['id']}: {ev['name']}")
        st.dataframe(events_df())

    with tab_members:
        st.subheader("Members")
        st.dataframe(members_df())

    with tab_import:
        st.subheader("Import Members (Public CSV)")
        url = st.text_input("Paste Google Sheet CSV URL")
        if st.button("Import"):
            try:
                df = fetch_public_csv(url)
                added, updated = 0, 0
                for _, r in df.iterrows():
                    before = len(list_members())
                    upsert_member(
                        r.get("first_name",""), r.get("last_name",""),
                        r.get("classification","freshman"),
                        r.get("v_number"), r.get("student_email"), r.get("personal_email")
                    )
                    after = len(list_members())
                    if after > before: added += 1
                    else: updated += 1
                st.success(f"Imported: {added} new, {updated} updated")
            except Exception as e:
                st.error(f"Import failed: {e}")





