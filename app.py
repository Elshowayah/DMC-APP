import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta

from src.members.manage_members import list_members, find_member, upsert_member, update_classification, graduate_member
from src.events.manage_events import list_events, create_event
from src.attendance.manage_attendance import check_in, list_attendance
from src.utils.sheets import fetch_public_csv

st.set_page_config(page_title="Org Attendance", page_icon="✅", layout="centered")
st.title("✅ Org Event Check-In")

# Helpers
def members_df(): return pd.DataFrame(list_members())
def events_df(): return pd.DataFrame(list_events())
def attendance_df(): return pd.DataFrame(list_attendance())

def attendance_for_event_df(event_id: int):
    a = attendance_df()
    if a.empty: return a
    a = a[a["event_id"].astype(int) == int(event_id)]
    m = members_df()
    if m.empty: return a
    merged = a.merge(m, left_on="member_id", right_on="id", how="left")
    merged["member_name"] = merged["first_name"].fillna("") + " " + merged["last_name"].fillna("")
    return merged[["checked_in_at","member_id","member_name","primary_email","classification","method"]]

# Sidebar navigation
mode = st.sidebar.radio("Mode", ["Check-In", "Add Member", "Create Event"])

# Event picker for check-in
if mode == "Check-In":
    evs = list_events()
    if evs:
        ev_map = {f"{e['id']} - {e['name']} ({e['event_date']})": int(e["id"]) for e in evs}
        current_event_id = ev_map[st.selectbox("Select Event", list(ev_map.keys()))]
    else:
        st.warning("No events yet. Add in Create Event.")
        current_event_id = None
else:
    current_event_id = None

# =========================
# CHECK-IN
# =========================
if mode == "Check-In" and current_event_id:
    q = st.text_input("Search by email or name").strip()
    if st.button("Search"):
        hits = find_member(q)
        if not hits: st.warning("No match. Register first.")
        else:
            for m in hits:
                st.write(f"{m['first_name']} {m['last_name']} ({m['primary_email']}) – {m['classification']}")
                if st.button(f"Check in {m['id']}", key=f"ci{m['id']}"):
                    res = check_in(current_event_id, int(m["id"]))
                    st.success("Checked in!" if res else "Already checked in ✅")

    st.subheader("Attendance")
    df = attendance_for_event_df(current_event_id)
    if not df.empty:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"event_{current_event_id}_attendance.csv", mime="text/csv")

# =========================
# ADD MEMBER
# =========================
elif mode == "Add Member":
    with st.form("add"):
        fn = st.text_input("First name")
        ln = st.text_input("Last name")
        cl = st.selectbox("Classification", ["freshman","sophomore","junior","senior","alumni"])
        v  = st.text_input("V-number")
        se = st.text_input("Student email")
        pe = st.text_input("Personal email")
        if st.form_submit_button("Save"):
            row = upsert_member(fn, ln, cl, v, se, pe)
            st.success(f"Saved {row['first_name']} {row['last_name']}")

# =========================
# ADMIN
# =========================
elif mode == "Create Event":
    tab_events, tab_members, tab_import = st.tabs(["Events","Members","Import"])

    with tab_events:
        with st.form("new_event"):
            name = st.text_input("Event name")
            dt = st.text_input("Event date", value=str(date.today()))
            loc = st.text_input("Location")
            if st.form_submit_button("Create"):
                ev = create_event(name, dt, loc)
                st.success(f"Created event {ev['id']}: {ev['name']}")
        st.dataframe(events_df())

    with tab_members:
        st.dataframe(members_df())

    with tab_import:
        url = st.text_input("Public CSV URL (must end with output=csv)")
        if st.button("Import"):
            try:
                df = fetch_public_csv(url)
                added, updated = 0,0
                for _,r in df.iterrows():
                    before = len(list_members())
                    upsert_member(
                        r.get("first_name",""), r.get("last_name",""),
                        r.get("classification","freshman"),
                        r.get("v_number"), r.get("student_email"), r.get("personal_email")
                    )
                    after = len(list_members())
                    if after > before: added+=1
                    else: updated+=1
                st.success(f"Imported {added} new, {updated} updated")
            except Exception as e:
                st.error(f"Import failed: {e}")






