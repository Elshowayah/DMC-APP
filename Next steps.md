-the two apps are pointing at different databases (secrets mismatch),
-the check-in app isn’t creating the member before inserting attendance,
-the Admin page is still reading CSV or a materialized view that isn’t refreshed, or
caching is hiding fresh data.
Here’s a copy-paste fix pack. Do them in order.
1) Prove both apps hit the same DB
Add this small banner near the top of both admin.py and checkin.py:
import streamlit as st
from sqlalchemy.engine.url import make_url
from db import ENGINE

try:
    u = make_url(ENGINE.url)
    st.caption(f"DB: {u.host}/{u.database}")   # no secrets, just host/db
except Exception as e:
    st.caption(f"DB not ready: {e}")
👉 In Streamlit Cloud, open each app. The caption should be identical.
If not: in each app’s Settings → Secrets, make DATABASE_URL exactly the same (include ?sslmode=require if using Neon/Supabase).
2) Make sure the check-in flow creates the member if missing
In checkin.py, the submit handler should look like this (drop-in):
from db import find_member_by_v, upsert_member, check_in, list_events
import streamlit as st

events = list_events(500)
event_map = {f"{e['name']} ({e['event_date']})": e["id"] for e in events}
event_label = st.selectbox("Event", list(event_map.keys()))
event_id = event_map[event_label]

v = st.text_input("V-number (e.g., V00123456)").strip()
first = st.text_input("First name").strip()
last = st.text_input("Last name").strip()
classification = st.selectbox("Classification", ["freshman","sophomore","junior","senior","alumni"], index=4)
major = st.text_input("Major (optional)").strip()

if st.button("Check In"):
    m = find_member_by_v(v)
    if not m:
        # register a minimal member
        upsert_member({
            "id": v,  # keep it consistent: use V-number as id (or your own scheme)
            "first_name": first or "Unknown",
            "last_name":  last or "Unknown",
            "classification": classification,
            "major": major or None,
            "v_number": v,
            "student_email": None,
            "personal_email": None,
            "created_at": None
        })
        member_id = v
    else:
        member_id = m["id"]

    check_in(event_id=event_id, member_id=member_id, method="kiosk")
    st.success("Checked in!")
If your members.id is not the V-number, use whatever ID you store consistently (but ensure find_member_by_v returns the right id).
3) Make the Admin “Data Browser” read live DB
Option A (simplest, recommended)
Replace the Admin Data Browser code with:
from db import latest_checkins
import streamlit as st

# Optional: quick cache with short TTL and a manual refresh
@st.cache_data(ttl=5, show_spinner=False)
def _latest():
    return latest_checkins(500)

if st.button("Refresh data"):
    st.cache_data.clear()

st.dataframe(_latest())