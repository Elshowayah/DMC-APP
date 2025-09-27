after member types the name doesnt clear so get it to clear.


# Optional: quick cache with short TTL and a manual refresh
@st.cache_data(ttl=5, show_spinner=False)
def _latest():
    return latest_checkins(500)

if st.button("Refresh data"):
    st.cache_data.clear()

st.dataframe(_latest())