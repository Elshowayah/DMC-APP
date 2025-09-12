from ..utils import read_csv, append_csv, next_id, now_iso
from ..members.manage_members import get_member_by_id
from ..events.manage_events import list_events

ATT_FILE = "attendance.csv"

def list_attendance():
    return read_csv(ATT_FILE)

def already_checked_in(event_id, member_id):
    for r in list_attendance():
        if int(r["event_id"]) == int(event_id) and int(r["member_id"]) == int(member_id):
            return True
    return False

def check_in(event_id: int, member_id: int, method="search", notes="") -> dict:
    if not get_member_by_id(member_id): return None
    if not any(int(e["id"]) == int(event_id) for e in list_events()): return None
    if already_checked_in(event_id, member_id): return None

    row = {
        "id": str(next_id(list_attendance())),
        "event_id": str(event_id),
        "member_id": str(member_id),
        "checked_in_at": now_iso(),
        "method": method,
        "notes": notes,
    }
    append_csv(ATT_FILE, row)
    return row
