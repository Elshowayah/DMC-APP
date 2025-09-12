from ..utils import read_csv, append_csv, next_id

EVENTS_FILE = "events.csv"

def list_events():
    return read_csv(EVENTS_FILE)

def create_event(name: str, date: str, location: str = "") -> dict:
    rows = list_events()
    new_id = next_id(rows)
    row = {
        "id": str(new_id),
        "name": name.strip(),
        "event_date": date.strip(),
        "location": location.strip(),
        "active": "true",
    }
    append_csv(EVENTS_FILE, row)
    return row
