from typing import Optional
from ..utils import read_csv, write_csv, append_csv, next_id, norm_email, now_iso

MEMBERS_FILE = "members.csv"

def list_members():
    return read_csv(MEMBERS_FILE)

def get_member_by_id(member_id: int) -> Optional[dict]:
    for r in list_members():
        if int(r["id"]) == int(member_id):
            return r
    return None

def _save_rows(rows: list[dict]) -> None:
    write_csv(MEMBERS_FILE, rows)

def _find_index(rows: list[dict], primary_email: str, v_number: str) -> Optional[int]:
    for i, r in enumerate(rows):
        if primary_email and norm_email(r.get("primary_email")) == norm_email(primary_email):
            return i
        if v_number and (r.get("v_number") or "") == (v_number or ""):
            return i
    return None

def upsert_member(
    first_name: str, last_name: str, classification: str,
    v_number: Optional[str], student_email: Optional[str],
    personal_email: Optional[str]
) -> dict:
    rows = list_members()
    primary_email = norm_email(student_email or personal_email or "")
    idx = _find_index(rows, primary_email, (v_number or ""))

    payload = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "classification": classification.strip().lower(),
        "v_number": (v_number or "").strip(),
        "student_email": norm_email(student_email),
        "personal_email": norm_email(personal_email),
        "primary_email": primary_email,
        "status": "active",
    }

    if idx is None:
        # insert
        new_id = next_id(rows)
        row = {"id": str(new_id), "last_verified_at": "", **payload}
        append_csv(MEMBERS_FILE, row)
        return row
    else:
        # update
        existing = rows[idx]
        existing.update(payload)
        existing["last_verified_at"] = now_iso()
        rows[idx] = existing
        _save_rows(rows)
        return existing

def update_classification(member_id: int, new_class: str) -> bool:
    rows = list_members()
    for r in rows:
        if int(r["id"]) == int(member_id):
            r["classification"] = new_class.strip().lower()
            r["last_verified_at"] = now_iso()
            _save_rows(rows)
            return True
    return False

def graduate_member(member_id: int, personal_email: str) -> bool:
    rows = list_members()
    for r in rows:
        if int(r["id"]) == int(member_id):
            r["classification"] = "alumni"
            r["v_number"] = ""
            r["personal_email"] = norm_email(personal_email)
            r["primary_email"] = r["personal_email"]
            r["last_verified_at"] = now_iso()
            _save_rows(rows)
            return True
    return False


