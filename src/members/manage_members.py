from typing import Optional
from ..utils import read_csv, write_csv, append_csv, next_id, norm_email, now_iso

MEMBERS_FILE = "members.csv"

def list_members():
    return read_csv(MEMBERS_FILE)

def find_member(query: str) -> list[dict]:
    q = (query or "").strip().lower()
    if not q: return []
    rows = list_members()

    # exact email match first
    exact = [r for r in rows if norm_email(r.get("primary_email")) == q]
    if exact: return exact

    # fuzzy match
    hits = []
    for r in rows:
        name = f"{r.get('first_name','')} {r.get('last_name','')}".lower()
        emails = " ".join([
            norm_email(r.get("primary_email")),
            norm_email(r.get("student_email")),
            norm_email(r.get("personal_email")),
        ])
        if q in name or q in emails:
            hits.append(r)
    return hits

def get_member_by_id(member_id: int) -> Optional[dict]:
    for r in list_members():
        if int(r["id"]) == int(member_id): return r
    return None

def _save(rows): write_csv(MEMBERS_FILE, rows)

def _find_index(rows, primary_email, v_number):
    for i,r in enumerate(rows):
        if primary_email and norm_email(r.get("primary_email")) == norm_email(primary_email):
            return i
        if v_number and (r.get("v_number") or "") == (v_number or ""):
            return i
    return None

def upsert_member(fn, ln, cl, vnum: Optional[str], stud_email: Optional[str], pers_email: Optional[str]) -> dict:
    rows = list_members()
    primary = norm_email(stud_email or pers_email or "")
    idx = _find_index(rows, primary, (vnum or ""))

    payload = {
        "first_name": fn.strip(),
        "last_name": ln.strip(),
        "classification": cl.strip().lower(),
        "v_number": (vnum or "").strip(),
        "student_email": norm_email(stud_email),
        "personal_email": norm_email(pers_email),
        "primary_email": primary,
        "status": "active",
    }

    if idx is None:
        new_id = next_id(rows)
        row = {"id": str(new_id), "last_verified_at": "", **payload}
        append_csv(MEMBERS_FILE, row)
        return row
    else:
        existing = rows[idx]
        existing.update(payload)
        existing["last_verified_at"] = now_iso()
        rows[idx] = existing
        _save(rows)
        return existing

def update_classification(member_id: int, new_class: str) -> bool:
    rows = list_members()
    for r in rows:
        if int(r["id"]) == int(member_id):
            r["classification"] = new_class.lower()
            r["last_verified_at"] = now_iso()
            _save(rows)
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
            _save(rows)
            return True
    return False



