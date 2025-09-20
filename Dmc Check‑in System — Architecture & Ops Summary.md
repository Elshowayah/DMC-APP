# DMC Check‑In System — Architecture & Ops Summary

## 1) Purpose

Replace CSV files with **Postgres** as the single source of truth for members, events, and attendance. Keep **S3** for exports and backups. Serve a public **check‑in** app and a protected **admin** app.

---

## 2) Components

* **Postgres DB** (local, RDS, Neon, or Supabase)

  * Tables: `members`, `events`, `attendance`
  * Optional: materialized view `databrowser` for fast dashboarding
* **Apps**

  * `checkin.py` (public kiosk): find/create member and insert attendance
  * `admin.py` (restricted): create/update events & members; view data browser
* **S3** (optional): CSV exports + `pg_dump` backups

---

## 3) Data Flow (live)

1. **Create/Update Member** → `upsert_member(...)` → writes to `members`.
2. **Create Event** → `create_event(...)` → writes to `events`.
3. **Check‑In** → `check_in(event_id, member_id, method)` → writes to `attendance`.
4. **Data Browser**

   * **Option A (recommended, always fresh):** query a JOIN via `latest_checkins(limit)`
   * **Option B (faster on large data):** query `databrowser` materialized view and refresh on writes

> CSVs are **not** used for live operations; only for initial import or exports.

---

## 4) Minimal Schema (excerpt)

```sql
CREATE TABLE members (
  id TEXT PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name  TEXT NOT NULL,
  classification TEXT,
  major TEXT,
  v_number TEXT UNIQUE,
  student_email TEXT UNIQUE,
  personal_email TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE events (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  event_date DATE NOT NULL,
  location TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE attendance (
  event_id  TEXT REFERENCES events(id) ON DELETE CASCADE,
  member_id TEXT REFERENCES members(id) ON DELETE CASCADE,
  checked_in_at TIMESTAMPTZ DEFAULT NOW(),
  method TEXT,
  PRIMARY KEY (event_id, member_id, checked_in_at)
);

-- Optional: one check‑in per member per event
-- CREATE UNIQUE INDEX uniq_once_per_event ON attendance(event_id, member_id);

-- Helpful index
CREATE INDEX IF NOT EXISTS idx_members_v ON members(v_number);
```

### Optional Materialized View

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS databrowser AS
SELECT a.event_id, e.name AS event_name, e.event_date,
       a.member_id, m.first_name, m.last_name, m.classification, m.major,
       a.checked_in_at, a.method
FROM attendance a
JOIN members m ON m.id = a.member_id
JOIN events  e ON e.id = a.event_id;

CREATE OR REPLACE FUNCTION refresh_databrowser() RETURNS VOID AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY databrowser;
EXCEPTION WHEN undefined_table THEN
  REFRESH MATERIALIZED VIEW databrowser;
END; $$ LANGUAGE plpgsql;
```

---

## 5) Core App Functions (db helpers)

* `list_events(limit)` → list recent events
* `create_event(ev)` → insert/update event by `id`
* `upsert_member(m)` → insert/update member by `id`
* `find_member_by_v(v_number)` → look up member by V‑number
* `check_in(event_id, member_id, method=None)` → insert into `attendance`
* `latest_checkins(limit)` → joined, recent check‑ins for the Data Browser

**Check‑in flow (kiosk):**

1. read V‑number; `find_member_by_v()`
2. if missing → `upsert_member()` minimal record
3. `check_in()`; optionally `SELECT refresh_databrowser()`

**Admin flow:**

* event/member management via `create_event()` / `upsert_member()`
* data browser via `latest_checkins()` **or** `SELECT * FROM databrowser ...`

---

## 6) Environment & Config

* **DATABASE\_URL** (example local):
  `postgresql+psycopg://dmc:dmc_password@localhost:5432/dmc_db`
* Store in `.env.local` (local) or **Streamlit Secrets** (Cloud).
* `.streamlit/config.toml` (Cloud minimal):

```toml
[server]
headless = true
[browser]
gatherUsageStats = false
[theme]
base = "light"
```

**requirements.txt**

```txt
streamlit>=1.37,<2
sqlalchemy>=2.0,<3
psycopg[binary]>=3.2,<4
pandas>=2.2,<3
python-dotenv>=1.0,<2
boto3>=1.34,<2  # if using S3 exports/backups
```

---

## 7) Imports/Exports (optional)

* **Initial import from CSVs:** load `members` → `events` → `attendance` (use a staging table + `ON CONFLICT` to avoid FK/duplicate errors).
* **Export to S3:** generate CSV from a JOIN and `boto3.upload_file(...)` to `s3://<bucket>/reports/…`
* **DB backup:** `pg_dump -Fc dmc_db.dump` → upload to `s3://<bucket>/db-backups/…`

---

## 8) Deployment Options

* **Easiest:** Streamlit Community Cloud + Neon/Supabase (paste DB URL with `?sslmode=require` in Secrets). Add a website button linking to the app URL.
* **Self‑host:** EC2 + Nginx reverse proxy (`/checkin`, `/admin`) + RDS Postgres. Protect `/admin` (basic auth or in‑app password).

---

## 9) Common Pitfalls & Fixes

* **`DATABASE_URL` missing/invalid** → ensure `.env.local` or Secrets is set; URL must start with `postgresql+psycopg://`.
* **FK errors on import** → load parents first; or stage CSV then backfill missing members/events.
* **Duplicate key on attendance** → use staging + `INSERT … ON CONFLICT DO NOTHING`.
* **Data Browser not updating** → switch to live JOIN (`latest_checkins`) or call `refresh_databrowser()` after writes.
* **Mixed Python envs** → use **either** venv **or** conda; not both.

---

## 10) Quick Health Checks

```bash
# DB up & reachable
psql "$DATABASE_URL" -c "SELECT 1;"

# Tables exist
psql "$DATABASE_URL" -c "\dt"

# Recent check‑ins
psql "$DATABASE_URL" -c "SELECT * FROM attendance ORDER BY checked_in_at DESC LIMIT 5;"
```

**End of summary.**
