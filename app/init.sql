-- ============================
-- DMC App - init.sql (updated)
-- ============================

-- Members
CREATE TABLE IF NOT EXISTS members (
  id                 TEXT PRIMARY KEY,
  first_name         TEXT NOT NULL,
  last_name          TEXT NOT NULL,
  classification     TEXT,
  major              TEXT,
  student_email      TEXT UNIQUE,
  linkedin_yes       BOOLEAN DEFAULT FALSE,
  updated_resume_yes BOOLEAN DEFAULT FALSE,
  had_internship     BOOLEAN,                      -- NEW: nullable, starts blank
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- In case the table existed before this change, add missing column safely
ALTER TABLE members
  ADD COLUMN IF NOT EXISTS had_internship BOOLEAN;

-- Auto-update updated_at on members
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc WHERE proname = 'tg_members_set_updated_at'
  ) THEN
    CREATE OR REPLACE FUNCTION tg_members_set_updated_at()
    RETURNS TRIGGER AS $fn$
    BEGIN
      NEW.updated_at := NOW();
      RETURN NEW;
    END
    $fn$ LANGUAGE plpgsql;
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_members_set_updated_at'
  ) THEN
    CREATE TRIGGER trg_members_set_updated_at
    BEFORE UPDATE ON members
    FOR EACH ROW
    EXECUTE FUNCTION tg_members_set_updated_at();
  END IF;
END
$$;

-- Helpful member indexes
CREATE INDEX IF NOT EXISTS idx_members_last_first
  ON members (last_name, first_name);
CREATE INDEX IF NOT EXISTS idx_members_email
  ON members (student_email);

-- (Optional) If you frequently filter/export by internship status:
-- CREATE INDEX IF NOT EXISTS idx_members_had_internship ON members (had_internship);

-- Events
CREATE TABLE IF NOT EXISTS events (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  event_date DATE NOT NULL,
  location   TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful event indexes
CREATE INDEX IF NOT EXISTS idx_events_date
  ON events (event_date DESC);
CREATE INDEX IF NOT EXISTS idx_events_name
  ON events (name);

-- Attendance
CREATE TABLE IF NOT EXISTS attendance (
  event_id      TEXT NOT NULL REFERENCES events(id)  ON DELETE CASCADE,
  member_id     TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
  checked_in_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  method        TEXT,
  PRIMARY KEY (event_id, member_id, checked_in_at)
);

-- Helpful attendance indexes
CREATE INDEX IF NOT EXISTS idx_attendance_event_time
  ON attendance (event_id, checked_in_at DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_member_time
  ON attendance (member_id, checked_in_at DESC);
