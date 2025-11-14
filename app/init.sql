-- ===========================================
-- DMC App - Database Bootstrap / Migration
-- Safe to run multiple times (idempotent)
-- ===========================================

BEGIN;

-- ---------
-- MEMBERS
-- ---------
CREATE TABLE IF NOT EXISTS members (
  id                 TEXT PRIMARY KEY,
  first_name         TEXT NOT NULL,
  last_name          TEXT NOT NULL,
  classification     TEXT,
  major              TEXT,
  student_email      TEXT UNIQUE,
  linkedin_yes       BOOLEAN NOT NULL DEFAULT FALSE,
  updated_resume_yes BOOLEAN NOT NULL DEFAULT FALSE,
  had_internship     BOOLEAN,                 -- nullable on purpose; starts blank
  personal_email     TEXT,
  v_number           TEXT,
  hoodie_size        TEXT,                    -- nullable: starts as NULL
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Bring older schemas up to date
ALTER TABLE members
  ADD COLUMN IF NOT EXISTS linkedin_yes       BOOLEAN     NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS updated_resume_yes BOOLEAN     NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS had_internship     BOOLEAN,
  ADD COLUMN IF NOT EXISTS personal_email     TEXT,
  ADD COLUMN IF NOT EXISTS v_number           TEXT,
  ADD COLUMN IF NOT EXISTS hoodie_size        TEXT,
  ADD COLUMN IF NOT EXISTS created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Normalize / constrain hoodie_size (nullable, no default)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name   = 'members'
      AND column_name  = 'hoodie_size'
  ) THEN
    -- Normalize existing values (if any)
    UPDATE members
       SET hoodie_size = LOWER(REPLACE(hoodie_size, ' ', ''))
     WHERE hoodie_size IS NOT NULL;

    UPDATE members
       SET hoodie_size = 'xl'
     WHERE hoodie_size IN ('xlarge', 'extra large', 'xl ');

    UPDATE members
       SET hoodie_size = '2xl'
     WHERE hoodie_size IN ('xxl', '2x', 'doublexl');

    -- Any weird values become NULL; we want "no answer" not "medium"
    UPDATE members
       SET hoodie_size = NULL
     WHERE hoodie_size IS NOT NULL
       AND hoodie_size NOT IN ('small','medium','large','xl','2xl');

    -- Drop old constraint if shape changed
    IF EXISTS (
      SELECT 1
      FROM pg_constraint
      WHERE conname = 'members_hoodie_size_chk'
        AND conrelid = 'members'::regclass
    ) THEN
      ALTER TABLE members DROP CONSTRAINT members_hoodie_size_chk;
    END IF;

    -- New constraint: allow NULL or one of the valid codes
    ALTER TABLE members
      ADD CONSTRAINT members_hoodie_size_chk
      CHECK (
        hoodie_size IS NULL
        OR hoodie_size IN ('small','medium','large','xl','2xl')
      );
  END IF;
END
$$;

-- updated_at trigger function
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

-- updated_at trigger hook
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

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_members_last_first ON members (last_name, first_name);
CREATE INDEX IF NOT EXISTS idx_members_email      ON members (student_email);

-- ---------
-- EVENTS
-- ---------
CREATE TABLE IF NOT EXISTS events (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  event_date DATE NOT NULL,
  location   TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_date ON events (event_date DESC);
CREATE INDEX IF NOT EXISTS idx_events_name ON events (name);

-- -----------
-- ATTENDANCE
-- -----------
CREATE TABLE IF NOT EXISTS attendance (
  event_id      TEXT NOT NULL REFERENCES events(id)  ON DELETE CASCADE,
  member_id     TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
  checked_in_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  method        TEXT,
  PRIMARY KEY (event_id, member_id, checked_in_at)
);

CREATE INDEX IF NOT EXISTS idx_attendance_event_time  ON attendance (event_id, checked_in_at DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_member_time ON attendance (member_id, checked_in_at DESC);

COMMIT;
