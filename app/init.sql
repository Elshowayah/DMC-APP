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
  had_internship     BOOLEAN,  -- nullable on purpose; starts blank
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- hoodie_size stored as normalized lowercase code with default
  hoodie_size        TEXT NOT NULL DEFAULT 'medium'
);

-- Add missing columns safely (for older schemas)
ALTER TABLE members
  ADD COLUMN IF NOT EXISTS linkedin_yes       BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS updated_resume_yes BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS had_internship     BOOLEAN,
  ADD COLUMN IF NOT EXISTS hoodie_size        TEXT,  -- add as nullable first for old rows
  ADD COLUMN IF NOT EXISTS created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Normalize / backfill hoodie_size if column exists
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='members' AND column_name='hoodie_size'
  ) THEN
    -- Normalize common aliases to lowercase codes
    UPDATE members
       SET hoodie_size = LOWER(REPLACE(hoodie_size,' ',''))
     WHERE hoodie_size IS NOT NULL;

    UPDATE members SET hoodie_size = 'xl'  WHERE hoodie_size IN ('xlarge','extra large','xl ');
    UPDATE members SET hoodie_size = '2xl' WHERE hoodie_size IN ('xxl','2x','doublexl');

    -- Default unknowns / NULLs to 'medium'
    UPDATE members
       SET hoodie_size = 'medium'
     WHERE hoodie_size IS NULL
        OR hoodie_size NOT IN ('small','medium','large','xl','2xl');

    -- Set default + NOT NULL
    ALTER TABLE members
      ALTER COLUMN hoodie_size SET DEFAULT 'medium',
      ALTER COLUMN hoodie_size SET NOT NULL;

    -- Add CHECK constraint if it doesn't exist yet
    IF NOT EXISTS (
      SELECT 1 FROM pg_constraint
      WHERE conname = 'members_hoodie_size_chk'
        AND conrelid = 'members'::regclass
    ) THEN
      ALTER TABLE members
        ADD CONSTRAINT members_hoodie_size_chk
        CHECK (hoodie_size IN ('small','medium','large','xl','2xl'));
    END IF;
  END IF;
END
$$;

-- Updater function for updated_at
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

-- Trigger (create only if missing)
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
-- Optional if you filter by internship often:
-- CREATE INDEX IF NOT EXISTS idx_members_had_internship ON members (had_internship);

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
