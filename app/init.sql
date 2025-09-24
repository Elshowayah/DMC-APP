CREATE TABLE IF NOT EXISTS members (
  id TEXT PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name  TEXT NOT NULL,
  classification TEXT,
  major TEXT,
  student_email TEXT UNIQUE,
  linkedin_yes BOOLEAN,
  updated_resume_yes BOOLEAN
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  event_date DATE NOT NULL,
  location TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendance (
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  member_id TEXT NOT NULL REFERENCES members(id) ON DELETE CASCADE,
  checked_in_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  method TEXT,
  PRIMARY KEY (event_id, member_id, checked_in_at)
);
