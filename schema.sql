CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    forecast_json TEXT NOT NULL,
    profile_name TEXT
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    forecast_date TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    observation_time TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('ИИ', 'Правила')),
    options_json TEXT
);

CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    answers_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_requested_at
    ON requests(requested_at DESC, id DESC);
