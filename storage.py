"""SQLite persistence for profiles, forecasts and recommendations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from profiles import restore_answers, validate_answers
from recommendations import OutfitOption
from weather import ForecastDay


def _parse_json(raw: str | None) -> object:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(schema)

    # Databases from the first app version lack these two columns.
    legacy_columns = (("requests", "profile_name"), ("recommendations", "options_json"))
    for table, column in legacy_columns:
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    db.commit()
    db.row_factory = sqlite3.Row
    return db


def list_profiles(db: sqlite3.Connection) -> list[dict]:
    rows = db.execute("SELECT id, name FROM profiles ORDER BY name COLLATE NOCASE")
    return [dict(row) for row in rows]


def get_profile(db: sqlite3.Connection, profile_id: int) -> dict | None:
    row = db.execute(
        "SELECT id, name, answers_json FROM profiles WHERE id = ?",
        (profile_id,),
    ).fetchone()
    if row is None:
        return None
    answers, needs_quiz = restore_answers(_parse_json(row["answers_json"]))
    return {
        "id": row["id"],
        "name": row["name"],
        "answers": answers,
        "needs_quiz": needs_quiz,
    }


def save_profile(
    db: sqlite3.Connection,
    name: str,
    answers: dict[str, str],
    profile_id: int | None = None,
) -> int:
    name = name.strip()
    if not name or len(name) > 50:
        raise ValueError("Название профиля должно содержать от 1 до 50 символов")

    payload = json.dumps(validate_answers(answers), ensure_ascii=False)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with db:
        if profile_id is None:
            cursor = db.execute(
                "INSERT INTO profiles "
                "(name, answers_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (name, payload, timestamp, timestamp),
            )
            return cursor.lastrowid

        cursor = db.execute(
            "UPDATE profiles SET name = ?, answers_json = ?, updated_at = ? "
            "WHERE id = ?",
            (name, payload, timestamp, profile_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Профиль не найден")
        return profile_id


def save_request(
    db: sqlite3.Connection,
    city: str,
    profile_name: str,
    rows: list[tuple[ForecastDay, OutfitOption, str]],
) -> int:
    if not rows:
        raise ValueError("Нет рекомендаций для сохранения")
    if any(not isinstance(option, OutfitOption) for _, option, _ in rows):
        raise ValueError("Нельзя сохранить день без комплекта одежды")

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    forecast = [asdict(day) for day, _, _ in rows]
    with db:
        cursor = db.execute(
            "INSERT INTO requests "
            "(city, requested_at, forecast_json, profile_name) "
            "VALUES (?, ?, ?, ?)",
            (city, timestamp, json.dumps(forecast, ensure_ascii=False), profile_name),
        )
        request_id = cursor.lastrowid
        recommendations = [
            (
                request_id,
                day.date,
                day.temperature,
                day.observation_time,
                option.description,
                method,
                json.dumps(asdict(option), ensure_ascii=False),
            )
            for day, option, method in rows
        ]
        db.executemany(
            "INSERT INTO recommendations "
            "(request_id, forecast_date, temperature_c, observation_time, "
            "recommendation, method, options_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            recommendations,
        )
    return request_id


def _saved_option(raw: str | None, recommendation: str) -> dict:
    """Read one outfit from current or older multi-outfit history records."""
    fallback = {"title": "Рекомендация", "description": recommendation}
    data = _parse_json(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return fallback

    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        description = item.get("description")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(description, str) or not description.strip():
            continue
        note = item.get("note", "")
        return {
            "title": title,
            "description": description,
            "note": note if isinstance(note, str) else "",
        }
    return fallback


def last_requests(db: sqlite3.Connection, limit: int = 10) -> list[dict]:
    if limit < 0:
        raise ValueError("Лимит истории не может быть отрицательным")
    requests_rows = db.execute(
        "SELECT id, city, requested_at, profile_name, forecast_json FROM requests "
        "ORDER BY requested_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    result = []
    for request in requests_rows:
        forecast_days = _parse_json(request["forecast_json"])
        if not isinstance(forecast_days, list):
            forecast_days = []
        forecast_by_date = {
            day["date"]: day
            for day in forecast_days
            if isinstance(day, dict) and isinstance(day.get("date"), str)
        }
        days = db.execute(
            "SELECT forecast_date, temperature_c, observation_time, "
            "recommendation, method, options_json FROM recommendations "
            "WHERE request_id = ? ORDER BY forecast_date",
            (request["id"],),
        ).fetchall()
        parsed_days = []
        for day in days:
            item = dict(day)
            item["forecast"] = forecast_by_date.get(item["forecast_date"])
            item["option"] = _saved_option(
                item["options_json"], item["recommendation"]
            )
            del item["options_json"]
            parsed_days.append(item)
        request_data = dict(request)
        del request_data["forecast_json"]
        result.append({**request_data, "days": parsed_days})

    return result
