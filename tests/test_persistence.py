"""Offline checks for quiz profiles and saved recommendation history."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from profiles import QUESTIONS, answer_labels, default_answers, validate_answers
from recommendations import OutfitOption
from storage import (
    connect,
    get_profile,
    last_requests,
    list_profiles,
    save_profile,
    save_request,
)
from weather import ForecastDay


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "history.sqlite3"
        self.db = connect(self.path)
        self.addCleanup(self.db.close)

    def test_quiz_answers_are_validated_and_profiles_can_be_updated(self) -> None:
        answers = {question.key: question.choices[0][0] for question in QUESTIONS}
        self.assertEqual(len(answer_labels(answers)), 10)

        profile_id = save_profile(self.db, "  Алиса  ", answers)
        self.assertEqual(list_profiles(self.db), [{"id": profile_id, "name": "Алиса"}])
        self.assertEqual(
            get_profile(self.db, profile_id),
            {
                "id": profile_id,
                "name": "Алиса",
                "answers": answers,
                "needs_quiz": False,
            },
        )

        updated = default_answers()
        self.assertEqual(
            save_profile(self.db, "Алиса", updated, profile_id), profile_id
        )
        self.assertEqual(get_profile(self.db, profile_id)["answers"], updated)

        with self.assertRaisesRegex(ValueError, "Выберите ответ"):
            validate_answers({"cold": "sensitive"})
        with self.assertRaisesRegex(ValueError, "Выберите ответ"):
            save_profile(self.db, "Неверный", {**updated, "cold": "unexpected"})
        with self.assertRaisesRegex(ValueError, "Выберите ответ"):
            validate_answers({**updated, "cold": []})
        self.assertEqual(len(list_profiles(self.db)), 1)

    def test_old_incomplete_profile_can_be_reopened_for_quiz(self) -> None:
        self.db.execute(
            "INSERT INTO profiles (name, answers_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("Старый", json.dumps({"cold": "hardy"}), "2025-01-01", "2025-01-01"),
        )
        profile_id = self.db.execute("SELECT id FROM profiles").fetchone()[0]
        profile = get_profile(self.db, profile_id)
        self.assertTrue(profile["needs_quiz"])
        self.assertEqual(profile["answers"]["cold"], "hardy")
        self.assertEqual(profile["answers"]["heat"], default_answers()["heat"])

        self.db.execute(
            "UPDATE profiles SET answers_json = ? WHERE id = ?",
            ("{broken", profile_id),
        )
        damaged = get_profile(self.db, profile_id)
        self.assertTrue(damaged["needs_quiz"])
        self.assertEqual(damaged["answers"], default_answers())

        self.db.execute(
            "UPDATE profiles SET answers_json = ? WHERE id = ?",
            (json.dumps({"cold": []}), profile_id),
        )
        self.assertTrue(get_profile(self.db, profile_id)["needs_quiz"])

    def test_history_keeps_every_weather_parameter_and_outfit(self) -> None:
        day = ForecastDay(
            "2026-10-27",
            7.5,
            "12:00",
            condition="небольшой дождь",
            precipitation_probability=0.8,
            rain_mm=4.2,
            snow_mm=0.0,
            feels_like=3.1,
            humidity=81,
            wind_speed=6.4,
            wind_gust=9.8,
        )
        option = OutfitOption(
            "Тёплый", "Куртка, брюки и ботинки", "Для долгой прогулки"
        )
        request_id = save_request(
            self.db, "Москва, RU", "Алиса", [(day, option, "ИИ")]
        )
        history = last_requests(self.db)

        self.assertEqual(history[0]["id"], request_id)
        self.assertEqual(history[0]["profile_name"], "Алиса")
        self.assertEqual(history[0]["days"][0]["forecast"], asdict(day))
        self.assertEqual(
            history[0]["days"][0]["option"], asdict(option)
        )
        self.assertEqual(history[0]["days"][0]["method"], "ИИ")
        saved_json = self.db.execute(
            "SELECT options_json FROM recommendations WHERE request_id = ?",
            (request_id,),
        ).fetchone()[0]
        self.assertEqual(json.loads(saved_json), asdict(option))

        with self.assertRaisesRegex(ValueError, "без комплекта"):
            save_request(self.db, "Москва, RU", "Алиса", [(day, None, "ИИ")])
        with self.assertRaisesRegex(ValueError, "Нет рекомендаций"):
            save_request(self.db, "Москва, RU", "Алиса", [])
        self.assertEqual(len(last_requests(self.db)), 1)

    def test_history_reads_first_valid_outfit_from_old_array(self) -> None:
        day = ForecastDay("2026-10-27", 7.5, "12:00")
        request_id = save_request(
            self.db,
            "Москва",
            "Алиса",
            [(day, OutfitOption("Первый", "Куртка и брюки"), "ИИ")],
        )
        old_options = [
            {"title": "", "description": "Повреждённая запись"},
            {"title": "Первый", "description": "Куртка и брюки"},
            {"title": "Второй", "description": "Пальто и брюки"},
        ]
        self.db.execute(
            "UPDATE recommendations SET options_json = ? WHERE request_id = ?",
            (json.dumps(old_options, ensure_ascii=False), request_id),
        )
        self.assertEqual(
            last_requests(self.db)[0]["days"][0]["option"], old_options[1]
            | {"note": ""},
        )

    def test_old_schema_migrates_and_damaged_options_use_saved_text(self) -> None:
        self.db.close()
        self.path.unlink()
        legacy = sqlite3.connect(self.path)
        legacy.executescript(
            """
            CREATE TABLE requests (
                id INTEGER PRIMARY KEY, city TEXT NOT NULL,
                requested_at TEXT NOT NULL, forecast_json TEXT NOT NULL
            );
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY, request_id INTEGER NOT NULL,
                forecast_date TEXT NOT NULL, temperature_c REAL NOT NULL,
                observation_time TEXT NOT NULL, recommendation TEXT NOT NULL,
                method TEXT NOT NULL
            );
            INSERT INTO requests VALUES (1, 'Казань', '2025-01-01', '[]');
            INSERT INTO recommendations VALUES
                (1, 1, '2025-01-02', -3, '12:00', 'Старая рекомендация', 'Правила');
            """
        )
        legacy.close()

        migrated = connect(self.path)
        self.addCleanup(migrated.close)
        self.assertIn(
            "profile_name",
            {row[1] for row in migrated.execute("PRAGMA table_info(requests)")},
        )
        self.assertIn(
            "options_json",
            {row[1] for row in migrated.execute("PRAGMA table_info(recommendations)")},
        )
        old = last_requests(migrated)[0]["days"][0]
        self.assertEqual(old["option"]["description"], "Старая рекомендация")

        migrated.execute("UPDATE recommendations SET options_json = '{broken'")
        migrated.execute("UPDATE requests SET forecast_json = '{broken'")
        self.assertEqual(
            last_requests(migrated)[0]["days"][0]["option"],
            {
                "title": "Рекомендация",
                "description": "Старая рекомендация",
            },
        )


if __name__ == "__main__":
    unittest.main()
