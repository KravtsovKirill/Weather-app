"""Offline checks for city lookup and OpenWeather forecast parsing."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import requests

from weather import (
    WeatherError,
    fetch_forecast,
    matching_location,
    select_noon_forecasts,
)

LOCAL_ZONE = timezone(timedelta(hours=3))


def slot(local_time: str, temp: float, **fields: object) -> dict:
    timestamp = int(
        datetime.fromisoformat(local_time).replace(tzinfo=LOCAL_ZONE).timestamp()
    )
    return {"dt": timestamp, "main": {"temp": temp}, **fields}


def forecast_payload() -> dict:
    return {
        "cod": "200",
        "city": {"timezone": 3 * 3600},
        "list": [
            slot(
                "2026-10-27T09:00:00",
                8,
                pop=0.2,
                rain={"3h": 0.4},
                wind={"speed": 3},
            ),
            slot(
                "2026-10-27T12:00:00",
                12,
                pop=0.8,
                rain={"3h": 1.2},
                wind={"speed": 8, "gust": 12},
                weather=[{"description": "небольшой дождь"}],
            ),
            slot(
                "2026-10-27T18:00:00",
                9,
                pop=0.4,
                snow={"3h": 0.3},
                wind={"speed": 4},
            ),
            slot("2026-10-28T00:00:00", 5, pop=0.7, rain={"3h": 0.8}),
            slot("2026-10-28T12:00:00", 15, pop=0.1),
            slot("2026-10-28T18:00:00", 11, pop=0.3, rain={"3h": 0.5}),
            slot("2026-10-29T00:00:00", 8, pop=0.2, rain={"3h": 0.2}),
        ],
    }


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> object:
        return self.payload


class BrokenJsonResponse(FakeResponse):
    def json(self) -> object:
        raise ValueError("invalid JSON")


class FakeSession:
    def __init__(self, *replies: FakeResponse | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict, int]] = []

    def get(self, url: str, *, params: dict, timeout: int) -> FakeResponse:
        self.calls.append((url, params, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class WeatherTests(unittest.TestCase):
    def test_forecast_uses_local_dates_and_all_available_weather(self) -> None:
        payload = forecast_payload()
        payload["list"][1]["main"].update({"feels_like": 7.5, "humidity": 72})

        first, second = select_noon_forecasts(payload, 2)

        self.assertEqual((first.date, first.observation_time), ("2026-10-27", "12:00"))
        self.assertEqual(
            (first.temperature, first.temperature_min, first.temperature_max),
            (12, 8, 12),
        )
        self.assertEqual((first.feels_like, first.humidity), (7.5, 72))
        self.assertEqual((first.wind_speed, first.wind_gust), (8, 12))
        self.assertEqual(first.condition, "небольшой дождь")
        self.assertAlmostEqual(first.precipitation_probability, 0.8)
        # Midnight's three-hour rain amount belongs to the preceding day.
        self.assertAlmostEqual(first.rain_mm, 2.4)
        self.assertAlmostEqual(first.snow_mm, 0.3)
        self.assertEqual(
            (second.temperature, second.temperature_min, second.temperature_max),
            (15, 5, 15),
        )
        self.assertAlmostEqual(second.precipitation_probability, 0.3)
        self.assertAlmostEqual(second.rain_mm, 0.7)

    def test_partial_first_day_uses_available_evening_slot(self) -> None:
        payload = {
            "city": {"timezone": 3 * 3600},
            "list": [
                slot("2026-10-27T21:00:00", 4),
                slot("2026-10-28T12:00:00", 10),
            ],
        }
        first, second = select_noon_forecasts(payload, 2)
        self.assertEqual(
            (first.observation_time, first.temperature_min, first.temperature_max),
            ("21:00", 4, 4),
        )
        self.assertEqual(second.observation_time, "12:00")

    def test_optional_weather_is_not_invented_when_provider_omits_it(self) -> None:
        item = slot("2026-10-27T12:00:00", 10, pop=0)
        item["main"].update({"feels_like": "nan", "humidity": 140})
        item["wind"] = {"speed": -1}
        payload = {"city": {"timezone": 3 * 3600}, "list": [item]}

        day = select_noon_forecasts(payload, 1)[0]

        self.assertEqual(day.precipitation_probability, 0)
        self.assertIsNone(day.rain_mm)
        self.assertIsNone(day.snow_mm)
        self.assertIsNone(day.feels_like)
        self.assertIsNone(day.humidity)
        self.assertIsNone(day.wind_speed)

    def test_malformed_or_short_forecast_is_rejected(self) -> None:
        for broken in (
            {"city": {"timezone": 0}, "list": []},
            {
                "city": {"timezone": 0},
                "list": [slot("2026-10-27T12:00:00", float("nan"))],
            },
            {"cod": "404", "city": {"timezone": 0}, "list": []},
        ):
            with self.subTest(broken=broken), self.assertRaises(WeatherError):
                select_noon_forecasts(broken, 1)
        with self.assertRaises(WeatherError):
            select_noon_forecasts(forecast_payload(), 5)

    def test_city_name_country_and_state_must_match(self) -> None:
        places = [
            {"name": "Springfield", "state": "Massachusetts", "country": "US"},
            {"name": "Springfield", "state": "Illinois", "country": "US"},
            {"name": "Korolyov", "local_names": {"ru": "Королёв"}, "country": "RU"},
        ]
        self.assertIs(matching_location("Springfield, Illinois, US", places), places[1])
        # OpenWeather resolves a two-letter state code before returning locations.
        self.assertIs(matching_location("Springfield, IL, US", [places[1]]), places[1])
        self.assertIsNone(matching_location("Springfield, Texas, US", places))
        self.assertIs(matching_location("Королев, RU", places), places[2])
        london = {"name": "London", "state": "England", "country": "GB"}
        self.assertIs(matching_location("London, England, GB", [london]), london)
        self.assertIsNone(matching_location("Неизвестный", places))

    def test_bare_russian_city_names_work_without_country_or_state(self) -> None:
        for city, location in (
            (
                "Москва",
                {"name": "Moscow", "local_names": {"ru": "Москва"}, "country": "RU"},
            ),
            (
                "Лондон",
                {"name": "London", "local_names": {"ru": "Лондон"}, "country": "GB"},
            ),
            ("London", {"name": "London", "country": "GB"}),
        ):
            with self.subTest(city=city):
                location.update({"lat": 50.0, "lon": 30.0})
                session = FakeSession(
                    FakeResponse([location]), FakeResponse(forecast_payload())
                )

                result = fetch_forecast(city, 1, "test-key", session)

                self.assertEqual(len(result.days), 1)
                self.assertEqual(session.calls[0][1]["q"], city)

    def test_bare_nonexistent_city_is_rejected(self) -> None:
        session = FakeSession(FakeResponse([]))
        with self.assertRaisesRegex(WeatherError, "не найден"):
            fetch_forecast("Несуществующийгород", 1, "test-key", session)
        self.assertEqual(len(session.calls), 1)

    def test_fetch_rejects_fuzzy_city_without_forecast_request(self) -> None:
        session = FakeSession(FakeResponse([{"name": "Somewhere", "country": "GB"}]))
        with self.assertRaisesRegex(WeatherError, "не найден"):
            fetch_forecast("Imaginarytown, GB", 1, "test-key", session)
        self.assertEqual(len(session.calls), 1)

    def test_fetch_uses_geocoded_coordinates_and_metric_forecast(self) -> None:
        location = {
            "name": "Moscow",
            "local_names": {"ru": "Москва"},
            "country": "RU",
            "lat": 55.75,
            "lon": 37.62,
        }
        session = FakeSession(
            FakeResponse([location]), FakeResponse(forecast_payload())
        )

        result = fetch_forecast("Москва, RU", 2, "test-key", session)

        self.assertEqual(result.city, "Moscow, RU")
        self.assertEqual(len(result.days), 2)
        self.assertEqual(
            session.calls[0][1],
            {"q": "Москва, RU", "limit": 5, "appid": "test-key"},
        )
        self.assertEqual(
            session.calls[1][1],
            {
                "lat": 55.75,
                "lon": 37.62,
                "appid": "test-key",
                "units": "metric",
                "lang": "ru",
            },
        )
        self.assertTrue(all(timeout == 10 for _, _, timeout in session.calls))

    def test_provider_and_input_errors_are_specific(self) -> None:
        with self.assertRaisesRegex(WeatherError, "Количество дней"):
            fetch_forecast("Москва", 6, "test-key", FakeSession())
        with self.assertRaisesRegex(WeatherError, "ключ в настройках"):
            fetch_forecast("Москва", 1, "test-key", FakeSession(FakeResponse({}, 401)))
        with self.assertRaisesRegex(WeatherError, "ограничил запросы"):
            fetch_forecast("Москва", 1, "test-key", FakeSession(FakeResponse({}, 429)))
        with self.assertRaisesRegex(WeatherError, "неожиданный ответ"):
            fetch_forecast(
                "Москва", 1, "test-key", FakeSession(FakeResponse({"cod": 401}))
            )
        with self.assertRaisesRegex(WeatherError, "нечитаемый ответ"):
            fetch_forecast(
                "Москва", 1, "test-key", FakeSession(BrokenJsonResponse(None))
            )
        with self.assertRaisesRegex(WeatherError, "не ответил вовремя"):
            fetch_forecast("Москва", 1, "test-key", FakeSession(requests.Timeout()))

    def test_invalid_coordinates_are_rejected_before_forecast_request(self) -> None:
        location = {
            "name": "Moscow",
            "country": "RU",
            "lat": float("nan"),
            "lon": 37.62,
        }
        session = FakeSession(FakeResponse([location]))
        with self.assertRaisesRegex(WeatherError, "координаты"):
            fetch_forecast("Moscow", 1, "test-key", session)
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
