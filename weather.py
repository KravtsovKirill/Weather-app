"""Validate cities and summarize OpenWeather's three-hour forecast by local day."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Iterable

import requests


class WeatherError(Exception):
    pass


@dataclass(frozen=True)
class ForecastDay:
    date: str
    temperature: float
    observation_time: str
    condition: str | None = None
    precipitation_probability: float | None = None
    rain_mm: float | None = None
    snow_mm: float | None = None
    feels_like: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    wind_gust: float | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None


@dataclass(frozen=True)
class Forecast:
    city: str
    days: list[ForecastDay]


def normalized_place_name(value: str) -> str:
    """Ignore case, accents and punctuation when comparing city names."""
    plain = unicodedata.normalize("NFKD", value.casefold().replace("ё", "е"))
    plain = "".join(char for char in plain if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", plain, flags=re.UNICODE))


def matching_location(query: str, locations: list[dict]) -> dict | None:
    parts = [part.strip() for part in query.split(",")]
    if not 1 <= len(parts) <= 3 or not all(parts):
        return None

    requested_name = normalized_place_name(parts[0])
    country_code = parts[-1].upper() if len(parts) >= 2 else None
    if country_code and (
        len(country_code) != 2
        or not country_code.isascii()
        or not country_code.isalpha()
    ):
        return None
    requested_state = parts[1] if len(parts) == 3 else None

    # The geocoder accepts US state codes in the query but returns full state
    # names. Let it resolve two-letter codes; compare full names when supplied.
    state_name = (
        normalized_place_name(requested_state)
        if requested_state and len(requested_state) != 2
        else None
    )

    for place in locations:
        if not isinstance(place, dict):
            continue
        if country_code and str(place.get("country") or "").upper() != country_code:
            continue
        if state_name and (
            normalized_place_name(str(place.get("state") or "")) != state_name
        ):
            continue

        names = [place.get("name", "")]
        local_names = place.get("local_names")
        if isinstance(local_names, dict):
            names.extend(name for name in local_names.values() if isinstance(name, str))
        if any(
            normalized_place_name(name) == requested_name
            for name in names
            if isinstance(name, str)
        ):
            return place

    return None


def optional_number(
    value: object, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    """Read an optional finite number within its valid range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        return None
    return number


def _metric_values(
    items: Iterable[dict],
    key: str,
    *,
    field: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    """Collect valid numbers from a forecast field or nested weather section."""
    values = []
    for item in items:
        value = item.get(key)
        if field is not None:
            value = value.get(field) if isinstance(value, dict) else None
        number = optional_number(value, minimum=minimum, maximum=maximum)
        if number is not None:
            values.append(number)
    return values


def _response_json(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise WeatherError("Сервис погоды вернул нечитаемый ответ") from exc


def select_noon_forecasts(payload: dict, count: int) -> list[ForecastDay]:
    """Pick noon conditions and summarize changing weather for each local date."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise ValueError("Количество дней должно быть от 1 до 5")

    try:
        if not isinstance(payload, dict) or str(payload.get("cod", "200")) != "200":
            raise ValueError("forecast provider returned an error")
        local_zone = timezone(timedelta(seconds=int(payload["city"]["timezone"])))
        items = payload["list"]
        if not isinstance(items, list):
            raise TypeError("forecast list is not an array")

        grouped: dict[str, list[tuple[datetime, dict, float]]] = {}
        precipitation_by_day: dict[str, list[dict]] = {}
        for item in items:
            local_time = datetime.fromtimestamp(int(item["dt"]), local_zone)
            temperature = optional_number(item["main"]["temp"])
            if temperature is None:
                raise ValueError("missing temperature")
            grouped.setdefault(local_time.date().isoformat(), []).append(
                (local_time, item, temperature)
            )
            # rain.3h and snow.3h describe the preceding three hours. Their
            # midpoint keeps a midnight forecast on the day it mostly covers.
            interval_day = (local_time - timedelta(hours=1, minutes=30)).date()
            precipitation_by_day.setdefault(interval_day.isoformat(), []).append(item)

        result = []
        for day in sorted(grouped)[:count]:
            local_time, sample, temperature = min(
                grouped[day],
                key=lambda entry: (
                    abs(entry[0].hour * 60 + entry[0].minute - 720),
                    entry[0],
                ),
            )
            precipitation_items = precipitation_by_day.get(day, [])
            probabilities = _metric_values(
                precipitation_items, "pop", minimum=0, maximum=1
            )
            rain_values = _metric_values(
                precipitation_items, "rain", field="3h", minimum=0
            )
            snow_values = _metric_values(
                precipitation_items, "snow", field="3h", minimum=0
            )
            wind_items = [item for _, item, _ in grouped[day]]
            wind_speeds = _metric_values(wind_items, "wind", field="speed", minimum=0)
            wind_gusts = _metric_values(
                wind_items, "wind", field="gust", minimum=0
            )
            weather = sample.get("weather")
            condition = None
            if isinstance(weather, list) and weather and isinstance(weather[0], dict):
                raw_condition = weather[0].get("description") or weather[0].get("main")
                if isinstance(raw_condition, str):
                    condition = raw_condition.strip() or None
            result.append(
                ForecastDay(
                    day,
                    round(temperature, 1),
                    local_time.strftime("%H:%M"),
                    condition=condition,
                    precipitation_probability=(
                        max(probabilities) if probabilities else None
                    ),
                    rain_mm=round(sum(rain_values), 1) if rain_values else None,
                    snow_mm=round(sum(snow_values), 1) if snow_values else None,
                    feels_like=optional_number(sample["main"].get("feels_like")),
                    humidity=optional_number(
                        sample["main"].get("humidity"), minimum=0, maximum=100
                    ),
                    wind_speed=max(wind_speeds) if wind_speeds else None,
                    wind_gust=max(wind_gusts) if wind_gusts else None,
                    temperature_min=round(
                        min(value for _, _, value in grouped[day]), 1
                    ),
                    temperature_max=round(
                        max(value for _, _, value in grouped[day]), 1
                    ),
                )
            )
        if len(result) < count:
            raise ValueError("too few forecast days")
        return result
    except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
        raise WeatherError("Сервис погоды вернул неполный прогноз") from exc


def fetch_forecast(
    city: str,
    count: int,
    api_key: str,
    session=requests,
) -> Forecast:
    city = city.strip()
    if not city:
        raise WeatherError("Введите название города")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise WeatherError("Количество дней должно быть от 1 до 5")
    if not api_key or not api_key.strip():
        raise WeatherError("Добавьте ключ сервиса погоды в настройки приложения")
    api_key = api_key.strip()

    try:
        geo = session.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 5, "appid": api_key},
            timeout=10,
        )
        geo.raise_for_status()
        locations = _response_json(geo)
        if not isinstance(locations, list):
            raise WeatherError("Сервис погоды вернул неожиданный ответ")
        place = matching_location(city, locations)
        if place is None:
            raise WeatherError(
                f"Город «{city}» не найден. Проверьте написание "
                "или укажите код страны, например London, GB"
            )

        latitude = optional_number(place.get("lat"))
        longitude = optional_number(place.get("lon"))
        if (
            latitude is None
            or longitude is None
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise WeatherError("Сервис погоды вернул неверные координаты города")
        place_name = place.get("name")
        country = place.get("country")
        if (
            not isinstance(place_name, str)
            or not place_name.strip()
            or not isinstance(country, str)
            or not country.strip()
        ):
            raise WeatherError("Сервис погоды вернул неполные данные города")

        response = session.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
                "lang": "ru",
            },
            timeout=10,
        )
        response.raise_for_status()
        days = select_noon_forecasts(_response_json(response), count)
        state = place.get("state")
        city_label = ", ".join(
            part
            for part in (place_name, state if isinstance(state, str) else None, country)
            if part
        )
        return Forecast(city_label, days)
    except WeatherError:
        raise
    except requests.Timeout as exc:
        raise WeatherError("Сервис погоды не ответил вовремя") from exc
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            raise WeatherError(
                "Не удалось подключиться к сервису погоды. Проверьте ключ в настройках"
            ) from exc
        if status == 429:
            raise WeatherError(
                "Сервис погоды временно ограничил запросы. Попробуйте позже"
            ) from exc
        raise WeatherError("Сервис погоды временно недоступен") from exc
    except requests.RequestException as exc:
        raise WeatherError(
            "Не удалось получить прогноз погоды. Проверьте подключение"
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError("Сервис погоды вернул неожиданный ответ") from exc
