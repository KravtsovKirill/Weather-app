"""Turn a personal profile and weather forecast into one outfit per day."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import requests

from profiles import answer_labels, validate_answers
from weather import ForecastDay


class RecommendationError(Exception):
    """An AI recommendation could not be produced or validated."""


class EmptyCompletionError(RecommendationError):
    """A provider finished a request without producing visible recommendation text."""

    def __init__(self, message: str, model: str = "") -> None:
        super().__init__(message)
        self.model = model


@dataclass(frozen=True)
class OutfitOption:
    title: str
    description: str
    note: str = ""


FREE_MODELS = (
    "google/gemma-4-26b-a4b-it:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
    "dots-studio/dots-3-note-preview:free",
)
JSON_MODELS = {*FREE_MODELS, "openrouter/free", "google/gemma-4-31b-it:free"}
DAY_KEYS = ("days", "recommendations", "results", "дни")
OPTION_KEYS = (
    "option",
    "recommendation",
    "комплект",
    "вариант",
    "options",
    "outfits",
    "варианты",
    "outfit",
)
DESCRIPTION_KEYS = ("description", "outfit", "описание")


def _validate_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecommendationError(f"В рекомендации не заполнено поле «{field}».")
    return value.strip()


def _first_field(item: dict, names: tuple[str, ...], default: object = None) -> object:
    for name in names:
        if name in item:
            return item[name]
    return default


def _json_from_text(content: str) -> object:
    """Accept JSON with a code fence or a short explanation around it."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            result, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and any(
            key in result for key in (*DAY_KEYS, *OPTION_KEYS, *DESCRIPTION_KEYS)
        ):
            return result
        if isinstance(result, list) and result and all(
            isinstance(item, (dict, list)) for item in result
        ):
            return result
    raise RecommendationError("Ответ сервиса рекомендаций имеет неожиданный формат.")


def _option_from_document(document: object) -> OutfitOption:
    """Read one outfit while tolerating older array-shaped model responses."""
    if isinstance(document, dict):
        document = next(
            (
                document[key]
                for key in OPTION_KEYS
                if isinstance(document.get(key), (dict, list))
            ),
            document,
        )
    if isinstance(document, list):
        if not document:
            raise RecommendationError("В ответе нет комплекта одежды.")
        document = document[0]
    if not isinstance(document, dict):
        raise RecommendationError("В ответе нет полного комплекта одежды.")
    title = _first_field(document, ("title", "name", "название"), "Ваш комплект")
    description = _first_field(document, DESCRIPTION_KEYS)
    note = _first_field(document, ("note", "reason", "пояснение"), "")
    if not isinstance(note, str):
        note = ""
    return OutfitOption(
        _validate_text(title, "название"),
        _validate_text(description, "описание"),
        note.strip(),
    )


def _read_batch_outfits(
    content: str, days: Sequence[ForecastDay]
) -> list[OutfitOption]:
    if not isinstance(content, str) or not content.strip():
        raise RecommendationError("Сервис рекомендаций не сформировал ответ.")
    document = _json_from_text(content.strip())
    if isinstance(document, dict):
        raw_days = _first_field(document, DAY_KEYS)
        if raw_days is None and len(days) == 1:
            return [_option_from_document(document)]
    else:
        raw_days = document

    if len(days) == 1 and isinstance(raw_days, list) and raw_days and all(
        isinstance(item, dict)
        and any(key in item for key in DESCRIPTION_KEYS)
        for item in raw_days
    ):
        return [_option_from_document(raw_days)]

    if not isinstance(raw_days, list) or len(raw_days) != len(days):
        raise RecommendationError("Получены рекомендации не для всех дней.")

    dated = []
    for raw_day in raw_days:
        if not isinstance(raw_day, (dict, list)):
            raise RecommendationError("В ответе не хватает данных для одного из дней.")
        date = (
            _first_field(raw_day, ("date", "forecast_date", "дата"))
            if isinstance(raw_day, dict)
            else None
        )
        dated.append((date, _option_from_document(raw_day)))

    dates = [date for date, _ in dated]
    if all(date is None for date in dates):
        return [option for _, option in dated]
    expected = [day.date for day in days]
    if any(not isinstance(date, str) for date in dates) or sorted(dates) != sorted(
        expected
    ):
        raise RecommendationError("Даты рекомендаций не совпадают с прогнозом.")
    by_date = {date: option for date, option in dated}
    return [by_date[day.date] for day in days]


def _response_text(payload: object) -> str:
    """Extract text from regular and multipart chat completion messages."""
    if not isinstance(payload, dict):
        raise RecommendationError("Сервис рекомендаций вернул неожиданный ответ.")
    choices = payload.get("choices")
    if isinstance(payload.get("error"), dict):
        raise RecommendationError(_embedded_error(payload["error"]))
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RecommendationError("Не удалось получить рекомендацию. Попробуйте позже.")
    choice = choices[0]
    if isinstance(choice.get("error"), dict) or choice.get("finish_reason") == "error":
        raise RecommendationError(_embedded_error(choice.get("error") or {}))
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RecommendationError("Не удалось получить рекомендацию. Попробуйте позже.")
    content = message.get("content")
    if isinstance(content, str):
        result = content
    elif isinstance(content, list):
        result = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") in (None, "text", "output_text")
            and isinstance(item.get("text"), str)
        )
    elif content is None:
        result = ""
    else:
        raise RecommendationError("Сервис рекомендаций вернул неожиданный ответ.")
    if result.strip():
        return result

    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        explanation = "Не удалось завершить подбор из-за ограничения длины ответа."
    elif message.get("reasoning") or message.get("reasoning_details"):
        explanation = "Сервис не сформировал готовую рекомендацию."
    else:
        explanation = "Сервис рекомендаций не сформировал ответ."
    model = payload.get("model")
    raise EmptyCompletionError(
        explanation,
        model if isinstance(model, str) else "",
    )


def _embedded_error(error: dict) -> str:
    """Turn an error inside an HTTP 200 completion into a useful short message."""
    code = error.get("code")
    return _http_error(int(code)) if str(code).isdigit() else _http_error(None)


def _http_error(status: int | None) -> str:
    if status in (401, 403):
        return "Не удалось войти в сервис рекомендаций. Проверьте ключ в настройках."
    if status == 402:
        return "Выбранный сервис требует платного доступа. Выберите бесплатный вариант."
    if status == 429:
        return "Сервис рекомендаций временно ограничил запросы. Повторите позже."
    if status == 404:
        return "Сейчас нет доступного способа сформировать рекомендацию."
    if status in (500, 502, 503, 504):
        return "Сервис рекомендаций сейчас недоступен. Повторите запрос позже."
    return "Не удалось сформировать рекомендацию. Проверьте настройки и повторите."


def _retry_after_hint(response: object) -> str:
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f" Повторите через {seconds} с."
    return f" Повторите примерно через {ceil(seconds / 60)} мин."


def _rate_limit_error(response: object) -> str:
    try:
        payload = response.json()
    except (AttributeError, ValueError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = {}
    message = error.get("message")
    words = (
        re.sub(r"[_-]+", " ", message.casefold()) if isinstance(message, str) else ""
    )
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    if (
        metadata.get("provider_code") is not None
        or metadata.get("provider_error_code") is not None
        or metadata.get("limit_source") == "provider"
    ) or ("provider" in words and "limit" in words):
        detail = "Сервис рекомендаций временно ограничил запросы. Повторите позже."
    elif any(
        marker in words
        for marker in ("daily", "per day", "day limit", "day quota", "24 hour")
    ):
        detail = (
            "Исчерпан дневной лимит бесплатных рекомендаций. "
            "Повторите попытку после обновления лимита."
        )
    elif any(marker in words for marker in ("per minute", "minute limit", "rpm")):
        detail = "Достигнут кратковременный лимит рекомендаций."
    else:
        detail = (
            "Сейчас действует ограничение бесплатных рекомендаций. "
            "Попробуйте позже."
        )
    return detail + _retry_after_hint(response)


def _request_content(
    session,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 3000,
    timeout: int = 45,
) -> str:
    selected_model = (model or "").strip() or "openrouter/free"
    if selected_model == "openrouter/free":
        model_choices: tuple[str, ...] | None = FREE_MODELS
    elif selected_model in FREE_MODELS:
        model_choices = (selected_model,) + tuple(
            candidate for candidate in FREE_MODELS if candidate != selected_model
        )
    elif selected_model == "google/gemma-4-31b-it:free":
        model_choices = (selected_model, *FREE_MODELS[1:])
    else:
        model_choices = None
    json_mode = selected_model in JSON_MODELS
    empty_recovered = False
    # All listed free models accept JSON mode and can produce text without reasoning.
    for _ in range(4):
        body = {
            "temperature": 0.45,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if model_choices:
            body["models"] = list(model_choices)
            body["reasoning"] = {"enabled": False}
        else:
            body["model"] = selected_model
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            return _response_text(response.json())
        except EmptyCompletionError as exc:
            if empty_recovered or (
                model_choices is None and not selected_model.endswith(":free")
            ):
                raise RecommendationError(
                    f"{exc} Повторный подбор тоже не удался."
                    if empty_recovered
                    else str(exc)
                ) from exc
            if model_choices:
                alternative = next(
                    (
                        candidate
                        for candidate in model_choices
                        if candidate != exc.model
                    ),
                    model_choices[-1],
                )
                model_choices = (alternative,)
            else:
                model_choices = FREE_MODELS
            json_mode = True
            empty_recovered = True
            continue
        except requests.Timeout as exc:
            raise RecommendationError(
                "Подбор занял слишком много времени. Попробуйте ещё раз."
            ) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                raise RecommendationError(_rate_limit_error(exc.response)) from exc
            if (
                model_choices is None
                and selected_model.endswith(":free")
                and status in (404, 500, 502, 503, 504)
            ):
                model_choices, json_mode = FREE_MODELS, True
                continue
            if json_mode and (status in (400, 422)):
                json_mode = False
                continue
            raise RecommendationError(_http_error(status)) from exc
        except (requests.RequestException, ValueError) as exc:
            raise RecommendationError(
                "Не удалось получить рекомендацию. Проверьте подключение и повторите."
            ) from exc
    raise RecommendationError("Не удалось получить рекомендацию. Попробуйте позже.")


def _weather_context(day: ForecastDay) -> dict[str, object]:
    return {
        "дата": day.date,
        "время_прогноза": day.observation_time,
        "температура_цельсий": day.temperature,
        "минимальная_температура_за_день_цельсий": day.temperature_min,
        "максимальная_температура_за_день_цельсий": day.temperature_max,
        "ощущается_как_цельсий": day.feels_like,
        "условия_около_полудня": day.condition,
        "влажность_около_полудня_проценты": day.humidity,
        "максимальная_скорость_ветра_за_день_м_с": day.wind_speed,
        "максимальный_порыв_ветра_за_день_м_с": day.wind_gust,
        "максимальная_вероятность_осадков_за_интервал_дня_проценты": (
            round(day.precipitation_probability * 100)
            if day.precipitation_probability is not None
            else None
        ),
        "дождь_за_день_мм": day.rain_mm,
        "снег_за_день_мм": day.snow_mm,
    }


SYSTEM_PROMPT = (
    "Ты консультант по повседневной одежде. Для КАЖДОЙ даты предложи ОДИН "
    "наиболее подходящий полный комплект. Учти все 10 ответов анкеты: переносимость "
    "холода и жары, перегрев, время и место на улице, активность, слои, стиль и обувь. "
    "Согласуй конфликтующие предпочтения с помощью снимаемых слоёв. Для каждого дня "
    "учитывай температуру около полудня, минимум и максимум, ощущаемую температуру, "
    "влажность, скорость и порывы ветра, вероятность и объём дождя или снега. "
    "Вероятность осадков — максимум среди трёхчасовых интервалов, а не вероятность "
    "осадков за весь день. Дневные сводки могут быть неполными; null означает "
    "неизвестные данные. Не утверждай, что осадки обязательно начнутся. "
    "В каждом комплекте назови конкретный верх, низ, обувь и необходимые аксессуары. "
    "Если погода меняется, подбирай одежду заново без одинаковых шаблонов. "
    "Не предлагай ненужную зимнюю или дождевую одежду. В note кратко объясни выбор "
    "через конкретные ответы анкеты и параметры прогноза. Иностранные слова допустимы. "
    "Верни только JSON без Markdown, по порядку дат: "
    '{"days":[{"date":"YYYY-MM-DD","option":'
    '{"title":"...","description":"...","note":"..."}}]}'
)


def recommend_batch(
    city: str,
    days: Sequence[ForecastDay],
    api_key: str,
    model: str,
    answers: dict[str, str],
    session=requests,
) -> list[tuple[OutfitOption, str]]:
    """Generate one outfit per day with one request for 1–5 days."""
    if not api_key or not api_key.strip():
        raise RecommendationError(
            "Добавьте ключ сервиса рекомендаций в настройки приложения."
        )
    days = list(days)
    if not 1 <= len(days) <= 5 or any(not isinstance(day, ForecastDay) for day in days):
        raise RecommendationError("Для подбора нужен прогноз на 1–5 дней.")
    if len({day.date for day in days}) != len(days):
        raise RecommendationError("В прогнозе повторяются даты.")

    context = {
        "город": city,
        "ответы_анкеты": answer_labels(validate_answers(answers)),
        "дни": [_weather_context(day) for day in days],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    for attempt in range(2):
        content = _request_content(
            session,
            api_key.strip(),
            model,
            messages,
            max_tokens=max(1100, min(3600, 750 * len(days))),
            timeout=max(55, 20 * len(days)),
        )
        try:
            outfits = _read_batch_outfits(content, days)
            return [(outfit, "ИИ") for outfit in outfits]
        except RecommendationError as exc:
            if attempt == 1:
                raise RecommendationError(
                    "Не удалось подготовить полные рекомендации. "
                    "Попробуйте ещё раз позже."
                ) from exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Предыдущий ответ не прошёл проверку: {exc} "
                        "Повтори рекомендации для всех указанных дат: один "
                        "комплект на каждую дату. Верни только JSON."
                    ),
                }
            )
    raise AssertionError("unreachable")
