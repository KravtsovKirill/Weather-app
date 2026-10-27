"""Offline checks for the OpenRouter recommendation flow."""

from __future__ import annotations

import copy
import json
import unittest

import requests

from profiles import answer_labels, default_answers
from recommendations import (
    RecommendationError,
    _read_batch_outfits,
    recommend_batch,
)
from weather import ForecastDay

FREE_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
    "dots-studio/dots-3-note-preview:free",
]


def completion(content: object) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def outfit_json() -> str:
    return json.dumps(
        {
            "option": {
                "title": "Ваш комплект",
                "description": "Рубашка, кардиган, брюки и ботинки.",
                "note": "Долгая прогулка и ветер требуют съёмного слоя.",
            }
        },
        ensure_ascii=False,
    )


def batch_json(days: list[ForecastDay], *, reverse: bool = False) -> str:
    entries = []
    for day in days:
        option = copy.deepcopy(json.loads(outfit_json())["option"])
        option["description"] = f"{day.date}: {option['description']}"
        entries.append({"date": day.date, "option": option})
    if reverse:
        entries.reverse()
    return json.dumps({"days": entries}, ensure_ascii=False)


class FakeResponse:
    def __init__(
        self, status: int, payload: object = None, headers: dict[str, str] | None = None
    ):
        self.status_code = status
        self.payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **copy.deepcopy(kwargs)})
        if not self.responses:
            raise AssertionError("Unexpected OpenRouter request")
        return self.responses.pop(0)


def day_with_full_weather() -> ForecastDay:
    return ForecastDay(
        "2026-11-01",
        5.0,
        "12:00",
        condition="небольшой дождь",
        precipitation_probability=0.75,
        rain_mm=4.2,
        snow_mm=0.0,
        feels_like=1.0,
        humidity=88.0,
        wind_speed=8.0,
        wind_gust=13.0,
        temperature_min=-2.0,
        temperature_max=9.0,
    )


def parse_one(content: str):
    return _read_batch_outfits(content, [day_with_full_weather()])[0]


def recommend_one(city, day, api_key, model, answers, session):
    return recommend_batch(city, [day], api_key, model, answers, session)[0]


class RecommendationTests(unittest.TestCase):
    def test_markdown_wrapped_json_and_foreign_words_are_accepted(self) -> None:
        content = (
            "Вариант [1]:\n```json\n"
            + outfit_json().replace("Рубашка", "oversize-рубашка")
            + "\n```\nПодберите по размеру."
        )
        outfit = parse_one(content)
        self.assertIn("oversize-рубашка", outfit.description)

    def test_equivalent_json_field_names_are_accepted(self) -> None:
        payload = json.loads(outfit_json())
        renamed = {
            "recommendation": {
                "name": payload["option"]["title"],
                "outfit": payload["option"]["description"],
                "reason": payload["option"]["note"],
            }
        }
        outfit = parse_one(json.dumps(renamed, ensure_ascii=False))
        self.assertEqual(outfit.note, payload["option"]["note"])

    def test_single_outfit_without_optional_title_or_note_is_accepted(self) -> None:
        outfit = parse_one(
            json.dumps({"outfit": "Лонгслив, брюки и ботинки."}, ensure_ascii=False)
        )
        self.assertEqual(outfit.title, "Ваш комплект")
        self.assertEqual(outfit.note, "")

    def test_missing_text_does_not_repeat_same_generation(self) -> None:
        session = FakeSession(
            FakeResponse(200, completion(None)),
            FakeResponse(200, completion(outfit_json())),
        )
        try:
            outfit, _ = recommend_one(
                "Москва",
                day_with_full_weather(),
                "test-key",
                "openrouter/free",
                default_answers(),
                session=session,
            )
        except RecommendationError as exc:
            self.assertNotIn("Модель дважды", str(exc))
            self.assertNotIn("Модель вернула пустой ответ", str(exc))
        else:
            self.assertTrue(outfit.description)
        self.assertLessEqual(len(session.calls), 2)
        if len(session.calls) == 2:
            first = {
                k: v for k, v in session.calls[0]["json"].items() if k != "messages"
            }
            second = {
                k: v for k, v in session.calls[1]["json"].items() if k != "messages"
            }
            self.assertNotEqual(first, second)

    def test_multipart_text_content_is_read(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                completion([{"type": "text", "text": outfit_json()}]),
            )
        )
        outfit, _ = recommend_one(
            "Москва",
            day_with_full_weather(),
            "test-key",
            "openrouter/free",
            default_answers(),
            session=session,
        )
        self.assertTrue(outfit.description)

    def test_json_mode_rejection_retries_without_it(self) -> None:
        session = FakeSession(
            FakeResponse(422), FakeResponse(200, completion(outfit_json()))
        )
        outfit, _ = recommend_one(
            "Москва",
            day_with_full_weather(),
            "test-key",
            "openrouter/free",
            default_answers(),
            session=session,
        )
        self.assertTrue(outfit.description)
        self.assertIn("response_format", session.calls[0]["json"])
        self.assertNotIn("response_format", session.calls[1]["json"])

    def test_gemma_json_mode_rejection_retries_without_it(self) -> None:
        session = FakeSession(
            FakeResponse(422), FakeResponse(200, completion(outfit_json()))
        )
        recommend_one(
            "Москва",
            day_with_full_weather(),
            "test-key",
            "google/gemma-4-26b-a4b-it:free",
            default_answers(),
            session=session,
        )
        self.assertIn("response_format", session.calls[0]["json"])
        self.assertNotIn("response_format", session.calls[1]["json"])
        self.assertEqual(session.calls[0]["json"]["reasoning"], {"enabled": False})
        self.assertEqual(session.calls[1]["json"]["models"], FREE_MODELS)

    def test_removed_named_free_model_switches_to_free_router(self) -> None:
        session = FakeSession(
            FakeResponse(404), FakeResponse(200, completion(outfit_json()))
        )
        recommend_one(
            "Москва",
            day_with_full_weather(),
            "test-key",
            "old/model:free",
            default_answers(),
            session=session,
        )
        self.assertEqual(session.calls[0]["json"]["model"], "old/model:free")
        self.assertEqual(session.calls[1]["json"]["models"], FREE_MODELS)

    def test_rate_limit_does_not_trigger_more_requests(self) -> None:
        session = FakeSession(FakeResponse(429))
        with self.assertRaisesRegex(
            RecommendationError, "ограничение бесплатных рекомендаций"
        ):
            recommend_one(
                "Москва",
                day_with_full_weather(),
                "test-key",
                "openrouter/free",
                default_answers(),
                session=session,
            )
        self.assertEqual(len(session.calls), 1)

    def test_legacy_multioption_response_uses_only_first(self) -> None:
        primary = json.loads(outfit_json())["option"]
        secondary = {**primary, "description": "Другой комплект."}
        outfit = parse_one(
            json.dumps({"options": [primary, secondary]}, ensure_ascii=False)
        )
        self.assertEqual(outfit.description, primary["description"])

    def test_missing_api_key_does_not_call_openrouter(self) -> None:
        session = FakeSession()
        with self.assertRaisesRegex(
            RecommendationError, "ключ сервиса рекомендаций"
        ) as raised:
            recommend_one(
                "Москва",
                day_with_full_weather(),
                "   ",
                "openrouter/free",
                default_answers(),
                session=session,
            )
        self.assertNotIn("OPENROUTER", str(raised.exception))
        self.assertEqual(session.calls, [])

    def test_batch_uses_one_request_for_all_days_and_quiz_answers(self) -> None:
        days = [
            day_with_full_weather(),
            ForecastDay("2026-11-02", 22.0, "12:00", condition="ясно"),
        ]
        session = FakeSession(
            FakeResponse(200, completion(batch_json(days, reverse=True)))
        )
        answers = default_answers()
        answers.update({"cold": "sensitive", "activity": "active", "shoes": "boots"})

        result = recommend_batch(
            "Москва, RU", days, "test-key", "openrouter/free", answers, session
        )

        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(source == "ИИ" for _, source in result))
        self.assertTrue(all(outfit.description for outfit, _ in result))
        self.assertIn(days[0].date, result[0][0].description)
        self.assertIn(days[1].date, result[1][0].description)

        body = session.calls[0]["json"]
        self.assertEqual(body["models"], FREE_MODELS)
        self.assertEqual(body["reasoning"], {"enabled": False})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertGreaterEqual(body["max_tokens"], 1100)
        context = json.loads(body["messages"][1]["content"])
        self.assertEqual(context["город"], "Москва, RU")
        self.assertEqual(len(context["ответы_анкеты"]), 10)
        for label in answer_labels(answers):
            self.assertIn(label, context["ответы_анкеты"])
        self.assertEqual(context["дни"][0]["дата"], "2026-11-01")
        self.assertEqual(context["дни"][0]["температура_цельсий"], 5.0)
        self.assertEqual(context["дни"][0]["условия_около_полудня"], "небольшой дождь")
        self.assertEqual(
            context["дни"][0]["минимальная_температура_за_день_цельсий"],
            -2.0,
        )
        self.assertEqual(
            context["дни"][0]["максимальная_температура_за_день_цельсий"], 9.0
        )
        self.assertEqual(context["дни"][0]["ощущается_как_цельсий"], 1.0)
        self.assertEqual(context["дни"][0]["влажность_около_полудня_проценты"], 88.0)
        self.assertEqual(
            context["дни"][0]["максимальная_скорость_ветра_за_день_м_с"], 8.0
        )
        self.assertEqual(
            context["дни"][0]["максимальный_порыв_ветра_за_день_м_с"], 13.0
        )
        self.assertEqual(
            context["дни"][0][
                "максимальная_вероятность_осадков_за_интервал_дня_проценты"
            ],
            75,
        )
        self.assertEqual(context["дни"][0]["дождь_за_день_мм"], 4.2)
        self.assertEqual(context["дни"][0]["снег_за_день_мм"], 0.0)
        self.assertIsNone(context["дни"][1]["дождь_за_день_мм"])
        self.assertIn("ОДИН", body["messages"][0]["content"])
        self.assertIn('"option"', body["messages"][0]["content"])
        self.assertIn("минимум и максимум", body["messages"][0]["content"])
        self.assertIn("ответы анкеты", body["messages"][0]["content"].casefold())
        self.assertIn(
            "а не вероятность осадков за весь день", body["messages"][0]["content"]
        )

    def test_batch_markdown_json_and_missing_dates_are_accepted(self) -> None:
        days = [day_with_full_weather()]
        content = "```json\n" + batch_json(days) + "\n```"
        session = FakeSession(FakeResponse(200, completion(content)))
        result = recommend_batch(
            "Москва", days, "test-key", "openrouter/free", default_answers(), session
        )
        self.assertTrue(result[0][0].description)
        self.assertEqual(len(session.calls), 1)

    def test_batch_missing_day_retries_once(self) -> None:
        days = [
            day_with_full_weather(),
            ForecastDay("2026-11-02", 22.0, "12:00"),
        ]
        incomplete = batch_json(days[:1])
        session = FakeSession(
            FakeResponse(200, completion(incomplete)),
            FakeResponse(200, completion(batch_json(days))),
        )
        result = recommend_batch(
            "Москва", days, "test-key", "openrouter/free", default_answers(), session
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(session.calls[1]["json"]["messages"]), 3)

    def test_batch_daily_limit_429_does_not_retry_or_switch_models(self) -> None:
        session = FakeSession(
            FakeResponse(
                429,
                {"error": {"message": "Free model daily rate limit exceeded"}},
                {"Retry-After": "120"},
            )
        )
        with self.assertRaises(RecommendationError) as raised:
            recommend_batch(
                "Москва",
                [day_with_full_weather()],
                "test-key",
                "google/gemma-4-26b-a4b-it:free",
                default_answers(),
                session,
            )
        self.assertIn("дневной лимит", str(raised.exception))
        self.assertIn("после обновления лимита", str(raised.exception))
        self.assertIn("через 2 мин", str(raised.exception))
        self.assertNotIn("OpenRouter", str(raised.exception))
        self.assertEqual(len(session.calls), 1)

    def test_batch_provider_limit_429_does_not_retry(self) -> None:
        session = FakeSession(
            FakeResponse(
                429,
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "metadata": {"provider_code": 429},
                    }
                },
            )
        )
        with self.assertRaisesRegex(RecommendationError, "Сервис рекомендаций"):
            recommend_batch(
                "Москва",
                [day_with_full_weather()],
                "test-key",
                "google/gemma-4-26b-a4b-it:free",
                default_answers(),
                session,
            )
        self.assertEqual(len(session.calls), 1)

    def test_batch_http_200_top_level_provider_error_is_not_called_empty(self) -> None:
        # OpenRouter can return a provider failure after generation starts.
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "error": {
                        "message": "Provider returned error while generating",
                        "code": 503,
                    },
                    "choices": [],
                },
            )
        )
        with self.assertRaises(RecommendationError) as raised:
            recommend_batch(
                "Москва",
                [day_with_full_weather()],
                "test-key",
                "openrouter/free",
                default_answers(),
                session,
            )
        self.assertNotIn("пустой ответ", str(raised.exception))
        self.assertNotIn("Модель дважды", str(raised.exception))
        self.assertNotIn("не вернул текст", str(raised.exception))
        self.assertEqual(len(session.calls), 1)

    def test_batch_http_200_choice_error_is_not_called_empty(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "error",
                            "error": {
                                "message": "The model provider failed while generating",
                                "code": 503,
                            },
                            "message": {"role": "assistant", "content": None},
                        }
                    ]
                },
            )
        )
        with self.assertRaises(RecommendationError) as raised:
            recommend_batch(
                "Москва",
                [day_with_full_weather()],
                "test-key",
                "openrouter/free",
                default_answers(),
                session,
            )
        self.assertNotIn("пустой ответ", str(raised.exception))
        self.assertNotIn("Модель дважды", str(raised.exception))
        self.assertEqual(len(session.calls), 1)

    def test_batch_empty_after_output_limit_is_classified_or_recovered(self) -> None:
        days = [day_with_full_weather()]
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "model": "some/reasoning-model:free",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "I should make three complete outfits...",
                            },
                        }
                    ],
                    "usage": {
                        "completion_tokens": 1800,
                        "completion_tokens_details": {"reasoning_tokens": 1800},
                    },
                },
            ),
            FakeResponse(200, completion(batch_json(days))),
        )
        try:
            result = recommend_batch(
                "Москва",
                days,
                "test-key",
                "openrouter/free",
                default_answers(),
                session,
            )
        except RecommendationError as exc:
            self.assertRegex(str(exc), "лимит|токен|длин")
            self.assertNotIn("Модель дважды", str(exc))
        else:
            self.assertTrue(result[0][0].description)
        self.assertLessEqual(len(session.calls), 2)
        if len(session.calls) == 2:
            first = {
                k: v for k, v in session.calls[0]["json"].items() if k != "messages"
            }
            second = {
                k: v for k, v in session.calls[1]["json"].items() if k != "messages"
            }
            self.assertNotEqual(first, second, "Do not repeat the same generation")

    def test_batch_reasoning_only_without_final_answer_is_not_retried_unchanged(
        self,
    ) -> None:
        days = [day_with_full_weather()]
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "model": "some/reasoning-model:free",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_details": [
                                    {
                                        "type": "reasoning.text",
                                        "text": "Need to compare days.",
                                    }
                                ],
                            },
                        }
                    ],
                },
            ),
            FakeResponse(200, completion(batch_json(days))),
        )
        try:
            result = recommend_batch(
                "Москва",
                days,
                "test-key",
                "openrouter/free",
                default_answers(),
                session,
            )
        except RecommendationError as exc:
            self.assertNotIn("Модель дважды", str(exc))
            self.assertNotIn("Модель вернула пустой ответ", str(exc))
        else:
            self.assertTrue(result[0][0].description)
        self.assertLessEqual(len(session.calls), 2)
        if len(session.calls) == 2:
            first = {
                k: v for k, v in session.calls[0]["json"].items() if k != "messages"
            }
            second = {
                k: v for k, v in session.calls[1]["json"].items() if k != "messages"
            }
            self.assertNotEqual(first, second, "Do not repeat the same generation")

    def test_empty_reply_from_third_free_model_uses_a_different_model(self) -> None:
        days = [day_with_full_weather()]
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "model": FREE_MODELS[2],
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": None},
                        }
                    ],
                },
            ),
            FakeResponse(200, completion(batch_json(days))),
        )

        result = recommend_batch(
            "Москва", days, "test-key", "openrouter/free", default_answers(), session
        )

        self.assertTrue(result[0][0].description)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["json"]["models"], FREE_MODELS)
        self.assertEqual(session.calls[1]["json"]["models"], FREE_MODELS[:1])
        self.assertEqual(session.calls[1]["json"]["reasoning"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()
