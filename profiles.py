from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    key: str
    section: str
    prompt: str
    choices: tuple[tuple[str, str], ...]


QUESTIONS = (
    Question(
        "cold",
        "Ощущения",
        "Как вы переносите холод?",
        (
            ("sensitive", "Мёрзну быстрее других"),
            ("average", "Примерно как большинство"),
            ("hardy", "Хорошо переношу холод"),
        ),
    ),
    Question(
        "heat",
        "Ощущения",
        "Как вы переносите жару?",
        (
            ("sensitive", "Мне быстро становится жарко"),
            ("average", "Примерно как большинство"),
            ("hardy", "Хорошо переношу жару"),
        ),
    ),
    Question(
        "hands_feet",
        "Ощущения",
        "Часто ли мёрзнут руки и ноги?",
        (
            ("often", "Часто"),
            ("sometimes", "Иногда"),
            ("rarely", "Редко"),
        ),
    ),
    Question(
        "sweat",
        "Ощущения",
        "Как быстро вам становится жарко при движении?",
        (
            ("quick", "Быстро"),
            ("average", "Обычно"),
            ("slow", "Редко перегреваюсь"),
        ),
    ),
    Question(
        "outside",
        "Ритм дня",
        "Сколько времени подряд вы обычно проводите на улице?",
        (
            ("short", "До 15 минут"),
            ("medium", "15–60 минут"),
            ("long", "Больше часа"),
        ),
    ),
    Question(
        "activity",
        "Ритм дня",
        "Как вы обычно двигаетесь на улице?",
        (
            ("low", "В основном стою или сижу"),
            ("walking", "Иду обычным шагом"),
            ("active", "Быстро иду или активно двигаюсь"),
        ),
    ),
    Question(
        "exposure",
        "Ритм дня",
        "Где чаще проходит время на улице?",
        (
            ("sheltered", "В городе, недолго между зданиями"),
            ("mixed", "По-разному"),
            ("open", "На открытых пространствах"),
        ),
    ),
    Question(
        "layers",
        "Одежда",
        "Как относитесь к многослойности?",
        (
            ("like", "Люблю несколько слоёв"),
            ("neutral", "По ситуации"),
            ("minimal", "Предпочитаю меньше слоёв"),
        ),
    ),
    Question(
        "style",
        "Одежда",
        "Какой стиль одежды ближе?",
        (
            ("casual", "Повседневный"),
            ("classic", "Классический"),
            ("sport", "Спортивный"),
        ),
    ),
    Question(
        "shoes",
        "Одежда",
        "Какую обувь удобнее носить?",
        (
            ("sneakers", "Кроссовки"),
            ("boots", "Ботинки"),
            ("flexible", "Без предпочтений"),
        ),
    ),
)


def default_answers() -> dict[str, str]:
    return {question.key: question.choices[1][0] for question in QUESTIONS}


def validate_answers(answers: Mapping[str, str]) -> dict[str, str]:
    """Require one known answer for every question before saving or prompting."""
    if not isinstance(answers, Mapping):
        raise TypeError("Ответы квиза должны быть набором значений")

    valid = {}
    for question in QUESTIONS:
        answer = answers.get(question.key)
        if not isinstance(answer, str) or answer not in dict(question.choices):
            raise ValueError(f"Выберите ответ на вопрос «{question.prompt}»")
        valid[question.key] = answer
    return valid


def restore_answers(saved: object) -> tuple[dict[str, str], bool]:
    """Keep valid old answers and mark incomplete profiles for a new quiz."""
    answers = default_answers()
    if not isinstance(saved, dict):
        return answers, True

    needs_quiz = False
    for question in QUESTIONS:
        value = saved.get(question.key)
        if isinstance(value, str) and value in dict(question.choices):
            answers[question.key] = value
        else:
            needs_quiz = True
    return answers, needs_quiz


def answer_labels(answers: dict[str, str]) -> list[str]:
    valid = validate_answers(answers)
    return [
        f"{question.prompt} — {dict(question.choices)[valid[question.key]]}"
        for question in QUESTIONS
    ]
