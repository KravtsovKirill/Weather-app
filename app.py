from __future__ import annotations

import os
import sqlite3
from dataclasses import fields
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from profiles import QUESTIONS, answer_labels, default_answers
from recommendations import (
    OutfitOption,
    RecommendationError,
    recommend_batch,
)
from storage import (
    connect,
    get_profile,
    last_requests,
    list_profiles,
    save_profile,
    save_request,
)
from weather import ForecastDay, WeatherError, fetch_forecast

PROJECT_DIR = Path(__file__).parent
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
load_dotenv(PROJECT_DIR / ".env", override=False)


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def icon(name: str, class_name: str = "ui-icon") -> str:
    """Render a bundled, openly licensed icon without a network request."""
    path = PROJECT_DIR / "assets" / "icons" / f"{name}.svg"
    if not path.is_file():
        return ""
    return (
        f'<span class="{class_name}" aria-hidden="true">'
        f"{path.read_text(encoding='utf-8')}</span>"
    )


def start_quiz(profiles: list[dict], profile: dict | None = None) -> None:
    st.session_state.quiz_mode = True
    st.session_state.quiz_step = 0
    st.session_state.quiz_answers = (
        dict(profile["answers"]) if profile else default_answers()
    )
    st.session_state.quiz_profile_id = profile["id"] if profile else None
    if profile:
        st.session_state.quiz_name = profile["name"]
    elif profiles:
        st.session_state.quiz_name = f"Профиль {len(profiles) + 1}"
    else:
        st.session_state.quiz_name = "Мой профиль"
    st.session_state.pop("quiz_name_input", None)
    st.session_state.latest_result = None


def page_header(title: str, subtitle: str, *, quiz: bool = False) -> None:
    kind = " quiz-header" if quiz else ""
    kicker = "Ваши предпочтения" if quiz else "Персональный подбор"
    mark = icon("sparkle" if quiz else "sun", "page-mark-icon")
    st.markdown(
        f'<div class="page-header{kind}">'
        f'<div class="page-mark">{mark}</div>'
        '<div class="page-heading">'
        f'<span class="page-kicker">{kicker}</span>'
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def section_heading(title: str, detail: str = "", *, icon_name: str = "") -> None:
    detail_html = f"<span>{escape(detail)}</span>" if detail else ""
    heading_icon = icon(icon_name, "section-icon") if icon_name else ""
    st.markdown(
        '<div class="section-heading">'
        f'<h2>{heading_icon}{escape(title)}</h2>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def formatted_day(raw: str) -> str:
    try:
        return date.fromisoformat(raw).strftime("%d.%m.%Y")
    except ValueError:
        return raw


def weather_facts(day: ForecastDay) -> list[tuple[str, str, str, str]]:
    """Build the same weather facts for the forecast card and saved history."""
    facts = []
    if day.feels_like is not None:
        value = f"{day.feels_like:+g} °C"
        facts.append(("Ощущается", value, "sun", f"ощущается как {value}"))
    if day.temperature_min is not None and day.temperature_max is not None:
        low, high = day.temperature_min, day.temperature_max
        facts.append(
            (
                "Диапазон",
                f"{low:+g}…{high:+g} °C",
                "",
                f"от {low:+g} до {high:+g} °C",
            )
        )

    chance = day.precipitation_probability
    metrics = (
        ("Ветер", day.wind_speed, " м/с", "wind", "ветер до"),
        ("Порывы", day.wind_gust, " м/с", "wind", "порывы до"),
        ("Влажность", day.humidity, "%", "", "влажность"),
        (
            "Осадки",
            round(chance * 100) if chance is not None else None,
            "%",
            "cloud-rain",
            "вероятность осадков до",
        ),
        ("Дождь", day.rain_mm, " мм", "cloud-rain", "дождь"),
        ("Снег", day.snow_mm, " мм", "cloud-rain", "снег"),
    )
    for label, amount, unit, icon_name, caption in metrics:
        if amount is None:
            continue
        value = f"{amount:g}{unit}"
        facts.append(
            (
                label,
                f"до {value}" if label == "Осадки" else value,
                icon_name,
                f"{caption} {value}",
            )
        )
    return facts


def forecast_details(day: ForecastDay) -> str:
    parts = [f"Около {day.observation_time}"]
    if day.condition:
        parts.append(day.condition)
    parts.extend(caption for _, _, _, caption in weather_facts(day))
    return " · ".join(parts)


def saved_forecast(day: dict) -> ForecastDay:
    saved = day["forecast"]
    values = {
        field.name: saved[field.name]
        for field in fields(ForecastDay)
        if field.name in saved
    }
    values.update(
        date=day["forecast_date"],
        temperature=day["temperature_c"],
        observation_time=day["observation_time"],
    )
    return ForecastDay(**values)


def weather_metric(label: str, value: str, icon_name: str = "") -> str:
    metric_icon = icon(icon_name, "metric-icon") if icon_name else ""
    return (
        '<div class="weather-metric">'
        f"{metric_icon}<span>{escape(label)}</span><strong>{escape(value)}</strong>"
        "</div>"
    )


def show_weather(day: ForecastDay) -> None:
    wet = bool(day.rain_mm or day.snow_mm) or (
        day.precipitation_probability is not None
        and day.precipitation_probability >= 0.4
    )
    weather_icon = icon("cloud-rain" if wet else "sun", "weather-icon")
    facts = [
        weather_metric(label, value, icon_name)
        for label, value, icon_name, _ in weather_facts(day)
    ]
    facts_html = (
        f'<div class="weather-metrics">{"".join(facts)}</div>' if facts else ""
    )
    st.markdown(
        '<section class="weather-panel">'
        '<div class="weather-lead">'
        f"{weather_icon}"
        '<div class="weather-headline">'
        f'<span class="weather-date">{icon("calendar-days", "date-icon")}'
        f"{escape(formatted_day(day.date))} · "
        f"около {escape(day.observation_time)}</span>"
        f'<div class="weather-temperature">{day.temperature:+g} °C</div>'
        '<div class="weather-condition">'
        f'{escape(day.condition or "Прогноз погоды")}</div>'
        "</div></div>"
        f"{facts_html}</section>",
        unsafe_allow_html=True,
    )


def show_outfit(option: OutfitOption) -> None:
    note = (
        '<div class="outfit-note"><span>Почему этот комплект</span>'
        f"<p>{escape(option.note)}</p></div>"
        if option.note
        else ""
    )
    st.markdown(
        '<article class="outfit-card">'
        '<div class="outfit-card-header">'
        f'{icon("shirt", "outfit-icon")}'
        '<div><span class="outfit-kicker">Рекомендация для вас</span>'
        f"<h3>{escape(option.title)}</h3></div>"
        "</div>"
        f'<p class="outfit-description">{escape(option.description)}</p>'
        f"{note}</article>",
        unsafe_allow_html=True,
    )


def show_quiz(db: sqlite3.Connection) -> None:
    step = st.session_state.quiz_step
    question = QUESTIONS[step]
    page_header(
        "Расскажите о себе",
        f"Ответьте на {len(QUESTIONS)} коротких вопросов, "
        "чтобы одежда подходила именно вам.",
        quiz=True,
    )

    stage_names = tuple(dict.fromkeys(item.section for item in QUESTIONS))
    stage = stage_names.index(question.section)
    stage_html = "".join(
        f'<span class="stage '
        f'{"active" if index == stage else "done" if index < stage else ""}">'
        f"{index + 1:02d} · {name}</span>"
        for index, name in enumerate(stage_names)
    )
    st.markdown(f'<div class="stages">{stage_html}</div>', unsafe_allow_html=True)
    st.progress(
        (step + 1) / len(QUESTIONS),
        text=f"Вопрос {step + 1} из {len(QUESTIONS)}",
    )

    values = [value for value, _ in question.choices]
    labels = dict(question.choices)
    with st.form("quiz_question"):
        selected = st.radio(
            question.prompt,
            values,
            index=values.index(st.session_state.quiz_answers[question.key]),
            format_func=lambda value: labels[value],
        )
        profile_name = st.session_state.quiz_name
        if step == len(QUESTIONS) - 1:
            profile_name = st.text_input(
                "Название профиля",
                value=profile_name,
                key="quiz_name_input",
                max_chars=50,
            )
        back_column, next_column = st.columns(2)
        back = back_column.form_submit_button(
            "Назад", disabled=step == 0, use_container_width=True
        )
        advance = next_column.form_submit_button(
            "Сохранить профиль" if step == len(QUESTIONS) - 1 else "Далее",
            type="primary",
            use_container_width=True,
        )

    if not (back or advance):
        st.caption(
            "Ваши ответы сохраняются в профиле и учитываются "
            "при каждом подборе. "
            "Имя профиля остаётся только в приложении."
        )
        return

    st.session_state.quiz_answers[question.key] = selected
    st.session_state.quiz_name = profile_name
    if back:
        st.session_state.quiz_step -= 1
        st.rerun()
    if step < len(QUESTIONS) - 1:
        st.session_state.quiz_step += 1
        st.rerun()

    try:
        profile_id = save_profile(
            db,
            st.session_state.quiz_name,
            st.session_state.quiz_answers,
            st.session_state.quiz_profile_id,
        )
    except sqlite3.IntegrityError:
        st.error("Профиль с таким названием уже существует.")
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state.quiz_mode = False
        st.session_state.active_profile_id = profile_id
        st.rerun()


def show_profile_controls(db: sqlite3.Connection, profiles: list[dict]) -> dict:
    section_heading("Ваш профиль", icon_name="user-round")
    profile_ids = [item["id"] for item in profiles]
    names = {item["id"]: item["name"] for item in profiles}
    if st.session_state.get("active_profile_id") not in profile_ids:
        st.session_state.active_profile_id = profile_ids[0]

    select_column, edit_column, add_column = st.columns(
        [2.4, 1, 1], vertical_alignment="bottom"
    )
    with select_column:
        profile_id = st.selectbox(
            "Кому подбираем одежду?",
            profile_ids,
            format_func=lambda value: names[value],
            key="active_profile_id",
        )
    profile = get_profile(db, profile_id)
    if profile is None:
        st.error("Профиль не найден. Обновите страницу и выберите другой.")
        st.stop()
    with edit_column:
        if st.button("Изменить ответы", use_container_width=True):
            start_quiz(profiles, profile)
            st.rerun()
    with add_column:
        if st.button("Новый профиль", use_container_width=True):
            start_quiz(profiles)
            st.rerun()
    with st.expander("Какие предпочтения учитываются"):
        st.caption(f"Подбор учитывает все {len(QUESTIONS)} ответов из анкеты.")
        for answer in answer_labels(profile["answers"]):
            st.write(f"• {answer}")
    if profile["needs_quiz"]:
        st.warning(
            "Этот профиль создан в старой версии или содержит неполную анкету. "
            "Обновите ответы перед подбором одежды."
        )
    return profile


def show_forecast_form(db: sqlite3.Connection, profile: dict) -> None:
    section_heading(
        "Прогноз", "Выберите город и число дней", icon_name="map-pin"
    )
    with st.form("forecast_form"):
        city_column, days_column = st.columns([3, 1])
        with city_column:
            city = st.text_input("Город", placeholder="Введите город")
        with days_column:
            count = st.selectbox("Дней прогноза", options=list(range(1, 6)), index=2)
        submitted = st.form_submit_button(
            "Подобрать комплект",
            type="primary",
            use_container_width=True,
            disabled=profile["needs_quiz"],
        )
    if not submitted:
        return
    try:
        ai_key = setting("OPENROUTER_API_KEY")
        if not ai_key:
            raise RecommendationError(
                "Сервис рекомендаций не настроен. "
                "Добавьте ключ доступа в настройках приложения."
            )
        with st.spinner("Получаем погоду и подбираем одежду…"):
            forecast = fetch_forecast(city, count, setting("OPENWEATHER_API_KEY"))
            generated = recommend_batch(
                forecast.city,
                forecast.days,
                ai_key,
                setting("OPENROUTER_MODEL", DEFAULT_MODEL),
                profile["answers"],
            )
            if len(generated) != len(forecast.days):
                raise RecommendationError(
                    "Не удалось подобрать одежду на все дни."
                )
            if any(not isinstance(option, OutfitOption) for option, _ in generated):
                raise RecommendationError(
                    "Не удалось подобрать один комплект на каждый день."
                )
            rows = [
                (day, option, method)
                for day, (option, method) in zip(forecast.days, generated)
            ]
            save_request(db, forecast.city, profile["name"], rows)
            st.session_state.latest_result = {
                "profile_id": profile["id"],
                "city": forecast.city,
                "requested_days": count,
                "weather_days": len(forecast.days),
                "rows": rows,
            }
    except (WeatherError, RecommendationError) as exc:
        st.error(str(exc))
    except (sqlite3.Error, OSError) as exc:
        st.error(f"Не удалось сохранить запрос: {exc}")


def show_latest_result(profile: dict) -> None:
    result = st.session_state.get("latest_result")
    if not result or result["profile_id"] != profile["id"]:
        return

    section_heading(
        f"Подбор: {result['city']}",
        "Один комплект на каждый день",
        icon_name="shirt",
    )
    if result.get("weather_days", len(result["rows"])) < result["requested_days"]:
        st.warning(
            f"Доступен прогноз только на {result['weather_days']} "
            f"из {result['requested_days']} выбранных дней."
        )
    tabs = st.tabs(
        [
            f"{formatted_day(day.date)} · {day.temperature:+g}°"
            for day, _, _ in result["rows"]
        ]
    )
    for tab, (day, option, _) in zip(tabs, result["rows"]):
        with tab:
            show_weather(day)
            show_outfit(option)


def show_history(db: sqlite3.Connection) -> None:
    section_heading("История", icon_name="history")
    with st.expander("Последние подборы"):
        history = last_requests(db)
        if not history:
            st.write("Пока нет сохранённых прогнозов.")
            return

        selected = st.selectbox(
            "Последние 10 запросов",
            range(len(history)),
            format_func=lambda index: (
                f"{history[index]['city']} · "
                f"{history[index]['requested_at'][:10]} · "
                f"{history[index]['profile_name'] or 'Без профиля'}"
            ),
        )
        for day in history[selected]["days"]:
            st.markdown(
                f"**{formatted_day(day['forecast_date'])} · "
                f"{day['temperature_c']:+g} °C**"
            )
            if day.get("forecast"):
                st.caption(forecast_details(saved_forecast(day)))
            option = day["option"]
            show_outfit(
                OutfitOption(
                    option["title"],
                    option["description"],
                    option.get("note", ""),
                )
            )


def main() -> None:
    st.set_page_config(
        page_title="Одежда по погоде",
        page_icon=str(PROJECT_DIR / "assets/icons/shirt.svg"),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    styles = (PROJECT_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

    database_path = Path(setting("DATABASE_PATH", "data/history.sqlite3"))
    if not database_path.is_absolute():
        database_path = PROJECT_DIR / database_path
    try:
        db = connect(database_path)
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Не удалось открыть базу данных: {exc}")
        st.stop()

    try:
        profiles = list_profiles(db)
        if "quiz_mode" not in st.session_state:
            st.session_state.quiz_mode = not bool(profiles)
            if st.session_state.quiz_mode:
                start_quiz(profiles)
        elif not profiles and not st.session_state.quiz_mode:
            start_quiz(profiles)

        if st.session_state.quiz_mode:
            show_quiz(db)
            return

        page_header(
            "Одежда по погоде",
            "Один подходящий комплект на каждый день "
            "с учётом прогноза и ваших привычек.",
        )
        profile = show_profile_controls(db, profiles)
        show_forecast_form(db, profile)
        show_latest_result(profile)
        show_history(db)
        st.markdown(
            '<p class="footer-note">Рекомендации учитывают температуру, осадки, '
            "ветер, влажность и ваши предпочтения, если эти данные есть "
            "в прогнозе.</p>",
            unsafe_allow_html=True,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
