"""
Streamlit-дашборд мониторинга API: визуализация SQLite-лога запросов.
Запуск:
    streamlit run src/monitoring/dashboard.py
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def find_project_root(marker="PROJECT_CONTEXT.md"):
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Не найден {marker}")


PROJECT_ROOT = find_project_root()
DB_PATH = PROJECT_ROOT / "data" / "monitoring" / "predictions.db"


# ============================================================
# Конфигурация страницы
# ============================================================

st.set_page_config(
    page_title="Metro-Bus ML — Monitoring",
    page_icon="🚌",
    layout="wide",
)

st.title("Мониторинг ML-сервиса прогнозирования пассажиропотока")
st.caption("Версия модели: v1 (XGBoost + Optuna, 32 признака, MAE на test = 10.75)")


# ============================================================
# Загрузка данных из SQLite
# ============================================================

@st.cache_data(ttl=10)
def load_logs() -> pd.DataFrame:
    """Загружает лог запросов из SQLite. Кешируется на 10 секунд."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, timestamp, request_json, response_json, n_predictions, model_version "
        "FROM predictions_log ORDER BY id ASC",
        conn,
    )
    conn.close()

    if df.empty:
        return df

    # Парсинг временных меток и JSON
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["request"] = df["request_json"].apply(json.loads)
    df["response"] = df["response_json"].apply(json.loads)

    # Извлечение полей запроса
    df["incident_hour"]   = df["request"].apply(lambda r: r.get("incident_hour"))
    df["lines_affected"]  = df["request"].apply(lambda r: tuple(r.get("lines_affected", [])))
    df["status_label"]    = df["request"].apply(lambda r: r.get("status_label"))
    df["duration_min"]    = df["request"].apply(lambda r: r.get("duration_min"))

    # Извлечение полей ответа
    df["boroughs_affected"] = df["response"].apply(
        lambda r: tuple(r.get("boroughs_affected", []))
    )

    # Производные поля
    df["incident_dt"] = pd.to_datetime(df["incident_hour"], errors="coerce")
    df["incident_hour_of_day"] = df["incident_dt"].dt.hour
    df["incident_day_of_week"] = df["incident_dt"].dt.day_name()
    df["lines_str"] = df["lines_affected"].apply(lambda lst: "/".join(lst) if lst else "—")
    df["request_date"] = df["timestamp"].dt.date

    return df


df = load_logs()

if df.empty:
    st.warning(
        "SQLite-лог пуст. Запусти симуляцию: `python scripts/simulate_requests.py`."
    )
    st.stop()


# ============================================================
# Верхний блок: KPI
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Всего запросов", f"{len(df):,}")

with col2:
    st.metric(
        "Уникальных версий модели",
        df["model_version"].nunique(),
        help=", ".join(df["model_version"].unique()),
    )

with col3:
    st.metric(
        "Среднее число маршрутов в зоне",
        f"{df['n_predictions'].mean():.0f}",
        help=f"min={df['n_predictions'].min()}, max={df['n_predictions'].max()}",
    )

with col4:
    period_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    if period_days > 0:
        rate = len(df) / max(period_days * 24, 0.001)
        st.metric("Запросов в час (среднее)", f"{rate:.1f}")
    else:
        st.metric("Запросов в час (среднее)", "—")


st.markdown("---")


# ============================================================
# График 1: распределение запросов по типу инцидента
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Распределение запросов по типу инцидента")
    status_counts = df["status_label"].value_counts().reset_index()
    status_counts.columns = ["status_label", "count"]

    fig1 = px.bar(
        status_counts,
        x="count",
        y="status_label",
        orientation="h",
        labels={"count": "Число запросов", "status_label": "Тип инцидента"},
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Распределение по часу инцидента")
    hour_counts = df["incident_hour_of_day"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["hour", "count"]

    fig2 = px.bar(
        hour_counts,
        x="hour",
        y="count",
        labels={"hour": "Час инцидента (NYC)", "count": "Число запросов"},
    )
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# График 2: топ комбинаций линий метро
# ============================================================

st.subheader("Топ-10 комбинаций затронутых линий метро")
lines_counts = df["lines_str"].value_counts().head(10).reset_index()
lines_counts.columns = ["lines", "count"]

fig3 = px.bar(
    lines_counts,
    x="count",
    y="lines",
    orientation="h",
    labels={"count": "Число запросов", "lines": "Линии метро"},
)
fig3.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# График 3: размер зоны влияния (число маршрутов)
# ============================================================

st.subheader("Распределение размера зоны влияния (маршрутов в ответе)")
fig4 = px.histogram(
    df,
    x="n_predictions",
    nbins=20,
    labels={"n_predictions": "Число маршрутов в зоне"},
)
fig4.update_yaxes(title_text="Частота")
fig4.update_layout(height=400, showlegend=False)
fig4.update_layout(height=400, showlegend=False)
st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# График 4: динамика запросов во времени
# ============================================================

st.subheader("Динамика запросов к API (время поступления)")
df["minute_bin"] = df["timestamp"].dt.floor("1min")
timeline = df.groupby("minute_bin").size().reset_index(name="count")

fig5 = px.line(
    timeline,
    x="minute_bin",
    y="count",
    labels={"minute_bin": "Время", "count": "Запросов в минуту"},
    markers=True,
)
fig5.update_layout(height=350)
st.plotly_chart(fig5, use_container_width=True)


# ============================================================
# Таблица последних запросов
# ============================================================

st.subheader("Последние 15 запросов")
display_cols = [
    "id", "timestamp", "incident_hour", "lines_str",
    "status_label", "duration_min", "n_predictions", "model_version",
]
recent = df[display_cols].tail(15).iloc[::-1]
recent.columns = [
    "ID", "Время запроса", "Время инцидента", "Линии",
    "Статус", "Длительность", "Маршрутов", "Модель",
]
st.dataframe(recent, use_container_width=True, hide_index=True)


# ============================================================
# Подвал
# ============================================================

st.markdown("---")
st.caption(
    f"Источник данных: `{DB_PATH.relative_to(PROJECT_ROOT)}`. "
    f"Дашборд обновляется каждые 10 секунд."
)