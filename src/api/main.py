"""
FastAPI-сервис для прогнозирования прироста пассажиропотока.
Запуск: uvicorn src.api.main:app --reload --port 8000
"""
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.model_loader import ModelLoader
from src.api.schemas import (
    HealthResponse,
    IncidentRequest,
    ModelInfoResponse,
    PredictionResponse,
    RoutePrediction,
)


def find_project_root(marker="PROJECT_CONTEXT.md"):
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Не найден корень проекта (нет файла {marker})")


PROJECT_ROOT = find_project_root()
DB_PATH = PROJECT_ROOT / "data" / "monitoring" / "predictions.db"

# Глобальный объект модели (загружается при старте приложения)
ml: ModelLoader = None


# ============================================================
# Lifespan: загрузка модели и инициализация БД при старте
# ============================================================

def init_database():
    """Создаёт SQLite-базу для логирования запросов, если её нет."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            request_json    TEXT NOT NULL,
            response_json   TEXT NOT NULL,
            n_predictions   INTEGER,
            model_version   TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(request_dict: dict, response_dict: dict, model_version: str):
    """Записывает запрос и ответ в SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions_log (timestamp, request_json, response_json, n_predictions, model_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                json.dumps(request_dict, ensure_ascii=False),
                json.dumps(response_dict, ensure_ascii=False),
                len(response_dict.get("predictions", [])),
                model_version,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[log_prediction] WARNING: не удалось записать в БД: {e}")


# ============================================================
# FastAPI-приложение
# ============================================================

app = FastAPI(
    title="Metro-Bus ML API",
    description="Прогнозирование прироста пассажиропотока на автобусных маршрутах при инцидентах в метро NYC",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    global ml
    print("[startup] Инициализация БД...")
    init_database()
    
    # Версия модели читается из переменной окружения
    model_version = os.environ.get("MODEL_VERSION", "v1")
    print(f"[startup] Загрузка модели версии {model_version}...")
    ml = ModelLoader(PROJECT_ROOT, model_version=model_version)
    print(f"[startup] Сервис готов к работе с моделью {model_version}")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    """Проверка живости сервиса и состояния модели."""
    return HealthResponse(
        status="ok",
        model_loaded=(ml is not None and ml.model is not None),
        model_version=ml.metadata.get("model_version", "unknown") if ml else "unknown",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["service"])
def model_info():
    """Возвращает метаданные модели: метрики, гиперпараметры, период обучения."""
    if ml is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return ModelInfoResponse(
        model_version=ml.metadata.get("model_version", "unknown"),
        model_type=ml.metadata.get("model_type", "unknown"),
        tuning=ml.metadata.get("tuning", "unknown"),
        metrics_on_test=ml.metadata.get("metrics_on_test", {}),
        hyperparameters=ml.metadata.get("hyperparameters", {}),
        training_data=ml.metadata.get("training_data", {}),
        n_features=len(ml.feature_columns),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: IncidentRequest):
    """
    Принимает описание инцидента и возвращает прогноз прироста пассажиропотока
    по всем автобусным маршрутам в зоне влияния.
    """
    if ml is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        incident_dt = datetime.fromisoformat(request.incident_hour)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат времени: {e}")
    
    boroughs_affected = ml.get_affected_boroughs(request.lines_affected)
    
    predictions = ml.predict(
        incident_dt=incident_dt,
        lines_affected=request.lines_affected,
        status_label=request.status_label,
        duration_min=request.duration_min,
    )
    
    response = PredictionResponse(
        incident_hour=request.incident_hour,
        lines_affected=request.lines_affected,
        boroughs_affected=boroughs_affected,
        n_routes_in_zone=len(predictions),
        model_version=ml.metadata.get("model_version", "unknown"),
        predictions=[RoutePrediction(**p) for p in predictions],
    )
    
    # Логирование в SQLite
    log_prediction(
        request_dict=request.model_dump(),
        response_dict=response.model_dump(),
        model_version=response.model_version,
    )
    
    return response


@app.get("/", tags=["service"])
def root():
    return {
        "service": "Metro-Bus ML API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }