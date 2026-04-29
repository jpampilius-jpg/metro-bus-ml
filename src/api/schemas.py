"""
Pydantic-схемы для FastAPI: валидация запросов и ответов.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# Запрос: описание инцидента в метро
# ============================================================

class IncidentRequest(BaseModel):
    """Описание инцидента, по которому делается прогноз."""
    
    incident_hour: str = Field(
        ...,
        description="Время начала инцидента в формате ISO 8601",
        examples=["2024-12-15T08:30:00"],
    )
    lines_affected: List[str] = Field(
        ...,
        description="Список затронутых линий метро NYC",
        examples=[["A", "C", "E"]],
    )
    status_label: str = Field(
        default="delays",
        description="Тип инцидента (delays, severe-delays, part-suspended, reroute и т.д.)",
        examples=["delays"],
    )
    duration_min: Optional[int] = Field(
        default=30,
        description="Предполагаемая длительность инцидента в минутах",
        examples=[30],
        ge=1,
        le=480,
    )


# ============================================================
# Ответ: прогноз uplift по каждому маршруту в зоне
# ============================================================

class RoutePrediction(BaseModel):
    """Прогноз для одного автобусного маршрута."""
    
    bus_route: str = Field(..., description="Идентификатор маршрута")
    route_borough: str = Field(..., description="Borough маршрута")
    baseline_t1: float = Field(..., description="Базовый поток (исторический средний)")
    predicted_uplift: float = Field(..., description="Прогноз прироста потока")
    predicted_total: float = Field(..., description="Итоговый прогноз потока (baseline + uplift)")


class PredictionResponse(BaseModel):
    """Ответ API с прогнозом по всем маршрутам в зоне влияния."""
    
    incident_hour: str
    lines_affected: List[str]
    boroughs_affected: List[str]
    n_routes_in_zone: int
    model_version: str
    predictions: List[RoutePrediction]


# ============================================================
# Health-check и model info
# ============================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    tuning: str
    metrics_on_test: dict
    hyperparameters: dict
    training_data: dict
    n_features: int