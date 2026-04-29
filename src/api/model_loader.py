"""
Загрузка модели и подготовка признаков для предсказания.
Поддержка нескольких версий модели через параметр model_version.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


class ModelLoader:
    """Загружает обученную модель XGBoost и готовит признаки для инференса."""

    LINE_TO_BOROUGHS = {
        "1": ["Manhattan", "Bronx"],
        "2": ["Manhattan", "Bronx", "Brooklyn"],
        "3": ["Manhattan", "Brooklyn"],
        "4": ["Manhattan", "Bronx", "Brooklyn"],
        "5": ["Manhattan", "Bronx", "Brooklyn"],
        "6": ["Manhattan", "Bronx"],
        "7": ["Manhattan", "Queens"],
        "A": ["Manhattan", "Brooklyn", "Queens"],
        "B": ["Manhattan", "Bronx", "Brooklyn"],
        "C": ["Manhattan", "Brooklyn"],
        "D": ["Manhattan", "Bronx", "Brooklyn"],
        "E": ["Manhattan", "Queens"],
        "F": ["Manhattan", "Brooklyn", "Queens"],
        "G": ["Brooklyn", "Queens"],
        "J": ["Manhattan", "Brooklyn", "Queens"],
        "L": ["Manhattan", "Brooklyn"],
        "M": ["Manhattan", "Brooklyn", "Queens"],
        "N": ["Manhattan", "Brooklyn", "Queens"],
        "Q": ["Manhattan", "Brooklyn"],
        "R": ["Manhattan", "Brooklyn", "Queens"],
        "W": ["Manhattan", "Queens"],
        "Z": ["Manhattan", "Brooklyn", "Queens"],
        "S": ["Manhattan", "Brooklyn"],
    }

    SUPPORTED_VERSIONS = ("v1", "v2")

    def __init__(self, project_root: Path, model_version: str = "v1"):
        if model_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Неизвестная версия модели: {model_version}. "
                f"Допустимые: {self.SUPPORTED_VERSIONS}"
            )

        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.model_version = model_version

        self.model: xgb.Booster = None
        self.feature_columns: List[str] = []
        self.metadata: dict = {}
        self.baseline_profile: pd.DataFrame = None
        self.route_borough_map: dict = {}

        self._load_model()
        self._load_baseline_profile()

    def _load_model(self) -> None:
        """Загружает XGBoost-модель версии self.model_version и метаданные."""
        if self.model_version == "v1":
            model_filename = "xgboost_v1.json"
            metadata_filename = "xgboost_v1_metadata.json"
            features_filename = "feature_columns.json"
        else:  # v2
            model_filename = "xgboost_v2.json"
            metadata_filename = "xgboost_v2_metadata.json"
            features_filename = "feature_columns_v2.json"

        model_path = self.models_dir / model_filename
        metadata_path = self.models_dir / metadata_filename
        features_path = self.models_dir / features_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Не найден файл модели: {model_path}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        print(f"[ModelLoader] Модель загружена: {model_path.name} (версия {self.model_version})")
        print(f"[ModelLoader] Признаков: {len(self.feature_columns)}")

    def _load_baseline_profile(self) -> None:
        """Загружает базовый профиль (route × dow × hour) из training set."""
        training_set_path = self.project_root / "data" / "processed" / "training_set_v1.parquet"

        if not training_set_path.exists():
            print(f"[ModelLoader] WARNING: training_set не найден")
            return

        df = pd.read_parquet(training_set_path)

        df["dow"] = df["incident_hour"].dt.dayofweek
        df["hour"] = df["incident_hour"].dt.hour

        self.baseline_profile = (
            df.groupby(["bus_route", "dow", "hour"])
              .agg(
                  baseline_t1=("baseline_t1", "mean"),
                  baseline_t0=("baseline_t0", "mean"),
                  actual_t0=("actual_t0", "mean"),
              )
              .reset_index()
        )

        if "route_borough" in df.columns:
            route_borough = df.groupby("bus_route")["route_borough"].first().to_dict()
        else:
            route_borough = {}

        if not route_borough:
            unique_routes = df["bus_route"].unique()
            for r in unique_routes:
                prefix = "".join(c for c in str(r) if c.isalpha())
                if prefix.startswith("BX"):
                    route_borough[r] = "Bronx"
                elif prefix.startswith("M"):
                    route_borough[r] = "Manhattan"
                elif prefix.startswith("B"):
                    route_borough[r] = "Brooklyn"
                elif prefix.startswith("Q"):
                    route_borough[r] = "Queens"
                elif prefix.startswith("S"):
                    route_borough[r] = "StatenIsland"
                else:
                    route_borough[r] = "Unknown"

        self.route_borough_map = route_borough
        print(f"[ModelLoader] Baseline-профиль: {len(self.baseline_profile):,} записей")
        print(f"[ModelLoader] Уникальных маршрутов: {self.baseline_profile['bus_route'].nunique()}")

    def get_affected_boroughs(self, lines_affected: List[str]) -> List[str]:
        boroughs = set()
        for line in lines_affected:
            if line in self.LINE_TO_BOROUGHS:
                boroughs.update(self.LINE_TO_BOROUGHS[line])
        return sorted(boroughs)

    def build_features(
        self,
        incident_dt: datetime,
        lines_affected: List[str],
        status_label: str,
        duration_min: int,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Собирает DataFrame признаков для всех маршрутов в зоне влияния.
        Учитывает feature_columns текущей версии модели (v1=32, v2=22).
        """
        boroughs_affected = self.get_affected_boroughs(lines_affected)

        if self.baseline_profile is None:
            raise RuntimeError("Baseline-профиль не загружен")

        routes_in_zone = [
            r for r, b in self.route_borough_map.items()
            if b in boroughs_affected
        ]

        if not routes_in_zone:
            return pd.DataFrame(), []

        dow = incident_dt.weekday()
        hour = incident_dt.hour

        sub = self.baseline_profile[
            (self.baseline_profile["bus_route"].isin(routes_in_zone))
            & (self.baseline_profile["dow"] == dow)
            & (self.baseline_profile["hour"] == hour)
        ].copy().reset_index(drop=True)

        if sub.empty:
            return pd.DataFrame(), []

        # Признаки инцидента (все, что нужны для v1; для v2 лишние будут отброшены при reindex)
        sub["hour"] = hour
        sub["day_of_week"] = dow
        sub["is_weekend"] = int(dow >= 5)
        sub["month"] = incident_dt.month
        sub["day_of_month"] = incident_dt.day
        sub["num_lines_affected"] = len(lines_affected)
        sub["n_boroughs_affected"] = len(boroughs_affected)

        sub["is_express"] = sub["bus_route"].str.contains(r"\+$|X$", regex=True, na=False).astype(int)
        sub["route_in_zone"] = 1
        sub["route_borough"] = sub["bus_route"].map(self.route_borough_map)

        if hour < 6:
            time_of_day = "night"
        elif hour < 10:
            time_of_day = "morning_rush"
        elif hour < 16:
            time_of_day = "midday"
        elif hour < 20:
            time_of_day = "evening_rush"
        else:
            time_of_day = "evening"
        sub["time_of_day"] = time_of_day

        sub["status_main"] = status_label.split()[0] if status_label else "unknown"

        cat_cols = ["time_of_day", "status_main", "route_borough"]
        sub_encoded = pd.get_dummies(sub, columns=cat_cols, drop_first=False)

        # Подгонка под feature_columns модели (v1 или v2):
        # лишние колонки удаляются, недостающие добавляются нулями
        for col in self.feature_columns:
            if col not in sub_encoded.columns:
                sub_encoded[col] = 0

        X = sub_encoded[self.feature_columns].astype(float).reset_index(drop=True)
        route_list = sub["bus_route"].tolist()

        return X, route_list

    def predict(
        self,
        incident_dt: datetime,
        lines_affected: List[str],
        status_label: str,
        duration_min: int,
    ) -> List[dict]:
        X, routes = self.build_features(incident_dt, lines_affected, status_label, duration_min)

        if X.empty:
            return []

        dmatrix = xgb.DMatrix(X.values, feature_names=self.feature_columns)
        preds = self.model.predict(dmatrix)

        results = []
        for i, route in enumerate(routes):
            bt1 = self._get_baseline_t1(route, incident_dt)
            uplift = float(preds[i])
            results.append({
                "bus_route": route,
                "route_borough": self.route_borough_map.get(route, "Unknown"),
                "baseline_t1": round(bt1, 2),
                "predicted_uplift": round(uplift, 2),
                "predicted_total": round(bt1 + uplift, 2),
            })

        results.sort(key=lambda r: abs(r["predicted_uplift"]), reverse=True)
        return results

    def _get_baseline_t1(self, route: str, incident_dt: datetime) -> float:
        sub = self.baseline_profile[
            (self.baseline_profile["bus_route"] == route)
            & (self.baseline_profile["dow"] == incident_dt.weekday())
            & (self.baseline_profile["hour"] == incident_dt.hour)
        ]
        if sub.empty:
            return 0.0
        return float(sub["baseline_t1"].iloc[0])