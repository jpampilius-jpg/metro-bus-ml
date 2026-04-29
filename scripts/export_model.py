"""
Экспорт лучшей модели XGBoost (Optuna) из MLflow в standalone-файл.
Запуск: python scripts/export_model.py
"""

import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


def find_project_root(marker="PROJECT_CONTEXT.md"):
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Не найден корень проекта (нет файла {marker})")


def main():
    project_root = find_project_root()
    print(f"Корень проекта: {project_root}")
    
    # Подключение к MLflow
    tracking_uri = f"file:{(project_root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Поиск лучшего Optuna-run'а в первом эксперименте
    exp = client.get_experiment_by_name("metro_bus_uplift_v1")
    if exp is None:
        raise RuntimeError("Эксперимент metro_bus_uplift_v1 не найден")
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.runName = 'xgboost_optuna_tuned'",
        max_results=1,
    )
    
    if not runs:
        raise RuntimeError("Run xgboost_optuna_tuned не найден в metro_bus_uplift_v1")
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    print(f"\nНайден лучший run:")
    print(f"  Run ID: {run_id}")
    print(f"  MAE   : {best_run.data.metrics.get('mae'):.4f}")
    print(f"  RMSE  : {best_run.data.metrics.get('rmse'):.4f}")
    
    # Загрузка модели
    model_uri = f"runs:/{run_id}/model"
    print(f"\nЗагрузка модели из {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    
    # Создание папки models/
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Сохранение модели в JSON-формате (нативный XGBoost)
    model_path = models_dir / "xgboost_v1.json"
    model.save_model(str(model_path))
    print(f"\nМодель сохранена: {model_path}")
    print(f"Размер файла   : {model_path.stat().st_size / 1024:.1f} КБ")
    
    # Сохранение метаданных
    metadata = {
        "model_version":   "v1",
        "model_type":      "xgboost",
        "tuning":          "optuna_30_trials",
        "mlflow_run_id":   run_id,
        "metrics_on_test": {
            "mae":  best_run.data.metrics.get("mae"),
            "rmse": best_run.data.metrics.get("rmse"),
            "mape": best_run.data.metrics.get("mape"),
        },
        "hyperparameters": {
            k.replace("hp_", ""): v 
            for k, v in best_run.data.params.items() 
            if k.startswith("hp_")
        },
        "training_data": {
            "n_train":      int(best_run.data.params.get("n_train", 0)),
            "n_test":       int(best_run.data.params.get("n_test", 0)),
            "n_features":   int(best_run.data.params.get("n_features", 0)),
            "train_period": best_run.data.params.get("train_period", ""),
            "test_period":  best_run.data.params.get("test_period", ""),
        },
        "feature_names":   list(model.get_booster().feature_names) if model.get_booster().feature_names else [],
    }
    
    metadata_path = models_dir / "xgboost_v1_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Метаданные     : {metadata_path}")
    
    # Сохранение списка признаков отдельно (для API)
    feature_columns_path = models_dir / "feature_columns.json"
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(metadata["feature_names"], f, indent=2, ensure_ascii=False)
    print(f"Список фичей   : {feature_columns_path}")
    print(f"Признаков всего: {len(metadata['feature_names'])}")
    
    print("\nЭкспорт завершён. Модель готова к использованию в API.")


if __name__ == "__main__":
    main()