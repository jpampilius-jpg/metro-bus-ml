"""
Просмотр SQLite-лога API-запросов.
Запуск: python scripts/check_predictions_log.py
"""

import json
import sqlite3
from pathlib import Path


def find_project_root(marker="PROJECT_CONTEXT.md"):
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Не найден корень проекта (нет файла {marker})")


def main():
    project_root = find_project_root()
    db_path = project_root / "data" / "monitoring" / "predictions.db"

    if not db_path.exists():
        print(f"База не найдена: {db_path}")
        print("Сделайте хотя бы один запрос к /predict через Swagger UI.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Общая статистика
    cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM predictions_log")
    total, first_ts, last_ts = cur.fetchone()

    print("=" * 70)
    print(f"SQLite-лог API: {db_path.relative_to(project_root)}")
    print("=" * 70)
    print(f"Всего запросов     : {total}")
    print(f"Первый запрос      : {first_ts}")
    print(f"Последний запрос   : {last_ts}")

    # Распределение по версиям модели
    cur.execute("SELECT model_version, COUNT(*) FROM predictions_log GROUP BY model_version")
    rows = cur.fetchall()
    print(f"\nЗапросы по версиям модели:")
    for ver, cnt in rows:
        print(f"  {ver}: {cnt}")

    # Распределение по числу предсказаний в ответе
    cur.execute("""
        SELECT 
            MIN(n_predictions) AS min_n,
            AVG(n_predictions) AS avg_n,
            MAX(n_predictions) AS max_n
        FROM predictions_log
    """)
    min_n, avg_n, max_n = cur.fetchone()
    print(f"\nРазмер ответа (число маршрутов в зоне):")
    print(f"  min: {min_n}, avg: {avg_n:.1f}, max: {max_n}")

    # Последние 5 запросов
    print(f"\nПоследние 5 запросов:")
    print("-" * 70)
    cur.execute("""
        SELECT id, timestamp, request_json, n_predictions, model_version
        FROM predictions_log
        ORDER BY id DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        req_id, ts, req_json, n_pred, ver = row
        req = json.loads(req_json)
        print(f"#{req_id} | {ts}")
        print(f"  Запрос  : lines={req.get('lines_affected')}, "
              f"hour={req.get('incident_hour')}, status={req.get('status_label')}")
        print(f"  Ответ   : {n_pred} маршрутов, модель {ver}")
        print()

    conn.close()


if __name__ == "__main__":
    main()