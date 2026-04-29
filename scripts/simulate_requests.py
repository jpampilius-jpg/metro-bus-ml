"""
Симуляция реалистичной нагрузки на API: 250 запросов с разными
параметрами инцидентов, чтобы наполнить SQLite-лог для дашборда.

Запуск (API должен быть запущен на http://localhost:8000):
    python scripts/simulate_requests.py
"""

import random
import time
from datetime import datetime, timedelta

import requests


API_URL = "http://127.0.0.1:8000/predict"
N_REQUESTS = 250

# Реалистичные сценарии для симуляции
LINES_POOL = [
    ["A", "C", "E"],         # West Side Manhattan + Brooklyn + Queens
    ["1", "2", "3"],         # Broadway-7th
    ["4", "5", "6"],         # Lexington Ave
    ["N", "Q", "R", "W"],    # Broadway BMT
    ["B", "D", "F", "M"],    # 6th Ave
    ["L"],                   # Только L
    ["G"],                   # Crosstown G
    ["7"],                   # Flushing
    ["J", "Z"],              # Nassau
    ["A"],                   # Только A
    ["F"],
    ["E", "F"],
    ["6"],
    ["N", "R"],
]

STATUSES = [
    "delays",
    "delays",
    "delays",
    "delays",         # delays — самый частый, дублируем для веса
    "severe-delays",
    "severe-delays",
    "part-suspended",
    "reroute",
    "stops-skipped",
    "express-to-local",
]

DURATIONS = [10, 15, 20, 30, 30, 30, 45, 60, 90, 120]  # минуты


def random_incident_hour():
    """Случайный час в пределах Q4 2024 (период обучения модели)."""
    base = datetime(2024, 10, 1, 0, 0, 0)
    days = random.randint(0, 91)
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])
    return (base + timedelta(days=days, hours=hour, minutes=minute)).isoformat()


def main():
    print(f"Симуляция: {N_REQUESTS} запросов к {API_URL}")
    print("=" * 60)

    # Сначала проверим, что API доступен
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        print(f"API health: {health}")
    except Exception as e:
        print(f"ОШИБКА: API недоступен. Проверь что сервис запущен.")
        print(f"  uvicorn src.api.main:app --port 8000")
        print(f"  ИЛИ docker run -d --name metro-bus-api -p 8000:8000 metro-bus-ml:v1")
        print(f"\nДетали: {e}")
        return

    successful = 0
    failed = 0
    t_start = time.time()

    for i in range(N_REQUESTS):
        payload = {
            "incident_hour": random_incident_hour(),
            "lines_affected": random.choice(LINES_POOL),
            "status_label": random.choice(STATUSES),
            "duration_min": random.choice(DURATIONS),
        }

        try:
            r = requests.post(API_URL, json=payload, timeout=10)
            if r.status_code == 200:
                successful += 1
            else:
                failed += 1
                print(f"  #{i+1}: HTTP {r.status_code}")
        except Exception as e:
            failed += 1
            print(f"  #{i+1}: ERROR {e}")

        # Прогресс каждые 25 запросов
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{N_REQUESTS}] успешно: {successful}, "
                  f"скорость: {rate:.1f} запросов/сек")

        # Лёгкая пауза, чтобы не задохнуть API
        time.sleep(0.05)

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Готово за {total_time:.1f} сек")
    print(f"Успешно: {successful}/{N_REQUESTS}")
    print(f"Ошибок : {failed}")


if __name__ == "__main__":
    main()