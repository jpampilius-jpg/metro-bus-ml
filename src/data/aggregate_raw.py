"""
Агрегация сырых данных MTA до уровня, подходящего для ML.

Sub: station_complex × hour (суммируем по payment_method и fare_class)
Bus: bus_route × hour (суммируем по payment_method и fare_class)

Результат сохраняем в data/interim/ как parquet.

Запуск:
    python src/data/aggregate_raw.py
"""

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_subway():
    """Агрегируем метро: station_complex_id × час."""
    print("\n[Subway] Чтение сырых данных...")
    df = pd.read_parquet(RAW_DIR / "subway_ridership.parquet")
    print(f"  Исходно: {len(df):,} строк")

    # Приводим timestamp к datetime
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])

    # Переводим ridership в число
    df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce")
    df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")

    # Агрегируем
    print("  Агрегация по station_complex × час...")
    grouped = (
        df.groupby(
            ["transit_timestamp", "station_complex_id", "station_complex",
             "borough", "latitude", "longitude"],
            as_index=False,
        )
        .agg({"ridership": "sum", "transfers": "sum"})
    )
    grouped["latitude"] = pd.to_numeric(grouped["latitude"], errors="coerce")
    grouped["longitude"] = pd.to_numeric(grouped["longitude"], errors="coerce")

    output = INTERIM_DIR / "subway_hourly.parquet"
    grouped.to_parquet(output, compression="snappy")

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"  ✅ Сохранено: {output}")
    print(f"     После агрегации: {len(grouped):,} строк ({size_mb:.1f} МБ)")
    print(f"     Уникальных станций: {grouped['station_complex_id'].nunique()}")


def aggregate_bus():
    """Агрегируем автобусы: bus_route × час."""
    print("\n[Bus] Чтение сырых данных...")
    df = pd.read_parquet(RAW_DIR / "bus_ridership.parquet")
    print(f"  Исходно: {len(df):,} строк")

    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce")
    df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")

    print("  Агрегация по bus_route × час...")
    grouped = (
        df.groupby(["transit_timestamp", "bus_route"], as_index=False)
        .agg({"ridership": "sum", "transfers": "sum"})
    )

    output = INTERIM_DIR / "bus_hourly.parquet"
    grouped.to_parquet(output, compression="snappy")

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"  ✅ Сохранено: {output}")
    print(f"     После агрегации: {len(grouped):,} строк ({size_mb:.1f} МБ)")
    print(f"     Уникальных маршрутов: {grouped['bus_route'].nunique()}")


def main():
    print("=" * 70)
    print("Агрегация сырых данных MTA")
    print("=" * 70)

    aggregate_subway()
    aggregate_bus()

    print("\n" + "=" * 70)
    print("Агрегация завершена.")
    print("=" * 70)


if __name__ == "__main__":
    main()