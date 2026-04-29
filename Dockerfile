# ============================================================
# Metro-Bus ML API — Dockerfile
# Базовый образ: Python 3.12 slim (минимальный размер)
# ============================================================

FROM python:3.12-slim

# Метаданные образа
LABEL maintainer="Yulia, УВИ-271"
LABEL description="ML API для прогнозирования прироста пассажиропотока на автобусных маршрутах при инцидентах в метро NYC"
LABEL version="1.0.0"

# Не создавать .pyc файлы, печатать stdout/stderr сразу
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Рабочая директория внутри контейнера
WORKDIR /app

# ============================================================
# Установка зависимостей (отдельным слоем для кеширования)
# ============================================================

# Сначала копируем только requirements — это позволит Docker
# переиспользовать слой с зависимостями при изменении кода
COPY requirements.txt .

# Устанавливаем зависимости production-сервиса
# Из полного requirements.txt берём только то, что нужно для API
# (pandas, xgboost, fastapi, uvicorn, pydantic)
RUN pip install --no-cache-dir \
    "fastapi==0.115.5" \
    "uvicorn==0.32.1" \
    "pydantic==2.10.2" \
    "pandas==2.2.3" \
    "numpy==1.26.4" \
    "pyarrow==18.0.0" \
    "xgboost==2.1.2" \
    "scikit-learn==1.5.2"

# ============================================================
# Копирование кода приложения и модели
# ============================================================

# Структура внутри /app:
#   /app/src/api/      — код API
#   /app/models/       — обученная модель и метаданные
#   /app/data/         — baseline-профиль (training_set_v1.parquet)
#   /app/PROJECT_CONTEXT.md  — маркер для find_project_root

COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/training_set_v1.parquet ./data/processed/
COPY PROJECT_CONTEXT.md ./

# Создаём папку для SQLite-лога мониторинга
RUN mkdir -p /app/data/monitoring

# ============================================================
# Сетевые настройки
# ============================================================

EXPOSE 8000

# Health check: каждые 30 секунд проверяем /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# ============================================================
# Запуск
# ============================================================

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]