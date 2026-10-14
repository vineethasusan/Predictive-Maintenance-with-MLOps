# Serving image: FastAPI + trained model
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
COPY serve ./serve
# Train at build time so the image is self-contained (in production the model comes from the MLflow registry)
RUN python -m fleetpm.data --out data/telemetry.csv.gz && python -m fleetpm.train --data data/telemetry.csv.gz --out artifacts
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
