FROM python:3.11-slim

WORKDIR /app

# System deps for XGBoost & scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset.csv      .
COPY ml_pipeline.py   .
COPY main.py          .
COPY train_and_save.py .

# If artifacts.pkl already exists (pre-built), it will be used at startup.
# Otherwise, the server trains from scratch on first boot (slower).
COPY artifacts.pkl* ./
COPY static/ ./static/
COPY templates/ ./templates/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
