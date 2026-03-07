# Production-ready Dockerfile for FastAPI + Uvicorn on Python 3.10
FROM python:3.10-slim

# Ensure Python output is unbuffered and no pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System deps (optional minimal)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

# [FIX-6] EXPOSE is documentation-only and may mislead on platforms like Render where $PORT is injected at runtime.
# [FIX-6] The command below already binds to ${PORT:-8000}, which is correct. We remove EXPOSE to avoid implying a fixed port.
# [FIX-6] If desired, one could write `EXPOSE ${PORT:-8000}`, but ENV substitution in EXPOSE isn't portable across Docker versions.
# EXPOSE removed intentionally.

# Command: bind to 0.0.0.0 and use Render's PORT env var with default 8000
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
