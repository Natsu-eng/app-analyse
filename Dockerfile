# Dockerfile optimisé pour l'application Streamlit + ML
FROM python:3.11-slim

LABEL maintainer="ML Analytics Team"
      description="Streamlit ML Analytics Application"
      version="2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update && apt-get install -y build-essential curl git && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement le code source de l'application
COPY src/ ./src

RUN mkdir -p logs mlflow_artifacts models reports && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Mettre à jour le point d'entrée pour le nouveau chemin
CMD ["streamlit", "run", "src/app/main.py"]