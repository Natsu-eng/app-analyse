"""
Constantes centralisées pour l'application de machine learning, organisées par catégorie.
Support des variables d'environnement via .env
"""

import os
from dotenv import load_dotenv

# Charger les variables depuis le fichier .env
load_dotenv()

# Constantes générales de l'application
APP_CONSTANTS = {
    "DEFAULT_TASK_TYPE": "classification",
    "DEFAULT_N_SPLITS": 3,
    "SUPPORTED_EXTENSIONS": {'csv', 'parquet', 'xlsx', 'xls', 'json'},
    "MAX_FILE_SIZE_MB": 1024,
    "MEMORY_WARNING_THRESHOLD": 85
}

# Constantes pour l'entraînement
TRAINING_CONSTANTS = {
    "MIN_SAMPLES_REQUIRED": 50,
    "MEMORY_LIMIT_MB": 4000,
    "MAX_FEATURES": 100,
    "MAX_TRAINING_TIME": 3600,
    "CV_FOLDS": 5,
    "RANDOM_STATE": 42,
    "N_JOBS": 1,
    "MAX_GRID_COMBINATIONS": 50,
    "MAX_VISUALIZATION_SAMPLES": 1000,
    "MAX_MODEL_FILES": 20,
    "HIGH_MEMORY_THRESHOLD": 85,
    "HIGH_CPU_THRESHOLD": 90,
    "MIN_NUMERIC_FEATURES": 2,
    "MAX_CLASSES": 100,
    "MAX_MISSING_RATIO": 0.5,
    "MONITOR_TIME_THRESHOLD": 10,
    "MONITOR_INTERVAL": 5,
    "MAX_MODELS": 4
}

# Constantes pour la validation des données
VALIDATION_CONSTANTS = {
    "MIN_ROWS_REQUIRED": 50,
    "MIN_COLS_REQUIRED": 2,
    "MAX_MISSING_RATIO": 0.5,
    "FEATURE_CORR_THRESHOLD": 0.3,
    "MEMORY_WARNING_THRESHOLD": 0.7,
    "MISSING_WARNING_THRESHOLD": 0.7,
    "MEMORY_MULTIPLIER": 3,
    "CLUSTERING_TIME_MULTIPLIER": 1.2,
    "CLASSIFICATION_TIME_MULTIPLIER": 1.5,
    "CLASSIFICATION_BASE_MULTIPLIER": 1.3,
    "REGRESSION_TIME_MULTIPLIER": 1.4,
    "DEFAULT_TIME_MULTIPLIER": 1.5,
    "OPTIMIZE_HP_MULTIPLIER": 3.5,
    "MAX_CATEGORICAL_UNIQUE": 20,
    "MIN_UNIQUE_VALUES": 10
}

# Constantes pour les prétraitements
PREPROCESSING_CONSTANTS = {
    "NUMERIC_IMPUTATION_DEFAULT": "mean",
    "CATEGORICAL_IMPUTATION_DEFAULT": "most_frequent",
    "SMOTE_K_NEIGHBORS": 3,
    "SCALING_METHOD": "standard",
    "ENCODING_METHOD": "onehot",
}

# Constantes pour les visualisations
VISUALIZATION_CONSTANTS = {
    "MAX_SAMPLES": 1000,
    "TRAIN_SIZES": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    "PLOTLY_TEMPLATE": "plotly_white",
    "BAR_CHART_COLOR": "#3498db"
}

# Constantes pour la journalisation
LOGGING_CONSTANTS = {
    "LOG_FILE": "datalab_pro.log",
    "MAX_FILE_SIZE_MB": 10,
    "BACKUP_COUNT": 5,
    "CONSOLE_LOGGING": True,
    "LOG_DIR": "logs",
    "MLFLOW_LOG_FILE": "mlflow.log",
    "SILENT_LIBRARIES": ['urllib3', 'requests', 'matplotlib', 'PIL', 'mlflow'],
    "SLOW_OPERATION_THRESHOLD": 30,
    "HIGH_MEMORY_THRESHOLD": 85,
    "MEMORY_LOG_INTERVAL": 5,
    "ERROR_LOG_FILE": "critical_errors.log",
    "ERROR_LOG_MAX_SIZE_MB": 5,
    "ERROR_LOG_BACKUP_COUNT": 3,
    "DEFAULT_LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
}

# Constantes MLflow (chargées depuis .env si dispo)
MLFLOW_CONSTANTS = {
    "AVAILABLE": False,  # Flag dynamique pour savoir si MLflow est installé
    "TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),  # fallback local
    "EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "datalab_pro_experimentations")
}

# Constantes DB Postgres (optionnelles si utilisées ailleurs)
DB_CONSTANTS = {
    "USER": os.getenv("POSTGRES_USER", "postgres"),
    "PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "DB": os.getenv("POSTGRES_DB", "mlflow_db"),
    "HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "PORT": int(os.getenv("POSTGRES_PORT", 5432)),
}
