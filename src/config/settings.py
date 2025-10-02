"""
Configuration centralisée via Pydantic, chargée depuis les variables d'environnement.
"""
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    DEFAULT_TASK_TYPE: str = "classification"
    DEFAULT_N_SPLITS: int = 5
    MAX_FILE_SIZE_MB: int = 1024

class TrainingSettings(BaseSettings):
    HIGH_MEMORY_THRESHOLD: float = 85.0
    HIGH_CPU_THRESHOLD: float = 90.0
    MIN_SAMPLES_REQUIRED: int = 10
    MAX_GRID_COMBINATIONS: int = 100
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42
    N_JOBS: int = -1
    MAX_TRAINING_TIME: int = 3600
    MAX_MODEL_FILES: int = 50
    MONITOR_TIME_THRESHOLD: int = 10
    MEMORY_LIMIT_MB: int = 2048

class MlflowSettings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT_NAME: str = "Default Experiment"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instanciation des configurations
app_settings = AppSettings()
training_settings = TrainingSettings()
mlflow_settings = MlflowSettings()