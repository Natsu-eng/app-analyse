import logging
import logging.handlers
import os
import sys
import platform
import psutil
from datetime import datetime
from typing import Optional
import traceback
from src.config.constants import LOGGING_CONSTANTS, MLFLOW_CONSTANTS  


def setup_logging(
    log_file: str = None,
    max_file_size_mb: int = None,
    backup_count: int = None,
    console_logging: bool = None,
    mlflow_integration: bool = False
):
    """
    Configuration du système de journalisation pour la production, avec niveau configurable et support MLflow.
    """

    # Lecture des constantes avec valeurs par défaut
    log_file = log_file or LOGGING_CONSTANTS["LOG_FILE"]
    max_file_size_mb = max_file_size_mb or LOGGING_CONSTANTS["MAX_FILE_SIZE_MB"]
    backup_count = backup_count or LOGGING_CONSTANTS["BACKUP_COUNT"]
    console_logging = (
        console_logging
        if console_logging is not None
        else LOGGING_CONSTANTS["CONSOLE_LOGGING"]
    )

    # Lire le niveau de journalisation depuis .env (via constants)
    log_level_str = LOGGING_CONSTANTS["DEFAULT_LOG_LEVEL"].upper()

    # Créer le dossier logs s'il n'existe pas
    log_dir = LOGGING_CONSTANTS["LOG_DIR"]
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, log_file)

    # Configuration du niveau de journalisation
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # Format des journaux
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Supprimer les gestionnaires existants pour éviter les doublons
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Gestionnaire de fichier avec rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.error(f"Échec de la configuration du gestionnaire de fichier : {str(e)}")
        raise

    # Gestionnaire de console (optionnel)
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = logging.DEBUG if log_level_str == "DEBUG" else logging.WARNING
        console_handler.setLevel(console_level)
        console_format = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

    # Configuration MLflow (si activé)
    if mlflow_integration:
        try:
            import mlflow

            if not MLFLOW_CONSTANTS["AVAILABLE"]:
                MLFLOW_CONSTANTS["AVAILABLE"] = True

            # Vérifier l'encodage de l'URI et du nom de l'expérience
            tracking_uri = MLFLOW_CONSTANTS["TRACKING_URI"]
            experiment_name = MLFLOW_CONSTANTS["EXPERIMENT_NAME"]
            if not tracking_uri or not experiment_name:
                raise ValueError("MLFLOW_TRACKING_URI ou MLFLOW_EXPERIMENT_NAME non défini")

            # Forcer l'encodage UTF-8 pour l'URI
            try:
                tracking_uri = tracking_uri.encode('utf-8').decode('utf-8')
                experiment_name = experiment_name.encode('utf-8').decode('utf-8')
            except UnicodeEncodeError as e:
                root_logger.error(f"Erreur d'encodage dans la configuration MLflow : {str(e)}")
                raise

            mlflow.set_tracking_uri(tracking_uri)

            # Créer l’expérience si elle n’existe pas
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)

            mlflow_logger = logging.getLogger("mlflow")
            mlflow_logger.setLevel(numeric_level)
            mlflow_handler = logging.handlers.RotatingFileHandler(
                filename=os.path.join(log_dir, LOGGING_CONSTANTS["MLFLOW_LOG_FILE"]),
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
            mlflow_handler.setFormatter(log_format)
            mlflow_logger.addHandler(mlflow_handler)

            root_logger.info(
                f"Journalisation MLflow initialisée avec l'URI : {tracking_uri} et l'expérience : {experiment_name}"
            )

        except ImportError:
            root_logger.warning(
                "MLflow non installé, configuration de la journalisation MLflow ignorée"
            )
        except Exception as e:
            root_logger.error(f"Échec de la configuration de la journalisation MLflow : {str(e)}")
            root_logger.debug(f"Détail de l'erreur : {traceback.format_exc()}")

def get_logger(name: str) -> logging.Logger:
    """Récupère un logger configuré."""
    return logging.getLogger(name)


def log_system_info():
    """Journalise les informations système au démarrage."""
    logger = get_logger(__name__)

    try:
        logger.info(f"Système : {platform.system()} {platform.release()}")
        logger.info(f"Version Python : {platform.python_version()}")
        logger.info(f"Architecture : {platform.architecture()[0]}")

        memory = psutil.virtual_memory()
        logger.info(f"Mémoire totale : {memory.total / (1024**3):.1f} Go")
        logger.info(f"Mémoire disponible : {memory.available / (1024**3):.1f} Go")

        logger.info(f"Cœurs CPU : {psutil.cpu_count()}")

    except Exception as e:
        logger.warning(f"Échec de la journalisation des informations système : {e}")


class PerformanceLogger:
    """Logger spécialisé pour les métriques de performance."""

    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.start_times = {}
        self.last_memory_log = 0

    def start_operation(self, operation_name: str):
        """Démarre le chronométrage d'une opération."""
        import time

        self.start_times[operation_name] = time.time()
        self.logger.info(f"Début de l'opération : {operation_name}")

    def end_operation(self, operation_name: str, additional_info: str = ""):
        """Termine le chronométrage d'une opération."""
        import time

        if operation_name not in self.start_times:
            self.logger.warning(f"L'opération {operation_name} n'a pas été démarrée")
            return

        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]

        log_message = f"Opération terminée : {operation_name} en {duration:.2f}s"
        if additional_info:
            log_message += f" - {additional_info}"

        self.logger.info(log_message)

        # Alerte si l'opération est lente
        if duration > LOGGING_CONSTANTS["SLOW_OPERATION_THRESHOLD"]:
            self.logger.warning(
                f"Opération lente détectée : {operation_name} a pris {duration:.2f}s"
            )

    def log_memory_usage(
        self,
        operation_name: str = "",
        interval: float = LOGGING_CONSTANTS["MEMORY_LOG_INTERVAL"],
    ):
        """Journalise l'utilisation mémoire actuelle, avec échantillonnage."""
        import time

        try:
            if time.time() - self.last_memory_log < interval:
                return

            memory = psutil.virtual_memory()
            message = f"Utilisation mémoire : {memory.percent:.1f}% ({memory.used / (1024**3):.1f} Go utilisés)"
            if operation_name:
                message = f"{operation_name} - {message}"

            self.logger.info(message)

            if memory.percent > LOGGING_CONSTANTS["HIGH_MEMORY_THRESHOLD"]:
                self.logger.warning(
                    f"Utilisation mémoire élevée détectée : {memory.percent:.1f}%"
                )

            self.last_memory_log = time.time()

        except Exception as e:
            self.logger.error(f"Échec de la journalisation de l'utilisation mémoire : {e}")


def setup_error_logging():
    """Configure un gestionnaire spécial pour les erreurs critiques."""
    error_logger = get_logger("erreurs_critiques")

    error_file_path = os.path.join(
        LOGGING_CONSTANTS["LOG_DIR"], LOGGING_CONSTANTS["ERROR_LOG_FILE"]
    )
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_file_path,
        maxBytes=LOGGING_CONSTANTS["ERROR_LOG_MAX_SIZE_MB"] * 1024 * 1024,
        backupCount=LOGGING_CONSTANTS["ERROR_LOG_BACKUP_COUNT"],
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)

    error_format = logging.Formatter(
        fmt="%(asctime)s - ERREUR CRITIQUE - %(name)s - %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s\n"
        + "-" * 80,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_handler.setFormatter(error_format)

    error_logger.addHandler(error_handler)
    error_logger.info("Journalisation des erreurs critiques initialisée")

    return error_logger