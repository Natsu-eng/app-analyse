import logging
import logging.handlers
import os
from datetime import datetime
import sys

def setup_logging(
    log_file: str = "datalab_pro.log",
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    console_logging: bool = True
):
    """
    Configuration du système de logging pour la production, avec niveau configurable.
    
    Args:
        log_file: Nom du fichier de log
        max_file_size_mb: Taille maximale du fichier de log en MB
        backup_count: Nombre de fichiers de sauvegarde à conserver
        console_logging: Si True, affiche aussi les logs dans la console
    """
    
    # CORRECTION: Lire le niveau de log depuis une variable d'environnement
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Créer le dossier logs s'il n'existe pas
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file_path = os.path.join(log_dir, log_file)
    
    # Configuration du niveau de log
    numeric_level = getattr(logging, log_level_str, logging.INFO)
    
    # Format des logs
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Supprimer les handlers existants pour éviter les doublons
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler pour fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_file_size_mb * 1024 * 1024,  # Conversion en bytes
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Handler pour console (optionnel)
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        # En mode DEBUG, la console est aussi plus verbeuse
        console_level = logging.DEBUG if log_level_str == 'DEBUG' else logging.WARNING
        console_handler.setLevel(console_level)
        console_format = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
    
    # Configuration spécifique pour certains loggers tiers
    # Réduire la verbosité des librairies externes
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log de démarrage
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized - Level: {log_level_str} - File: {log_file_path}")
    logger.info(f"Application started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger avec le nom spécifié.
    
    Args:
        name: Nom du logger (généralement __name__)
    
    Returns:
        Instance de logger configuré
    """
    return logging.getLogger(name)

def log_system_info():
    """Log des informations système au démarrage."""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    try:
        # Informations système
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Architecture: {platform.architecture()[0]}")
        
        # Informations mémoire
        memory = psutil.virtual_memory()
        logger.info(f"Total memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
        
        # Informations CPU
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")

class PerformanceLogger:
    """
    Logger spécialisé pour les métriques de performance.
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Démarre le chronométrage d'une opération."""
        import time
        self.start_times[operation_name] = time.time()
        self.logger.info(f"Started operation: {operation_name}")
    
    def end_operation(self, operation_name: str, additional_info: str = ""):
        """Termine le chronométrage d'une opération."""
        import time
        
        if operation_name not in self.start_times:
            self.logger.warning(f"Operation {operation_name} was not started")
            return
        
        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        
        log_message = f"Completed operation: {operation_name} in {duration:.2f}s"
        if additional_info:
            log_message += f" - {additional_info}"
            
        self.logger.info(log_message)
        
        # Alerte si l'opération prend trop de temps
        if duration > 30:
            self.logger.warning(f"Slow operation detected: {operation_name} took {duration:.2f}s")
    
    def log_memory_usage(self, operation_name: str = ""):
        """Log de l'utilisation mémoire actuelle."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            message = f"Memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB used)"
            if operation_name:
                message = f"{operation_name} - {message}"
                
            self.logger.info(message)
            
            # Alerte si utilisation mémoire élevée
            if memory.percent > 85:
                self.logger.warning(f"High memory usage detected: {memory.percent:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to log memory usage: {e}")

def setup_error_logging():
    """
    Configure un handler spécial pour les erreurs critiques.
    """
    error_logger = get_logger("critical_errors")
    
    # Handler séparé pour les erreurs critiques
    error_file_path = os.path.join("logs", "critical_errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_file_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    
    error_format = logging.Formatter(
        fmt='%(asctime)s - CRITICAL ERROR - %(name)s - %(message)s\n%(pathname)s:%(lineno)d\n%(exc_info)s\n' + '-'*80,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_format)
    
    error_logger.addHandler(error_handler)
    error_logger.info("Critical error logging initialized")
    
    return error_logger