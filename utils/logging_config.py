
import logging
import logging.config
import os

def setup_logging():
    """
    Configure le logging pour l'ensemble de l'application avec une configuration par défaut.
    Cette fonction doit être appelée une seule fois au démarrage de l'application.
    """
    log_file = 'logs/pipeline.log'
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'formatter': 'standard',
                'level': 'INFO',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'INFO',
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['file', 'console']
        }
    }
    
    try:
        logging.config.dictConfig(logging_config)
        logging.info("Logging configured successfully from hardcoded dictConfig.")
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"Failed to set up logging from dictConfig: {e}. Falling back to basicConfig.")

