"""
Sauvegarde et chargement des modèles.
"""
import joblib
import os
from typing import Any

from src.shared.logging import get_logger

logger = get_logger(__name__)

def save_model(model: Any, path: str):
    """Sauvegarde un modèle sur le disque."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise

def load_model(path: str) -> Any:
    """Charge un modèle depuis le disque."""
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise
