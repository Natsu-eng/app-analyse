"""
Fonctions pour le monitoring des ressources système.
"""
import psutil
import time
from typing import Dict, Any

from src.shared.logging import get_logger

logger = get_logger(__name__)

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système actuelles."""
    try:
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {}
