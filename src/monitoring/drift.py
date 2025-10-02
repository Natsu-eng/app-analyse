"""
Détection de dérive des données et du concept.
"""
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, Optional

from src.shared.logging import get_logger

logger = get_logger(__name__)

class DriftDetector:
    def __init__(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

    def detect_data_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Détecte la dérive des données entre la baseline et les données actuelles."""
        drift_report = {}
        common_cols = list(set(self.baseline_data.columns) & set(current_data.columns))

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(self.baseline_data[col]):
                try:
                    stat, p_value = ks_2samp(
                        self.baseline_data[col].dropna(), 
                        current_data[col].dropna()
                    )
                    if p_value < threshold:
                        drift_report[col] = {"drift_detected": True, "p_value": p_value, "statistic": stat}
                except Exception as e:
                    logger.warning(f"Could not perform KS test on {col}: {e}")
        
        logger.info(f"Data drift check completed. {len(drift_report)} columns drifted.")
        return drift_report
