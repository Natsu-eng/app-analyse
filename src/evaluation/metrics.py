"""
Module de calcul de métriques robuste pour l'évaluation des modèles ML.
Version Production - Complètement refactorée pour la robustesse
"""

from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    explained_variance_score, mean_squared_log_error
)
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import time
import gc
import logging
import json
from functools import wraps
from joblib import Parallel, delayed
import threading
from contextlib import contextmanager

# Imports conditionnels
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from src.config.constants import TRAINING_CONSTANTS, VALIDATION_CONSTANTS, LOGGING_CONSTANTS
except ImportError:
    # Fallback pour tests
    TRAINING_CONSTANTS = {
        "N_JOBS": -1,
        "RANDOM_STATE": 42
    }
    VALIDATION_CONSTANTS = {
        "MIN_ROWS_REQUIRED": 10,
        "MAX_CLASSES": 50,
        "MAX_MISSING_RATIO": 0.5,
        "MIN_COLS_REQUIRED": 1
    }
    LOGGING_CONSTANTS = {
        "DEFAULT_LOG_LEVEL": "INFO",
        "LOG_DIR": "logs",
        "LOG_FILE": "metrics.log",
        "CONSOLE_LOGGING": True,
        "SLOW_OPERATION_THRESHOLD": 30.0,
        "HIGH_MEMORY_THRESHOLD": 100.0
    }

# =============================
# CONFIGURATION LOGGING ROBUSTE
# =============================

class MetricsLogger:
    """Logger structuré et thread-safe pour les métriques."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging de façon robuste."""
        try:
            os.makedirs(LOGGING_CONSTANTS["LOG_DIR"], exist_ok=True)
            
            self.logger = logging.getLogger('metrics')
            self.logger.setLevel(getattr(logging, LOGGING_CONSTANTS["DEFAULT_LOG_LEVEL"]))
            
            # Éviter les handlers dupliqués
            if not self.logger.handlers:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
                )
                
                # Handler fichier
                file_handler = logging.FileHandler(
                    os.path.join(LOGGING_CONSTANTS["LOG_DIR"], LOGGING_CONSTANTS["LOG_FILE"])
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                # Handler console
                if LOGGING_CONSTANTS["CONSOLE_LOGGING"]:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
            
        except Exception as e:
            # Fallback basic
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('metrics_fallback')
    
    def log_structured(self, level: str, message: str, extra: Dict = None):
        """Log structuré thread-safe."""
        with self._lock:
            try:
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level.upper(),
                    "message": message,
                    "module": "metrics"
                }
                if extra:
                    log_data.update(extra)
                
                log_message = json.dumps(log_data, ensure_ascii=False, default=str)
                getattr(self.logger, level.lower())(log_message)
            except Exception as e:
                # Fallback ultra-robuste
                print(f"METRICS_ERROR: {message} - {str(e)}")

# Instance globale
METRICS_LOGGER = MetricsLogger()

def log_metrics(level: str, message: str, extra: Dict = None):
    """Interface de logging simplifiée."""
    METRICS_LOGGER.log_structured(level, message, extra)

# =============================
# GESTION D'ÉTAT ET SÉCURITÉ
# =============================

class MetricsStateManager:
    """Gestionnaire d'état pour les calculs de métriques."""
    
    def __init__(self):
        self._calculation_lock = threading.RLock()
        self._active_calculations = 0
        self._calculation_stats = {
            "total_calculations": 0,
            "failed_calculations": 0,
            "last_calculation_time": None
        }
    
    @contextmanager
    def calculation_context(self):
        """Context manager pour suivre les calculs."""
        with self._calculation_lock:
            self._active_calculations += 1
            self._calculation_stats["total_calculations"] += 1
            start_time = time.time()
            
            try:
                yield
            except Exception as e:
                self._calculation_stats["failed_calculations"] += 1
                raise
            finally:
                self._active_calculations -= 1
                self._calculation_stats["last_calculation_time"] = time.time() - start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de calcul."""
        with self._calculation_lock:
            return self._calculation_stats.copy()

# Instance globale
METRICS_STATE = MetricsStateManager()

# =============================
# DÉCORATEURS AVANCÉS
# =============================

def monitor_performance(func):
    """Décorateur de monitoring performant avec gestion d'erreurs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        with METRICS_STATE.calculation_context():
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                memory_delta = _get_memory_usage() - start_memory
                
                log_metrics("DEBUG", f"Performance {func.__name__}", {
                    "duration_s": round(duration, 4),
                    "memory_delta_mb": round(memory_delta, 2),
                    "success": True
                })
                
                # Alertes performances
                slow_threshold = LOGGING_CONSTANTS.get("SLOW_OPERATION_THRESHOLD", 30.0)
                memory_threshold = LOGGING_CONSTANTS.get("HIGH_MEMORY_THRESHOLD", 100.0)
                
                if duration > slow_threshold:
                    log_metrics("WARNING", f"Opération lente détectée", {
                        "function": func.__name__,
                        "duration_s": round(duration, 2),
                        "threshold_s": slow_threshold
                    })
                
                if memory_delta > memory_threshold:
                    log_metrics("WARNING", f"Usage mémoire élevé détecté", {
                        "function": func.__name__,
                        "memory_mb": round(memory_delta, 2),
                        "threshold_mb": memory_threshold
                    })
                    
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_metrics("ERROR", f"Erreur dans {func.__name__}", {
                    "error": str(e),
                    "duration_s": round(duration, 4),
                    "success": False
                })
                raise
    
    return wrapper

def safe_metric_calculation(fallback_value=None, max_retries=1):
    """Décorateur robuste pour calculs de métriques avec retry."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        log_metrics("WARNING", f"Tentative {attempt + 1} échouée, retry...", {
                            "function": func.__name__,
                            "error": str(e)
                        })
                        time.sleep(0.1)  # Backoff minimal
                    else:
                        log_metrics("ERROR", f"Échec après {max_retries + 1} tentatives", {
                            "function": func.__name__,
                            "error": str(e)
                        })
            
            return fallback_value
        return wrapper
    return decorator

# =============================
# FONCTIONS UTILITAIRES ROBUSTES
# =============================

@safe_metric_calculation(fallback_value=np.array([]))
def safe_array_conversion(data: Any, max_samples: int = 100000, sample: bool = True) -> np.ndarray:
    """
    Convertit les données en array numpy de façon ultra-robuste.
    """
    try:
        if data is None:
            log_metrics("WARNING", "Données None fournies à safe_array_conversion")
            return np.array([])
        
        # Conversion robuste selon le type
        if isinstance(data, pd.Series):
            result = data.values
        elif isinstance(data, pd.DataFrame):
            result = data.values.flatten() if data.shape[1] == 1 else data.values
        elif isinstance(data, list):
            result = np.array(data, dtype=object)
        elif isinstance(data, np.ndarray):
            result = data.copy()
        else:
            # Tentative générique
            result = np.array(data, dtype=object)
        
        # Nettoyage des données
        if hasattr(result, 'size') and result.size == 0:
            log_metrics("WARNING", "Tableau vide après conversion")
            return np.array([])
        
        # Échantillonnage intelligent
        if sample and hasattr(result, 'shape') and len(result) > max_samples:
            log_metrics("INFO", "Application échantillonnage", {
                "original_size": len(result),
                "max_samples": max_samples
            })
            
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            rng = np.random.RandomState(random_state)
            
            try:
                indices = rng.choice(len(result), size=max_samples, replace=False)
                result = result[indices]
            except Exception as e:
                log_metrics("WARNING", "Échec échantillonnage, prise des premiers échantillons", {
                    "error": str(e)
                })
                result = result[:max_samples]
        
        # Aplatissement si nécessaire
        if hasattr(result, 'ndim') and result.ndim > 1 and result.shape[1] == 1:
            result = result.flatten()
            
        return result
        
    except Exception as e:
        log_metrics("ERROR", "Échec critique conversion tableau", {
            "data_type": str(type(data)),
            "error": str(e)
        })
        return np.array([])

def _get_memory_usage() -> float:
    """Obtient l'utilisation mémoire en MB de façon robuste."""
    try:
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    except Exception:
        return 0.0

@safe_metric_calculation(fallback_value={})
def get_system_metrics() -> Dict[str, Any]:
    """
    Retourne les métriques système complètes.
    """
    try:
        if not PSUTIL_AVAILABLE:
            return {"psutil_available": False}
        
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "timestamp": time.time(),
            "active_calculations": METRICS_STATE._active_calculations,
            "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
            
        }
        
        log_metrics("DEBUG", "Métriques système collectées", metrics)
        return metrics
        
    except Exception as e:
        log_metrics("ERROR", "Échec collecte métriques système", {"error": str(e)})
        return {"error": str(e), "timestamp": time.time()}

# =============================
# VALIDATION DES DONNÉES ROBUSTE
# =============================

@monitor_performance
def validate_input_data(y_true: Any, y_pred: Any, task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entrée de façon exhaustive.
    """
    validation = {
        "is_valid": False,
        "issues": [],
        "warnings": [],
        "n_samples": 0,
        "task_type": task_type.lower().strip()
    }
    
    try:
        # Normalisation task_type
        if validation["task_type"] in ['unsupervised', 'cluster']:
            validation["task_type"] = 'clustering'
        
        # Conversion sécurisée
        y_true_flat = safe_array_conversion(y_true, sample=False)
        y_pred_flat = safe_array_conversion(y_pred, sample=False)
        
        # Validation de base
        if len(y_true_flat) == 0 or len(y_pred_flat) == 0:
            validation["issues"].append("Données vides après conversion")
            return validation
        
        if len(y_true_flat) != len(y_pred_flat):
            validation["issues"].append(
                f"Dimensions incohérentes: y_true={len(y_true_flat)}, y_pred={len(y_pred_flat)}"
            )
            return validation
        
        validation["n_samples"] = len(y_true_flat)
        
        # Validation spécifique au task_type
        if validation["task_type"] == "classification":
            _validate_classification_data(y_true_flat, y_pred_flat, validation)
        elif validation["task_type"] == "regression":
            _validate_regression_data(y_true_flat, y_pred_flat, validation)
        elif validation["task_type"] == "clustering":
            _validate_clustering_data(y_true_flat, y_pred_flat, validation)
        else:
            validation["issues"].append(f"Type de tâche non supporté: {validation['task_type']}")
            return validation
        
        # Validation de taille
        min_samples = VALIDATION_CONSTANTS.get("MIN_ROWS_REQUIRED", 2)
        if validation["n_samples"] < min_samples:
            validation["warnings"].append(
                f"Peu d'échantillons: {validation['n_samples']} < {min_samples}"
            )
        
        # Si pas d'issues critiques, validation réussie
        if not validation["issues"]:
            validation["is_valid"] = True
        
        log_metrics("DEBUG", "Validation données terminée", {
            "is_valid": validation["is_valid"],
            "n_issues": len(validation["issues"]),
            "n_warnings": len(validation["warnings"])
        })
        
    except Exception as e:
        log_metrics("ERROR", "Erreur critique validation données", {"error": str(e)})
        validation["issues"].append(f"Erreur validation: {str(e)}")
    
    return validation

def _validate_classification_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique classification."""
    try:
        unique_true = np.unique(y_true[~np.isnan(y_true)])
        unique_pred = np.unique(y_pred[~np.isnan(y_pred)])
        
        if len(unique_true) < 2:
            validation["issues"].append("Moins de 2 classes dans y_true")
        
        if len(unique_pred) < 2:
            validation["warnings"].append("Moins de 2 classes dans y_pred")
        
        max_classes = VALIDATION_CONSTANTS.get("MAX_CLASSES", 50)
        if len(unique_true) > max_classes:
            validation["warnings"].append(f"Trop de classes: {len(unique_true)} > {max_classes}")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation classification: {str(e)}")

def _validate_regression_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique régression."""
    try:
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            validation["warnings"].append("Valeurs infinies détectées")
        
        nan_count = np.sum(np.isnan(y_true)) + np.sum(np.isnan(y_pred))
        max_missing_ratio = VALIDATION_CONSTANTS.get("MAX_MISSING_RATIO", 0.5)
        
        if nan_count > len(y_true) * max_missing_ratio:
            validation["warnings"].append(f"Trop de NaN: {nan_count}/{len(y_true)} valeurs")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation régression: {str(e)}")

def _validate_clustering_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique clustering."""
    try:
        unique_labels = np.unique(y_pred[~np.isnan(y_pred)])
        
        if len(unique_labels) < 2 and -1 not in unique_labels:
            validation["warnings"].append("Moins de 2 clusters valides")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation clustering: {str(e)}")

# =============================
# CLASSE PRINCIPALE REFACTORISÉE
# =============================

class EvaluationMetrics:
    """
    Classe robuste et thread-safe pour calculer les métriques ML.
    Version Production avec gestion d'erreurs avancée.
    """
    
    def __init__(self, task_type: str):
        self.task_type = self._normalize_task_type(task_type)
        self.metrics = {}
        self.warnings = []
        self._calculation_lock = threading.RLock()
        
    def _normalize_task_type(self, task_type: str) -> str:
        """Normalise le type de tâche."""
        task_type = task_type.lower().strip()
        if task_type in ['unsupervised', 'cluster']:
            return 'clustering'
        if task_type not in ['classification', 'regression', 'clustering']:
            self.warnings.append(f"Type de tâche '{task_type}' non reconnu, utilisation classification par défaut")
            return 'classification'
        return task_type
    
    @safe_metric_calculation(fallback_value=None)
    def safe_metric_calculation(self, metric_func, *args, **kwargs) -> Any:
        """
        Calcule une métrique avec gestion d'erreurs robuste.
        """
        try:
            result = metric_func(*args, **kwargs)
            
            # Validation du résultat
            if result is None:
                raise ValueError("Résultat None")
            
            if np.isscalar(result):
                if np.isnan(result):
                    raise ValueError("Résultat NaN")
                if np.isinf(result):
                    raise ValueError("Résultat infini")
            
            return result
            
        except Exception as e:
            func_name = getattr(metric_func, '__name__', str(metric_func))
            warning_msg = f"Erreur calcul {func_name}: {str(e)}"
            self.warnings.append(warning_msg)
            log_metrics("WARNING", warning_msg)
            return None

    @monitor_performance
    def calculate_classification_metrics(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray, 
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calcule les métriques classification de façon robuste.
        """
        with self._calculation_lock:
            metrics = {
                "task_type": "classification",
                "success": False,
                "n_samples": len(y_true) if y_true is not None else 0,
                "warnings": self.warnings.copy()
            }
            
            try:
                # Validation
                validation = validate_input_data(y_true, y_pred, "classification")
                if not validation["is_valid"]:
                    metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                    metrics["warnings"].extend(validation["warnings"])
                    return metrics
                
                # Métriques de base
                metrics['accuracy'] = self.safe_metric_calculation(accuracy_score, y_true, y_pred)
                metrics['precision'] = self.safe_metric_calculation(
                    precision_score, y_true, y_pred, average='weighted', zero_division=0
                )
                metrics['recall'] = self.safe_metric_calculation(
                    recall_score, y_true, y_pred, average='weighted', zero_division=0
                )
                metrics['f1_score'] = self.safe_metric_calculation(
                    f1_score, y_true, y_pred, average='weighted', zero_division=0
                )
                
                # Rapport de classification
                try:
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    metrics['classification_report'] = report
                except Exception as e:
                    self.warnings.append(f"Erreur rapport classification: {str(e)}")
                
                # ROC-AUC si probabilités disponibles
                if y_proba is not None and len(y_proba) > 0:
                    n_classes = len(np.unique(y_true))
                    try:
                        if n_classes > 2:
                            metrics['roc_auc'] = self.safe_metric_calculation(
                                roc_auc_score, y_true, y_proba, multi_class='ovr', average='weighted'
                            )
                        else:
                            metrics['roc_auc'] = self.safe_metric_calculation(
                                roc_auc_score, y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                            )
                    except Exception as e:
                        self.warnings.append(f"ROC-AUC échoué: {str(e)}")
                
                # Matrice de confusion
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    metrics['confusion_matrix'] = cm.tolist()
                except Exception as e:
                    self.warnings.append(f"Matrice confusion échouée: {str(e)}")
                
                metrics['success'] = True
                metrics['warnings'] = self.warnings
                
                log_metrics("INFO", "Métriques classification calculées", {
                    "n_samples": metrics['n_samples'],
                    "accuracy": metrics.get('accuracy'),
                    "success": True
                })
                
            except Exception as e:
                log_metrics("ERROR", "Erreur critique calcul classification", {"error": str(e)})
                metrics['error'] = str(e)
                metrics['success'] = False
            
            return metrics

    @monitor_performance
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calcule les métriques régression de façon robuste.
        """
        with self._calculation_lock:
            metrics = {
                "task_type": "regression",
                "success": False,
                "n_samples": len(y_true) if y_true is not None else 0,
                "warnings": self.warnings.copy()
            }
            
            try:
                # Validation
                validation = validate_input_data(y_true, y_pred, "regression")
                if not validation["is_valid"]:
                    metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                    return metrics
                
                # Métriques de base
                metrics['mse'] = self.safe_metric_calculation(mean_squared_error, y_true, y_pred)
                metrics['mae'] = self.safe_metric_calculation(mean_absolute_error, y_true, y_pred)
                metrics['r2'] = self.safe_metric_calculation(r2_score, y_true, y_pred)
                metrics['explained_variance'] = self.safe_metric_calculation(
                    explained_variance_score, y_true, y_pred
                )
                
                # RMSE dérivé
                if metrics['mse'] is not None and metrics['mse'] >= 0:
                    metrics['rmse'] = np.sqrt(metrics['mse'])
                else:
                    metrics['rmse'] = None
                
                # MSLE conditionnel
                if (np.all(y_true > 0) and np.all(y_pred > 0) and 
                    not np.any(np.isinf(y_true)) and not np.any(np.isinf(y_pred))):
                    metrics['msle'] = self.safe_metric_calculation(mean_squared_log_error, y_true, y_pred)
                else:
                    metrics['msle'] = None
                    self.warnings.append("MSLE non calculé: valeurs non positives détectées")
                
                # Statistiques d'erreur
                try:
                    errors = np.abs(y_true - y_pred)
                    metrics['error_stats'] = {
                        'mean_error': float(np.nanmean(errors)),
                        'std_error': float(np.nanstd(errors)),
                        'max_error': float(np.nanmax(errors)),
                        'median_error': float(np.nanmedian(errors)),
                        'q95_error': float(np.nanpercentile(errors, 95))
                    }
                except Exception as e:
                    self.warnings.append(f"Statistiques erreur échouées: {str(e)}")
                
                metrics['success'] = True
                metrics['warnings'] = self.warnings
                
                log_metrics("INFO", "Métriques régression calculées", {
                    "n_samples": metrics['n_samples'],
                    "r2": metrics.get('r2'),
                    "success": True
                })
                
            except Exception as e:
                log_metrics("ERROR", "Erreur critique calcul régression", {"error": str(e)})
                metrics['error'] = str(e)
                metrics['success'] = False
            
            return metrics

    @monitor_performance
    def calculate_unsupervised_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calcule les métriques clustering de façon robuste.
        """
        with self._calculation_lock:
            metrics = {
                "task_type": "clustering",
                "success": False,
                "n_samples": len(X) if X is not None else 0,
                "warnings": self.warnings.copy()
            }
            
            try:
                # Validation
                validation = validate_input_data(X, labels, "clustering")
                if not validation["is_valid"]:
                    metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                    return metrics
                
                # Filtrage des points valides
                valid_mask = labels >= 0
                valid_labels = labels[valid_mask]
                valid_X = X[valid_mask]
                
                n_clusters = len(np.unique(valid_labels))
                metrics['n_clusters'] = n_clusters
                metrics['n_valid_samples'] = len(valid_labels)
                metrics['n_outliers'] = int(np.sum(labels == -1))
                
                # Métriques de clustering (seulement si conditions remplies)
                if n_clusters > 1 and len(valid_labels) >= n_clusters:
                    metrics['silhouette_score'] = self.safe_metric_calculation(
                        silhouette_score, valid_X, valid_labels
                    )
                    metrics['davies_bouldin_score'] = self.safe_metric_calculation(
                        davies_bouldin_score, valid_X, valid_labels
                    )
                    metrics['calinski_harabasz_score'] = self.safe_metric_calculation(
                        calinski_harabasz_score, valid_X, valid_labels
                    )
                else:
                    self.warnings.append(
                        f"Pas assez de clusters valides: {n_clusters} clusters, {len(valid_labels)} échantillons"
                    )
                
                # Distribution des clusters
                try:
                    cluster_sizes = np.bincount(valid_labels)
                    metrics['cluster_sizes'] = {
                        f"cluster_{i}": int(count) for i, count in enumerate(cluster_sizes)
                    }
                    metrics['cluster_size_stats'] = {
                        'min': int(np.min(cluster_sizes)),
                        'max': int(np.max(cluster_sizes)),
                        'mean': float(np.mean(cluster_sizes)),
                        'std': float(np.std(cluster_sizes))
                    }
                except Exception as e:
                    self.warnings.append(f"Distribution clusters échouée: {str(e)}")
                
                metrics['success'] = True
                metrics['warnings'] = self.warnings
                
                log_metrics("INFO", "Métriques clustering calculées", {
                    "n_samples": metrics['n_samples'],
                    "n_clusters": n_clusters,
                    "success": True
                })
                
            except Exception as e:
                log_metrics("ERROR", "Erreur critique calcul clustering", {"error": str(e)})
                metrics['error'] = str(e)
                metrics['success'] = False
            
            return metrics

# =============================
# FONCTIONS PRINCIPALES ROBUSTES
# =============================

@monitor_performance
def calculate_global_metrics(
    y_true_all: List[Any],
    y_pred_all: List[Any],
    y_proba_all: List[Any] = None,
    task_type: str = "classification",
    label_encoder: Any = None,
    X_data: Any = None,
    sample_metrics: bool = True,
    max_samples_metrics: int = 100000
) -> Dict[str, Any]:
    """
    Calcule les métriques globales de façon robuste et parallélisée.
    """
    start_time = time.time()
    
    metrics = {
        "task_type": task_type.lower().strip(),
        "success": False,
        "computation_time": 0,
        "warnings": [],
        "batch_processing": {
            "total_batches": len(y_true_all),
            "processed_batches": 0,
            "failed_batches": 0
        }
    }
    
    # Normalisation task_type
    if metrics["task_type"] in ['unsupervised', 'cluster']:
        metrics["task_type"] = 'clustering'
    
    try:
        def process_batch(i: int, y_true: Any, y_pred: Any, y_proba: Any = None) -> Tuple:
            """Traite un batch de données."""
            try:
                y_true_flat = safe_array_conversion(
                    y_true, max_samples=max_samples_metrics, sample=sample_metrics
                )
                y_pred_flat = safe_array_conversion(
                    y_pred, max_samples=max_samples_metrics, sample=sample_metrics
                )
                y_proba_flat = None
                
                if y_proba is not None:
                    y_proba_flat = safe_array_conversion(
                        y_proba, max_samples=max_samples_metrics, sample=sample_metrics
                    )
                
                return y_true_flat, y_pred_flat, y_proba_flat, True
                
            except Exception as e:
                log_metrics("WARNING", f"Échec traitement batch {i}", {"error": str(e)})
                return None, None, None, False
        
        # Traitement parallèle robuste
        batch_args = []
        for i, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):
            y_proba = y_proba_all[i] if y_proba_all and i < len(y_proba_all) else None
            batch_args.append((i, y_true, y_pred, y_proba))
        
        # Exécution parallèle avec gestion d'erreurs
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", 1)
        results = []
        
        if n_jobs == 1 or len(batch_args) == 1:
            # Mode séquentiel pour stabilité
            for args in batch_args:
                results.append(process_batch(*args))
        else:
            # Mode parallèle
            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_batch)(*args) for args in batch_args
                )
            except Exception as e:
                log_metrics("ERROR", "Échec parallélisme, fallback séquentiel", {"error": str(e)})
                results = [process_batch(*args) for args in batch_args]
        
        # Agrégation des résultats
        y_true_aggregated = []
        y_pred_aggregated = []
        y_proba_aggregated = []
        
        for y_true_flat, y_pred_flat, y_proba_flat, success in results:
            metrics["batch_processing"]["processed_batches"] += 1
            
            if not success:
                metrics["batch_processing"]["failed_batches"] += 1
                continue
            
            if (y_true_flat is not None and y_pred_flat is not None and 
                len(y_true_flat) == len(y_pred_flat) and len(y_true_flat) > 0):
                
                y_true_aggregated.extend(y_true_flat)
                y_pred_aggregated.extend(y_pred_flat)
                
                if y_proba_flat is not None:
                    if len(y_proba_aggregated) == 0:
                        y_proba_aggregated = y_proba_flat
                    else:
                        try:
                            y_proba_aggregated = np.vstack([y_proba_aggregated, y_proba_flat])
                        except Exception as e:
                            log_metrics("WARNING", "Échec empilement probabilités", {"error": str(e)})
        
        # Vérification données agrégées
        if len(y_true_aggregated) == 0:
            metrics["error"] = "Aucune donnée valide après agrégation"
            metrics["computation_time"] = time.time() - start_time
            return metrics
        
        # Conversion finale
        y_true_array = np.array(y_true_aggregated)
        y_pred_array = np.array(y_pred_aggregated)
        y_proba_array = np.array(y_proba_aggregated) if len(y_proba_aggregated) > 0 else None
        
        # Décodage des labels si encodeur disponible
        if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
            try:
                y_true_decoded = label_encoder.inverse_transform(y_true_array.astype(int))
                y_pred_decoded = label_encoder.inverse_transform(y_pred_array.astype(int))
            except Exception as e:
                log_metrics("WARNING", "Échec décodage labels", {"error": str(e)})
                y_true_decoded = y_true_array
                y_pred_decoded = y_pred_array
        else:
            y_true_decoded = y_true_array
            y_pred_decoded = y_pred_array
        
        # Calcul des métriques finales
        evaluator = EvaluationMetrics(metrics["task_type"])
        
        if metrics["task_type"] == "classification":
            final_metrics = evaluator.calculate_classification_metrics(
                y_true_decoded, y_pred_decoded, y_proba_array
            )
        elif metrics["task_type"] == "regression":
            final_metrics = evaluator.calculate_regression_metrics(y_true_decoded, y_pred_decoded)
        elif metrics["task_type"] == "clustering":
            if X_data is not None:
                X_flat = safe_array_conversion(X_data, max_samples=max_samples_metrics, sample=sample_metrics)
                if len(X_flat) == len(y_pred_array):
                    final_metrics = evaluator.calculate_unsupervised_metrics(X_flat, y_pred_array)
                else:
                    final_metrics = {"error": "Dimensions X et labels incohérentes"}
            else:
                final_metrics = {"error": "Données X requises pour clustering"}
        else:
            final_metrics = {"error": f"Type de tâche non supporté: {metrics['task_type']}"}
        
        # Fusion des résultats
        metrics.update(final_metrics)
        metrics["computation_time"] = time.time() - start_time
        
        if final_metrics.get("success", False):
            metrics["success"] = True
        
        log_metrics("INFO", "Métriques globales calculées", {
            "n_samples": metrics.get("n_samples", 0),
            "computation_time": metrics["computation_time"],
            "success": metrics["success"]
        })
        
    except Exception as e:
        log_metrics("ERROR", "Erreur critique calcul métriques globales", {"error": str(e)})
        metrics["error"] = str(e)
        metrics["computation_time"] = time.time() - start_time
    
    # Nettoyage mémoire
    gc.collect()
    
    return metrics

@safe_metric_calculation(
    fallback_value={"error": "Erreur évaluation", "success": False, "warnings": []}
)
@monitor_performance
def evaluate_single_train_test_split(
    model: Any,
    X_test: Any,
    y_test: Any,
    task_type: str = "classification",
    label_encoder: Any = None,
    sample_metrics: bool = True,
    max_samples_metrics: int = 100000
) -> Dict[str, Any]:
    """
    Évalue un modèle sur un jeu de test de façon robuste.
    """
    task_type = task_type.lower().strip()
    if task_type in ['unsupervised', 'cluster']:
        task_type = 'clustering'
    
    metrics = {
        "success": False, 
        "warnings": [],
        "task_type": task_type
    }
    
    try:
        # Validation des entrées
        if X_test is None:
            metrics["error"] = "X_test est None"
            return metrics
        
        if hasattr(X_test, 'size') and X_test.size == 0:
            metrics["error"] = "X_test est vide"
            return metrics
        
        # Échantillonnage intelligent
        if (sample_metrics and hasattr(X_test, 'shape') and 
            X_test.shape[0] > max_samples_metrics):
            
            log_metrics("INFO", "Application échantillonnage évaluation", {
                "original_size": X_test.shape[0],
                "max_samples": max_samples_metrics
            })
            
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            rng = np.random.RandomState(random_state)
            indices = rng.choice(X_test.shape[0], max_samples_metrics, replace=False)
            
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.iloc[indices]
            else:
                X_test = X_test[indices]
                
            if y_test is not None:
                if isinstance(y_test, pd.Series):
                    y_test = y_test.iloc[indices]
                else:
                    y_test = y_test[indices]
        
        # Prédictions selon le type de tâche
        if task_type == 'clustering':
            try:
                # Gestion spécifique par type de modèle
                model_str = str(type(model)).upper()
                
                if 'DBSCAN' in model_str:
                    y_pred = model.fit_predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                
                y_pred = np.asarray(y_pred)
                
                # Validation des prédictions clustering
                unique_clusters = np.unique(y_pred)
                n_clusters = len(unique_clusters[unique_clusters >= 0])  # Exclure les outliers (-1)
                
                if n_clusters < 2:
                    metrics["error"] = f"Seulement {n_clusters} cluster(s) valide(s) détecté(s)"
                    return metrics
                
                # Calcul des métriques clustering
                clustering_metrics = calculate_global_metrics(
                    [X_test], [y_pred], task_type="clustering",
                    sample_metrics=sample_metrics, max_samples_metrics=max_samples_metrics
                )
                
                metrics.update(clustering_metrics)
                
            except Exception as e:
                metrics["error"] = f"Erreur clustering: {str(e)}"
                return metrics
                
        else:
            # Classification ou régression
            try:
                y_pred = model.predict(X_test)
                y_proba = None
                
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception as e:
                        metrics["warnings"].append(f"predict_proba échoué: {str(e)}")
                
                # Calcul des métriques
                supervised_metrics = calculate_global_metrics(
                    [y_test], [y_pred], [y_proba] if y_proba is not None else [],
                    task_type=task_type, label_encoder=label_encoder,
                    sample_metrics=sample_metrics, max_samples_metrics=max_samples_metrics
                )
                
                metrics.update(supervised_metrics)
                
            except Exception as e:
                metrics["error"] = f"Erreur prédiction: {str(e)}"
                return metrics
        
        metrics["success"] = True
        
    except Exception as e:
        log_metrics("ERROR", "Erreur critique évaluation", {"error": str(e)})
        metrics["error"] = str(e)
    
    return metrics

# =============================
# FONCTIONS DE RAPPORT AVANCÉES
# =============================

@monitor_performance
def generate_evaluation_report(metrics: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    """Génère un rapport d'évaluation structuré."""
    report = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "status": "UNKNOWN",
            "message": "",
            "primary_metric": "",
            "primary_score": 0.0
        },
        "detailed_metrics": _sanitize_metrics_for_output(metrics),
        "recommendations": [],
        "warnings": metrics.get("warnings", [])
    }
    
    try:
        if metrics.get("error"):
            report["summary"]["status"] = "ERROR"
            report["summary"]["message"] = metrics["error"]
            return report
        
        if not metrics.get("success", False):
            report["summary"]["status"] = "FAILED"
            report["summary"]["message"] = "Calcul des métriques échoué"
            return report
        
        task_type = metrics.get("task_type", "classification")
        report["summary"]["task_type"] = task_type
        report["summary"]["n_samples"] = metrics.get("n_samples", 0)
        
        # Analyse selon le type de tâche
        if task_type == "classification":
            accuracy = metrics.get("accuracy", 0)
            report["summary"]["primary_metric"] = "accuracy"
            report["summary"]["primary_score"] = accuracy
            
            if accuracy > 0.9:
                report["summary"]["status"] = "EXCELLENT"
                report["recommendations"].append("Performance excellente - prêt pour la production")
            elif accuracy > 0.7:
                report["summary"]["status"] = "GOOD" 
                report["recommendations"].append("Bonne performance - utilisable en production")
            else:
                report["summary"]["status"] = "NEEDS_IMPROVEMENT"
                report["recommendations"].append("Performance modérée - envisager l'optimisation")
                
        elif task_type == "regression":
            r2 = metrics.get("r2", 0)
            report["summary"]["primary_metric"] = "r2"
            report["summary"]["primary_score"] = r2
            
            if r2 > 0.8:
                report["summary"]["status"] = "EXCELLENT"
                report["recommendations"].append("Très bon pouvoir prédictif")
            elif r2 > 0.5:
                report["summary"]["status"] = "GOOD"
                report["recommendations"].append("Performance acceptable")
            else:
                report["summary"]["status"] = "NEEDS_IMPROVEMENT"
                report["recommendations"].append("Faible pouvoir prédictif - revoir les features")
                
        elif task_type == "clustering":
            silhouette = metrics.get("silhouette_score", 0)
            report["summary"]["primary_metric"] = "silhouette_score"
            report["summary"]["primary_score"] = silhouette
            
            if silhouette > 0.7:
                report["summary"]["status"] = "EXCELLENT"
                report["recommendations"].append("Excellente séparation des clusters")
            elif silhouette > 0.5:
                report["summary"]["status"] = "GOOD"
                report["recommendations"].append("Bonne séparation des clusters")
            else:
                report["summary"]["status"] = "NEEDS_IMPROVEMENT"
                report["recommendations"].append("Séparation faible - essayer d'autres algorithmes")
        
        report["summary"]["computation_time"] = metrics.get("computation_time", 0)
        
        log_metrics("INFO", "Rapport d'évaluation généré", {
            "model_name": model_name,
            "status": report["summary"]["status"]
        })
        
    except Exception as e:
        log_metrics("ERROR", "Erreur génération rapport", {"error": str(e)})
        report["summary"]["status"] = "ERROR"
        report["summary"]["message"] = f"Erreur génération: {str(e)}"
    
    return report

@monitor_performance
def compare_models_performance(models_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """Compare les performances de plusieurs modèles."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models_count": len(models_metrics),
        "ranking": [],
        "best_model": None,
        "comparison_metrics": {},
        "warnings": []
    }
    
    try:
        if not models_metrics:
            comparison["error"] = "Aucune métrique fournie"
            return comparison
        
        # Détermination du type de tâche et métrique principale
        task_type = None
        primary_metric = None
        metric_direction = 1  # 1 pour maximize, -1 pour minimize
        
        for model_name, metrics in models_metrics.items():
            if "accuracy" in metrics:
                task_type = "classification"
                primary_metric = "accuracy"
                metric_direction = 1
                break
            elif "r2" in metrics:
                task_type = "regression" 
                primary_metric = "r2"
                metric_direction = 1
                break
            elif "silhouette_score" in metrics:
                task_type = "clustering"
                primary_metric = "silhouette_score"
                metric_direction = 1
                break
            elif "mse" in metrics:
                task_type = "regression"
                primary_metric = "mse"
                metric_direction = -1
                break
        
        if not primary_metric:
            comparison["error"] = "Impossible de déterminer la métrique de comparaison"
            return comparison
        
        # Classement des modèles
        ranking_data = []
        for model_name, metrics in models_metrics.items():
            score = metrics.get(primary_metric)
            if score is not None and np.isfinite(score):
                ranking_data.append((model_name, score, metrics))
        
        if metric_direction == -1:
            ranking_data.sort(key=lambda x: x[1])  # Tri croissant pour les métriques à minimiser
        else:
            ranking_data.sort(key=lambda x: x[1], reverse=True)  # Tri décroissant pour maximiser
        
        # Construction du classement
        comparison["ranking"] = []
        for i, (model_name, score, metrics) in enumerate(ranking_data):
            comparison["ranking"].append({
                "rank": i + 1,
                "model_name": model_name,
                "score": float(score),
                "metrics": _sanitize_metrics_for_output(metrics),
                "warnings": metrics.get("warnings", [])
            })
        
        if ranking_data:
            comparison["best_model"] = ranking_data[0][0]
            comparison["best_score"] = float(ranking_data[0][1])
        
        comparison["task_type"] = task_type
        comparison["primary_metric"] = primary_metric
        
        log_metrics("INFO", "Comparaison modèles terminée", {
            "n_models": len(models_metrics),
            "best_model": comparison.get("best_model"),
            "best_score": comparison.get("best_score")
        })
        
    except Exception as e:
        log_metrics("ERROR", "Erreur comparaison modèles", {"error": str(e)})
        comparison["error"] = str(e)
    
    return comparison

def _sanitize_metrics_for_output(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Nettoie les métriques pour la sortie (supprime les objets complexes)."""
    sanitized = {}
    for key, value in metrics.items():
        if key in ['error', 'warnings', 'success']:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (list, dict)) and not isinstance(value, (np.ndarray)):
            # Conversion récursive pour les structures simples
            try:
                json.dumps(value)  # Test de sérialisation
                sanitized[key] = value
            except (TypeError, ValueError):
                continue  # Ignore les structures complexes
    return sanitized

# =============================
# EXPORT ET INITIALISATION
# =============================

__all__ = [
    'EvaluationMetrics',
    'MetricsLogger',
    'MetricsStateManager',
    'safe_array_conversion',
    'validate_input_data',
    'calculate_global_metrics',
    'evaluate_single_train_test_split',
    'generate_evaluation_report',
    'compare_models_performance',
    'get_system_metrics',
    'log_metrics',
    'METRICS_LOGGER',
    'METRICS_STATE'
]

# Initialisation au chargement
log_metrics("INFO", "Module metrics initialisé", {
    "version": "2.0.0",
    "psutil_available": PSUTIL_AVAILABLE
})