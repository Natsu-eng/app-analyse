"""
Module de calcul de métriques robuste pour l'évaluation des modèles ML.
Gère classification, régression et clustering avec gestion d'erreurs avancée.
Conforme aux standards MLOps et prêt pour la production.
"""

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
from functools import wraps

# Configuration du logging
logger = logging.getLogger(__name__)

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import conditionnel pour éviter les dépendances facultatives
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible - monitoring système limité")

# =============================
# Décorateurs de monitoring
# =============================

def monitor_performance(func):
    """Décorateur pour monitorer les performances des fonctions critiques"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = _get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.debug(f"{func.__name__} - Duration: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            # Alertes pour performances dégradées
            if duration > 30:
                logger.warning(f"⏰ {func.__name__} took {duration:.2f}s - performance issue")
            if memory_delta > 500:
                logger.warning(f"💾 {func.__name__} used {memory_delta:.1f}MB - high memory usage")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def safe_execute(fallback_value=None, log_errors=True):
    """Décorateur pour l'exécution sécurisée avec fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"❌ Safe execution failed in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

# =============================
# Classe principale
# =============================

class EvaluationMetrics:
    """
    Classe robuste pour calculer et gérer les métriques d'évaluation ML.
    Supporte classification, régression et clustering avec gestion d'erreurs avancée.
    """
    
    def __init__(self, task_type: str):
        """
        Args:
            task_type: Type de tâche ML ('classification', 'regression', 'clustering')
        """
        self.task_type = self._normalize_task_type(task_type)
        self.metrics = {}
        self.error_messages = []
        self.warning_messages = []
        
    def _normalize_task_type(self, task_type: str) -> str:
        """Normalise le type de tâche pour cohérence interne"""
        task_type = task_type.lower().strip()
        if task_type in ['unsupervised', 'cluster']:
            return 'clustering'
        return task_type
    
    def safe_metric_calculation(self, metric_func, *args, **kwargs) -> Any:
        """
        Calcule une métrique avec gestion robuste des erreurs.
        
        Args:
            metric_func: Fonction de calcul de métrique
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat du calcul ou None en cas d'erreur
        """
        try:
            result = metric_func(*args, **kwargs)
            
            # Validation du résultat
            if result is not None and np.isscalar(result):
                if np.isnan(result) or np.isinf(result):
                    raise ValueError(f"Résultat invalide: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur calcul {metric_func.__name__}: {str(e)}"
            self.error_messages.append(error_msg)
            logger.warning(f"⚠️ {error_msg}")
            return None
    
    @monitor_performance
    def calculate_classification_metrics(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray, 
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calcule les métriques pour les problèmes de classification.       
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)         
        Returns:
            Dictionnaire des métriques de classification
        """
        metrics = {
            "task_type": "classification",
            "success": False,
            "n_samples": len(y_true),
            "n_classes": len(np.unique(y_true))
        }
      
        try:
            # Validation des données
            validation = self._validate_classification_data(y_true, y_pred)
            if not validation["is_valid"]:
                metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                return metrics
            
            # Métriques de base avec gestion d'erreurs
            metrics['accuracy'] = self.safe_metric_calculation(accuracy_score, y_true, y_pred)
            metrics['precision'] = self.safe_metric_calculation(precision_score, y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = self.safe_metric_calculation(recall_score, y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = self.safe_metric_calculation(f1_score, y_true, y_pred, average='weighted', zero_division=0)
            
            # Rapport de classification détaillé
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            class_metrics = {
                str(key): {
                    'precision': float(value.get('precision', 0)),
                    'recall': float(value.get('recall', 0)),
                    'f1_score': float(value.get('f1-score', 0)),
                    'support': int(value.get('support', 0))
                } for key, value in report.items() if isinstance(value, dict) and key not in ['accuracy', 'macro avg', 'weighted avg']
            }
            metrics['class_metrics'] = class_metrics

            # AUC-ROC si les probabilités sont disponibles
            if y_proba is not None and len(y_proba) > 0:
                try:
                    n_classes = len(np.unique(y_true))
                    metrics['roc_auc'] = self.safe_metric_calculation(
                        roc_auc_score, 
                        y_true, 
                        y_proba[:, 1] if n_classes == 2 else y_proba,
                        multi_class='ovr' if n_classes > 2 else None,
                        average='weighted' if n_classes > 2 else None
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Calcul AUC-ROC échoué: {e}")
                    metrics['roc_auc'] = None

            # Matrice de confusion
            try:
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            except Exception as e:
                logger.warning(f"⚠️ Matrice confusion échouée: {e}")
                metrics['confusion_matrix'] = None
                
            metrics['success'] = True
            logger.info(f"✅ Métriques classification calculées: {metrics['n_samples']} échantillons, {metrics['n_classes']} classes")
            
        except Exception as e:
            logger.error(f"❌ Erreur critique calcul métriques classification: {e}")
            metrics['error'] = str(e)
            metrics['success'] = False
        
        return metrics
    
    def _validate_classification_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Valide les données pour la classification"""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            if len(y_true) != len(y_pred):
                validation["is_valid"] = False
                validation["issues"].append(f"Dimensions incohérentes: y_true={len(y_true)}, y_pred={len(y_pred)}")
                return validation
            
            if len(y_true) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Moins de 2 échantillons")
                return validation
            
            unique_true = np.unique(y_true)
            if len(unique_true) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Moins de 2 classes dans y_true")
            
            unique_pred = np.unique(y_pred)
            if len(unique_pred) < 2:
                validation["warnings"].append("Moins de 2 classes dans y_pred")
            
            # Vérification des types
            if not all(isinstance(x, (int, float, np.number)) for x in y_true):
                validation["warnings"].append("Types de données non numériques détectés")
                
        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur validation: {str(e)}")
        
        return validation
    
    @monitor_performance
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calcule les métriques pour les problèmes de régression.    
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions          
        Returns:
            Dictionnaire des métriques de régression
        """
        metrics = {
            "task_type": "regression",
            "success": False,
            "n_samples": len(y_true)
        }
        
        try:
            # Validation des données
            validation = self._validate_regression_data(y_true, y_pred)
            if not validation["is_valid"]:
                metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                return metrics
            
            # Métriques de base
            metrics['mse'] = self.safe_metric_calculation(mean_squared_error, y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse']) if metrics['mse'] is not None else None
            metrics['mae'] = self.safe_metric_calculation(mean_absolute_error, y_true, y_pred)
            metrics['r2'] = self.safe_metric_calculation(r2_score, y_true, y_pred)
            metrics['explained_variance'] = self.safe_metric_calculation(explained_variance_score, y_true, y_pred)
            
            # MSE logarithmique (si valeurs positives)
            if (np.all(y_true > 0) and np.all(y_pred > 0) and 
                not np.any(np.isinf(y_true)) and not np.any(np.isinf(y_pred))):
                metrics['msle'] = self.safe_metric_calculation(mean_squared_log_error, y_true, y_pred)
            else:
                metrics['msle'] = None
                logger.warning("⚠️ MSLE non calculé: valeurs négatives ou infinies détectées")
            
            # Statistiques des erreurs
            if metrics['mae'] is not None and metrics['rmse'] is not None:
                errors = np.abs(y_true - y_pred)
                metrics['error_stats'] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(errors)),
                    'median_error': float(np.median(errors)),
                    'q95_error': float(np.percentile(errors, 95))
                }
            
            # Ratio d'amélioration par rapport à la baseline
            baseline_pred = np.full_like(y_true, np.mean(y_true))
            baseline_mse = mean_squared_error(y_true, baseline_pred)
            if metrics['mse'] is not None and baseline_mse > 0:
                metrics['improvement_ratio'] = 1 - (metrics['mse'] / baseline_mse)
            
            metrics['success'] = True
            logger.info(f"✅ Métriques régression calculées: {metrics['n_samples']} échantillons")
            
        except Exception as e:
            logger.error(f"❌ Erreur critique calcul métriques régression: {e}")
            metrics['error'] = str(e)
            metrics['success'] = False
        
        return metrics
    
    def _validate_regression_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Valide les données pour la régression"""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            if len(y_true) != len(y_pred):
                validation["is_valid"] = False
                validation["issues"].append(f"Dimensions incohérentes: y_true={len(y_true)}, y_pred={len(y_pred)}")
                return validation
            
            if len(y_true) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Moins de 2 échantillons")
                return validation
            
            # Vérification des valeurs infinies
            if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
                validation["warnings"].append("Valeurs infinies détectées")
            
            # Vérification des NaN
            nan_count_true = np.sum(np.isnan(y_true))
            nan_count_pred = np.sum(np.isnan(y_pred))
            if nan_count_true > 0 or nan_count_pred > 0:
                validation["warnings"].append(f"NaN détectés: y_true={nan_count_true}, y_pred={nan_count_pred}")
            
            # Vérification de la variance
            if np.var(y_true) == 0:
                validation["warnings"].append("Variance nulle dans y_true")
                
        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur validation: {str(e)}")
        
        return validation
    
    @monitor_performance
    def calculate_unsupervised_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calcule les métriques pour les problèmes de clustering.
        
        Args:
            X: Données d'entrée
            labels: Labels des clusters
            
        Returns:
            Dictionnaire des métriques de clustering
        """
        metrics = {
            "task_type": "clustering",
            "success": False,
            "n_samples": len(X)
        }
        
        try:
            # Validation des données
            validation = self._validate_clustering_data(X, labels)
            if not validation["is_valid"]:
                metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
                return metrics
            
            # Filtrage des données valides
            valid_mask = labels >= 0  # Exclure le bruit (-1) et valeurs négatives
            valid_labels = labels[valid_mask]
            valid_X = X[valid_mask]
            
            n_clusters = len(np.unique(valid_labels))
            metrics['n_clusters'] = n_clusters
            metrics['n_valid_samples'] = len(valid_labels)
            metrics['n_outliers'] = int(np.sum(labels == -1))
            
            # Calcul des métriques seulement si assez de clusters valides
            if n_clusters > 1 and len(valid_labels) >= n_clusters:
                metrics['silhouette_score'] = self.safe_metric_calculation(
                    silhouette_score, valid_X, labels=valid_labels
                )
                metrics['davies_bouldin_score'] = self.safe_metric_calculation(
                    davies_bouldin_score, valid_X, labels=valid_labels
                )
                metrics['calinski_harabasz_score'] = self.safe_metric_calculation(
                    calinski_harabasz_score, valid_X, labels=valid_labels
                )
            else:
                metrics['silhouette_score'] = None
                metrics['davies_bouldin_score'] = None
                metrics['calinski_harabasz_score'] = None
                self.warning_messages.append("Pas assez de clusters valides pour calculer les métriques")
            
            # Statistiques des clusters
            try:
                cluster_sizes = {}
                for i, count in enumerate(np.bincount(valid_labels)):
                    cluster_sizes[f"Cluster_{i}"] = int(count)
                metrics['cluster_sizes'] = cluster_sizes
                
                # Qualité du clustering
                cluster_quality = self._evaluate_cluster_quality(valid_X, valid_labels)
                metrics.update(cluster_quality)
                
            except Exception as e:
                logger.warning(f"⚠️ Statistiques clusters échouées: {e}")
            
            metrics['success'] = True
            logger.info(f"✅ Métriques clustering calculées: {metrics['n_samples']} échantillons, {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"❌ Erreur critique calcul métriques clustering: {e}")
            metrics['error'] = str(e)
            metrics['success'] = False
        
        return metrics
    
    def _validate_clustering_data(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Valide les données pour le clustering"""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            if len(X) != len(labels):
                validation["is_valid"] = False
                validation["issues"].append(f"Incohérence dimensions: X={len(X)}, labels={len(labels)}")
                return validation
            
            if len(X) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Moins de 2 échantillons")
                return validation
            
            # Vérification des valeurs valides dans les labels
            valid_labels = labels[~np.isnan(labels)]
            if len(valid_labels) == 0:
                validation["is_valid"] = False
                validation["issues"].append("Aucun label valide")
                return validation
            
            n_clusters = len(np.unique(valid_labels))
            if n_clusters < 2:
                validation["warnings"].append("Moins de 2 clusters détectés")
            
            # Vérification de la dimensionalité
            if X.shape[1] < 2:
                validation["warnings"].append("Seulement 1 dimension - clustering potentiellement peu informatif")
                
        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur validation: {str(e)}")
        
        return validation
    
    def _evaluate_cluster_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Évalue la qualité du clustering avec des métriques additionnelles"""
        quality = {}
        
        try:
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                return quality
            
            # Calcul des centroïdes
            cluster_centers = []
            intra_cluster_distances = []
            
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    cluster_centers.append(centroid)
                    
                    # Distances intra-cluster
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_cluster_distances.extend(distances)
            
            # Distance inter-clusters
            if len(cluster_centers) > 1:
                from scipy.spatial.distance import pdist
                center_distances = pdist(cluster_centers)
                quality['avg_inter_cluster_distance'] = float(np.mean(center_distances))
                quality['min_inter_cluster_distance'] = float(np.min(center_distances))
                quality['max_inter_cluster_distance'] = float(np.max(center_distances))
            
            # Métriques intra-cluster
            if intra_cluster_distances:
                quality['avg_intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
                quality['max_intra_cluster_distance'] = float(np.max(intra_cluster_distances))
            
            # Ratio de séparation
            if 'avg_inter_cluster_distance' in quality and 'avg_intra_cluster_distance' in quality:
                quality['separation_ratio'] = (
                    quality['avg_inter_cluster_distance'] / quality['avg_intra_cluster_distance']
                    if quality['avg_intra_cluster_distance'] > 0 else float('inf')
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Évaluation qualité clusters échouée: {e}")
        
        return quality

# =============================
# Fonctions utilitaires
# =============================

@safe_execute(fallback_value=np.array([]))
def safe_array_conversion(data: Any, max_samples: int = 100000, sample: bool = True) -> np.ndarray:
    """
    Convertit les données en array numpy de façon sécurisée.   
    Args:
        data: Données à convertir (peut être une liste, une série, un DataFrame ou un tableau numpy).
        max_samples: Nombre maximum d'échantillons à conserver (utilisé si sample est True).
        sample: Booléen pour contrôler si un échantillonnage doit être effectué si les données dépassent max_samples.       
    Returns:
        Array numpy
    """
    try:
        # Vérification du type de données d'entrée
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Vérification si les données sont vides ou de type inattendu
        if data.size == 0:
            logger.warning("Les données d'entrée sont vides, retour d'un tableau vide.")
            return np.array([])
        
        # Échantillonnage si trop de données et si l'option d'échantillonnage est activée
        if sample and len(data) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(data), size=max_samples, replace=False)
            data = data[indices]
        
        return data.flatten() if data.ndim > 1 and data.shape[1] == 1 else data

    except Exception as e:
        logger.error(f"❌ Erreur lors de la conversion en array: {e}")
        return np.array([])
    
@monitor_performance
def validate_input_data(y_true: Any, y_pred: Any, task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entrée pour l'évaluation.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        task_type: Type de tâche
        
    Returns:
        Résultat de validation
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "n_samples": 0
    }
    
    try:
        # Normalisation du type de tâche
        task_type = task_type.lower().strip()
        if task_type in ['unsupervised', 'cluster']:
            task_type = 'clustering'
        
        # Conversion sécurisée
        y_true_flat = safe_array_conversion(y_true)
        y_pred_flat = safe_array_conversion(y_pred)
        
        # Vérification des dimensions
        if len(y_true_flat) == 0 or len(y_pred_flat) == 0:
            validation["is_valid"] = False
            validation["issues"].append("Données vides après conversion")
            return validation
        
        if len(y_true_flat) != len(y_pred_flat):
            validation["is_valid"] = False
            validation["issues"].append(f"Dimensions incohérentes: y_true={len(y_true_flat)}, y_pred={len(y_pred_flat)}")
            return validation
        
        validation["n_samples"] = len(y_true_flat)
        
        # Vérifications spécifiques au type de tâche
        if task_type == "classification":
            unique_true = np.unique(y_true_flat)
            unique_pred = np.unique(y_pred_flat)
            
            if len(unique_true) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Moins de 2 classes dans y_true")
            
            if len(unique_pred) < 2:
                validation["warnings"].append("Moins de 2 classes dans y_pred")
        
        elif task_type == "regression":
            if np.any(np.isinf(y_true_flat)) or np.any(np.isinf(y_pred_flat)):
                validation["warnings"].append("Valeurs infinies détectées")
            
            if np.any(np.isnan(y_true_flat)) or np.any(np.isnan(y_pred_flat)):
                validation["warnings"].append("Valeurs NaN détectées")
        
        elif task_type == "clustering":
            unique_labels = np.unique(y_pred_flat)
            if len(unique_labels) < 2 and -1 not in unique_labels:
                validation["warnings"].append("Moins de 2 clusters valides détectés")
        
        if validation["n_samples"] < 10:
            validation["warnings"].append("Très peu d'échantillons pour l'évaluation")
        
        logger.debug(f"✅ Validation données: {validation['n_samples']} échantillons, {len(validation['issues'])} issues")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"❌ Erreur validation données: {e}")
    
    return validation

@monitor_performance
def calculate_global_metrics(
    y_true_all: List[Any],
    y_pred_all: List[Any], 
    y_proba_all: List[Any] = None,
    task_type: str = "classification",
    label_encoder: Any = None,
    X_data: Any = None
) -> Dict[str, Any]:
    """
    Calcule les métriques de performance sur un ensemble agrégé de prédictions.
    
    Args:
        y_true_all: Liste des valeurs réelles
        y_pred_all: Liste des prédictions
        y_proba_all: Liste des probabilités (classification)
        task_type: Type de tâche
        label_encoder: Encodeur de labels
        X_data: Données d'entrée (clustering)
        
    Returns:
        Dictionnaire des métriques
    """
    start_time = time.time()
    task_type = task_type.lower().strip()
    if task_type in ['unsupervised', 'cluster']:
        task_type = 'clustering'
        
    evaluator = EvaluationMetrics(task_type)
    metrics = {
        "task_type": task_type,
        "success": False,
        "computation_time": 0
    }
    
    try:
        # Agrégation des données
        y_true_aggregated = []
        y_pred_aggregated = []
        y_proba_aggregated = []
        
        for i, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):
            try:
                y_true_flat = safe_array_conversion(y_true)
                y_pred_flat = safe_array_conversion(y_pred)
                
                if len(y_true_flat) > 0 and len(y_true_flat) == len(y_pred_flat):
                    y_true_aggregated.extend(y_true_flat)
                    y_pred_aggregated.extend(y_pred_flat)
                    
                    if y_proba_all and i < len(y_proba_all):
                        y_proba = y_proba_all[i]
                        if y_proba is not None:
                            y_proba_flat = safe_array_conversion(y_proba)
                            if len(y_proba_flat) == len(y_true_flat):
                                if len(y_proba_aggregated) == 0:
                                    y_proba_aggregated = y_proba_flat
                                else:
                                    y_proba_aggregated = np.vstack([y_proba_aggregated, y_proba_flat])
            except Exception as e:
                logger.warning(f"⚠️ Erreur agrégation batch {i}: {e}")
                continue
        
        # Conversion en arrays numpy
        y_true_array = np.array(y_true_aggregated)
        y_pred_array = np.array(y_pred_aggregated)
        y_proba_array = np.array(y_proba_aggregated) if len(y_proba_aggregated) > 0 else None
        
        # Décodage des labels si encodeur disponible
        if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
            try:
                y_true_decoded = label_encoder.inverse_transform(y_true_array.astype(int))
                y_pred_decoded = label_encoder.inverse_transform(y_pred_array.astype(int))
            except Exception as e:
                logger.warning(f"⚠️ Erreur décodage labels: {e}")
                y_true_decoded = y_true_array
                y_pred_decoded = y_pred_array
        else:
            y_true_decoded = y_true_array
            y_pred_decoded = y_pred_array
        
        # Validation des données
        validation = validate_input_data(y_true_decoded, y_pred_decoded, task_type)
        
        if not validation["is_valid"]:
            metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
            logger.error(f"❌ {metrics['error']}")
            return metrics
        
        # Calcul des métriques selon le type de tâche
        metrics["n_samples"] = validation["n_samples"]
        metrics["validation_warnings"] = validation["warnings"]
        
        if task_type == "classification":
            classification_metrics = evaluator.calculate_classification_metrics(
                y_true_decoded, y_pred_decoded, y_proba_array
            )
            metrics.update(classification_metrics)
            
        elif task_type == "regression":
            regression_metrics = evaluator.calculate_regression_metrics(y_true_decoded, y_pred_decoded)
            metrics.update(regression_metrics)
            
        elif task_type == "clustering":
            if X_data is not None:
                X_flat = safe_array_conversion(X_data)
                if len(X_flat) == len(y_pred_array):
                    unsupervised_metrics = evaluator.calculate_unsupervised_metrics(X_flat, y_pred_array)
                    metrics.update(unsupervised_metrics)
                else:
                    metrics["error"] = f"Incohérence dimensions X_data={len(X_flat)}, labels={len(y_pred_array)}"
            else:
                metrics["error"] = "Données X requises pour l'évaluation clustering"
        
        else:
            metrics["error"] = f"Type de tâche non supporté: {task_type}"
        
        # Métriques de performance du calcul
        metrics["computation_time"] = time.time() - start_time
        metrics["success"] = True
        
        # Ajout des warnings du calculateur
        if evaluator.error_messages:
            metrics["calculation_warnings"] = evaluator.error_messages
        if evaluator.warning_messages:
            metrics["calculation_warnings"] = metrics.get("calculation_warnings", []) + evaluator.warning_messages
        
        logger.info(f"✅ Métriques globales calculées: {metrics['n_samples']} échantillons en {metrics['computation_time']:.2f}s")
        
    except Exception as e:
        metrics["error"] = f"Erreur critique calcul métriques: {str(e)}"
        metrics["success"] = False
        logger.error(f"❌ Erreur critique calculate_global_metrics: {e}", exc_info=True)
    
    # Nettoyage mémoire
    gc.collect()
    
    return metrics

@safe_execute(fallback_value={"error": "Erreur évaluation", "success": False})
def evaluate_single_train_test_split(
    model: Any,
    X_test: Any,
    y_test: Any,
    task_type: str = "classification",
    label_encoder: Any = None
) -> Dict[str, Any]:
    """
    Évalue un modèle sur un unique jeu de test.
    
    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        task_type: Type de tâche
        label_encoder: Encodeur de labels
        
    Returns:
        Métriques d'évaluation
    """
    task_type = task_type.lower().strip()
    if task_type in ['unsupervised', 'cluster']:
        task_type = 'clustering'
        
    try:
        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"⚠️ predict_proba échoué: {e}")
        
        # Calcul des métriques
        metrics = calculate_global_metrics(
            [y_test], [y_pred], [y_proba] if y_proba is not None else [],
            task_type, label_encoder, X_data=X_test if task_type == 'clustering' else None
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur evaluate_single_train_test_split: {e}")
        return {"error": str(e), "success": False}

@monitor_performance
def generate_evaluation_report(metrics: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    """
    Génère un rapport d'évaluation structuré.
    
    Args:
        metrics: Métriques calculées
        model_name: Nom du modèle
        
    Returns:
        Rapport structuré
    """
    report = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
        "detailed_metrics": metrics,
        "recommendations": []
    }
    
    try:
        if metrics.get("error"):
            report["summary"]["status"] = "ERROR"
            report["summary"]["message"] = metrics["error"]
            return report
        
        # Résumé selon le type de tâche
        task_type = metrics.get("task_type", "classification")
        
        if task_type == "classification":
            report["summary"]["task_type"] = "classification"
            report["summary"]["primary_metric"] = "accuracy"
            report["summary"]["primary_score"] = metrics.get("accuracy", 0)
            report["summary"]["status"] = "SUCCESS"
            
            accuracy = metrics.get("accuracy", 0)
            if accuracy > 0.9:
                report["recommendations"].append("🎯 Excellente performance - modèle très fiable")
            elif accuracy > 0.7:
                report["recommendations"].append("✅ Bonne performance - modèle utilisable en production")
            else:
                report["recommendations"].append("⚠️ Performance modérée - envisager l'optimisation")
                
        elif task_type == "regression":
            report["summary"]["task_type"] = "regression"
            report["summary"]["primary_metric"] = "r2"
            report["summary"]["primary_score"] = metrics.get("r2", 0)
            report["summary"]["status"] = "SUCCESS"
            
            r2 = metrics.get("r2", 0)
            if r2 > 0.8:
                report["recommendations"].append("🎯 Excellente performance - modèle très prédictif")
            elif r2 > 0.5:
                report["recommendations"].append("✅ Performance acceptable - modèle utilisable")
            else:
                report["recommendations"].append("⚠️ Performance faible - revoir les features")
                
        elif task_type == "clustering":
            report["summary"]["task_type"] = "clustering"
            report["summary"]["primary_metric"] = "silhouette_score"
            report["summary"]["primary_score"] = metrics.get("silhouette_score", 0)
            report["summary"]["status"] = "SUCCESS"
            
            silhouette = metrics.get("silhouette_score", 0)
            if silhouette > 0.7:
                report["recommendations"].append("🎯 Excellente séparation des clusters")
            elif silhouette > 0.5:
                report["recommendations"].append("✅ Bonne séparation - clustering utilisable")
            else:
                report["recommendations"].append("⚠️ Séparation faible - envisager autre algorithme")
        
        report["summary"]["n_samples"] = metrics.get("n_samples", 0)
        report["summary"]["computation_time"] = metrics.get("computation_time", 0)
        
        if metrics.get("validation_warnings"):
            report["warnings"] = metrics["validation_warnings"]
        
        if metrics.get("calculation_warnings"):
            report["calculation_warnings"] = metrics["calculation_warnings"]
        
        logger.info(f"✅ Rapport évaluation généré pour {model_name}")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        report["summary"]["status"] = "ERROR"
        report["summary"]["message"] = f"Erreur génération rapport: {str(e)}"
    
    return report

@monitor_performance
def compare_models_performance(models_metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare les performances de plusieurs modèles.
    
    Args:
        models_metrics: Dictionnaire {nom_modèle: métriques}
        
    Returns:
        Comparaison structurée
    """
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_count": len(models_metrics),
        "ranking": [],
        "best_model": None,
        "comparison_metrics": {}
    }
    
    try:
        # Déterminer le type de tâche et la métrique principale
        primary_metric = None
        task_type = None
        
        for model_name, metrics in models_metrics.items():
            if "accuracy" in metrics:
                task_type = "classification"
                primary_metric = "accuracy"
                break
            elif "r2" in metrics:
                task_type = "regression"
                primary_metric = "r2"
                break
            elif "silhouette_score" in metrics:
                task_type = "clustering"
                primary_metric = "silhouette_score"
                break
        
        if not primary_metric:
            comparison["error"] = "Impossible de déterminer le type de tâche"
            return comparison
        
        # Classement des modèles
        ranking = []
        for model_name, metrics in models_metrics.items():
            score = metrics.get(primary_metric, float('-inf'))
            if score is not None:
                ranking.append((model_name, score, metrics))
        
        # Tri selon la métrique (plus grand = mieux, sauf pour certaines métriques)
        if primary_metric in ["mse", "mae", "rmse", "davies_bouldin_score"]:
            ranking.sort(key=lambda x: x[1])  # Croissant (plus petit = mieux)
        else:
            ranking.sort(key=lambda x: x[1], reverse=True)  # Décroissant
        
        comparison["ranking"] = [
            {
                "rank": i + 1,
                "model_name": name,
                "score": score,
                "metrics": _sanitize_metrics_for_output(metrics)
            }
            for i, (name, score, metrics) in enumerate(ranking)
        ]
        
        if ranking:
            comparison["best_model"] = ranking[0][0]
            comparison["best_score"] = ranking[0][1]
        
        comparison["task_type"] = task_type
        comparison["primary_metric"] = primary_metric
        
        logger.info(f"✅ Comparaison de {len(models_metrics)} modèles terminée")
        
    except Exception as e:
        comparison["error"] = f"Erreur comparaison modèles: {str(e)}"
        logger.error(f"❌ Erreur compare_models_performance: {e}")
    
    return comparison

def _sanitize_metrics_for_output(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Nettoie les métriques pour l'output (enlève les objets complexes)"""
    sanitized = {}
    for key, value in metrics.items():
        if not isinstance(value, (dict, list, np.ndarray)) and key != 'error':
            if isinstance(value, (int, float, np.number)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
    return sanitized

@safe_execute(fallback_value={})
def get_system_metrics() -> Dict[str, Any]:
    """
    Retourne les métriques système utiles pour le suivi des ressources.
    
    Returns:
        Dictionnaire des métriques système
    """
    try:
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil non disponible"}
            
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024 ** 3),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Erreur métriques système: {e}")
        return {"error": str(e), "timestamp": time.time()}

def _get_memory_usage() -> float:
    """
    Obtient l'utilisation mémoire actuelle en MB.
    
    Returns:
        Mémoire utilisée en MB
    """
    try:
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    except:
        return 0.0

# Export des fonctions principales
__all__ = [
    'EvaluationMetrics',
    'safe_array_conversion',
    'validate_input_data',
    'calculate_global_metrics',
    'evaluate_single_train_test_split',
    'generate_evaluation_report',
    'compare_models_performance',
    'get_system_metrics'
]