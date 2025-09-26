import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    explained_variance_score, mean_squared_log_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
import time
import gc
from io import BytesIO
import base64

# Configuration des warnings
warnings.filterwarnings("ignore")

from utils.logging_config import get_logger

logger = get_logger(__name__)

class EvaluationMetrics:
    """Classe pour calculer et gérer les métriques d'évaluation de façon robuste"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.metrics = {}
        self.error_messages = []
        
    def safe_metric_calculation(self, metric_func, *args, **kwargs) -> Any:
        """Calcule une métrique avec gestion robuste des erreurs"""
        try:
            return metric_func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Erreur calcul {metric_func.__name__}: {str(e)}"
            self.error_messages.append(error_msg)
            logger.warning(error_msg)
            return None
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calcule les métriques pour les problèmes de classification"""
        metrics = {}
        
        try:
            # Métriques de base
            metrics['accuracy'] = self.safe_metric_calculation(accuracy_score, y_true, y_pred)
            metrics['precision'] = self.safe_metric_calculation(precision_score, y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = self.safe_metric_calculation(recall_score, y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = self.safe_metric_calculation(f1_score, y_true, y_pred, average='weighted', zero_division=0)
            
            # Rapport de classification détaillé
            try:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics['classification_report'] = report
                
                # Extraire les métriques par classe
                class_metrics = {}
                for key, value in report.items():
                    if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
                        class_metrics[key] = {
                            'precision': value.get('precision', 0),
                            'recall': value.get('recall', 0),
                            'f1_score': value.get('f1-score', 0),
                            'support': value.get('support', 0)
                        }
                metrics['class_metrics'] = class_metrics
                
            except Exception as e:
                logger.warning(f"Erreur rapport classification: {e}")
                metrics['classification_report'] = {}
                metrics['class_metrics'] = {}
            
            # AUC-ROC si les probabilités sont disponibles
            if y_proba is not None and len(y_proba) > 0:
                try:
                    n_classes = len(np.unique(y_true))
                    if n_classes == 2:
                        metrics['roc_auc'] = self.safe_metric_calculation(
                            roc_auc_score, y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                        )
                    else:
                        metrics['roc_auc'] = self.safe_metric_calculation(
                            roc_auc_score, y_true, y_proba, multi_class='ovr', average='weighted'
                        )
                except Exception as e:
                    logger.warning(f"Erreur calcul AUC-ROC: {e}")
                    metrics['roc_auc'] = None
            
            # Matrice de confusion
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                metrics['n_classes'] = len(np.unique(y_true))
            except Exception as e:
                logger.warning(f"Erreur matrice confusion: {e}")
                metrics['confusion_matrix'] = None
            
        except Exception as e:
            logger.error(f"Erreur critique calcul métriques classification: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcule les métriques pour les problèmes de régression"""
        metrics = {}
        
        try:
            # Métriques de base
            metrics['mse'] = self.safe_metric_calculation(mean_squared_error, y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse']) if metrics['mse'] is not None else None
            metrics['mae'] = self.safe_metric_calculation(mean_absolute_error, y_true, y_pred)
            metrics['r2'] = self.safe_metric_calculation(r2_score, y_true, y_pred)
            metrics['explained_variance'] = self.safe_metric_calculation(explained_variance_score, y_true, y_pred)
            
            # MSE logarithmique (si valeurs positives)
            if np.all(y_true > 0) and np.all(y_pred > 0):
                metrics['msle'] = self.safe_metric_calculation(mean_squared_log_error, y_true, y_pred)
            else:
                metrics['msle'] = None
            
            # Statistiques des erreurs
            if metrics['mae'] is not None and metrics['rmse'] is not None:
                errors = np.abs(y_true - y_pred)
                metrics['error_stats'] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(errors)),
                    'median_error': float(np.median(errors))
                }
            
            # Ratio d'amélioration par rapport à la baseline (moyenne)
            baseline_pred = np.full_like(y_true, np.mean(y_true))
            baseline_mse = mean_squared_error(y_true, baseline_pred)
            if metrics['mse'] is not None and baseline_mse > 0:
                metrics['improvement_ratio'] = 1 - (metrics['mse'] / baseline_mse)
            
        except Exception as e:
            logger.error(f"Erreur critique calcul métriques régression: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def calculate_unsupervised_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calcule les métriques pour les problèmes non supervisés"""
        metrics = {}
        
        try:
            n_clusters = len(np.unique(labels))
            
            if n_clusters > 1:
                metrics['silhouette_score'] = self.safe_metric_calculation(silhouette_score, X, labels)
                metrics['davies_bouldin_score'] = self.safe_metric_calculation(davies_bouldin_score, X, labels)
                metrics['calinski_harabasz_score'] = self.safe_metric_calculation(calinski_harabasz_score, X, labels)
            else:
                metrics['silhouette_score'] = None
                metrics['davies_bouldin_score'] = None
                metrics['calinski_harabasz_score'] = None
            
            metrics['n_clusters'] = n_clusters
            metrics['cluster_sizes'] = {f"Cluster {i}": int(count) for i, count in enumerate(np.bincount(labels))}
            
            # Qualité du clustering
            if n_clusters > 1:
                cluster_quality = self._evaluate_cluster_quality(X, labels)
                metrics.update(cluster_quality)
            
        except Exception as e:
            logger.error(f"Erreur critique calcul métriques non supervisées: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _evaluate_cluster_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Évalue la qualité du clustering"""
        quality = {}
        
        try:
            n_clusters = len(np.unique(labels))
            
            # Séparation des clusters
            cluster_centers = []
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    cluster_centers.append(np.mean(cluster_points, axis=0))
            
            if len(cluster_centers) > 1:
                # Distance moyenne entre les centres des clusters
                from scipy.spatial.distance import pdist
                center_distances = pdist(cluster_centers)
                quality['avg_inter_cluster_distance'] = float(np.mean(center_distances))
            
            # Cohésion des clusters
            intra_cluster_distances = []
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_cluster_distances.extend(distances)
            
            if intra_cluster_distances:
                quality['avg_intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
            
        except Exception as e:
            logger.warning(f"Erreur évaluation qualité clusters: {e}")
        
        return quality

def safe_array_conversion(data: Any, max_samples: int = 100000) -> np.ndarray:
    """
    Convertit les données en array numpy de façon sécurisée.
    
    Args:
        data: Données à convertir
        max_samples: Nombre maximum d'échantillons à conserver
    
    Returns:
        Array numpy
    """
    try:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values
        
        if isinstance(data, np.ndarray):
            # Échantillonnage si trop grand
            if len(data) > max_samples:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(data), size=max_samples, replace=False)
                data = data[indices]
            
            return data.flatten() if data.ndim > 1 else data
        
        # Conversion depuis d'autres types
        return np.array(data).flatten()
        
    except Exception as e:
        logger.error(f"Erreur conversion array: {e}")
        return np.array([])

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
            # Vérifier les valeurs extrêmes
            if np.any(np.isinf(y_true_flat)) or np.any(np.isinf(y_pred_flat)):
                validation["warnings"].append("Valeurs infinies détectées")
            
            if np.any(np.isnan(y_true_flat)) or np.any(np.isnan(y_pred_flat)):
                validation["warnings"].append("Valeurs NaN détectées")
        
        # Vérification du nombre d'échantillons
        if validation["n_samples"] < 10:
            validation["warnings"].append("Très peu d'échantillons pour l'évaluation")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"Erreur validation données: {e}")
    
    return validation

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
    Version robuste avec gestion complète des erreurs.
    
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
    evaluator = EvaluationMetrics(task_type)
    metrics = {}
    
    try:
        # Agrégation des données
        y_true_aggregated = []
        y_pred_aggregated = []
        y_proba_aggregated = []
        
        # Traitement des listes d'entrée
        for i, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):
            try:
                y_true_flat = safe_array_conversion(y_true)
                y_pred_flat = safe_array_conversion(y_pred)
                
                if len(y_true_flat) > 0 and len(y_true_flat) == len(y_pred_flat):
                    y_true_aggregated.extend(y_true_flat)
                    y_pred_aggregated.extend(y_pred_flat)
                    
                    # Agrégation des probabilités si disponibles
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
                logger.warning(f"Erreur agrégation batch {i}: {e}")
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
                logger.warning(f"Erreur décodage labels: {e}")
                y_true_decoded = y_true_array
                y_pred_decoded = y_pred_array
        else:
            y_true_decoded = y_true_array
            y_pred_decoded = y_pred_array
        
        # Validation des données
        validation = validate_input_data(y_true_decoded, y_pred_decoded, task_type)
        
        if not validation["is_valid"]:
            metrics["error"] = f"Données invalides: {', '.join(validation['issues'])}"
            logger.error(metrics["error"])
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
            
        elif task_type == "unsupervised":
            if X_data is not None:
                X_flat = safe_array_conversion(X_data)
                if len(X_flat) == len(y_pred_array):
                    unsupervised_metrics = evaluator.calculate_unsupervised_metrics(X_flat, y_pred_array)
                    metrics.update(unsupervised_metrics)
            else:
                metrics["error"] = "Données X requises pour l'évaluation non supervisée"
        
        else:
            metrics["error"] = f"Type de tâche non supporté: {task_type}"
        
        # Métriques de performance du calcul
        metrics["computation_time"] = time.time() - start_time
        metrics["success"] = True
        
        # Messages d'erreur accumulés
        if evaluator.error_messages:
            metrics["calculation_warnings"] = evaluator.error_messages
        
        logger.info(f"✅ Métriques calculées pour {metrics['n_samples']} échantillons en {metrics['computation_time']:.2f}s")
        
    except Exception as e:
        metrics["error"] = f"Erreur critique calcul métriques: {str(e)}"
        metrics["success"] = False
        logger.error(f"❌ Erreur critique calculate_global_metrics: {e}", exc_info=True)
    
    # Nettoyage mémoire
    gc.collect()
    
    return metrics

def evaluate_single_train_test_split(
    model: Any,
    X_test: Any,
    y_test: Any,
    task_type: str = "classification",
    label_encoder: Any = None
) -> Dict[str, Any]:
    """
    Évalue un modèle sur un unique jeu de test (fallback si la CV échoue).
    
    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        task_type: Type de tâche
        label_encoder: Encodeur de labels
    
    Returns:
        Métriques d'évaluation
    """
    try:
        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Erreur predict_proba: {e}")
        
        # Calcul des métriques
        metrics = calculate_global_metrics(
            [y_test], [y_pred], [y_proba] if y_proba is not None else [],
            task_type, label_encoder, X_data=X_test
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur evaluate_single_train_test_split: {e}")
        return {"error": str(e), "success": False}

def generate_evaluation_report(metrics: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    """
    Génère un rapport d'évaluation structuré et détaillé.
    
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
        task_type = "classification"  # À détecter depuis les métriques
        
        if "accuracy" in metrics:
            task_type = "classification"
            report["summary"]["task_type"] = "classification"
            report["summary"]["primary_metric"] = "accuracy"
            report["summary"]["primary_score"] = metrics.get("accuracy", 0)
            report["summary"]["status"] = "SUCCESS"
            
            # Recommandations pour la classification
            accuracy = metrics.get("accuracy", 0)
            if accuracy > 0.9:
                report["recommendations"].append("Excellente performance - modèle très fiable")
            elif accuracy > 0.7:
                report["recommendations"].append("Bonne performance - modèle utilisable en production")
            else:
                report["recommendations"].append("Performance modérée - envisager l'optimisation")
                
        elif "r2" in metrics:
            task_type = "regression"
            report["summary"]["task_type"] = "regression"
            report["summary"]["primary_metric"] = "r2"
            report["summary"]["primary_score"] = metrics.get("r2", 0)
            report["summary"]["status"] = "SUCCESS"
            
            # Recommandations pour la régression
            r2 = metrics.get("r2", 0)
            if r2 > 0.8:
                report["recommendations"].append("Excellente performance - modèle très prédictif")
            elif r2 > 0.5:
                report["recommendations"].append("Performance acceptable - modèle utilisable")
            else:
                report["recommendations"].append("Performance faible - revoir les features")
                
        elif "silhouette_score" in metrics:
            task_type = "unsupervised"
            report["summary"]["task_type"] = "unsupervised"
            report["summary"]["primary_metric"] = "silhouette_score"
            report["summary"]["primary_score"] = metrics.get("silhouette_score", 0)
            report["summary"]["status"] = "SUCCESS"
        
        # Informations générales
        report["summary"]["n_samples"] = metrics.get("n_samples", 0)
        report["summary"]["computation_time"] = metrics.get("computation_time", 0)
        
        # Alertes
        if metrics.get("validation_warnings"):
            report["warnings"] = metrics["validation_warnings"]
        
        if metrics.get("calculation_warnings"):
            report["calculation_warnings"] = metrics["calculation_warnings"]
        
    except Exception as e:
        logger.error(f"Erreur génération rapport: {e}")
        report["summary"]["status"] = "ERROR"
        report["summary"]["message"] = f"Erreur génération rapport: {str(e)}"
    
    return report

def create_confusion_matrix_plot(confusion_matrix: List[List[int]], class_names: List[str] = None) -> str:
    """
    Crée une visualisation de matrice de confusion en base64.
    
    Args:
        confusion_matrix: Matrice de confusion
        class_names: Noms des classes
    
    Returns:
        Image base64 encodée
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraies étiquettes')
        plt.xlabel('Étiquettes prédites')
        
        # Conversion en base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Erreur création matrice confusion: {e}")
        return ""

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
                task_type = "unsupervised"
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
        
        # Tri selon la métrique (ordre décroissant sauf pour certaines métriques)
        if primary_metric in ["mse", "mae", "rmse", "davies_bouldin_score"]:
            ranking.sort(key=lambda x: x[1])  # Croissant (plus petit = mieux)
        else:
            ranking.sort(key=lambda x: x[1], reverse=True)  # Décroissant
        
        comparison["ranking"] = [
            {
                "rank": i + 1,
                "model_name": name,
                "score": score,
                "metrics": metrics
            }
            for i, (name, score, metrics) in enumerate(ranking)
        ]
        
        # Meilleur modèle
        if ranking:
            comparison["best_model"] = ranking[0][0]
            comparison["best_score"] = ranking[0][1]
        
        # Métriques de comparaison
        comparison["task_type"] = task_type
        comparison["primary_metric"] = primary_metric
        
        logger.info(f"✅ Comparaison de {len(models_metrics)} modèles terminée")
        
    except Exception as e:
        comparison["error"] = f"Erreur comparaison modèles: {str(e)}"
        logger.error(f"❌ Erreur compare_models_performance: {e}")
    
    return comparison

import psutil
def get_system_metrics():
    """
    Retourne les métriques système utiles pour le suivi des ressources.
    """
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory": psutil.virtual_memory().available / (1024 ** 3),  # en Go
            "total_memory": psutil.virtual_memory().total / (1024 ** 3),          # en Go
            "disk_usage": psutil.disk_usage('/').percent
        }
    except Exception as e:
        return {"error": str(e)}
