"""
Module de visualisation des modèles ML - Version Production
Visualisations avancées pour l'évaluation des modèles avec gestion robuste des erreurs
"""

import tempfile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
from typing import Dict, List, Any, Optional
from functools import wraps
import concurrent.futures
import gc
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    silhouette_samples, silhouette_score, confusion_matrix, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve

# Configuration des imports conditionnels avec fallbacks robustes
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit non disponible - mode standalone activé")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.info("SHAP non disponible - analyse d'importance avancée désactivée")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.info("psutil non disponible - monitoring système limité")

try:
    from matplotlib import cm
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.info("matplotlib non disponible - palettes de couleurs limitées")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_samples, silhouette_score, 
        confusion_matrix, roc_curve, precision_recall_curve, auc,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.model_selection import learning_curve, validation_curve
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logging.error(f"scikit-learn non disponible - fonctionnalités réduites: {e}")

# Import des constantes avec fallback
try:
    from src.config.constants import (
        VISUALIZATION_CONSTANTS, LOGGING_CONSTANTS, 
        VALIDATION_CONSTANTS, TRAINING_CONSTANTS
    )
except ImportError:
    # Fallback des constantes en cas d'import échoué
    VISUALIZATION_CONSTANTS = {
        "PLOTLY_TEMPLATE": "plotly_white",
        "MAX_SAMPLES": 10000,
        "TRAIN_SIZES": np.linspace(0.1, 1.0, 10),
        "COLOR_PALETTE": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "HEATMAP_COLORMAP": "Viridis"
    }
    LOGGING_CONSTANTS = {
        "DEFAULT_LOG_LEVEL": "INFO",
        "LOG_DIR": "logs",
        "LOG_FILE": "model_plots.log",
        "CONSOLE_LOGGING": True,
        "SLOW_OPERATION_THRESHOLD": 30,
        "HIGH_MEMORY_THRESHOLD": 500
    }
    VALIDATION_CONSTANTS = {
        "MIN_SAMPLES_PLOT": 10,
        "MAX_FEATURES_PLOT": 50
    }
    TRAINING_CONSTANTS = {
        "RANDOM_STATE": 42,
        "CV_FOLDS": 5,
        "N_JOBS": -1
    }

# Configuration du logging structuré
# CORRECTION - Supprimez cette méthode de la classe StructuredLogger
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure le logging avec format JSON structuré"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, LOGGING_CONSTANTS["DEFAULT_LOG_LEVEL"]))
    
    def log_structured(self, level: str, message: str, extra: Dict = None):
        """Log structuré en JSON"""
        log_dict = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "module": "model_plots"
        }
        if extra:
            log_dict.update(extra)
        
        log_message = json.dumps(log_dict, ensure_ascii=False, default=str)
        getattr(self.logger, level.lower())(log_message)

def _safe_get_model_task_type(model_result: Dict) -> str:
    """
    Détection robuste du type de tâche - VERSION CORRIGÉE
    """
    try:
        if not model_result or not isinstance(model_result, dict):
            return 'unknown'
        
        # 1. Vérification par métriques (METHODE PRINCIPALE)
        metrics = model_result.get('metrics', {})
        
        # Classification metrics
        if any(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']):
            return 'classification'
        
        # Regression metrics  
        if any(metric in metrics for metric in ['r2', 'mse', 'mae', 'rmse']):
            return 'regression'
            
        # Clustering metrics
        if any(metric in metrics for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']):
            return 'clustering'
        
        # 2. Vérification par nom du modèle
        model_name = model_result.get('model_name', '').lower()
        
        clustering_keywords = ['kmeans', 'dbscan', 'gmm', 'gaussianmixture', 'hierarchical', 'agglomerative']
        if any(kw in model_name for kw in clustering_keywords):
            return 'clustering'
            
        regression_keywords = ['regression', 'regressor', 'linear', 'lasso', 'ridge']
        if any(kw in model_name for kw in regression_keywords):
            return 'regression'
            
        classification_keywords = ['classifier', 'classification', 'logistic', 'randomforest', 'xgboost']
        if any(kw in model_name for kw in classification_keywords):
            return 'classification'
        
        # 3. Fallback basé sur les données disponibles
        has_y_test = model_result.get('y_test') is not None
        has_labels = model_result.get('labels') is not None
        
        if has_y_test and not has_labels:
            # Essayer de déterminer si c'est classification ou regression
            y_test = model_result.get('y_test')
            try:
                y_array = np.array(y_test)
                unique_vals = len(np.unique(y_array))
                if unique_vals <= 20:  # Classification si peu de valeurs uniques
                    return 'classification'
                else:
                    return 'regression'
            except:
                return 'classification'  # Default safe
        elif has_labels and not has_y_test:
            return 'clustering'
        
        return 'unknown'
        
    except Exception as e:
        print(f"DEBUG - Erreur détection type tâche: {e}")
        return 'unknown'


# Initialisation du logger
logger = StructuredLogger(__name__)
# Décorateurs de gestion d'erreurs et de performance
def timeout(seconds: int = 300):
    """Décorateur de timeout pour les opérations longues"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.log_structured("ERROR", f"Timeout: {func.__name__} > {seconds}s")
                    return None
                except Exception as e:
                    logger.log_structured("ERROR", f"Exception in {func.__name__}: {str(e)}")
                    return None
        return wrapper
    return decorator

def monitor_operation(func):
    """Décorateur de monitoring des performances"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            memory_used = _get_memory_usage() - start_memory
            
            logger.log_structured("DEBUG", f"Performance {func.__name__}", {
                "duration_s": round(elapsed, 2),
                "memory_delta_mb": round(memory_used, 1)
            })
            
            # Alertes si opération lente ou mémoire élevée
            if elapsed > LOGGING_CONSTANTS["SLOW_OPERATION_THRESHOLD"]:
                logger.log_structured("WARNING", f"Opération lente: {func.__name__} = {elapsed:.1f}s")
            if memory_used > LOGGING_CONSTANTS["HIGH_MEMORY_THRESHOLD"]:
                logger.log_structured("WARNING", f"Mémoire élevée: {func.__name__} = {memory_used:.1f}MB")
                
            return result
        except Exception as e:
            logger.log_structured("ERROR", f"{func.__name__} échoué: {str(e)}")
            return None
    return wrapper

def safe_execute(fallback_value=None, log_errors: bool = True):
    """Décorateur d'exécution sécurisée"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.log_structured("ERROR", f"Safe execute failed in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

# Utilitaires de base
def _get_memory_usage() -> float:
    """Récupère l'utilisation mémoire actuelle"""
    try:
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    except Exception:
        return 0.0

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système"""
    try:
        if not PSUTIL_AVAILABLE:
            return {
                'memory_percent': 0, 
                'memory_available_mb': 0, 
                'timestamp': time.time()
            }
            
        memory = psutil.virtual_memory()
        metrics = {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'timestamp': time.time()
        }
        return metrics
    except Exception as e:
        logger.log_structured("ERROR", f"Erreur métriques système: {str(e)}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def _safe_get(obj: Any, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    """Récupération sécurisée d'attributs imbriqués"""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current if current is not None else default

def _create_empty_plot(message: str, height: int = 400) -> go.Figure:
    """Crée un graphique vide avec message d'erreur"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#e74c3c"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#e74c3c",
        borderwidth=1
    )
    fig.update_layout(
        title="Visualisation non disponible",
        template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
        height=height,
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(240, 240, 240, 0.1)',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

def _format_metric_value(value: Any, precision: int = 3) -> str:
    """Formate une valeur métrique pour l'affichage"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    try:
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        
        if isinstance(value, (float, np.floating)):
            if abs(value) < 0.001 or abs(value) > 10000:
                return f"{value:.2e}"
            return f"{value:.{precision}f}"
        
        return str(value)
    except (ValueError, TypeError):
        return str(value)

def _export_plot_to_png(fig: go.Figure, width: int = 1200, height: int = 600) -> bytes:
    """Exporte un graphique Plotly en PNG"""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return img_bytes
    except Exception as e:
        logger.log_structured("ERROR", f"Export PNG échoué: {str(e)}")
        return b""

def _generate_color_palette(n_colors: int) -> List[str]:
    """Génère une palette de couleurs"""
    if MATPLOTLIB_AVAILABLE and n_colors <= 20:
        try:
            cmap = cm.get_cmap('viridis', n_colors)
            return [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                   for r, g, b, _ in cmap(np.linspace(0, 1, n_colors))]
        except Exception:
            pass
    
    # Fallback vers une palette prédéfinie
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Génération de couleurs supplémentaires si nécessaire
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors

# Classe principale de visualisation
class ModelEvaluationVisualizer:
    """
    Visualisateur robuste pour l'évaluation des modèles ML
    Gère les visualisations pour classification, régression et clustering
    """
    
    def __init__(self, ml_results: List[Dict[str, Any]]):
        self.ml_results = ml_results or []
        self.validation_result = self._validate_data()
        self._temp_dir = Path(tempfile.gettempdir()) / "ml_plots_cache"
        self._temp_dir.mkdir(exist_ok=True)
        self._plot_cache = {}
        
        logger.log_structured("INFO", "Visualizer initialisé", {
            "n_results": len(self.ml_results),
            "task_type": self.validation_result.get("task_type", "unknown")
        })

    @monitor_operation
    def _validate_data(self) -> Dict[str, Any]:
        """Valide les données d'entrée et extrait les métadonnées"""
        validation = {
            "has_results": False,
            "results_count": 0,
            "task_type": "unknown",
            "best_model": None,
            "successful_models": [],
            "failed_models": [],
            "errors": [],
            "warnings": [],
            "metrics_summary": {}
        }
        
        try:
            if not self.ml_results:
                validation["errors"].append("Aucun résultat ML fourni")
                return validation
            
            validation["results_count"] = len(self.ml_results)
            validation["has_results"] = True

            # Séparation modèles réussis/échoués
            for result in self.ml_results:
                if not isinstance(result, dict):
                    validation["warnings"].append("Résultat non-dictionnaire ignoré")
                    continue
                    
                has_error = _safe_get(result, ['metrics', 'error']) is not None
                model_name = _safe_get(result, ['model_name'], 'Unknown')
                
                if has_error:
                    validation["failed_models"].append(result)
                else:
                    validation["successful_models"].append(result)
            
            # Détermination du type de tâche
            if validation["successful_models"]:
                task_type = self._detect_task_type(validation["successful_models"])
                validation["task_type"] = task_type
                
                # Détermination du meilleur modèle
                validation["best_model"] = self._find_best_model(
                    validation["successful_models"], 
                    task_type
                )
                
                # Résumé des métriques
                validation["metrics_summary"] = self._compute_metrics_summary(
                    validation["successful_models"], 
                    task_type
                )
            
            logger.log_structured("INFO", "Validation données terminée", {
                "n_successful": len(validation['successful_models']),
                "n_failed": len(validation['failed_models']),
                "task_type": validation["task_type"]
            })
            
        except Exception as e:
            validation["errors"].append(f"Erreur validation: {str(e)}")
            logger.log_structured("ERROR", f"Erreur validation évaluation: {str(e)}")
        
        return validation

    @st.cache_data(ttl=3600, max_entries=10)
    def cached_plot(fig, plot_key: str):
        """Cache les graphiques avec gestion de taille - VERSION CORRIGÉE"""
        try:
            if fig is None:
                return None
            
            # Vérifier si c'est une figure Plotly valide
            if hasattr(fig, 'to_json'):
                return fig
            else:
                logger.log_structured("WARNING", f"Objet non-Plotly fourni au cache", {"plot_key": plot_key})
                return fig
                
        except Exception as e:
            logger.log_structured("ERROR", f"Erreur cache graphique", {
                "plot_key": plot_key, 
                "error": str(e)
            })
            return fig

    def _detect_task_type(self, successful_models: List[Dict]) -> str:
        """Détecte automatiquement le type de tâche - VERSION ULTRA ROBUSTE"""
        if not successful_models:
            return "unknown"
        
        # PRIORITÉ 1 : Utiliser le task_type stocké dans st.session_state (le plus fiable)
        if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            stored_task = getattr(st.session_state, 'task_type', None)
            if stored_task and stored_task in ['classification', 'regression', 'clustering']:
                logger.log_structured("INFO", f"Type tâche récupéré depuis session_state: {stored_task}")
                return stored_task
        
        # PRIORITÉ 2 : Détection par analyse de chaque modèle
        task_types = []
        for model in successful_models:
            task_type = _safe_get_model_task_type(model)
            if task_type != 'unknown':
                task_types.append(task_type)
        
        # Prendre le type le plus fréquent
        if task_types:
            from collections import Counter
            most_common = Counter(task_types).most_common(1)
            detected_type = most_common[0][0] if most_common else 'unknown'
            logger.log_structured("INFO", f"Type tâche détecté par analyse: {detected_type}")
            return detected_type
        
        # PRIORITÉ 3 : Fallback sur analyse du premier modèle
        first_model = successful_models[0]
        
        # Vérification structure des données
        has_labels = _safe_get(first_model, ['labels']) is not None
        has_y_test = _safe_get(first_model, ['y_test']) is not None
        
        if has_labels and not has_y_test:
            return 'clustering'
        
        if has_y_test:
            try:
                y_test = _safe_get(first_model, ['y_test'])
                y_array = np.array(y_test).ravel()
                unique_vals = np.unique(y_array)
                n_unique = len(unique_vals)
                n_total = len(y_array)
                
                # Classification si peu de valeurs uniques
                is_classification = (
                    n_unique <= 20 or
                    (n_unique / n_total < 0.1 and n_unique > 1)
                )
                
                return 'classification' if is_classification else 'regression'
            except Exception as e:
                logger.log_structured("ERROR", f"Erreur analyse y_test: {str(e)}")
        
        # PRIORITÉ 4 : Analyse des métriques
        metrics = _safe_get(first_model, ['metrics'], {})
        
        clustering_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        regression_metrics = ['r2', 'mse', 'mae', 'rmse']
        
        if any(m in metrics for m in clustering_metrics):
            return 'clustering'
        elif any(m in metrics for m in classification_metrics):
            return 'classification'
        elif any(m in metrics for m in regression_metrics):
            return 'regression'
        
        logger.log_structured("WARNING", "Type de tâche non détecté après toutes tentatives")
        return 'unknown'

    def _find_best_model(self, models: List[Dict], task_type: str) -> Optional[str]:
        """Trouve le meilleur modèle selon le type de tâche"""
        if not models:
            return None
            
        try:
            if task_type == 'classification':
                best_model = max(models, key=lambda x: (
                    _safe_get(x, ['metrics', 'accuracy'], 0),
                    _safe_get(x, ['metrics', 'f1'], 0)
                ))
            elif task_type == 'regression':
                best_model = max(models, key=lambda x: _safe_get(x, ['metrics', 'r2'], -999))
            elif task_type == 'clustering':
                best_model = max(models, key=lambda x: _safe_get(x, ['metrics', 'silhouette_score'], -999))
            else:
                best_model = models[0]
            
            return _safe_get(best_model, ['model_name'])
        except Exception as e:
            logger.log_structured("WARNING", f"Erreur recherche meilleur modèle: {str(e)}")
            return _safe_get(models[0], ['model_name'])

    def _compute_metrics_summary(self, models: List[Dict], task_type: str) -> Dict[str, Any]:
        """Calcule un résumé des métriques pour tous les modèles"""
        summary = {}
        
        try:
            if task_type == 'classification':
                accuracies = [_safe_get(m, ['metrics', 'accuracy'], 0) for m in models]
                f1_scores = [_safe_get(m, ['metrics', 'f1'], 0) for m in models]
                summary = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'f1_mean': float(np.mean(f1_scores)),
                    'f1_std': float(np.std(f1_scores))
                }
            elif task_type == 'regression':
                r2_scores = [_safe_get(m, ['metrics', 'r2'], 0) for m in models]
                rmse_scores = [_safe_get(m, ['metrics', 'rmse'], 0) for m in models]
                summary = {
                    'r2_mean': float(np.mean(r2_scores)),
                    'r2_std': float(np.std(r2_scores)),
                    'rmse_mean': float(np.mean(rmse_scores)),
                    'rmse_std': float(np.std(rmse_scores))
                }
            elif task_type == 'clustering':
                silhouette_scores = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in models]
                summary = {
                    'silhouette_mean': float(np.mean(silhouette_scores)),
                    'silhouette_std': float(np.std(silhouette_scores))
                }
        except Exception as e:
            logger.log_structured("WARNING", f"Erreur calcul résumé métriques: {str(e)}")
        
        return summary

    # Méthodes principales de visualisation
    @monitor_operation
    @timeout(seconds=60)
    def create_comparison_plot(self) -> go.Figure:
        """Crée un graphique de comparaison des modèles"""
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models:
                return _create_empty_plot("Aucun modèle valide à comparer")
            
            model_names = [_safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(successful_models)]
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                return self._create_classification_comparison(successful_models, model_names)
            elif task_type == 'regression':
                return self._create_regression_comparison(successful_models, model_names)
            elif task_type == 'clustering':
                return self._create_clustering_comparison(successful_models, model_names)
            else:
                return _create_empty_plot(f"Type de tâche '{task_type}' non supporté")
                
        except Exception as e:
            logger.log_structured("ERROR", f"Graphique comparaison échoué: {str(e)}")
            return _create_empty_plot(f"Erreur création graphique: {str(e)}")

    def _create_classification_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour classification"""
        metrics_data = {
            'Accuracy': [_safe_get(m, ['metrics', 'accuracy'], 0) for m in models],
            'F1-Score': [_safe_get(m, ['metrics', 'f1'], 0) for m in models],
            'Precision': [_safe_get(m, ['metrics', 'precision'], 0) for m in models],
            'Recall': [_safe_get(m, ['metrics', 'recall'], 0) for m in models]
        }
        
        fig = go.Figure()
        colors = _generate_color_palette(len(metrics_data))
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Bar(
                name=metric_name,
                x=model_names, 
                y=values,
                marker_color=colors[i],
                text=[f"{v:.3f}" for v in values],
                textposition='auto',
                hovertemplate=f"{metric_name}: %{{y:.3f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Comparaison des Modèles - Classification",
            xaxis_title="Modèles",
            yaxis_title="Score",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest'
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    def _create_regression_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour régression"""
        r2_scores = [_safe_get(m, ['metrics', 'r2'], 0) for m in models]
        rmse_scores = [_safe_get(m, ['metrics', 'rmse'], 0) for m in models]
        mae_scores = [_safe_get(m, ['metrics', 'mae'], 0) for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score (plus haut = mieux)', 'Erreurs (plus bas = mieux)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² scores
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R²', 
                   marker_color='#2ecc71', 
                   text=[f"{v:.3f}" for v in r2_scores],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Erreurs
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', 
                   marker_color='#e74c3c',
                   text=[f"{v:.3f}" for v in rmse_scores],
                   textposition='auto'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=model_names, y=mae_scores, name='MAE', 
                   marker_color='#f39c12',
                   text=[f"{v:.3f}" for v in mae_scores],
                   textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparaison des Modèles - Régression",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    def _create_clustering_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour clustering"""
        silhouette_scores = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in models]
        n_clusters = [_safe_get(m, ['metrics', 'n_clusters'], 0) for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score de Silhouette', 'Nombre de Clusters'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scores de silhouette avec code couleur
        colors_sil = []
        for score in silhouette_scores:
            if score > 0.5:
                colors_sil.append('#27ae60')  # Vert - bon
            elif score > 0.3:
                colors_sil.append('#f39c12')  # Orange - moyen
            else:
                colors_sil.append('#e74c3c')  # Rouge - faible
        
        fig.add_trace(
            go.Bar(x=model_names, y=silhouette_scores, name='Silhouette',
                   marker_color=colors_sil,
                   text=[f"{v:.3f}" for v in silhouette_scores],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Nombre de clusters
        fig.add_trace(
            go.Bar(x=model_names, y=n_clusters, name='Clusters',
                   marker_color='#3498db',
                   text=[f"{int(v)}" for v in n_clusters],
                   textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparaison des Modèles - Clustering",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    @monitor_operation
    @timeout(seconds=60)
    def create_performance_distribution(self) -> go.Figure:
        """Crée un histogramme de distribution des performances"""
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models:
                return _create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                values = [_safe_get(m, ['metrics', 'accuracy'], 0) for m in successful_models]
                title = "Distribution des Scores d'Accuracy"
                x_title = "Score d'Accuracy"
                color = '#3498db'
                
            elif task_type == 'regression':
                values = [_safe_get(m, ['metrics', 'r2'], 0) for m in successful_models]
                title = "Distribution des Scores R²"
                x_title = "Score R²"
                color = '#2ecc71'
                
            elif task_type == 'clustering':
                values = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in successful_models]
                title = "Distribution des Scores de Silhouette"
                x_title = "Score de Silhouette"
                color = '#9b59b6'
            else:
                return _create_empty_plot("Type de tâche non supporté")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=values, 
                nbinsx=min(15, len(values)), 
                marker_color=color,
                opacity=0.7,
                name="Distribution",
                hovertemplate="Score: %{x:.3f}<br>Fréquence: %{y}<extra></extra>"
            ))
            
            # Ligne de moyenne
            mean_val = np.mean(values)
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Moyenne: {mean_val:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Nombre de Modèles",
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                height=450,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.log_structured("ERROR", f"Distribution performances échouée: {str(e)}")
            return _create_empty_plot(f"Erreur: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_feature_importance_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée un graphique d'importance des features - VERSION CORRIGÉE"""
        try:
            model = _safe_get(model_result, ['model'])
            feature_names = _safe_get(model_result, ['feature_names'], [])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if model is None:
                return _create_empty_plot("Modèle non disponible")
            
            # Extraction du modèle final du pipeline si nécessaire
            actual_model = model
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                if pipeline_steps:
                    model_step = pipeline_steps[-1]
                    actual_model = model.named_steps[model_step]
            
            # Extraction des importances
            importances = None
            method_used = ""
            
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
                method_used = "Feature Importances"
            elif hasattr(actual_model, 'coef_'):
                coef = actual_model.coef_
                if coef.ndim == 1:
                    importances = np.abs(coef)
                else:
                    importances = np.mean(np.abs(coef), axis=0)
                method_used = "Coefficients"
            else:
                return _create_empty_plot("Importance des features non disponible pour ce modèle")
            
            if importances is None or len(importances) == 0:
                return _create_empty_plot("Impossible d'extraire l'importance des features")
            
            # Vérification que les importances ne sont pas toutes nulles
            if np.all(importances == 0):
                return _create_empty_plot("Toutes les importances sont nulles")
            
            # Création du DataFrame
            if not feature_names or len(feature_names) != len(importances):
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Limitation du nombre de features affichées
            top_n = min(VALIDATION_CONSTANTS.get("MAX_FEATURES_PLOT", 20), len(importance_df))
            importance_df = importance_df.tail(top_n)
            
            # Vérification que le DataFrame n'est pas vide
            if importance_df.empty:
                return _create_empty_plot("Aucune donnée d'importance après filtrage")
            
            # Création du graphique
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{imp:.4f}" for imp in importance_df['importance']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Features - {model_name}<br><sub>{method_used}</sub>",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(400, top_n * 25),
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                margin=dict(l=150, r=50, t=80, b=50)
            )
            
            logger.log_structured("INFO", f"Graphique importance créé", {
                "model": model_name,
                "n_features": top_n,
                "method": method_used
            })
            return fig
            
        except Exception as e:
            logger.log_structured("ERROR", f"Graphique importance features échoué: {str(e)}")
            return _create_empty_plot(f"Erreur importance features: {str(e)}")

    @monitor_operation
    @timeout(seconds=180)
    def create_shap_analysis(self, model_result: Dict[str, Any], max_samples: int = 1000) -> go.Figure:
        """Crée une analyse SHAP des features - VERSION CORRIGÉE"""
        if not SHAP_AVAILABLE:
            return _create_empty_plot("SHAP non disponible. Installez avec: pip install shap")
        
        try:
            model = _safe_get(model_result, ['model'])
            X_sample = _safe_get(model_result, ['X_sample'])  
            feature_names = _safe_get(model_result, ['feature_names'], [])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if model is None or X_sample is None:
                return _create_empty_plot("Données manquantes pour l'analyse SHAP")
            
            # Conversion robuste des données
            try:
                X_sample = np.array(X_sample)
                if X_sample.size == 0:
                    return _create_empty_plot("Données X_sample vides")
            except Exception as e:
                return _create_empty_plot(f"Erreur conversion données: {str(e)}")
            
            # Échantillonnage pour les performances
            n_samples = min(max_samples, len(X_sample))
            if n_samples < 2:
                return _create_empty_plot("Trop peu d'échantillons pour l'analyse SHAP")
                
            X_shap = X_sample[:n_samples]
            
            # Extraction du modèle du pipeline
            actual_model = model
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                if pipeline_steps:
                    actual_model = model.named_steps[pipeline_steps[-1]]
            
            # Vérification que le modèle peut faire des prédictions
            if not hasattr(actual_model, 'predict') and not hasattr(actual_model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas les prédictions")
            
            # Calcul des valeurs SHAP
            explainer = None
            shap_values = None
            
            try:
                # Essayer TreeExplainer pour les modèles tree-based
                if hasattr(actual_model, 'feature_importances_') or hasattr(actual_model, 'tree_'):
                    try:
                        explainer = shap.TreeExplainer(actual_model)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as tree_error:
                        logger.log_structured("WARNING", f"TreeExplainer échoué: {str(tree_error)}")
                        # Continuer avec d'autres explainers
                        
                # Essayer LinearExplainer pour les modèles linéaires
                if shap_values is None and hasattr(actual_model, 'coef_'):
                    try:
                        explainer = shap.LinearExplainer(actual_model, X_shap)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as linear_error:
                        logger.log_structured("WARNING", f"LinearExplainer échoué: {str(linear_error)}")
                
                # Fallback KernelExplainer
                if shap_values is None:
                    try:
                        background = shap.sample(X_shap, min(10, len(X_shap)))
                        if hasattr(actual_model, 'predict_proba'):
                            explainer = shap.KernelExplainer(actual_model.predict_proba, background)
                        else:
                            explainer = shap.KernelExplainer(actual_model.predict, background)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as kernel_error:
                        logger.log_structured("ERROR", f"KernelExplainer échoué: {str(kernel_error)}")
                        return _create_empty_plot(f"Erreur calcul SHAP: {str(kernel_error)[:100]}")
                
                # Gestion des formats de sortie SHAP
                if shap_values is None:
                    return _create_empty_plot("Impossible de calculer les valeurs SHAP")
                    
                if isinstance(shap_values, list):
                    # Prendre la première classe pour la classification binaire
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                return self._create_shap_summary_plot(shap_values, X_shap, feature_names, model_name)
                
            except Exception as shap_error:
                logger.log_structured("ERROR", f"Erreur calcul SHAP: {str(shap_error)}")
                return _create_empty_plot(f"Erreur calcul SHAP: {str(shap_error)[:100]}")
            
        except Exception as e:
            logger.log_structured("ERROR", f"Analyse SHAP échouée: {str(e)}")
            return _create_empty_plot(f"Erreur analyse SHAP: {str(e)}")

    def _create_shap_summary_plot(self, shap_values: np.ndarray, X: np.ndarray, 
                                feature_names: List[str], model_name: str) -> go.Figure:
        """Crée un graphique summary SHAP personnalisé"""
        try:
            if len(shap_values.shape) != 2:
                return _create_empty_plot("Format de valeurs SHAP incorrect")
                
            # Calcul de l'importance moyenne absolue
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Noms de features par défaut si manquants
            if not feature_names or len(feature_names) != shap_values.shape[1]:
                feature_names = [f'Feature_{i}' for i in range(shap_values.shape[1])]
            
            # Sélection des top features
            top_n = min(15, len(mean_abs_shap))
            top_indices = np.argsort(mean_abs_shap)[-top_n:]
            
            fig = go.Figure()
            
            for idx, feature_idx in enumerate(top_indices):
                feature_name = feature_names[feature_idx]
                shap_vals = shap_values[:, feature_idx]
                feature_vals = X[:, feature_idx]
                
                # Normalisation des valeurs de feature pour la couleur
                if len(np.unique(feature_vals)) > 1:
                    norm_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals))
                else:
                    norm_vals = np.zeros_like(feature_vals)
                
                fig.add_trace(go.Scatter(
                    x=shap_vals,
                    y=[idx] * len(shap_vals),
                    mode='markers',
                    marker=dict(
                        color=norm_vals,
                        colorscale='RdYlBu',
                        size=6,
                        opacity=0.6,
                        line=dict(width=0.5, color='white'),
                        colorbar=dict(title="Valeur Feature<br>(normalisée)", x=1.02) 
                    ),
                    name=feature_name,
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{feature_name}</b><br>'
                        f'SHAP: %{{x:.4f}}<br>'
                        f'Valeur: %{{marker.color:.3f}}<extra></extra>'
                    )
                ))
            
            fig.update_layout(
                title=f"SHAP Summary - {model_name}",
                xaxis_title="Valeur SHAP (impact sur la prédiction)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(top_indices))),
                    ticktext=[feature_names[i] for i in top_indices],
                    title="Features"
                ),
                height=600,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=False,
                margin=dict(l=150, r=100, t=80, b=50)
            )
            
            # Ligne verticale à zéro
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            return _create_empty_plot(f"Erreur création SHAP: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_confusion_matrix(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une matrice de confusion"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la matrice de confusion")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()

            # Prédictions
            y_pred = model.predict(X_test)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Noms des classes
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            class_names = [str(label) for label in unique_labels]

            # Création du heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale='Blues',
                hovertemplate=(
                    'Classe Réelle: %{y}<br>'
                    'Classe Prédite: %{x}<br>'
                    'Nombre: %{z}<extra></extra>'
                )
            ))

            fig.update_layout(
                title=f"Matrice de Confusion - {model_name}",
                xaxis_title="Classe Prédite",
                yaxis_title="Classe Réelle",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                annotations=[
                    dict(
                        x=xi, y=yi, text=str(val),
                        xref='x1', yref='y1',
                        font=dict(color='white' if val > cm.max() / 2 else 'black'),
                        showarrow=False
                    ) for yi, row in enumerate(cm) for xi, val in enumerate(row)
                ]
            )

            logger.log_structured("INFO", f"Matrice de confusion créée", {"model": model_name})
            return fig

        except Exception as e:
            logger.log_structured("ERROR", f"Matrice de confusion échouée: {str(e)}")
            return _create_empty_plot(f"Erreur matrice de confusion: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_roc_curve(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une courbe ROC"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe ROC")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas predict_proba")

            # Conversion des données
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()

            # Probabilités prédites
            y_score = model.predict_proba(X_test)

            # Courbe ROC (binary ou multi-class)
            if y_score.shape[1] == 2:
                # Cas binaire
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    line=dict(color='#2ecc71', width=3),
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ))
                
            else:
                # Cas multi-class
                fig = go.Figure()
                n_classes = y_score.shape[1]
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'Classe {i} (AUC = {roc_auc:.3f})',
                        hovertemplate=f'Classe {i}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
                    ))

            # Ligne diagonale de référence
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Aléatoire',
                showlegend=False
            ))

            fig.update_layout(
                title=f"Courbe ROC - {model_name}",
                xaxis_title="Taux de Faux Positifs (FPR)",
                yaxis_title="Taux de Vrais Positifs (TPR)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            logger.log_structured("INFO", f"Courbe ROC créée", {"model": model_name})
            return fig

        except Exception as e:
            logger.log_structured("ERROR", f"Courbe ROC échouée: {str(e)}")
            return _create_empty_plot(f"Erreur courbe ROC: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_cluster_visualization(self, model_result: Dict[str, Any]) -> go.Figure:
        """Visualisation 2D des clusters"""
        try:
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour la visualisation")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des données valides
            valid_mask = ~np.isnan(labels)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            # Réduction de dimension si nécessaire
            if X.shape[1] > 2 and SKLEARN_AVAILABLE:
                try:
                    # Essai PCA d'abord
                    pca = PCA(n_components=2, random_state=TRAINING_CONSTANTS["RANDOM_STATE"])
                    X_reduced = pca.fit_transform(X)
                    x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
                    y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
                except Exception:
                    # Fallback vers les deux premières features
                    X_reduced = X[:, :2]
                    x_label = "Feature 1"
                    y_label = "Feature 2"
            else:
                X_reduced = X[:, :2] if X.shape[1] >= 2 else X
                x_label = "Feature 1"
                y_label = "Feature 2" if X.shape[1] >= 2 else "Feature 1"
            
            unique_labels = np.unique(labels)
            colors = _generate_color_palette(len(unique_labels))
            
            fig = go.Figure()
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) == 0:
                    continue
                    
                if label == -1:
                    # Points de bruit
                    color = 'gray'
                    name = 'Bruit'
                    size = 6
                    opacity = 0.4
                else:
                    color = colors[i % len(colors)]
                    name = f'Cluster {int(label)}'
                    size = 8
                    opacity = 0.7
                
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], 
                    y=X_reduced[mask, 1] if X_reduced.shape[1] > 1 else np.zeros(np.sum(mask)),
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=size,
                        line=dict(width=0.5, color='white'),
                        opacity=opacity
                    ),
                    hovertemplate=(
                        f'<b>{name}</b><br>'
                        f'X: %{{x:.2f}}<br>'
                        f'Y: %{{y:.2f}}<extra></extra>'
                    )
                ))
            
            fig.update_layout(
                title=f"Visualisation des Clusters - {model_name}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.log_structured("ERROR", f"Visualisation clusters échouée: {str(e)}")
            return _create_empty_plot(f"Erreur visualisation clusters: {str(e)}")

    @monitor_operation
    @timeout(seconds=90)
    def create_silhouette_analysis(self, model_result: Dict[str, Any]) -> go.Figure:
        """Analyse silhouette pour le clustering"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour l'analyse silhouette")
                
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour l'analyse silhouette")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des données valides
            valid_mask = ~np.isnan(labels) & (labels != -1)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return _create_empty_plot("Au moins 2 clusters requis")
            
            # Calcul des scores silhouette
            silhouette_vals = silhouette_samples(X, labels)
            avg_score = silhouette_score(X, labels)
            
            fig = go.Figure()
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                cluster_sil = silhouette_vals[labels == label]
                cluster_sil.sort()
                
                cluster_size = len(cluster_sil)
                y_upper = y_lower + cluster_size
                
                color = _generate_color_palette(len(unique_labels))[i]
                
                fig.add_trace(go.Scatter(
                    x=cluster_sil,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    line=dict(width=2, color=color),
                    name=f'Cluster {int(label)} ({cluster_size} pts)',
                    fill='tozerox',
                    fillcolor=color,
                    opacity=0.7,
                    hovertemplate=(
                        f'Cluster {int(label)}<br>'
                        f'Score Silhouette: %{{x:.3f}}<extra></extra>'
                    )
                ))
                
                y_lower = y_upper + 10
            
            # Ligne du score moyen
            fig.add_vline(
                x=avg_score, 
                line_dash="dash", 
                line_color="red", 
                line_width=3,
                annotation_text=f"Score moyen: {avg_score:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Analyse Silhouette - {model_name}",
                xaxis_title="Coefficient de Silhouette",
                yaxis_title="Échantillons (par cluster)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.log_structured("ERROR", f"Analyse silhouette échouée: {str(e)}")
            return _create_empty_plot(f"Erreur analyse silhouette: {str(e)}")

    @monitor_operation
    @timeout(seconds=60)
    def create_residuals_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Graphique des résidus pour la régression"""
        try:
            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            # Prédictions et résidus
            y_pred = pd.Series(model.predict(X_test))
            residuals = y_test - y_pred

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color='#e74c3c', 
                    size=8, 
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name='Résidus',
                hovertemplate=(
                    'Prédiction: %{x:.3f}<br>'
                    'Résidu: %{y:.3f}<extra></extra>'
                )
            ))

            # Ligne horizontale à zéro
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.8)

            fig.update_layout(
                title=f"Analyse des Résidus - {model_name}",
                xaxis_title="Valeurs Prédites",
                yaxis_title="Résidus (Réel - Prédit)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.log_structured("ERROR", f"Graphique résidus échoué: {str(e)}")
            return _create_empty_plot(f"Erreur graphique résidus: {str(e)}")

    @monitor_operation
    @timeout(seconds=60)
    def create_predicted_vs_actual(self, model_result: Dict[str, Any]) -> go.Figure:
        """Graphique prédictions vs valeurs réelles"""
        try:
            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            # Prédictions
            y_pred = pd.Series(model.predict(X_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(
                    color='#2ecc71', 
                    size=8, 
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name='Prédictions',
                hovertemplate=(
                    'Réel: %{x:.3f}<br>'
                    'Prédit: %{y:.3f}<extra></extra>'
                )
            ))

            # Ligne y=x
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='y = x',
                showlegend=False
            ))

            fig.update_layout(
                title=f"Prédictions vs Réelles - {model_name}",
                xaxis_title="Valeurs Réelles",
                yaxis_title="Valeurs Prédites",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.log_structured("ERROR", f"Graphique prédictions vs réelles échoué: {str(e)}")
            return _create_empty_plot(f"Erreur graphique prédictions: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_learning_curve(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une courbe d'apprentissage"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe d'apprentissage")

            model = _safe_get(model_result, ['model'])
            X_train = _safe_get(model_result, ['X_train'])
            y_train = _safe_get(model_result, ['y_train'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_train is None or y_train is None:
                return _create_empty_plot("Données d'entraînement manquantes")

            # Conversion des données
            X_train = np.array(X_train)
            y_train = np.array(y_train).ravel()

            # Calcul de la courbe d'apprentissage
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=min(5, TRAINING_CONSTANTS.get("CV_FOLDS", 5)),
                n_jobs=1,  # Éviter les problèmes de parallélisation
                random_state=TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig = go.Figure()

            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_scores_mean,
                mode='lines+markers',
                name='Score entraînement',
                line=dict(color='#3498db', width=3),
                error_y=dict(
                    type='data',
                    array=train_scores_std,
                    visible=True,
                    color='#3498db',
                    thickness=1.5,
                    width=3
                )
            ))

            # Courbe de validation
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_scores_mean,
                mode='lines+markers',
                name='Score validation',
                line=dict(color='#e74c3c', width=3),
                error_y=dict(
                    type='data',
                    array=test_scores_std,
                    visible=True,
                    color='#e74c3c',
                    thickness=1.5,
                    width=3
                )
            ))

            fig.update_layout(
                title=f"Courbe d'Apprentissage - {model_name}",
                xaxis_title="Nombre d'échantillons d'entraînement",
                yaxis_title="Score",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig

        except Exception as e:
            logger.log_structured("ERROR", f"Courbe d'apprentissage échouée: {str(e)}")
            return _create_empty_plot(f"Erreur courbe d'apprentissage: {str(e)}")

    @monitor_operation
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame de comparaison des modèles"""
        comparison_data = []
        
        for result in self.ml_results:
            model_name = _safe_get(result, ['model_name'], 'Unknown')
            training_time = _safe_get(result, ['training_time'], 0)
            metrics = _safe_get(result, ['metrics'], {})
            
            has_error = _safe_get(metrics, ['error']) is not None
            
            row = {
                'Modèle': model_name,
                'Statut': '❌ Échec' if has_error else '✅ Succès',
                'Temps (s)': f"{training_time:.2f}" if isinstance(training_time, (int, float)) else 'N/A'
            }
            
            if not has_error:
                task_type = self.validation_result["task_type"]
                
                if task_type == 'classification':
                    row.update({
                        'Accuracy': _format_metric_value(_safe_get(metrics, ['accuracy'])),
                        'F1-Score': _format_metric_value(_safe_get(metrics, ['f1'])),
                        'Precision': _format_metric_value(_safe_get(metrics, ['precision'])),
                        'Recall': _format_metric_value(_safe_get(metrics, ['recall'])),
                        'AUC': _format_metric_value(_safe_get(metrics, ['auc']))
                    })
                elif task_type == 'regression':
                    row.update({
                        'R²': _format_metric_value(_safe_get(metrics, ['r2'])),
                        'MAE': _format_metric_value(_safe_get(metrics, ['mae'])),
                        'RMSE': _format_metric_value(_safe_get(metrics, ['rmse'])),
                        'MSE': _format_metric_value(_safe_get(metrics, ['mse']))
                    })
                elif task_type == 'clustering':
                    row.update({
                        'Silhouette': _format_metric_value(_safe_get(metrics, ['silhouette_score'])),
                        'N_Clusters': _format_metric_value(_safe_get(metrics, ['n_clusters'])),
                        'DB_Index': _format_metric_value(_safe_get(metrics, ['davies_bouldin_score']))
                    })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Nettoyage des types de données
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        logger.log_structured("INFO", "DataFrame comparaison généré", {"n_models": len(df)})
        return df

    @monitor_operation
    def get_export_data(self) -> Dict[str, Any]:
        """Prépare les données pour l'export"""
        try:
            models_data = []
            
            for result in self.validation_result["successful_models"]:
                model_data = {
                    "model_name": _safe_get(result, ["model_name"], "Unknown"),
                    "task_type": self.validation_result["task_type"],
                    "training_time": _safe_get(result, ["training_time"], 0),
                    "metrics": {}
                }
                
                metrics = _safe_get(result, ["metrics"], {})
                
                # Filtrage des métriques non-sérialisables
                for key, value in metrics.items():
                    if not isinstance(value, (dict, list, np.ndarray)) and key != 'error':
                        try:
                            if isinstance(value, (np.integer, np.floating)):
                                model_data["metrics"][key] = float(value)
                            else:
                                model_data["metrics"][key] = value
                        except (TypeError, ValueError):
                            continue
                
                models_data.append(model_data)
            
            export_data = {
                "export_timestamp": time.time(),
                "export_date": datetime.now().isoformat(),
                "task_type": self.validation_result["task_type"],
                "best_model": self.validation_result["best_model"],
                "total_models": len(self.validation_result["successful_models"]),
                "failed_models": len(self.validation_result["failed_models"]),
                "success_rate": len(self.validation_result["successful_models"]) / self.validation_result["results_count"] * 100 if self.validation_result["results_count"] > 0 else 0,
                "global_statistics": self.validation_result["metrics_summary"],
                "models": models_data,
                "system_info": get_system_metrics(),
                "warnings": self.validation_result["warnings"],
                "errors": self.validation_result["errors"]
            }
            
            logger.log_structured("INFO", "Données d'export préparées", {
                "n_models": len(models_data),
                "task_type": self.validation_result["task_type"]
            })
            
            gc.collect()
            return export_data
            
        except Exception as e:
            logger.log_structured("ERROR", f"Préparation données export échouée: {str(e)}")
            return {
                "error": str(e), 
                "export_timestamp": time.time(),
                "task_type": self.validation_result.get("task_type", "unknown")
            }

    def cleanup(self):
        """Nettoie les ressources temporaires"""
        try:
            # Nettoyage du cache
            self._plot_cache.clear()
            
            # Nettoyage des fichiers temporaires
            for file_path in self._temp_dir.glob("*.png"):
                try:
                    file_path.unlink()
                except Exception:
                    pass
            
            logger.log_structured("INFO", "Ressources nettoyées")
        except Exception as e:
            logger.log_structured("WARNING", f"Nettoyage partiellement échoué: {str(e)}")

# Fonctions utilitaires exportées
def create_model_comparison(ml_results: List[Dict[str, Any]]) -> go.Figure:
    """Fonction utilitaire pour créer un graphique de comparaison"""
    visualizer = ModelEvaluationVisualizer(ml_results)
    return visualizer.create_comparison_plot()

def create_feature_importance_plot(model_result: Dict[str, Any]) -> go.Figure:
    """Fonction utilitaire pour créer un graphique d'importance des features"""
    visualizer = ModelEvaluationVisualizer([model_result])
    return visualizer.create_feature_importance_plot(model_result)

# Export des symboles principaux
__all__ = [
    'ModelEvaluationVisualizer',
    'create_model_comparison',
    'create_feature_importance_plot',
    'get_system_metrics',
    'logger'
]

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Exemple d'utilisation
    sample_results = [
        {
            "model_name": "RandomForest",
            "training_time": 10.5,
            "metrics": {
                "accuracy": 0.85,
                "f1": 0.83,
                "precision": 0.84,
                "recall": 0.82
            }
        }
    ]
    
    visualizer = ModelEvaluationVisualizer(sample_results)
    fig = visualizer.create_comparison_plot()
    
    if fig:
        print("Graphique créé avec succès!")
    else:
        print("Échec de création du graphique")
    
    visualizer.cleanup()