import os
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
from src.shared.logging import get_logger

# Configuration du logging
logger = logging.getLogger(__name__)

# Import conditionnel de dépendances optionnelles
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit non disponible - certaines fonctionnalités limitées")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP non disponible - analyse d'importance avancée limitée")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible - monitoring système limité")

try:
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib non disponible - certaines couleurs limitées")

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
    from sklearn.metrics.pairwise import euclidean_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn non disponible - fonctionnalités critiques manquantes")

# =============================
# Décorateurs de monitoring
# =============================

def timeout(seconds: int = 300):
    """Décorateur pour limiter le temps d'exécution des visualisations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Timeout: {func.__name__} took too long (> {seconds}s)")
                    return None
        return wrapper
    return decorator

def monitor_evaluation_operation(func):
    """Décorateur pour monitorer les opérations d'évaluation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            memory_used = _get_memory_usage() - start_memory
            
            if elapsed > 5:
                logger.warning(f"⏰ {func.__name__} took {elapsed:.2f}s, memory: {memory_used:+.1f}MB")
                
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} failed: {e}")
            return None
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
# Fonctions utilitaires
# =============================

def _get_memory_usage() -> float:
    """Obtient l'utilisation mémoire actuelle en MB"""
    try:
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    except:
        return 0.0

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système"""
    try:
        if not PSUTIL_AVAILABLE:
            return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}
            
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"❌ System metrics failed: {e}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def _safe_get(obj: Any, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    """
    Accès sécurisé aux données nested.
    
    Args:
        obj: Objet à explorer
        keys: Liste des clés à accéder
        default: Valeur par défaut
        
    Returns:
        Valeur trouvée ou default
    """
    if obj is None:
        return default
    
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

def _create_empty_plot(message: str) -> go.Figure:
    """
    Crée un graphique vide avec un message d'information.
    
    Args:
        message: Message à afficher
        
    Returns:
        Figure Plotly vide
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#e74c3c")
    )
    fig.update_layout(
        title="Visualisation non disponible",
        template="plotly_white",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def _format_metric_value(value: Any) -> str:
    """
    Formate les valeurs métriques avec gestion des cas spéciaux.
    
    Args:
        value: Valeur à formater
        
    Returns:
        Valeur formatée
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    try:
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        
        return str(value)
    except (ValueError, TypeError):
        return str(value)

def _export_plot_to_png(fig: go.Figure) -> bytes:
    """
    Convertit un graphique Plotly en PNG pour l'export.
    
    Args:
        fig: Figure Plotly à exporter
        
    Returns:
        Bytes du fichier PNG
    """
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        return img_bytes
    except Exception as e:
        logger.error(f"❌ Export PNG échoué: {e}")
        return b""

# =============================
# Classe principale
# =============================

class ModelEvaluationVisualizer:
    """
    Classe robuste pour la visualisation des résultats d'évaluation des modèles ML.
    Supporte classification, régression et clustering avec gestion d'erreurs avancée.
    """
    
    def __init__(self, ml_results: List[Dict[str, Any]]):
        """
        Args:
            ml_results: Liste des résultats des modèles ML
        """
        self.ml_results = ml_results or []
        self.validation_result = self._validate_data()
        self._temp_dir = os.path.join(tempfile.gettempdir(), "ml_temp_plots")
        os.makedirs(self._temp_dir, exist_ok=True)
        
        logger.info(f"🔧 Visualizer initialisé: {len(self.ml_results)} résultats")
    
    @monitor_evaluation_operation
    def _validate_data(self) -> Dict[str, Any]:
        """
        Valide les données d'évaluation avec gestion robuste.
        
        Returns:
            Résultat de validation
        """
        validation: Dict[str, Any] = {
            "has_results": False,
            "results_count": 0,
            "task_type": "unknown",
            "best_model": None,
            "successful_models": [],
            "failed_models": [],
            "errors": []
        }
        
        try:
            if not self.ml_results:
                validation["errors"].append("Aucun résultat ML fourni")
                return validation
            
            validation["results_count"] = len(self.ml_results)
            validation["has_results"] = True

            # Séparation des modèles réussis/échoués
            for result in self.ml_results:
                if not isinstance(result, dict):
                    continue
                    
                has_error = _safe_get(result, ['metrics', 'error']) is not None
                model_name = _safe_get(result, ['model_name'], 'Unknown')
                
                if has_error:
                    validation["failed_models"].append(result)
                else:
                    validation["successful_models"].append(result)
            
            # Détection du type de tâche
            if validation["successful_models"]:
                first_success = validation["successful_models"][0]
                task_type = _safe_get(first_success, ['task_type'])
                
                if not task_type:
                    # Détection automatique basée sur les métriques
                    metrics = _safe_get(first_success, ['metrics'], {})
                    
                    if any(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1_score']):
                        task_type = 'classification'
                    elif any(k in metrics for k in ['r2', 'mae', 'rmse', 'mse']):
                        task_type = 'regression'
                    elif any(k in metrics for k in ['silhouette_score', 'n_clusters']):
                        task_type = 'clustering'
                    elif _safe_get(first_success, ['labels']) is not None:
                        task_type = 'clustering'
                    else:
                        task_type = 'unknown'
                
                validation["task_type"] = task_type
            
            # Identification du meilleur modèle
            if validation["successful_models"]:
                successful = validation["successful_models"]
                
                if validation["task_type"] == 'classification':
                    best = max(successful, key=lambda x: (
                        _safe_get(x, ['metrics', 'accuracy'], 0),
                        _safe_get(x, ['metrics', 'f1_score'], 0)
                    ))
                elif validation["task_type"] == 'regression':
                    best = max(successful, key=lambda x: _safe_get(x, ['metrics', 'r2'], -999))
                elif validation["task_type"] == 'clustering':
                    best = max(successful, key=lambda x: _safe_get(x, ['metrics', 'silhouette_score'], -999))
                else:
                    best = successful[0]
                
                validation["best_model"] = _safe_get(best, ['model_name'], 'Unknown')
            
            logger.info(f"✅ Validation terminée: {len(validation['successful_models'])} modèles OK, {len(validation['failed_models'])} erreurs")
            
        except Exception as e:
            validation["errors"].append(f"Erreur validation: {str(e)}")
            logger.error(f"❌ Erreur validation évaluation: {e}")
        
        return validation

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def get_comparison_plot(self) -> go.Figure:
        """
        Crée le graphique de comparaison des modèles selon le type de tâche.
        
        Returns:
            Figure Plotly de comparaison
        """
        try:
            successful_results = self.validation_result["successful_models"]
            
            if not successful_results:
                return _create_empty_plot("Aucun modèle valide à comparer")
            
            model_names = [_safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(successful_results)]
            
            task_type = self.validation_result["task_type"]
            
            # Graphiques spécifiques par type de tâche
            if task_type == 'classification':
                return self._create_classification_comparison_plot(successful_results, model_names)
            elif task_type == 'regression':
                return self._create_regression_comparison_plot(successful_results, model_names)
            elif task_type == 'clustering':
                return self._create_clustering_comparison_plot(successful_results, model_names)
            else:
                return _create_empty_plot(f"Type de tâche '{task_type}' non supporté")
                
        except Exception as e:
            logger.error(f"❌ Création graphique comparaison échouée: {e}")
            return _create_empty_plot(f"Erreur: {str(e)}")

    def _create_classification_comparison_plot(self, results: List, model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour la classification"""
        metrics_data = {
            'Accuracy': [_safe_get(r, ['metrics', 'accuracy'], 0) for r in results],
            'F1-Score': [_safe_get(r, ['metrics', 'f1_score'], 0) for r in results],
            'Precision': [_safe_get(r, ['metrics', 'precision'], 0) for r in results],
            'Recall': [_safe_get(r, ['metrics', 'recall'], 0) for r in results]
        }
        return self._create_metrics_bar_plot(metrics_data, model_names, 'Classification')

    def _create_regression_comparison_plot(self, results: List, model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour la régression"""
        r2_scores = [_safe_get(r, ['metrics', 'r2'], 0) for r in results]
        mae_scores = [_safe_get(r, ['metrics', 'mae'], 0) for r in results]
        rmse_scores = [_safe_get(r, ['metrics', 'rmse'], 0) for r in results]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score (plus haut = mieux)', 'Erreurs (plus bas = mieux)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² scores
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R²', 
                  marker_color='#2ecc71', text=[f"{v:.3f}" for v in r2_scores]),
            row=1, col=1
        )
        
        # Erreurs (MAE et RMSE)
        fig.add_trace(
            go.Bar(x=model_names, y=mae_scores, name='MAE', 
                  marker_color='#e74c3c', text=[f"{v:.3f}" for v in mae_scores]),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', 
                  marker_color='#f39c12', text=[f"{v:.3f}" for v in rmse_scores]),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparaison des Performances - Régression",
            height=450,
            template="plotly_white",
            showlegend=True
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    def _create_clustering_comparison_plot(self, results: List, model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour le clustering"""
        silhouette_scores = [_safe_get(r, ['metrics', 'silhouette_score'], 0) for r in results]
        n_clusters = [_safe_get(r, ['metrics', 'n_clusters'], 0) for r in results]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score de Silhouette', 'Nombre de Clusters'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scores de silhouette
        colors_sil = ['#27ae60' if s > 0.5 else '#f39c12' if s > 0.3 else '#e74c3c' 
                     for s in silhouette_scores]
        
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
            title="Comparaison des Performances - Clustering",
            height=450,
            template="plotly_white",
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _create_metrics_bar_plot(self, metrics_data: Dict, model_names: List[str], task_type: str) -> go.Figure:
        """Crée un graphique à barres pour les métriques"""
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Bar(
                name=metric_name,
                x=model_names, 
                y=values,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=f"Comparaison des Performances - {task_type}",
            xaxis_title="Modèles",
            yaxis_title="Score",
            height=450,
            template="plotly_white",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    @monitor_evaluation_operation
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Crée un DataFrame comparatif des modèles avec formatage conditionnel.
        
        Returns:
            DataFrame de comparaison
        """
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
                        'F1-Score': _format_metric_value(_safe_get(metrics, ['f1_score'])),
                        'Precision': _format_metric_value(_safe_get(metrics, ['precision'])),
                        'Recall': _format_metric_value(_safe_get(metrics, ['recall']))
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
        
        # Conversion explicite des colonnes pour éviter les erreurs PyArrow
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        logger.info(f"✅ DataFrame comparaison généré: {len(df)} modèles")
        return df

    @monitor_evaluation_operation
    @timeout(seconds=120)
    def get_performance_distribution_plot(self) -> go.Figure:
        """
        Crée un graphique de distribution des performances selon le type de tâche.
        
        Returns:
            Figure Plotly de distribution
        """
        try:
            if not self.validation_result["successful_models"]:
                return _create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                values = [_safe_get(r, ['metrics', 'accuracy'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores d'Accuracy"
                x_title = "Score d'Accuracy"
                color = 'lightblue'
                
            elif task_type == 'regression':
                values = [_safe_get(r, ['metrics', 'r2'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores R²"
                x_title = "Score R²"
                color = 'lightgreen'
                
            elif task_type == 'clustering':
                values = [_safe_get(r, ['metrics', 'silhouette_score'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores de Silhouette"
                x_title = "Score de Silhouette"
                color = 'mediumpurple'
            else:
                return _create_empty_plot("Type de tâche non supporté pour la distribution")
            
            fig = go.Figure()
            
            # Histogramme
            fig.add_trace(go.Histogram(
                x=values, 
                nbinsx=min(10, len(values)), 
                marker_color=color,
                opacity=0.7,
                name="Distribution"
            ))
            
            # Ligne de moyenne
            mean_val = np.mean(values)
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Moyenne: {mean_val:.3f}")
            
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Fréquence",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Graphique distribution performances échoué: {e}")
            return _create_empty_plot(f"Erreur: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=90)
    def create_silhouette_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un silhouette plot pour l'analyse de clustering.
        
        Args:
            model_result: Résultat du modèle
            
        Returns:
            Figure Plotly de silhouette
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour le silhouette plot")
                
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour le silhouette plot")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des valeurs valides
            valid_mask = ~np.isnan(labels) & (labels != -1)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return _create_empty_plot("Au moins 2 clusters requis pour le silhouette plot")
            
            # Calcul des scores silhouette
            try:
                silhouette_vals = silhouette_samples(X, labels)
                avg_score = silhouette_score(X, labels)
            except Exception as e:
                return _create_empty_plot(f"Erreur calcul silhouette: {str(e)}")
            
            fig = go.Figure()
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                cluster_sil = silhouette_vals[labels == label]
                cluster_sil.sort()
                
                cluster_size = len(cluster_sil)
                y_upper = y_lower + cluster_size
                
                # Couleurs
                if MATPLOTLIB_AVAILABLE:
                    color_rgb = cm.nipy_spectral(float(i) / len(unique_labels))
                    color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                else:
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=cluster_sil,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    line=dict(width=2, color=color),
                    name=f'Cluster {int(label)} ({cluster_size} pts)',
                    fill='tozerox',
                    fillcolor=color,
                    opacity=0.7,
                    hovertemplate=f'Cluster {int(label)}<br>Silhouette: %{{x:.3f}}<extra></extra>'
                ))
                
                y_lower = y_upper + 10
            
            # Ligne de score moyen
            fig.add_vline(
                x=avg_score, 
                line_dash="dash", 
                line_color="red", 
                line_width=3,
                annotation_text=f"Score moyen: {avg_score:.3f}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title=f"Analyse Silhouette - {model_name}",
                xaxis_title="Coefficient de Silhouette",
                yaxis_title="Index des échantillons",
                height=500,
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Silhouette plot échoué: {e}")
            return _create_empty_plot(f"Erreur: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_feature_importance_plot(self, model, feature_names: List[str]) -> go.Figure:
        """
        Crée un graphique d'importance des features.
        
        Args:
            model: Modèle entraîné
            feature_names: Noms des features
            
        Returns:
            Figure Plotly d'importance des features
        """
        try:
            importances = None
            method_used = ""
            final_feature_names = feature_names
            
            # Extraction du modèle depuis un pipeline si nécessaire
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                model_step = pipeline_steps[-1]
                actual_model = model.named_steps[model_step]
                
                # Extraire les noms des features après prétraitement
                try:
                    if 'preprocessor' in model.named_steps:
                        preprocessor = model.named_steps['preprocessor']
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            final_feature_names = preprocessor.get_feature_names_out()
                            logger.info(f"✅ Noms des features extraits du préprocesseur: {len(final_feature_names)} features")
                        else:
                            logger.warning("⚠️ Le préprocesseur ne supporte pas get_feature_names_out")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'extraction des noms des features: {e}")
                    return _create_empty_plot(f"Erreur extraction noms des features: {str(e)[:100]}...")
            else:
                actual_model = model
            
            # Extraction de l'importance selon le type de modèle
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
                method_used = "Feature Importances (Tree-based)"
            elif hasattr(actual_model, 'coef_'):
                coef = actual_model.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
                method_used = "Coefficients (Linear model)"
            else:
                return _create_empty_plot("Modèle ne supporte pas l'extraction d'importance des features")
            
            if importances is None or len(importances) != len(final_feature_names):
                logger.error(f"❌ Incompatibilité: {len(importances) if importances is not None else 'None'} importances vs {len(final_feature_names)} features")
                return _create_empty_plot(f"Incompatibilité entre features ({len(final_feature_names)}) et importances ({len(importances) if importances is not None else 'None'})")
            
            # Préparation des données
            importance_df = pd.DataFrame({
                'feature': final_feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Top N features pour la lisibilité
            top_n = min(20, len(importance_df))
            importance_df = importance_df.tail(top_n)
            
            # Normalisation pour la couleur
            max_importance = importance_df['importance'].max()
            normalized_importance = importance_df['importance'] / max_importance if max_importance > 0 else importance_df['importance']
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(
                    color=normalized_importance,
                    colorscale='Viridis',
                    colorbar=dict(title="Importance Normalisée")
                ),
                text=[f"{imp:.4f}" for imp in importance_df['importance']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} - Importance des Features<br><sub>{method_used}</sub>",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(400, top_n * 25),
                template="plotly_white",
                margin=dict(l=150)
            )
            
            logger.info(f"✅ Graphique d'importance créé pour {method_used} avec {top_n} features")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Graphique importance features échoué: {e}")
            return _create_empty_plot(f"Erreur lors de la création du graphique d'importance: {str(e)[:100]}...")

    @monitor_evaluation_operation
    @timeout(seconds=180)
    def create_shap_plot(self, model_result: Dict[str, Any], max_samples: int = 100) -> go.Figure:
        """
        Crée un SHAP summary plot si SHAP est disponible.
        
        Args:
            model_result: Résultat du modèle contenant 'model', 'X_sample', et 'feature_names'
            max_samples: Nombre maximum d'échantillons à utiliser pour SHAP (défaut: 100)
            
        Returns:
            Figure Plotly SHAP
        """
        if not SHAP_AVAILABLE:
            return _create_empty_plot("SHAP n'est pas installé.\nInstallez avec: pip install shap")
        
        try:
            model = _safe_get(model_result, ['model'])
            X_sample = _safe_get(model_result, ['X_sample'])  
            feature_names = _safe_get(model_result, ['feature_names'], [])
            
            # Extraction du modèle depuis un pipeline si nécessaire
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                model_step = pipeline_steps[-1]
                actual_model = model.named_steps[model_step]
            else:
                actual_model = model
            
            if actual_model is None or X_sample is None:
                return _create_empty_plot("Données manquantes pour l'analyse SHAP")
            
            X_sample = np.array(X_sample)
            
            # Limitation pour performance
            n_samples = min(max_samples, len(X_sample))
            X_shap = X_sample[:n_samples]
            
            # Sélection de l'explainer
            explainer = None
            shap_values = None
            
            try:
                if hasattr(actual_model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(actual_model)
                    shap_values = explainer.shap_values(X_shap)
                    
                elif hasattr(actual_model, 'coef_'):
                    explainer = shap.LinearExplainer(actual_model, X_shap[:min(20, len(X_shap))])
                    shap_values = explainer.shap_values(X_shap)
                    
                else:
                    background = shap.sample(X_shap, min(10, len(X_shap)), random_state=42)
                    explainer = shap.KernelExplainer(
                        actual_model.predict_proba if hasattr(actual_model, 'predict_proba') else actual_model.predict, 
                        background
                    )
                    shap_values = explainer.shap_values(X_shap[:min(20, len(X_shap))])
                
                # Gestion des valeurs SHAP multi-classes
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    shap_values = shap_values[0]
                
                if shap_values is None:
                    return _create_empty_plot("Impossible de calculer les valeurs SHAP")
                
                return self._create_custom_shap_plot(shap_values, X_shap[:len(shap_values)], feature_names)
                
            except Exception as shap_error:
                logger.error(f"Erreur calcul SHAP : {str(shap_error)}")
                return _create_empty_plot(f"Erreur SHAP: {str(shap_error)[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ SHAP plot échoué: {e}")
            return _create_empty_plot(f"Erreur analyse SHAP: {str(e)[:50]}...")

    def _create_custom_shap_plot(self, shap_values: np.ndarray, X: np.ndarray, feature_names: List[str]) -> go.Figure:
        """Crée un SHAP summary plot personnalisé avec Plotly"""
        try:
            if len(shap_values.shape) != 2:
                return _create_empty_plot("Format de valeurs SHAP incorrect")
                
            # Calcul de l'importance moyenne des features
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            if not feature_names or len(feature_names) != shap_values.shape[1]:
                feature_names = [f'Feature_{i}' for i in range(shap_values.shape[1])]
            
            # Top 15 features pour la lisibilité
            top_indices = np.argsort(mean_abs_shap)[-15:]
            
            fig = go.Figure()
            
            for idx, i in enumerate(top_indices):
                feature_name = feature_names[i]
                feature_shap_vals = shap_values[:, i]
                feature_vals = X[:, i]
                
                # Normaliser les valeurs de features pour la couleur
                if len(np.unique(feature_vals)) > 1:
                    norm_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals))
                else:
                    norm_vals = np.zeros_like(feature_vals)
                
                fig.add_trace(go.Scatter(
                    x=feature_shap_vals,
                    y=[idx] * len(feature_shap_vals),
                    mode='markers',
                    marker=dict(
                        color=norm_vals,
                        colorscale='RdYlBu',
                        size=8,
                        opacity=0.7,
                        line=dict(width=0.5, color='white'),
                        colorbar=dict(title="Valeur Feature<br>(normalisée)", x=1.02) if idx == 0 else None
                    ),
                    name=feature_name,
                    showlegend=False,
                    hovertemplate=f'<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Valeur: {feature_vals[0]:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title="SHAP Summary Plot - Impact des Features sur les Prédictions",
                xaxis_title="Valeur SHAP (impact sur le modèle)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(top_indices))),
                    ticktext=[feature_names[i] for i in top_indices],
                    title="Features"
                ),
                height=600,
                template="plotly_white",
                showlegend=False
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            return _create_empty_plot(f"Erreur création SHAP plot: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=120)
    def create_cluster_scatter_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Scatter plot des clusters avec gestion robuste des données.
        
        Args:
            model_result: Résultat du modèle
            
        Returns:
            Figure Plotly de scatter plot
        """
        try:
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour la visualisation")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Gestion des valeurs manquantes
            valid_mask = ~np.isnan(labels)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide pour la visualisation")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            # Réduction de dimension si nécessaire
            if X.shape[1] > 2 and SKLEARN_AVAILABLE:
                try:
                    pca = PCA(n_components=2, random_state=42)
                    X_reduced = pca.fit_transform(X)
                    x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
                    y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
                except Exception:
                    X_reduced = X[:, :2]
                    x_label = "Feature 1"
                    y_label = "Feature 2"
            else:
                X_reduced = X[:, :2] if X.shape[1] >= 2 else X
                x_label = "Feature 1"
                y_label = "Feature 2" if X.shape[1] >= 2 else "Feature 1"
            
            unique_labels = np.unique(labels)
            
            fig = go.Figure()
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) == 0:
                    continue
                    
                if label == -1:
                    color = 'gray'
                    name = 'Bruit'
                    size = 6
                    opacity = 0.4
                else:
                    if MATPLOTLIB_AVAILABLE:
                        color_rgb = cm.nipy_spectral(float(i) / max(1, len(unique_labels)))
                        color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                    else:
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
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
                    hovertemplate=f'<b>{name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"Visualisation des Clusters - {model_name}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Scatter plot clusters échoué: {e}")
            return _create_empty_plot(f"Erreur visualisation clusters: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_confusion_matrix_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une matrice de confusion pour les tâches de classification."""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la matrice de confusion")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes (X_test, y_test ou modèle)")

            # Sécurisation de y_test en array 1D
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values.ravel()
            else:
                y_test = np.array(y_test).ravel()

            # Prédictions
            y_pred = model.predict(X_test)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            class_names = np.unique(y_test).astype(str)

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues',
                hovertemplate='Vrai: %{y}<br>Prédit: %{x}<br>Count: %{z}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Matrice de Confusion - {model_name}",
                xaxis_title="Classe Prédite",
                yaxis_title="Classe Réelle",
                height=500,
                template="plotly_white"
            )

            logger.info(f"✅ Matrice de confusion créée pour {model_name}")
            return fig

        except Exception as e:
            logger.error(f"❌ Matrice de confusion échouée: {e}")
            return _create_empty_plot(f"Erreur matrice de confusion: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_roc_curve_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une courbe ROC pour les tâches de classification binaire."""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe ROC")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes (X_test, y_test ou modèle)")

            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas predict_proba")

            # Sécurisation de y_test en array 1D
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values.ravel()
            else:
                y_test = np.array(y_test).ravel()

            # Probabilités
            y_score = model.predict_proba(X_test)

            if y_score.shape[1] == 2:  # binaire
                y_score = y_score[:, 1]
            else:
                return _create_empty_plot("Courbe ROC supportée uniquement pour classification binaire")

            # Courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color='#2ecc71', width=2),
                name=f'ROC (AUC = {roc_auc:.3f})',
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))

            fig.update_layout(
                title=f"Courbe ROC - {model_name}",
                xaxis_title="Taux de Faux Positifs (FPR)",
                yaxis_title="Taux de Vrais Positifs (TPR)",
                height=500,
                template="plotly_white"
            )

            logger.info(f"✅ Courbe ROC créée pour {model_name} avec AUC={roc_auc:.3f}")
            return fig

        except Exception as e:
            logger.error(f"❌ Courbe ROC échouée: {e}")
            return _create_empty_plot(f"Erreur courbe ROC: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_precision_recall_curve_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une courbe de précision-rappel pour les tâches de classification binaire."""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe PR")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes (X_test, y_test ou modèle)")

            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas predict_proba")

            # Sécurisation de y_test en array 1D
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values.ravel()
            else:
                y_test = np.array(y_test).ravel()

            # Probabilités
            y_score = model.predict_proba(X_test)

            if y_score.shape[1] == 2:  # binaire
                y_score = y_score[:, 1]
            else:
                return _create_empty_plot("Courbe PR supportée uniquement pour classification binaire")

            # Courbe PR
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            pr_auc = auc(recall, precision)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                line=dict(color='#3498db', width=2),
                name=f'PR (AUC = {pr_auc:.3f})',
                hovertemplate='Rappel: %{x:.3f}<br>Précision: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Courbe de Précision-Rappel - {model_name}",
                xaxis_title="Rappel",
                yaxis_title="Précision",
                height=500,
                template="plotly_white"
            )

            logger.info(f"✅ Courbe PR créée pour {model_name} avec AUC={pr_auc:.3f}")
            return fig

        except Exception as e:
            logger.error(f"❌ Courbe PR échouée: {e}")
            return _create_empty_plot(f"Erreur courbe PR: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_residuals_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphique des résidus pour les tâches de régression.
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour le graphique des résidus")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes (X_test, y_test ou modèle)")

            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            y_pred = pd.Series(model.predict(X_test))
            residuals = y_test - y_pred

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='#e74c3c', size=8, opacity=0.6),
                name='Résidus'
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title=f"Graphique des Résidus - {model_name}",
                xaxis_title="Valeurs Prédites",
                yaxis_title="Résidus (y_test - y_pred)",
                height=500,
                template="plotly_white",
                showlegend=True
            )

            return fig

        except Exception as e:
            return _create_empty_plot(f"Erreur graphique des résidus: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=60)
    def create_predicted_vs_actual_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphique de prédictions vs. réelles pour les tâches de régression.
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour le graphique prédictions vs. réelles")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes (X_test, y_test ou modèle)")

            # Convertir en DataFrame / Series pour compatibilité Plotly
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            y_pred = pd.Series(model.predict(X_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(color='#2ecc71', size=8, opacity=0.6),
                name='Prédictions'
            ))

            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='y=x',
                showlegend=False
            ))

            fig.update_layout(
                title=f"Prédictions vs. Réelles - {model_name}",
                xaxis_title="Valeurs Réelles",
                yaxis_title="Valeurs Prédites",
                height=500,
                template="plotly_white",
                showlegend=True
            )

            return fig

        except Exception as e:
            return _create_empty_plot(f"Erreur graphique prédictions vs. réelles: {str(e)}")

    @monitor_evaluation_operation
    @timeout(seconds=90)
    def create_intra_cluster_distance_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphique de dispersion intra-cluster pour le clustering.
        
        Args:
            model_result: Résultat du modèle avec X_sample et labels
            
        Returns:
            Figure Plotly de la dispersion intra-cluster
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour le graphique de dispersion intra-cluster")
                
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour la dispersion intra-cluster")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des valeurs valides
            valid_mask = ~np.isnan(labels) & (labels != -1)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return _create_empty_plot("Au moins 2 clusters requis")
            
            # Calculer la distance moyenne intra-cluster
            distances = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    dist = np.mean(euclidean_distances(cluster_points))
                else:
                    dist = 0
                distances.append(dist)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'Cluster {int(label)}' for label in unique_labels],
                y=distances,
                marker_color='#3498db',
                text=[f"{d:.3f}" for d in distances],
                textposition='auto',
                hovertemplate='Cluster: %{x}<br>Distance moyenne: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Dispersion Intra-Cluster - {model_name}",
                xaxis_title="Clusters",
                yaxis_title="Distance Moyenne Intra-Cluster",
                height=500,
                template="plotly_white",
                showlegend=False
            )
            
            logger.info(f"✅ Graphique de dispersion intra-cluster créé pour {model_name}")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Graphique de dispersion intra-cluster échoué: {e}")
            return _create_empty_plot(f"Erreur graphique de dispersion intra-cluster: {str(e)}")

    from sklearn.model_selection import learning_curve
    @monitor_evaluation_operation
    @timeout(seconds=300)  # Augmenter à 300s pour grands datasets
    def create_learning_curve_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée une courbe d'apprentissage pour un modèle supervisé.
        Args:
            model_result: Dictionnaire contenant 'model', 'X_train', 'y_train', 'task_type'
        Returns:
            Figure Plotly de la courbe d'apprentissage
        """
        try:
            model = _safe_get(model_result, ['model'])
            X_train = _safe_get(model_result, ['X_train'])
            y_train = _safe_get(model_result, ['y_train'])
            task_type = _safe_get(model_result, ['task_type'], 'classification')
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if not all([model, X_train is not None, y_train is not None]):
                logger.warning("Données manquantes pour la courbe d'apprentissage")
                return _create_empty_plot("Données manquantes (modèle, X_train ou y_train)")

            # Convertir en formats compatibles
            X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
            y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train.ravel()

            scoring = 'accuracy' if task_type == 'classification' else 'r2'

            train_sizes, train_scores, test_scores = learning_curve(
                estimator=model,
                X=X_train,
                y=y_train,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            fig = go.Figure()

            # Courbes principales
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean,
                mode="lines+markers",
                name="Score Entraînement",
                line=dict(color="#1f77b4"),
                error_y=dict(type="data", array=train_std, visible=True)
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_mean,
                mode="lines+markers",
                name="Score Validation",
                line=dict(color="#ff7f0e"),
                error_y=dict(type="data", array=test_std, visible=True)
            ))

            fig.update_layout(
                title=f"Courbe d'Apprentissage - {model_name}",
                xaxis_title="Taille de l'ensemble d'entraînement",
                yaxis_title=scoring.title(),
                template="plotly_white",
                height=500
            )

            logger.info(f"✅ Courbe d'apprentissage créée pour {model_name}")
            return fig

        except Exception as e:
            logger.error(f"❌ Courbe d'apprentissage échouée: {e}")
            return _create_empty_plot(f"Erreur création courbe d'apprentissage: {str(e)[:100]}...")
        
    @monitor_evaluation_operation
    @timeout(seconds=90)
    def create_predicted_proba_distribution_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphique de distribution des probabilités prédites pour un modèle de classification.
        """
        try:
            model = model_result.get('model')
            X_test = model_result.get('X_test')
            model_name = model_result.get('model_name', 'Modèle')

            if model is None or X_test is None:
                return _create_empty_plot("Données manquantes pour distribution des probabilités")
            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas predict_proba")

            # Convertir X_test en DataFrame si ce n'est pas déjà un DataFrame
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)

            y_proba = model.predict_proba(X_test)

            fig = go.Figure()
            # Binaire
            if y_proba.shape[1] == 2:
                fig.add_trace(go.Histogram(x=y_proba[:, 1], nbinsx=20, name='Classe 1', opacity=0.7, marker_color='#2ecc71'))
                fig.add_trace(go.Histogram(x=y_proba[:, 0], nbinsx=20, name='Classe 0', opacity=0.7, marker_color='#e74c3c'))
            # Multi-classes
            else:
                for i in range(y_proba.shape[1]):
                    fig.add_trace(go.Histogram(x=y_proba[:, i], nbinsx=20, name=f'Classe {i}', opacity=0.7))

            fig.update_layout(
                title=f"Distribution des probabilités prédites - {model_name}",
                xaxis_title="Probabilité prédite",
                yaxis_title="Nombre d'échantillons",
                barmode='overlay',
                template="plotly_white",
                height=500
            )
            return fig

        except Exception as e:
            return _create_empty_plot(f"Erreur distribution probabilités: {str(e)}")

    def create_feature_correlation_heatmap(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée une heatmap des corrélations entre features pour l'ensemble d'entraînement.
        Args:
            model_result: Dictionnaire contenant 'X_train', 'model_name'
        Returns:
            Figure Plotly de la matrice de corrélation
        """
        try:
            X_train = model_result.get('X_train')
            model_name = model_result.get('model_name', 'Modèle')

            if X_train is None:
                return _create_empty_plot("Données manquantes pour heatmap des corrélations")

            # Convertir en DataFrame si ce n'est pas déjà le cas
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)

            corr_matrix = X_train.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hovertemplate='Feature %{x} vs %{y}: %{z:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Heatmap des corrélations - {model_name}",
                template="plotly_white",
                height=600
            )

            logger.info(f"✅ Heatmap des corrélations créée pour {model_name}")
            return fig

        except Exception as e:
            logger.error(f"❌ Création heatmap corrélations échouée: {e}")
            return _create_empty_plot(f"Erreur heatmap corrélations: {str(e)}")
        
    
    @monitor_evaluation_operation
    @timeout(seconds=120)
    def create_elbow_curve_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée une courbe du coude pour KMeans dans le clustering.
        Args:
            model_result: Résultat du modèle avec 'X_sample', 'model_name', 'metrics'
        Returns:
            Figure Plotly de la courbe du coude
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe du coude")
            
            X = _safe_get(model_result, ['X_sample'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or model_name != 'K-Means':
                return _create_empty_plot("Données manquantes ou modèle non-KMeans")
            
            X = np.array(X)
            if X.shape[0] < 2:
                return _create_empty_plot("Trop peu de données pour la courbe du coude")
            
            from sklearn.cluster import KMeans
            inertias = []
            k_range = range(2, min(11, len(X)))
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(k_range),
                y=inertias,
                mode='lines+markers',
                line=dict(color='#3498db'),
                name='Inertie'
            ))
            
            n_clusters = _safe_get(model_result, ['metrics', 'n_clusters'])
            if n_clusters:
                idx = list(k_range).index(n_clusters) if n_clusters in k_range else None
                if idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[n_clusters], y=[inertias[idx]],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name='Clusters sélectionnés'
                    ))
            
            fig.update_layout(
                title=f"Courbe du Coude - {model_name}",
                xaxis_title="Nombre de Clusters",
                yaxis_title="Inertie",
                template="plotly_white",
                height=500
            )
            
            logger.info(f"✅ Courbe du coude créée pour {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Courbe du coude échouée: {e}")
            return _create_empty_plot(f"Erreur courbe du coude: {str(e)[:100]}...")


    @monitor_evaluation_operation
    def get_export_data(self) -> Dict[str, Any]:
        """
        Prépare les données pour l'export avec informations complètes.
        
        Returns:
            Données structurées pour export
        """
        try:
            models_data = []
            
            for result in self.validation_result["successful_models"]:
                model_data = {
                    "model_name": _safe_get(result, ["model_name"], "Unknown"),
                    "task_type": self.validation_result["task_type"],
                    "training_time": _safe_get(result, ["training_time"], 0),
                    "metrics": {}
                }
                
                # Extraction des métriques selon le type de tâche
                metrics = _safe_get(result, ["metrics"], {})
                
                # Filtrer les métriques exportables
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
            
            # Calcul de statistiques globales
            stats = {}
            if models_data:
                task_type = self.validation_result["task_type"]
                
                if task_type == 'classification' and models_data:
                    accuracies = [m["metrics"].get("accuracy", 0) for m in models_data]
                    stats = {
                        "mean_accuracy": float(np.mean(accuracies)),
                        "std_accuracy": float(np.std(accuracies)),
                        "best_accuracy": float(np.max(accuracies))
                    }
                elif task_type == 'regression' and models_data:
                    r2_scores = [m["metrics"].get("r2", 0) for m in models_data]
                    stats = {
                        "mean_r2": float(np.mean(r2_scores)),
                        "std_r2": float(np.std(r2_scores)),
                        "best_r2": float(np.max(r2_scores))
                    }
                elif task_type == 'clustering' and models_data:
                    silhouette_scores = [m["metrics"].get("silhouette_score", 0) for m in models_data]
                    stats = {
                        "mean_silhouette": float(np.mean(silhouette_scores)),
                        "std_silhouette": float(np.std(silhouette_scores)),
                        "best_silhouette": float(np.max(silhouette_scores))
                    }
            
            return {
                "export_timestamp": time.time(),
                "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_type": self.validation_result["task_type"],
                "best_model": self.validation_result["best_model"],
                "total_models": len(self.validation_result["successful_models"]),
                "failed_models": len(self.validation_result["failed_models"]),
                "success_rate": len(self.validation_result["successful_models"]) / self.validation_result["results_count"] * 100 if self.validation_result["results_count"] > 0 else 0,
                "global_statistics": stats,
                "models": models_data,
                "system_info": get_system_metrics()
            }
            
        except Exception as e:
            logger.error(f"❌ Préparation données export échouée: {e}")
            return {
                "error": str(e), 
                "export_timestamp": time.time(),
                "task_type": self.validation_result.get("task_type", "unknown")
            }

# Export des fonctions principales
__all__ = [
    'ModelEvaluationVisualizer',
    'get_system_metrics'
]