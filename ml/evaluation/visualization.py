import os
import tempfile
from matplotlib import cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
import psutil
from typing import Dict, List, Any, Optional
from functools import wraps
import streamlit as st
import concurrent.futures
import uuid
import base64
from io import BytesIO

# Import conditionnel de SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)

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
                    logger.error(f"Timeout: {func.__name__} took too long (> {seconds}s)")
                    return None
        return wrapper
    return decorator

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système"""
    try:
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def monitor_evaluation_operation(func):
    """Décorateur pour monitorer les opérations d'évaluation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 5:
                logger.warning(f"Evaluation operation {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Evaluation operation {func.__name__} failed: {e}")
            return None
    return wrapper

class ModelEvaluationVisualizer:
    """Classe principale pour la visualisation des résultats d'évaluation des modèles ML"""
    
    def __init__(self, ml_results: List[Dict[str, Any]]):
        self.ml_results = ml_results or []
        self.validation_result = self._validate_data()
        self._temp_dir = os.path.join(tempfile.gettempdir(), "ml_temp_plots")
        os.makedirs(self._temp_dir, exist_ok=True)
    
    @monitor_evaluation_operation
    def _validate_data(self) -> Dict[str, Any]:
        """Valide les données d'évaluation avec gestion robuste"""
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
                    
                has_error = self._safe_get(result, ['metrics', 'error']) is not None
                model_name = self._safe_get(result, ['model_name'], 'Unknown')
                
                if has_error:
                    validation["failed_models"].append(result)
                else:
                    validation["successful_models"].append(result)
            
            # Détection du type de tâche améliorée
            if validation["successful_models"]:
                first_success = validation["successful_models"][0]
                task_type = self._safe_get(first_success, ['task_type'])
                
                if not task_type:
                    # Détection automatique basée sur les métriques
                    metrics = self._safe_get(first_success, ['metrics'], {})
                    
                    # Classification: présence de accuracy, precision, recall, f1_score
                    classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    if any(k in metrics for k in classification_metrics):
                        task_type = 'classification'
                    
                    # Régression: présence de r2, mae, rmse, mse
                    elif any(k in metrics for k in ['r2', 'mae', 'rmse', 'mse']):
                        task_type = 'regression'
                    
                    # Clustering: présence de silhouette_score, n_clusters
                    elif any(k in metrics for k in ['silhouette_score', 'n_clusters']):
                        task_type = 'clustering'
                    
                    # Détection par la présence de labels/predictions
                    elif self._safe_get(first_success, ['labels']) is not None:
                        task_type = 'clustering'
                    elif self._safe_get(first_success, ['y_pred']) is not None:
                        task_type = 'classification'  # Par défaut
                    else:
                        task_type = 'unknown'
                
                validation["task_type"] = task_type
            
            # Identification du meilleur modèle selon le type de tâche
            if validation["successful_models"]:
                successful = validation["successful_models"]
                
                if validation["task_type"] == 'classification':
                    # Critère principal: accuracy, puis f1_score
                    best = max(successful, key=lambda x: (
                        self._safe_get(x, ['metrics', 'accuracy'], 0),
                        self._safe_get(x, ['metrics', 'f1_score'], 0)
                    ))
                elif validation["task_type"] == 'regression':
                    # Critère principal: r2 score
                    best = max(successful, key=lambda x: self._safe_get(x, ['metrics', 'r2'], -999))
                elif validation["task_type"] == 'clustering':
                    # Critère principal: silhouette score
                    best = max(successful, key=lambda x: self._safe_get(x, ['metrics', 'silhouette_score'], -999))
                else:
                    # Type inconnu: prendre le premier modèle
                    best = successful[0]
                
                validation["best_model"] = self._safe_get(best, ['model_name'], 'Unknown')
            
            logger.info(f"Validation terminée: {len(validation['successful_models'])} modèles OK, {len(validation['failed_models'])} erreurs")
            
        except Exception as e:
            validation["errors"].append(f"Erreur validation: {str(e)}")
            logger.error(f"Evaluation validation error: {e}")
        
        return validation
    
    @staticmethod
    def _safe_get(obj: Any, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
        """Accès sécurisé aux données nested"""
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

    def _create_empty_plot(self, message: str) -> go.Figure:
        """Crée un graphique vide avec un message"""
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

    @monitor_evaluation_operation
    def get_comparison_plot(self) -> go.Figure:
        """Crée le graphique de comparaison des modèles selon le type de tâche"""
        try:
            successful_results = self.validation_result["successful_models"]
            
            if not successful_results:
                return self._create_empty_plot("Aucun modèle valide à comparer")
            
            model_names = [self._safe_get(r, ['model_name'], f'Modèle_{i}') 
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
                return self._create_empty_plot(f"Type de tâche '{task_type}' non supporté")
                
        except Exception as e:
            logger.error(f"Comparison plot creation failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)}")

    def _create_classification_comparison_plot(self, results: List, model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour la classification"""
        metrics_data = {
            'Accuracy': [self._safe_get(r, ['metrics', 'accuracy'], 0) for r in results],
            'F1-Score': [self._safe_get(r, ['metrics', 'f1_score'], 0) for r in results],
            'Precision': [self._safe_get(r, ['metrics', 'precision'], 0) for r in results],
            'Recall': [self._safe_get(r, ['metrics', 'recall'], 0) for r in results]
        }
        return self._create_metrics_bar_plot(metrics_data, model_names, 'Classification')

    def _create_regression_comparison_plot(self, results: List, model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour la régression"""
        # Pour la régression, on inverse MAE et RMSE (plus bas = mieux)
        r2_scores = [self._safe_get(r, ['metrics', 'r2'], 0) for r in results]
        mae_scores = [self._safe_get(r, ['metrics', 'mae'], 0) for r in results]
        rmse_scores = [self._safe_get(r, ['metrics', 'rmse'], 0) for r in results]
        
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
        silhouette_scores = [self._safe_get(r, ['metrics', 'silhouette_score'], 0) for r in results]
        n_clusters = [self._safe_get(r, ['metrics', 'n_clusters'], 0) for r in results]
        
        # Création d'un graphique combiné
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
        """Crée un DataFrame comparatif des modèles avec formatage conditionnel"""
        comparison_data = []
        
        for result in self.ml_results:
            model_name = self._safe_get(result, ['model_name'], 'Unknown')
            training_time = self._safe_get(result, ['training_time'], 0)
            metrics = self._safe_get(result, ['metrics'], {})
            
            has_error = self._safe_get(metrics, ['error']) is not None
            
            row = {
                'Modèle': model_name,
                'Statut': '❌ Échec' if has_error else '✅ Succès',
                'Temps (s)': f"{training_time:.2f}" if isinstance(training_time, (int, float)) else 'N/A'
            }
            
            if not has_error:
                task_type = self.validation_result["task_type"]
                
                # Métriques spécifiques par type de tâche
                if task_type == 'classification':
                    row.update({
                        'Accuracy': self._format_metric_value(self._safe_get(metrics, ['accuracy'])),
                        'F1-Score': self._format_metric_value(self._safe_get(metrics, ['f1_score'])),
                        'Precision': self._format_metric_value(self._safe_get(metrics, ['precision'])),
                        'Recall': self._format_metric_value(self._safe_get(metrics, ['recall']))
                    })
                elif task_type == 'regression':
                    row.update({
                        'R²': self._format_metric_value(self._safe_get(metrics, ['r2'])),
                        'MAE': self._format_metric_value(self._safe_get(metrics, ['mae'])),
                        'RMSE': self._format_metric_value(self._safe_get(metrics, ['rmse'])),
                        'MSE': self._format_metric_value(self._safe_get(metrics, ['mse']))
                    })
                elif task_type == 'clustering':
                    row.update({
                        'Silhouette': self._format_metric_value(self._safe_get(metrics, ['silhouette_score'])),
                        'N_Clusters': self._format_metric_value(self._safe_get(metrics, ['n_clusters'])),
                        'Inertie': self._format_metric_value(self._safe_get(metrics, ['inertia']))
                    })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Conversion explicite des colonnes pour éviter les erreurs PyArrow
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        return df
    
    def _format_metric_value(self, value: Any) -> str:
        """Formate les valeurs métriques avec gestion des cas spéciaux"""
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

    @monitor_evaluation_operation
    def get_performance_distribution_plot(self) -> go.Figure:
        """Crée un graphique de distribution des performances selon le type de tâche"""
        try:
            if not self.validation_result["successful_models"]:
                return self._create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                values = [self._safe_get(r, ['metrics', 'accuracy'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores d'Accuracy"
                x_title = "Score d'Accuracy"
                color = 'lightblue'
                
            elif task_type == 'regression':
                values = [self._safe_get(r, ['metrics', 'r2'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores R²"
                x_title = "Score R²"
                color = 'lightgreen'
                
            elif task_type == 'clustering':
                values = [self._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores de Silhouette"
                x_title = "Score de Silhouette"
                color = 'mediumpurple'
            else:
                return self._create_empty_plot("Type de tâche non supporté pour la distribution")
            
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
            logger.error(f"Cluster scatter plot failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)}")

    @monitor_evaluation_operation
    def create_silhouette_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Silhouette plot avec gestion robuste des données"""
        try:
            X = self._safe_get(model_result, ['X_sample'])
            labels = self._safe_get(model_result, ['labels'])
            model_name = self._safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return self._create_empty_plot("Données manquantes pour le silhouette plot")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des valeurs valides
            valid_mask = ~np.isnan(labels) & (labels != -1)  # Exclure le bruit
            if not np.any(valid_mask):
                return self._create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return self._create_empty_plot("Au moins 2 clusters requis pour le silhouette plot")
            
            # Calcul des scores silhouette
            try:
                silhouette_vals = silhouette_samples(X, labels)
                avg_score = silhouette_score(X, labels)
            except Exception as e:
                return self._create_empty_plot(f"Erreur calcul silhouette: {str(e)}")
            
            fig = go.Figure()
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                cluster_sil = silhouette_vals[labels == label]
                cluster_sil.sort()
                
                cluster_size = len(cluster_sil)
                y_upper = y_lower + cluster_size
                
                color_rgb = cm.nipy_spectral(float(i) / len(unique_labels))
                color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                
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
            logger.error(f"Silhouette plot failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)}")

    @monitor_evaluation_operation
    def create_feature_importance_plot(self, model, feature_names: List[str]) -> go.Figure:
        """Crée un graphique d'importance des features avec support étendu"""
        try:
            importances = None
            method_used = ""
            
            # Différentes méthodes pour extraire l'importance des features
            if hasattr(model, 'feature_importances_'):
                # Random Forest, Gradient Boosting, etc.
                importances = model.feature_importances_
                method_used = "Feature Importances (Tree-based)"
                
            elif hasattr(model, 'coef_'):
                # Linear models, SVM, etc.
                coef = model.coef_
                if coef.ndim > 1:
                    # Multi-class: moyenne des valeurs absolues
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
                method_used = "Coefficients (Linear model)"
                
            elif hasattr(model, 'dual_coef_') and hasattr(model, 'support_vectors_'):
                # SVM avec kernel
                return self._create_empty_plot("Feature importance non disponible pour SVM avec kernel")
                
            else:
                return self._create_empty_plot("Modèle ne supporte pas l'extraction d'importance des features")
            
            if importances is None or len(importances) != len(feature_names):
                return self._create_empty_plot("Incompatibilité entre features et importances")
            
            # Préparation des données
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Prendre les top N features pour la lisibilité
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
                height=max(400, top_n * 25),  # Hauteur adaptative
                template="plotly_white",
                margin=dict(l=150)  # Marge gauche pour les noms de features
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Feature importance plot failed: {e}")
            return self._create_empty_plot("Erreur importance features")

    @monitor_evaluation_operation  
    def create_shap_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée un SHAP summary plot si SHAP est disponible - VERSION CORRIGÉE"""
        if not SHAP_AVAILABLE:
            return self._create_empty_plot("SHAP n'est pas installé.\nInstallez avec: pip install shap")
        
        try:
            # Extraction des données nécessaires
            model = self._safe_get(model_result, ['model'])
            X_sample = self._safe_get(model_result, ['X_sample'])  
            feature_names = self._safe_get(model_result, ['feature_names'], [])
            
            # Essayer d'extraire le modèle depuis un pipeline si nécessaire
            if hasattr(model, 'named_steps'):
                # C'est un pipeline, extraire le modèle final
                pipeline_steps = list(model.named_steps.keys())
                model_step = pipeline_steps[-1]  # Dernier step = modèle
                actual_model = model.named_steps[model_step]
            else:
                actual_model = model
            
            if actual_model is None or X_sample is None:
                return self._create_empty_plot("Données manquantes pour l'analyse SHAP")
            
            X_sample = np.array(X_sample)
            
            # Limitation pour performance (SHAP peut être lent)
            n_samples = min(50, len(X_sample))
            X_shap = X_sample[:n_samples]
            
            # Sélection intelligente de l'explainer
            explainer = None
            shap_values = None
            
            try:
                # Essayer TreeExplainer pour les modèles tree-based
                if hasattr(actual_model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(actual_model)
                    shap_values = explainer.shap_values(X_shap)
                    
                # Essayer LinearExplainer pour les modèles linéaires  
                elif hasattr(actual_model, 'coef_'):
                    explainer = shap.LinearExplainer(actual_model, X_shap[:20])  # Background plus petit
                    shap_values = explainer.shap_values(X_shap)
                    
                # Fallback: KernelExplainer (plus lent mais universel)
                else:
                    background = shap.sample(X_shap, min(10, len(X_shap)))  # Échantillon background
                    explainer = shap.KernelExplainer(actual_model.predict_proba if hasattr(actual_model, 'predict_proba') else actual_model.predict, background)
                    shap_values = explainer.shap_values(X_shap[:20])  # Encore plus limité
                
                # Gestion des valeurs SHAP multi-classes
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    # Classification multi-classe: prendre la première classe
                    shap_values = shap_values[0]
                
                if shap_values is None:
                    return self._create_empty_plot("Impossible de calculer les valeurs SHAP")
                
                # Créer le plot personnalisé
                return self._create_custom_shap_plot(shap_values, X_shap[:len(shap_values)], feature_names)
                
            except Exception as shap_error:
                return self._create_empty_plot(f"Erreur SHAP: {str(shap_error)[:100]}...")
            
        except Exception as e:
            logger.error(f"SHAP plot failed: {e}")
            return self._create_empty_plot(f"Erreur analyse SHAP: {str(e)[:50]}...")

    def _create_custom_shap_plot(self, shap_values: np.ndarray, X: np.ndarray, feature_names: List[str]) -> go.Figure:
        """Crée un SHAP summary plot personnalisé avec Plotly - VERSION CORRIGÉE"""
        try:
            if len(shap_values.shape) != 2:
                return self._create_empty_plot("Format de valeurs SHAP incorrect")
                
            # Calcul de l'importance moyenne des features
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            if not feature_names or len(feature_names) != shap_values.shape[1]:
                feature_names = [f'Feature_{i}' for i in range(shap_values.shape[1])]
            
            # Top 15 features pour la lisibilité
            top_indices = np.argsort(mean_abs_shap)[-15:]
            
            fig = go.Figure()
            
            # Créer le scatter plot pour chaque feature
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
                    y=[idx] * len(feature_shap_vals),  # Position Y fixe pour chaque feature
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
            
            # Mise à jour des axes
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
            
            # Ligne verticale à x=0
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            return self._create_empty_plot(f"Erreur création SHAP plot: {str(e)}")

    def show_model_details(self, selected_result: Dict[str, Any], task_type: str):
        """Affiche les détails d'un modèle spécifique selon le type de tâche - VERSION CORRIGÉE"""
        try:
            model_name = self._safe_get(selected_result, ['model_name'], 'Unknown')
            st.subheader(f"🔍 Analyse Détaillée: {model_name}")
            
            # Informations générales du modèle - DONNÉES DYNAMIQUES PAR MODÈLE
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Informations Générales:**")
                
                # Temps d'entraînement spécifique au modèle sélectionné
                training_time = self._safe_get(selected_result, ['training_time'], 0)
                st.write(f"• **Temps d'entraînement:** {training_time:.2f}s")
                st.write(f"• **Type de tâche:** {task_type.title()}")
                
                # Paramètres spécifiques au modèle sélectionné
                model = self._safe_get(selected_result, ['model'])
                if model is not None:
                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        # Filtrer les paramètres principaux
                        main_params = {}
                        for key, value in params.items():
                            if not callable(value) and not key.startswith('_') and len(str(value)) < 50:
                                main_params[key] = value
                        
                        if main_params:
                            st.write("• **Paramètres du modèle:**")
                            for key, value in list(main_params.items())[:5]:  # Top 5 paramètres
                                st.write(f"  - {key}: {value}")
                
                # Score de validation croisée si disponible
                cv_score = self._safe_get(selected_result, ['cv_score'])
                if cv_score is not None:
                    st.write(f"• **Score CV:** {cv_score:.3f}")
            
            with col2:
                st.markdown("**📊 Métriques de Performance:**")
                
                # Métriques spécifiques au modèle sélectionné
                metrics = self._safe_get(selected_result, ['metrics'], {})
                if metrics:
                    # Filtrer et formater les métriques pour éviter les objets complexes
                    clean_metrics = {}
                    for k, v in metrics.items():
                        if not isinstance(v, (dict, list, np.ndarray)) and k != 'error':
                            if isinstance(v, (int, float, np.number)):
                                clean_metrics[k] = f"{float(v):.4f}" if abs(float(v)) < 1000 else f"{float(v):.2e}"
                            else:
                                clean_metrics[k] = str(v)
                    
                    if clean_metrics:
                        # Créer un DataFrame avec conversion de types appropriée
                        metrics_data = []
                        for key, value in clean_metrics.items():
                            metrics_data.append({'Métrique': str(key), 'Valeur': str(value)})
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, width=400, height=200)
                    else:
                        st.write("Aucune métrique numérique disponible")
                else:
                    st.write("Aucune métrique disponible")
            
            st.markdown("---")
            
            # Visualisations spécifiques selon le type de tâche
            if task_type == 'clustering':
                self._show_clustering_details(selected_result)
            elif task_type in ['classification', 'regression']:
                self._show_supervised_details(selected_result, task_type)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'affichage des détails: {str(e)}")
            logger.error(f"Model details display failed: {e}")

    def _show_clustering_details(self, selected_result: Dict[str, Any]):
        """Affiche les détails spécifiques au clustering"""
        st.markdown("### 🎯 Analyses de Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Visualisation des Clusters")
            cluster_plot = self.create_cluster_scatter_plot(selected_result)
            st.plotly_chart(cluster_plot, width='stretch')
        
        with col2:
            st.markdown("#### 📈 Analyse Silhouette")
            silhouette_plot = self.create_silhouette_plot(selected_result)
            st.plotly_chart(silhouette_plot, width='stretch')
        
        # Statistiques détaillées des clusters
        labels = self._safe_get(selected_result, ['labels'])
        if labels is not None:
            st.markdown("#### 📋 Statistiques des Clusters")
            
            labels = np.array(labels)
            valid_labels = labels[~np.isnan(labels)]
            
            if len(valid_labels) > 0:
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                
                cluster_stats = []
                for label, count in zip(unique_labels, counts):
                    if label == -1:
                        cluster_stats.append({
                            'Cluster': 'Bruit',
                            'Nombre d\'échantillons': int(count),
                            'Pourcentage': f"{count/len(valid_labels)*100:.1f}%"
                        })
                    else:
                        cluster_stats.append({
                            'Cluster': f'Cluster {int(label)}',
                            'Nombre d\'échantillons': int(count),
                            'Pourcentage': f"{count/len(valid_labels)*100:.1f}%"
                        })
                
                cluster_df = pd.DataFrame(cluster_stats)
                # Conversion explicite pour éviter les erreurs PyArrow
                for col in cluster_df.columns:
                    cluster_df[col] = cluster_df[col].astype(str)
                
                st.dataframe(cluster_df, width='stretch')

    def _show_supervised_details(self, selected_result: Dict[str, Any], task_type: str):
        """Affiche les détails pour classification et régression - VERSION AMÉLIORÉE"""
        st.markdown(f"### 🎯 Analyses de {task_type.title()}")
        
        # Ajout des courbes d'apprentissage
        st.markdown("#### 📈 Courbes d'Apprentissage")
        learning_curve_plot = self.create_learning_curve_plot(selected_result)
        st.plotly_chart(learning_curve_plot, width='stretch')
        
        # Analyses des features
        model = self._safe_get(selected_result, ['model'])
        feature_names = self._safe_get(selected_result, ['feature_names'], [])
        
        if model and feature_names:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Importance des Features")
                importance_plot = self.create_feature_importance_plot(model, feature_names)
                st.plotly_chart(importance_plot, width='stretch')
            
            with col2:
                st.markdown("#### 🔍 Analyse SHAP")
                if SHAP_AVAILABLE:
                    with st.spinner("Calcul des valeurs SHAP en cours..."):
                        shap_plot = self.create_shap_plot(selected_result)
                        st.plotly_chart(shap_plot, width='stretch')
                else:
                    st.info("📦 SHAP n'est pas installé.\n\nInstallez avec: `pip install shap`")
                    
                    # Affichage alternatif des coefficients/importances
                    if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                        st.markdown("**💡 Informations disponibles:**")
                        
                        if hasattr(model, 'feature_importances_'):
                            top_features = sorted(zip(feature_names, model.feature_importances_), 
                                                key=lambda x: abs(x[1]), reverse=True)[:10]
                            st.write("**Top 10 Features (Tree-based):**")
                            for name, importance in top_features:
                                st.write(f"• {name}: {importance:.4f}")
                                
                        elif hasattr(model, 'coef_'):
                            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                            top_features = sorted(zip(feature_names, coef), 
                                                key=lambda x: abs(x[1]), reverse=True)[:10]
                            st.write("**Top 10 Features (Coefficients):**")
                            for name, coef_val in top_features:
                                st.write(f"• {name}: {coef_val:.4f}")
        else:
            st.warning("⚠️ Données du modèle ou noms des features manquants")
        
        # Matrice de confusion pour la classification
        if task_type == 'classification':
            confusion_matrix = self._safe_get(selected_result, ['confusion_matrix'])
            if confusion_matrix is not None:
                st.markdown("#### 📊 Matrice de Confusion")
                self._show_confusion_matrix(confusion_matrix)

    def _show_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Affiche la matrice de confusion"""
        try:
            cm = np.array(confusion_matrix)
            
            # Création du heatmap avec Plotly
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="Matrice de Confusion",
                xaxis_title="Prédictions",
                yaxis_title="Valeurs Réelles",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Calcul des métriques par classe
            if cm.shape[0] == cm.shape[1]:  # Matrice carrée
                n_classes = cm.shape[0]
                class_metrics = []
                
                for i in range(n_classes):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_metrics.append({
                        'Classe': f'Classe {i}',
                        'Précision': f"{precision:.3f}",
                        'Rappel': f"{recall:.3f}",
                        'F1-Score': f"{f1:.3f}"
                    })
                
                st.markdown("**📊 Métriques par Classe:**")
                class_df = pd.DataFrame(class_metrics)
                # Conversion explicite pour éviter les erreurs PyArrow
                for col in class_df.columns:
                    class_df[col] = class_df[col].astype(str)
                st.dataframe(class_df, width='stretch')
                
        except Exception as e:
            st.error(f"Erreur affichage matrice de confusion: {str(e)}")

    def get_export_data(self) -> Dict[str, Any]:
        """Prépare les données pour l'export avec informations complètes"""
        try:
            models_data = []
            
            for result in self.validation_result["successful_models"]:
                model_data = {
                    "model_name": self._safe_get(result, ["model_name"], "Unknown"),
                    "task_type": self.validation_result["task_type"],
                    "training_time": self._safe_get(result, ["training_time"], 0),
                    "metrics": {}
                }
                
                # Extraction des métriques selon le type de tâche
                metrics = self._safe_get(result, ["metrics"], {})
                
                # Filtrer les métriques exportables (pas d'objets complexes)
                for key, value in metrics.items():
                    if not isinstance(value, (dict, list, np.ndarray)) and key != 'error':
                        try:
                            # Convertir en types JSON-compatibles
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
            logger.error(f"Export data preparation failed: {e}")
            return {
                "error": str(e), 
                "export_timestamp": time.time(),
                "task_type": self.validation_result.get("task_type", "unknown")
            } 
        except Exception as e:
            logger.error(f"Performance distribution plot failed: {e}")
            return self._create_empty_plot("Erreur création distribution")

    @monitor_evaluation_operation
    def create_learning_curve_plot(self, selected_result: Dict[str, Any]) -> go.Figure:
        """Crée les courbes d'apprentissage pour un modèle"""
        try:
            model = self._safe_get(selected_result, ['model'])
            X_train = self._safe_get(selected_result, ['X_train'])
            y_train = self._safe_get(selected_result, ['y_train'])
            model_name = self._safe_get(selected_result, ['model_name'], 'Modèle')
            
            if not all([model, X_train is not None, y_train is not None]):
                return self._create_empty_plot("Données d'entraînement manquantes pour les courbes d'apprentissage")
            
            # Limitation de la taille pour éviter les timeouts
            max_samples = min(2000, len(X_train))
            
            # Calcul des courbes d'apprentissage
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train[:max_samples], y_train[:max_samples],
                cv=3,
                n_jobs=1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42,
                scoring='accuracy' if self.validation_result["task_type"] == 'classification' else 'r2'
            )
            
            # Calcul des moyennes et écart-types
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Score d\'entraînement',
                line=dict(color='#2ecc71'),
                marker=dict(size=6)
            ))
            
            # Zone d'incertitude train
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Incertitude train',
                showlegend=False
            ))
            
            # Courbe de validation
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Score de validation',
                line=dict(color='#e74c3c'),
                marker=dict(size=6)
            ))
            
            # Zone d'incertitude validation
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Incertitude validation',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"Courbes d'Apprentissage - {model_name}",
                xaxis_title="Nombre d'échantillons d'entraînement",
                yaxis_title="Score",
                template="plotly_white",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Learning curve plot failed: {e}")
            return self._create_empty_plot(f"Erreur courbes d'apprentissage: {str(e)}")

    @monitor_evaluation_operation
    def create_cluster_scatter_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Scatter plot des clusters avec gestion robuste des données"""
        try:
            X = self._safe_get(model_result, ['X_sample'])
            labels = self._safe_get(model_result, ['labels'])
            model_name = self._safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return self._create_empty_plot("Données manquantes pour la visualisation")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Gestion des valeurs manquantes
            valid_mask = ~np.isnan(labels)
            if not np.any(valid_mask):
                return self._create_empty_plot("Aucune donnée valide pour la visualisation")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            # Réduction de dimension si nécessaire
            if X.shape[1] > 2:
                try:
                    pca = PCA(n_components=2, random_state=42)
                    X_reduced = pca.fit_transform(X)
                    x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
                    y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
                except Exception:
                    # Fallback: prendre les deux premières dimensions
                    X_reduced = X[:, :2]
                    x_label = "Feature 1"
                    y_label = "Feature 2"
            else:
                X_reduced = X
                x_label = "Feature 1"
                y_label = "Feature 2"
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])  # Exclure le bruit (-1)
            
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
                    color_rgb = cm.nipy_spectral(float(i) / max(1, n_clusters))
                    color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                    name = f'Cluster {int(label)}'
                    size = 8
                    opacity = 0.7
                
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], 
                    y=X_reduced[mask, 1],
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
            logger.error(f"Cluster scatter plot failed: {e}")
            return self._create_empty_plot(f"Erreur visualisation clusters: {str(e)}")