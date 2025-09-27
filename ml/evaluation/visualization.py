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
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import learning_curve
import streamlit as st
import concurrent.futures
import uuid
import base64
from io import BytesIO

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
        """Valide les données d'évaluation"""
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
            
            # Détection du type de tâche
            if validation["successful_models"]:
                first_success = validation["successful_models"][0]
                task_type = self._safe_get(first_success, ['task_type'])
                
                if not task_type:
                    metrics = self._safe_get(first_success, ['metrics'], {})
                    if any(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1_score']):
                        task_type = 'classification'
                    elif any(k in metrics for k in ['r2', 'mae', 'rmse', 'mse']):
                        task_type = 'regression'
                    elif any(k in metrics for k in ['silhouette_score', 'n_clusters']):
                        task_type = 'clustering'
                    else:
                        task_type = 'unknown'
                
                validation["task_type"] = task_type
            
            # Identification du meilleur modèle
            if validation["successful_models"]:
                successful = validation["successful_models"]
                
                if validation["task_type"] == 'classification':
                    best = max(successful, key=lambda x: self._safe_get(x, ['metrics', 'accuracy'], 0))
                elif validation["task_type"] == 'regression':
                    best = max(successful, key=lambda x: self._safe_get(x, ['metrics', 'r2'], -999))
                elif validation["task_type"] == 'clustering':
                    best = max(successful, key=lambda x: self._safe_get(x, ['metrics', 'silhouette_score'], -999))
                else:
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
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualisation non disponible",
            template="plotly_white",
            height=400
        )
        return fig

    @monitor_evaluation_operation
    def get_comparison_plot(self) -> go.Figure:
        """Crée le graphique de comparaison des modèles"""
        try:
            successful_results = self.validation_result["successful_models"]
            
            if not successful_results:
                return self._create_empty_plot("Aucun modèle valide à comparer")
            
            model_names = [self._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(successful_results)]
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                metrics_data = {
                    'Accuracy': [self._safe_get(r, ['metrics', 'accuracy'], 0) for r in successful_results],
                    'F1-Score': [self._safe_get(r, ['metrics', 'f1_score'], 0) for r in successful_results],
                    'Precision': [self._safe_get(r, ['metrics', 'precision'], 0) for r in successful_results]
                }
                return self._create_metrics_bar_plot(metrics_data, model_names, 'Classification')
                
            elif task_type == 'regression':
                metrics_data = {
                    'R² Score': [self._safe_get(r, ['metrics', 'r2'], 0) for r in successful_results],
                    'MAE': [self._safe_get(r, ['metrics', 'mae'], 0) for r in successful_results],
                    'RMSE': [self._safe_get(r, ['metrics', 'rmse'], 0) for r in successful_results]
                }
                return self._create_metrics_bar_plot(metrics_data, model_names, 'Régression')
                
            else:  # clustering
                silhouette_scores = [self._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                                   for r in successful_results]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=model_names, 
                    y=silhouette_scores,
                    marker_color='#9467bd',
                    text=[f"{v:.3f}" for v in silhouette_scores],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Comparaison des Scores de Silhouette - Clustering",
                    xaxis_title="Modèles",
                    yaxis_title="Score Silhouette",
                    height=450,
                    template="plotly_white"
                )
                fig.update_xaxes(tickangle=45)
                
                return fig
                
        except Exception as e:
            logger.error(f"Comparison plot creation failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)}")
    
    def _create_metrics_bar_plot(self, metrics_data: Dict, model_names: List[str], task_type: str) -> go.Figure:
        """Crée un graphique à barres pour les métriques"""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
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
            barmode='group'
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    @monitor_evaluation_operation
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Crée un DataFrame comparatif des modèles"""
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
                        'RMSE': self._format_metric_value(self._safe_get(metrics, ['rmse']))
                    })
                elif task_type == 'clustering':
                    row.update({
                        'Silhouette': self._format_metric_value(self._safe_get(metrics, ['silhouette_score'])),
                        'Clusters': self._format_metric_value(self._safe_get(metrics, ['n_clusters']))
                    })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _format_metric_value(self, value: Any) -> str:
        """Formate les valeurs métriques"""
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
        """Crée un graphique de distribution des performances"""
        try:
            if not self.validation_result["successful_models"]:
                return self._create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                values = [self._safe_get(r, ['metrics', 'accuracy'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores d'Accuracy"
                color = 'lightblue'
                
            elif task_type == 'regression':
                values = [self._safe_get(r, ['metrics', 'r2'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores R²"
                color = 'lightgreen'
                
            elif task_type == 'clustering':
                values = [self._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                         for r in self.validation_result["successful_models"]]
                title = "Distribution des Scores de Silhouette"
                color = 'purple'
            else:
                return self._create_empty_plot("Type de tâche non supporté")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=values, 
                nbinsx=10, 
                marker_color=color,
                opacity=0.7
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Score",
                yaxis_title="Nombre de Modèles",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Performance distribution plot failed: {e}")
            return self._create_empty_plot("Erreur création distribution")

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
                pca = PCA(n_components=2, random_state=42)
                X_reduced = pca.fit_transform(X)
                x_label = "Composante Principale 1"
                y_label = "Composante Principale 2"
            else:
                X_reduced = X
                x_label = "Feature 1"
                y_label = "Feature 2"
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            fig = go.Figure()
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) == 0:
                    continue
                    
                if label == -1:
                    color = 'gray'
                    name = 'Bruit'
                    size = 6
                else:
                    color_rgb = cm.nipy_spectral(float(i) / max(1, n_clusters))
                    color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                    name = f'Cluster {int(label)}'
                    size = 8
                
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], 
                    y=X_reduced[mask, 1],
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=size,
                        line=dict(width=0.5, color='white')
                    ),
                    opacity=0.7
                ))
            
            fig.update_layout(
                title=f"Visualisation des Clusters - {model_name}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template="plotly_white",
                showlegend=True
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
            valid_mask = ~np.isnan(labels)
            if not np.any(valid_mask):
                return self._create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return self._create_empty_plot("Au moins 2 clusters requis pour le silhouette plot")
            
            # Calcul des scores silhouette
            silhouette_vals = silhouette_samples(X, labels)
            avg_score = silhouette_score(X, labels)
            
            fig = go.Figure()
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # Ignorer le bruit
                    continue
                    
                cluster_sil = silhouette_vals[labels == label]
                cluster_sil.sort()
                
                y_upper = y_lower + len(cluster_sil)
                
                color_rgb = cm.nipy_spectral(float(i) / len(unique_labels))
                color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                
                fig.add_trace(go.Scatter(
                    x=cluster_sil,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    line=dict(width=1, color=color),
                    name=f'Cluster {int(label)}',
                    fill='tozerox'
                ))
                
                y_lower = y_upper + 10
            
            # Ligne de score moyen
            fig.add_shape(
                type="line",
                x0=avg_score, y0=0, x1=avg_score, y1=y_lower,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=avg_score, y=y_lower * 0.9,
                text=f"Score moyen: {avg_score:.3f}",
                showarrow=True,
                arrowhead=1,
                ax=-50,
                ay=0,
                bgcolor="red",
                font=dict(color="white")
            )
            
            fig.update_layout(
                title=f"Analyse Silhouette - {model_name}",
                xaxis_title="Coefficient Silhouette",
                yaxis_title="Cluster",
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Silhouette plot failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)}")

    def show_model_details(self, selected_result: Dict[str, Any], task_type: str):
        """Affiche les détails d'un modèle spécifique"""
        try:
            model_name = self._safe_get(selected_result, ['model_name'], 'Unknown')
            st.subheader(f"🔍 Analyse Détaillée: {model_name}")
            
            # Métriques principales
            metrics = self._safe_get(selected_result, ['metrics'], {})
            if metrics:
                st.write("**📊 Métriques de Performance:**")
                metrics_df = pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}])
                st.dataframe(metrics_df.T.rename(columns={0: 'Valeur'}), use_container_width=True)
            
            # Visualisations selon le type de tâche
            if task_type == 'clustering':
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(self.create_cluster_scatter_plot(selected_result), 
                                  use_container_width=True)
                with col2:
                    st.plotly_chart(self.create_silhouette_plot(selected_result), 
                                  use_container_width=True)
            
            elif task_type in ['classification', 'regression']:
                # Feature importance si disponible
                model = self._safe_get(selected_result, ['model'])
                feature_names = self._safe_get(selected_result, ['feature_names'], [])
                
                if model and feature_names:
                    st.plotly_chart(self.create_feature_importance_plot(model, feature_names), 
                                  use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'affichage des détails: {str(e)}")

    def create_feature_importance_plot(self, model, feature_names: List[str]) -> go.Figure:
        """Crée un graphique d'importance des features"""
        try:
            importances = None
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            
            if importances is None or len(importances) != len(feature_names):
                return self._create_empty_plot("Importance des features non disponible")
            
            # Création du graphique
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True).tail(15)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color=importance_df['importance'],
                colorscale='Viridis',
                text=[f"{imp:.3f}" for imp in importance_df['importance']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top 15 - Importance des Features",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=500,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Feature importance plot failed: {e}")
            return self._create_empty_plot("Erreur importance features")

    def get_export_data(self) -> Dict[str, Any]:
        """Prépare les données pour l'export"""
        try:
            models_data = []
            for result in self.validation_result["successful_models"]:
                models_data.append({
                    "model_name": self._safe_get(result, ["model_name"], "Unknown"),
                    "metrics": self._safe_get(result, ["metrics"], {}),
                    "training_time": self._safe_get(result, ["training_time"], 0),
                    "task_type": self.validation_result["task_type"]
                })
            
            return {
                "export_timestamp": time.time(),
                "task_type": self.validation_result["task_type"],
                "best_model": self.validation_result["best_model"],
                "total_models": len(self.validation_result["successful_models"]),
                "models": models_data
            }
        except Exception as e:
            return {"error": str(e)}