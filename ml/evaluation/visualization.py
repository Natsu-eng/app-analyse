import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
import psutil
from typing import Dict, List, Any
from functools import wraps
import shap
import platform
import sklearn
import shap
from sklearn.model_selection import learning_curve


logger = logging.getLogger(__name__)

class ModelEvaluationVisualizer:
    """Classe principale pour la visualisation des résultats d'évaluation des modèles ML"""
    
    def __init__(self, ml_results: List[Dict]):
        self.ml_results = ml_results
        self.validation_result = self._validate_data()
    
    def _validate_data(self) -> Dict[str, Any]:
        """Valide les données d'évaluation"""
        validation = {
            "has_results": False,
            "results_count": 0,
            "task_type": None,
            "best_model": None,
            "successful_models": [],
            "failed_models": [],
            "errors": []
        }
        
        try:
            if not self.ml_results or not isinstance(self.ml_results, list):
                validation["errors"].append("Aucun résultat ML valide")
                return validation
            
            validation["results_count"] = len(self.ml_results)
            validation["has_results"] = True

            # Guard rail mémoire
            mem = get_system_metrics()
            if mem['memory_percent'] > 90:
                logger.warning(f"⚠️ Validation lancée avec RAM critique: {mem['memory_percent']}%")
            
            # Séparation des modèles réussis/échoués
            for result in self.ml_results:
                if not isinstance(result, dict):
                    continue
                    
                has_error = self._safe_get(result, ['metrics', 'error']) is not None
                if has_error:
                    validation["failed_models"].append(result)
                else:
                    validation["successful_models"].append(result)
            
            # Détection du type de tâche
            if validation["successful_models"]:
                first_success = validation["successful_models"][0]
                task_type = self._safe_get(first_success, ['task_type'])
                
                if not task_type:
                    # Déduction depuis les métriques
                    metrics = self._safe_get(first_success, ['metrics'], {})
                    if 'accuracy' in metrics:
                        task_type = 'classification'
                    elif 'r2' in metrics:
                        task_type = 'regression'
                    elif 'silhouette_score' in metrics:
                        task_type = 'unsupervised'
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
                elif validation["task_type"] == 'unsupervised':
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
    def _safe_get(obj, keys, default=None):
        """Accès sécurisé aux données nested"""
        if obj is None:
            return default
        try:
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key, default)
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    return default
            return current if current is not None else default
        except (KeyError, TypeError, IndexError, AttributeError):
            return default
    
    def get_comparison_plot(self) -> go.Figure:
        """Crée le graphique de comparaison des modèles"""
        try:
            successful_results = [r for r in self.ml_results 
                                if not self._safe_get(r, ['metrics', 'error'])]
            
            if not successful_results:
                return self._create_empty_plot("Aucune métrique valide disponible")
            
            model_names = [self._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(successful_results)]
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                metrics_data = {
                    'Accuracy': [self._safe_get(r, ['metrics', 'accuracy'], 0) for r in successful_results],
                    'F1-Score': [self._safe_get(r, ['metrics', 'f1_score'], 0) for r in successful_results],
                    'AUC-ROC': [self._safe_get(r, ['metrics', 'roc_auc'], 0) for r in successful_results]
                }
                return self._create_subplots(metrics_data, model_names, 'Classification')
                
            elif task_type == 'regression':
                metrics_data = {
                    'R² Score': [self._safe_get(r, ['metrics', 'r2'], 0) for r in successful_results],
                    'MAE': [self._safe_get(r, ['metrics', 'mae'], 0) for r in successful_results],
                    'RMSE': [self._safe_get(r, ['metrics', 'rmse'], 0) for r in successful_results]
                }
                return self._create_subplots(metrics_data, model_names, 'Regression')
                
            else:  # unsupervised
                silhouette_scores = [self._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                                   for r in successful_results]
                n_clusters = [self._safe_get(r, ['metrics', 'n_clusters'], 0) 
                            for r in successful_results]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Score Silhouette', 'Nombre de Clusters')
                )
                
                fig.add_trace(
                    go.Bar(
                        x=model_names, 
                        y=silhouette_scores, 
                        name='Silhouette',
                        marker_color='#9467bd',
                        text=[f"{v:.3f}" for v in silhouette_scores],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=model_names, 
                        y=n_clusters, 
                        name='Clusters',
                        marker_color='#8c564b',
                        text=[f"{v:.0f}" for v in n_clusters],
                        textposition='outside'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title="Comparaison des Performances - UNSUPERVISED",
                    height=450,
                    showlegend=False,
                    template="plotly_white"
                )
                fig.update_xaxes(tickangle=45)
                
                return fig
                
        except Exception as e:
            logger.error(f"Comparison plot creation failed: {e}")
            return self._create_empty_plot(f"Erreur: {str(e)[:50]}")
    
    def _create_subplots(self, metrics_data: Dict, model_names: List[str], task_type: str) -> go.Figure:
        """Crée des subplots pour les métriques"""
        n_metrics = len(metrics_data)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=list(metrics_data.keys()),
            specs=[[{"secondary_y": False}] * n_metrics]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            fig.add_trace(
                go.Bar(
                    x=model_names, 
                    y=values, 
                    name=metric_name,
                    marker_color=colors[i % len(colors)],
                    text=[f"{v:.3f}" for v in values],
                    textposition='outside'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=f"Comparaison des Performances - {task_type.upper()}",
            height=450,
            showlegend=False,
            template="plotly_white"
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Crée un graphique vide avec un message d'erreur"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(title="Comparaison des Modèles")
        return fig
    
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
                        'Accuracy': self._format_metric_value(self._safe_get(metrics, ['accuracy']), 'accuracy'),
                        'F1-Score': self._format_metric_value(self._safe_get(metrics, ['f1_score']), 'f1_score'),
                        'Précision': self._format_metric_value(self._safe_get(metrics, ['precision']), 'precision'),
                        'Rappel': self._format_metric_value(self._safe_get(metrics, ['recall']), 'recall')
                    })
                elif task_type == 'regression':
                    row.update({
                        'R²': self._format_metric_value(self._safe_get(metrics, ['r2']), 'r2'),
                        'MAE': self._format_metric_value(self._safe_get(metrics, ['mae']), 'mae'),
                        'RMSE': self._format_metric_value(self._safe_get(metrics, ['rmse']), 'rmse')
                    })
                elif task_type == 'unsupervised':
                    row.update({
                        'Silhouette': self._format_metric_value(self._safe_get(metrics, ['silhouette_score']), 'silhouette_score'),
                        'Clusters': self._format_metric_value(self._safe_get(metrics, ['n_clusters']), 'n_clusters')
                    })
            else:
                error_msg = str(self._safe_get(metrics, ['error'], 'Erreur inconnue'))
                row['Erreur'] = error_msg[:50] + '...' if len(error_msg) > 50 else error_msg
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _format_metric_value(self, value, metric_name: str) -> str:
        """Formate les valeurs métriques"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        
        try:
            if isinstance(value, (int, np.integer)):
                return f"{value:,}"
            
            if isinstance(value, (float, np.floating)):
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'r2', 'silhouette_score']:
                    return f"{value:.3f}"
                elif metric_name in ['mse', 'mae', 'rmse']:
                    return f"{value:.4f}"
                else:
                    return f"{value:.3f}"
            
            return str(value)
        except (ValueError, TypeError):
            return str(value)
    
    def get_export_data(self) -> Dict[str, Any]:
        """Prépare les données pour l'export"""
        export_data = {
            'timestamp': time.time(),
            'task_type': self.validation_result["task_type"],
            'best_model': self.validation_result["best_model"],
            'total_models': len(self.ml_results),
            'successful_models': len(self.validation_result["successful_models"]),
            'failed_models': len(self.validation_result["failed_models"]),
            'metadata': {
                'python_version': platform.python_version(),
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'sklearn_version': sklearn.__version__,
                'platform': platform.platform()
            },
            'models': []
        }
        
        for result in self.ml_results:
            model_data = {
                'name': self._safe_get(result, ['model_name'], 'Unknown'),
                'training_time': self._safe_get(result, ['training_time'], 0),
                'parameters': self._safe_get(result, ['best_params'], {}),
                'metrics': self._safe_get(result, ['metrics'], {}),
                'has_error': self._safe_get(result, ['metrics', 'error']) is not None
            }
            export_data['models'].append(model_data)
        
        return export_data
    
    def create_shap_plot(self, model, X_sample, feature_names: List[str]) -> go.Figure:
        """Crée un graphique SHAP bar plot pour expliquer les features"""
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=True).tail(15)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(color=importance_df['importance'], colorscale='RdBu'),
                text=[f"{imp:.3f}" for imp in importance_df['importance']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top 15 - Importance des Features (SHAP)",
                xaxis_title="Impact Moyen Absolu",
                yaxis_title="Features",
                height=500,
                template="plotly_white"
            )
            return fig
        
        except Exception as e:
            logger.error(f"SHAP plot creation failed: {e}")
            return self._create_empty_plot("Erreur création SHAP")


    def create_learning_curve_plot(self, model, X, y, cv=5, scoring='accuracy') -> go.Figure:
        """Affiche les courbes d'apprentissage pour détecter overfitting/underfitting"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean, mode='lines+markers',
                name="Train", line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=val_mean, mode='lines+markers',
                name="Validation", line=dict(color='orange')
            ))
            
            fig.update_layout(
                title="Courbes d'Apprentissage",
                xaxis_title="Taille des données d'entraînement",
                yaxis_title=f"Score ({scoring})",
                template="plotly_white",
                height=500
            )
            return fig
        
        except Exception as e:
            logger.error(f"Learning curve plot failed: {e}")
            return self._create_empty_plot("Erreur courbes d'apprentissage")
        
    def create_confusion_matrix_plot(self, confusion_matrix: List[List[int]], class_names: List[str] = None) -> go.Figure:
        """Crée une heatmap de matrice de confusion"""
        try:
            if not confusion_matrix or not isinstance(confusion_matrix, (list, np.ndarray)):
                return self._create_empty_plot("Matrice de confusion non disponible")
            
            cm_array = np.array(confusion_matrix)
            
            if cm_array.size == 0:
                return self._create_empty_plot("Matrice de confusion vide")
            
            n_classes = cm_array.shape[0]
            
            if class_names is None or len(class_names) != n_classes:
                class_names = [f"Classe {i}" for i in range(n_classes)]
            
            # Normalisation pour les couleurs
            cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_array,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Nombre")
            ))
            
            # Annotations avec texte
            annotations = []
            for i in range(n_classes):
                for j in range(n_classes):
                    annotations.append(
                        dict(
                            x=class_names[j],
                            y=class_names[i],
                            text=str(cm_array[i, j]),
                            showarrow=False,
                            font=dict(
                                color="white" if cm_normalized[i, j] > 0.5 else "black",
                                size=12
                            )
                        )
                    )
            
            fig.update_layout(
                title="Matrice de Confusion",
                xaxis_title="Prédictions",
                yaxis_title="Valeurs Réelles",
                annotations=annotations,
                width=500,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Confusion matrix plot creation failed: {e}")
            return self._create_empty_plot("Erreur création matrice")
    
    def create_feature_importance_plot(self, model, feature_names: List[str]) -> go.Figure:
        """Crée un graphique d'importance des features"""
        try:
            importances = None
            
            # Extraction sécurisée de l'importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    importances = np.abs(coef).mean(axis=0)
                else:
                    importances = np.abs(coef)
            
            if importances is None:
                return self._create_empty_plot("Importance des features non disponible")
            
            if len(importances) != len(feature_names):
                return self._create_empty_plot("Incompatibilité taille features")
            
            # Tri et sélection des top features
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Limiter aux 15 plus importantes
            top_features = feature_importance_df.tail(15)
            
            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"{imp:.3f}" for imp in top_features['importance']],
                textposition='outside'
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
            logger.error(f"Feature importance plot creation failed: {e}")
            return self._create_empty_plot("Erreur importance features")
    
    def get_performance_distribution_plot(self) -> go.Figure:
        """Crée un graphique de distribution des performances"""
        try:
            if not self.validation_result["successful_models"]:
                return self._create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                accuracies = [self._safe_get(r, ['metrics', 'accuracy'], 0) 
                            for r in self.validation_result["successful_models"]]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=accuracies,
                    nbinsx=10,
                    name='Accuracy',
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Distribution des Scores d'Accuracy",
                    xaxis_title="Accuracy",
                    yaxis_title="Nombre de Modèles"
                )
                return fig
                
            elif task_type == 'regression':
                r2_scores = [self._safe_get(r, ['metrics', 'r2'], 0) 
                           for r in self.validation_result["successful_models"]]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=r2_scores,
                    nbinsx=10,
                    name='R² Score',
                    marker_color='lightgreen'
                ))
                fig.update_layout(
                    title="Distribution des Scores R²",
                    xaxis_title="R² Score",
                    yaxis_title="Nombre de Modèles"
                )
                return fig
            
            else:
                return self._create_empty_plot("Distribution non disponible pour ce type de tâche")
                
        except Exception as e:
            logger.error(f"Performance distribution plot failed: {e}")
            return self._create_empty_plot("Erreur création distribution")

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
            raise
    return wrapper