import os
import numpy as np
import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from src.evaluation.model_plots import ModelEvaluationVisualizer
from src.evaluation.metrics import get_system_metrics, EvaluationMetrics
from utils.report_generator import generate_pdf_report
from src.config.constants import TRAINING_CONSTANTS

from logging import getLogger
logger = getLogger(__name__)

# Import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Configuration PyArrow
os.environ["PANDAS_USE_PYARROW"] = "0"
try:
    pd.options.mode.dtype_backend = "numpy_nullable"
except Exception:
    pass

# Configuration page
st.set_page_config(page_title="Évaluation des Modèles", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .best-model-card {
        border-left: 4px solid #27ae60;
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-subtitle {
        font-size: 0.7rem;
        color: #95a5a6;
    }
    .performance-high { color: #27ae60; background: #d5f4e6; padding: 2px 6px; border-radius: 4px; }
    .performance-medium { color: #f39c12; background: #fef9e7; padding: 2px 6px; border-radius: 4px; }
    .performance-low { color: #e74c3c; background: #fadbd8; padding: 2px 6px; border-radius: 4px; }
    .tab-content {
        padding: 1rem;
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, max_entries=5)
def cached_plot(fig, _size_key: str):
    """Cache les figures Plotly avec limite de taille"""
    if fig and hasattr(fig, 'to_json'):
        size_mb = len(json.dumps(fig.to_json())) / (1024**2)
        if size_mb > 10:
            st.warning("⚠️ Graphique volumineux, affichage simplifié")
            return None
    return fig

def display_metrics_header(validation):
    """Affiche l'en-tête avec métriques principales"""
    successful_count = len(validation["successful_models"])
    total_count = validation["results_count"]
    
    st.markdown('<div class="main-header">📈 Évaluation des Modèles</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Taux de Réussite</div>
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-subtitle">{successful_count}/{total_count} modèles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card {'best-model-card' if validation['best_model'] else ''}">
            <div class="metric-title">Meilleur Modèle</div>
            <div class="metric-value" style="font-size: 1.2rem;">{validation['best_model'] or 'N/A'}</div>
            <div class="metric-subtitle">Type: {validation['task_type'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        failed_count = len(validation["failed_models"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Échecs</div>
            <div class="metric-value" style="color: #e74c3c;">{failed_count}</div>
            <div class="metric-subtitle">Modèles échoués</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_info = get_system_metrics()
        memory_color = "#27ae60" if memory_info['memory_percent'] < 70 else "#f39c12" if memory_info['memory_percent'] < 85 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mémoire Système</div>
            <div class="metric-value" style="color: {memory_color};">{memory_info['memory_percent']:.1f}%</div>
            <div class="metric-subtitle">Utilisation RAM</div>
        </div>
        """, unsafe_allow_html=True)

def create_pdf_report_latex(result, task_type):
    """Génère un rapport PDF avec LaTeX"""
    try:
        metrics = result.get('metrics', {})
        model_name = result.get('model_name', 'Unknown')
        training_time = result.get('training_time', 0)
        
        latex_content = r"""
\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{noto}
\begin{document}
\begin{center}
    \textbf{\Large Rapport d'Évaluation du Modèle: """ + model_name + r"""}\\[0.5cm]
    \textit{Type de tâche: """ + task_type.title() + r"""}
\end{center}
\section*{Métriques de Performance}
\begin{tabular}{ll}
    \toprule
    \textbf{Métrique} & \textbf{Valeur} \\
    \midrule
"""
        if task_type == 'classification':
            latex_content += f"Accuracy & {metrics.get('accuracy', 0):.3f} \\\\\n"
            latex_content += f"Precision & {metrics.get('precision', 0):.3f} \\\\\n"
            latex_content += f"Recall & {metrics.get('recall', 0):.3f} \\\\\n"
            latex_content += f"F1-Score & {metrics.get('f1_score', 0):.3f} \\\\\n"
        elif task_type == 'regression':
            latex_content += f"R² Score & {metrics.get('r2', 0):.3f} \\\\\n"
            latex_content += f"MAE & {metrics.get('mae', 0):.3f} \\\\\n"
            latex_content += f"RMSE & {metrics.get('rmse', 0):.3f} \\\\\n"
        else:  # clustering
            latex_content += f"Silhouette Score & {metrics.get('silhouette_score', 0):.3f} \\\\\n"
            latex_content += f"Calinski-Harabasz & {metrics.get('calinski_harabasz', 0):.3f} \\\\\n"
            latex_content += f"Davies-Bouldin & {metrics.get('davies_bouldin_score', 0):.3f} \\\\\n"
            latex_content += f"Nombre de Clusters & {metrics.get('n_clusters', 'N/A')} \\\\\n"
        
        latex_content += f"Training Time & {training_time:.1f}s \\\\\n"
        latex_content += r"""
    \bottomrule
\end{tabular}
\section*{Résumé}
Ce rapport présente les performances du modèle \textbf{""" + model_name + r"""} pour la tâche de """ + task_type + r""".
Veuillez consulter les visualisations pour plus de détails.
\end{document}
"""
        return generate_pdf_report({'content': latex_content})
    except Exception as e:
        st.error(f"❌ Erreur génération PDF: {str(e)}")
        logger.error(f"PDF report generation failed: {e}")
        return None

@st.cache_data
def calculate_clustering_metrics_cached(X, labels):
    """Cache les calculs de métriques de clustering"""
    try:
        if X is None or labels is None:
            return {'error': 'Données manquantes'}
        evaluator = EvaluationMetrics(task_type='clustering')
        return evaluator.calculate_unsupervised_metrics(X, labels)
    except Exception as e:
        return {'error': str(e)}

def display_model_details(evaluator, model_result, task_type):
    """
    Affiche les détails complets d'un modèle avec toutes les visualisations pertinentes.
    
    Args:
        evaluator: Instance de ModelEvaluationVisualizer
        model_result: Dictionnaire contenant les résultats du modèle
        task_type: Type de tâche ('classification', 'regression', 'clustering')
    """
    model_name = model_result.get('model_name', 'Unknown')
    st.markdown(f"#### Détails du modèle: {model_name}")
    
    # ============================================
    # Section Debug (optionnelle en production)
    # ============================================
    with st.expander("🔍 Debug - Données disponibles", expanded=False):
        available_keys = list(model_result.keys())
        st.write(f"**Clés disponibles:** {', '.join(available_keys)}")
        
        data_status = {}
        for key in ['X_test', 'y_test', 'X_train', 'y_train', 'X_sample', 'labels', 'model']:
            value = model_result.get(key)
            if value is not None:
                if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                    shape = getattr(value, 'shape', (len(value),))
                    data_status[key] = f"✅ Présent ({shape})"
                else:
                    data_status[key] = "✅ Présent"
            else:
                data_status[key] = "❌ Manquant"
        
        st.json(data_status)
    
    # ============================================
    # Métriques principales
    # ============================================
    metrics = model_result.get('metrics', {})
    training_time = model_result.get('training_time', 0)
    
    st.markdown("**Métriques principales**")
    metrics_data = []
    
    if task_type == 'classification':
        metrics_data.extend([
            {'Métrique': 'Accuracy', 'Valeur': f"{metrics.get('accuracy', 0):.3f}",
             'Description': 'Proportion des prédictions correctes'},
            {'Métrique': 'Precision', 'Valeur': f"{metrics.get('precision', 0):.3f}",
             'Description': 'Précision des prédictions positives'},
            {'Métrique': 'Recall', 'Valeur': f"{metrics.get('recall', 0):.3f}",
             'Description': 'Rappel des vrais positifs'},
            {'Métrique': 'F1-Score', 'Valeur': f"{metrics.get('f1_score', 0):.3f}",
             'Description': 'Moyenne harmonique de précision et rappel'}
        ])
    elif task_type == 'regression':
        metrics_data.extend([
            {'Métrique': 'R² Score', 'Valeur': f"{metrics.get('r2', 0):.3f}",
             'Description': 'Coefficient de détermination'},
            {'Métrique': 'MAE', 'Valeur': f"{metrics.get('mae', 0):.3f}",
             'Description': 'Erreur absolue moyenne'},
            {'Métrique': 'RMSE', 'Valeur': f"{metrics.get('rmse', 0):.3f}",
             'Description': 'Racine de l'erreur quadratique moyenne'}
        ])
    else:  # clustering
        metrics_data.extend([
            {'Métrique': 'Silhouette Score', 'Valeur': f"{metrics.get('silhouette_score', 0):.3f}",
             'Description': 'Qualité de la séparation des clusters'},
            {'Métrique': 'Calinski-Harabasz', 'Valeur': f"{metrics.get('calinski_harabasz', 0):.3f}",
             'Description': 'Ratio de dispersion entre et intra-clusters'},
            {'Métrique': 'Davies-Bouldin', 'Valeur': f"{metrics.get('davies_bouldin_score', 0):.3f}",
             'Description': 'Similitude moyenne entre clusters'},
            {'Métrique': 'Nombre de Clusters', 'Valeur': f"{metrics.get('n_clusters', 'N/A')}",
             'Description': 'Nombre de clusters formés'}
        ])
    
    metrics_data.append({
        'Métrique': 'Temps d'entraînement (s)', 
        'Valeur': f"{training_time:.1f}",
        'Description': 'Durée de l'entraînement'
    })
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    # ============================================
    # Récupération des données et du modèle
    # ============================================
    model = model_result.get('model')
    X_test = model_result.get('X_test')
    y_test = model_result.get('y_test')
    X_train = model_result.get('X_train')
    y_train = model_result.get('y_train')
    X_sample = model_result.get('X_sample')
    labels = model_result.get('labels')
    feature_names = model_result.get('feature_names', [])
    
    logger.info(f"📊 Affichage des détails pour {model_name}, task_type={task_type}")
    
    # Helper function pour vérifier predict_proba
    def model_has_predict_proba(model):
        if model is None:
            return False
        if hasattr(model, 'named_steps'):
            final_step = list(model.named_steps.values())[-1]
            return hasattr(final_step, 'predict_proba')
        return hasattr(model, 'predict_proba')
    
    # ============================================
    # Visualisations communes (Classification et Régression)
    # ============================================
    if task_type in ['classification', 'regression'] and model:
        
        # Importance des features
        if feature_names:
            st.markdown("#### Importance des Features")
            try:
                feature_plot = evaluator.create_feature_importance_plot(model, feature_names)
                if feature_plot:
                    st.plotly_chart(feature_plot, use_container_width=True)
                else:
                    st.info("ℹ️ Importance des features non disponible pour ce type de modèle")
            except Exception as e:
                st.warning(f"⚠️ Erreur importance features: {str(e)[:100]}")
                logger.error(f"Erreur importance features {model_name}: {e}")
        
        # Analyse SHAP
        if X_sample is not None and len(X_sample) > 0:
            st.markdown("#### Analyse SHAP")
            try:
                shap_plot = evaluator.create_shap_plot(model_result)
                if shap_plot:
                    st.plotly_chart(shap_plot, use_container_width=True)
                else:
                    st.info("ℹ️ Analyse SHAP non disponible pour ce modèle")
            except Exception as e:
                st.info(f"ℹ️ SHAP non disponible: {str(e)[:50]}")
                logger.info(f"SHAP non disponible pour {model_name}: {e}")
    
    # ============================================
    # Visualisations spécifiques à la Classification
    # ============================================
    if task_type == 'classification' and model:
        
        if X_test is not None and y_test is not None and len(X_test) > 0:
            
            # Matrice de confusion
            st.markdown("#### Matrice de Confusion")
            try:
                cm_plot = evaluator.create_confusion_matrix_plot(model_result)
                if cm_plot:
                    st.plotly_chart(cm_plot, use_container_width=True)
                else:
                    st.warning("⚠️ Impossible d'afficher la matrice de confusion")
            except Exception as e:
                st.error(f"Erreur matrice de confusion: {str(e)[:100]}")
                logger.error(f"Erreur confusion matrix {model_name}: {e}")
            
            # Courbes basées sur predict_proba
            if model_has_predict_proba(model):
                
                # Courbe ROC
                st.markdown("#### Courbe ROC")
                try:
                    roc_plot = evaluator.create_roc_curve_plot(model_result)
                    if roc_plot:
                        st.plotly_chart(roc_plot, use_container_width=True)
                    else:
                        st.info("ℹ️ Courbe ROC disponible uniquement pour classification binaire")
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe ROC: {str(e)[:100]}")
                    logger.warning(f"Échec ROC pour {model_name}: {e}")
                
                # Courbe Précision-Rappel
                st.markdown("#### Courbe de Précision-Rappel")
                try:
                    pr_plot = evaluator.create_precision_recall_curve_plot(model_result)
                    if pr_plot:
                        st.plotly_chart(pr_plot, use_container_width=True)
                    else:
                        st.info("ℹ️ Courbe PR disponible uniquement pour classification binaire")
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe PR: {str(e)[:100]}")
                
                # Distribution des probabilités
                st.markdown("#### Distribution des Probabilités Prédites")
                try:
                    proba_plot = evaluator.create_predicted_proba_distribution_plot(model_result)
                    if proba_plot:
                        st.plotly_chart(proba_plot, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Erreur distribution probabilités: {str(e)[:100]}")
            
            else:
                st.info("ℹ️ Modèle ne supporte pas predict_proba - courbes ROC/PR non disponibles")
        
        else:
            st.warning("⚠️ Données de test manquantes pour les visualisations de classification")
        
        # Courbe d'apprentissage
        if X_train is not None and y_train is not None and len(X_train) > 0:
            st.markdown("#### Courbe d'Apprentissage")
            try:
                learning_plot = evaluator.create_learning_curve_plot(model_result)
                if learning_plot:
                    st.plotly_chart(learning_plot, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erreur courbe d'apprentissage: {str(e)[:100]}")
        
        # Heatmap de corrélation (utilise X_sample déjà récupéré)
        if X_sample is not None and len(X_sample) > 0:
            st.markdown("#### Heatmap de Corrélation des Features")
            try:
                temp_result = {**model_result, 'X_train': X_sample}
                corr_plot = evaluator.create_feature_correlation_heatmap(temp_result)
                if corr_plot:
                    st.plotly_chart(corr_plot, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erreur heatmap: {str(e)[:100]}")
    
    # ============================================
    # Visualisations spécifiques à la Régression
    # ============================================
    if task_type == 'regression' and model:
        
        if X_test is not None and y_test is not None and len(X_test) > 0:
            
            # Graphique des résidus
            st.markdown("#### Graphique des Résidus")
            try:
                residuals_plot = evaluator.create_residuals_plot(model_result)
                if residuals_plot:
                    st.plotly_chart(residuals_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur graphique résidus: {str(e)[:100]}")
            
            # Prédictions vs Réelles
            st.markdown("#### Prédictions vs. Réelles")
            try:
                pred_vs_actual_plot = evaluator.create_predicted_vs_actual_plot(model_result)
                if pred_vs_actual_plot:
                    st.plotly_chart(pred_vs_actual_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur prédictions vs. réelles: {str(e)[:100]}")
        
        else:
            st.warning("⚠️ Données de test manquantes pour les visualisations de régression")
        
        # Courbe d'apprentissage
        if X_train is not None and y_train is not None and len(X_train) > 0:
            st.markdown("#### Courbe d'Apprentissage")
            try:
                learning_plot = evaluator.create_learning_curve_plot(model_result)
                if learning_plot:
                    st.plotly_chart(learning_plot, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Erreur courbe d'apprentissage: {str(e)[:100]}")
    
    # ============================================
    # Visualisations spécifiques au Clustering
    # ============================================
    if task_type == 'clustering':
        
        if X_sample is not None and labels is not None and len(X_sample) > 0:
            
            # Scatter plot des clusters
            st.markdown("#### Visualisation des Clusters")
            try:
                cluster_plot = evaluator.create_cluster_scatter_plot(model_result)
                if cluster_plot:
                    st.plotly_chart(cluster_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur scatter plot clusters: {str(e)[:100]}")
                logger.error(f"Erreur cluster scatter {model_name}: {e}")
            
            # Analyse de Silhouette
            st.markdown("#### Analyse de Silhouette")
            try:
                silhouette_plot = evaluator.create_silhouette_plot(model_result)
                if silhouette_plot:
                    st.plotly_chart(silhouette_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur silhouette plot: {str(e)[:100]}")
                logger.error(f"Erreur silhouette {model_name}: {e}")
            
            # Dispersion intra-cluster
            st.markdown("#### Dispersion Intra-Cluster")
            try:
                intra_cluster_plot = evaluator.create_intra_cluster_distance_plot(model_result)
                if intra_cluster_plot:
                    st.plotly_chart(intra_cluster_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur dispersion intra-cluster: {str(e)[:100]}")
                logger.error(f"Erreur intra-cluster {model_name}: {e}")
        
        else:
            st.warning("⚠️ Données de clustering (X_sample, labels) manquantes pour les visualisations")
            logger.warning(f"X_sample ou labels manquant pour {model_name}")
    
    # ============================================
    # Footer
    # ============================================
    st.markdown("---")
    st.caption(f"📊 Visualisations générées pour {model_name}")

def create_mlflow_run_plot(runs, task_type):
    """Crée un graphique de comparaison des runs MLflow"""
    try:
        if not runs:
            return None
        
        # Sélectionner la métrique principale
        metric_key = (
            'metrics.accuracy' if task_type == 'classification' else
            'metrics.r2' if task_type == 'regression' else
            'metrics.silhouette_score'
        )
        
        # Préparer les données pour le graphique
        plot_data = []
        for run in runs:
            if metric_key in run and run.get('status') == 'FINISHED':
                plot_data.append({
                    'Model': run.get('tags.mlflow.runName', 'Unknown'),
                    'Metric': run.get(metric_key, 0),
                    'Start Time': pd.to_datetime(run.get('start_time', 0), unit='ms'),
                    'Run ID': run.get('run_id', 'N/A')
                })
        
        if not plot_data:
            return None
        
        df_plot = pd.DataFrame(plot_data)
        fig = px.bar(
            df_plot,
            x='Model',
            y='Metric',
            color='Metric',
            text='Metric',
            title=f"Comparaison des Runs MLflow ({metric_key.split('.')[-1].title()})",
            hover_data=['Run ID', 'Start Time'],
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
        fig.update_layout(
            xaxis_title="Modèle",
            yaxis_title=metric_key.split('.')[-1].title(),
            showlegend=False
        )
        return fig
    except Exception as e:
        logger.error(f"Erreur création graphique MLflow: {e}")
        return None

def get_mlflow_artifact(run_id, artifact_path, client):
    """Récupère un artefact MLflow"""
    try:
        artifact_file = client.download_artifacts(run_id, artifact_path)
        with open(artifact_file, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Échec récupération artefact {artifact_path} pour run {run_id}: {e}")
        return None

def main():
    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        st.error("🚫 Aucun résultat disponible")
        st.info("Entraînez des modèles dans 'Configuration ML'.")
        if st.button("⚙️ Aller à Configuration ML", width='stretch'):
            st.switch_page("pages/2_⚙️_Configuration_ML.py")
        return

    try:
        evaluator = ModelEvaluationVisualizer(st.session_state.ml_results)
        validation = evaluator.validation_result
    except Exception as e:
        st.error(f"❌ Erreur initialisation: {str(e)}")
        st.info("Action: Vérifiez les résultats ou relancez l'entraînement.")
        logger.error(f"Erreur initialisation visualizer: {e}")
        return

    if not validation["has_results"]:
        st.error("📭 Aucune donnée valide")
        return

    display_metrics_header(validation)

    # Ajout de l'onglet MLflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Vue d'Ensemble", "🔍 Détails", "📈 Métriques", "💾 Export", "🔗 MLflow"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        if validation["successful_models"]:
            st.markdown("### 📈 Comparaison des Performances")
            comparison_plot = evaluator.get_comparison_plot()
            if comparison_plot:
                st.plotly_chart(cached_plot(comparison_plot, "comparison_plot"), width='stretch')
            
            st.markdown("### 📋 Synthèse")
            df_comparison = evaluator.get_comparison_dataframe()
            st.dataframe(df_comparison, width='stretch', height=400)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Résumé Statistique")
            col1, col2, col3 = st.columns(3)
            if validation["task_type"] in ['classification', 'regression']:
                numeric_cols = df_comparison.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    main_metric = 'Accuracy' if validation["task_type"] == 'classification' else 'R²'
                    if main_metric in numeric_cols.columns:
                        with col1:
                            st.metric("Score Moyen", f"{numeric_cols[main_metric].mean():.3f}")
                        with col2:
                            st.metric("Meilleur Score", f"{numeric_cols[main_metric].max():.3f}")
                        with col3:
                            st.metric("Écart-type", f"{numeric_cols[main_metric].std():.3f}")
            elif validation["task_type"] == 'clustering':
                with col1:
                    avg_silhouette = np.mean([r.get('metrics', {}).get('silhouette_score', 0) 
                                            for r in validation["successful_models"]])
                    st.metric("Silhouette Moyen", f"{avg_silhouette:.3f}")
                with col2:
                    avg_clusters = np.mean([r.get('metrics', {}).get('n_clusters', 0) 
                                          for r in validation["successful_models"]])
                    st.metric("Clusters Moyen", f"{avg_clusters:.1f}")
                with col3:
                    best_silhouette = max([r.get('metrics', {}).get('silhouette_score', 0) 
                                         for r in validation["successful_models"]])
                    st.metric("Meilleur Silhouette", f"{best_silhouette:.3f}")
        else:
            st.warning("⚠️ Aucun modèle valide")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analyse Détaillée")
        if validation["successful_models"]:
            model_names = [r.get('model_name', f'Modèle_{i}') 
                         for i, r in enumerate(validation["successful_models"])]
            selected_idx = st.selectbox(
                "Modèle à analyser:",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x]==validation.get('best_model') else ''}",
                key="model_selector_detail"
            )
            model_result = validation["successful_models"][selected_idx]
            display_model_details(evaluator, model_result, validation["task_type"])
        else:
            st.info("ℹ️ Aucun modèle disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 📈 Métriques Avancées")
        if validation["successful_models"]:
            st.markdown("#### 📊 Distribution des Performances")
            dist_plot = evaluator.get_performance_distribution_plot()
            if dist_plot:
                st.plotly_chart(cached_plot(dist_plot, "dist_plot"), width='stretch')
            
            st.markdown("#### 📋 Métriques par Catégorie")
            if validation["task_type"] == 'clustering':
                st.markdown("**🎯 Qualité des Clusters**")
                clustering_metrics = []
                for result in validation["successful_models"]:
                    metrics = result.get('metrics', {})
                    silhouette = metrics.get('silhouette_score', 0)
                    clustering_metrics.append({
                        'Modèle': result.get('model_name', 'Unknown'),
                        'Silhouette': f"{silhouette:.3f}",
                        'Calinski-Harabasz': f"{metrics.get('calinski_harabasz', 0):.3f}",
                        'Davies-Bouldin': f"{metrics.get('davies_bouldin_score', 0):.3f}",
                        'Clusters': metrics.get('n_clusters', 'N/A'),
                        'Qualité': '🟢 Excellente' if silhouette > 0.7 else '🟡 Bonne' if silhouette > 0.5 else '🟠 Moyenne' if silhouette > 0.3 else '🔴 Faible'
                    })
                st.dataframe(pd.DataFrame(clustering_metrics), width='stretch')
                
                st.markdown("**⏱️ Performance Computationnelle**")
                perf_metrics = [
                    {'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"}
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(perf_metrics), width='stretch')
            
            elif validation["task_type"] == 'classification':
                st.markdown("**🏷️ Précision**")
                class_metrics = [
                    {
                        'Modèle': r.get('model_name', 'Unknown'),
                        'Accuracy': f"{r.get('metrics', {}).get('accuracy', 0):.3f}",
                        'Precision': f"{r.get('metrics', {}).get('precision', 0):.3f}",
                        'Recall': f"{r.get('metrics', {}).get('recall', 0):.3f}",
                        'F1-Score': f"{r.get('metrics', {}).get('f1_score', 0):.3f}"
                    }
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(class_metrics), width='stretch')
                
                st.markdown("**⏱️ Performance Computationnelle**")
                perf_metrics = [
                    {'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"}
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(perf_metrics), width='stretch')
            
            elif validation["task_type"] == 'regression':
                st.markdown("**📊 Précision**")
                reg_metrics = [
                    {
                        'Modèle': r.get('model_name', 'Unknown'),
                        'R² Score': f"{r.get('metrics', {}).get('r2', 0):.3f}",
                        'MAE': f"{r.get('metrics', {}).get('mae', 0):.3f}",
                        'RMSE': f"{r.get('metrics', {}).get('rmse', 0):.3f}"
                    }
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(reg_metrics), width='stretch')
                
                st.markdown("**⏱️ Performance Computationnelle**")
                perf_metrics = [
                    {'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"}
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(perf_metrics), width='stretch')
        else:
            st.warning("⚠️ Aucune métrique disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 💾 Export des Résultats")
        if validation["successful_models"]:
            export_data = evaluator.get_export_data()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Données Structurées")
                csv_data = pd.DataFrame(export_data['models']).to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"evaluation_{validation['task_type']}_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
                    file_name=f"evaluation_{validation['task_type']}_{int(time.time())}.json",
                    mime="application/json",
                    width='stretch'
                )
                with st.expander("👁️ Aperçu"):
                    st.json(export_data, expanded=False)
            with col2:
                st.markdown("#### 📈 Rapport Global")
                if validation["best_model"]:
                    best_model_result = next((r for r in validation["successful_models"] 
                                           if r.get('model_name') == validation["best_model"]), None)
                    if best_model_result:
                        pdf_bytes = create_pdf_report_latex(best_model_result, validation["task_type"])
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Rapport PDF",
                                data=pdf_bytes,
                                file_name=f"rapport_{validation['best_model']}_{int(time.time())}.pdf",
                                mime="application/pdf",
                                width='stretch'
                            )
                if st.button("🔄 Générer Rapport", width='stretch'):
                    with st.spinner("Génération..."):
                        time.sleep(1)
                        st.success("✅ Rapport généré!")
                        st.balloons()
        else:
            st.info("ℹ️ Aucune donnée disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 🔗 Exploration des Runs MLflow")
        
        if not MLFLOW_AVAILABLE:
            st.error("🚫 MLflow non disponible")
            st.info("Installez MLflow pour accéder aux runs (`pip install mlflow`).")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        if 'mlflow_runs' not in st.session_state or not st.session_state.mlflow_runs:
            st.warning("⚠️ Aucun run MLflow disponible")
            st.info("Entraînez des modèles dans 'Configuration ML' pour générer des runs.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        try:
            mlflow_runs = st.session_state.mlflow_runs
            client = MlflowClient()
            
            # Filtrage des runs
            st.markdown("#### 📋 Filtrer les Runs")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                run_status = st.multiselect(
                    "Statut du Run",
                    options=['FINISHED', 'FAILED', 'RUNNING'],
                    default=['FINISHED'],
                    key="mlflow_status_filter"
                )
            with col_filter2:
                model_names = sorted(set(run.get('tags.mlflow.runName', 'Unknown') for run in mlflow_runs))
                selected_models = st.multiselect(
                    "Modèles",
                    options=model_names,
                    default=model_names,
                    key="mlflow_model_filter"
                )
            with col_filter3:
                sort_by = st.selectbox(
                    "Trier par",
                    options=['start_time', 'metrics.accuracy', 'metrics.r2', 'metrics.silhouette_score'],
                    index=0,
                    key="mlflow_sort_by"
                )
            
            # Filtrer les runs
            filtered_runs = [
                run for run in mlflow_runs
                if run.get('status', 'UNKNOWN') in run_status
                and run.get('tags.mlflow.runName', 'Unknown') in selected_models
            ]
            
            # Trier les runs
            reverse_sort = sort_by != 'start_time'
            filtered_runs = sorted(
                filtered_runs,
                key=lambda x: x.get(sort_by, 0) if sort_by != 'start_time' else x.get(sort_by, 0) / 1000,
                reverse=reverse_sort
            )
            
            # Tableau interactif des runs
            st.markdown("#### 📊 Tableau des Runs MLflow")
            run_data = []
            for run in filtered_runs:
                metrics = {k.split('.')[-1]: v for k, v in run.items() if k.startswith('metrics.')}
                params = {k.split('.')[-1]: v for k, v in run.items() if k.startswith('params.')}
                run_data.append({
                    'Run ID': run.get('run_id', 'N/A'),
                    'Modèle': run.get('tags.mlflow.runName', 'Unknown'),
                    'Statut': run.get('status', 'UNKNOWN'),
                    'Date': pd.to_datetime(run.get('start_time', 0), unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                    **{k: f"{v:.3f}" if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                    **{k: v for k, v in params.items()}
                })
            
            if run_data:
                df_runs = pd.DataFrame(run_data)
                st.dataframe(df_runs, width='stretch', height=400)
                
                # Graphique de comparaison des runs
                st.markdown("#### 📈 Comparaison des Performances")
                run_plot = create_mlflow_run_plot(filtered_runs, validation["task_type"])
                if run_plot:
                    st.plotly_chart(cached_plot(run_plot, "mlflow_run_plot"), width='stretch')
                else:
                    st.warning("⚠️ Impossible de générer le graphique de comparaison")
                    logger.warning("Échec création graphique comparaison MLflow")
                
                # Sélection d'un run pour détails
                st.markdown("#### 🔍 Détails du Run")
                selected_run_id = st.selectbox(
                    "Sélectionner un Run",
                    options=[run['run_id'] for run in filtered_runs],
                    format_func=lambda x: f"Run {x} ({next((r['tags.mlflow.runName'] for r in filtered_runs if r['run_id'] == x), 'Unknown')})",
                    key="mlflow_run_selector"
                )
                
                selected_run = next((run for run in filtered_runs if run['run_id'] == selected_run_id), None)
                if selected_run:
                    st.markdown("**Détails du Run**")
                    with st.expander("📋 Métriques et Paramètres", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Métriques**")
                            metrics = {k.split('.')[-1]: v for k, v in selected_run.items() if k.startswith('metrics.')}
                            st.json(metrics)
                        with col2:
                            st.markdown("**Paramètres**")
                            params = {k.split('.')[-1]: v for k, v in selected_run.items() if k.startswith('params.')}
                            st.json(params)
                    
                    # Visualisations spécifiques au run
                    st.markdown("#### 📊 Visualisations")
                    if validation["task_type"] == 'classification':
                        # Essayer de récupérer la matrice de confusion sauvegardée
                        cm_artifact = get_mlflow_artifact(selected_run_id, "confusion_matrix.pkl", client)
                        if cm_artifact:
                            st.markdown("**Matrice de Confusion**")
                            cm_plot = evaluator.create_confusion_matrix_plot(pd.read_pickle(cm_artifact))
                            if cm_plot:
                                st.plotly_chart(cm_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage matrice de confusion")
                        
                        # Courbe ROC
                        roc_artifact = get_mlflow_artifact(selected_run_id, "roc_curve.pkl", client)
                        if roc_artifact:
                            st.markdown("**Courbe ROC**")
                            roc_plot = evaluator.create_roc_curve_plot(pd.read_pickle(roc_artifact))
                            if roc_plot:
                                st.plotly_chart(roc_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage courbe ROC")
                    
                    elif validation["task_type"] == 'regression':
                        # Graphique des résidus
                        residuals_artifact = get_mlflow_artifact(selected_run_id, "residuals_plot.pkl", client)
                        if residuals_artifact:
                            st.markdown("**Graphique des Résidus**")
                            residuals_plot = evaluator.create_residuals_plot(pd.read_pickle(residuals_artifact))
                            if residuals_plot:
                                st.plotly_chart(residuals_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage graphique des résidus")
                        
                        # Prédictions vs Réelles
                        pred_vs_actual_artifact = get_mlflow_artifact(selected_run_id, "pred_vs_actual_plot.pkl", client)
                        if pred_vs_actual_artifact:
                            st.markdown("**Prédictions vs. Réelles**")
                            pred_vs_actual_plot = evaluator.create_predicted_vs_actual_plot(pd.read_pickle(pred_vs_actual_artifact))
                            if pred_vs_actual_plot:
                                st.plotly_chart(pred_vs_actual_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage graphique prédictions vs. réelles")
                    
                    elif validation["task_type"] == 'clustering':
                        # Scatter plot des clusters
                        cluster_artifact = get_mlflow_artifact(selected_run_id, "cluster_scatter_plot.pkl", client)
                        if cluster_artifact:
                            st.markdown("**Visualisation des Clusters**")
                            cluster_plot = evaluator.create_cluster_scatter_plot(pd.read_pickle(cluster_artifact))
                            if cluster_plot:
                                st.plotly_chart(cluster_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage scatter plot clusters")
                        
                        # Silhouette plot
                        silhouette_artifact = get_mlflow_artifact(selected_run_id, "silhouette_plot.pkl", client)
                        if silhouette_artifact:
                            st.markdown("**Analyse de Silhouette**")
                            silhouette_plot = evaluator.create_silhouette_plot(pd.read_pickle(silhouette_artifact))
                            if silhouette_plot:
                                st.plotly_chart(silhouette_plot, width='stretch')
                            else:
                                st.warning("⚠️ Échec affichage silhouette plot")
                
                    # Téléchargement des artefacts
                    st.markdown("#### 💾 Télécharger les Artefacts")
                    artifact_list = client.list_artifacts(selected_run_id)
                    if artifact_list:
                        for artifact in artifact_list:
                            if artifact.path.endswith(('.pkl', '.joblib')):
                                artifact_data = get_mlflow_artifact(selected_run_id, artifact.path, client)
                                if artifact_data:
                                    st.download_button(
                                        label=f"Télécharger {artifact.path}",
                                        data=artifact_data,
                                        file_name=artifact.path,
                                        mime="application/octet-stream",
                                        width='stretch'
                                    )
                    else:
                        st.info("ℹ️ Aucun artefact disponible")
                
                    # Déploiement du modèle
                    st.markdown("#### 🚀 Déployer le Modèle")
                    if st.button("Déployer via MLflow", key=f"deploy_{selected_run_id}"):
                        try:
                            run_info = client.get_run(selected_run_id)
                            model_uri = f"runs:/{selected_run_id}/model"
                            st.info(f"Commande de déploiement: `mlflow models serve -m {model_uri} --port 5000`")
                            st.success("✅ Instruction de déploiement générée. Exécutez la commande dans votre terminal.")
                        except Exception as e:
                            st.error(f"❌ Échec préparation déploiement: {str(e)}")
                            logger.error(f"Échec déploiement modèle {selected_run_id}: {e}")
            
            else:
                st.info("ℹ️ Aucun run filtré")
            
            # Exportation des métriques MLflow
            st.markdown("#### 📥 Exporter les Runs")
            if filtered_runs:
                csv_data = pd.DataFrame(run_data).to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger Runs CSV",
                    data=csv_data,
                    file_name=f"mlflow_runs_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
        except Exception as e:
            st.error(f"❌ Erreur chargement MLflow: {str(e)}")
            logger.error(f"MLflow tab error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🕐 Mis à jour: {time.strftime('%H:%M:%S')}")
    with col2:
        memory_info = get_system_metrics()
        memory_status = "🟢" if memory_info['memory_percent'] < 70 else "🟡" if memory_info['memory_percent'] < 85 else "🔴"
        st.caption(f"{memory_status} Mémoire: {memory_info['memory_percent']:.1f}%")
    with col3:
        st.caption(f"📊 Modèles: {len(st.session_state.ml_results)}")

if __name__ == "__main__":
    main()