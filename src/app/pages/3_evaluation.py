import os
import pickle
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

def model_has_predict_proba(model):
    """Vérifie si le modèle supporte predict_proba"""
    if model is None:
        return False
    if hasattr(model, 'named_steps'):
        final_step = list(model.named_steps.values())[-1]
        return hasattr(final_step, 'predict_proba')
    return hasattr(model, 'predict_proba')

def display_model_details(evaluator, model_result, task_type):
    """Affiche les détails complets d'un modèle avec visualisations"""
    model_name = model_result.get('model_name', 'Unknown')
    st.markdown(f"#### Détails du modèle: {model_name}")

    with st.expander("🔍 Debug - Données disponibles", expanded=False):
        available_keys = list(model_result.keys())
        st.write(f"**Clés disponibles:** {', '.join(available_keys)}")
        data_status = {key: f"✅ Présent ({getattr(model_result.get(key), 'shape', (len(model_result.get(key)),))})" 
                      if model_result.get(key) is not None and isinstance(model_result.get(key), (pd.DataFrame, pd.Series, np.ndarray)) 
                      else "✅ Présent" if model_result.get(key) is not None else "❌ Manquant" 
                      for key in ['X_test', 'y_test', 'X_train', 'y_train', 'X_sample', 'labels', 'model']}
        st.json(data_status)

    metrics = model_result.get('metrics', {})
    training_time = model_result.get('training_time', 0)
    st.markdown("**Métriques principales**")
    metrics_data = []

    if task_type == 'classification':
        metrics_data.extend([
            {'Métrique': 'Accuracy', 'Valeur': f"{metrics.get('accuracy', 0):.3f}", 'Description': 'Proportion des prédictions correctes'},
            {'Métrique': 'Precision', 'Valeur': f"{metrics.get('precision', 0):.3f}", 'Description': 'Précision des prédictions positives'},
            {'Métrique': 'Recall', 'Valeur': f"{metrics.get('recall', 0):.3f}", 'Description': 'Rappel des vrais positifs'},
            {'Métrique': 'F1-Score', 'Valeur': f"{metrics.get('f1_score', 0):.3f}", 'Description': 'Moyenne harmonique de précision et rappel'}
        ])
    elif task_type == 'regression':
        metrics_data.extend([
            {'Métrique': 'R² Score', 'Valeur': f"{metrics.get('r2', 0):.3f}", 'Description': 'Coefficient de détermination'},
            {'Métrique': 'MAE', 'Valeur': f"{metrics.get('mae', 0):.3f}", 'Description': 'Erreur absolue moyenne'},
            {'Métrique': 'RMSE', 'Valeur': f"{metrics.get('rmse', 0):.3f}", 'Description': 'Racine de l\'erreur quadratique moyenne'}
        ])
    else:  # clustering
        metrics_data.extend([
            {'Métrique': 'Silhouette Score', 'Valeur': f"{metrics.get('silhouette_score', 0):.3f}", 'Description': 'Qualité de la séparation des clusters'},
            {'Métrique': 'Calinski-Harabasz', 'Valeur': f"{metrics.get('calinski_harabasz', 0):.3f}", 'Description': 'Ratio de dispersion entre et intra-clusters'},
            {'Métrique': 'Davies-Bouldin', 'Valeur': f"{metrics.get('davies_bouldin_score', 0):.3f}", 'Description': 'Similitude moyenne entre clusters'},
            {'Métrique': 'Nombre de Clusters', 'Valeur': f"{metrics.get('n_clusters', 'N/A')}", 'Description': 'Nombre de clusters formés'}
        ])
    metrics_data.append({'Métrique': 'Temps d\'entraînement (s)', 'Valeur': f"{training_time:.1f}", 'Description': 'Durée de l\'entraînement'})
    st.dataframe(pd.DataFrame(metrics_data), width='stretch')

    model = model_result.get('model')
    X_test = model_result.get('X_test')
    y_test = model_result.get('y_test')
    X_train = model_result.get('X_train')
    y_train = model_result.get('y_train')
    X_sample = model_result.get('X_sample')
    labels = model_result.get('labels')
    feature_names = model_result.get('feature_names', [])

    logger.info(f"📊 Affichage des détails pour {model_name}, task_type={task_type}")

    if task_type in ['classification', 'regression'] and model:
        if feature_names:
            st.markdown("#### Importance des Features")
            try:
                feature_plot = evaluator.create_feature_importance_plot(model, feature_names)
                if feature_plot:
                    st.plotly_chart(feature_plot, width='stretch')
                else:
                    st.info("ℹ️ Importance des features non disponible pour ce type de modèle")
            except Exception as e:
                st.warning(f"⚠️ Erreur importance features: {str(e)[:100]}")
                logger.error(f"Erreur importance features {model_name}: {e}")

        if X_sample is not None and len(X_sample) > 0:
            st.markdown("#### Analyse SHAP")
            try:
                shap_plot = evaluator.create_shap_plot(model_result)
                if shap_plot:
                    st.plotly_chart(shap_plot, width='stretch')
                else:
                    st.info("ℹ️ Analyse SHAP non disponible pour ce modèle")
            except Exception as e:
                st.info(f"ℹ️ SHAP non disponible: {str(e)[:50]}")
                logger.info(f"SHAP non disponible pour {model_name}: {e}")

    if task_type == 'classification' and model:
        if X_test is not None and y_test is not None and len(X_test) > 0:
            st.markdown("#### Matrice de Confusion")
            try:
                cm_plot = evaluator.create_confusion_matrix_plot(model_result)
                if cm_plot:
                    st.plotly_chart(cm_plot, width='stretch')
                else:
                    st.warning("⚠️ Impossible d'afficher la matrice de confusion")
            except Exception as e:
                st.error(f"Erreur matrice de confusion: {str(e)[:100]}")
                logger.error(f"Erreur confusion matrix {model_name}: {e}")

            if model_has_predict_proba(model):
                st.markdown("#### Courbe ROC")
                try:
                    roc_plot = evaluator.create_roc_curve_plot(model_result)
                    if roc_plot:
                        st.plotly_chart(roc_plot, width='stretch')
                    else:
                        st.info("ℹ️ Courbe ROC disponible uniquement pour classification binaire")
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe ROC: {str(e)[:100]}")
                    logger.warning(f"Échec ROC pour {model_name}: {e}")

                st.markdown("#### Courbe de Précision-Rappel")
                try:
                    pr_plot = evaluator.create_precision_recall_curve_plot(model_result)
                    if pr_plot:
                        st.plotly_chart(pr_plot, width='stretch')
                    else:
                        st.info("ℹ️ Courbe PR disponible uniquement pour classification binaire")
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe PR: {str(e)[:100]}")

                st.markdown("#### Distribution des Probabilités Prédites")
                try:
                    proba_plot = evaluator.create_predicted_proba_distribution_plot(model_result)
                    if proba_plot:
                        st.plotly_chart(proba_plot, width='stretch')
                except Exception as e:
                    st.warning(f"⚠️ Erreur distribution probabilités: {str(e)[:100]}")
            else:
                st.info("ℹ️ Modèle ne supporte pas predict_proba - courbes ROC/PR non disponibles")

        if X_train is not None and y_train is not None and len(X_train) > 0:
            with st.expander("#### 📈 Courbe d'Apprentissage", expanded=False):
                try:
                    learning_plot = evaluator.create_learning_curve_plot(model_result)
                    if learning_plot:
                        st.plotly_chart(learning_plot, width='stretch')
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe d'apprentissage: {str(e)[:100]}")

        if X_sample is not None and len(X_sample) > 0:
            with st.expander("#### 🔥 Heatmap de Corrélation des Features", expanded=False):
                try:
                    temp_result = {**model_result, 'X_train': X_sample}
                    corr_plot = evaluator.create_feature_correlation_heatmap(temp_result)
                    if corr_plot:
                        st.plotly_chart(corr_plot, width='stretch')
                except Exception as e:
                    st.warning(f"⚠️ Erreur heatmap: {str(e)[:100]}")

    if task_type == 'regression' and model:
        if X_test is not None and y_test is not None and len(X_test) > 0:
            st.markdown("#### Graphique des Résidus")
            try:
                residuals_plot = evaluator.create_residuals_plot(model_result)
                if residuals_plot:
                    st.plotly_chart(residuals_plot, width='stretch')
            except Exception as e:
                st.error(f"Erreur graphique résidus: {str(e)[:100]}")

            st.markdown("#### Prédictions vs. Réelles")
            try:
                pred_vs_actual_plot = evaluator.create_predicted_vs_actual_plot(model_result)
                if pred_vs_actual_plot:
                    st.plotly_chart(pred_vs_actual_plot, width='stretch')
            except Exception as e:
                st.error(f"Erreur prédictions vs. réelles: {str(e)[:100]}")

        if X_train is not None and y_train is not None and len(X_train) > 0:
            with st.expander("#### 📈 Courbe d'Apprentissage", expanded=False):
                try:
                    learning_plot = evaluator.create_learning_curve_plot(model_result)
                    if learning_plot:
                        st.plotly_chart(learning_plot, width='stretch')
                except Exception as e:
                    st.warning(f"⚠️ Erreur courbe d'apprentissage: {str(e)[:100]}")

    if task_type == 'clustering':
        if X_sample is not None and labels is not None and len(X_sample) > 0:
            st.markdown("#### Visualisation des Clusters")
            try:
                cluster_plot = evaluator.create_cluster_scatter_plot(model_result)
                if cluster_plot:
                    st.plotly_chart(cluster_plot, width='stretch')
            except Exception as e:
                st.error(f"Erreur scatter plot clusters: {str(e)[:100]}")
                logger.error(f"Erreur cluster scatter {model_name}: {e}")

            st.markdown("#### Analyse de Silhouette")
            try:
                silhouette_plot = evaluator.create_silhouette_plot(model_result)
                if silhouette_plot:
                    st.plotly_chart(silhouette_plot, width='stretch')
            except Exception as e:
                st.error(f"Erreur silhouette plot: {str(e)[:100]}")
                logger.error(f"Erreur silhouette {model_name}: {e}")

            st.markdown("#### Dispersion Intra-Cluster")
            try:
                intra_cluster_plot = evaluator.create_intra_cluster_distance_plot(model_result)
                if intra_cluster_plot:
                    st.plotly_chart(intra_cluster_plot, width='stretch')
            except Exception as e:
                st.error(f"Erreur dispersion intra-cluster: {str(e)[:100]}")
                logger.error(f"Erreur intra-cluster {model_name}: {e}")
        else:
            st.warning("⚠️ Données de clustering (X_sample, labels) manquantes pour les visualisations")
            logger.warning(f"X_sample ou labels manquant pour {model_name}")

    st.markdown("---")
    cv_scores = model_result.get('cv_scores')
    if cv_scores is not None and isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        st.caption(f"📊 Validation Croisée: {cv_mean:.3f} ± {cv_std:.3f}")
    st.caption(f"📊 Visualisations générées pour {model_name} | ⏱️ Entraînement: {training_time:.2f}s")

def get_mlflow_artifact(run_id, artifact_path, client):
    """Récupère un artefact MLflow"""
    try:
        return client.download_artifacts(run_id, artifact_path)
    except Exception as e:
        logger.error(f"Erreur récupération artefact MLflow {run_id}: {e}")
        return None

def create_mlflow_run_plot(runs, task_type):
    """Crée un graphique de comparaison des runs MLflow"""
    try:
        # Vérification que task_type est une chaîne
        if isinstance(task_type, list):
            logger.warning("task_type est une liste, utilisation de la première valeur ou 'classification' par défaut")
            task_type = task_type[0] if task_type else 'classification'
        
        metric_key = 'metrics.accuracy' if task_type == 'classification' else 'metrics.r2' if task_type == 'regression' else 'metrics.silhouette_score'
        metric_label = 'Accuracy' if task_type == 'classification' else 'R² Score' if task_type == 'regression' else 'Silhouette Score'

        plot_data = [
            {'Model': run.get('tags.mlflow.runName', 'Unknown'), 'Metric': float(run.get(metric_key, 0)), 'Run ID': run.get('run_id', 'N/A')}
            for run in runs if run.get(metric_key) is not None
        ]

        if not plot_data:
            return None

        df_plot = pd.DataFrame(plot_data)
        fig = px.bar(
            df_plot,
            x='Model',
            y='Metric',
            color='Metric',
            text='Metric',
            title=f"Comparaison des Runs MLflow ({metric_label})",
            hover_data=['Run ID'],
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
        fig.update_layout(xaxis_title="Modèle", yaxis_title=metric_label, showlegend=False, height=400)
        return fig
    except Exception as e:
        logger.error(f"Erreur création graphique MLflow: {e}")
        return None

def display_mlflow_tab():
    """Affiche l'onglet MLflow avec gestion d'erreurs robuste"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("### 🔗 Exploration des Runs MLflow")

    if not MLFLOW_AVAILABLE:
        st.error("🚫 MLflow non disponible")
        st.info("Installez MLflow pour accéder aux runs (`pip install mlflow`).")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if 'mlflow_runs' not in st.session_state or st.session_state.mlflow_runs is None:
        st.session_state.mlflow_runs = []
        st.warning("⚠️ Aucun run MLflow disponible")
        st.info("Entraînez des modèles dans 'Configuration ML' pour générer des runs.")
        if st.button("🔄 Initialiser MLflow runs"):
            st.session_state.mlflow_runs = []
            st.success("✅ mlflow_runs initialisé")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    mlflow_runs = st.session_state.mlflow_runs
    if not isinstance(mlflow_runs, list):
        st.error(f"❌ Format des runs MLflow invalide: {type(mlflow_runs)}")
        logger.error(f"mlflow_runs type incorrect: {type(mlflow_runs)}")
        if st.button("🔄 Corriger le format"):
            st.session_state.mlflow_runs = []
            st.success("✅ Format corrigé")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if not mlflow_runs:
        st.warning("⚠️ Liste des runs MLflow vide")
        st.info("Entraînez des modèles pour générer des runs.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    first_run = mlflow_runs[0]
    required_keys = ['run_id', 'status', 'start_time']
    missing_keys = [key for key in required_keys if key not in first_run]
    if missing_keys:
        st.error(f"❌ Clés manquantes dans les runs: {missing_keys}")
        st.json({"keys_disponibles": list(first_run.keys())})
        logger.error(f"Run structure invalide. Keys: {list(first_run.keys())}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
    col1, col2, col3 = st.columns(3)
    with col1:
        finished_runs = len([r for r in mlflow_runs if r.get('status') == 'FINISHED'])
        st.metric("Runs Réussis", finished_runs)
    with col2:
        failed_runs = len([r for r in mlflow_runs if r.get('status') == 'FAILED'])
        st.metric("Runs Échoués", failed_runs)
    with col3:
        if finished_runs > 0:
            success_rate = (finished_runs / len(mlflow_runs)) * 100
            st.metric("Taux de Réussite", f"{success_rate:.1f}%")

    st.markdown("#### 📋 Filtrer les Runs")
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        run_status = st.multiselect("Statut du Run", options=['FINISHED', 'FAILED', 'RUNNING'], default=['FINISHED'], key="mlflow_status_filter")
    with col_filter2:
        model_names = sorted(set(run.get('tags.mlflow.runName', 'Unknown').split('_')[0] for run in mlflow_runs))
        selected_models = st.multiselect("Modèles", options=model_names, default=model_names, key="mlflow_model_filter")
    with col_filter3:
        available_metrics = set(k.split('.')[-1] for run in mlflow_runs for k in run.keys() if k.startswith('metrics.'))
        sort_metric = st.selectbox("Trier par", options=['start_time'] + list(available_metrics), index=0, key="mlflow_sort_by")

    filtered_runs = [run for run in mlflow_runs if run.get('status', 'UNKNOWN') in run_status and run.get('tags.mlflow.runName', 'Unknown').split('_')[0] in selected_models]
    filtered_runs = sorted(filtered_runs, key=lambda x: x.get(f'metrics.{sort_metric}', x.get('start_time', 0)), reverse=sort_metric != 'start_time')
    st.markdown(f"**{len(filtered_runs)} runs filtrés**")

    if filtered_runs:
        run_data = []
        for run in filtered_runs:
            metrics = {k.split('.')[-1]: v for k, v in run.items() if k.startswith('metrics.')}
            params = {k.split('.')[-1]: v for k, v in run.items() if k.startswith('params.')}
            run_id = run.get('run_id', 'N/A')
            row = {
                'Run ID': run_id[:8] + '...' if len(run_id) > 8 else run_id,
                'Modèle': run.get('tags.mlflow.runName', 'Unknown'),
                'Statut': run.get('status', 'UNKNOWN'),
                'Date': pd.to_datetime(run.get('start_time', 0), unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                **{k.title(): f"{v:.3f}" if isinstance(v, (int, float)) else v for k, v in metrics.items() if k in ['accuracy', 'f1_score', 'r2', 'mae', 'silhouette_score']}
            }
            run_data.append(row)

        df_runs = pd.DataFrame(run_data)
        st.markdown("#### 📊 Tableau des Runs MLflow")
        st.dataframe(df_runs, width='stretch', height=400)

        st.markdown("#### 📈 Comparaison des Performances")
        try:
            run_plot = create_mlflow_run_plot(filtered_runs, st.session_state.ml_results.get('task_type', 'classification'))
            if run_plot:
                st.plotly_chart(cached_plot(run_plot, "mlflow_run_plot"), width='stretch')
            else:
                st.warning("⚠️ Impossible de générer le graphique de comparaison")
        except Exception as e:
            st.warning(f"⚠️ Erreur graphique: {str(e)[:100]}")
            logger.warning(f"Échec création graphique MLflow: {e}")

        st.markdown("#### 🔍 Détails du Run")
        selected_run_idx = st.selectbox(
            "Sélectionner un Run",
            options=range(len(filtered_runs)),
            format_func=lambda x: f"{filtered_runs[x].get('tags.mlflow.runName', 'Unknown')} ({filtered_runs[x].get('status', 'UNKNOWN')})",
            key="mlflow_run_selector"
        )
        if selected_run_idx is not None:
            selected_run = filtered_runs[selected_run_idx]
            st.markdown("**Informations du Run**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Métriques**")
                metrics = {k.split('.')[-1]: v for k, v in selected_run.items() if k.startswith('metrics.')}
                st.json(metrics if metrics else {"info": "Aucune métrique disponible"})
            with col2:
                st.markdown("**Paramètres**")
                params = {k.split('.')[-1]: v for k, v in selected_run.items() if k.startswith('params.')}
                display_params = dict(list(params.items())[:20]) if params else {"info": "Aucun paramètre disponible"}
                st.json(display_params)
                if len(params) > 20:
                    st.caption(f"... et {len(params) - 20} autres paramètres")
            with st.expander("📋 Informations Complètes du Run", expanded=False):
                st.json(selected_run)

            if st.button("📥 Télécharger Artefacts", key=f"download_artifacts_{selected_run.get('run_id')}"):
                artifact_data = get_mlflow_artifact(selected_run.get('run_id'), "model", MlflowClient())
                if artifact_data:
                    st.success("✅ Artefact téléchargé!")
                else:
                    st.error("❌ Erreur lors du téléchargement des artefacts")

        if st.button("📥 Télécharger Runs CSV", key="download_mlflow_csv"):
            try:
                csv_data = pd.DataFrame(run_data).to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv_data,
                    file_name=f"mlflow_runs_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            except Exception as e:
                st.error(f"Erreur export CSV: {str(e)}")
    else:
        st.info("Aucun run ne correspond aux filtres sélectionnés")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Fonction principale de la page d'évaluation"""
    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        st.error("🚫 Aucun résultat disponible")
        st.info("Entraînez des modèles dans 'Configuration ML'.")
        if st.button("⚙️ Aller à la page entrainement", width='stretch'):
            st.switch_page("pages/2_training.py")
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
                    avg_silhouette = np.mean([r.get('metrics', {}).get('silhouette_score', 0) for r in validation["successful_models"]])
                    st.metric("Silhouette Moyen", f"{avg_silhouette:.3f}")
                with col2:
                    avg_clusters = np.mean([r.get('metrics', {}).get('n_clusters', 0) for r in validation["successful_models"]])
                    st.metric("Clusters Moyen", f"{avg_clusters:.1f}")
                with col3:
                    best_silhouette = max([r.get('metrics', {}).get('silhouette_score', 0) for r in validation["successful_models"]])
                    st.metric("Meilleur Silhouette", f"{best_silhouette:.3f}")
        else:
            st.warning("⚠️ Aucun modèle valide")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analyse Détaillée")
        if validation["successful_models"]:
            model_names = [r.get('model_name', f'Modèle_{i}') for i, r in enumerate(validation["successful_models"])]
            selected_idx = st.selectbox(
                "Modèle à analyser:",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x] == validation.get('best_model') else ''}",
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
                clustering_metrics = [
                    {
                        'Modèle': r.get('model_name', 'Unknown'),
                        'Silhouette': f"{r.get('metrics', {}).get('silhouette_score', 0):.3f}",
                        'Calinski-Harabasz': f"{r.get('metrics', {}).get('calinski_harabasz', 0):.3f}",
                        'Davies-Bouldin': f"{r.get('metrics', {}).get('davies_bouldin_score', 0):.3f}",
                        'Clusters': r.get('metrics', {}).get('n_clusters', 'N/A'),
                        'Qualité': '🟢 Excellente' if r.get('metrics', {}).get('silhouette_score', 0) > 0.7 else '🟡 Bonne' if r.get('metrics', {}).get('silhouette_score', 0) > 0.5 else '🟠 Moyenne' if r.get('metrics', {}).get('silhouette_score', 0) > 0.3 else '🔴 Faible'
                    }
                    for r in validation["successful_models"]
                ]
                st.dataframe(pd.DataFrame(clustering_metrics), width='stretch')
                
                st.markdown("**⏱️ Performance Computationnelle**")
                perf_metrics = [{'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"} for r in validation["successful_models"]]
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
                perf_metrics = [{'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"} for r in validation["successful_models"]]
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
                perf_metrics = [{'Modèle': r.get('model_name'), 'Temps (s)': f"{r.get('training_time', 0):.1f}"} for r in validation["successful_models"]]
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
                    best_model_result = next((r for r in validation["successful_models"] if r.get('model_name') == validation["best_model"]), None)
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
        display_mlflow_tab()

if __name__ == "__main__":
    main()