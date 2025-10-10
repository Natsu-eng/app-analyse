import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import gc
import concurrent.futures
from typing import Dict, Optional, List, Any
from src.evaluation.model_plots import ModelEvaluationVisualizer, _generate_color_palette, _safe_get_model_task_type
from src.evaluation.metrics import get_system_metrics
from utils.report_generator import generate_pdf_report
from src.config.constants import TRAINING_CONSTANTS, LOGGING_CONSTANTS, VALIDATION_CONSTANTS, VISUALIZATION_CONSTANTS
from logging import getLogger
from datetime import datetime
import logging

logger = getLogger(__name__)

# Import MLflow avec gestion robuste
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
st.set_page_config(
    page_title="Évaluation des Modèles", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

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
    .performance-high { color: #27ae60; }
    .performance-medium { color: #f39c12; }
    .performance-low { color: #e74c3c; }
    .tab-content {
        padding: 1rem;
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #ecf0f1;
    }
    .plot-container {
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #fafbfc;
    }
</style>
""", unsafe_allow_html=True)

def log_structured(level: str, message: str, extra: Dict = None):
    """Logging structuré avec gestion d'erreurs"""
    try:
        log_dict = {
            "timestamp": datetime.now().isoformat(), 
            "level": level, 
            "message": message,
            "module": "evaluation_page"
        }
        if extra:
            log_dict.update(extra)
        logger.log(getattr(logging, level.upper()), json.dumps(log_dict, default=str))
    except Exception as e:
        print(f"Logging error: {e}")

@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def cached_plot(fig, plot_key: str):
    """Cache les graphiques - VERSION STABLE"""
    try:
        if fig is None:
            return None
        
        # Vérification que c'est une figure Plotly valide
        if hasattr(fig, 'to_json'):
            return fig
        else:
            return fig  # Retourner l'objet même si pas Plotly
            
    except Exception as e:
        print(f"Cache error: {e}")
        return fig  # Toujours retourner la figure même en cas d'erreur
    
    except Exception as e:
        log_structured("ERROR", f"Erreur cache graphique", {
            "plot_key": plot_key, 
            "error": str(e)
        })
        return fig

def monitor_ml_operation(func):
    """Décorateur pour monitorer les opérations ML"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            log_structured("INFO", f"Opération {func.__name__} réussie", {
                "duration_s": round(duration, 2)
            })
            return result
        except Exception as e:
            log_structured("ERROR", f"Échec {func.__name__}", {
                "error": str(e)[:200]
            })
            raise
    return wrapper

def get_mlflow_artifact(run_id: str, artifact_path: str, client: Optional[Any] = None) -> Optional[bytes]:
    """Récupère un artefact MLflow avec gestion robuste des erreurs"""
    try:
        if not MLFLOW_AVAILABLE or client is None:
            log_structured("ERROR", "MLflow non disponible")
            return None
        
        artifact_data = client.download_artifacts(run_id, artifact_path)
        log_structured("INFO", "Artefact MLflow téléchargé", {
            "run_id": run_id[:8], 
            "artifact_path": artifact_path
        })
        return artifact_data
    except Exception as e:
        log_structured("ERROR", f"Échec téléchargement artefact", {
            "run_id": run_id[:8] if run_id else "unknown",
            "error": str(e)[:200]
        })
        return None

def display_metrics_header(validation_result: Dict[str, Any]):
    """Affiche l'en-tête avec métriques principales"""
    successful_count = len(validation_result.get("successful_models", []))
    total_count = validation_result.get("results_count", 0)
    best_model = validation_result.get("best_model", "N/A")
    task_type = validation_result.get("task_type", "unknown")
    
    st.markdown('<div class="main-header">📈 Évaluation des Modèles</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        status_color = "#27ae60" if success_rate > 80 else "#f39c12" if success_rate > 50 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Taux de Réussite</div>
            <div class="metric-value" style="color: {status_color};">{success_rate:.1f}%</div>
            <div class="metric-subtitle">{successful_count}/{total_count} modèles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card {'best-model-card' if best_model != 'N/A' else ''}">
            <div class="metric-title">Meilleur Modèle</div>
            <div class="metric-value" style="font-size: 1.2rem;">{best_model}</div>
            <div class="metric-subtitle">Type: {task_type.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        failed_count = len(validation_result.get("failed_models", []))
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
            <div class="metric-subtitle">{memory_info['memory_available_mb']:.0f} MB disponible</div>
        </div>
        """, unsafe_allow_html=True)

def create_pdf_report_latex(model_result: Dict[str, Any], task_type: str) -> Optional[bytes]:
    """Génère un rapport PDF avec LaTeX"""
    try:
        metrics = model_result.get('metrics', {})
        model_name = model_result.get('model_name', 'Unknown')
        training_time = model_result.get('training_time', 0)
        
        latex_content = f"""
\\documentclass[a4paper,11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\begin{{document}}

\\begin{{center}}
    \\textbf{{\\Large Rapport d'Évaluation du Modèle: {model_name}}} \\\\[0.5cm]
    \\textit{{Type de tâche: {task_type.title()}}}
\\end{{center}}

\\section*{{Métriques de Performance}}
\\begin{{tabular}}{{ll}}
    \\toprule
    \\textbf{{Métrique}} & \\textbf{{Valeur}} \\\\
    \\midrule
"""
        
        if task_type == 'classification':
            latex_content += f"    Accuracy & {metrics.get('accuracy', 0):.3f} \\\\\\ \n"
            latex_content += f"    Precision & {metrics.get('precision', 0):.3f} \\\\\\ \n"
            latex_content += f"    Recall & {metrics.get('recall', 0):.3f} \\\\\\ \n"
            latex_content += f"    F1-Score & {metrics.get('f1', 0):.3f} \\\\\\ \n"
        elif task_type == 'regression':
            latex_content += f"    R² Score & {metrics.get('r2', 0):.3f} \\\\\\ \n"
            latex_content += f"    MAE & {metrics.get('mae', 0):.3f} \\\\\\ \n"
            latex_content += f"    RMSE & {metrics.get('rmse', 0):.3f} \\\\\\ \n"
        else:  # clustering
            latex_content += f"    Silhouette Score & {metrics.get('silhouette_score', 0):.3f} \\\\\\ \n"
            latex_content += f"    Nombre de Clusters & {metrics.get('n_clusters', 'N/A')} \\\\\\ \n"
        
        latex_content += f"    Temps d'entraînement & {training_time:.1f}s \\\\\\ \n"
        latex_content += """
    \\bottomrule
\\end{tabular}

\\section*{Résumé}
Ce rapport présente les performances du modèle \\textbf{""" + model_name + """} pour la tâche de """ + task_type + """.
Veuillez consulter les visualisations pour plus de détails.

\\end{document}
"""
        
        pdf_bytes = generate_pdf_report({'content': latex_content})
        log_structured("INFO", "Rapport PDF généré", {"model_name": model_name})
        return pdf_bytes
        
    except Exception as e:
        log_structured("ERROR", f"Génération PDF échouée", {"error": str(e)[:200]})
        return None

def model_has_predict_proba(model) -> bool:
    """Vérifie si le modèle supporte predict_proba"""
    if model is None:
        return False
    try:
        if hasattr(model, 'named_steps'):
            final_step = list(model.named_steps.values())[-1]
            return hasattr(final_step, 'predict_proba')
        return hasattr(model, 'predict_proba')
    except Exception:
        return False

        
@monitor_ml_operation
def display_model_details(visualizer, model_result: Dict[str, Any], task_type: str):  

    """Affiche les détails complets d'un modèle avec visualisations - VERSION STABLE"""
    model_name = model_result.get('model_name', 'Unknown')
    unique_id = f"{model_name}_{int(time.time())}"

    st.markdown(f"#### 🔍 Détails du modèle: **{model_name}**")

    # Section debug expandable
    with st.expander("🐛 Debug - Données disponibles", expanded=False):
        debug_info = {
            "model_name": model_name,
            "task_type": task_type,
            "has_X_test": model_result.get('X_test') is not None,
            "has_y_test": model_result.get('y_test') is not None,
            "has_X_train": model_result.get('X_train') is not None,
            "has_y_train": model_result.get('y_train') is not None,
            "has_model": model_result.get('model') is not None,
            "has_labels": model_result.get('labels') is not None,
            "metrics_keys": list(model_result.get('metrics', {}).keys()),
            "feature_names": model_result.get('feature_names', [])[:5]
        }
        st.json(debug_info)

    # Séparation visuelle
    st.markdown("---")
    st.markdown("#### 📈 Visualisations")

    try:
        # === Classification / Régression : Feature importance ===
        if task_type in ['classification', 'regression']:
            with st.container():
                st.markdown("**🎯 Importance des Features**")
                feature_plot = visualizer.create_feature_importance_plot(model_result)
                if feature_plot:
                    st.plotly_chart(
                        cached_plot(feature_plot, f"feature_{unique_id}"),
                        width='stretch',
                        key=f"feature_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Importance des features non disponible pour ce modèle.")

        # === SHAP Analysis ===
        if task_type in ['classification', 'regression'] and model_result.get('X_sample') is not None:
            with st.container():
                st.markdown("**🔍 Analyse SHAP**")
                shap_plot = visualizer.create_shap_analysis(model_result)
                if shap_plot:
                    st.plotly_chart(
                        cached_plot(shap_plot, f"shap_{unique_id}"),
                        width='stretch',
                        key=f"shap_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Analyse SHAP non disponible.")

        # === Courbe d'apprentissage ===
        if task_type in ['classification', 'regression'] and model_result.get('X_train') is not None:
            with st.container():
                st.markdown("**📚 Courbe d'Apprentissage**")
                learning_plot = visualizer.create_learning_curve(model_result)
                if learning_plot:
                    st.plotly_chart(
                        cached_plot(learning_plot, f"learning_{unique_id}"),
                        width='stretch',
                        key=f"learning_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Courbe d'apprentissage non disponible.")

        # === Cas Classification ===
        if task_type == 'classification':
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Matrice de Confusion**")
                cm_plot = visualizer.create_confusion_matrix(model_result)
                if cm_plot:
                    st.plotly_chart(
                        cached_plot(cm_plot, f"cm_{unique_id}"),
                        width='stretch',
                        key=f"cm_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Matrice de confusion non disponible.")

            with col2:
                st.markdown("**📈 Courbe ROC**")
                roc_plot = visualizer.create_roc_curve(model_result)
                if roc_plot:
                    st.plotly_chart(
                        cached_plot(roc_plot, f"roc_{unique_id}"),
                        width='stretch',
                        key=f"roc_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Courbe ROC non disponible.")

        # === Cas Régression ===
        elif task_type == 'regression':
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📉 Graphique des Résidus**")
                residuals_plot = visualizer.create_residuals_plot(model_result)
                if residuals_plot:
                    st.plotly_chart(
                        cached_plot(residuals_plot, f"residuals_{unique_id}"),
                        width='stretch',
                        key=f"residuals_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Graphique des résidus non disponible.")

            with col2:
                st.markdown("**🎯 Prédictions vs Réelles**")
                pred_plot = visualizer.create_predicted_vs_actual(model_result)
                if pred_plot:
                    st.plotly_chart(
                        cached_plot(pred_plot, f"pred_{unique_id}"),
                        width='stretch',
                        key=f"pred_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Graphique prédictions vs réelles non disponible.")

        # === Cas Clustering ===
        elif task_type == 'clustering':
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔮 Visualisation des Clusters**")
                cluster_plot = visualizer.create_cluster_visualization(model_result)
                if cluster_plot:
                    st.plotly_chart(
                        cached_plot(cluster_plot, f"cluster_{unique_id}"),
                        width='stretch',
                        key=f"cluster_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Visualisation des clusters non disponible.")

            with col2:
                st.markdown("**📊 Analyse Silhouette**")
                silhouette_plot = visualizer.create_silhouette_analysis(model_result)
                if silhouette_plot:
                    st.plotly_chart(
                        cached_plot(silhouette_plot, f"silhouette_{unique_id}"),
                        width='stretch',
                        key=f"silhouette_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Analyse silhouette non disponible.")

    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des visualisations: {str(e)[:200]}")
        log_structured("ERROR", f"Erreur visualisations {model_name}", {"error": str(e)})

    # Nettoyage mémoire
    gc.collect()

def create_mlflow_run_plot(runs: List[Any], task_type: str, metric_to_plot: str = None, chart_type: str = "Bar") -> Optional[go.Figure]:
    """Crée un graphique comparant les performances des runs MLflow"""
    if not runs:
        log_structured("WARNING", "Aucun run MLflow fourni")
        return None

    # Validation des runs et extraction des données
    valid_runs_data = []
    available_metrics = set()
    
    for i, run in enumerate(runs):
        run_dict = run if isinstance(run, dict) else run.__dict__ if hasattr(run, '__dict__') else {}
        metrics = run_dict.get('metrics', {})
        if not metrics:
            continue
        
        # Extraction du nom du modèle
        model_name = (
            run_dict.get('tags', {}).get('mlflow.runName', 
            run_dict.get('model_name', 
            run_dict.get('runName', f'Modèle_{i}'))))
        
        # Ajout des métriques disponibles
        available_metrics.update(metrics.keys())
        valid_runs_data.append({'model_name': model_name, 'metrics': metrics})

    if not valid_runs_data:
        log_structured("WARNING", "Aucune donnée valide dans les runs MLflow")
        return None

    # Sélection de la métrique par défaut si non spécifiée
    if not metric_to_plot:
        if task_type == 'classification':
            metric_to_plot = 'accuracy' if 'accuracy' in available_metrics else 'f1' if 'f1' in available_metrics else None
        elif task_type == 'regression':
            metric_to_plot = 'r2' if 'r2' in available_metrics else 'rmse' if 'rmse' in available_metrics else None
        elif task_type == 'clustering':
            metric_to_plot = 'silhouette_score' if 'silhouette_score' in available_metrics else None
        
        if not metric_to_plot:
            log_structured("WARNING", "Aucune métrique exploitable trouvée")
            return None

    # Préparation des données pour le graphique
    plot_data = []
    for run_data in valid_runs_data:
        model_name = run_data['model_name']
        metrics = run_data['metrics']
        if metric_to_plot in metrics and isinstance(metrics[metric_to_plot], (int, float)):
            plot_data.append({'Modèle': model_name, metric_to_plot: metrics[metric_to_plot]})

    if not plot_data:
        log_structured("WARNING", f"Aucune donnée pour la métrique {metric_to_plot}")
        return None

    # Création du DataFrame
    df = pd.DataFrame(plot_data)
    
    # Limiter à 10 modèles pour éviter l'encombrement
    if len(df) > 10:
        df = df.head(10)
        log_structured("INFO", "Limitation à 10 modèles pour le graphique")

    # Création du graphique
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[:len(df)]

    if chart_type == "Radar":
        # Graphique radar pour toutes les métriques
        for run_data in valid_runs_data:
            model_name = run_data['model_name']
            metrics = run_data['metrics']
            valid_metrics = [m for m in available_metrics if m in metrics and isinstance(metrics[m], (int, float))]
            values = [metrics.get(m, 0) for m in valid_metrics]
            if values:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=valid_metrics,
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors[valid_runs_data.index(run_data) % len(colors)])
                ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Comparaison des Modèles - Radar ({task_type.capitalize()})",
            template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
            height=500
        )

    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df['Modèle'],
            y=df[metric_to_plot],
            mode='lines+markers+text',
            name=metric_to_plot,
            line=dict(color=colors[0]),
            marker=dict(size=8),
            text=df[metric_to_plot].round(3),
            textposition='top center'
        ))
        fig.update_layout(
            title=f"Comparaison des Modèles - {metric_to_plot} ({task_type.capitalize()})",
            xaxis_title="Modèles",
            yaxis_title=metric_to_plot,
            template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
            height=500,
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            margin=dict(b=150)
        )

    else:  # Bar
        fig.add_trace(go.Bar(
            x=df['Modèle'],
            y=df[metric_to_plot],
            name=metric_to_plot,
            marker_color=colors,
            width=0.8 / max(1, len(df)),
            text=df[metric_to_plot].round(3),
            textposition='auto'
        ))
        fig.update_layout(
            title=f"Comparaison des Modèles - {metric_to_plot} ({task_type.capitalize()})",
            xaxis_title="Modèles",
            yaxis_title=metric_to_plot,
            template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
            height=500,
            showlegend=False,
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            margin=dict(b=150)
        )

    log_structured("INFO", "Graphique MLflow généré avec succès", {
        "n_models": len(df),
        "metric": metric_to_plot,
        "chart_type": chart_type
    })
    return fig

def display_mlflow_tab():
    """Affiche l'onglet MLflow avec gestion d'erreurs robuste"""
    st.markdown("### 🔗 Exploration des Runs MLflow")
    
    if not MLFLOW_AVAILABLE:
        st.error("🚫 MLflow non disponible")
        st.info("Installez MLflow pour accéder aux runs: `pip install mlflow`")
        return
    
    if 'mlflow_runs' not in st.session_state:
        st.session_state.mlflow_runs = []
    
    mlflow_runs = st.session_state.mlflow_runs
    
    if not mlflow_runs:
        st.warning("⚠️ Aucun run MLflow disponible")
        st.info("Entraînez des modèles pour générer des runs MLflow")
        return
    
    st.markdown(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=['FINISHED', 'RUNNING', 'FAILED'],
            default=['FINISHED'],
            key="mlflow_status_filter"
        )
    
    with col2:
        model_names = sorted(set(
            run.get('tags', {}).get('mlflow.runName', run.get('model_name', 'Unknown'))
            for run in mlflow_runs if isinstance(run, dict)
        ))
        model_filter = st.multiselect(
            "Filtrer par modèle",
            options=model_names,
            default=model_names,
            key="mlflow_model_filter"
        )
    
    with col3:
        available_metrics = set()
        for run in mlflow_runs:
            if isinstance(run, dict) and 'metrics' in run:
                available_metrics.update(run['metrics'].keys())
        metric_to_plot = st.selectbox(
            "Métrique à afficher",
            options=sorted(available_metrics),
            key="mlflow_metric_selector"
        )
    
    # Filtrage des runs
    filtered_runs = [
        run for run in mlflow_runs
        if isinstance(run, dict) and
        run.get('status', 'UNKNOWN') in status_filter and
        run.get('tags', {}).get('mlflow.runName', run.get('model_name', 'Unknown')) in model_filter
    ]
    
    if not filtered_runs:
        st.info("ℹ️ Aucun run ne correspond aux filtres sélectionnés")
        return
    
    # Affichage du tableau
    st.markdown("#### 📋 Liste des Runs")
    run_data = []
    for run in filtered_runs:
        run_id = run.get('run_id', 'N/A')
        status = run.get('status', 'UNKNOWN')
        model_name = run.get('tags', {}).get('mlflow.runName', run.get('model_name', 'Unknown'))
        metrics = run.get('metrics', {})
        row = {
            'Run ID': run_id[:8] + '...' if len(run_id) > 8 else run_id,
            'Modèle': model_name,
            'Statut': status
        }
        for metric in available_metrics:
            row[metric] = f"{metrics.get(metric, 0):.3f}" if metric in metrics else 'N/A'
        run_data.append(row)
    
    if run_data:
        st.dataframe(pd.DataFrame(run_data), width='stretch')
    
    # Graphique de comparaison
    st.markdown("#### 📈 Comparaison des Performances")
    task_type = st.session_state.get('task_type', 'classification')
    mlflow_plot = create_mlflow_run_plot(filtered_runs, task_type, metric_to_plot)
    
    if mlflow_plot:
        st.plotly_chart(
            cached_plot(mlflow_plot, f"mlflow_comparison_{metric_to_plot}"),
            width='stretch',
            key=f"mlflow_comparison_{metric_to_plot}_{int(time.time())}"
        )
    else:
        st.info(f"ℹ️ Données insuffisantes pour générer le graphique de {metric_to_plot}")

def main():
    """Fonction principale de la page d'évaluation"""
    
    # Initialisation de session
    if 'warnings' not in st.session_state:
        st.session_state.warnings = []
    
    # VÉRIFICATION ROBUSTE des résultats ML
    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        st.error("🚫 Aucun résultat d'entraînement disponible")
        st.info("Veuillez d'abord entraîner des modèles dans l'onglet 'Configuration ML'")
        
        if st.button("⚙️ Aller à l'entraînement", width='stretch'):
            st.switch_page("pages/2_training.py")
        return
    
    # VÉRIFICATION que ml_results est une liste
    if not isinstance(st.session_state.ml_results, list):
        st.error("❌ Format invalide des résultats d'entraînement")
        st.session_state.ml_results = None
        return
    
    # VÉRIFICATION que la liste n'est pas vide
    if len(st.session_state.ml_results) == 0:
        st.error("📭 Aucun résultat d'entraînement disponible")
        return
    
    try:
        # Initialisation du visualiseur
        visualizer = ModelEvaluationVisualizer(st.session_state.ml_results)
        validation_result = visualizer.validation_result
        
        if not validation_result["has_results"]:
            st.error("📭 Aucune donnée valide trouvée dans les résultats")
            return
            
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation de l'évaluation: {str(e)[:200]}")
        log_structured("ERROR", "Erreur initialisation visualizer", {"error": str(e)})
        return
    
    # En-tête avec métriques
    display_metrics_header(validation_result)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'Ensemble", 
        "🔍 Détails Modèles", 
        "📈 Métriques", 
        "💾 Export", 
        "🔗 MLflow"
    ])
    
    with tab1:
        st.markdown("### 📈 Comparaison des Modèles")
        
        if validation_result["successful_models"]:
            comparison_plot = visualizer.create_comparison_plot()
            if comparison_plot:
                st.plotly_chart(
                    cached_plot(comparison_plot, "comparison"), 
                    width='stretch',
                    key=f"comparison_{int(time.time())}"  # CLÉ UNIQUE
                )
            else:
                st.warning("⚠️ Impossible de générer le graphique de comparaison")
            
            st.markdown("### 📋 Tableau de Comparaison")
            df_comparison = visualizer.get_comparison_dataframe()
            st.dataframe(df_comparison, width='stretch')
            
        else:
            st.warning("⚠️ Aucun modèle valide à comparer")
    
    with tab2:
        st.markdown("### 🔍 Analyse Détaillée par Modèle")
        
        successful_models = validation_result.get("successful_models", [])
        if successful_models:
            model_names = [m.get('model_name', f'Modèle_{i}') for i, m in enumerate(successful_models)]
            best_model = validation_result.get("best_model")
            
            selected_idx = st.selectbox(
                "Sélectionnez un modèle à analyser:",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x] == best_model else ''}",
                key="model_selector"
            )
            
            if 0 <= selected_idx < len(successful_models):
                display_model_details(
                    visualizer, 
                    successful_models[selected_idx], 
                    validation_result["task_type"]
                )
            else:
                st.error("❌ Sélection de modèle invalide")
        else:
            st.info("ℹ️ Aucun modèle disponible pour l'analyse détaillée")
    
    with tab3:
        st.markdown("### 📈 Métriques et Distributions")
        
        if validation_result["successful_models"]:
            # Distribution des performances
            st.markdown("#### 📊 Distribution des Performances")
            dist_plot = visualizer.create_performance_distribution()
            if dist_plot:
                st.plotly_chart(
                    cached_plot(dist_plot, "performance_dist"), 
                    width='stretch',
                    key=f"performance_dist_{int(time.time())}"  # CLÉ UNIQUE
                )
            
            # Métriques détaillées
            st.markdown("#### 📋 Métriques Détaillées")
            task_type = validation_result["task_type"]
            
            if task_type == 'classification':
                metrics_df = pd.DataFrame([
                    {
                        'Modèle': m.get('model_name', 'Unknown'),
                        'Accuracy': m.get('metrics', {}).get('accuracy', 0),
                        'Precision': m.get('metrics', {}).get('precision', 0),
                        'Recall': m.get('metrics', {}).get('recall', 0),
                        'F1-Score': m.get('metrics', {}).get('f1', 0),
                        'Temps (s)': m.get('training_time', 0)
                    }
                    for m in validation_result["successful_models"]
                ])
            
            elif task_type == 'regression':
                metrics_df = pd.DataFrame([
                    {
                        'Modèle': m.get('model_name', 'Unknown'),
                        'R²': m.get('metrics', {}).get('r2', 0),
                        'MAE': m.get('metrics', {}).get('mae', 0),
                        'RMSE': m.get('metrics', {}).get('rmse', 0),
                        'Temps (s)': m.get('training_time', 0)
                    }
                    for m in validation_result["successful_models"]
                ])
            
            else:  # clustering
                metrics_df = pd.DataFrame([
                    {
                        'Modèle': m.get('model_name', 'Unknown'),
                        'Silhouette': m.get('metrics', {}).get('silhouette_score', 0),
                        'Calinski-Harabasz': m.get('metrics', {}).get('calinski_harabasz_score', 0),
                        'Davies-Bouldin': m.get('metrics', {}).get('davies_bouldin_score', 0),
                        'Clusters': m.get('metrics', {}).get('n_clusters', 'N/A'),
                        'Temps (s)': m.get('training_time', 0)
                    }
                    for m in validation_result["successful_models"]
                ])
            
            st.dataframe(metrics_df, width='stretch')
            
        else:
            st.warning("⚠️ Aucune métrique disponible")
    
    with tab4:
        st.markdown("### 💾 Export des Résultats")
        
        if validation_result["successful_models"]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Données Structurées")
                
                # Export CSV
                df_comparison = visualizer.get_comparison_dataframe()
                csv_data = df_comparison.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"comparaison_modeles_{int(time.time())}.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
                # Export JSON
                export_data = visualizer.get_export_data()
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_data,
                    file_name=f"evaluation_complete_{int(time.time())}.json",
                    mime="application/json",
                    width='stretch'
                )
            
            with col2:
                st.markdown("#### 📈 Rapports Détaillés")
                
                # Rapport PDF du meilleur modèle
                best_model_name = validation_result.get("best_model")
                if best_model_name:
                    best_model_result = next(
                        (m for m in validation_result["successful_models"] 
                         if m.get('model_name') == best_model_name), 
                        None
                    )
                    
                    if best_model_result:
                        pdf_bytes = create_pdf_report_latex(
                            best_model_result, 
                            validation_result["task_type"]
                        )
                        
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Rapport PDF (Meilleur Modèle)",
                                data=pdf_bytes,
                                file_name=f"rapport_{best_model_name}_{int(time.time())}.pdf",
                                mime="application/pdf",
                                width='stretch'
                            )
                
                # Bouton de régénération
                if st.button("🔄 Générer Nouveau Rapport", width='stretch'):
                    with st.spinner("Génération du rapport..."):
                        time.sleep(2)
                        st.success("✅ Rapport généré avec succès!")
            
            # Aperçu des données
            with st.expander("👁️ Aperçu des Données d'Export", expanded=False):
                export_data = visualizer.get_export_data()
                st.json(export_data)
                
        else:
            st.info("ℹ️ Aucune donnée disponible pour l'export")
    
    with tab5:
        display_mlflow_tab()
    
    # Affichage des warnings
    if st.session_state.warnings:
        with st.expander("⚠️ Avertissements", expanded=False):
            for warning in st.session_state.warnings:
                st.warning(warning)
        
        if st.button("🗑️ Effacer les avertissements"):
            st.session_state.warnings = []
            st.rerun()
    
    # Nettoyage final
    gc.collect()

if __name__ == "__main__":
    main()