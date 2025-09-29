import os
import numpy as np
import streamlit as st
import pandas as pd
import time
import json
from ml.evaluation.visualization import ModelEvaluationVisualizer
from ml.evaluation.metrics_calculation import get_system_metrics, EvaluationMetrics
from utils.report_generator import generate_pdf_report

from logging import getLogger
logger = getLogger(__name__)

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
    """Affiche les détails d'un modèle sélectionné"""
    st.markdown(f"#### Détails du modèle: {model_result.get('model_name', 'Unknown')}")
    
    # Afficher les métriques
    metrics = model_result.get('metrics', {})
    training_time = model_result.get('training_time', 0)
    
    st.markdown("**Métriques principales**")
    metrics_data = []
    if task_type == 'classification':
        metrics_data.append({
            'Métrique': 'Accuracy', 'Valeur': f"{metrics.get('accuracy', 0):.3f}",
            'Description': 'Proportion des prédictions correctes'
        })
        metrics_data.append({
            'Métrique': 'Precision', 'Valeur': f"{metrics.get('precision', 0):.3f}",
            'Description': 'Précision des prédictions positives'
        })
        metrics_data.append({
            'Métrique': 'Recall', 'Valeur': f"{metrics.get('recall', 0):.3f}",
            'Description': 'Rappel des vrais positifs'
        })
        metrics_data.append({
            'Métrique': 'F1-Score', 'Valeur': f"{metrics.get('f1_score', 0):.3f}",
            'Description': 'Moyenne harmonique de précision et rappel'
        })
    elif task_type == 'regression':
        metrics_data.append({
            'Métrique': 'R² Score', 'Valeur': f"{metrics.get('r2', 0):.3f}",
            'Description': 'Coefficient de détermination'
        })
        metrics_data.append({
            'Métrique': 'MAE', 'Valeur': f"{metrics.get('mae', 0):.3f}",
            'Description': 'Erreur absolue moyenne'
        })
        metrics_data.append({
            'Métrique': 'RMSE', 'Valeur': f"{metrics.get('rmse', 0):.3f}",
            'Description': 'Racine de l’erreur quadratique moyenne'
        })
    else:  # clustering
        metrics_data.append({
            'Métrique': 'Silhouette Score', 'Valeur': f"{metrics.get('silhouette_score', 0):.3f}",
            'Description': 'Qualité de la séparation des clusters'
        })
        metrics_data.append({
            'Métrique': 'Calinski-Harabasz', 'Valeur': f"{metrics.get('calinski_harabasz', 0):.3f}",
            'Description': 'Ratio de dispersion entre et intra-clusters'
        })
        metrics_data.append({
            'Métrique': 'Davies-Bouldin', 'Valeur': f"{metrics.get('davies_bouldin_score', 0):.3f}",
            'Description': 'Similitude moyenne entre clusters'
        })
        metrics_data.append({
            'Métrique': 'Nombre de Clusters', 'Valeur': f"{metrics.get('n_clusters', 'N/A')}",
            'Description': 'Nombre de clusters formés'
        })
    
    metrics_data.append({
        'Métrique': 'Temps d’entraînement (s)', 'Valeur': f"{training_time:.1f}",
        'Description': 'Durée de l’entraînement'
    })
    
    st.dataframe(pd.DataFrame(metrics_data), width='stretch')

    # Visualisations spécifiques
    model_name = model_result.get('model_name', 'Unknown')
    model = model_result.get('model')
    logger.info(f"📊 Affichage des détails pour {model_name}, task_type={task_type}")

    # Importance des features (classification et régression)
    if task_type in ['classification', 'regression'] and model and model_result.get('feature_names'):
        st.markdown("#### Importance des Features")
        feature_plot = evaluator.create_feature_importance_plot(model, model_result['feature_names'])
        if feature_plot:
            st.plotly_chart(feature_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher l’importance des features")
            logger.warning(f"⚠️ Échec création importance des features pour {model_name}")

    # SHAP Summary Plot (classification et régression)
    if task_type in ['classification', 'regression'] and model and model_result.get('X_sample') is not None and not model_result.get('X_sample').empty:
        st.markdown("#### Analyse SHAP")
        shap_plot = evaluator.create_shap_plot(model_result)
        if shap_plot:
            st.plotly_chart(shap_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le SHAP plot")
            logger.warning(f"⚠️ Échec création SHAP plot pour {model_name}")

    # Nouvelles visualisations pour classification
    if task_type == 'classification' and model and model_result.get('X_test') is not None and model_result.get('y_test') is not None:
        # Matrice de confusion
        st.markdown("#### Matrice de Confusion")
        cm_plot = evaluator.create_confusion_matrix_plot(model_result)
        if cm_plot:
            st.plotly_chart(cm_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher la matrice de confusion")
            logger.warning(f"⚠️ Échec création matrice de confusion pour {model_name}: X_test={model_result.get('X_test') is not None}, y_test={model_result.get('y_test') is not None}")

        # Courbe ROC
        if hasattr(model, 'predict_proba'):
            st.markdown("#### Courbe ROC")
            roc_plot = evaluator.create_roc_curve_plot(model_result)
            if roc_plot:
                st.plotly_chart(roc_plot, width='stretch')
            else:
                st.warning("⚠️ Impossible d’afficher la courbe ROC")
                logger.warning(f"⚠️ Échec création courbe ROC pour {model_name}")
        else:
            st.warning("⚠️ Modèle ne supporte pas predict_proba pour la courbe ROC")
            logger.warning(f"⚠️ Modèle {model_name} sans predict_proba")

        # Courbe de Précision-Rappel
        if hasattr(model, 'predict_proba'):
            st.markdown("#### Courbe de Précision-Rappel")
            pr_plot = evaluator.create_precision_recall_curve_plot(model_result)
            if pr_plot:
                st.plotly_chart(pr_plot, width='stretch')
            else:
                st.warning("⚠️ Impossible d’afficher la courbe de précision-rappel")
                logger.warning(f"⚠️ Échec création courbe PR pour {model_name}")
        else:
            st.warning("⚠️ Modèle ne supporte pas predict_proba pour la courbe PR")
            logger.warning(f"⚠️ Modèle {model_name} sans predict_proba")

        # Courbe d’apprentissage
        if model_result.get('X_train') is not None and model_result.get('y_train') is not None:
            st.markdown("#### Courbe d’Apprentissage")
            learning_plot = evaluator.create_learning_curve_plot(model_result)
            if learning_plot:
                st.plotly_chart(learning_plot, width='stretch')
            else:
                st.warning("⚠️ Impossible d’afficher la courbe d’apprentissage")
                logger.warning(f"⚠️ Échec création courbe d’apprentissage pour {model_name}")
        else:
            st.warning("⚠️ Données d’entraînement (X_train, y_train) manquantes pour la courbe d’apprentissage")
            logger.warning(f"⚠️ X_train ou y_train manquant pour {model_name}")

        # Distribution des probabilités prédites
        if hasattr(model, 'predict_proba'):
            st.markdown("#### Distribution des Probabilités Prédites (Classe Positive)")
            proba_plot = evaluator.create_predicted_proba_distribution_plot(model_result)
            if proba_plot:
                st.plotly_chart(proba_plot, width='stretch')
            else:
                st.warning("⚠️ Impossible d’afficher la distribution des probabilités")
                logger.warning(f"⚠️ Échec création distribution probabilités pour {model_name}")
        else:
            st.warning("⚠️ Modèle ne supporte pas predict_proba pour la distribution des probabilités")
            logger.warning(f"⚠️ Modèle {model_name} sans predict_proba")

        # Heatmap de corrélation des features
        if model_result.get('X_sample') is not None and not model_result.get('X_sample').empty:
            st.markdown("#### Heatmap de Corrélation des Features")
            corr_plot = evaluator.create_feature_correlation_heatmap(model_result)
            if corr_plot:
                st.plotly_chart(corr_plot, width='stretch')
            else:
                st.warning("⚠️ Impossible d’afficher la heatmap de corrélation")
                logger.warning(f"⚠️ Échec création heatmap corrélation pour {model_name}")
        else:
            st.warning("⚠️ Données (X_sample) manquantes pour la heatmap de corrélation")
            logger.warning(f"⚠️ X_sample manquant pour {model_name}")

    # Régression : Graphique des résidus, Prédictions vs. Réelles
    if task_type == 'regression' and model and model_result.get('X_test') is not None and model_result.get('y_test') is not None:
        st.markdown("#### Graphique des Résidus")
        residuals_plot = evaluator.create_residuals_plot(model_result)
        if residuals_plot:
            st.plotly_chart(residuals_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le graphique des résidus")
            logger.warning(f"⚠️ Échec création graphique résidus pour {model_name}")

        st.markdown("#### Prédictions vs. Réelles")
        pred_vs_actual_plot = evaluator.create_predicted_vs_actual_plot(model_result)
        if pred_vs_actual_plot:
            st.plotly_chart(pred_vs_actual_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le graphique prédictions vs. réelles")
            logger.warning(f"⚠️ Échec création graphique prédictions vs. réelles pour {model_name}")

    # Clustering : Scatter plot, Silhouette plot, Dispersion intra-cluster
    if task_type == 'clustering' and model_result.get('X_sample') is not None and model_result.get('labels') is not None:
        st.markdown("#### Visualisation des Clusters")
        cluster_plot = evaluator.create_cluster_scatter_plot(model_result)
        if cluster_plot:
            st.plotly_chart(cluster_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le scatter plot des clusters")
            logger.warning(f"⚠️ Échec création scatter plot clusters pour {model_name}")
        
        st.markdown("#### Analyse de Silhouette")
        silhouette_plot = evaluator.create_silhouette_plot(model_result)
        if silhouette_plot:
            st.plotly_chart(silhouette_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le silhouette plot")
            logger.warning(f"⚠️ Échec création silhouette plot pour {model_name}")

        st.markdown("#### Dispersion Intra-Cluster")
        intra_cluster_plot = evaluator.create_intra_cluster_distance_plot(model_result)
        if intra_cluster_plot:
            st.plotly_chart(intra_cluster_plot, width='stretch')
        else:
            st.warning("⚠️ Impossible d’afficher le graphique de dispersion intra-cluster")
            logger.warning(f"⚠️ Échec création dispersion intra-cluster pour {model_name}")

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
        return

    if not validation["has_results"]:
        st.error("📭 Aucune donnée valide")
        return

    display_metrics_header(validation)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'Ensemble", "🔍 Détails", "📈 Métriques", "💾 Export"])
    
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