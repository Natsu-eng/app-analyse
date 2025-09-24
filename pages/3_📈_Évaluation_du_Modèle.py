import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import time
import logging
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
import os

# Configuration
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="Évaluation des Modèles", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration Production ---
def setup_evaluation_environment():
    """Configuration pour l'environnement de production"""
    if 'evaluation_setup_done' not in st.session_state:
        st.session_state.evaluation_setup_done = True
        
        # Masquer les éléments Streamlit en production
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_evaluation_environment()

# --- Styles CSS personnalisés ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin: 10px 0px;
    }
    .best-model {
        background-color: #e6f7ff;
        border-left: 4px solid #1890ff;
    }
    .model-comparison {
        background-color: #f6f6f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
    }
    .feature-importance {
        background-color: #fff7e6;
        border-left: 4px solid #fa8c16;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions utilitaires ---
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
        logger.error(f"Failed to get system metrics: {e}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def safe_get(obj, keys, default=None):
    """Accès sécurisé aux données nested"""
    try:
        for key in keys:
            obj = obj[key]
        return obj
    except (KeyError, TypeError, IndexError):
        return default

def format_metric_value(value, metric_name: str) -> str:
    """Formate les valeurs métriques selon leur type"""
    if value is None:
        return "N/A"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    
    if isinstance(value, float):
        if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'r2']:
            return f"{value:.3f}"
        elif metric_name in ['mse', 'mae', 'rmse']:
            return f"{value:.4f}"
        else:
            return f"{value:.3f}"
    
    return str(value)

def create_download_link(data, filename: str, file_type: str = "csv") -> str:
    """Crée un lien de téléchargement"""
    try:
        if file_type == "csv":
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Télécharger CSV</a>'
        elif file_type == "json":
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(json_str.encode()).decode()
            return f'<a href="data:application/json;base64,{b64}" download="{filename}.json">📥 Télécharger JSON</a>'
    except Exception as e:
        logger.error(f"Erreur création lien téléchargement: {e}")
    return ""

# --- Vérification des données ---
def validate_evaluation_data() -> Dict[str, Any]:
    """Valide la présence des données d'évaluation"""
    validation = {
        "has_results": False,
        "results_count": 0,
        "task_type": None,
        "best_model": None,
        "errors": []
    }
    
    try:
        if 'ml_results' not in st.session_state or not st.session_state.ml_results:
            validation["errors"].append("Aucun résultat d'expérimentation trouvé")
            return validation
        
        results = st.session_state.ml_results
        validation["results_count"] = len(results)
        validation["has_results"] = True
        
        # Déterminer le type de tâche
        if results and 'task_type' in results[0]:
            validation["task_type"] = results[0]['task_type']
        else:
            # Déduire du premier modèle qui a des métriques valides
            for result in results:
                metrics = result.get('metrics', {})
                if 'accuracy' in metrics:
                    validation["task_type"] = 'classification'
                    break
                elif 'r2' in metrics:
                    validation["task_type"] = 'regression'
                    break
                elif 'silhouette_score' in metrics:
                    validation["task_type"] = 'unsupervised'
                    break
        
        # Trouver le meilleur modèle
        successful_models = [r for r in results if not r.get('metrics', {}).get('error')]
        if successful_models:
            if validation["task_type"] == 'classification':
                successful_models.sort(key=lambda x: x.get('metrics', {}).get('accuracy', 0), reverse=True)
            elif validation["task_type"] == 'regression':
                successful_models.sort(key=lambda x: x.get('metrics', {}).get('r2', -float('inf')), reverse=True)
            elif validation["task_type"] == 'unsupervised':
                successful_models.sort(key=lambda x: x.get('metrics', {}).get('silhouette_score', -float('inf')), reverse=True)
            
            if successful_models:
                validation["best_model"] = successful_models[0]['model_name']
        
    except Exception as e:
        validation["errors"].append(f"Erreur validation données: {str(e)}")
        logger.error(f"Erreur validation évaluation: {e}")
    
    return validation

# --- Visualisations ---
def create_metrics_comparison_plot(results: List[Dict], task_type: str) -> go.Figure:
    """Crée un graphique de comparaison des métriques"""
    try:
        # Filtrer les modèles avec des métriques valides
        valid_results = [r for r in results if not r.get('metrics', {}).get('error')]
        
        if not valid_results:
            fig = go.Figure()
            fig.add_annotation(text="Aucune métrique valide disponible", x=0.5, y=0.5, showarrow=False)
            return fig
        
        model_names = [r['model_name'] for r in valid_results]
        
        if task_type == 'classification':
            # Métriques de classification
            accuracies = [safe_get(r, ['metrics', 'accuracy'], 0) for r in valid_results]
            f1_scores = [safe_get(r, ['metrics', 'f1_score'], 0) for r in valid_results]
            roc_aucs = [safe_get(r, ['metrics', 'roc_auc'], 0) for r in valid_results]
            
            fig = make_subplots(rows=1, cols=3, subplot_titles=('Accuracy', 'F1-Score', 'AUC-ROC'))
            
            fig.add_trace(go.Bar(x=model_names, y=accuracies, name='Accuracy'), 1, 1)
            fig.add_trace(go.Bar(x=model_names, y=f1_scores, name='F1-Score'), 1, 2)
            fig.add_trace(go.Bar(x=model_names, y=roc_aucs, name='AUC-ROC'), 1, 3)
            
        elif task_type == 'regression':
            # Métriques de régression
            r2_scores = [safe_get(r, ['metrics', 'r2'], 0) for r in valid_results]
            mae_scores = [safe_get(r, ['metrics', 'mae'], 0) for r in valid_results]
            rmse_scores = [safe_get(r, ['metrics', 'rmse'], 0) for r in valid_results]
            
            fig = make_subplots(rows=1, cols=3, subplot_titles=('R² Score', 'MAE', 'RMSE'))
            
            fig.add_trace(go.Bar(x=model_names, y=r2_scores, name='R²'), 1, 1)
            fig.add_trace(go.Bar(x=model_names, y=mae_scores, name='MAE'), 1, 2)
            fig.add_trace(go.Bar(x=model_names, y=rmse_scores, name='RMSE'), 1, 3)
            
        else:
            # Métriques non supervisées
            silhouette_scores = [safe_get(r, ['metrics', 'silhouette_score'], 0) for r in valid_results]
            n_clusters = [safe_get(r, ['metrics', 'n_clusters'], 0) for r in valid_results]
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Silhouette Score', 'Nombre de Clusters'))
            
            fig.add_trace(go.Bar(x=model_names, y=silhouette_scores, name='Silhouette'), 1, 1)
            fig.add_trace(go.Bar(x=model_names, y=n_clusters, name='Clusters'), 1, 2)
        
        fig.update_layout(
            title=f"Comparaison des Modèles - {task_type.upper()}",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création graphique comparaison: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Erreur lors de la création du graphique", x=0.5, y=0.5, showarrow=False)
        return fig

def create_confusion_matrix_plot(confusion_matrix: List[List[int]], class_names: List[str] = None) -> go.Figure:
    """Crée une heatmap de matrice de confusion"""
    try:
        if confusion_matrix is None:
            fig = go.Figure()
            fig.add_annotation(text="Matrice de confusion non disponible", x=0.5, y=0.5, showarrow=False)
            return fig
        
        cm_array = np.array(confusion_matrix)
        
        if class_names is None:
            class_names = [f"Classe {i}" for i in range(len(cm_array))]
        
        fig = px.imshow(
            cm_array,
            labels=dict(x="Prédit", y="Réel", color="Count"),
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            aspect="auto"
        )
        
        # Ajouter les annotations
        for i in range(len(cm_array)):
            for j in range(len(cm_array[i])):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm_array[i, j]),
                    showarrow=False,
                    font=dict(color="red" if cm_array[i, j] > cm_array.max() / 2 else "black")
                )
        
        fig.update_layout(
            title="Matrice de Confusion",
            xaxis_title="Prédit",
            yaxis_title="Réel"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création matrice confusion: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Erreur création matrice de confusion", x=0.5, y=0.5, showarrow=False)
        return fig

def create_feature_importance_plot(model, feature_names: List[str]) -> go.Figure:
    """Crée un graphique d'importance des features"""
    try:
        # Vérifier si le modèle a une importance des features
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_.flatten())
        else:
            fig = go.Figure()
            fig.add_annotation(text="Importance des features non disponible", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Créer le graphique
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        fig = go.Figure(go.Bar(
            x=sorted_importances[:10],  # Top 10 features
            y=sorted_features[:10],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Top 10 - Importance des Features",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création importance features: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Erreur création importance features", x=0.5, y=0.5, showarrow=False)
        return fig

# --- Interface principale ---
st.title("📈 Évaluation des Modèles")

# Vérification des données
validation = validate_evaluation_data()

if not validation["has_results"]:
    st.error("❌ Aucun résultat d'expérimentation trouvé")
    st.info("""
    **Pour utiliser cette page :**
    1. Allez sur la page **⚙️ Configuration ML**
    2. Configurez et lancez une expérimentation
    3. Revenez sur cette page pour voir les résultats
    """)
    st.page_link("pages/2_⚙️_Configuration_ML.py", label="🔧 Aller à la configuration ML", icon="⚙️")
    st.stop()

# Métriques système
system_metrics = get_system_metrics()
with st.sidebar:
    st.subheader("📊 Statistiques")
    st.write(f"**Modèles évalués :** {validation['results_count']}")
    if validation['best_model']:
        st.write(f"**Meilleur modèle :** {validation['best_model']}")
    st.write(f"**Type de tâche :** {validation['task_type']}")
    
    st.subheader("🔧 Actions")
    if st.button("🔄 Actualiser les données"):
        st.rerun()
    
    if st.button("🧹 Nettoyer la mémoire"):
        gc.collect()
        st.success("Mémoire nettoyée!")
    
    st.subheader("💾 Export")
    if st.button("📊 Exporter les résultats"):
        # Créer un rapport exportable
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": validation["task_type"],
            "models_count": validation["results_count"],
            "best_model": validation["best_model"],
            "results": st.session_state.ml_results
        }
        
        # Convertir en JSON téléchargeable
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Télécharger le rapport JSON",
            data=json_str,
            file_name=f"rapport_ml_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Header avec résumé
st.success(f"✅ Expérimentation terminée - {validation['results_count']} modèles évalués")

# Métriques rapides
successful_models = [r for r in st.session_state.ml_results if not r.get('metrics', {}).get('error')]
failed_models = [r for r in st.session_state.ml_results if r.get('metrics', {}).get('error')]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modèles réussis", len(successful_models))
with col2:
    st.metric("Modèles échoués", len(failed_models))
with col3:
    if validation['task_type'] == 'classification':
        best_score = safe_get(successful_models[0] if successful_models else {}, ['metrics', 'accuracy'], 0)
        st.metric("Meilleure accuracy", f"{best_score:.3f}")
    elif validation['task_type'] == 'regression':
        best_score = safe_get(successful_models[0] if successful_models else {}, ['metrics', 'r2'], 0)
        st.metric("Meilleur R²", f"{best_score:.3f}")
with col4:
    st.metric("RAM utilisée", f"{system_metrics['memory_percent']:.1f}%")

# --- Onglets principaux ---
tab_comparison, tab_individual, tab_details, tab_export = st.tabs([
    "📊 Comparaison Globale", 
    "🔍 Analyse Individuelle", 
    "📋 Détails Techniques", 
    "💾 Export & Rapport"
])

# Onglet 1: Comparaison Globale
with tab_comparison:
    st.header("📊 Comparaison des Performances")
    
    # Graphique de comparaison
    fig = create_metrics_comparison_plot(st.session_state.ml_results, validation["task_type"])
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de comparaison détaillé
    st.subheader("📋 Tableau Comparatif")
    
    comparison_data = []
    for result in st.session_state.ml_results:
        model_info = {
            'Modèle': result['model_name'],
            'Statut': '✅ Succès' if not result.get('metrics', {}).get('error') else '❌ Échec',
            'Temps (s)': safe_get(result, ['training_time'], 'N/A')
        }
        
        metrics = result.get('metrics', {})
        if validation["task_type"] == 'classification':
            model_info.update({
                'Accuracy': safe_get(metrics, ['accuracy']),
                'F1-Score': safe_get(metrics, ['f1_score']),
                'AUC-ROC': safe_get(metrics, ['roc_auc']),
                'Précision': safe_get(metrics, ['precision']),
                'Rappel': safe_get(metrics, ['recall'])
            })
        elif validation["task_type"] == 'regression':
            model_info.update({
                'R²': safe_get(metrics, ['r2']),
                'MAE': safe_get(metrics, ['mae']),
                'RMSE': safe_get(metrics, ['rmse']),
                'MSE': safe_get(metrics, ['mse'])
            })
        elif validation["task_type"] == 'unsupervised':
            model_info.update({
                'Silhouette': safe_get(metrics, ['silhouette_score']),
                'Clusters': safe_get(metrics, ['n_clusters']),
                'Davies-Bouldin': safe_get(metrics, ['davies_bouldin_score'])
            })
        
        comparison_data.append(model_info)
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, height=400)
    
    # Bouton d'export
    st.markdown(create_download_link(df_comparison, "comparaison_modeles", "csv"), unsafe_allow_html=True)

# Onglet 2: Analyse Individuelle
with tab_individual:
    st.header("🔍 Analyse par Modèle")
    
    # Sélecteur de modèle
    model_names = [r['model_name'] for r in st.session_state.ml_results]
    selected_model = st.selectbox("Sélectionnez un modèle à analyser", model_names)
    
    # Récupérer les données du modèle sélectionné
    selected_result = next((r for r in st.session_state.ml_results if r['model_name'] == selected_model), None)
    
    if selected_result:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Performances de {selected_model}")
            
            metrics = selected_result.get('metrics', {})
            if metrics.get('error'):
                st.error(f"❌ Erreur lors de l'entraînement : {metrics['error']}")
            else:
                # Affichage des métriques principales
                if validation["task_type"] == 'classification':
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Accuracy", format_metric_value(metrics.get('accuracy'), 'accuracy'))
                    with cols[1]:
                        st.metric("F1-Score", format_metric_value(metrics.get('f1_score'), 'f1_score'))
                    with cols[2]:
                        st.metric("AUC-ROC", format_metric_value(metrics.get('roc_auc'), 'roc_auc'))
                    with cols[3]:
                        st.metric("Précision", format_metric_value(metrics.get('precision'), 'precision'))
                    
                    # Matrice de confusion si disponible
                    if 'confusion_matrix' in metrics:
                        st.subheader("🎯 Matrice de Confusion")
                        cm_fig = create_confusion_matrix_plot(metrics['confusion_matrix'])
                        st.plotly_chart(cm_fig, use_container_width=True)
                
                elif validation["task_type"] == 'regression':
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("R² Score", format_metric_value(metrics.get('r2'), 'r2'))
                    with cols[1]:
                        st.metric("MAE", format_metric_value(metrics.get('mae'), 'mae'))
                    with cols[2]:
                        st.metric("RMSE", format_metric_value(metrics.get('rmse'), 'rmse'))
                    with cols[3]:
                        st.metric("MSE", format_metric_value(metrics.get('mse'), 'mse'))
                
                elif validation["task_type"] == 'unsupervised':
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Score Silhouette", format_metric_value(metrics.get('silhouette_score'), 'silhouette_score'))
                    with cols[1]:
                        st.metric("Nombre de Clusters", metrics.get('n_clusters', 'N/A'))
                    with cols[2]:
                        st.metric("Davies-Bouldin", format_metric_value(metrics.get('davies_bouldin_score'), 'davies_bouldin_score'))
        
        with col2:
            st.subheader("📊 Informations")
            
            # Carte d'information du modèle
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Informations Modèle</h4>
                <p><strong>Statut:</strong> {'✅ Entraîné' if not metrics.get('error') else '❌ Erreur'}</p>
                <p><strong>Temps d'entraînement:</strong> {safe_get(selected_result, ['training_time'], 'N/A')}s</p>
                <p><strong>Type de tâche:</strong> {validation['task_type']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Importance des features si disponible
            if 'model' in selected_result and selected_result['model'] is not None:
                try:
                    feature_names = selected_result.get('feature_names', [])
                    if feature_names:
                        importance_fig = create_feature_importance_plot(selected_result['model'], feature_names)
                        st.plotly_chart(importance_fig, use_container_width=True)
                except Exception as e:
                    logger.debug(f"Importance features non disponible: {e}")

# Onglet 3: Détails Techniques
with tab_details:
    st.header("📋 Détails Techniques")
    
    # Sélecteur de modèle pour les détails
    model_for_details = st.selectbox("Modèle pour détails techniques", model_names, key="details_selector")
    detailed_result = next((r for r in st.session_state.ml_results if r['model_name'] == model_for_details), None)
    
    if detailed_result:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Hyperparamètres
            st.write("**Hyperparamètres optimisés :**")
            best_params = detailed_result.get('best_params', {})
            if best_params:
                for param, value in best_params.items():
                    st.write(f"- `{param}`: `{value}`")
            else:
                st.info("Aucun hyperparamètre optimisé (entraînement standard)")
            
            # Features utilisées
            feature_list = detailed_result.get('feature_names', [])
            if feature_list:
                st.write(f"**Features utilisées ({len(feature_list)}) :**")
                features_text = ", ".join([f"`{f}`" for f in feature_list[:10]])
                if len(feature_list) > 10:
                    features_text += f" ... et {len(feature_list) - 10} de plus"
                st.write(features_text)
        
        with col2:
            st.subheader("📈 Métriques Détaillées")
            
            metrics = detailed_result.get('metrics', {})
            if not metrics.get('error'):
                # Affichage formaté des métriques
                metric_df_data = []
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ['confusion_matrix', 'classification_report', 'error_stats']:
                        metric_df_data.append({
                            'Métrique': metric_name.replace('_', ' ').title(),
                            'Valeur': format_metric_value(metric_value, metric_name)
                        })
                
                if metric_df_data:
                    metric_df = pd.DataFrame(metric_df_data)
                    st.dataframe(metric_df, use_container_width=True, height=300)
                
                # Rapport de classification détaillé
                if 'classification_report' in metrics and validation["task_type"] == 'classification':
                    with st.expander("📊 Rapport de Classification Détaillé"):
                        try:
                            report_df = pd.DataFrame(metrics['classification_report']).transpose()
                            st.dataframe(report_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur affichage rapport: {e}")

# Onglet 4: Export & Rapport
with tab_export:
    st.header("💾 Export des Résultats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export des Données")
        
        # Format d'export
        export_format = st.radio("Format d'export", ["CSV", "JSON", "Les deux"])
        
        # Options d'export
        include_metrics = st.checkbox("Inclure les métriques détaillées", value=True)
        include_config = st.checkbox("Inclure la configuration", value=True)
        include_models = st.checkbox("Inclure les modèles (fichiers .joblib)", value=False)
        
        # Bouton d'export
        if st.button("🚀 Générer l'export complet", type="primary"):
            with st.spinner("Génération de l'export en cours..."):
                try:
                    # Préparation des données d'export
                    export_data = {
                        "metadata": {
                            "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "task_type": validation["task_type"],
                            "models_count": len(st.session_state.ml_results),
                            "best_model": validation["best_model"]
                        },
                        "results": []
                    }
                    
                    # Ajout des résultats détaillés
                    for result in st.session_state.ml_results:
                        result_export = {
                            "model_name": result['model_name'],
                            "training_time": result.get('training_time'),
                            "status": "success" if not result.get('metrics', {}).get('error') else "error"
                        }
                        
                        if include_metrics:
                            result_export["metrics"] = result.get('metrics', {})
                        
                        if include_config:
                            result_export["best_params"] = result.get('best_params', {})
                            result_export["feature_names"] = result.get('feature_names', [])
                        
                        export_data["results"].append(result_export)
                    
                    # Génération des fichiers
                    if export_format in ["JSON", "Les deux"]:
                        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Télécharger JSON",
                            data=json_str,
                            file_name=f"rapport_ml_complet_{time.strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    if export_format in ["CSV", "Les deux"]:
                        # Création d'un CSV simplifié
                        csv_data = []
                        for result in st.session_state.ml_results:
                            row = {
                                "model_name": result['model_name'],
                                "training_time": result.get('training_time'),
                                "status": "success" if not result.get('metrics', {}).get('error') else "error"
                            }
                            
                            # Ajout des métriques principales
                            metrics = result.get('metrics', {})
                            for metric_name in ['accuracy', 'r2', 'silhouette_score', 'f1_score', 'mae', 'rmse']:
                                if metric_name in metrics:
                                    row[metric_name] = metrics[metric_name]
                            
                            csv_data.append(row)
                        
                        df_export = pd.DataFrame(csv_data)
                        csv_export = df_export.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv_export,
                            file_name=f"rapport_ml_simplifie_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    st.success("✅ Export généré avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'export: {e}")
    
    with col2:
        st.subheader("📋 Rapport Automatique")
        
        # Génération de rapport synthétique
        if st.button("📄 Générer le rapport synthétique"):
            try:
                # Analyse des résultats
                successful_count = len(successful_models)
                failed_count = len(failed_models)
                
                # Meilleur modèle
                best_model_info = successful_models[0] if successful_models else None
                
                # Création du rapport
                report = f"""
# 📊 RAPPORT D'EXPÉRIMENTATION ML

## 📋 Résumé Exécutif
- **Date de génération**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Type de tâche**: {validation['task_type'].upper()}
- **Modèles évalués**: {len(st.session_state.ml_results)}
- **Taux de succès**: {successful_count}/{len(st.session_state.ml_results)} ({(successful_count/len(st.session_state.ml_results)*100):.1f}%)

## 🏆 Meilleur Modèle
- **Nom**: {best_model_info['model_name'] if best_model_info else 'Aucun'}
- **Score principal**: {safe_get(best_model_info, ['metrics', 'accuracy' if validation['task_type'] == 'classification' else 'r2'], 'N/A')}

## 📈 Recommandations
"""
                
                if validation['task_type'] == 'classification':
                    best_accuracy = safe_get(best_model_info, ['metrics', 'accuracy'], 0)
                    if best_accuracy > 0.9:
                        report += "- ✅ Performance excellente, modèle prêt pour la production\n"
                    elif best_accuracy > 0.7:
                        report += "- ⚠️ Performance acceptable, envisager l'optimisation\n"
                    else:
                        report += "- ❌ Performance faible, revoir l'approche\n"
                
                elif validation['task_type'] == 'regression':
                    best_r2 = safe_get(best_model_info, ['metrics', 'r2'], 0)
                    if best_r2 > 0.8:
                        report += "- ✅ Très bon pouvoir prédictif\n"
                    elif best_r2 > 0.5:
                        report += "- ⚠️ Pouvoir prédictif acceptable\n"
                    else:
                        report += "- ❌ Faible pouvoir prédictif\n"
                
                st.text_area("Rapport synthétique", report, height=300)
                
                # Bouton de téléchargement du rapport
                st.download_button(
                    label="📥 Télécharger le rapport",
                    data=report,
                    file_name=f"rapport_synthetique_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Erreur génération rapport: {e}")

# --- Footer avec monitoring ---
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption(f"📊 Données chargées: {time.strftime('%H:%M:%S')}")

with footer_col2:
    st.caption(f"💾 RAM: {system_metrics['memory_percent']:.1f}%")

with footer_col3:
    if st.button("🔄 Rafraîchir la page"):
        st.rerun()

# Message d'aide
with st.expander("ℹ️ Aide et Informations", expanded=False):
    st.markdown("""
    **Guide d'utilisation :**
    - **Comparaison Globale** : Vue d'ensemble des performances de tous les modèles
    - **Analyse Individuelle** : Détails spécifiques à chaque modèle
    - **Détails Techniques** : Configuration et métriques avancées
    - **Export & Rapport** : Génération de rapports et exports
    
    **Conseils :**
    - Utilisez l'export JSON pour une analyse approfondie
    - Consultez les métriques détaillées pour comprendre les performances
    - Téléchargez les rapports pour documentation
    """)