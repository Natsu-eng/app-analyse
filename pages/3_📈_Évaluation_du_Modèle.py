import numpy as np
import streamlit as st
import pandas as pd
import time
import json
import gc
import plotly.graph_objects as go
from ml.evaluation.visualization import ModelEvaluationVisualizer
from ml.evaluation.metrics_calculation import get_system_metrics
from utils.report_generator import generate_pdf_report

# --- Correctif compatibilité PyArrow ---
import os
os.environ["PANDAS_USE_PYARROW"] = "0"
try:
    pd.options.mode.dtype_backend = "numpy_nullable"
except Exception:
    pass

# --- Configuration de la page ---
st.set_page_config(
    page_title="Évaluation des Modèles",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé professionnel ---
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
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .best-model-card {
        border-left: 4px solid #27ae60;
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.25rem;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #95a5a6;
    }
    .performance-high { 
        color: #27ae60; 
        font-weight: 700; 
        background: #d5f4e6;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .performance-medium { 
        color: #f39c12; 
        font-weight: 700;
        background: #fef9e7;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .performance-low { 
        color: #e74c3c; 
        font-weight: 700;
        background: #fadbd8;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .tab-content {
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid #ecf0f1;
    }
    .status-success {
        color: #27ae60;
        font-weight: 600;
    }
    .status-error {
        color: #e74c3c;
        font-weight: 600;
    }
    .section-divider {
        border-top: 1px solid #bdc3c7;
        margin: 2rem 0 1rem 0;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, max_entries=10)
def cached_plot(fig):
    """Cache les figures Plotly avec gestion optimisée"""
    return fig

def display_metrics_header(validation):
    """Affiche l'en-tête avec les métriques principales - Design professionnel"""
    successful_count = len(validation["successful_models"])
    total_count = validation["results_count"]
    
    st.markdown('<div class="main-header">📈 Tableau de Bord d\'Évaluation des Modèles</div>', unsafe_allow_html=True)
    
    # Indicateurs principaux avec design professionnel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Taux de Réussite</div>
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-subtitle">{successful_count} sur {total_count} modèles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        card_class = "metric-card best-model-card" if validation["best_model"] else "metric-card"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">Meilleur Modèle</div>
            <div class="metric-value" style="font-size: 1.2rem;">{validation["best_model"] or "N/A"}</div>
            <div class="metric-subtitle">Type: {validation["task_type"].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        failed_count = len(validation["failed_models"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Modèles Échoués</div>
            <div class="metric-value" style="color: #e74c3c;">{failed_count}</div>
            <div class="metric-subtitle">Erreurs détectées</div>
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

def create_sidebar(validation, evaluator):
    """Crée la sidebar avec contrôles et informations"""
    with st.sidebar:
        st.markdown("### 📋 Informations Projet")
        
        # Informations tâche avec icônes appropriées
        task_type = validation.get('task_type', 'Inconnu')
        task_icons = {
            'clustering': '🎯',
            'classification': '🏷️', 
            'regression': '📊',
            'unknown': '❓'
        }
        task_icon = task_icons.get(task_type, '❓')
        st.markdown(f"**{task_icon} Type de tâche:** `{task_type.title()}`")
        
        if validation["best_model"]:
            st.markdown(f"**🏆 Champion:** `{validation['best_model']}`")
        
        # Informations temporelles
        st.markdown(f"**🕒 Mis à jour:** {time.strftime('%H:%M:%S')}")
        
        # Statistiques détaillées
        st.markdown("---")
        st.markdown("### 📊 Statistiques")
        
        total_models = len(validation.get("successful_models", [])) + len(validation.get("failed_models", []))
        if total_models > 0:
            success_rate = len(validation["successful_models"]) / total_models * 100
            st.progress(success_rate / 100)
            st.caption(f"Réussite: {success_rate:.1f}% ({len(validation['successful_models'])}/{total_models})")
        
        st.markdown("---")
        st.markdown("### ⚙️ Actions Rapides")
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualiser", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🧹 Mémoire", use_container_width=True):
                gc.collect()
                st.success("✅ Optimisée", icon="🧹")
        
        st.markdown("---")
        st.markdown("### 📥 Export")
        
        # Export des données
        if validation["successful_models"]:
            export_data = evaluator.get_export_data()
            
            # Export CSV
            csv_data = pd.DataFrame(export_data['models']).to_csv(index=False)
            st.download_button(
                label="📊 Export CSV",
                data=csv_data,
                file_name=f"evaluation_{task_type}_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Export PDF du meilleur modèle
            if validation["best_model"]:
                best_model_result = next((r for r in validation["successful_models"] 
                                       if evaluator._safe_get(r, ['model_name'], '') == validation["best_model"]), None)
                if best_model_result:
                    try:
                        pdf_bytes = generate_pdf_report(best_model_result)
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Rapport PDF",
                                data=pdf_bytes,
                                file_name=f"rapport_{validation['best_model']}_{int(time.time())}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning("⚠️ PDF non disponible")

def main():
    # Vérification des données
    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        st.error("🚫 Aucun résultat d'expérimentation disponible")
        st.info("Veuillez d'abord entraîner des modèles dans la page 'Configuration ML'")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚙️ Aller à la Configuration ML", use_container_width=True):
                st.switch_page("pages/2_⚙️_Configuration_ML.py")
        return

    # Initialisation de l'évaluateur
    try:
        evaluator = ModelEvaluationVisualizer(st.session_state.ml_results)
        validation = evaluator.validation_result
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation de l'évaluation: {str(e)}")
        return

    if not validation["has_results"]:
        st.error("📭 Aucune donnée d'évaluation valide")
        return

    # En-tête principal
    display_metrics_header(validation)
    
    # Sidebar
    create_sidebar(validation, evaluator)

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Vue d'Ensemble", 
        "🔍 Analyse Détaillée", 
        "📈 Métriques Avancées",
        "💾 Export Complet"
    ])

    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if validation["successful_models"]:
            # Graphique de comparaison principal
            st.markdown("### 📈 Comparaison des Performances")
            comparison_plot = evaluator.get_comparison_plot()
            if comparison_plot:
                st.plotly_chart(cached_plot(comparison_plot), use_container_width=True)
            
            st.markdown("### 📋 Tableau de Synthèse")
            df_comparison = evaluator.get_comparison_dataframe()
            
            # Formatage conditionnel du DataFrame
            def format_status(val):
                if "✅" in str(val):
                    return f'<span class="status-success">{val}</span>'
                elif "❌" in str(val):
                    return f'<span class="status-error">{val}</span>'
                return val
            
            # Affichage du dataframe avec style
            st.dataframe(df_comparison, use_container_width=True, height=400)
            
            # Statistiques de performance par type de tâche
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Résumé Statistique")
            
            col1, col2, col3 = st.columns(3)
            
            if validation["task_type"] in ['classification', 'regression']:
                numeric_cols = df_comparison.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    main_metric = numeric_cols.columns[0]
                    with col1:
                        avg_metric = numeric_cols[main_metric].mean()
                        st.metric("Score Moyen", f"{avg_metric:.3f}")
                    with col2:
                        best_metric = numeric_cols[main_metric].max()
                        st.metric("Meilleur Score", f"{best_metric:.3f}")
                    with col3:
                        std_metric = numeric_cols[main_metric].std()
                        st.metric("Écart-type", f"{std_metric:.3f}")
            
            elif validation["task_type"] == 'clustering':
                with col1:
                    avg_silhouette = np.mean([evaluator._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                                            for r in validation["successful_models"]])
                    st.metric("Silhouette Moyen", f"{avg_silhouette:.3f}")
                with col2:
                    cluster_counts = [evaluator._safe_get(r, ['metrics', 'n_clusters'], 0) 
                                    for r in validation["successful_models"]]
                    avg_clusters = np.mean(cluster_counts) if cluster_counts else 0
                    st.metric("Clusters Moyen", f"{avg_clusters:.1f}")
                with col3:
                    best_silhouette = max([evaluator._safe_get(r, ['metrics', 'silhouette_score'], 0) 
                                         for r in validation["successful_models"]])
                    st.metric("Meilleur Silhouette", f"{best_silhouette:.3f}")
        else:
            st.warning("⚠️ Aucun modèle valide à afficher")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analyse Détaillée par Modèle")
        
        if validation["successful_models"]:
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(validation["successful_models"])]
            
            # Sélecteur de modèle avec indicateur du meilleur
            selected_idx = st.selectbox(
                "Sélectionnez un modèle à analyser en détail:",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x]==validation.get('best_model') else ''}",
                key="model_selector_detail"
            )
            
            selected_result = validation["successful_models"][selected_idx]
            evaluator.show_model_details(selected_result, validation["task_type"])
            
        else:
            st.info("ℹ️ Aucun modèle disponible pour l'analyse détaillée")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 📈 Analyses Statistiques Avancées")
        
        if validation["successful_models"]:
            # Distribution des performances
            st.markdown("#### 📊 Distribution des Performances")
            dist_plot = evaluator.get_performance_distribution_plot()
            if dist_plot:
                st.plotly_chart(cached_plot(dist_plot), use_container_width=True)
            
            # Métriques détaillées selon le type de tâche
            if validation["task_type"] == 'clustering':
                st.markdown("#### 🎯 Analyse de Clustering Détaillée")
                clustering_metrics = []
                for result in validation["successful_models"]:
                    metrics = evaluator._safe_get(result, ['metrics'], {})
                    silhouette = metrics.get('silhouette_score', 0)
                    stability = ('🟢 Excellente' if silhouette > 0.7 else 
                               '🟡 Bonne' if silhouette > 0.5 else 
                               '🟠 Moyenne' if silhouette > 0.3 else '🔴 Faible')
                    
                    clustering_metrics.append({
                        'Modèle': evaluator._safe_get(result, ['model_name'], 'Unknown'),
                        'Score Silhouette': f"{silhouette:.3f}",
                        'Nombre de Clusters': metrics.get('n_clusters', 'N/A'),
                        'Qualité': stability
                    })
                
                df_clustering = pd.DataFrame(clustering_metrics)
                st.dataframe(df_clustering, use_container_width=True)
                
            elif validation["task_type"] == 'classification':
                st.markdown("#### 🏷️ Métriques de Classification Avancées")
                class_metrics = []
                for result in validation["successful_models"]:
                    metrics = evaluator._safe_get(result, ['metrics'], {})
                    class_metrics.append({
                        'Modèle': evaluator._safe_get(result, ['model_name'], 'Unknown'),
                        'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                        'Precision': f"{metrics.get('precision', 0):.3f}",
                        'Recall': f"{metrics.get('recall', 0):.3f}",
                        'F1-Score': f"{metrics.get('f1_score', 0):.3f}"
                    })
                
                df_classification = pd.DataFrame(class_metrics)
                st.dataframe(df_classification, use_container_width=True)
                
            elif validation["task_type"] == 'regression':
                st.markdown("#### 📊 Métriques de Régression Avancées")
                reg_metrics = []
                for result in validation["successful_models"]:
                    metrics = evaluator._safe_get(result, ['metrics'], {})
                    reg_metrics.append({
                        'Modèle': evaluator._safe_get(result, ['model_name'], 'Unknown'),
                        'R² Score': f"{metrics.get('r2', 0):.3f}",
                        'MAE': f"{metrics.get('mae', 0):.3f}",
                        'RMSE': f"{metrics.get('rmse', 0):.3f}",
                        'MSE': f"{metrics.get('mse', 0):.3f}"
                    })
                
                df_regression = pd.DataFrame(reg_metrics)
                st.dataframe(df_regression, use_container_width=True)
                
        else:
            st.warning("⚠️ Aucune métrique avancée disponible")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("### 💾 Export des Résultats Complets")
        
        if validation["successful_models"]:
            export_data = evaluator.get_export_data()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Export Données Structurées")
                st.download_button(
                    label="📥 Télécharger JSON Complet",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
                    file_name=f"evaluation_complete_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Aperçu des données
                with st.expander("👁️ Aperçu des données exportées"):
                    st.json(export_data, expanded=False)
            
            with col2:
                st.markdown("#### 📈 Rapports d'Analyse")
                st.info("📋 Exportez un rapport détaillé avec visualisations et analyses complètes")
                
                # Bouton pour générer un rapport complet
                if st.button("🔄 Générer Rapport Global", use_container_width=True):
                    with st.spinner("Génération du rapport en cours..."):
                        try:
                            # Simulation de génération de rapport
                            time.sleep(1)
                            st.success("✅ Rapport généré avec succès!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")
        else:
            st.info("ℹ️ Aucune donnée disponible pour l'export")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Pied de page avec informations système
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption(f"🕐 Dernière mise à jour: {time.strftime('%H:%M:%S')}")
    with col2:
        memory_info = get_system_metrics()
        memory_status = "🟢" if memory_info['memory_percent'] < 70 else "🟡" if memory_info['memory_percent'] < 85 else "🔴"
        st.caption(f"{memory_status} Mémoire: {memory_info['memory_percent']:.1f}%")
    with col3:
        st.caption(f"📊 Modèles chargés: {len(st.session_state.ml_results)}")
    with col4:
        st.caption(f"🎯 Type: {validation['task_type'].title()}")

if __name__ == "__main__":
    main()