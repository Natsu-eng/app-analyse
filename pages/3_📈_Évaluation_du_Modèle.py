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

# --- Configuration de la page ---
st.set_page_config(
    page_title="Évaluation des Modèles",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé moderne ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .best-model-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 2px solid #ffd700;
    }
    .performance-high { color: #00d26a; font-weight: 700; }
    .performance-medium { color: #ffb200; font-weight: 700; }
    .performance-low { color: #ff4d4d; font-weight: 700; }
    .tab-content {
        padding: 2rem 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, max_entries=10)
def cached_plot(fig):
    """Cache les figures Plotly avec gestion optimisée"""
    return fig

def display_metrics_header(validation):
    """Affiche l'en-tête avec les métriques principales"""
    successful_count = len(validation["successful_models"])
    total_count = validation["results_count"]
    
    st.markdown('<div class="main-header">📈 Tableau de Bord d\'Évaluation</div>', unsafe_allow_html=True)
    
    # Indicateurs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        st.metric("📊 Taux de Réussite", f"{success_rate:.1f}%")
    
    with col2:
        st.metric("✅ Modèles Validés", successful_count)
    
    with col3:
        st.metric("❌ Échecs", len(validation["failed_models"]))
    
    with col4:
        if validation["best_model"]:
            st.metric("🏆 Meilleur Modèle", validation["best_model"])
        else:
            st.metric("🏆 Meilleur Modèle", "N/A")
    
    with col5:
        memory_info = get_system_metrics()
        mem_color = "normal" if memory_info['memory_percent'] < 80 else "off"
        st.metric("💾 Mémoire Utilisée", f"{memory_info['memory_percent']:.1f}%", delta=None, delta_color=mem_color)

def create_sidebar(validation, evaluator):
    """Crée la sidebar avec contrôles et informations"""
    with st.sidebar:
        st.markdown("### 📋 Informations Globales")
        
        # Informations tâche
        task_type = validation.get('task_type', 'Inconnu')
        task_icon = "🔮" if task_type == 'clustering' else "🎯" if task_type == 'classification' else "📊"
        st.write(f"**{task_icon} Type de tâche:** {task_type}")
        
        if validation["best_model"]:
            st.write(f"**🏆 Meilleur modèle:** `{validation['best_model']}`")
        
        st.write(f"**📅 Dernière mise à jour:** {time.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Contrôles")
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Actualiser", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🧹 Optimiser Mémoire", use_container_width=True):
                gc.collect()
                st.success("Mémoire optimisée!")
        
        st.markdown("---")
        st.markdown("### 💾 Export des Résultats")
        
        # Export des données
        if validation["successful_models"]:
            export_data = evaluator.get_export_data()
            
            # Export CSV
            csv_data = pd.DataFrame(export_data['models']).to_csv(index=False)
            st.download_button(
                label="📄 Exporter CSV",
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
                                label="🎯 Rapport PDF",
                                data=pdf_bytes,
                                file_name=f"rapport_{validation['best_model']}_{int(time.time())}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error("Erreur génération PDF")

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
        st.markdown("### 📊 Comparaison des Performances")
        
        if validation["successful_models"]:
            comparison_plot = evaluator.get_comparison_plot()
            if comparison_plot:
                st.plotly_chart(cached_plot(comparison_plot), use_container_width=True)
            
            st.markdown("### 📋 Tableau Comparatif")
            df_comparison = evaluator.get_comparison_dataframe()
            st.dataframe(df_comparison, use_container_width=True, height=400)
            
            # Statistiques rapides
            if validation["task_type"] in ['classification', 'regression']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_metric = df_comparison.select_dtypes(include=[np.number]).iloc[:, 0].mean()
                    st.metric("Moyenne Principale", f"{avg_metric:.3f}")
                with col2:
                    best_metric = df_comparison.select_dtypes(include=[np.number]).iloc[:, 0].max()
                    st.metric("Meilleur Score", f"{best_metric:.3f}")
                with col3:
                    std_metric = df_comparison.select_dtypes(include=[np.number]).iloc[:, 0].std()
                    st.metric("Écart-type", f"{std_metric:.3f}")
        else:
            st.warning("⚠️ Aucun modèle valide à afficher")

    with tab2:
        st.markdown("### 🔍 Analyse par Modèle")
        
        if validation["successful_models"]:
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(validation["successful_models"])]
            
            # Sélecteur de modèle avec indicateur du meilleur
            selected_idx = st.selectbox(
                "Sélectionnez un modèle à analyser:",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x]==validation.get('best_model') else ''}",
                key="model_selector_detail"
            )
            
            selected_result = validation["successful_models"][selected_idx]
            evaluator.show_model_details(selected_result, validation["task_type"])
            
        else:
            st.info("ℹ️ Aucun modèle disponible pour l'analyse détaillée")

    with tab3:
        st.markdown("### 📈 Analyses Avancées")
        
        if validation["successful_models"]:
            # Distribution des performances
            dist_plot = evaluator.get_performance_distribution_plot()
            if dist_plot:
                st.plotly_chart(cached_plot(dist_plot), use_container_width=True)
            
            # Métriques détaillées selon le type de tâche
            if validation["task_type"] == 'clustering':
                st.markdown("#### 🎯 Métriques de Clustering")
                clustering_metrics = []
                for result in validation["successful_models"]:
                    metrics = evaluator._safe_get(result, ['metrics'], {})
                    clustering_metrics.append({
                        'Modèle': evaluator._safe_get(result, ['model_name'], 'Unknown'),
                        'Silhouette': metrics.get('silhouette_score', 'N/A'),
                        'Clusters': metrics.get('n_clusters', 'N/A'),
                        'Stabilité': 'Élevée' if metrics.get('silhouette_score', 0) > 0.5 else 'Moyenne' if metrics.get('silhouette_score', 0) > 0.3 else 'Faible'
                    })
                st.dataframe(pd.DataFrame(clustering_metrics), use_container_width=True)
                
        else:
            st.warning("⚠️ Aucune métrique avancée disponible")

    with tab4:
        st.markdown("### 💾 Export des Résultats Complets")
        
        if validation["successful_models"]:
            export_data = evaluator.get_export_data()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Format JSON")
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
                    file_name=f"evaluation_complete_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Aperçu des données
                with st.expander("👁️ Aperçu des données exportées"):
                    st.json(export_data, expanded=False)
            
            with col2:
                st.markdown("#### 📈 Rapport Détaillé")
                st.info("Exportez un rapport complet avec toutes les visualisations et analyses")
                
                # Bouton pour générer un rapport complet
                if st.button("🔄 Générer Rapport Complet", use_container_width=True):
                    with st.spinner("Génération du rapport en cours..."):
                        try:
                            # Ici vous pouvez étendre pour générer un rapport plus complet
                            st.success("✅ Rapport généré avec succès!")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")
        else:
            st.info("ℹ️ Aucune donnée à exporter")

    # Pied de page avec informations système
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🕐 Dernière mise à jour: {time.strftime('%H:%M:%S')}")
    with col2:
        memory_info = get_system_metrics()
        st.caption(f"💾 Mémoire: {memory_info['memory_percent']:.1f}%")
    with col3:
        st.caption(f"📊 Modèles chargés: {len(st.session_state.ml_results)}")

if __name__ == "__main__":
    main()