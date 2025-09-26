import numpy as np
import streamlit as st
import pandas as pd
import time
import base64
import json
import gc
import plotly.graph_objects as go
from ml.evaluation.visualization import ModelEvaluationVisualizer
from ml.evaluation.metrics_calculation import get_system_metrics

from utils.report_generator import generate_pdf_report 

# Configuration de la page
st.set_page_config(
    page_title="Évaluation des Modèles",
    page_icon="📈",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .best-model {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-left-color: #2196f3;
    }
    .performance-high { color: #4caf50; font-weight: bold; }
    .performance-medium { color: #ff9800; font-weight: bold; }
    .performance-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def create_download_link(data, filename: str, file_type: str = "csv"):
    """Crée un lien de téléchargement"""
    try:
        if file_type == "csv" and hasattr(data, 'to_csv'):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Télécharger CSV</a>'
        elif file_type == "json":
            json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            return f'<a href="data:application/json;base64,{b64}" download="{filename}.json">📥 Télécharger JSON</a>'
    except Exception as e:
        st.error(f"Erreur création lien: {e}")
    return ""

def get_performance_class(value: float, metric_type: str) -> str:
    """Détermine la classe de performance pour le styling"""
    try:
        if metric_type in ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'r2']:
            if value >= 0.8: return "performance-high"
            elif value >= 0.6: return "performance-medium"
            else: return "performance-low"
        elif metric_type in ['mae', 'mse', 'rmse']:
            if value <= 0.1: return "performance-high"
            elif value <= 0.3: return "performance-medium"
            else: return "performance-low"
    except: pass
    return ""

def generate_model_pdf_report(model_result: dict) -> bytes:
    """Génère un rapport PDF pour un modèle spécifique"""
    try:
        return generate_pdf_report(model_result)
    except Exception as e:
        st.error(f"Erreur génération PDF: {e}")
        return None

def main():
    st.title("📈 Évaluation Détaillée des Modèles")
    
    # Vérification des données
    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        st.error("❌ Aucun résultat d'expérimentation disponible")
        st.info("Veuillez d'abord entraîner des modèles dans la page Configuration ML")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Retour Accueil"):
                st.switch_page("app.py")
        with col2:
            if st.button("⚙️ Configuration ML"):
                st.switch_page("pages/2_⚙️_Configuration_ML.py")
        return
    
    # Initialisation de l'évaluateur
    evaluator = ModelEvaluationVisualizer(st.session_state.ml_results)
    validation = evaluator.validation_result
    
    if not validation["has_results"]:
        st.error("❌ Données d'évaluation invalides")
        return
    
    # En-tête avec métriques
    successful_count = len(validation["successful_models"])
    total_count = validation["results_count"]
    
    st.success(f"✅ Évaluation terminée - {successful_count}/{total_count} modèles réussis")
    
    # Métriques rapides
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modèles Réussis", successful_count)
    with col2:
        st.metric("Modèles Échoués", len(validation["failed_models"]))
    with col3:
        if validation["best_model"]:
            st.metric("Meilleur Modèle", validation["best_model"])
    with col4:
        system_metrics = get_system_metrics()
        memory_pct = system_metrics['memory_percent']
        st.metric("RAM Utilisée", f"{memory_pct:.1f}%")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Informations")
        st.write(f"**Type de tâche:** {validation['task_type'] or 'Inconnu'}")
        if validation["best_model"]:
            st.write(f"**Meilleur modèle:** {validation['best_model']}")
        
        st.header("🔧 Actions")
        if st.button("🔄 Actualiser"):
            st.rerun()
        
        if st.button("🧹 Nettoyer Mémoire"):
            gc.collect()
            st.success("Mémoire nettoyée")
        
        # Export rapide
        if validation["successful_models"]:
            st.header("📥 Export Rapide")
            
            try:
                export_data = []
                for result in validation["successful_models"]:
                    metrics = evaluator._safe_get(result, ['metrics'], {})
                    row = {
                        'modele': evaluator._safe_get(result, ['model_name'], 'Unknown'),
                        'temps_entrainement': evaluator._safe_get(result, ['training_time'], 0)
                    }
                    
                    if validation["task_type"] == 'classification':
                        row.update({
                            'accuracy': evaluator._safe_get(metrics, ['accuracy'], 0),
                            'f1_score': evaluator._safe_get(metrics, ['f1_score'], 0)
                        })
                    elif validation["task_type"] == 'regression':
                        row.update({
                            'r2': evaluator._safe_get(metrics, ['r2'], 0),
                            'mae': evaluator._safe_get(metrics, ['mae'], 0)
                        })
                    
                    export_data.append(row)
                
                if export_data:
                    df_export = pd.DataFrame(export_data)
                    csv_data = df_export.to_csv(index=False)
                    
                    # CSV Rapide
                    st.download_button(
                        label="📄 CSV Résumé",
                        data=csv_data,
                        file_name=f"resultats_{validation['task_type']}_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # NOUVEAU : PDF Rapide du meilleur modèle
                    if validation["best_model"]:
                        st.markdown("---")
                        st.subheader("📊 PDF Rapide")
                        
                        best_model_result = next(
                            (r for r in validation["successful_models"] 
                            if evaluator._safe_get(r, ['model_name'], '') == validation["best_model"]), 
                            None
                        )
                        
                        if best_model_result:
                            try:
                                pdf_bytes = generate_pdf_report(best_model_result)
                                if pdf_bytes:
                                    st.download_button(
                                        label="🎯 PDF Meilleur Modèle",
                                        data=pdf_bytes,
                                        file_name=f"rapport_{validation['best_model']}_{int(time.time())}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            except Exception as e:
                                st.error(f"Erreur PDF: {e}")
                    
            except Exception as e:
                st.error(f"❌ Erreur export: {e}")
        
    
    # Onglets principaux
    tab_comparison, tab_individual, tab_analysis, tab_explain, tab_learning, tab_export = st.tabs([
    "📊 Vue Globale", 
    "🔍 Analyse Détaillée", 
    "📈 Métriques Avancées",
    "🧩 Explicabilité",
    "📉 Courbes d'Apprentissage",
    "💾 Export Complet"
    ])
    
    with tab_comparison:
        st.header("Comparaison des Modèles")
        
        if validation["successful_models"]:
            # Graphique comparatif
            fig = evaluator.get_comparison_plot()
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau comparatif
            st.subheader("Tableau Comparatif Détaillé")
            df_comparison = evaluator.get_comparison_dataframe()
            st.dataframe(df_comparison, use_container_width=True, height=400)
            
            # Analyse rapide
            if successful_count > 0:
                st.subheader("📈 Analyse Rapide")
                success_rate = (successful_count / total_count) * 100
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Taux de Succès", f"{success_rate:.1f}%")
                
                with col2:
                    avg_time = np.mean([
                        evaluator._safe_get(r, ['training_time'], 0) 
                        for r in validation["successful_models"]
                    ])
                    st.metric("Temps Moyen", f"{avg_time:.1f}s")
                
                with col3:
                    if validation["best_model"]:
                        st.metric("Champion", validation["best_model"])
        else:
            st.warning("Aucun modèle réussi à comparer")
    
    with tab_individual:
        st.header("Analyse Détaillée par Modèle")
        
        if validation["successful_models"]:
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(st.session_state.ml_results)]
            
            # Sélecteur de modèle
            selected_idx = st.selectbox(
                "Sélectionnez un modèle",
                range(len(model_names)),
                format_func=lambda x: f"{model_names[x]} {'🏆' if model_names[x] == validation.get('best_model') else ''}",
                key="model_selector"
            )
            
            selected_result = st.session_state.ml_results[selected_idx]
            model_name = model_names[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Performances de {model_name}")
                metrics = evaluator._safe_get(selected_result, ['metrics'], {})
                has_error = evaluator._safe_get(metrics, ['error']) is not None
                
                if has_error:
                    st.error(f"❌ Erreur: {evaluator._safe_get(metrics, ['error'], 'Erreur inconnue')}")
                else:
                    if validation["task_type"] == 'classification':
                        metric_cols = st.columns(4)
                        metric_values = [
                            ('Accuracy', evaluator._safe_get(metrics, ['accuracy'], 0)),
                            ('F1-Score', evaluator._safe_get(metrics, ['f1_score'], 0)),
                            ('Précision', evaluator._safe_get(metrics, ['precision'], 0)),
                            ('Rappel', evaluator._safe_get(metrics, ['recall'], 0))
                        ]
                        
                        for i, (name, value) in enumerate(metric_values):
                            with metric_cols[i]:
                                formatted_value = evaluator._format_metric_value(value, name.lower())
                                perf_class = get_performance_class(value if isinstance(value, (int, float)) else 0, name.lower())
                                st.metric(name, formatted_value)
                    
                    elif validation["task_type"] == 'regression':
                        metric_cols = st.columns(4)
                        metric_values = [
                            ('R²', evaluator._safe_get(metrics, ['r2'], 0)),
                            ('MAE', evaluator._safe_get(metrics, ['mae'], 0)),
                            ('RMSE', evaluator._safe_get(metrics, ['rmse'], 0)),
                            ('MSE', evaluator._safe_get(metrics, ['mse'], 0))
                        ]
                        
                        for i, (name, value) in enumerate(metric_values):
                            with metric_cols[i]:
                                formatted_value = evaluator._format_metric_value(value, name.lower())
                                st.metric(name, formatted_value)
            
            with col2:
                st.subheader("ℹ️ Informations")
                training_time = evaluator._safe_get(selected_result, ['training_time'], 0)
                best_params = evaluator._safe_get(selected_result, ['best_params'], {})
                
                st.write(f"**Nom:** {model_name}")
                st.write(f"**Statut:** {'✅ Succès' if not has_error else '❌ Échec'}")
                st.write(f"**Temps:** {training_time:.2f}s")
                st.write(f"**Paramètres:** {len(best_params)} optimisés")
                
                if model_name == validation.get('best_model'):
                    st.success("🏆 **Meilleur modèle**")
        
        else:
            st.warning("Aucun modèle disponible pour l'analyse")
    
    with tab_analysis:
        st.header("Métriques Avancées")
        
        if validation["successful_models"]:
            # Distribution des performances
            if validation["task_type"] == 'classification':
                accuracies = [evaluator._safe_get(r, ['metrics', 'accuracy'], 0) 
                            for r in validation["successful_models"]]
                
                if accuracies:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=accuracies, nbinsx=10, name='Accuracy'))
                    fig.update_layout(title="Distribution des Scores d'Accuracy")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des temps
            training_times = [evaluator._safe_get(r, ['training_time'], 0) 
                            for r in validation["successful_models"]]
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(validation["successful_models"])]
            
            if training_times:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=model_names, y=training_times, name='Temps (s)'))
                fig.update_layout(title="Temps d'Entraînement par Modèle")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Aucune métrique avancée disponible")

    with tab_explain:
        st.header("🧩 Explicabilité des Modèles")
        
        if validation["successful_models"]:
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                        for i, r in enumerate(validation["successful_models"])]
            
            selected_model = st.selectbox("Sélectionnez un modèle pour l'explication", model_names)
            selected_result = next((r for r in validation["successful_models"] 
                                if evaluator._safe_get(r, ['model_name'], '') == selected_model), None)
            
            if selected_result:
                st.subheader(f"SHAP - Importance des Features ({selected_model})")
                
                try:
                    model = evaluator._safe_get(selected_result, ['fitted_model'])
                    X_sample = evaluator._safe_get(selected_result, ['X_sample'])  # petit subset stocké pendant training
                    feature_names = evaluator._safe_get(selected_result, ['feature_names'], [])
                    
                    if model is not None and X_sample is not None:
                        fig = evaluator.create_shap_plot(model, X_sample, feature_names)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Données nécessaires pour SHAP non disponibles")
                except Exception as e:
                    st.error(f"Erreur SHAP: {e}")
                
                st.subheader("Importance des Features (classique)")
                try:
                    model = evaluator._safe_get(selected_result, ['fitted_model'])
                    feature_names = evaluator._safe_get(selected_result, ['feature_names'], [])
                    if model is not None and feature_names:
                        fig_imp = evaluator.create_feature_importance_plot(model, feature_names)
                        st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur importance features: {e}")
        else:
            st.warning("Aucun modèle disponible pour l'explication")

    with tab_learning:
        st.header("📉 Courbes d'Apprentissage - Détection Overfitting/Underfitting")
        
        if validation["successful_models"]:
            model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                        for i, r in enumerate(validation["successful_models"])]
            
            selected_model = st.selectbox("Sélectionnez un modèle", model_names, key="learning_selector")
            selected_result = next((r for r in validation["successful_models"] 
                                if evaluator._safe_get(r, ['model_name'], '') == selected_model), None)
            
            if selected_result:
                try:
                    model = evaluator._safe_get(selected_result, ['fitted_model'])
                    X_train = evaluator._safe_get(selected_result, ['X_train'])
                    y_train = evaluator._safe_get(selected_result, ['y_train'])
                    if model is not None and X_train is not None and y_train is not None:
                        fig = evaluator.create_learning_curve_plot(model, X_train, y_train)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Données nécessaires pour courbes d'apprentissage non disponibles")
                except Exception as e:
                    st.error(f"Erreur courbes d'apprentissage: {e}")
        else:
            st.warning("Aucun modèle disponible pour analyse des courbes d'apprentissage")
    
    with tab_export:
        st.header("Export Complet des Résultats")
        
        if validation["successful_models"]:
            # Section Export CSV/JSON
            col1, col2 = st.columns(2)
            
            with col1:
                # Export CSV
                df_export = evaluator.get_comparison_dataframe()
                st.download_button(
                    label="📄 Exporter en CSV",
                    data=df_export.to_csv(index=False),
                    file_name=f"resultats_ml_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export JSON
                export_data = evaluator.get_export_data()
                st.download_button(
                    label="📊 Exporter en JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
                    file_name=f"resultats_ml_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # SECTION : RAPPORTS PDF
            st.markdown("---")
            st.subheader("📊 Rapports PDF Détaillés")
            
            # Sélecteur de modèle pour PDF
            pdf_model_names = [evaluator._safe_get(r, ['model_name'], f'Modèle_{i}') 
                             for i, r in enumerate(validation["successful_models"])]
            
            if pdf_model_names:
                selected_pdf_model = st.selectbox(
                    "Sélectionnez un modèle pour le rapport PDF détaillé",
                    pdf_model_names,
                    key="pdf_model_selector"
                )
                
                # Trouver le résultat correspondant
                selected_pdf_result = next(
                    (r for r in validation["successful_models"] 
                     if evaluator._safe_get(r, ['model_name'], '') == selected_pdf_model), 
                    None
                )
                
                col_pdf1, col_pdf2 = st.columns([3, 1])
                
                with col_pdf1:
                    if selected_pdf_result and st.button("📄 Générer le rapport PDF détaillé", 
                                                       use_container_width=True, 
                                                       key="generate_pdf"):
                        with st.spinner("Génération du rapport PDF en cours..."):
                            try:
                                pdf_bytes = generate_pdf_report(selected_pdf_result)
                                
                                if pdf_bytes:
                                    st.success("✅ Rapport PDF généré avec succès!")
                            except Exception as e:
                                st.error(f"Erreur génération PDF: {e}")
                                    
                with col_pdf2:
                    if selected_pdf_result:
                        try:
                            pdf_bytes = generate_pdf_report(selected_pdf_result)
                            if pdf_bytes:
                                st.download_button(
                                    label="📥 Télécharger PDF",
                                    data=pdf_bytes,
                                    file_name=f"rapport_{selected_pdf_model}_{int(time.time())}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Erreur génération PDF: {e}")
            
            # Rapport PDF global
            st.markdown("---")
            st.subheader("📚 Rapport PDF Global")
            
            if st.button("🌐 Générer rapport global PDF", use_container_width=True):
                with st.spinner("Génération du rapport global..."):
                    try:
                        # Générer PDF pour le meilleur modèle comme rapport global
                        if validation["best_model"]:
                            best_model_result = next(
                                (r for r in validation["successful_models"] 
                                 if evaluator._safe_get(r, ['model_name'], '') == validation["best_model"]), 
                                None
                            )
                            
                            if best_model_result:
                                pdf_bytes = generate_pdf_report(best_model_result)
                                
                                if pdf_bytes:
                                    st.success("✅ Rapport global PDF généré!")
                                    st.download_button(
                                        label="📥 Télécharger rapport global PDF",
                                        data=pdf_bytes,
                                        file_name=f"rapport_global_{validation['task_type']}_{int(time.time())}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur génération rapport global: {e}")
            
            # Aperçu des données (existant)
            st.markdown("---")
            st.subheader("📋 Aperçu des données exportées")
            st.dataframe(df_export, use_container_width=True)
            
            # Statistiques d'export
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.info(f"**Total modèles:** {len(export_data['models'])}")
                st.info(f"**Type de tâche:** {export_data['task_type']}")
            with col_stat2:
                st.info(f"**Meilleur modèle:** {export_data['best_model']}")
                st.info(f"**Date export:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(export_data['timestamp']))}")
        
        else:
            st.warning("❌ Aucune donnée à exporter")

if __name__ == "__main__":
    main()