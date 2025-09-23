

import streamlit as st
import pandas as pd
import numpy as np
import logging
from plots.model_evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)
from st_aggrid import AgGrid, GridOptionsBuilder

def show_evaluation():
    """Affiche la section d'évaluation des modèles de manière robuste."""
    st.header("📈 Évaluation du Modèle")
    
    if not st.session_state.get('ml_results'):
        st.info("Veuillez lancer l'entraînement des modèles pour voir les résultats.")
        return

    results = st.session_state.ml_results
    
    # --- Tableau de comparaison ---
    with st.container(border=True):
        st.subheader("🏆 Comparaison des Modèles")
        try:
            results_list = []
            for r in results:
                res_dict = {"Modèle": r["model_name"]}
                for k, v in r["metrics"].items():
                    if isinstance(v, (int, float, np.number)):
                        res_dict[k] = round(v, 4)
                    elif isinstance(v, str) and "\n" not in v:
                        res_dict[k] = v
                results_list.append(res_dict)
            
            if not results_list:
                st.warning("Aucune métrique à afficher.")
                return

            results_df = pd.DataFrame(results_list).set_index("Modèle")
            AgGrid(results_df.reset_index(), fit_columns_on_load=True, key="eval_grid")
        except Exception as e:
            st.error(f"Erreur lors de l'affichage du tableau de comparaison : {e}")

    st.markdown("---")
    
    # --- Inspection détaillée ---
    st.subheader("🔍 Inspection Détaillée")
    model_names = [r["model_name"] for r in results]
    if not model_names:
        return
        
    selected_model_name = st.selectbox("Choisissez un modèle à inspecter", model_names, key="model_inspect_select")
    selected_result = next((r for r in results if r["model_name"] == selected_model_name), None)
    
    if not selected_result:
        st.error("Modèle sélectionné non trouvé.")
        return

    with st.container(border=True):
        tab_metrics, tab_plots = st.tabs(["Métrique & Notes", "Graphiques de Performance"])
        
        with tab_metrics:
            st.subheader(f"🎯 Performance de {selected_model_name}")
            
            if selected_result.get("notes"):
                with st.expander("📝 Notes d'évaluation du pipeline", expanded=True):
                    for note in selected_result["notes"]:
                        st.info(note)
            
            metrics = selected_result.get("metrics", {})
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                st.markdown("**Scores Clés**")
                for key, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        st.metric(label=key, value=f"{value:.4f}")
                    elif isinstance(value, str) and len(value) < 50:
                        st.metric(label=key, value=value)
            
            with col2:
                if "classification_report" in metrics:
                    st.text_area("Rapport de Classification", metrics["classification_report"], height=250)

        with tab_plots:
            st.subheader("📊 Graphiques de Performance")
            le = selected_result.get('label_encoder')
            y_test_encoded = selected_result.get('y_test_encoded')
            y_pred_encoded = selected_result.get('y_pred_encoded')
            y_proba = selected_result.get("y_proba")

            # Vérifier que les données nécessaires sont présentes et que le LabelEncoder est disponible
            if y_test_encoded is not None and y_pred_encoded is not None and le is not None:
                # Décoder les labels pour un affichage clair
                y_test_decoded = le.inverse_transform(y_test_encoded)
                
                # Gérer les labels prédits non vus par l'encodeur
                label_map = {i: label for i, label in enumerate(le.classes_)}
                y_pred_decoded = np.array([label_map.get(int(pred), "CLASSE_INCONNUE") for pred in y_pred_encoded])

                st.plotly_chart(plot_confusion_matrix(y_test_decoded, y_pred_decoded, le.classes_), use_container_width=True)
            else:
                st.info("Matrice de confusion non disponible (données de test/prédictions manquantes ou LabelEncoder non trouvé).")

            if y_test_encoded is not None and y_proba is not None and le is not None and selected_result["metrics"].get("roc_auc") != "N/A":
                st.plotly_chart(plot_roc_curve(y_test_encoded, y_proba, le.classes_), use_container_width=True)
                st.plotly_chart(plot_precision_recall_curve(y_test_encoded, y_proba, le.classes_), use_container_width=True)
            else:
                st.info("Les courbes ROC et Précision/Rappel ne sont pas disponibles (données de test/probabilités manquantes ou AUC non calculable).")
