# E:\gemini\app-analyse\pages\3_🤖_Modélisation.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from components.reporting import display_results
from utils.report_generator import generate_pdf_report

st.set_page_config(layout="wide")
st.title("🤖 Modélisation")

if 'ml_results' not in st.session_state or not st.session_state.ml_results:
    st.info("Aucun modèle n'a encore été entraîné.")
    st.stop()

all_models_results = st.session_state.ml_results

tab_overview, tab_details = st.tabs(["Vue d'ensemble & Comparaison", "Détail par Modèle & Téléchargement"])

with tab_overview:
    st.header("Comparaison des Performances des Modèles")
    comparison_data = []
    for res in all_models_results:
        metrics = res.get("metrics_summary", {}).get("global_metrics", {})
        if not metrics:
            continue
        row = {"Modèle": res.get("model_name", "Inconnu")}
        task_type = res.get("config", {}).get("task_type")
        
        if task_type == "classification":
            # Ajouter plusieurs métriques clés
            row["Accuracy"] = f"{metrics.get('accuracy', 0):.4f}"
            row["ROC AUC"] = f"{metrics.get('roc_auc', 0):.4f}"
            row["Precision"] = f"{metrics.get('precision', 0):.4f}" if "precision" in metrics else "N/A"
            row["Recall"] = f"{metrics.get('recall', 0):.4f}" if "recall" in metrics else "N/A"
            row["F1-Score"] = f"{metrics.get('f1_score', 0):.4f}" if "f1_score" in metrics else "N/A"
        
        elif task_type == "regression":
            row["R² Score"] = f"{metrics.get('r2', 0):.4f}"
            row["RMSE"] = f"{metrics.get('rmse', 0):.4f}"
            row["MAE"] = f"{metrics.get('mae', 0):.4f}"
            row["MSE"] = f"{metrics.get('mse', 0):.4f}"
        
        elif task_type == "unsupervised":
            row["Silhouette Score"] = f"{metrics.get('silhouette_score', 0):.4f}"
            row["Davies-Bouldin"] = f"{metrics.get('davies_bouldin_score', 0):.4f}"
            row["Calinski-Harabasz"] = f"{metrics.get('calinski_harabasz_score', 0):.2f}"
            row["Nombre de Clusters"] = f"{metrics.get('n_clusters', 'N/A')}"
        
        comparison_data.append(row)
    
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data).set_index("Modèle"))


with tab_details:
    st.header("Analyse Détaillée et Export")
    for model_result in all_models_results:
        model_name = model_result.get("model_name", "Modèle Inconnu")
        with st.expander(f"Résultats pour : {model_name}", expanded=True):
            display_results(
                metrics_summary=model_result.get("metrics_summary"),
                task_type=model_result.get("config", {}).get("task_type"),
                X_test_processed=model_result.get("X_test_processed"),
                y_test=model_result.get("y_test"),
                y_pred=model_result.get("y_pred"),
                model=model_result.get("model"),
                label_encoder=model_result.get("label_encoder"),
                X_test_raw=model_result.get("X_test_raw")
            )

            st.subheader("Téléchargements")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    model_pipeline = model_result.get("model")
                    if model_pipeline:
                        model_buffer = io.BytesIO()
                        joblib.dump(model_pipeline, model_buffer)
                        model_buffer.seek(0)
                        st.download_button(
                            label="📥 Télécharger le Modèle (.joblib)",
                            data=model_buffer,
                            file_name=f"{model_name}_pipeline.joblib",
                            mime="application/octet-stream",
                            key=f"download_model_{model_name}"
                        )
                except Exception as e:
                    st.error(f"Erreur lors de la sérialisation du modèle : {e}")

            with col2:
                try:
                    with st.spinner("Génération du rapport PDF..."):
                        pdf_bytes = generate_pdf_report(model_result)
                    st.download_button(
                        label="📄 Télécharger le Rapport (.pdf)",
                        data=pdf_bytes,
                        file_name=f"{model_name}_report.pdf",
                        mime="application/pdf",
                        key=f"download_report_{model_name}"
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport PDF : {e}")
