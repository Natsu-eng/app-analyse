import streamlit as st
import pandas as pd
import numpy as np
import shap

# Import des fonctions de plotting centralisées
from plots.model_evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_elbow_method,
    plot_shap_summary
)


def display_results(metrics_summary, task_type="classification", X_test_processed=None, y_test=None, y_pred=None, model=None, label_encoder=None, X_test_raw=None):
    if not metrics_summary or "global_metrics" not in metrics_summary:
        st.warning("Aucun résultat d'évaluation à afficher.")
        return

    agg_metrics = metrics_summary["global_metrics"]
    if "error" in agg_metrics:
        st.error(f"Erreur lors de l'évaluation : {agg_metrics['error']}")
        return

    st.subheader("Métriques Globales")

    # --- Classification ---
    if task_type == "classification":
        st.write("#### Métriques de Classification")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{agg_metrics.get('accuracy', 0):.4f}")
        roc_auc_val = agg_metrics.get('roc_auc', 'N/A')
        col2.metric("ROC AUC", f"{roc_auc_val:.4f}" if isinstance(roc_auc_val, (int, float)) else roc_auc_val)
        st.markdown("**Accuracy**: Pourcentage global de prédictions correctes. **ROC AUC**: Capacité du modèle à distinguer les classes.")

        if "classification_report" in agg_metrics:
            st.write("##### Rapport de Classification Détaillé")
            # Assurer que le rapport est un dictionnaire avant de créer le DataFrame
            report_data = agg_metrics["classification_report"]
            if isinstance(report_data, dict):
                report_df = pd.DataFrame(report_data).transpose()
                st.dataframe(report_df)
                st.markdown("**Précision**: Sur toutes les prédictions positives, combien étaient correctes. **Rappel (Recall)**: Sur toutes les vraies valeurs positives, combien ont été trouvées. **F1-Score**: Moyenne harmonique de la précision et du rappel.")
            else:
                st.text(report_data) # Afficher comme texte brut si ce n'est pas un dict

        if y_test is not None and y_pred is not None:
            st.subheader("Visualisations de Classification")
            class_labels = label_encoder.classes_ if label_encoder is not None else np.unique(y_test)
            
            st.plotly_chart(plot_confusion_matrix(y_test, y_pred, class_labels=class_labels.tolist()), use_container_width=True)
            st.markdown("**Interprétation**: La matrice de confusion montre les performances du modèle. On souhaite maximiser les valeurs sur la diagonale (prédictions correctes) et minimiser les autres (erreurs).")

            if hasattr(model, 'predict_proba') and X_test_processed is not None:
                y_proba = model.predict_proba(X_test_processed)
                st.plotly_chart(plot_roc_curve(y_test, y_proba, class_labels=class_labels.tolist()), use_container_width=True)
                st.markdown("**Interprétation**: La courbe ROC illustre la capacité du modèle à distinguer les classes. Une courbe proche du coin supérieur gauche (AUC proche de 1) indique un excellent modèle.")

    # --- Regression ---
    elif task_type == "regression":
        st.write("#### Métriques de Régression")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R2 Score", f"{agg_metrics.get('r2', 0):.4f}")
        col2.metric("RMSE", f"{agg_metrics.get('rmse', 0):.4f}")
        col3.metric("MAE", f"{agg_metrics.get('mae', 0):.4f}")
        col4.metric("MSE", f"{agg_metrics.get('mse', 0):.4f}")
        st.markdown("**R²**: Part de la variance expliquée. **RMSE/MAE/MSE**: Erreurs de prédiction (plus bas = mieux).")
        
        if y_test is not None and y_pred is not None:
            st.subheader("Visualisations de Régression")
            st.plotly_chart(plot_predictions_vs_actual(y_test, y_pred), use_container_width=True)
            st.plotly_chart(plot_residuals(y_test, y_pred), use_container_width=True)

    # --- Unsupervised ---
    elif task_type == "unsupervised":
        st.write("#### Métriques de Clustering")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Clusters", f"{agg_metrics.get('n_clusters', 'N/A')}")
        col2.metric("Score de Silhouette", f"{agg_metrics.get('silhouette_score', 0):.4f}")
        col3.metric("Davies-Bouldin Score", f"{agg_metrics.get('davies_bouldin_score', 0):.4f}")
        col4.metric("Calinski-Harabasz Score", f"{agg_metrics.get('calinski_harabasz_score', 0):.2f}")
        st.markdown("**Silhouette**: Mesure la qualité de la séparation des clusters. **Davies-Bouldin**: Similarité des clusters. **Calinski-Harabasz**: Ratio de dispersion.")

        if "inertias_for_elbow" in agg_metrics:
            k_range = range(1, len(agg_metrics["inertias_for_elbow"]) + 1)
            st.plotly_chart(plot_elbow_method(agg_metrics["inertias_for_elbow"], k_range), use_container_width=True)

    # --- SHAP (tous modèles sauf non supervisé) ---
    if model is not None and X_test_processed is not None and task_type != 'unsupervised':
        st.subheader("Interprétabilité du Modèle (SHAP)")
        with st.spinner("Calcul des valeurs SHAP pour l'interprétabilité (peut être long)..."):
            try:
                # Accès robuste au dernier estimateur du pipeline
                model_to_explain = model.steps[-1][1]

                X_shap = X_test_processed.select_dtypes(include=np.number).copy()
                if X_shap.shape[1] == 0:
                    st.warning("Aucune colonne numérique exploitable pour SHAP. Graphiques ignorés.")
                    return

                # Création de l'explainer
                if hasattr(model_to_explain, 'predict_proba'):
                    explainer = shap.Explainer(model_to_explain, X_shap)
                else:
                    explainer = shap.Explainer(model_to_explain.predict, X_shap)

                shap_values = explainer(X_shap)

                st.write("##### Importance Globale des Features (SHAP Summary)")
                # Utilise la fonction de plotting centralisée
                plot_shap_summary(shap_values, X_shap)
                st.markdown("**Interprétation**: Ce graphique montre l'impact de chaque caractéristique sur la prédiction du modèle. Les features en haut sont les plus importantes.")

            except Exception as e:
                st.error(f"Erreur lors de la génération des graphiques SHAP : {e}")
