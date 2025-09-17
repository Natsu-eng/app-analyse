import json
from typing import List, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio

# Updated imports
from utils import (
    load_data,
    summarize_missing_values,
    handle_missing_values,
    auto_detect_variable_types,
    get_task_type,
)
from plots import (
    barplot_counts,
    pie_proportions,
    histogram_with_density,
    boxplot,
    qqplot,
    scatterplot,
    boxplot_bivariate,
    plot_predictions_vs_actual,
    plot_residuals_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from analyses import (
    univariate_stats,
    correlations,
    contingency_and_tests,
    group_numeric_by_category,
    tests_cat_vs_num,
)
from models import MODEL_CATALOG, train_and_evaluate

st.set_page_config(page_title="Data Science App", layout="wide")
pio.templates.default = "plotly_white"

# ===== Header =====
st.markdown('''
<style>
.kpi-card {background: #FFFFFF; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb;}
.kpi-title {font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em;}
.kpi-value {font-size: 22px; font-weight: 700; color: #111827;}
.section-title {margin-top: 0;}
</style>
''', unsafe_allow_html=True)

st.title("Data Science")
st.caption("Import, EDA, Modélisation et Prédictions — propulsé par Streamlit & Plotly")

# --- Sidebar for data import and cleaning ---
with st.sidebar:
    st.header("1. Import des données")
    upload = st.file_uploader("Fichier (CSV, Excel, Parquet, JSON)", type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"])
    st.divider()
    st.header("2. Valeurs manquantes")
    missing_strategy = st.selectbox("Stratégie", [
        "aucun", "supprimer_lignes", "remplir_moyenne", "remplir_mediane", "remplir_mode", "remplir_constante"
    ])
    fill_value = None
    if missing_strategy == "remplir_constante":
        fill_value = st.number_input("Valeur constante", value=0.0)
    st.divider()

if upload is None:
    st.info("Importez un fichier pour démarrer l’analyse.")
    st.stop()

try:
    df_raw = load_data(upload.getvalue(), upload.name)
except Exception as e:
    st.error(f"Erreur de chargement: {e}")
    st.stop()

# --- Data processing ---
df = handle_missing_values(df_raw, strategy=missing_strategy, fill_constant=fill_value, columns=list(df_raw.columns))
qual_cols, quant_cols = auto_detect_variable_types(df)

# --- Main app layout ---
# KPI row
missing_summary = summarize_missing_values(df_raw)
nb_rows, nb_cols = df_raw.shape
nb_missing_cells = int(missing_summary["nb_nan"].sum())
memory_mb = float(df_raw.memory_usage(deep=True).sum()) / (1024 ** 2)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Lignes</div><div class="kpi-value">{nb_rows:,}</div></div>''', unsafe_allow_html=True)
with c2:
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Colonnes</div><div class="kpi-value">{nb_cols:,}</div></div>''', unsafe_allow_html=True)
with c3:
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Cellules manquantes</div><div class="kpi-value">{nb_missing_cells:,}</div></div>''', unsafe_allow_html=True)
with c4:
    st.markdown(f'''<div class="kpi-card"><div class="kpi-title">Mémoire</div><div class="kpi-value">{memory_mb:.2f} Mo</div></div>''', unsafe_allow_html=True)

# Tabs with updated names
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Aperçu", "Analyse Univariée", "Analyse Bivariée", "Machine Learning", "Prédictions"
])

with tab1:
    st.subheader("Aperçu des données")
    st.write("Aperçu (100 premières lignes):")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("Valeurs manquantes (par colonne):")
    st.dataframe(summarize_missing_values(df), use_container_width=True)

with tab2:
    st.subheader("Analyse univariée")
    col = st.selectbox("Variable", options=list(df.columns), key="univar_col")
    if col:
        if col in quant_cols:
            stats_dict = univariate_stats(df, col)
            mc1, mc2 = st.columns([1,1])
            with mc1:
                st.metric("Moyenne", f"{stats_dict.get('moyenne', np.nan):.3f}")
                st.metric("Écart-type", f"{stats_dict.get('ecart_type', np.nan):.3f}")
            with mc2:
                st.metric("Médiane", f"{stats_dict.get('mediane', np.nan):.3f}")
                st.metric("Skewness", f"{stats_dict.get('skewness', np.nan):.3f}")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(histogram_with_density(df, col), use_container_width=True)
                st.plotly_chart(boxplot(df, col), use_container_width=True)
            with c2:
                st.plotly_chart(qqplot(df, col), use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(barplot_counts(df, col), use_container_width=True)
            with c2:
                st.plotly_chart(pie_proportions(df, col), use_container_width=True)

with tab3:
    st.subheader("Analyses bivariées")
    left, right = st.columns(2)
    with left:
        x = st.selectbox("Variable X", options=list(df.columns), index=0, key="bivar_x")
    with right:
        y = st.selectbox("Variable Y", options=list(df.columns), index=1 if len(df.columns) > 1 else 0, key="bivar_y")
    if x and y and x != y:
        if x in quant_cols and y in quant_cols:
            st.plotly_chart(scatterplot(df, x, y), use_container_width=True)
            st.expander("Coefficients de corrélation").json(correlations(df, x, y))
        elif x in qual_cols and y in qual_cols:
            res = contingency_and_tests(df, x, y)
            st.expander("Tableau et tests").json(res)
            st.dataframe(pd.crosstab(df[x], df[y]), use_container_width=True)
        else:
            cat = x if x in qual_cols else y
            num = y if x in qual_cols else x
            st.plotly_chart(boxplot_bivariate(df, cat, num), use_container_width=True)
            st.dataframe(group_numeric_by_category(df, cat, num), use_container_width=True)
            st.expander("Tests").json(tests_cat_vs_num(df, cat, num))

# New Machine Learning Tab
with tab4:
    st.subheader("Entraînement de modèles de Machine Learning")

    st.markdown("#### 1. Configuration de la tâche")
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Sélectionnez la variable cible (Y)", options=df.columns, key="ml_target")
    
    if target_col:
        task_type, unique_values = get_task_type(df[target_col])
        st.info(f"**Tâche détectée :** `{task_type.upper()}`" + (f" ({len(unique_values)} classes)" if task_type == "classification" else ""))

        with col2:
            feature_cols = st.multiselect(
                "Sélectionnez les variables explicatives (X)",
                options=[c for c in df.columns if c != target_col],
                default=[c for c in df.columns if c != target_col],
                key="ml_features"
            )

        st.markdown("#### 2. Configuration du modèle")
        col1, col2 = st.columns([1, 2])
        with col1:
            model_name = st.selectbox("Choisissez un modèle", options=list(MODEL_CATALOG[task_type].keys()), key="ml_model")
            test_size = st.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05, key="ml_test_size")

        st.sidebar.header("3. Hyperparamètres du modèle")
        params_config = MODEL_CATALOG[task_type][model_name]["params"]
        model_params = {}
        for name, config in params_config.items():
            if config.get("type") == "static":
                model_params[name] = config["value"]
                continue
            
            key = f"ml_{model_name}_{name}"
            if config["type"] == "int":
                model_params[name] = st.sidebar.number_input(
                    name, min_value=config["min"], max_value=config["max"], value=config["default"], step=config.get("step", 1), key=key
                )
            elif config["type"] == "float":
                model_params[name] = st.sidebar.number_input(
                    name, min_value=config["min"], max_value=config["max"], value=config["default"], step=config.get("step", 0.01), format="%.3f", key=key
                )
            elif config["type"] == "select":
                model_params[name] = st.sidebar.selectbox(
                    name, options=config["options"], index=config["options"].index(config["default"]), key=key
                )

        if st.button("🚀 Entraîner le modèle", use_container_width=True, key="ml_train_button"):
            if not feature_cols:
                st.warning("Veuillez sélectionner au moins une variable explicative.")
            else:
                with st.spinner("Entraînement en cours..."):
                    try:
                        results = train_and_evaluate(
                            df=df,
                            target_column=target_col,
                            feature_columns=feature_cols,
                            model_name=model_name,
                            task_type=task_type,
                            test_size=test_size,
                            model_params=model_params,
                        )
                        st.session_state['last_model_results'] = results # Save for prediction tab

                        st.success("Entraînement terminé !")
                        st.markdown("#### Métriques de performance")
                        metrics_df = pd.DataFrame([results["metrics"]]).T
                        metrics_df.columns = ["Valeur"]
                        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

                        st.markdown("#### Visualisations")
                        if task_type == "regression":
                            fig1 = plot_predictions_vs_actual(results["y_test"], results["y_pred"])
                            st.plotly_chart(fig1, use_container_width=True)
                            fig2 = plot_residuals_distribution(results["y_test"], results["y_pred"])
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            labels = sorted(list(unique_values))
                            fig1 = plot_confusion_matrix(results["y_test"], results["y_pred"], labels)
                            st.plotly_chart(fig1, use_container_width=True)
                            fig2 = plot_roc_curve(results["y_test"], results["y_proba"], labels)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        fig3 = plot_feature_importance(results["pipeline"], results["feature_names"])
                        st.plotly_chart(fig3, use_container_width=True)

                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")
                        st.exception(e)

# Prediction Tab (updated to use the new model)
with tab5:
    st.subheader("Prédictions sur de nouvelles données")
    if 'last_model_results' not in st.session_state:
        st.info("Ajustez un modèle dans l'onglet 'Machine Learning' pour activer les prédictions.")
    else:
        model_results = st.session_state['last_model_results']
        pipeline = model_results['pipeline']
        
        st.write("Le dernier modèle entraîné est prêt pour les prédictions.")
        
        upload_new = st.file_uploader("Nouvelles données", type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"], key="newdata")
        if upload_new is not None:
            try:
                df_new = load_data(upload_new.getvalue(), upload_new.name)
                
                # Ensure all feature columns are present
                feature_cols = model_results['pipeline'].feature_names_in_
                if not all(c in df_new.columns for c in feature_cols):
                    st.error(f"Les données de prédiction doivent contenir les colonnes : {list(feature_cols)}")
                else:
                    with st.spinner("Prédiction en cours..."):
                        predictions = pipeline.predict(df_new[feature_cols])
                        res = df_new.copy()
                        res["prediction"] = predictions
                        st.dataframe(res, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur de prédiction: {e}")
                st.exception(e)