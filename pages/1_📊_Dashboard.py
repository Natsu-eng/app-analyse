
import streamlit as st
import pandas as pd
from utils.data_analysis import (
    get_data_profile, 
    auto_detect_column_types, 
    is_dask_dataframe, 
    compute_if_dask,
    sanitize_column_types_for_display
)
from plots.exploratory import (
    plot_overview_metrics,
    plot_missing_values_overview,
    plot_cardinality_overview,
    plot_distribution,
    plot_bivariate_analysis
)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard Exploratoire")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Veuillez d'abord charger un jeu de données depuis la page d'accueil.")
    st.page_link("app.py", label="Retour à l'accueil", icon="🏠")
    st.stop()

df = st.session_state.df

with st.spinner("Analyse du jeu de données..."):
    if st.session_state.get('column_types') is None:
        st.session_state.column_types = auto_detect_column_types(df)

column_types = st.session_state.column_types

st.header("Vue d'ensemble du jeu de données")

n_rows = compute_if_dask(df.shape[0])
n_cols = df.shape[1]
total_missing = compute_if_dask(df.isna().sum().sum())
missing_percentage = (total_missing / (n_rows * n_cols)) * 100 if (n_rows * n_cols) > 0 else 0
duplicate_rows = compute_if_dask(df.duplicated().sum())
memory_usage = compute_if_dask(df.memory_usage(deep=True).sum()) / (1024**2)

overview_metrics = {
    'n_rows': n_rows,
    'n_cols': n_cols,
    'missing_percentage': missing_percentage,
    'duplicate_rows': duplicate_rows,
    'memory_usage': memory_usage
}
st.plotly_chart(plot_overview_metrics(overview_metrics), use_container_width=True)

tab_overview, tab_univariate, tab_bivariate, tab_preview = st.tabs(
    ["📈 Qualité des Données", "🔬 Analyse par Variable", "🔗 Relations Bivariées", "📄 Aperçu des Données Brutes"]
)

with tab_overview:
    st.subheader("Valeurs Manquantes et Cardinalité")
    col1, col2 = st.columns(2)
    with col1:
        missing_fig = plot_missing_values_overview(df)
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
    with col2:
        if 'text' not in column_types:
            column_types['text'] = []
        cardinality_fig = plot_cardinality_overview(df, column_types)
        if cardinality_fig:
            st.plotly_chart(cardinality_fig, use_container_width=True)

with tab_univariate:
    st.subheader("Analyse Approfondie d'une Variable")
    selected_col = st.selectbox("Choisissez une variable à analyser", df.columns, key="univariate_selected_col")
    if selected_col:
        col_type = 'numeric' if selected_col in column_types['numeric'] else 'categorical'
        sample_df = df
        if is_dask_dataframe(df) or len(df) > 50000:
            sample_df = df.sample(frac=0.1).head(50000)
        sample_df = compute_if_dask(sample_df)
        if col_type == 'numeric':
            st.plotly_chart(plot_distribution(sample_df[selected_col], selected_col), use_container_width=True)
        else:
            st.write(f"Analyse de la variable catégorielle : **{selected_col}**")
            value_counts_df = sample_df[selected_col].value_counts().reset_index()
            value_counts_df.columns = ['Catégorie', 'Comptage']
            st.dataframe(value_counts_df)

with tab_bivariate:
    st.subheader("Analyse des Relations entre Deux Variables")
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable 1", df.columns, key="bivariate_var1_select")
    with col2:
        var2 = st.selectbox("Variable 2", df.columns, index=min(1, len(df.columns)-1), key="bivariate_var2_select")
    if var1 and var2 and var1 != var2:
        type1 = 'numeric' if var1 in column_types['numeric'] else 'categorical'
        type2 = 'numeric' if var2 in column_types['numeric'] else 'categorical'
        sample_df = df
        if is_dask_dataframe(df) or len(df) > 20000:
             sample_df = df.sample(frac=0.1).head(20000)
        sample_df = compute_if_dask(sample_df)
        biv_fig = plot_bivariate_analysis(sample_df, var1, var2, type1, type2)
        if biv_fig:
            st.plotly_chart(biv_fig, use_container_width=True)
    elif var1 == var2:
        st.warning("Veuillez sélectionner deux variables différentes.")

with tab_preview:
    st.subheader("Aperçu des Données Brutes")
    raw_df_sample = compute_if_dask(st.session_state.df.head(500))

    try:
        st.info("Tentative d'affichage des données brutes. Peut échouer si les types de données sont trop hétérogènes.")
        st.dataframe(raw_df_sample)
    except Exception as e:
        st.warning(
            f"""L'affichage direct des données brutes a échoué. C'est probablement dû à des types de données mixtes dans une ou plusieurs colonnes.
            (Erreur : `{e}`)

            Affichage d'une version sécurisée (convertie en texte) à la place."""
        )
        st.dataframe(raw_df_sample.astype(str))
