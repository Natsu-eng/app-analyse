

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.express as px
import logging

from utils.data_analysis import (
    auto_detect_column_types, 
    get_column_profile,
    detect_useless_columns,
    compute_if_dask,
    sanitize_column_types_for_display,
    is_dask_dataframe
)
from plots.exploratory import plot_correlation_heatmap, plot_distribution, plot_missing_values
from analyses import correlations, contingency_and_tests, tests_cat_vs_num

logger = logging.getLogger(__name__)

def show_dashboard():
    st.header("📊 Dashboard & Analyse Exploratoire")

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Veuillez d'abord charger un jeu de données depuis la page d'Accueil.")
        return

    df = st.session_state.df
    current_shape = (compute_if_dask(df.shape[0]), df.shape[1]) if is_dask_dataframe(df) else df.shape

    if 'column_types' not in st.session_state or st.session_state.get('df_shape') != current_shape:
        with st.spinner("Analyse des types de colonnes..."):
            st.session_state.column_types = auto_detect_column_types(df)
        st.session_state.df_shape = current_shape

    column_types = st.session_state.column_types

    tab1, tab2, tab3 = st.tabs(["📈 Vue d'Ensemble", "🔬 Analyse Interactive", "📝 Profil des Variables"])

    with tab1:
        useless_cols = detect_useless_columns(df)
        if useless_cols:
            with st.container():
                st.warning(f"Colonnes potentiellement inutiles détectées ({len(useless_cols)}): `{', '.join(useless_cols)}`")
                if st.button(f"🗑️ Supprimer les {len(useless_cols)} colonnes", key="delete_useless_cols_dashboard"):
                    df = df.drop(columns=useless_cols, errors='ignore')
                    st.session_state.df = df
                    # Réinitialiser les états dépendants pour forcer leur recalcul
                    st.session_state.pop('column_types', None)
                    st.session_state.pop('df_shape', None)
                    logger.info(f"Useless columns dropped: {useless_cols}")
                    st.toast(f"{len(useless_cols)} colonnes supprimées.", icon="✅")
                    st.rerun()
        
        col1, col2, col3 = st.columns(3)
        n_rows = compute_if_dask(df.shape[0])
        n_cols = df.shape[1]
        n_missing = compute_if_dask(df.isna().sum().sum())
        col1.metric("Nombre de lignes", f"{n_rows:,}")
        col2.metric("Nombre de colonnes", f"{n_cols:,}")
        col3.metric("Valeurs manquantes", f"{n_missing:,}")

        st.subheader("Aperçu interactif des données")
        
        df_display, _ = sanitize_column_types_for_display(df)

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
        gb.configure_grid_options(domLayout='normal')
        AgGrid(df_display, gridOptions=gb.build(), height=400, width='100%', allow_unsafe_jscode=True, enable_enterprise_modules=False)

    with tab2:
        st.subheader("Relations entre les variables")
        c1, c2 = st.columns(2)
        available_columns = df.columns.tolist()
        var1 = c1.selectbox("Variable 1", available_columns, key="dashboard_var1_select")
        var2 = c2.selectbox("Variable 2", available_columns, index=min(1, len(available_columns)-1), key="dashboard_var2_select")

        if var1 and var2 and var1 != var2:
            if st.button("Générer l'analyse", key="dashboard_generate_bivariate_analysis"):
                df_computed = compute_if_dask(df) if is_dask_dataframe(df) else df
                type1 = 'numeric' if var1 in column_types['numeric'] else 'categorical'
                type2 = 'numeric' if var2 in column_types['numeric'] else 'categorical'

                with st.spinner("Calcul en cours..."):
                    try:
                        if type1 == 'numeric' and type2 == 'numeric':
                            corr_results = correlations(df_computed, var1, var2)
                            fig = px.scatter(df_computed, x=var1, y=var2, title=f"{var1} vs {var2}", trendline="ols", trendline_color_override="red")
                            st.plotly_chart(fig, use_container_width=True)
                            if corr_results:
                                st.metric("Corrélation Pearson", f"{corr_results['pearson_r']:.3f}", f"p-value: {corr_results['pearson_p']:.3g}")

                        elif type1 == 'categorical' and type2 == 'categorical':
                            test_results = contingency_and_tests(df_computed, var1, var2)
                            st.dataframe(pd.crosstab(df_computed[var1], df_computed[var2]))
                            if test_results:
                                st.metric("V de Cramer", f"{test_results['cramers_v']:.3f}", f"p-value Chi²: {test_results['p_value']:.3g}")

                        else:
                            num_var, cat_var = (var1, var2) if type1 == 'numeric' else (var2, var1)
                            test_results = tests_cat_vs_num(df_computed, cat_var, num_var)
                            fig = px.box(df_computed, x=cat_var, y=num_var, color=cat_var, title=f"{num_var} par {cat_var}")
                            st.plotly_chart(fig, use_container_width=True)
                            if test_results and test_results.get('anova_stat'): st.metric("ANOVA F-stat", f"{test_results['anova_stat']:.3f}", f"p-value: {test_results['anova_p']:.3g}")
                            if test_results and test_results.get('t_stat'): st.metric("T-test stat", f"{test_results['t_stat']:.3f}", f"p-value: {test_results['t_p']:.3g}")
                    except Exception as e:
                        st.error(f"Impossible de générer l'analyse pour {var1} et {var2}: {e}")
                        logger.error(f"Bivariate analysis failed for {var1} and {var2}: {e}")

        st.subheader("Visualisations générales")
        try:
            with st.spinner("Génération des graphiques..."):
                st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
                st.plotly_chart(plot_missing_values(df), use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de générer les graphiques généraux. L'un des types de données n'est peut-être pas supporté. Erreur: {e}")
            logger.warning(f"Global plot generation failed: {e}")

    with tab3:
        st.subheader("Profil approfondi des variables")
        selected_col = st.selectbox("Choisir une colonne", df.columns, key="dashboard_profile_col_select")
        if selected_col:
            try:
                with st.spinner("Génération du profil de la colonne..."):
                    profile = get_column_profile(df[selected_col])
                    profile = compute_if_dask(profile)
                    
                    profile_df = pd.DataFrame(list(profile.items()), columns=['Métrique', 'Valeur'])
                    st.dataframe(profile_df)
                
                with st.spinner("Affichage de la distribution..."):
                    st.plotly_chart(plot_distribution(df, selected_col), use_container_width=True)
            except Exception as e:
                st.error(f"Impossible de générer le profil pour la colonne '{selected_col}': {e}")
                logger.error(f"Column profiling failed for '{selected_col}': {e}")
