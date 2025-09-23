import streamlit as st
import pandas as pd
from utils.preprocessing import handle_missing_values, engineer_date_features, engineer_text_features, handle_outliers_iqr
from utils.data_analysis import auto_detect_column_types
from plots.exploratory import plot_missing_values
from st_aggrid import AgGrid

def show_preprocessing():
    """Affiche la section de prétraitement des données avec aperçu en temps réel."""
    st.header("⚙️ Prétraitement & Feature Engineering")
    st.markdown("Nettoyez, transformez et enrichissez vos données. Les modifications sont appliquées et prévisualisées ci-dessous.")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Veuillez d'abord charger des données.")
        return

    # Initialiser le dataframe traité s'il n'existe pas
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = st.session_state.df.copy()

    col1, col2 = st.columns([0.4, 0.6])

    # --- Colonne de Gauche: Options de Transformation ---
    with col1:
        with st.container(border=True):
            st.subheader("🛠️ Options de Transformation")
            
            column_types = auto_detect_column_types(st.session_state.df_processed)

            option_tabs = st.tabs(["Manquants", "Feature Eng.", "Outliers"])
            
            with option_tabs[0]:
                st.markdown("**1. Gestion des Valeurs Manquantes**")
                num_strat = st.selectbox("Stratégie numérique", ['median', 'mean', 'knn', 'delete'], key='num_strat')
                cat_strat = st.selectbox("Stratégie catégorielle", ['mode', 'delete'], key='cat_strat')

            with option_tabs[1]:
                st.markdown("**2. Feature Engineering**")
                date_cols = st.multiselect("Colonnes Date à transformer", column_types.get('datetime', []))
                text_cols = st.multiselect("Colonnes Texte à analyser (TF-IDF)", column_types.get('text', []))

            with option_tabs[2]:
                st.markdown("**3. Gestion des Outliers (IQR)**")
                outlier_cols = st.multiselect("Colonnes numériques à nettoyer", column_types.get('numeric', []))

            # Bouton pour appliquer les transformations
            if st.button("Appliquer les Transformations", type="primary", use_container_width=True):
                df_temp = st.session_state.df.copy() # Toujours partir de l'original
                with st.spinner("Application des transformations..."):
                    try:
                        # Appliquer les transformations choisies
                        df_temp = handle_missing_values(df_temp, num_strategy=num_strat, cat_strategy=cat_strat)
                        if date_cols:
                            df_temp = engineer_date_features(df_temp, date_cols)
                        if text_cols:
                            df_temp = engineer_text_features(df_temp, text_cols)
                        if outlier_cols:
                            df_temp = handle_outliers_iqr(df_temp, outlier_cols)
                        
                        st.session_state.df_processed = df_temp
                        st.success("Transformations appliquées !")
                    except Exception as e:
                        st.error(f"Erreur lors de la transformation : {e}")

            if st.button("Réinitialiser", use_container_width=True):
                st.session_state.df_processed = st.session_state.df.copy()
                st.toast("Modifications réinitialisées.")

    # --- Colonne de Droite: Aperçu des Données ---
    with col2:
        with st.container(border=True):
            st.subheader("👀 Aperçu des Données Transformées")
            
            original_rows = len(st.session_state.df)
            processed_rows = len(st.session_state.df_processed)
            
            if original_rows != processed_rows:
                st.metric("Changement du nombre de lignes", f"{processed_rows:,}", f"{processed_rows - original_rows:,}")
            else:
                st.metric("Nombre de lignes", f"{processed_rows:,}")

            view_tabs = st.tabs(["Données Transformées", "Valeurs Manquantes (Après)"])
            
            with view_tabs[0]:
                st.markdown(f"**Aperçu des {min(100, len(st.session_state.df_processed))} premières lignes**")
                AgGrid(st.session_state.df_processed.head(100), height=350, fit_columns_on_grid_load=True, key="proc_grid_preview")

            with view_tabs[1]:
                st.markdown("**Visualisation des valeurs manquantes après traitement**")
                try:
                    fig = plot_missing_values(st.session_state.df_processed)
                    st.plotly_chart(fig, use_container_width=True, key="preproc_missing_chart")
                except Exception as e:
                    st.warning(f"Impossible d'afficher le graphique des valeurs manquantes. Erreur: {e}")