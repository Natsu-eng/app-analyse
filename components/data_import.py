import streamlit as st
from utils.data_loading import load_data

def show_data_import():
    """Affiche la section d'importation des données"""
    with st.expander("📥 1. Importation des Données", expanded=st.session_state.df is None):
        upload = st.file_uploader(
            "Importez un dataset (CSV, Excel, Parquet, JSON)", 
            type=["csv", "xlsx", "parquet", "json"]
        )
        
        if upload:
            try:
                st.session_state.df = load_data(upload.getvalue(), upload.name)
                st.session_state.df_processed = st.session_state.df.copy()
                st.session_state.upload_name = upload.name
                st.success("Données chargées avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {str(e)}")