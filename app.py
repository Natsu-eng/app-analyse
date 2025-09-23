import streamlit as st
import pandas as pd
import logging
import warnings

from utils.data_loading import load_data
from utils.data_analysis import sanitize_column_types_for_display
from utils.logging_config import setup_logging

# --- Initialisation Centrale ---
setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')

# --- Constantes de l'Application ---
# Les valeurs par défaut sont maintenant gérées ici
DEFAULT_TASK_TYPE = "classification"
DEFAULT_N_SPLITS = 5

# --- Fonctions de Gestion d'État ---
def initialize_session():
    """Initialise l'état de base de la session s'il n'existe pas."""
    if 'df' not in st.session_state or st.session_state.df is None:
        reset_app_state()

def reset_app_state():
    """
    Réinitialise toutes les variables de session liées à un jeu de données.
    Permet de repartir d'un état propre lors du chargement d'un nouveau fichier.
    """
    logger.info("Resetting application state for new file upload.")
    st.toast("Nouveau fichier détecté, réinitialisation de l'application...")
    
    # Réinitialisation des variables d'état
    st.session_state.df = None
    st.session_state.uploaded_file_name = None
    st.session_state.target_column_for_ml_config = None
    st.session_state.task_type = DEFAULT_TASK_TYPE
    st.session_state.config = None
    st.session_state.model_name = None
    st.session_state.model_params = {}
    st.session_state.preprocessing = {}
    st.session_state.n_splits = DEFAULT_N_SPLITS
    st.session_state.model = None
    st.session_state.metrics_summary = None
    st.session_state.preprocessor = None
    st.session_state.ml_results = [] # Assurer que les résultats sont initialisés
    logger.debug("Session state variables have been reset.")
    logger.debug("Session state variables have been reset.")

# --- Interface Principale ---
st.set_page_config(
    page_title="DataLab Pro | Accueil",
    page_icon="🧪",
    layout="centered"
)

# Initialiser la session au début de chaque exécution
initialize_session()

st.title("🧪 DataLab Pro")
st.markdown("Votre plateforme d'analyse de données et de Machine Learning automatisé.")
st.markdown("---")

st.header("1. Importez votre jeu de données")

uploaded_file = st.file_uploader(
    "Choisissez un fichier (CSV, Parquet, Excel)",
    type=["csv", "parquet", "xlsx", "xls"], key="file_uploader")

if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        reset_app_state()
        
        try:
            with st.spinner("Chargement et nettoyage des données..."):
                df, report = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Une erreur est survenue lors du chargement ou du traitement du fichier : {e}")
            logger.error(f"Failed to load data from {uploaded_file.name}: {e}")
            df = None
            report = None
        
        if df is not None:
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            logger.info(f"Successfully loaded and processed file: {uploaded_file.name}")
            
            if report and report.get("actions"):
                for action in report["actions"]:
                    st.toast(action, icon="✅")
            st.rerun()

# --- Affichage de l'état actuel ---
if st.session_state.df is not None:
    st.success(f"Fichier **{st.session_state.uploaded_file_name}** chargé et prêt pour l'analyse.")
    
    st.subheader("Aperçu des données nettoyées")
    df_to_display = st.session_state.df
    
    df_display, display_changes = sanitize_column_types_for_display(df_to_display)
    
    if display_changes:
        with st.expander("ℹ️ Rapport des conversions de types automatiques", expanded=False):
            st.markdown("Pour garantir la stabilité et la performance, les conversions suivantes ont été effectuées :")
            change_details = [f"- **`{col}`** : converti en `{desc.split(' -> ')[-1]}`" for col, desc in display_changes.items()]
            st.markdown("\n".join(change_details))

    st.dataframe(df_display)
    
    st.info("🎉 Explorez les données et lancez vos modèles via les pages dans la barre latérale.")
else:
    st.info("Veuillez charger un jeu de données pour commencer l'analyse.")