"""
Gestion centralisée de l'état de session Streamlit.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.config.settings import APP_CONFIG
from src.shared.logging import get_logger

logger = get_logger(__name__)

def initialize_session_state():
    """Initialise l'état de la session avec des valeurs par défaut."""
    defaults = {
        'df': None,
        'df_raw': None,
        'uploaded_file_name': None,
        'column_types': None,
        'ml_results': None,
        'task_type': APP_CONFIG["default_task_type"],
        'target_column': None,
        'feature_list': [],
        'selected_models': [],
        'preprocessing_choices': {},
        'optimize_hp': False,
        'test_size': 20,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state():
    """Réinitialise l'état de la session."""
    logger.info("Resetting session state.")
    # Garder les clés qui ne doivent pas être réinitialisées
    keys_to_keep = ['user_info'] 
    
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    initialize_session_state()
    st.cache_data.clear()
    st.cache_resource.clear()
