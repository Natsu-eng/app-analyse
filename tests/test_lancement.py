import pytest
import pandas as pd
from sklearn.datasets import make_classification
from src.models.training import train_models

def test_training_storage():
    """Vérifie que les résultats sont correctement stockés dans session_state."""
    import streamlit as st
    df, target = make_classification(n_samples=100, random_state=42)
    df = pd.DataFrame(df)
    df['target'] = target
    
    training_config = {
        'df': df,
        'target_column': 'target',
        'model_names': ['Régression Logistique'],
        'task_type': 'classification',
        'test_size': 0.2,
        'optimize': False,
        'feature_list': list(df.columns[:-1]),
        'use_smote': False,
        'preprocessing_choices': {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'scale_features': True,
            'use_smote': False
        }
    }
    
    results = train_models(**training_config)
    st.session_state.ml_results = results
    
    assert 'ml_results' in st.session_state, "ml_results non stocké"
    assert isinstance(st.session_state.ml_results, list), "ml_results doit être une liste"
    assert len(st.session_state.ml_results) == 1, "Un seul modèle attendu"
    assert st.session_state.ml_results[0]['model_name'] == 'Régression Logistique', "Nom du modèle incorrect"
    assert st.session_state.ml_results[0]['success'], "Entraînement devrait réussir"
    assert 'accuracy' in st.session_state.ml_results[0]['metrics'], "Métrique accuracy manquante"