import streamlit as st
import pandas as pd
from typing import Dict, List

# Imports des modules de l'application
from ml.catalog import MODEL_CATALOG
from utils.data_analysis import get_task_type, detect_imbalance
from ml.training import train_models # NOUVEL IMPORT

st.set_page_config(page_title="Configuration ML", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration Détaillée de l'Expérience ML")

# --- Vérification initiale des données ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Veuillez d'abord charger un jeu de données depuis la page d'Accueil.")
    st.stop()

df = st.session_state.df

# Initialisation de l'état de session pour cette page
def init_ml_config_state():
    if 'target_column_for_ml_config' not in st.session_state:
        st.session_state.target_column_for_ml_config = None
    if 'feature_list_for_ml_config' not in st.session_state:
        st.session_state.feature_list_for_ml_config = list(df.columns)
    if 'preprocessing_choices' not in st.session_state:
        st.session_state.preprocessing_choices = {}
    if 'selected_models_for_training' not in st.session_state:
        st.session_state.selected_models_for_training = []
    if 'test_split_for_ml_config' not in st.session_state:
        st.session_state.test_split_for_ml_config = 20
    if 'optimize_hp_for_ml_config' not in st.session_state:
        st.session_state.optimize_hp_for_ml_config = False

init_ml_config_state()

# --- Définition des onglets ---
tab_target, tab_preprocess, tab_models, tab_launch = st.tabs([
    "1. Cible & Features", 
    "2. Prétraitement", 
    "3. Sélection des Modèles", 
    "🚀 4. Lancement"
])

# --- Onglet 1: Cible & Features ---
with tab_target:
    st.header("Définition de la Cible et des Variables Explicatives")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.target_column_for_ml_config = st.selectbox(
            "Variable Cible (Y)", 
            options=df.columns,
            key="config_target_select"
        )
        if st.session_state.target_column_for_ml_config:
            task_type, n_classes = get_task_type(df[st.session_state.target_column_for_ml_config])
            st.session_state.task_type = task_type # Sauvegarde pour les autres onglets
            st.info(f"Tâche détectée : **{task_type.upper()}**")
    
    with col2:
        if st.session_state.target_column_for_ml_config:
            all_features = [col for col in df.columns if col != st.session_state.target_column_for_ml_config]
            st.session_state.feature_list_for_ml_config = st.multiselect(
                "Variables Explicatives (X)",
                options=all_features,
                default=all_features,
                key="config_features_select"
            )

# --- Onglet 2: Prétraitement ---
with tab_preprocess:
    st.header("Options de Prétraitement des Données")
    st.info("Ces étapes seront appliquées à l'intérieur de la validation croisée pour éviter les fuites de données.")
    
    choices = st.session_state.preprocessing_choices
    
    # Imputation
    st.subheader("Gestion des valeurs manquantes")
    choices['numeric_imputation'] = st.selectbox("Stratégie numérique", ['mean', 'median', 'knn'], key='cfg_num_strat')
    choices['categorical_imputation'] = st.selectbox("Stratégie catégorielle", ['most_frequent', 'constant'], key='cfg_cat_strat')

    # Équilibrage (si classification)
    if st.session_state.get('task_type') == 'classification':
        st.subheader("Équilibrage des classes")
        if detect_imbalance(df, st.session_state.target_column_for_ml_config):
            st.warning("Un déséquilibre a été détecté dans la variable cible.")
        choices['use_smote'] = st.checkbox("Activer SMOTE pour ré-échantillonner", key="cfg_smote")

# --- Onglet 3: Sélection des Modèles ---
with tab_models:
    st.header("Sélection et Configuration des Modèles")
    task_type = st.session_state.get('task_type', 'classification')
    
    st.session_state.selected_models_for_training = st.multiselect(
        "Modèles à entraîner et comparer",
        options=list(MODEL_CATALOG.get(task_type, {}).keys()),
        key="cfg_model_select"
    )
    
    st.subheader("Configuration de l'entraînement")
    st.session_state.test_split_for_ml_config = st.slider("Taille du jeu de test (%)", 10, 50, 20, 5, key="cfg_test_size")
    st.session_state.optimize_hp_for_ml_config = st.checkbox("Optimiser les hyperparamètres (GridSearch - plus long)", key="cfg_optimize")

# --- Onglet 4: Lancement ---
with tab_launch:
    st.header("Lancer l'Expérimentation")
    
    with st.expander("Récapitulatif de la Configuration", expanded=True):
        st.write(f"- **Cible**: `{st.session_state.target_column_for_ml_config}`")
        st.write(f"- **Features**: {len(st.session_state.feature_list_for_ml_config)} sélectionnées")
        st.write(f"- **Modèles**: `{st.session_state.selected_models_for_training}`")
        st.write(f"- **Prétraitement**: Imputation num: `{choices.get('numeric_imputation')}`, cat: `{choices.get('categorical_imputation')}`, SMOTE: `{choices.get('use_smote')}`")
        st.write(f"- **Optimisation HP**: `{st.session_state.optimize_hp_for_ml_config}`")

    if st.button("🚀 Lancer l'Expérimentation", type="primary", use_container_width=True):
        if not st.session_state.target_column_for_ml_config or not st.session_state.feature_list_for_ml_config or not st.session_state.selected_models_for_training:
            st.error("Configuration incomplète. Veuillez vérifier les onglets précédents.")
        else:
            with st.spinner("L'expérimentation est en cours... Veuillez patienter."):
                # APPEL À LA NOUVELLE FONCTION CENTRALISÉE
                results = train_models(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column_for_ml_config,
                    model_names=st.session_state.selected_models_for_training,
                    task_type=st.session_state.task_type,
                    test_size=st.session_state.test_split_for_ml_config / 100,
                    optimize=st.session_state.optimize_hp_for_ml_config,
                    feature_list=st.session_state.feature_list_for_ml_config,
                    use_smote=st.session_state.preprocessing_choices.get('use_smote', False),
                    preprocessing_choices=st.session_state.preprocessing_choices
                )
            
            st.session_state.ml_results = results
            st.success("Expérimentation terminée !")
            st.balloons()

            # Afficher un résumé des résultats
            st.subheader("Résultats de l'Expérimentation")
            for res in results:
                if res['metrics'].get('error'):
                    st.error(f"**{res['model_name']}**: Échec - {res['metrics']['error']}")
                else:
                    score = res['metrics'].get('accuracy') or res['metrics'].get('r2') or res['metrics'].get('silhouette_score', 0)
                    st.success(f"**{res['model_name']}**: Terminé avec un score de {score:.3f}")
            
            st.info("Vous pouvez maintenant consulter les résultats détaillés dans la page 'Évaluation du Modèle'.")