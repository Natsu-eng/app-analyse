import streamlit as st
import logging
from utils.data_analysis import auto_detect_target, get_task_type, detect_imbalance
from utils.data_analysis import get_relevant_features
from models.catalog import MODEL_CATALOG
from models.training import train_models

def show_sidebar():
    """Affiche la sidebar avec la configuration de l'expérience"""
    with st.sidebar:
        st.header("🛠️ Configuration de l'Expérience")
        st.markdown("---")
        
        with st.expander("📖 Guide d'Utilisation"):
            st.markdown("""
            1. **Importez des données** : Chargez un fichier depuis la page d'Accueil.
            2. **Configuration** : Choisissez la cible, les features et les modèles ici.
            3. **Lancer** : Entraînez et évaluez les modèles.
            4. **Consulter** : Analysez les résultats dans les pages du Dashboard et de l'Évaluation.
            5. **Export** : Téléchargez les modèles ou un rapport PDF.
            """)
        
        if st.session_state.get('df') is not None:
            df = st.session_state.df
            # Détection automatique de la cible
            auto_target = auto_detect_target(df)
            
            # Construction de la liste des choix pour la cible
            target_options = df.columns.tolist()
            try:
                # Mettre la cible auto-détectée en premier si elle existe
                if auto_target:
                    target_options.insert(0, target_options.pop(target_options.index(auto_target)))
            except (ValueError, IndexError):
                pass # auto_target n'était pas dans la liste
            
            target_col = st.selectbox(
                "1. Variable Cible (Y)", 
                options=target_options,
                key="target_select"
            )
            
            if target_col:
                st.session_state.target_column_for_ml_config = target_col
                try:
                    task_type, n_classes = get_task_type(df[target_col])
                    st.session_state.task_type = task_type
                    st.info(f"Tâche détectée : **{task_type.upper()}** (Classes: {n_classes})")
                    
                    # Sélection des features
                    all_features = [col for col in df.columns if col != target_col]
                    st.session_state.selected_features = st.multiselect(
                        "2. Features X à utiliser", 
                        all_features, 
                        default=get_relevant_features(df, target_col)
                    )
                    
                    # Gestion déséquilibre pour classification
                    if task_type == 'classification':
                        imbalance = detect_imbalance(df, target_col)
                        if imbalance:
                            st.warning("Déséquilibre détecté dans la cible. Activez SMOTE pour équilibrer.")
                        st.session_state.balance_enabled = st.checkbox(
                            "Équilibrer les classes (SMOTE)", 
                            value=imbalance
                        )
                    
                    # Sélection des modèles
                    models_to_train = st.multiselect(
                        "3. Modèles à comparer", 
                        list(MODEL_CATALOG.get(task_type, {}).keys())
                    )
                    
                    optimize = st.checkbox("Optimiser les hyperparamètres (plus lent)")
                    test_size = st.slider("4. Taille du jeu de test (%)", 10, 50, 20, 5)
                    
                    # Bouton pour lancer l'entraînement
                    if st.button("🚀 Lancer l'Expérimentation", use_container_width=True, type="primary"):
                        # On passe le dataframe traité s'il existe, sinon l'original
                        df_to_train = st.session_state.get('df_processed', df)
                        
                        st.session_state.ml_results = train_models(
                            df_to_train, 
                            target_col, 
                            models_to_train, 
                            task_type, 
                            test_size / 100, # Convertir en ratio
                            optimize, 
                            st.session_state.selected_features, 
                            st.session_state.get('balance_enabled', False),
                            st.session_state.get('preprocessing_choices', {}) # Passer les choix
                        )
                        st.success("Expérimentation terminée !")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"Erreur de configuration : {str(e)}")
                    logging.exception("Erreur de configuration dans la sidebar")
        else:
            st.info("Chargez des données pour commencer.")
        
        st.markdown("---")