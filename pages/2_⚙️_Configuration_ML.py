import streamlit as st
import pandas as pd
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional
import os

# Imports des modules de l'application
from ml.catalog import MODEL_CATALOG
from utils.data_analysis import get_target_and_task, detect_imbalance, auto_detect_column_types
from ml.training import train_models
from utils.logging_config import get_logger

# Configuration
logger = get_logger(__name__)
st.set_page_config(page_title="Configuration ML", page_icon="⚙️", layout="wide")

# --- Configuration Production ---
def setup_ml_config_environment():
    """Configuration pour l'environnement de production ML"""
    if 'ml_config_setup_done' not in st.session_state:
        st.session_state.ml_config_setup_done = True
        
        # Masquer les éléments Streamlit en production
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_ml_config_environment()

# --- Vérification initiale des données ---
@st.cache_data(ttl=300)
def validate_dataframe_for_ml(df: pd.DataFrame) -> Dict[str, Any]:
    """Valide le DataFrame pour l'analyse ML"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "min_rows_required": 50,
        "min_cols_required": 2
    }
    
    try:
        if df is None or df.empty:
            validation["is_valid"] = False
            validation["issues"].append("DataFrame vide ou non chargé")
            return validation
            
        n_rows, n_cols = df.shape
        
        # Vérification des dimensions minimales
        if n_rows < validation["min_rows_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop peu de lignes ({n_rows} < {validation['min_rows_required']})")
            
        if n_cols < validation["min_cols_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop peu de colonnes ({n_cols} < {validation['min_cols_required']})")
            
        # Vérification des valeurs manquantes excessives
        missing_ratio = df.isnull().mean().max()
        if missing_ratio > 0.8:
            validation["warnings"].append(f"Certaines colonnes ont {missing_ratio:.1%} de valeurs manquantes")
            
        # Vérification de la mémoire
        try:
            memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
            if memory_usage > 500:  # 500MB threshold
                validation["warnings"].append(f"Dataset volumineux ({memory_usage:.1f} MB)")
        except:
            validation["warnings"].append("Impossible de calculer l'utilisation mémoire")
            
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur de validation: {str(e)}")
        logger.error(f"DataFrame validation error: {e}")
        
    return validation

# --- Initialisation de l'état ML ---
def initialize_ml_config_state():
    """Initialise l'état de configuration ML de façon robuste"""
    required_keys = {
        'target_column_for_ml_config': None,
        'feature_list_for_ml_config': [],
        'preprocessing_choices': {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'use_smote': False,
            'remove_constant_cols': True,
            'remove_identifier_cols': True
        },
        'selected_models_for_training': [],
        'test_split_for_ml_config': 20,
        'optimize_hp_for_ml_config': False,
        'task_type': 'classification',
        'ml_training_in_progress': False,
        'ml_last_training_time': None,
        'ml_error_count': 0
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def safe_get_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Version sécurisée de la détection du type de tâche avec dictionnaire"""
    try:
        if target_column not in df.columns:
            return {"task_type": "unknown", "n_classes": 0, "error": "Colonne cible non trouvée"}
            
        # get_target_and_task retourne un dictionnaire, pas un tuple
        result_dict = get_target_and_task(df, target_column)
        
        # Extraction des valeurs du dictionnaire
        task_type = result_dict.get("task", "unknown")
        target_type = result_dict.get("target_type", "unknown")
        
        # Calcul du nombre de classes si classification
        n_classes = 0
        if task_type == "classification" and target_column in df.columns:
            n_classes = df[target_column].nunique()
            
        return {
            "task_type": task_type, 
            "target_type": target_type,
            "n_classes": n_classes, 
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Task type detection failed: {e}")
        return {"task_type": "unknown", "n_classes": 0, "error": str(e)}

# --- Interface Principale ---
st.title("⚙️ Configuration Détaillée de l'Expérience ML")

# Vérification initiale des données
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("📊 Veuillez d'abord charger un jeu de données depuis la page d'Accueil.")
    st.page_link("app.py", label="📋 Retour à l'accueil", icon="🏠")
    st.stop()

df = st.session_state.df

# Validation du DataFrame
validation_result = validate_dataframe_for_ml(df)
if not validation_result["is_valid"]:
    st.error("❌ Dataset incompatible avec l'analyse ML")
    for issue in validation_result["issues"]:
        st.write(f"• {issue}")
    st.stop()

# Affichage des avertissements
if validation_result["warnings"]:
    with st.expander("⚠️ Avertissements", expanded=False):
        for warning in validation_result["warnings"]:
            st.warning(warning)

# Initialisation de l'état
initialize_ml_config_state()

# Métriques du dataset
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Lignes", f"{len(df):,}")
with col2:
    st.metric("Colonnes", f"{len(df.columns)}")
with col3:
    try:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Mémoire", f"{memory_mb:.1f} MB")
    except:
        st.metric("Mémoire", "N/A")
with col4:
    st.metric("Type", "Pandas")

st.markdown("---")

# --- Définition des onglets ---
tab_target, tab_preprocess, tab_models, tab_launch = st.tabs([
    "🎯 1. Cible & Features", 
    "🔧 2. Prétraitement", 
    "🤖 3. Sélection des Modèles", 
    "🚀 4. Lancement"
])

# --- Onglet 1: Cible & Features ---
with tab_target:
    st.header("🎯 Définition de la Cible et des Variables Explicatives")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Variable Cible (Y)")
        target_column = st.selectbox(
            "Sélectionnez la variable à prédire", 
            options=df.columns,
            key="config_target_select",
            help="Cette variable sera utilisée comme cible pour l'apprentissage"
        )
        
        if target_column:
            task_info = safe_get_task_type(df, target_column)
            
            if task_info["error"]:
                st.error(f"Erreur de détection: {task_info['error']}")
            else:
                # Affichage stylisé du type de tâche
                task_type = task_info["task_type"]
                n_classes = task_info["n_classes"]
                
                if task_type == "classification":
                    st.success(f"**Tâche détectée : CLASSIFICATION**")
                    st.info(f"Nombre de classes : {n_classes}")
                    
                    # Détection du déséquilibre
                    imbalance_result = detect_imbalance(df, target_column)
                    if imbalance_result.get("is_imbalanced", False):
                        st.warning("⚖️ **Déséquilibre détecté** - Pensez à activer SMOTE dans l'onglet Prétraitement")
                    
                elif task_type == "regression":
                    st.success(f"**Tâche détectée : RÉGRESSION**")
                    # Statistiques de la variable cible
                    target_stats = df[target_column].describe()
                    st.write(f"**Plage de valeurs :** {target_stats['min']:.2f} à {target_stats['max']:.2f}")
                    
                elif task_type == "unsupervised":
                    st.info("**Tâche détectée : NON SUPERVISÉ**")
                    st.caption("Clustering ou réduction de dimension")
                
                st.session_state.task_type = task_type
                st.session_state.target_column_for_ml_config = target_column
    
    with col2:
        if st.session_state.target_column_for_ml_config:
            st.subheader("Variables Explicatives (X)")
            
            all_features = [col for col in df.columns if col != st.session_state.target_column_for_ml_config]
            
            # Détection automatique des types de colonnes pour le guide
            with st.spinner("Analyse des variables..."):
                column_types = auto_detect_column_types(df[all_features])
            
            # Interface de sélection avec informations
            selected_features = st.multiselect(
                "Sélectionnez les variables d'entrée",
                options=all_features,
                default=all_features,
                key="config_features_select",
                help="Variables utilisées pour prédire la cible"
            )
            
            st.session_state.feature_list_for_ml_config = selected_features
            
            # Statistiques des features sélectionnées
            if selected_features:
                st.success(f"✅ {len(selected_features)} variables sélectionnées")
                
                # Répartition par type
                numeric_count = len([f for f in selected_features if f in column_types.get('numeric', [])])
                categorical_count = len([f for f in selected_features if f in column_types.get('categorical', [])])
                other_count = len(selected_features) - numeric_count - categorical_count
                
                st.caption(f"📊 {numeric_count} numériques • {categorical_count} catégorielles • {other_count} autres")
            else:
                st.error("❌ Aucune variable sélectionnée")

# --- Onglet 2: Prétraitement ---
with tab_preprocess:
    st.header("🔧 Options de Prétraitement des Données")
    
    st.info("""
    ⚠️ **Important** : Ces traitements sont appliqués à l'intérieur de la validation croisée 
    pour éviter les fuites de données (data leakage). Chaque fold est traité indépendamment.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧩 Gestion des valeurs manquantes")
        
        st.session_state.preprocessing_choices['numeric_imputation'] = st.selectbox(
            "Stratégie pour les variables numériques",
            options=['mean', 'median', 'constant', 'knn'],
            index=0,
            key='cfg_num_strat',
            help="Moyenne, Médiane, Valeur constante (0), ou K-plus proches voisins"
        )
        
        st.session_state.preprocessing_choices['categorical_imputation'] = st.selectbox(
            "Stratégie pour les variables catégorielles",
            options=['most_frequent', 'constant'],
            index=0,
            key='cfg_cat_strat',
            help="Valeur la plus fréquente ou valeur constante ('missing')"
        )
        
        st.session_state.preprocessing_choices['remove_constant_cols'] = st.checkbox(
            "Supprimer les colonnes constantes",
            value=True,
            key="cfg_remove_constant",
            help="Élimine les colonnes sans variance"
        )
        
        st.session_state.preprocessing_choices['remove_identifier_cols'] = st.checkbox(
            "Supprimer les colonnes de type ID",
            value=True,
            key="cfg_remove_id",
            help="Élimine les colonnes avec des valeurs uniques pour chaque ligne"
        )
    
    with col2:
        st.subheader("⚖️ Équilibrage des données")
        
        # Afficher SMOTE seulement pour la classification
        if st.session_state.get('task_type') == 'classification':
            imbalance_info = detect_imbalance(df, st.session_state.target_column_for_ml_config)
            
            if imbalance_info.get("is_imbalanced", False):
                st.warning("📉 **Déséquilibre détecté**")
                st.write(f"Ratio de déséquilibre : {imbalance_info.get('imbalance_ratio', 'N/A'):.2f}")
                
                st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                    "Activer SMOTE (Synthetic Minority Over-sampling Technique)",
                    value=True,
                    key="cfg_smote",
                    help="Génère des échantillons synthétiques pour les classes minoritaires"
                )
                
                if st.session_state.preprocessing_choices['use_smote']:
                    st.success("✅ SMOTE sera appliqué pendant l'entraînement")
            else:
                st.success("✅ Les classes sont équilibrées")
                st.session_state.preprocessing_choices['use_smote'] = False
        else:
            st.info("🔒 L'équilibrage SMOTE n'est disponible que pour la classification")
            st.session_state.preprocessing_choices['use_smote'] = False

# --- Onglet 3: Sélection des Modèles ---
with tab_models:
    st.header("🤖 Sélection et Configuration des Modèles")
    
    task_type = st.session_state.get('task_type', 'classification')
    available_models = list(MODEL_CATALOG.get(task_type, {}).keys())
    
    if not available_models:
        st.error(f"❌ Aucun modèle disponible pour la tâche '{task_type}'")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Modèles disponibles")
        
        selected_models = st.multiselect(
            "Sélectionnez les modèles à entraîner et comparer",
            options=available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models,
            key="cfg_model_select",
            help="Les modèles seront entraînés et comparés automatiquement"
        )
        
        st.session_state.selected_models_for_training = selected_models
        
        # Informations sur les modèles sélectionnés
        if selected_models:
            st.success(f"✅ {len(selected_models)} modèles sélectionnés")
            
            # Afficher les détails des modèles
            with st.expander("📋 Détails des modèles sélectionnés", expanded=False):
                for model_name in selected_models:
                    model_config = MODEL_CATALOG[task_type][model_name]
                    st.write(f"**{model_name}**")
                    st.caption(f"Type: {type(model_config['model']).__name__}")
                    if model_config.get('params'):
                        st.caption(f"Hyperparamètres à optimiser: {len(model_config['params'])}")
    
    with col2:
        st.subheader("⚙️ Configuration")
        
        st.session_state.test_split_for_ml_config = st.slider(
            "Taille du jeu de test (%)", 
            min_value=10, 
            max_value=40, 
            value=20, 
            step=5,
            key="cfg_test_size",
            help="Pourcentage des données réservé pour le test"
        )
        
        st.session_state.optimize_hp_for_ml_config = st.checkbox(
            "Optimisation des hyperparamètres", 
            value=False,
            key="cfg_optimize",
            help="Recherche systématique des meilleurs paramètres (plus long)"
        )
        
        if st.session_state.optimize_hp_for_ml_config:
            st.warning("⏰ L'optimisation peut multiplier le temps d'entraînement")

# --- Onglet 4: Lancement ---
with tab_launch:
    st.header("🚀 Lancement de l'Expérimentation")
    
    # Vérification de la configuration
    config_errors = []
    
    if not st.session_state.target_column_for_ml_config:
        config_errors.append("Variable cible non définie")
    
    if not st.session_state.feature_list_for_ml_config:
        config_errors.append("Aucune variable explicative sélectionnée")
    
    if not st.session_state.selected_models_for_training:
        config_errors.append("Aucun modèle sélectionné")
    
    # Affichage du récapitulatif
    with st.expander("📋 Récapitulatif de la Configuration", expanded=True):
        if config_errors:
            st.error("❌ Configuration incomplète:")
            for error in config_errors:
                st.write(f"• {error}")
        else:
            st.success("✅ Configuration valide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Données**")
                st.write(f"• Cible: `{st.session_state.target_column_for_ml_config}`")
                st.write(f"• Features: {len(st.session_state.feature_list_for_ml_config)} variables")
                st.write(f"• Test: {st.session_state.test_split_for_ml_config}%")
                
            with col2:
                st.write("**Modèles**")
                st.write(f"• {len(st.session_state.selected_models_for_training)} modèles")
                st.write(f"• Optimisation: {'✅' if st.session_state.optimize_hp_for_ml_config else '❌'}")
                st.write(f"• SMOTE: {'✅' if st.session_state.preprocessing_choices.get('use_smote') else '❌'}")
    
    # Bouton de lancement
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        launch_disabled = len(config_errors) > 0 or st.session_state.get('ml_training_in_progress', False)
        
        if st.button(
            "🚀 Lancer l'Expérimentation", 
            type="primary", 
            use_container_width=True,
            disabled=launch_disabled,
            help="Démarrer l'entraînement des modèles"
        ):
            st.session_state.ml_training_in_progress = True
            st.session_state.ml_last_training_time = time.time()
            
            # Lancement de l'entraînement
            with st.spinner("🧠 Entraînement des modèles en cours... Cette opération peut prendre plusieurs minutes."):
                try:
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
                    st.session_state.ml_training_in_progress = False
                    st.session_state.ml_error_count = 0
                    
                    st.success("✅ Expérimentation terminée avec succès!")
                    st.balloons()
                    
                    # Affichage des résultats
                    st.subheader("📊 Résultats de l'Expérimentation")
                    
                    successful_models = 0
                    for res in results:
                        if res['metrics'].get('error'):
                            st.error(f"**{res['model_name']}**: ❌ Échec - {res['metrics']['error']}")
                        else:
                            successful_models += 1
                            # Score principal selon le type de tâche
                            if st.session_state.task_type == "classification":
                                score = res['metrics'].get('accuracy', 0)
                                st.success(f"**{res['model_name']}**: ✅ Exactitude = {score:.3f}")
                            elif st.session_state.task_type == "regression":
                                score = res['metrics'].get('r2', 0)
                                st.success(f"**{res['model_name']}**: ✅ R² = {score:.3f}")
                            else:
                                score = res['metrics'].get('silhouette_score', 0)
                                st.success(f"**{res['model_name']}**: ✅ Score = {score:.3f}")
                    
                    st.info(f"📈 {successful_models}/{len(results)} modèles entraînés avec succès")
                    
                    # Navigation vers les résultats
                    st.page_link("pages/4_📈_Évaluation_du_Modèle.py", label="📊 Voir les résultats détaillés", icon="📈")
                    
                except Exception as e:
                    st.session_state.ml_training_in_progress = False
                    st.session_state.ml_error_count += 1
                    st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                    logger.error(f"Training failed: {e}", exc_info=True)
    
    with col_info:
        if st.session_state.get('ml_training_in_progress', False):
            st.info("⏳ Entraînement en cours... Veuillez patienter.")
        elif st.session_state.get('ml_last_training_time'):
            last_time = st.session_state.ml_last_training_time
            st.caption(f"Dernier entraînement: {time.strftime('%H:%M:%S', time.localtime(last_time))}")
        
        if st.session_state.get('ml_error_count', 0) > 0:
            st.warning(f"⚠️ {st.session_state.ml_error_count} erreur(s) lors des entraînements")

# Footer avec monitoring
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if st.session_state.get('ml_error_count', 0) > 0:
        st.caption(f"⚠️ Erreurs ML: {st.session_state.ml_error_count}")

with footer_col2:
    current_time = time.strftime("%H:%M:%S")
    st.caption(f"⏰ Session: {current_time}")

with footer_col3:
    if st.button("🧹 Nettoyer cache ML", help="Libère la mémoire des modèles"):
        gc.collect()
        if 'ml_results' in st.session_state:
            del st.session_state.ml_results
        st.success("Cache ML nettoyé")
        st.rerun()

# Gestion d'erreurs globale
if st.session_state.get('ml_error_count', 0) > 5:
    st.error("⚠️ Plusieurs erreurs détectées. Considérez recharger l'application.")
    if st.button("🔄 Recharger la page ML"):
        st.session_state.ml_error_count = 0
        st.rerun()

# Ajoutez cette fonction dans votre Configuration_ML.py
def clear_cache_and_restart():
    """Nettoie le cache et redémarre l'application"""
    try:
        st.cache_data.clear()
        st.success("Cache nettoyé avec succès!")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du nettoyage du cache : {e}")

# Bouton de nettoyage dans la sidebar
if st.sidebar.button("🔄 Nettoyer le cache"):
    clear_cache_and_restart()