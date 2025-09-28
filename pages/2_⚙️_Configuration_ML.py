import streamlit as st
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Any
from functools import wraps
import numpy as np

# Imports des modules ML
from ml.catalog import MODEL_CATALOG
from utils.data_analysis import get_target_and_task, detect_imbalance, auto_detect_column_types
from ml.training import train_models
from utils.logging_config import get_logger

# Configuration
logger = get_logger(__name__)
st.set_page_config(page_title="Configuration ML", page_icon="⚙️", layout="wide")

# Configuration production
def setup_ml_config_environment():
    """Configuration pour l'environnement de production ML"""
    if 'ml_config_setup_done' not in st.session_state:
        st.session_state.ml_config_setup_done = True
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            .stAlert > div {padding: 0.5rem;}
            .stProgress .st-bo {background-color: #3498db;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_ml_config_environment()

# Décorateur de monitoring
def monitor_ml_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 10:
                logger.warning(f"ML operation {func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"ML operation {func.__name__} failed: {e}")
            raise
    return wrapper

# Validation des features pour clustering
def validate_clustering_features(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Valide que les features sont adaptées au clustering avec suggestions"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "valid_features": [],
        "suggested_features": []
    }
    
    try:
        # Analyse des corrélations pour suggestions
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            low_corr_cols = [col for col in numeric_cols if col in features and corr_matrix[col].mean() < 0.7]
            validation["suggested_features"] = low_corr_cols[:10]  # Limite à 10

        for feature in features:
            if feature not in df.columns:
                validation["issues"].append(f"Colonne '{feature}' non trouvée")
                continue
                
            if df[feature].std() == 0:
                validation["warnings"].append(f"'{feature}' est constante")
                continue
                
            missing_ratio = df[feature].isnull().mean()
            if missing_ratio > 0.5:
                validation["warnings"].append(f"'{feature}' a {missing_ratio:.1%} de valeurs manquantes")
                continue
                
            if df[feature].nunique() == 1:
                validation["warnings"].append(f"'{feature}' n'a qu'une seule valeur unique")
                continue
                
            validation["valid_features"].append(feature)
        
        validation["is_valid"] = len(validation["valid_features"]) >= 2
        if not validation["is_valid"]:
            validation["issues"].append("Minimum 2 variables numériques requises")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
    
    return validation

# Estimation temps d'entraînement
def estimate_training_time(df: pd.DataFrame, n_models: int, task_type: str, 
                         optimize_hp: bool, n_features: int, use_smote: bool = False) -> int:
    """Estime le temps d'entraînement de façon réaliste"""
    try:
        n_samples = len(df)
        base_complexity = (n_samples * n_features) / 1000
        time_multipliers = {
            'clustering': 1.2,
            'classification': 1.5 if use_smote else 1.3,
            'regression': 1.4
        }
        time_multiplier = time_multipliers.get(task_type, 1.5)
        if optimize_hp:
            time_multiplier *= 3.5
        estimated_seconds = base_complexity * n_models * time_multiplier
        return max(30, min(int(estimated_seconds), 3600))
    except Exception as e:
        logger.warning(f"Erreur estimation temps: {e}")
        return 60

# Vérification ressources système
def check_system_resources(df: pd.DataFrame, n_models: int) -> Dict[str, Any]:
    """Vérifie si le système a assez de ressources"""
    check_result = {
        "has_enough_resources": True,
        "issues": [],
        "warnings": [],
        "available_memory_mb": 0,
        "estimated_needed_mb": 0
    }
    
    try:
        df_memory = df.memory_usage(deep=True).sum() / (1024**2)
        estimated_needed = df_memory * n_models * 3
        available_memory = psutil.virtual_memory().available / (1024**2)
        check_result["available_memory_mb"] = available_memory
        check_result["estimated_needed_mb"] = estimated_needed
        
        if estimated_needed > available_memory:
            check_result["has_enough_resources"] = False
            check_result["issues"].append(f"Mémoire insuffisante: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        elif estimated_needed > available_memory * 0.7:
            check_result["warnings"].append(f"Mémoire limite: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            check_result["warnings"].append(f"CPU élevé: {cpu_percent:.1f}%")
            
    except Exception as e:
        check_result["warnings"].append("Vérification ressources échouée")
        logger.warning(f"Erreur vérification ressources: {e}")
    
    return check_result

# Validation DataFrame
@st.cache_data(ttl=300, max_entries=3)
def validate_dataframe_for_ml(df: pd.DataFrame) -> Dict[str, Any]:
    """Valide le DataFrame pour ML"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "min_rows_required": 50,
        "min_cols_required": 2,
        "stats": {}
    }
    
    try:
        if df is None or df.empty:
            validation["is_valid"] = False
            validation["issues"].append("DataFrame vide ou non chargé")
            return validation
        
        n_rows, n_cols = df.shape
        validation["stats"] = {"n_rows": n_rows, "n_cols": n_cols}
        
        if n_rows < validation["min_rows_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant: {n_rows} lignes (minimum: {validation['min_rows_required']})")
        
        if n_cols < validation["min_cols_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant: {n_cols} colonnes (minimum: {validation['min_cols_required']})")
        
        missing_ratio = df.isnull().mean().max()
        if missing_ratio > 0.95:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop de valeurs manquantes: {missing_ratio:.1%}")
        elif missing_ratio > 0.7:
            validation["warnings"].append(f"Beaucoup de valeurs manquantes: {missing_ratio:.1%}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if len(constant_cols) == len(numeric_cols):
                validation["warnings"].append("Toutes les colonnes numériques sont constantes")
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        validation["stats"]["memory_mb"] = memory_usage
        if memory_usage > 1000:
            validation["warnings"].append(f"Dataset volumineux: {memory_usage:.1f} MB")
            
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"DataFrame validation error: {e}")
    
    return validation

# Initialisation état
def initialize_ml_config_state():
    """Initialise l'état de configuration ML"""
    defaults = {
        'target_column_for_ml_config': None,
        'feature_list_for_ml_config': [],
        'preprocessing_choices': {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'use_smote': False,
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'pca_preprocessing': False
        },
        'selected_models_for_training': [],
        'test_split_for_ml_config': 20,
        'optimize_hp_for_ml_config': False,
        'task_type': 'classification',
        'ml_training_in_progress': False,
        'ml_last_training_time': None,
        'ml_error_count': 0,
        'ml_session_id': int(time.time()),
        'current_step': 1,
        'previous_task_type': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@monitor_ml_operation
def safe_get_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Détection sécurisée du type de tâche"""
    try:
        if not target_column or target_column not in df.columns:
            return {"task_type": "unknown", "n_classes": 0, "error": "Colonne cible invalide"}
        
        if df[target_column].nunique() == len(df):
            return {"task_type": "unknown", "n_classes": 0, "error": "Variable cible est un identifiant"}
        
        result_dict = get_target_and_task(df, target_column)
        task_type = result_dict.get("task", "unknown")
        target_type = result_dict.get("target_type", "unknown")
        
        n_classes = 0
        if task_type == "classification":
            n_classes = df[target_column].nunique()
            if n_classes > 100:
                return {"task_type": "unknown", "n_classes": n_classes, "error": f"Trop de classes ({n_classes})"}
        
        return {"task_type": task_type, "target_type": target_type, "n_classes": n_classes, "error": None}
    except Exception as e:
        logger.error(f"Task type detection failed: {e}")
        return {"task_type": "unknown", "n_classes": 0, "error": str(e)}

def get_task_specific_models(task_type: str) -> List[str]:
    """Retourne les modèles disponibles pour une tâche"""
    try:
        return list(MODEL_CATALOG.get(task_type, {}).keys())
    except Exception as e:
        logger.error(f"Error getting models for {task_type}: {e}")
        return []

def get_default_models_for_task(task_type: str) -> List[str]:
    """Retourne les modèles par défaut pour une tâche"""
    default_models = {
        'classification': ['RandomForest', 'XGBoost', 'LogisticRegression'],
        'regression': ['RandomForest', 'XGBoost', 'LinearRegression'],
        'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture']
    }
    available_models = get_task_specific_models(task_type)
    return [model for model in default_models.get(task_type, []) if model in available_models]

# Interface principale
st.title("⚙️ Configuration de l'Expérimentation ML")
st.markdown("Configurez votre analyse en 4 étapes simples et lancez l'entraînement des modèles.")

# Vérification des données
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("📊 Aucun dataset chargé")
    st.info("Chargez un dataset depuis la page d'accueil.")
    if st.button("🏠 Retour à l'accueil"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state.df

# Validation DataFrame
validation_result = validate_dataframe_for_ml(df)
if not validation_result["is_valid"]:
    st.error("❌ Dataset non compatible avec l'analyse ML")
    with st.expander("🔍 Détails des problèmes", expanded=True):
        for issue in validation_result["issues"]:
            st.error(f"• {issue}")
        st.info("**Critères requis**:\n- Minimum 50 lignes\n- Minimum 2 colonnes\n- Moins de 95% de valeurs manquantes")
    if st.button("🔄 Revérifier"):
        st.rerun()
    st.stop()

if validation_result["warnings"]:
    with st.expander("⚠️ Avertissements qualité données", expanded=False):
        for warning in validation_result["warnings"]:
            st.warning(f"• {warning}")

# Initialisation état
initialize_ml_config_state()

# Métriques dataset
st.markdown("### 📊 Aperçu du Dataset")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📏 Lignes", f"{validation_result['stats']['n_rows']:,}")
with col2:
    st.metric("📋 Colonnes", validation_result["stats"]["n_cols"])
with col3:
    memory_mb = validation_result["stats"].get("memory_mb", 0)
    st.metric("💾 Mémoire", f"{memory_mb:.1f} MB" if memory_mb > 0 else "N/A")
with col4:
    missing_pct = df.isnull().mean().mean() * 100
    st.metric("🕳️ Manquant", f"{missing_pct:.1f}%")
with col5:
    sys_memory = psutil.virtual_memory().percent
    color = "🟢" if sys_memory < 70 else "🟡" if sys_memory < 85 else "🔴"
    st.metric(f"{color} RAM Sys", f"{sys_memory:.0f}%")

st.markdown("---")

# Navigation par étapes
steps = ["🎯 Cible", "🔧 Préprocess", "🤖 Modèles", "🚀 Lancement"]
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav2:
    selected_step = st.radio("Étapes", steps, index=st.session_state.current_step - 1, horizontal=True, key="step_selector")
st.session_state.current_step = steps.index(selected_step) + 1

# Étape 1: Configuration cible
if st.session_state.current_step == 1:
    st.header("🎯 Configuration de la Tâche et Cible")
    
    task_options = ["Classification Supervisée", "Régression Supervisée", "Clustering Non Supervisé"]
    task_descriptions = {
        "Classification Supervisée": "Prédire des catégories (ex: spam/non-spam)",
        "Régression Supervisée": "Prédire des valeurs numériques (ex: prix, score)", 
        "Clustering Non Supervisé": "Découvrir des groupes naturels dans les données"
    }
    
    current_task_idx = {'classification': 0, 'regression': 1, 'clustering': 2}.get(st.session_state.task_type, 0)
    task_selection = st.selectbox(
        "Type de problème ML",
        options=task_options,
        index=current_task_idx,
        key="ml_task_selection",
        help="Sélectionnez le type d'apprentissage adapté à vos données"
    )
    st.info(f"**{task_selection}** - {task_descriptions[task_selection]}")
    
    selected_task_type = {
        "Classification Supervisée": "classification",
        "Régression Supervisée": "regression",
        "Clustering Non Supervisé": "clustering"
    }[task_selection]
    
    if st.session_state.previous_task_type != selected_task_type:
        st.session_state.target_column_for_ml_config = None
        st.session_state.feature_list_for_ml_config = []
        st.session_state.preprocessing_choices['use_smote'] = False
        st.session_state.previous_task_type = selected_task_type
        st.session_state.task_type = selected_task_type
        st.rerun()
    else:
        st.session_state.task_type = selected_task_type
    
    if selected_task_type in ['classification', 'regression']:
        st.subheader("🎯 Variable Cible (Y)")
        available_targets = (
            [col for col in df.columns if df[col].nunique() <= 50 or not pd.api.types.is_numeric_dtype(df[col])]
            if selected_task_type == 'classification' else
            [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10]
        )
        
        if not available_targets:
            st.error("❌ Aucune variable cible appropriée trouvée")
            st.info("Classification: ≤50 valeurs uniques\nRégression: numérique, >10 valeurs uniques")
        else:
            available_targets = [None] + available_targets
            target_idx = available_targets.index(st.session_state.target_column_for_ml_config) if st.session_state.target_column_for_ml_config in available_targets else 0
            target_column = st.selectbox(
                "Variable à prédire",
                options=available_targets,
                index=target_idx,
                key="ml_target_selector",
                help="Variable que le modèle apprendra à prédire"
            )
            
            if target_column != st.session_state.target_column_for_ml_config:
                st.session_state.target_column_for_ml_config = target_column
                st.session_state.feature_list_for_ml_config = []
            
            if target_column:
                task_info = safe_get_task_type(df, target_column)
                if task_info["error"]:
                    st.error(f"❌ Erreur: {task_info['error']}")
                    st.info("Action: Sélectionnez une autre colonne ou vérifiez les données.")
                else:
                    if selected_task_type == "classification":
                        st.success(f"✅ **Classification** ({task_info['n_classes']} classes)")
                        class_dist = df[target_column].value_counts()
                        if len(class_dist) <= 10:
                            st.bar_chart(class_dist)
                            st.caption(f"Distribution des classes")
                        imbalance_info = detect_imbalance(df, target_column)
                        if imbalance_info.get("is_imbalanced", False):
                            st.warning(f"⚠️ Déséquilibre (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                    else:
                        st.success("✅ **Régression**")
                        target_stats = df[target_column].describe()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Moyenne", f"{target_stats['mean']:.3f}")
                        with col2:
                            st.metric("Médiane", f"{target_stats['50%']:.3f}")
                        with col3:
                            st.metric("Écart-type", f"{target_stats['std']:.3f}")
                        with col4:
                            st.metric("Plage", f"{target_stats['max'] - target_stats['min']:.3f}")
        
        st.subheader("📊 Variables Explicatives (X)")
        all_features = [col for col in df.columns if col != target_column] if target_column else list(df.columns)
        
        if all_features:
            recommend_features = st.checkbox(
                "Sélection automatique des features",
                value=len(st.session_state.feature_list_for_ml_config) == 0,
                help="Sélectionne automatiquement les variables pertinentes"
            )
            
            if recommend_features and target_column:
                with st.spinner("🤖 Analyse des features..."):
                    column_types = auto_detect_column_types(df)
                    recommended_features = column_types.get('numeric', []) + [
                        col for col in column_types.get('categorical', []) if df[col].nunique() <= 20
                    ]
                    recommended_features = [col for col in recommended_features if col != target_column and col in all_features][:25]
                    st.session_state.feature_list_for_ml_config = recommended_features
                    st.success(f"✅ {len(recommended_features)} features sélectionnées")
            else:
                selected_features = st.multiselect(
                    "Variables d'entrée",
                    options=all_features,
                    default=st.session_state.feature_list_for_ml_config,
                    key="ml_features_selector",
                    help="Variables utilisées pour la prédiction"
                )
                st.session_state.feature_list_for_ml_config = selected_features
            
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} features sélectionnées")
                st.caption(f"📋 {', '.join(st.session_state.feature_list_for_ml_config[:10])}{' ...' if len(st.session_state.feature_list_for_ml_config) > 10 else ''}")
                if len(st.session_state.feature_list_for_ml_config) > 30:
                    st.warning("⚠️ Nombre élevé de features - risque de surapprentissage")
                    st.info("Action: Réduisez le nombre de features ou activez PCA.")
            else:
                st.warning("⚠️ Aucune feature sélectionnée")
                st.info("Action: Sélectionnez au moins une variable.")
        else:
            st.error("❌ Aucune feature disponible")
            st.info("Action: Vérifiez votre dataset.")
    
    else:  # Clustering
        st.session_state.target_column_for_ml_config = None
        st.success("✅ **Clustering Non Supervisé**")
        st.info("🔍 Le modèle identifiera des groupes naturels dans les données.")
        
        all_numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        if not all_numeric_features:
            st.error("❌ Aucune variable numérique pour le clustering")
            st.info("Action: Ajoutez des variables numériques au dataset.")
        else:
            st.subheader("📊 Variables pour le Clustering")
            auto_cluster_features = st.checkbox(
                "Sélection automatique",
                value=len(st.session_state.feature_list_for_ml_config) == 0,
                help="Sélectionne les variables numériques adaptées"
            )
            
            if auto_cluster_features:
                validation_result = validate_clustering_features(df, all_numeric_features)
                st.session_state.feature_list_for_ml_config = validation_result["valid_features"]
                if validation_result["suggested_features"]:
                    st.info(f"💡 Suggestion: {', '.join(validation_result['suggested_features'][:5])}")
                if validation_result["warnings"]:
                    with st.expander("⚠️ Avertissements", expanded=True):
                        for warning in validation_result["warnings"]:
                            st.warning(f"• {warning}")
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées")
            else:
                clustering_features = st.multiselect(
                    "Variables pour clustering",
                    options=all_numeric_features,
                    default=st.session_state.feature_list_for_ml_config or all_numeric_features[:10],
                    key="clustering_features_selector",
                    help="Variables numériques pour identifier les clusters"
                )
                validation_result = validate_clustering_features(df, clustering_features)
                st.session_state.feature_list_for_ml_config = validation_result["valid_features"]
                if validation_result["suggested_features"]:
                    st.info(f"💡 Suggestion: {', '.join(validation_result['suggested_features'][:5])}")
                if validation_result["warnings"]:
                    with st.expander("⚠️ Avertissements", expanded=True):
                        for warning in validation_result["warnings"]:
                            st.warning(f"• {warning}")
            
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées")
                if len(st.session_state.feature_list_for_ml_config) < 2:
                    st.warning("⚠️ Minimum 2 variables pour clustering")
                    st.info("Action: Ajoutez des variables numériques.")
                elif len(st.session_state.feature_list_for_ml_config) > 15:
                    st.warning("⚠️ Nombre élevé de variables - risque de malédiction dimensionnelle")
                    st.info("Action: Activez PCA ou réduisez les variables.")
                with st.expander("📈 Aperçu statistiques", expanded=False):
                    st.dataframe(df[st.session_state.feature_list_for_ml_config].describe().style.format("{:.3f}"), width='stretch')
            else:
                st.warning("⚠️ Aucune variable sélectionnée")
                st.info("Action: Sélectionnez des variables numériques.")

# Étape 2: Prétraitement
elif st.session_state.current_step == 2:
    st.header("🔧 Configuration du Prétraitement")
    task_type = st.session_state.get('task_type', 'classification')
    
    st.info(f"**Pipeline pour {task_type.upper()}**: Transformations appliquées séparément sur train/validation pour éviter le data leakage.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧩 Valeurs Manquantes")
        st.session_state.preprocessing_choices['numeric_imputation'] = st.selectbox(
            "Variables numériques",
            options=['mean', 'median', 'constant', 'knn'],
            index=['mean', 'median', 'constant', 'knn'].index(st.session_state.preprocessing_choices.get('numeric_imputation', 'mean')),
            key='numeric_imputation_selector',
            help="mean=moyenne, median=médiane, constant=0, knn=k-voisins"
        )
        st.session_state.preprocessing_choices['categorical_imputation'] = st.selectbox(
            "Variables catégorielles",
            options=['most_frequent', 'constant'],
            index=['most_frequent', 'constant'].index(st.session_state.preprocessing_choices.get('categorical_imputation', 'most_frequent')),
            key='categorical_imputation_selector',
            help="most_frequent=mode, constant='missing'"
        )
        
        st.subheader("🧹 Nettoyage")
        st.session_state.preprocessing_choices['remove_constant_cols'] = st.checkbox(
            "Supprimer colonnes constantes",
            value=st.session_state.preprocessing_choices.get('remove_constant_cols', True),
            key="remove_constant_checkbox",
            help="Élimine variables sans variance"
        )
        st.session_state.preprocessing_choices['remove_identifier_cols'] = st.checkbox(
            "Supprimer colonnes identifiantes",
            value=st.session_state.preprocessing_choices.get('remove_identifier_cols', True),
            key="remove_id_checkbox",
            help="Élimine variables avec valeurs uniques (ID)"
        )
    
    with col2:
        st.subheader("📏 Normalisation")
        scale_help = {
            'classification': "Recommandé pour SVM, KNN, réseaux de neurones",
            'regression': "Recommandé pour la plupart des algorithmes",
            'clustering': "ESSENTIEL pour le clustering (KMeans, DBSCAN)"
        }
        st.session_state.preprocessing_choices['scale_features'] = st.checkbox(
            "Normaliser les features",
            value=st.session_state.preprocessing_choices.get('scale_features', True),
            key="scale_features_checkbox",
            help=scale_help.get(task_type, "Recommandé")
        )
        if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
            st.error("❌ Normalisation critique pour le clustering!")
            st.info("Action: Activez la normalisation pour de meilleurs résultats.")
        
        if task_type == 'classification':
            st.subheader("⚖️ Déséquilibre")
            if st.session_state.target_column_for_ml_config:
                imbalance_info = detect_imbalance(df, st.session_state.target_column_for_ml_config)
                if imbalance_info.get("is_imbalanced", False):
                    st.warning(f"📉 Déséquilibre (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                    st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                        "Activer SMOTE",
                        value=st.session_state.preprocessing_choices.get('use_smote', True),
                        key="smote_checkbox",
                        help="Équilibre les classes minoritaires"
                    )
                else:
                    st.success("✅ Classes équilibrées")
                    st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                        "SMOTE (optionnel)",
                        value=False,
                        key="smote_optional_checkbox"
                    )
            else:
                st.info("🔒 Variable cible requise")
                st.session_state.preprocessing_choices['use_smote'] = False
        
        elif task_type == 'clustering':
            st.subheader("🔍 Clustering")
            st.session_state.preprocessing_choices['pca_preprocessing'] = st.checkbox(
                "Réduction dimension (PCA)",
                value=st.session_state.preprocessing_choices.get('pca_preprocessing', False),
                help="Réduit le bruit pour données haute dimension"
            )

# Étape 3: Modèles
elif st.session_state.current_step == 3:
    st.header("🤖 Sélection des Modèles")
    task_type = st.session_state.get('task_type', 'classification')
    available_models = get_task_specific_models(task_type)
    
    if not available_models:
        st.error(f"❌ Aucun modèle disponible pour '{task_type}'")
        st.info("Action: Vérifiez le catalogue de modèles.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Modèles")
        if not st.session_state.selected_models_for_training:
            st.session_state.selected_models_for_training = get_default_models_for_task(task_type)
        
        selected_models = st.multiselect(
            f"Modèles {task_type}",
            options=available_models,
            default=st.session_state.selected_models_for_training,
            key="models_multiselect",
            help="Modèles à entraîner et comparer"
        )
        st.session_state.selected_models_for_training = selected_models
        
        if selected_models:
            st.success(f"✅ {len(selected_models)} modèles sélectionnés")
            with st.expander("📋 Détails des modèles", expanded=False):
                for model_name in selected_models:
                    model_config = MODEL_CATALOG[task_type].get(model_name, {})
                    st.write(f"**{model_name}**")
                    st.caption(f"• {model_config.get('description', 'N/A')}")
                    if task_type == 'clustering':
                        if model_name == 'KMeans':
                            st.caption("💡 Clusters sphériques, taille similaire")
                        elif model_name == 'DBSCAN':
                            st.caption("💡 Robustes au bruit, forme arbitraire")
                        elif model_name == 'GaussianMixture':
                            st.caption("💡 Probabiliste, taille variable")
        else:
            st.warning("⚠️ Aucun modèle sélectionné")
            st.info("Action: Sélectionnez au moins un modèle.")
    
    with col2:
        st.subheader("⚙️ Configuration")
        if task_type != 'clustering':
            test_split = st.slider(
                "Jeu de test (%)",
                min_value=10,
                max_value=40,
                value=st.session_state.get('test_split_for_ml_config', 20),
                step=5,
                key="test_split_slider",
                help="Données réservées pour l'évaluation"
            )
            st.session_state.test_split_for_ml_config = test_split
            st.caption(f"📊 {test_split}% test, {100-test_split}% entraînement")
        else:
            st.info("🔍 Clustering: 100% des données utilisées")
            st.session_state.test_split_for_ml_config = 0
        
        optimize_hp = st.checkbox(
            "Optimisation hyperparamètres",
            value=st.session_state.get('optimize_hp_for_ml_config', False),
            key="optimize_hp_checkbox",
            help="Recherche des meilleurs paramètres (plus long)"
        )
        st.session_state.optimize_hp_for_ml_config = optimize_hp
        
        if optimize_hp:
            st.warning("⏰ Temps d'entraînement +3-5x")
            optimization_method = st.selectbox(
                "Méthode",
                options=['Silhouette Score', 'Davies-Bouldin'] if task_type == 'clustering' else ['GridSearch', 'RandomSearch'],
                index=0,
                key="optimization_method_selector",
                help="Silhouette=qualité clusters, GridSearch=exhaustif, RandomSearch=rapide"
            )
            st.session_state.optimization_method = optimization_method
        
        n_features = len(st.session_state.feature_list_for_ml_config)
        estimated_seconds = estimate_training_time(df, len(selected_models), task_type, optimize_hp, n_features, st.session_state.preprocessing_choices.get('use_smote', False))
        st.info(f"⏱️ Temps estimé: {max(1, estimated_seconds // 60)} minute(s)")
        
        if selected_models:
            resource_check = check_system_resources(df, len(selected_models))
            if not resource_check["has_enough_resources"]:
                st.error("❌ Ressources insuffisantes")
                for issue in resource_check["issues"]:
                    st.error(f"• {issue}")
                st.info("Action: Réduisez le nombre de modèles ou libérez de la mémoire.")
            elif resource_check["warnings"]:
                st.warning("⚠️ Ressources limites")
                for warning in resource_check["warnings"]:
                    st.warning(f"• {warning}")

# Étape 4: Lancement
elif st.session_state.current_step == 4:
    st.header("🚀 Lancement de l'Expérimentation")
    task_type = st.session_state.get('task_type', 'classification')
    
    config_issues = []
    if task_type in ['classification', 'regression'] and not st.session_state.target_column_for_ml_config:
        config_issues.append("Variable cible non définie")
    if not st.session_state.feature_list_for_ml_config:
        config_issues.append("Aucune variable explicative sélectionnée")
    elif len(st.session_state.feature_list_for_ml_config) < 2 and task_type == 'clustering':
        config_issues.append("Minimum 2 variables pour clustering")
    if not st.session_state.selected_models_for_training:
        config_issues.append("Aucun modèle sélectionné")
    
    if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
        config_issues.append("Normalisation requise pour clustering")
    if len(st.session_state.feature_list_for_ml_config) > 30:
        config_issues.append("Trop de features - risque de surapprentissage")
    
    resource_check = check_system_resources(df, len(st.session_state.selected_models_for_training))
    config_issues.extend(resource_check["issues"])
    
    with st.expander("📋 Récapitulatif", expanded=True):
        if config_issues:
            st.error("❌ Configuration incomplète:")
            for issue in config_issues:
                st.error(f"• {issue}")
                st.info("Action: Revenez aux étapes précédentes pour corriger.")
        else:
            st.success("✅ Configuration valide")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Données**")
            st.write(f"• Type: {task_type.upper()}")
            if task_type != 'clustering':
                st.write(f"• Cible: `{st.session_state.target_column_for_ml_config}`")
            st.write(f"• Features: {len(st.session_state.feature_list_for_ml_config)}")
            if task_type != 'clustering':
                st.write(f"• Test: {st.session_state.test_split_for_ml_config}%")
            else:
                st.write("• Test: 0% (clustering)")
        with col2:
            st.markdown("**🤖 Modèles**")
            st.write(f"• Modèles: {len(st.session_state.selected_models_for_training)}")
            st.write(f"• Optimisation: {'✅' if st.session_state.optimize_hp_for_ml_config else '❌'}")
            if task_type == 'classification':
                st.write(f"• SMOTE: {'✅' if st.session_state.preprocessing_choices.get('use_smote') else '❌'}")
            st.write(f"• Normalisation: {'✅' if st.session_state.preprocessing_choices.get('scale_features') else '❌'}")
    
    col_launch, col_reset = st.columns([2, 1])
    with col_launch:
        launch_disabled = len(config_issues) > 0 or st.session_state.get('ml_training_in_progress', False)
        if st.button("🚀 Lancer", type="primary", width='stretch', disabled=launch_disabled):
            st.session_state.ml_training_in_progress = True
            st.session_state.ml_last_training_time = time.time()
            
            training_config = {
                'df': df,
                'target_column': st.session_state.target_column_for_ml_config,
                'model_names': st.session_state.selected_models_for_training,
                'task_type': task_type,
                'test_size': st.session_state.test_split_for_ml_config / 100 if task_type != 'clustering' else 0.0,
                'optimize': st.session_state.optimize_hp_for_ml_config,
                'feature_list': st.session_state.feature_list_for_ml_config,
                'use_smote': st.session_state.preprocessing_choices.get('use_smote', False),
                'preprocessing_choices': st.session_state.preprocessing_choices
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            try:
                status_text.text("📊 Préparation des données...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                n_models = len(st.session_state.selected_models_for_training)
                results = []
                for i, model_name in enumerate(st.session_state.selected_models_for_training):
                    status_text.text(f"🔧 Entraînement {i+1}/{n_models}: {model_name}")
                    progress_bar.progress(10 + int((i / n_models) * 80))
                    model_config = training_config.copy()
                    model_config['model_names'] = [model_name]
                    try:
                        model_results = train_models(**model_config)
                        results.extend(model_results)
                    except Exception as e:
                        logger.error(f"Erreur entraînement {model_name}: {e}")
                        results.append({
                            "model_name": model_name,
                            "metrics": {"error": str(e)},
                            "success": False,
                            "training_time": 0,
                            "X_sample": None,
                            "y_test": None,
                            "labels": None,
                            "feature_names": st.session_state.feature_list_for_ml_config
                        })
                    time.sleep(0.5)
                
                status_text.text("✅ Finalisation...")
                progress_bar.progress(95)
                time.sleep(0.5)
                
                elapsed_time = time.time() - st.session_state.ml_last_training_time
                status_text.text(f"✅ Terminé en {elapsed_time:.1f}s")
                progress_bar.progress(100)
                
                st.session_state.ml_results = results
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = 0
                
                successful_models = [r for r in results if r.get('success', False) and not r.get('metrics', {}).get('error')]
                with results_container.container():
                    st.success(f"✅ {len(successful_models)}/{len(results)} modèles réussis")
                    if successful_models:
                        key_metric = 'silhouette_score' if task_type == 'clustering' else 'r2' if task_type == 'regression' else 'accuracy'
                        best_model = max(successful_models, key=lambda x: x.get('metrics', {}).get(key_metric, -999))
                        best_score = best_model.get('metrics', {}).get(key_metric, 0)
                        st.info(f"🏆 Meilleur modèle: **{best_model['model_name']}** ({key_metric.title()}: {best_score:.3f})")
                    if st.button("📈 Voir résultats", width='stretch'):
                        st.switch_page("pages/3_📈_Évaluation_du_Modèle.py")
                
            except Exception as e:
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = st.session_state.get('ml_error_count', 0) + 1
                status_text.text("❌ Échec")
                progress_bar.progress(0)
                st.error(f"❌ Erreur: {str(e)[:200]}")
                st.info("Action: Vérifiez la configuration ou contactez le support.")
                logger.error(f"Training failed: {e}", exc_info=True)
    
    with col_reset:
        if st.button("🔄 Reset", width='stretch'):
            ml_keys_to_reset = [
                'target_column_for_ml_config', 'feature_list_for_ml_config',
                'selected_models_for_training', 'ml_results', 'task_type',
                'previous_task_type'
            ]
            for key in ml_keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_step = 1
            st.success("Configuration réinitialisée")
            st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    progress = (st.session_state.current_step / len(steps)) * 100
    st.caption(f"📊 Étape {st.session_state.current_step}/{len(steps)} ({progress:.0f}%)")
with col2:
    st.caption(f"🎯 {st.session_state.get('task_type', 'Non défini').upper()}")
with col3:
    st.caption(f"⏰ {time.strftime('%H:%M:%S')}")

# Navigation
col_prev, col_next = st.columns(2)
with col_prev:
    if st.session_state.current_step > 1:
        if st.button("◀️ Précédent", width='stretch'):
            st.session_state.current_step -= 1
            st.rerun()
with col_next:
    if st.session_state.current_step < 4:
        if st.button("Suivant ▶️", width='stretch', type="primary"):
            if st.session_state.current_step == 1:
                if st.session_state.task_type in ['classification', 'regression'] and not st.session_state.target_column_for_ml_config:
                    st.error("Veuillez sélectionner une variable cible")
                elif not st.session_state.feature_list_for_ml_config:
                    st.error("Veuillez sélectionner au moins une variable")
                else:
                    st.session_state.current_step += 1
                    st.rerun()
            else:
                st.session_state.current_step += 1
                st.rerun()

# Debug
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    with st.expander("🔍 Debug", expanded=False):
        st.json({
            "current_step": st.session_state.current_step,
            "task_type": st.session_state.get('task_type'),
            "target_column": st.session_state.get('target_column_for_ml_config'),
            "num_features": len(st.session_state.get('feature_list_for_ml_config', [])),
            "num_models": len(st.session_state.get('selected_models_for_training', [])),
            "test_split": st.session_state.get('test_split_for_ml_config'),
            "training_in_progress": st.session_state.get('ml_training_in_progress', False),
            "error_count": st.session_state.get('ml_error_count', 0)
        })