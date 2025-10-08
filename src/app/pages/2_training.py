import mlflow
import streamlit as st
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Any
from functools import wraps
import numpy as np

# Imports des modules ML
from src.models.catalog import MODEL_CATALOG
from src.data.data_analysis import get_target_and_task, detect_imbalance, auto_detect_column_types
from src.models.training import train_models, is_mlflow_available
from src.shared.logging import get_logger
from src.config.constants import VALIDATION_CONSTANTS, PREPROCESSING_CONSTANTS, TRAINING_CONSTANTS, MLFLOW_CONSTANTS

# Configuration
logger = get_logger(__name__)
st.set_page_config(page_title="Configuration ML", page_icon="⚙️", layout="wide")

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
            div[data-testid="stHorizontalBlock"] > div {display: flex; flex-wrap: nowrap; justify-content: center;}
            div[data-testid="stHorizontalBlock"] label {margin: 0 10px; white-space: nowrap;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        if is_mlflow_available():
            logger.info("MLflow est disponible pour le suivi des expériences")
            try:
                mlflow.set_tracking_uri(MLFLOW_CONSTANTS["TRACKING_URI"])
                logger.info("URI de suivi MLflow configuré avec succès")
                experiment = mlflow.get_experiment_by_name(MLFLOW_CONSTANTS["EXPERIMENT_NAME"])
                if experiment is None:
                    mlflow.create_experiment(MLFLOW_CONSTANTS["EXPERIMENT_NAME"])
                logger.info(f"Expérience MLflow '{MLFLOW_CONSTANTS['EXPERIMENT_NAME']}' configurée")
            except Exception as e:
                logger.warning(f"Échec de la configuration de l'URI MLflow : {e}")

setup_ml_config_environment()

def monitor_ml_operation(func):
    """Décorateur de monitoring"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > TRAINING_CONSTANTS["MONITOR_TIME_THRESHOLD"]:
                logger.warning(f"Opération ML {func.__name__} a pris {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Échec de l'opération ML {func.__name__} : {e}")
            raise
    return wrapper

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
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            low_corr_cols = [col for col in numeric_cols if col in features and corr_matrix[col].mean() < VALIDATION_CONSTANTS["FEATURE_CORR_THRESHOLD"]]
            validation["suggested_features"] = low_corr_cols[:10]

        for feature in features:
            if feature not in df.columns:
                validation["issues"].append(f"Colonne '{feature}' introuvable")
                continue
                
            if df[feature].std() == 0:
                validation["warnings"].append(f"'{feature}' est constante")
                continue
                
            missing_ratio = df[feature].isnull().mean()
            if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
                validation["warnings"].append(f"'{feature}' a {missing_ratio:.1%} de valeurs manquantes")
                continue
                
            if df[feature].nunique() == 1:
                validation["warnings"].append(f"'{feature}' n'a qu'une seule valeur unique")
                continue
                
            validation["valid_features"].append(feature)
        
        validation["is_valid"] = len(validation["valid_features"]) >= VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]
        if not validation["is_valid"]:
            validation["issues"].append(f"Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables numériques requises")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur de validation : {str(e)}")
    
    return validation

def estimate_training_time(df: pd.DataFrame, n_models: int, task_type: str, 
                         optimize_hp: bool, n_features: int, use_smote: bool = False) -> int:
    """Estime le temps d'entraînement de façon réaliste"""
    try:
        n_samples = len(df)
        base_complexity = (n_samples * n_features) / 1000
        time_multipliers = {
            'clustering': VALIDATION_CONSTANTS["CLUSTERING_TIME_MULTIPLIER"],
            'classification': VALIDATION_CONSTANTS["CLASSIFICATION_TIME_MULTIPLIER"] if use_smote else VALIDATION_CONSTANTS["CLASSIFICATION_BASE_MULTIPLIER"],
            'regression': VALIDATION_CONSTANTS["REGRESSION_TIME_MULTIPLIER"]
        }
        time_multiplier = time_multipliers.get(task_type, VALIDATION_CONSTANTS["DEFAULT_TIME_MULTIPLIER"])
        if optimize_hp:
            time_multiplier *= VALIDATION_CONSTANTS["OPTIMIZE_HP_MULTIPLIER"]
        estimated_seconds = base_complexity * n_models * time_multiplier
        return max(30, min(int(estimated_seconds), TRAINING_CONSTANTS["MAX_TRAINING_TIME"]))
    except Exception as e:
        logger.warning(f"Erreur estimation temps : {e}")
        return 60

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
        estimated_needed = df_memory * n_models * VALIDATION_CONSTANTS["MEMORY_MULTIPLIER"]
        available_memory = psutil.virtual_memory().available / (1024**2)
        check_result["available_memory_mb"] = available_memory
        check_result["estimated_needed_mb"] = estimated_needed
        
        if estimated_needed > available_memory:
            check_result["has_enough_resources"] = False
            check_result["issues"].append(f"Mémoire insuffisante : {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        elif estimated_needed > available_memory * VALIDATION_CONSTANTS["MEMORY_WARNING_THRESHOLD"]:
            check_result["warnings"].append(f"Mémoire limite : {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > TRAINING_CONSTANTS["HIGH_CPU_THRESHOLD"]:
            check_result["warnings"].append(f"CPU élevé : {cpu_percent:.1f}%")
            
    except Exception as e:
        check_result["warnings"].append("Échec vérification ressources")
        logger.warning(f"Erreur vérification ressources : {e}")
    
    return check_result

@st.cache_data(ttl=300, max_entries=3)
def validate_dataframe_for_ml(df: pd.DataFrame) -> Dict[str, Any]:
    """Valide le DataFrame pour ML"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "min_rows_required": VALIDATION_CONSTANTS["MIN_ROWS_REQUIRED"],
        "min_cols_required": VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"],
        "stats": {}
    }
    
    try:
        if df is None or df.empty:
            validation["is_valid"] = False
            validation["issues"].append("DataFrame vide ou non chargé")
            return validation
        
        n_rows, n_cols = df.shape
        validation["stats"] = {"n_rows": n_rows, "n_cols": n_cols}
        
        if n_rows < VALIDATION_CONSTANTS["MIN_ROWS_REQUIRED"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant : {n_rows} lignes (minimum : {VALIDATION_CONSTANTS['MIN_ROWS_REQUIRED']})")
        
        if n_cols < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant : {n_cols} colonnes (minimum : {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']})")
        
        missing_ratio = df.isnull().mean().max()
        if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop de valeurs manquantes : {missing_ratio:.1%}")
        elif missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"] * VALIDATION_CONSTANTS["MISSING_WARNING_THRESHOLD"]:
            validation["warnings"].append(f"Beaucoup de valeurs manquantes : {missing_ratio:.1%}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if len(constant_cols) == len(numeric_cols):
                validation["warnings"].append("Toutes les colonnes numériques sont constantes")
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        validation["stats"]["memory_mb"] = memory_usage
        if memory_usage > TRAINING_CONSTANTS["MEMORY_LIMIT_MB"] / 4:
            validation["warnings"].append(f"Dataset volumineux : {memory_usage:.1f} MB")
            
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation : {str(e)}")
        logger.error(f"Erreur validation DataFrame : {e}")
    
    return validation

def initialize_ml_config_state():
    """Initialise l'état de configuration ML"""
    defaults = {
        'target_column_for_ml_config': None,
        'feature_list_for_ml_config': [],
        'preprocessing_choices': {
            'numeric_imputation': PREPROCESSING_CONSTANTS["NUMERIC_IMPUTATION_DEFAULT"],
            'categorical_imputation': PREPROCESSING_CONSTANTS["CATEGORICAL_IMPUTATION_DEFAULT"],
            'use_smote': False,
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'pca_preprocessing': False,
            'encoding_method': PREPROCESSING_CONSTANTS["ENCODING_METHOD"],
            'scaling_method': PREPROCESSING_CONSTANTS["SCALING_METHOD"]
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
        'previous_task_type': None,
        'ml_results': None,
        'mlflow_runs': None
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
            if n_classes > TRAINING_CONSTANTS["MAX_CLASSES"]:
                return {"task_type": "unknown", "n_classes": n_classes, "error": f"Trop de classes ({n_classes})"}
        
        return {"task_type": task_type, "target_type": target_type, "n_classes": n_classes, "error": None}
    except Exception as e:
        logger.error(f"Échec détection type tâche : {e}")
        return {"task_type": "unknown", "n_classes": 0, "error": str(e)}

def get_task_specific_models(task_type: str) -> List[str]:
    """Retourne les modèles disponibles pour une tâche"""
    try:
        return list(MODEL_CATALOG.get(task_type, {}).keys())
    except Exception as e:
        logger.error(f"Erreur récupération modèles pour {task_type} : {e}")
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
        st.info(f"**Critères requis**:\n- Minimum {VALIDATION_CONSTANTS['MIN_ROWS_REQUIRED']} lignes\n- Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} colonnes\n- Moins de {VALIDATION_CONSTANTS['MAX_MISSING_RATIO']*100:.0f}% de valeurs manquantes")
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
    color = "🟢" if sys_memory < 70 else "🟡" if sys_memory < TRAINING_CONSTANTS["HIGH_MEMORY_THRESHOLD"] else "🔴"
    st.metric(f"{color} RAM Sys", f"{sys_memory:.0f}%")

st.markdown("---")

# Navigation par étapes
steps = ["🎯 Cible", "🔧 Préprocess", "🤖 Modèles", "🚀 Lancement"]
st.radio("Étapes", steps, index=st.session_state.current_step - 1, horizontal=True, key="step_selector")
st.session_state.current_step = steps.index(st.session_state.get('step_selector', steps[0])) + 1

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
        "Type de problème",
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
            [col for col in df.columns if df[col].nunique() <= TRAINING_CONSTANTS["MAX_CLASSES"] or not pd.api.types.is_numeric_dtype(df[col])]
            if selected_task_type == 'classification' else
            [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > VALIDATION_CONSTANTS["MIN_UNIQUE_VALUES"]]
        )
        
        if not available_targets:
            st.error("❌ Aucune variable cible appropriée trouvée")
            st.info(f"Classification : ≤{TRAINING_CONSTANTS['MAX_CLASSES']} valeurs uniques\nRégression : numérique, >{VALIDATION_CONSTANTS['MIN_UNIQUE_VALUES']} valeurs uniques")
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
                    st.error(f"❌ Erreur : {task_info['error']}")
                    st.info("Action : Sélectionnez une autre colonne ou vérifiez les données.")
                else:
                    if selected_task_type == "classification":
                        st.success(f"✅ **Classification** ({task_info['n_classes']} classes)")
                        class_dist = df[target_column].value_counts()
                        if len(class_dist) <= 10:
                            st.bar_chart(class_dist, height=300)
                            st.caption(f"Distribution des classes")
                        imbalance_info = detect_imbalance(df, target_column)
                        if imbalance_info.get("is_imbalanced", False):
                            st.warning(f"⚠️ Déséquilibre (ratio : {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
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
                        col for col in column_types.get('categorical', []) if df[col].nunique() <= VALIDATION_CONSTANTS["MAX_CATEGORICAL_UNIQUE"]
                    ]
                    recommended_features = [col for col in recommended_features if col != target_column and col in all_features][:TRAINING_CONSTANTS["MAX_FEATURES"]]
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
                if len(st.session_state.feature_list_for_ml_config) > TRAINING_CONSTANTS["MAX_FEATURES"]:
                    st.warning("⚠️ Nombre élevé de features - risque de surapprentissage")
                    st.info("Action : Réduisez le nombre de features ou activez PCA.")
            else:
                st.warning("⚠️ Aucune feature sélectionnée")
                st.info("Action : Sélectionnez au moins une variable.")
        else:
            st.error("❌ Aucune feature disponible")
            st.info("Action : Vérifiez votre dataset.")
    
    else:  # Clustering
        st.session_state.target_column_for_ml_config = None
        st.success("✅ **Clustering Non Supervisé**")
        st.info("🔍 Le modèle identifiera des groupes naturels dans les données.")
        
        all_numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        if not all_numeric_features:
            st.error("❌ Aucune variable numérique pour le clustering")
            st.info("Action : Ajoutez des variables numériques au dataset.")
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
                    st.info(f"💡 Suggestion : {', '.join(validation_result['suggested_features'][:5])}")
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
                    st.info(f"💡 Suggestion : {', '.join(validation_result['suggested_features'][:5])}")
                if validation_result["warnings"]:
                    with st.expander("⚠️ Avertissements", expanded=True):
                        for warning in validation_result["warnings"]:
                            st.warning(f"• {warning}")
            
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées")
                if len(st.session_state.feature_list_for_ml_config) < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
                    st.warning(f"⚠️ Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables pour clustering")
                    st.info("Action : Ajoutez des variables numériques.")
                elif len(st.session_state.feature_list_for_ml_config) > TRAINING_CONSTANTS["MAX_FEATURES"]:
                    st.warning("⚠️ Nombre élevé de variables - risque de malédiction dimensionnelle")
                    st.info("Action : Activez PCA ou réduisez les variables.")
                with st.expander("📈 Aperçu statistiques", expanded=False):
                    st.dataframe(df[st.session_state.feature_list_for_ml_config].describe().style.format("{:.3f}"), width="stretch")
            else:
                st.warning("⚠️ Aucune variable sélectionnée")
                st.info("Action : Sélectionnez des variables numériques.")

# Étape 2: Prétraitement
elif st.session_state.current_step == 2:
    st.header("🔧 Configuration du Prétraitement")
    task_type = st.session_state.get('task_type', 'classification')
    
    st.info(f"**Pipeline pour {task_type.upper()}** : Transformations appliquées séparément sur train/validation pour éviter le data leakage.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧩 Valeurs Manquantes")
        st.session_state.preprocessing_choices['numeric_imputation'] = st.selectbox(
            "Variables numériques",
            options=['mean', 'median', 'constant', 'knn'],
            index=['mean', 'median', 'constant', 'knn'].index(st.session_state.preprocessing_choices.get('numeric_imputation', PREPROCESSING_CONSTANTS["NUMERIC_IMPUTATION_DEFAULT"])),
            key='numeric_imputation_selector',
            help="mean=moyenne, median=médiane, constant=0, knn=k-voisins"
        )
        st.session_state.preprocessing_choices['categorical_imputation'] = st.selectbox(
            "Variables catégorielles",
            options=['most_frequent', 'constant'],
            index=['most_frequent', 'constant'].index(st.session_state.preprocessing_choices.get('categorical_imputation', PREPROCESSING_CONSTANTS["CATEGORICAL_IMPUTATION_DEFAULT"])),
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

        # Analyse des colonnes pour nettoyage
        if st.session_state.preprocessing_choices['remove_constant_cols'] or st.session_state.preprocessing_choices['remove_identifier_cols']:
            with st.spinner("Analyse des colonnes..."):
                column_types = auto_detect_column_types(df)
                # Filtrer les colonnes numériques avec tous les types numériques
                numeric_cols = df.select_dtypes(include='number').columns
                constant_cols = []
                if len(numeric_cols) > 0:
                    constant_cols = [col for col in numeric_cols if df[col].std() == 0]
                else:
                    st.warning("⚠️ Aucune colonne numérique détectée pour vérifier les colonnes constantes.")
                    logger.warning("Aucune colonne numérique pour la vérification des colonnes constantes.")
                
                identifier_cols = [col for col in df.columns if df[col].nunique() == len(df)]
                if constant_cols or identifier_cols:
                    st.info(f"🧹 Nettoyage : {len(constant_cols)} colonnes constantes, {len(identifier_cols)} colonnes identifiantes détectées")
                else:
                    st.info("🧹 Aucune colonne constante ou identifiant détectée.")
    
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

        if task_type in ['classification', 'regression']:
            st.subheader("🔍 Réduction Dimensionnelle")
            st.session_state.preprocessing_choices['pca_preprocessing'] = st.checkbox(
                "Réduction dimension (PCA)",
                value=st.session_state.preprocessing_choices.get('pca_preprocessing', False),
                key="pca_preprocessing_checkbox_supervised",
                help="Réduit le bruit pour données haute dimension"
            )
            if len(st.session_state.feature_list_for_ml_config) > TRAINING_CONSTANTS["MAX_FEATURES"]:
                st.info("💡 PCA recommandé pour réduire le nombre de features.")

        if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
            st.error("❌ Normalisation critique pour le clustering !")
            st.info("Action : Activez la normalisation pour de meilleurs résultats.")
        
        if task_type == 'classification':
            st.subheader("⚖️ Déséquilibre")
            if st.session_state.target_column_for_ml_config:
                imbalance_info = detect_imbalance(df, st.session_state.target_column_for_ml_config)
                if imbalance_info.get("is_imbalanced", False):
                    st.warning(f"📉 Déséquilibre détecté (ratio : {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                    st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                        "Activer SMOTE",
                        value=st.session_state.preprocessing_choices.get('use_smote', True),
                        key="smote_checkbox",
                        help="Génère des données synthétiques pour équilibrer les classes minoritaires"
                    )
                    if st.session_state.preprocessing_choices['use_smote']:
                        # Ajouter des options de configuration pour SMOTE
                        with st.expander("⚙️ Paramètres SMOTE", expanded=False):
                            st.session_state.preprocessing_choices['smote_k_neighbors'] = st.number_input(
                                "Nombre de voisins (k)",
                                min_value=1,
                                max_value=20,
                                value=st.session_state.preprocessing_choices.get('smote_k_neighbors', 5),
                                step=1,
                                key="smote_k_neighbors_input",
                                help="Nombre de voisins utilisés pour générer les samples synthétiques"
                            )
                            st.session_state.preprocessing_choices['smote_sampling_strategy'] = st.selectbox(
                                "Stratégie d'échantillonnage",
                                options=['auto', 'minority', 'not minority', 'not majority', 'all'],
                                index=['auto', 'minority', 'not minority', 'not majority', 'all'].index(
                                    st.session_state.preprocessing_choices.get('smote_sampling_strategy', 'auto')
                                ),
                                key="smote_sampling_strategy_select",
                                help="Détermine quelles classes rééquilibrer (auto = classe minoritaire)"
                            )
                            # Validation du nombre de samples dans la classe minoritaire
                            min_class_count = min(df[st.session_state.target_column_for_ml_config].value_counts())
                            if min_class_count < st.session_state.preprocessing_choices['smote_k_neighbors']:
                                st.warning(
                                    f"⚠️ Classe minoritaire trop petite ({min_class_count} samples) pour k={st.session_state.preprocessing_choices['smote_k_neighbors']}. "
                                    "Réduisez k ou collectez plus de données."
                                )
                else:
                    st.success("✅ Classes équilibrées")
                    st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                        "SMOTE (optionnel)",
                        value=st.session_state.preprocessing_choices.get('use_smote', False),
                        key="smote_optional_checkbox",
                        help="Génère des données synthétiques même si les classes sont équilibrées"
                    )
                    if st.session_state.preprocessing_choices['use_smote']:
                        with st.expander("⚙️ Paramètres SMOTE", expanded=False):
                            st.session_state.preprocessing_choices['smote_k_neighbors'] = st.number_input(
                                "Nombre de voisins (k)",
                                min_value=1,
                                max_value=20,
                                value=st.session_state.preprocessing_choices.get('smote_k_neighbors', 5),
                                step=1,
                                key="smote_k_neighbors_input_optional",
                                help="Nombre de voisins utilisés pour générer les samples synthétiques"
                            )
                            st.session_state.preprocessing_choices['smote_sampling_strategy'] = st.selectbox(
                                "Stratégie d'échantillonnage",
                                options=['auto', 'minority', 'not minority', 'not majority', 'all'],
                                index=['auto', 'minority', 'not minority', 'not majority', 'all'].index(
                                    st.session_state.preprocessing_choices.get('smote_sampling_strategy', 'auto')
                                ),
                                key="smote_sampling_strategy_select_optional",
                                help="Détermine quelles classes rééquilibrer (auto = classe minoritaire)"
                            )
                            min_class_count = min(df[st.session_state.target_column_for_ml_config].value_counts())
                            if min_class_count < st.session_state.preprocessing_choices['smote_k_neighbors']:
                                st.warning(
                                    f"⚠️ Classe minoritaire trop petite ({min_class_count} samples) pour k={st.session_state.preprocessing_choices['smote_k_neighbors']}. "
                                    "Réduisez k ou collectez plus de données."
                                )
            else:
                st.info("🔒 Variable cible requise pour activer SMOTE")
                st.session_state.preprocessing_choices['use_smote'] = False
        elif task_type == 'clustering':
            st.subheader("🔍 Clustering")
            st.session_state.preprocessing_choices['pca_preprocessing'] = st.checkbox(
                "Réduction dimension (PCA)",
                value=st.session_state.preprocessing_choices.get('pca_preprocessing', False),
                key="pca_preprocessing_checkbox",
                help="Réduit le bruit pour données haute dimension"
            )


# Dans Étape 3 : Sélection des Modèles
elif st.session_state.current_step == 3:
    st.header("🤖 Sélection des Modèles")
    task_type = st.session_state.get('task_type', 'classification')
    available_models = get_task_specific_models(task_type)
    
    if not available_models:
        st.error(f"❌ Aucun modèle disponible pour '{task_type}'")
        st.info("Action : Vérifiez le catalogue de modèles.")
        st.stop()
    
    # Sélection des modèles et configuration
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
            if len(selected_models) > 5:
                st.warning("⚠️ Maximum 5 modèles recommandés pour éviter une surcharge système.")
                st.session_state.selected_models_for_training = selected_models[:5]
            st.success(f"✅ {len(st.session_state.selected_models_for_training)} modèles sélectionnés")
            with st.expander("📋 Détails des modèles", expanded=False):
                for model_name in selected_models:
                    model_config = MODEL_CATALOG[task_type].get(model_name, {})
                    st.write(f"**{model_name}**")
                    st.caption(f"• {model_config.get('description', 'Description non disponible')}")
        else:
            st.warning("⚠️ Aucun modèle sélectionné")
            st.info("Action : Sélectionnez au moins un modèle.")
            
    # Configuration supplémentaire
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
            st.info("🔍 Clustering : 100% des données utilisées")
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
            st.session_state.preprocessing_choices['optimization_method'] = st.selectbox(
                "Méthode",
                options=['Silhouette Score', 'Davies-Bouldin'] if task_type == 'clustering' else ['GridSearch', 'RandomSearch'],
                index=0,
                key="optimization_method_selector",
                help="Silhouette=qualité clusters, Davies-Bouldin=compacité, GridSearch=exhaustif, RandomSearch=rapide"
            )
        
        n_features = len(st.session_state.feature_list_for_ml_config)
        estimated_seconds = estimate_training_time(df, len(selected_models), task_type, optimize_hp, n_features, st.session_state.preprocessing_choices.get('use_smote', False))
        st.info(f"⏱️ Temps estimé : {max(1, estimated_seconds // 60)} minute(s)")
        
        if selected_models:
            resource_check = check_system_resources(df, len(selected_models))
            if not resource_check["has_enough_resources"]:
                st.error("❌ Ressources insuffisantes")
                for issue in resource_check["issues"]:
                    st.error(f"• {issue}")
                st.info("Action : Réduisez le nombre de modèles ou libérez de la mémoire.")
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
    elif len(st.session_state.feature_list_for_ml_config) < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"] and task_type == 'clustering':
        config_issues.append(f"Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables pour clustering")
    if not st.session_state.selected_models_for_training:
        config_issues.append("Aucun modèle sélectionné")
    
    if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
        config_issues.append("Normalisation requise pour clustering")
    if len(st.session_state.feature_list_for_ml_config) > TRAINING_CONSTANTS["MAX_FEATURES"]:
        config_issues.append("Trop de features - risque de surapprentissage")
    
    resource_check = check_system_resources(df, len(st.session_state.selected_models_for_training))
    config_issues.extend(resource_check["issues"])
    
    with st.expander("📋 Récapitulatif", expanded=True):
        if config_issues:
            st.error("❌ Configuration incomplète :")
            for issue in config_issues:
                st.error(f"• {issue}")
                st.info("Action : Revenez aux étapes précédentes pour corriger.")
        else:
            st.success("✅ Configuration valide")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Données**")
            st.write(f"• Type : {task_type.upper()}")
            if task_type != 'clustering':
                st.write(f"• Cible : `{st.session_state.target_column_for_ml_config}`")
            st.write(f"• Features : {len(st.session_state.feature_list_for_ml_config)}")
            if task_type != 'clustering':
                st.write(f"• Test : {st.session_state.test_split_for_ml_config}%")
            else:
                st.write("• Test : 0% (clustering)")
        with col2:
            st.markdown("**🤖 Modèles**")
            st.write(f"• Modèles : {len(st.session_state.selected_models_for_training)}")
            st.write(f"• Optimisation : {'✅' if st.session_state.optimize_hp_for_ml_config else '❌'}")
            if task_type == 'classification':
                st.write(f"• SMOTE : {'✅' if st.session_state.preprocessing_choices.get('use_smote') else '❌'}")
            st.write(f"• Normalisation : {'✅' if st.session_state.preprocessing_choices.get('scale_features') else '❌'}")
    
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
                    status_text.text(f"🔧 Entraînement {i+1}/{n_models} : {model_name}")
                    progress_bar.progress(10 + int((i / n_models) * 80))
                    model_config = training_config.copy()
                    model_config['model_names'] = [model_name]
                    try:
                        model_results = train_models(**model_config)
                        results.extend(model_results)
                    except Exception as e:
                        logger.error(f"Erreur entraînement {model_name} : {e}")
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
                
                # Stocker les runs MLflow avec gestion d'erreur
                try:
                    import mlflow
                    runs = mlflow.search_runs()
                    st.session_state.mlflow_runs = runs.to_dict(orient='records')
                    logger.info(f"{len(runs)} runs MLflow stockés dans session_state")
                except ImportError:
                    logger.info("MLflow n'est pas installé, tracking désactivé")
                    st.session_state.mlflow_runs = []
                except Exception as e:
                    logger.error(f"Échec récupération runs MLflow : {e}")
                    st.session_state.mlflow_runs = []
                                        
                successful_models = [r for r in results if r.get('success', False) and not r.get('metrics', {}).get('error')]
                with results_container.container():
                    st.success(f"✅ {len(successful_models)}/{len(results)} modèles réussis")
                    if successful_models:
                        key_metric = 'silhouette_score' if task_type == 'clustering' else 'r2' if task_type == 'regression' else 'accuracy'
                        best_model = max(successful_models, key=lambda x: x.get('metrics', {}).get(key_metric, -999))
                        best_score = best_model.get('metrics', {}).get(key_metric, 0)
                        st.info(f"🏆 Meilleur modèle : {best_model.get('model_name')}")
                    if st.button("📈 Voir résultats détaillés", width='stretch'):
                        st.switch_page("pages/3_evaluation.py")
                
            except Exception as e:
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = st.session_state.get('ml_error_count', 0) + 1
                status_text.text("❌ Échec")
                progress_bar.progress(0)
                st.error(f"❌ Erreur: {str(e)}")
                st.info("Action: Vérifiez la configuration ou contactez le support.")
                logger.error(f"Training failed: {e}", exc_info=True)
    
    with col_reset:
        if st.button("🔄 Reset", width='stretch'):
            ml_keys_to_reset = [
                'target_column_for_ml_config', 'feature_list_for_ml_config',
                'selected_models_for_training', 'ml_results', 'task_type', 
                'previous_task_type', 'test_split_for_ml_config', 'optimize_hp_for_ml_config',
                'preprocessing_choices', 'ml_training_in_progress', 'ml_last_training_time', 
                'ml_error_count', 'mlflow_runs', 'model_performance_history'
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