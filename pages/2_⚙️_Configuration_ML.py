import streamlit as st
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Any
from functools import wraps

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

# NOUVELLE FONCTION : Validation des features pour clustering
def validate_clustering_features(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Valide que les features sont adaptées au clustering"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "valid_features": []
    }
    
    try:
        for feature in features:
            if feature not in df.columns:
                validation["issues"].append(f"Colonne '{feature}' non trouvée")
                continue
                
            # Vérifier si constante
            if df[feature].std() == 0:
                validation["warnings"].append(f"'{feature}' est constante")
                continue
                
            # Vérifier valeurs manquantes
            missing_ratio = df[feature].isnull().mean()
            if missing_ratio > 0.5:
                validation["warnings"].append(f"'{feature}' a {missing_ratio:.1%} de valeurs manquantes")
                continue
                
            # Vérifier variance acceptable
            if df[feature].nunique() == 1:
                validation["warnings"].append(f"'{feature}' n'a qu'une seule valeur unique")
                continue
                
            validation["valid_features"].append(feature)
        
        validation["is_valid"] = len(validation["valid_features"]) >= 2
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
    
    return validation

# NOUVELLE FONCTION : Estimation temps plus précise
def estimate_training_time(df: pd.DataFrame, n_models: int, task_type: str, 
                          optimize_hp: bool, n_features: int, use_smote: bool = False) -> int:
    """Estime le temps d'entraînement de façon plus réaliste"""
    try:
        n_samples = len(df)
        
        # Complexité de base basée sur taille données
        base_complexity = (n_samples * n_features) / 1000
        
        # Multiplicateurs selon les paramètres
        time_multipliers = {
            'clustering': 1.2,
            'classification': 1.5 if use_smote else 1.3,
            'regression': 1.4
        }
        
        if optimize_hp:
            time_multipliers = {k: v * 3.5 for k, v in time_multipliers.items()}
        
        time_multiplier = time_multipliers.get(task_type, 1.5)
        
        # Estimation en secondes
        estimated_seconds = base_complexity * n_models * time_multiplier
        
        # Bornes raisonnables
        min_seconds = 30  # Minimum 30 secondes
        max_seconds = 3600  # Maximum 1 heure
        
        estimated_seconds = max(min_seconds, min(estimated_seconds, max_seconds))
        
        return int(estimated_seconds)
        
    except Exception as e:
        logger.warning(f"Erreur estimation temps: {e}")
        return 60  # Valeur par défaut

# NOUVELLE FONCTION : Vérification ressources système
def check_system_resources(df: pd.DataFrame, n_models: int) -> Dict[str, Any]:
    """Vérifie si le système a assez de ressources pour l'entraînement"""
    check_result = {
        "has_enough_resources": True,
        "issues": [],
        "warnings": [],
        "available_memory_mb": 0,
        "estimated_needed_mb": 0
    }
    
    try:
        # Estimation mémoire nécessaire
        df_memory = df.memory_usage(deep=True).sum() / (1024**2)
        estimated_needed = df_memory * n_models * 3  # Buffer 3x
        
        # Mémoire disponible
        available_memory = psutil.virtual_memory().available / (1024**2)
        
        check_result["available_memory_mb"] = available_memory
        check_result["estimated_needed_mb"] = estimated_needed
        
        # Vérifications
        if estimated_needed > available_memory:
            check_result["has_enough_resources"] = False
            check_result["issues"].append(
                f"Mémoire insuffisante (nécessaire: {estimated_needed:.0f}MB, disponible: {available_memory:.0f}MB)"
            )
        elif estimated_needed > available_memory * 0.7:
            check_result["warnings"].append(
                f"Mémoire limite (nécessaire: {estimated_needed:.0f}MB, disponible: {available_memory:.0f}MB)"
            )
        
        # Vérification CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            check_result["warnings"].append(f"CPU élevé: {cpu_percent:.1f}%")
            
    except Exception as e:
        logger.warning(f"Erreur vérification ressources: {e}")
        check_result["warnings"].append("Impossible de vérifier les ressources système")
    
    return check_result

# Validation sécurisée du DataFrame
@st.cache_data(ttl=300, max_entries=3)
def validate_dataframe_for_ml(df: pd.DataFrame) -> Dict[str, Any]:
    """Valide le DataFrame pour l'analyse ML avec critères stricts"""
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
        
        # Vérifications dimensionnelles
        if n_rows < validation["min_rows_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant: {n_rows} lignes (minimum: {validation['min_rows_required']})")
        
        if n_cols < validation["min_cols_required"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant: {n_cols} colonnes (minimum: {validation['min_cols_required']})")
        
        # Analyse qualité des données
        try:
            missing_ratio = df.isnull().mean().max()
            if missing_ratio > 0.95:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop de valeurs manquantes: {missing_ratio:.1%}")
            elif missing_ratio > 0.7:
                validation["warnings"].append(f"Beaucoup de valeurs manquantes: {missing_ratio:.1%}")
            
            # Vérification variance
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                low_variance_cols = []
                for col in numeric_cols:
                    if df[col].std() == 0:
                        low_variance_cols.append(col)
                
                if len(low_variance_cols) == len(numeric_cols):
                    validation["warnings"].append("Toutes les colonnes numériques sont constantes")
            
        except Exception as e:
            validation["warnings"].append(f"Analyse qualité échouée: {str(e)[:50]}")
        
        # Vérification mémoire
        try:
            memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
            validation["stats"]["memory_mb"] = memory_usage
            if memory_usage > 1000:  # 1GB
                validation["warnings"].append(f"Dataset volumineux: {memory_usage:.1f} MB")
        except:
            validation["warnings"].append("Calcul mémoire impossible")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"DataFrame validation error: {e}")
    
    return validation

# Initialisation robuste de l'état 
def initialize_ml_config_state():
    """Initialise l'état de configuration ML de façon robuste"""
    defaults = {
        'target_column_for_ml_config': None,
        'feature_list_for_ml_config': [],
        'preprocessing_choices': {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'use_smote': False,
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True
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
        'previous_task_type': None  # NOUVEAU : pour détection changement
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@monitor_ml_operation
def safe_get_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Version sécurisée de la détection du type de tâche - AMÉLIORÉE"""
    try:
        if not target_column or target_column not in df.columns:
            return {"task_type": "unknown", "n_classes": 0, "error": "Colonne cible invalide"}
        
        # Vérifier si c'est un identifiant (valeurs uniques)
        if df[target_column].nunique() == len(df):
            return {
                "task_type": "unknown", 
                "n_classes": df[target_column].nunique(),
                "error": "Variable cible a des valeurs uniques pour chaque ligne (probable identifiant)"
            }
        
        # Appel sécurisé à get_target_and_task
        result_dict = get_target_and_task(df, target_column)
        
        if not isinstance(result_dict, dict):
            return {"task_type": "unknown", "n_classes": 0, "error": "Résultat invalide"}
        
        task_type = result_dict.get("task", "unknown")
        target_type = result_dict.get("target_type", "unknown")
        
        # Calcul sécurisé du nombre de classes
        n_classes = 0
        try:
            if task_type == "classification":
                n_classes = df[target_column].nunique()
                if n_classes > 100:
                    return {
                        "task_type": "unknown",
                        "n_classes": n_classes,
                        "error": f"Trop de classes ({n_classes}) pour une classification standard"
                    }
        except Exception as e:
            logger.debug(f"Class count calculation failed: {e}")
        
        return {
            "task_type": task_type,
            "target_type": target_type,
            "n_classes": n_classes,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Task type detection failed: {e}")
        return {"task_type": "unknown", "n_classes": 0, "error": str(e)}

def get_task_specific_models(task_type: str) -> List[str]:
    """Retourne les modèles disponibles pour un type de tâche spécifique"""
    try:
        if task_type == 'clustering':
            return list(MODEL_CATALOG.get('clustering', {}).keys())
        elif task_type == 'regression':
            return list(MODEL_CATALOG.get('regression', {}).keys())
        else:  # classification par défaut
            return list(MODEL_CATALOG.get('classification', {}).keys())
    except Exception as e:
        logger.error(f"Error getting models for {task_type}: {e}")
        return []

def get_default_models_for_task(task_type: str) -> List[str]:
    """Retourne les modèles par défaut pour chaque type de tâche"""
    default_models = {
        'classification': ['RandomForest', 'XGBoost', 'LogisticRegression'],
        'regression': ['RandomForest', 'XGBoost', 'LinearRegression'],
        'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture']
    }
    available_models = get_task_specific_models(task_type)
    return [model for model in default_models.get(task_type, []) if model in available_models]

# Interface principale
st.title("⚙️ Configuration Détaillée de l'Expériences")

# Vérification des données avec validation stricte
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("📊 Aucun dataset chargé")
    st.info("Chargez d'abord un dataset depuis la page d'accueil pour configurer l'expérience ML.")
    if st.button("🏠 Retour à l'accueil"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state.df

# Validation stricte du DataFrame
validation_result = validate_dataframe_for_ml(df)
if not validation_result["is_valid"]:
    st.error("❌ Dataset non compatible avec l'analyse ML")
    with st.expander("🔍 Détails des problèmes", expanded=True):
        for issue in validation_result["issues"]:
            st.error(f"• {issue}")
    
    st.info("""
    **Critères requis pour l'analyse ML:**
    - Minimum 50 lignes de données
    - Minimum 2 colonnes
    - Moins de 95% de valeurs manquantes par colonne
    """)
    
    if st.button("🔄 Revérifier"):
        st.rerun()
    st.stop()

# Avertissements non-bloquants
if validation_result["warnings"]:
    with st.expander("⚠️ Avertissements qualité données", expanded=False):
        for warning in validation_result["warnings"]:
            st.warning(f"• {warning}")

# Initialisation de l'état
initialize_ml_config_state()

# Métriques du dataset avec design amélioré
st.markdown("### 📊 Aperçu du Dataset")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    n_rows = validation_result["stats"]["n_rows"]
    st.metric("📏 Lignes", f"{n_rows:,}")

with col2:
    n_cols = validation_result["stats"]["n_cols"]
    st.metric("📋 Colonnes", f"{n_cols}")

with col3:
    memory_mb = validation_result["stats"].get("memory_mb", 0)
    if memory_mb > 0:
        st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
    else:
        st.metric("💾 Mémoire", "N/A")

with col4:
    missing_pct = df.isnull().mean().mean() * 100
    st.metric("🕳️ Manquant", f"{missing_pct:.1f}%")

with col5:
    try:
        sys_memory = psutil.virtual_memory().percent
        color = "🔴" if sys_memory > 85 else "🟡" if sys_memory > 70 else "🟢"
        st.metric(f"{color} RAM Sys", f"{sys_memory:.0f}%")
    except:
        st.metric("🔧 RAM Sys", "N/A")

st.markdown("---")

# Navigation par étapes avec état persistant
steps = ["🎯 Cible", "🔧 Préprocess", "🤖 Modèles", "🚀 Lancement"]
selected_step = st.radio("Étapes de configuration", steps, index=st.session_state.current_step - 1, horizontal=True)
st.session_state.current_step = steps.index(selected_step) + 1

# Étape 1: Configuration de la cible
if st.session_state.current_step == 1:
    st.header("🎯 Configuration de la Tâche et Cible")
    
    # Sélection du type de tâche avec état stable
    task_options = ["Classification Supervisée", "Régression Supervisée", "Clustering Non Supervisé"]
    task_descriptions = {
        "Classification Supervisée": "Prédire des catégories (ex: spam/non-spam)",
        "Régression Supervisée": "Prédire des valeurs numériques (ex: prix, score)", 
        "Clustering Non Supervisé": "Découvrir des groupes naturels dans les données"
    }
    
    # Déterminer l'index initial basé sur l'état actuel
    if st.session_state.task_type == 'clustering':
        current_task_idx = 2
    elif st.session_state.task_type == 'regression':
        current_task_idx = 1
    else:
        current_task_idx = 0
    
    task_selection = st.selectbox(
        "Type de problème ML à résoudre",
        options=task_options,
        index=current_task_idx,
        key="ml_task_selection_stable",
        help="Sélectionnez le type d'apprentissage adapté à vos données"
    )
    
    # Afficher la description
    st.info(f"**{task_selection}** - {task_descriptions[task_selection]}")
    
    # Mapper la sélection au type de tâche
    task_mapping = {
        "Classification Supervisée": "classification",
        "Régression Supervisée": "regression", 
        "Clustering Non Supervisé": "clustering"
    }
    
    selected_task_type = task_mapping[task_selection]
    
    # NOUVEAU : Reset automatique quand le type de tâche change
    if st.session_state.previous_task_type != selected_task_type:
        st.info("🔄 Type de tâche modifié - réinitialisation des sélections...")
        
        # Reset des configurations qui ne sont plus valides
        if selected_task_type == 'clustering':
            st.session_state.target_column_for_ml_config = None
            st.session_state.preprocessing_choices['use_smote'] = False
            # Reset features pour clustering
            st.session_state.feature_list_for_ml_config = []
        elif selected_task_type in ['classification', 'regression']:
            # Reset des sélections inappropriées
            if st.session_state.target_column_for_ml_config and st.session_state.target_column_for_ml_config not in df.columns:
                st.session_state.target_column_for_ml_config = None
        
        st.session_state.previous_task_type = selected_task_type
        st.session_state.task_type = selected_task_type
        st.rerun()
    else:
        st.session_state.task_type = selected_task_type
    
    # Configuration spécifique selon le type de tâche
    if selected_task_type in ['classification', 'regression']:
        st.subheader("🎯 Variable Cible (Y)")
        
        # Sélecteur de cible adapté au type de tâche
        if selected_task_type == 'classification':
            # Pour classification: privilégier les colonnes catégorielles ou avec peu de valeurs uniques
            available_targets = [col for col in df.columns if df[col].nunique() <= 50 or not pd.api.types.is_numeric_dtype(df[col])]
        else:  # regression
            # Pour régression: privilégier les colonnes numériques continues
            available_targets = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10]
        
        if not available_targets:
            st.error("❌ Aucune variable cible appropriée trouvée")
            if selected_task_type == 'classification':
                st.info("Pour la classification, la variable cible doit avoir un nombre limité de valeurs uniques (≤50)")
            else:
                st.info("Pour la régression, la variable cible doit être numérique avec plusieurs valeurs uniques")
        else:
            # Ajouter option "Aucune" en premier
            available_targets = [None] + available_targets
            
            if not st.session_state.target_column_for_ml_config or st.session_state.target_column_for_ml_config not in available_targets:
                target_idx = 0
            else:
                try:
                    target_idx = available_targets.index(st.session_state.target_column_for_ml_config)
                except ValueError:
                    target_idx = 0
            
            target_column = st.selectbox(
                "Sélectionnez la variable à prédire",
                options=available_targets,
                index=target_idx,
                key="ml_target_selector_stable",
                help="Variable que le modèle apprendra à prédire"
            )
            
            # Mise à jour de l'état cible
            if target_column != st.session_state.target_column_for_ml_config:
                st.session_state.target_column_for_ml_config = target_column
                # Reset features si changement de cible
                st.session_state.feature_list_for_ml_config = []
            
            if target_column:
                # Analyse de la cible avec feedback utilisateur
                with st.spinner("🔍 Analyse de la variable cible..."):
                    task_info = safe_get_task_type(df, target_column)
                
                if task_info["error"]:
                    st.error(f"❌ Erreur analyse cible: {task_info['error']}")
                else:
                    # Affichage des informations sur la tâche
                    if selected_task_type == "classification":
                        st.success(f"✅ **Tâche: CLASSIFICATION** ({task_info['n_classes']} classes détectées)")
                        
                        # Affichage distribution des classes
                        class_dist = df[target_column].value_counts()
                        if len(class_dist) <= 10:
                            st.bar_chart(class_dist)
                            st.caption(f"Distribution des {len(class_dist)} classes")
                        
                        # Vérification déséquilibre
                        try:
                            imbalance_info = detect_imbalance(df, target_column)
                            if imbalance_info and imbalance_info.get("is_imbalanced"):
                                st.warning(f"⚠️ **Déséquilibre détecté** (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                                st.info("💡 **Conseil**: Activez SMOTE dans l'étape de prétraitement pour améliorer les performances")
                        except Exception as e:
                            logger.debug(f"Imbalance detection failed: {e}")
                            
                    elif selected_task_type == "regression":
                        st.success("✅ **Tâche: RÉGRESSION**")
                        
                        # Statistiques de la variable cible
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
        
        # Sélection des features avec validation
        st.subheader("📊 Variables Explicatives (X)")
        all_features = [col for col in df.columns if col != target_column] if target_column else list(df.columns)
        
        if all_features:
            # Features recommandées vs toutes
            recommend_features = st.checkbox(
                "Sélection automatique des features pertinentes", 
                value=len(st.session_state.feature_list_for_ml_config) == 0,
                help="Sélectionne automatiquement les variables les plus prometteuses"
            )
            
            if recommend_features and target_column:
                with st.spinner("🤖 Analyse des features..."):
                    try:
                        # Sélection intelligente basée sur les types
                        column_types = auto_detect_column_types(df)
                        recommended_features = []
                        
                        # Ajouter colonnes numériques (généralement bonnes pour ML)
                        recommended_features.extend(
                            col for col in column_types.get('numeric', []) 
                            if col != target_column and col in all_features
                        )
                        
                        # Ajouter quelques catégorielles avec peu de modalités
                        categorical_features = [
                            col for col in column_types.get('categorical', [])
                            if col != target_column and col in all_features and df[col].nunique() <= 20
                        ]
                        recommended_features.extend(categorical_features[:8])  # Limite à 8
                        
                        if recommended_features:
                            st.session_state.feature_list_for_ml_config = recommended_features[:25]  # Limite globale
                            st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} features sélectionnées automatiquement")
                        else:
                            st.session_state.feature_list_for_ml_config = all_features[:15]
                            st.info("ℹ️ Sélection par défaut appliquée")
                    except Exception as e:
                        logger.error(f"Auto feature selection failed: {e}")
                        st.session_state.feature_list_for_ml_config = all_features[:15]
            else:
                # Sélection manuelle
                selected_features = st.multiselect(
                    "Variables d'entrée pour la prédiction",
                    options=all_features,
                    default=st.session_state.feature_list_for_ml_config if st.session_state.feature_list_for_ml_config else [],
                    key="ml_features_selector_stable",
                    help="Variables utilisées pour prédire la cible"
                )
                st.session_state.feature_list_for_ml_config = selected_features
            
            # Affichage des features sélectionnées
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} features sélectionnées")
                if len(st.session_state.feature_list_for_ml_config) > 12:
                    features_display = st.session_state.feature_list_for_ml_config[:10]
                    st.caption(f"📋 {', '.join(features_display)} ... +{len(st.session_state.feature_list_for_ml_config)-10} autres")
                else:
                    st.caption(f"📋 {', '.join(st.session_state.feature_list_for_ml_config)}")
                
                # Avertissement si trop de features
                if len(st.session_state.feature_list_for_ml_config) > 30:
                    st.warning("⚠️ Nombre élevé de features - risque de surapprentissage")
            else:
                st.warning("⚠️ Aucune feature sélectionnée")
        else:
            st.error("❌ Aucune feature disponible")
    
    else:  # Non supervisé (Clustering)
        st.session_state.target_column_for_ml_config = None
        st.success("✅ **Tâche: CLUSTERING NON SUPERVISÉ**")
        st.info("🔍 Le modèle identifiera automatiquement des groupes naturels dans les données sans variable cible")
        
        # Sélection features pour clustering - uniquement numériques
        all_numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        
        if not all_numeric_features:
            st.error("❌ Aucune variable numérique disponible pour le clustering")
            st.info("Le clustering nécessite des variables numériques. Vérifiez les types de données de votre dataset.")
        else:
            st.subheader("📊 Variables pour le Clustering")
            st.info("💡 **Conseil**: Sélectionnez des variables numériques représentatives pour obtenir de bons clusters")
            
            # Sélection automatique pour clustering
            auto_cluster_features = st.checkbox(
                "Sélection automatique des variables numériques",
                value=len(st.session_state.feature_list_for_ml_config) == 0,
                help="Sélectionne toutes les variables numériques adaptées au clustering"
            )
            
            if auto_cluster_features:
                # NOUVEAU : Validation des features pour clustering
                validation_result = validate_clustering_features(df, all_numeric_features[:20])
                st.session_state.feature_list_for_ml_config = validation_result["valid_features"]
                
                if validation_result["warnings"]:
                    with st.expander("⚠️ Avertissements clustering", expanded=True):
                        for warning in validation_result["warnings"]:
                            st.warning(f"• {warning}")
                
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables numériques sélectionnées")
            else:
                # Sélection manuelle
                clustering_features = st.multiselect(
                    "Variables pour l'analyse de clusters",
                    options=all_numeric_features,
                    default=st.session_state.feature_list_for_ml_config if st.session_state.feature_list_for_ml_config else all_numeric_features[:10],
                    key="clustering_features_selector",
                    help="Variables numériques utilisées pour identifier les patterns et clusters"
                )
                
                # Validation des features sélectionnées
                if clustering_features:
                    validation_result = validate_clustering_features(df, clustering_features)
                    st.session_state.feature_list_for_ml_config = validation_result["valid_features"]
                    
                    if validation_result["warnings"]:
                        with st.expander("⚠️ Avertissements clustering", expanded=True):
                            for warning in validation_result["warnings"]:
                                st.warning(f"• {warning}")
                else:
                    st.session_state.feature_list_for_ml_config = clustering_features
            
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées pour le clustering")
                
                # Vérification de la qualité des features pour clustering
                if len(st.session_state.feature_list_for_ml_config) < 2:
                    st.warning("⚠️ Au moins 2 variables sont recommandées pour un clustering significatif")
                elif len(st.session_state.feature_list_for_ml_config) > 15:
                    st.warning("⚠️ Nombre élevé de variables - risque de 'malédiction de la dimensionnalité'")
                
                # Aperçu statistique
                with st.expander("📈 Aperçu des variables sélectionnées", expanded=False):
                    cluster_stats = df[st.session_state.feature_list_for_ml_config].describe()
                    st.dataframe(cluster_stats.style.format("{:.3f}"), use_container_width=True)
            else:
                st.warning("⚠️ Aucune variable sélectionnée pour le clustering")

# Étape 2: Prétraitement
elif st.session_state.current_step == 2:
    st.header("🔧 Configuration du Prétraitement")
    
    task_type = st.session_state.get('task_type', 'classification')
    
    st.info(f"""
    ℹ️ **Pipeline de prétraitement pour {task_type.upper()}**: 
    Les transformations sont appliquées dans l'ordre suivant, séparément sur train/validation pour éviter le data leakage.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧩 Gestion des Valeurs Manquantes")
        
        # Stratégies avec explications adaptées au type de tâche
        st.session_state.preprocessing_choices['numeric_imputation'] = st.selectbox(
            "Variables numériques",
            options=['mean', 'median', 'constant', 'knn'],
            index=['mean', 'median', 'constant', 'knn'].index(
                st.session_state.preprocessing_choices.get('numeric_imputation', 'mean')
            ),
            key='numeric_imputation_selector',
            help="mean=moyenne (robuste), median=médiane (extrêmes), constant=0, knn=k-voisins (précis)"
        )
        
        st.session_state.preprocessing_choices['categorical_imputation'] = st.selectbox(
            "Variables catégorielles",
            options=['most_frequent', 'constant'],
            index=['most_frequent', 'constant'].index(
                st.session_state.preprocessing_choices.get('categorical_imputation', 'most_frequent')
            ),
            key='categorical_imputation_selector',
            help="most_frequent=mode (fréquent), constant='missing' (explicite)"
        )
        
        st.subheader("🧹 Nettoyage Automatique")
        
        st.session_state.preprocessing_choices['remove_constant_cols'] = st.checkbox(
            "Supprimer colonnes constantes",
            value=st.session_state.preprocessing_choices.get('remove_constant_cols', True),
            key="remove_constant_checkbox",
            help="Élimine variables sans variance (utile pour tous les types)"
        )
        
        st.session_state.preprocessing_choices['remove_identifier_cols'] = st.checkbox(
            "Supprimer colonnes identifiantes",
            value=st.session_state.preprocessing_choices.get('remove_identifier_cols', True),
            key="remove_id_checkbox",
            help="Élimine variables avec valeurs uniques (ID, etc.)"
        )
    
    with col2:
        st.subheader("📏 Normalisation et Mise à l'échelle")
        
        scale_help = {
            'classification': "Recommandé pour SVM, KNN, réseaux de neurones",
            'regression': "Recommandé pour la plupart des algorithmes", 
            'clustering': "ESSENTIEL pour le clustering (KMeans, DBSCAN)"
        }
        
        st.session_state.preprocessing_choices['scale_features'] = st.checkbox(
            "Normaliser les features",
            value=st.session_state.preprocessing_choices.get('scale_features', True),
            key="scale_features_checkbox",
            help=scale_help.get(task_type, "Recommandé pour la plupart des algorithmes")
        )
        
        if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
            st.error("❌ **ATTENTION**: La normalisation est CRITIQUE pour le clustering!")
            st.info("Les algorithmes comme KMeans sont sensibles à l'échelle des variables")
        
        # Options spécifiques au type de tâche
        if task_type == 'classification':
            st.subheader("⚖️ Gestion du Déséquilibre")
            
            if st.session_state.target_column_for_ml_config:
                try:
                    with st.spinner("Analyse du déséquilibre..."):
                        imbalance_info = detect_imbalance(df, st.session_state.target_column_for_ml_config)
                    
                    if imbalance_info and imbalance_info.get("is_imbalanced", False):
                        st.warning("📉 **Déséquilibre de classes détecté**")
                        ratio = imbalance_info.get('imbalance_ratio', 0)
                        majority_class = imbalance_info.get('majority_class', '')
                        minority_class = imbalance_info.get('minority_class', '')
                        
                        st.write(f"**Ratio**: {ratio:.2f}")
                        st.write(f"**Classe majoritaire**: {majority_class}")
                        st.write(f"**Classe minoritaire**: {minority_class}")
                        
                        st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                            "Activer SMOTE (Sur-échantillonnage)",
                            value=st.session_state.preprocessing_choices.get('use_smote', True),
                            key="smote_checkbox",
                            help="Génère des échantillons synthétiques pour équilibrer les classes minoritaires"
                        )
                        
                        if st.session_state.preprocessing_choices['use_smote']:
                            st.success("✅ SMOTE activé - améliorera les performances sur les classes minoritaires")
                    else:
                        st.success("✅ Classes équilibrées")
                        st.session_state.preprocessing_choices['use_smote'] = False
                        st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                            "Activer SMOTE (optionnel)",
                            value=False,
                            key="smote_optional_checkbox",
                            help="Peut être activé même si les classes sont équilibrées"
                        )
                        
                except Exception as e:
                    logger.error(f"Imbalance detection error: {e}")
                    st.warning("⚠️ Impossible d'analyser le déséquilibre")
                    st.session_state.preprocessing_choices['use_smote'] = False
            else:
                st.info("🔒 Variable cible requise pour l'analyse de déséquilibre")
        
        elif task_type == 'clustering':
            st.subheader("🔍 Options de Clustering")
            
            st.info("""
            **Recommandations pour le clustering:**
            - ✅ Normalisation CRITIQUE
            - ✅ Suppression des variables constantes
            - ✅ Gestion des valeurs manquantes
            """)
            
            # Option spécifique au clustering
            st.session_state.preprocessing_choices['pca_preprocessing'] = st.checkbox(
                "Réduction de dimension (PCA optionnel)",
                value=st.session_state.preprocessing_choices.get('pca_preprocessing', False),
                help="Réduit le bruit et amliore les performances sur données haute dimension"
            )

# Étape 3: Sélection des modèles
elif st.session_state.current_step == 3:
    st.header("🤖 Sélection et Configuration des Modèles")
    
    task_type = st.session_state.get('task_type', 'classification')
    available_models = get_task_specific_models(task_type)
    
    if not available_models:
        st.error(f"❌ Aucun modèle disponible pour '{task_type}'")
        st.info("Vérifiez la configuration du catalogue de modèles")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Modèles Disponibles")
        
        # Pré-sélection intelligente basée sur le type de tâche
        if not st.session_state.selected_models_for_training:
            default_models = get_default_models_for_task(task_type)
            st.session_state.selected_models_for_training = default_models
        
        selected_models = st.multiselect(
            f"Modèles {task_type} à entraîner et comparer",
            options=available_models,
            default=st.session_state.selected_models_for_training,
            key="models_multiselect_stable",
            help="Chaque modèle sera entraîné et évalué automatiquement"
        )
        
        st.session_state.selected_models_for_training = selected_models
        
        # Informations détaillées sur les modèles
        if selected_models:
            st.success(f"✅ {len(selected_models)} modèles sélectionnés")
            
            with st.expander("📋 Détails des modèles sélectionnés", expanded=False):
                for model_name in selected_models:
                    try:
                        model_config = MODEL_CATALOG[task_type][model_name]
                        st.write(f"**{model_name}**")
                        
                        if 'description' in model_config:
                            st.caption(f"• {model_config['description']}")
                        
                        st.caption(f"• Type: {type(model_config['model']).__name__}")
                        
                        if model_config.get('params'):
                            param_count = len(model_config['params'])
                            st.caption(f"• Hyperparamètres: {param_count} disponibles")
                            
                        # Conseils spécifiques
                        if task_type == 'clustering':
                            if model_name == 'KMeans':
                                st.caption("💡 **Conseil**: Excellent pour clusters sphériques de taille similaire")
                            elif model_name == 'DBSCAN':
                                st.caption("💡 **Conseil**: Robustes au bruit, trouve clusters de forme arbitraire")
                            elif model_name == 'GaussianMixture':
                                st.caption("💡 **Conseil**: Modèle probabiliste, bon pour clusters de taille variable")
                            
                    except Exception as e:
                        logger.error(f"Model info error for {model_name}: {e}")
                        st.caption(f"• {model_name}: Informations non disponibles")
        else:
            st.warning("⚠️ Aucun modèle sélectionné")
    
    with col2:
        st.subheader("⚙️ Configuration Avancée")
        
        # Configuration différente selon le type de tâche
        if task_type != 'clustering':
            # Taille du jeu de test avec validation - UNIQUEMENT pour supervisé
            test_split = st.slider(
                "Jeu de test (%)",
                min_value=10,
                max_value=40,
                value=st.session_state.get('test_split_for_ml_config', 20),
                step=5,
                key="test_split_slider_stable",
                help="Pourcentage de données réservées pour l'évaluation finale"
            )
            st.session_state.test_split_for_ml_config = test_split
            st.caption(f"📊 {test_split}% pour test, {100-test_split}% pour entraînement")
        else:
            # Pour non supervisé, pas de split
            st.info("🔍 **Clustering**: Utilisation de 100% des données")
            st.session_state.test_split_for_ml_config = 0
            st.caption("Le clustering utilise tout le dataset pour trouver des patterns")
        
        # Optimisation des hyperparamètres
        optimize_hp = st.checkbox(
            "Optimisation hyperparamètres",
            value=st.session_state.get('optimize_hp_for_ml_config', False),
            key="optimize_hp_checkbox_stable",
            help="Recherche automatique des meilleurs paramètres (plus long mais meilleures performances)"
        )
        st.session_state.optimize_hp_for_ml_config = optimize_hp
        
        if optimize_hp:
            st.warning("⏰ Temps d'entraînement multiplié par 3-5x")
            
            # Options d'optimisation adaptées
            if task_type == 'clustering':
                optimization_method = st.selectbox(
                    "Méthode d'optimisation",
                    options=['Silhouette Score', 'Davies-Bouldin'],
                    index=0,
                    key="optimization_method_selector",
                    help="Silhouette=qualité clusters, Davies-Bouldin=compacité"
                )
            else:
                optimization_method = st.selectbox(
                    "Méthode d'optimisation",
                    options=['GridSearch', 'RandomSearch'],
                    index=0,
                    key="optimization_method_selector",
                    help="GridSearch=exhaustif (précis), RandomSearch=échantillonnage (rapide)"
                )
            st.session_state.optimization_method = optimization_method
        
        # NOUVELLE ESTIMATION DU TEMPS plus précise
        n_features = len(st.session_state.feature_list_for_ml_config)
        use_smote = st.session_state.preprocessing_choices.get('use_smote', False)
        
        estimated_seconds = estimate_training_time(
            df, len(selected_models), task_type, optimize_hp, n_features, use_smote
        )
        
        estimated_minutes = max(1, estimated_seconds // 60)
        
        st.info(f"⏱️ Temps estimé: {estimated_minutes} minute(s)")
        
        # Vérification des ressources système
        if selected_models:
            resource_check = check_system_resources(df, len(selected_models))
            
            if not resource_check["has_enough_resources"]:
                st.error("❌ Ressources système insuffisantes")
                for issue in resource_check["issues"]:
                    st.error(f"• {issue}")
            elif resource_check["warnings"]:
                st.warning("⚠️ Ressources système limites")
                for warning in resource_check["warnings"]:
                    st.warning(f"• {warning}")
        
        # Avertissements spécifiques
        if task_type == 'clustering' and len(selected_models) > 3:
            st.warning("⚠️ Le clustering peut être long avec beaucoup de données")

# Étape 4: Lancement
elif st.session_state.current_step == 4:
    st.header("🚀 Lancement de l'Expérimentation")
    
    task_type = st.session_state.get('task_type', 'classification')
    
    # Validation complète de la configuration
    config_issues = []
    config_warnings = []
    
    # Vérifications obligatoires
    if task_type in ['classification', 'regression'] and not st.session_state.target_column_for_ml_config:
        config_issues.append("Variable cible non définie")
    
    if not st.session_state.feature_list_for_ml_config:
        config_issues.append("Aucune variable explicative sélectionnée")
    elif len(st.session_state.feature_list_for_ml_config) < 2 and task_type == 'clustering':
        config_issues.append("Au moins 2 variables requises pour le clustering")
    
    if not st.session_state.selected_models_for_training:
        config_issues.append("Aucun modèle sélectionné")
    
    # Vérifications de qualité spécifiques
    if task_type == 'clustering':
        if not st.session_state.preprocessing_choices.get('scale_features', True):
            config_warnings.append("⚠️ La normalisation est CRITIQUE pour le clustering!")
        
        if len(st.session_state.feature_list_for_ml_config) > 15:
            config_warnings.append("Beaucoup de variables - risque de malédiction dimensionnelle")
    
    elif task_type == 'classification':
        if len(st.session_state.feature_list_for_ml_config) > 30:
            config_warnings.append("Beaucoup de features - risque de surapprentissage")
    
    if len(st.session_state.selected_models_for_training) > 5:
        config_warnings.append("Beaucoup de modèles sélectionnés (temps long)")
    
    # Vérification finale des ressources
    resource_check = check_system_resources(df, len(st.session_state.selected_models_for_training))
    if not resource_check["has_enough_resources"]:
        config_issues.extend(resource_check["issues"])
    config_warnings.extend(resource_check["warnings"])
    
    # Récapitulatif de configuration adapté
    with st.expander("📋 Récapitulatif Configuration", expanded=True):
        if config_issues:
            st.error("❌ Configuration incomplète:")
            for issue in config_issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ Configuration valide")
        
        if config_warnings:
            st.warning("⚠️ Avertissements:")
            for warning in config_warnings:
                st.write(f"• {warning}")
        
        # Détails de la configuration adaptés au type de tâche
        if not config_issues:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Configuration Données**")
                st.write(f"• Type: {task_type.upper()}")
                if task_type != 'clustering':
                    st.write(f"• Cible: `{st.session_state.target_column_for_ml_config}`")
                st.write(f"• Features: {len(st.session_state.feature_list_for_ml_config)}")
                if task_type != 'clustering':
                    st.write(f"• Test: {st.session_state.test_split_for_ml_config}%")
                else:
                    st.write("• Test: 0% (clustering)")
            
            with col2:
                st.markdown("**🤖 Configuration Modèles**")
                st.write(f"• Modèles: {len(st.session_state.selected_models_for_training)}")
                st.write(f"• Optimisation: {'✅' if st.session_state.optimize_hp_for_ml_config else '❌'}")
                
                if task_type == 'classification':
                    st.write(f"• SMOTE: {'✅' if st.session_state.preprocessing_choices.get('use_smote') else '❌'}")
                
                st.write(f"• Normalisation: {'✅' if st.session_state.preprocessing_choices.get('scale_features') else '❌'}")
                
            # Informations ressources
            st.markdown("**💻 Ressources Système**")
            st.write(f"• Mémoire disponible: {resource_check['available_memory_mb']:.0f} MB")
            st.write(f"• Mémoire estimée nécessaire: {resource_check['estimated_needed_mb']:.0f} MB")
            st.write(f"• Statut: {'✅ Suffisante' if resource_check['has_enough_resources'] else '❌ Insuffisante'}")
    
    # Boutons d'action
    col_launch, col_reset, col_info = st.columns([2, 1, 2])
    
    with col_launch:
        launch_disabled = len(config_issues) > 0 or st.session_state.get('ml_training_in_progress', False)
        
        launch_button = st.button(
            "🚀 Lancer l'Expérimentation",
            type="primary",
            use_container_width=True,
            disabled=launch_disabled,
            help="Démarrer l'entraînement avec la configuration actuelle"
        )
        
        if launch_button:
            # Préparation du lancement
            st.session_state.ml_training_in_progress = True
            st.session_state.ml_last_training_time = time.time()
            
            # Configuration finale adaptée au type de tâche
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
            
            # NOUVEAU : Lancement avec progression détaillée
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            try:
                # Étape 1: Préparation
                status_text.text("📊 Préparation des données...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # Étape 2: Entraînement des modèles
                status_text.text("🤖 Entraînement des modèles en cours...")
                
                # Appel principal avec progression
                n_models = len(st.session_state.selected_models_for_training)
                results = []
                
                for i, model_name in enumerate(st.session_state.selected_models_for_training):
                    status_text.text(f"🔧 Entraînement {i+1}/{n_models}: {model_name}")
                    progress_bar.progress(10 + int((i / n_models) * 80))
                    
                    # Entraînement du modèle individuel
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
                            "success": False
                        })
                    
                    time.sleep(0.5)  # Pause pour feedback utilisateur
                
                # Étape 3: Finalisation
                status_text.text("✅ Finalisation de l'expérimentation...")
                progress_bar.progress(95)
                time.sleep(1)
                
                elapsed_time = time.time() - st.session_state.ml_last_training_time
                status_text.text(f"✅ Expérimentation terminée en {elapsed_time:.1f}s")
                progress_bar.progress(100)
                
                # Sauvegarde des résultats
                st.session_state.ml_results = results
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = 0
                
                # Analyse rapide des résultats adaptée
                successful_models = [r for r in results if r.get('success', False) and not r.get('metrics', {}).get('error')]
                failed_models = [r for r in results if not r.get('success', False) or r.get('metrics', {}).get('error')]
                
                with results_container.container():
                    st.success(f"✅ Expérimentation terminée! {len(successful_models)}/{len(results)} modèles réussis")
                    
                    if successful_models:
                        # Affichage du meilleur modèle adapté
                        if task_type == 'classification':
                            best_model = max(successful_models, key=lambda x: x.get('metrics', {}).get('accuracy', 0))
                            best_score = best_model.get('metrics', {}).get('accuracy', 0)
                            st.info(f"🏆 Meilleur modèle: **{best_model['model_name']}** (Accuracy: {best_score:.3f})")
                        elif task_type == 'regression':
                            best_model = max(successful_models, key=lambda x: x.get('metrics', {}).get('r2', -999))
                            best_score = best_model.get('metrics', {}).get('r2', 0)
                            st.info(f"🏆 Meilleur modèle: **{best_model['model_name']}** (R²: {best_score:.3f})")
                        else:  # clustering
                            best_model = max(successful_models, key=lambda x: x.get('metrics', {}).get('silhouette_score', -999))
                            best_score = best_model.get('metrics', {}).get('silhouette_score', 0)
                            st.info(f"🏆 Meilleur modèle: **{best_model['model_name']}** (Silhouette: {best_score:.3f})")
                    
                    if failed_models:
                        st.warning(f"⚠️ {len(failed_models)} modèles ont échoué")
                    
                    # Navigation
                    st.balloons()
                    if st.button("📈 Voir les résultats détaillés", use_container_width=True):
                        st.switch_page("pages/3_📈_Évaluation_du_Modèle.py")
                    
            except Exception as e:
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = st.session_state.get('ml_error_count', 0) + 1
                
                error_msg = str(e)
                status_text.text("❌ Expérimentation échouée")
                progress_bar.progress(0)
                
                st.error(f"❌ Erreur durant l'entraînement: {error_msg[:200]}")
                logger.error(f"Training failed: {e}", exc_info=True)
    
    with col_reset:
        if st.button("🔄 Reset Config", use_container_width=True, help="Remet à zéro la configuration"):
            # Reset sélectif des paramètres ML
            ml_keys_to_reset = [
                'target_column_for_ml_config',
                'feature_list_for_ml_config',
                'selected_models_for_training',
                'ml_results',
                'task_type',
                'previous_task_type'
            ]
            for key in ml_keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state.current_step = 1
            st.success("Configuration réinitialisée")
            st.rerun()
    
    with col_info:
        # État actuel
        if st.session_state.get('ml_training_in_progress'):
            st.info("⏳ Entraînement en cours...")
        elif st.session_state.get('ml_last_training_time'):
            last_time = time.strftime('%H:%M:%S', time.localtime(st.session_state.ml_last_training_time))
            st.caption(f"Dernier: {last_time}")
        
        if st.session_state.get('ml_error_count', 0) > 0:
            st.warning(f"⚠️ {st.session_state.ml_error_count} erreurs")

# Footer avec monitoring et navigation
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    progress = (st.session_state.current_step / 4) * 100
    st.caption(f"📊 Étape {st.session_state.current_step}/4 ({progress:.0f}%)")

with footer_col2:
    task_type_display = st.session_state.get('task_type', 'Non défini')
    st.caption(f"🎯 {task_type_display.upper()}")

with footer_col3:
    try:
        sys_memory = psutil.virtual_memory().percent
        color = "🔴" if sys_memory > 85 else "🟡" if sys_memory > 70 else "🟢"
        st.caption(f"{color} RAM: {sys_memory:.0f}%")
    except:
        st.caption("🔧 RAM: N/A")

with footer_col4:
    st.caption(f"⏰ {time.strftime('%H:%M:%S')}")

# Navigation entre les étapes
st.markdown("---")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.session_state.current_step > 1:
        if st.button("◀️ Étape précédente", use_container_width=True):
            st.session_state.current_step -= 1
            st.rerun()

with nav_col4:
    if st.session_state.current_step < 4:
        if st.button("Étape suivante ▶️", use_container_width=True, type="primary"):
            # Validation avant passage à l'étape suivante
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

# Debug conditionnel
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    with st.expander("🔍 Debug ML Config", expanded=False):
        debug_info = {
            "current_step": st.session_state.current_step,
            "task_type": st.session_state.get('task_type'),
            "target_column": st.session_state.get('target_column_for_ml_config'),
            "num_features": len(st.session_state.get('feature_list_for_ml_config', [])),
            "num_models": len(st.session_state.get('selected_models_for_training', [])),
            "test_split": st.session_state.get('test_split_for_ml_config'),
            "training_in_progress": st.session_state.get('ml_training_in_progress', False),
            "error_count": st.session_state.get('ml_error_count', 0)
        }
        st.json(debug_info)