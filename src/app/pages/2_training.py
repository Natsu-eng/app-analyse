import logging
import mlflow
import streamlit as st
import pandas as pd
import time
import psutil
import os
import gc
import json
from datetime import datetime
from typing import Dict, List, Any
from functools import wraps
import numpy as np
import concurrent.futures

# Imports des modules ML
from src.models.catalog import MODEL_CATALOG
from src.data.data_analysis import get_target_and_task, detect_imbalance, auto_detect_column_types
from src.models.training import train_models, is_mlflow_available
from src.shared.logging import get_logger
from src.config.constants import (
    VALIDATION_CONSTANTS, PREPROCESSING_CONSTANTS, TRAINING_CONSTANTS,
    MLFLOW_CONSTANTS, LOGGING_CONSTANTS, VISUALIZATION_CONSTANTS
)

# Configuration
logger = get_logger(__name__)
st.set_page_config(page_title="Configuration ML", page_icon="⚙️", layout="wide")

def log_structured(level: str, message: str, extra: Dict = None):
    log_dict = {"timestamp": datetime.now().isoformat(), "level": level, "message": message}
    if extra:
        log_dict.update(extra)
    logger.log(getattr(logging, level.upper()), json.dumps(log_dict))

def safe_train_models(**kwargs):
    """Wrapper sécurisé autour de train_models avec gestion d'erreurs étendue"""
    try:
        # Validation des paramètres critiques
        required_params = ['df', 'model_names', 'task_type']
        for param in required_params:
            if param not in kwargs or kwargs[param] is None:
                raise ValueError(f"Paramètre manquant: {param}")
        
        if kwargs['df'].empty:
            raise ValueError("DataFrame vide")
            
        if not kwargs['model_names']:
            raise ValueError("Aucun modèle sélectionné")
        
        # Appel sécurisé
        results = train_models(**kwargs)
        
        # Validation des résultats
        if not results or not isinstance(results, list):
            raise ValueError("Aucun résultat retourné par l'entraînement")
            
        return results
        
    except Exception as e:
        log_structured("ERROR", f"Erreur entraînement: {str(e)}", {
            "error_type": type(e).__name__,
            "task_type": kwargs.get('task_type'),
            "n_models": len(kwargs.get('model_names', [])),
            "df_shape": kwargs.get('df').shape if kwargs.get('df') is not None else None
        })
        raise

def setup_ml_config_environment():
    """Configuration robuste pour l'environnement de production ML"""
    if 'ml_config_setup_done' not in st.session_state:
        st.session_state.ml_config_setup_done = True
        
        # Configuration MLflow robuste
        if is_mlflow_available():
            try:
                mlflow.set_tracking_uri(MLFLOW_CONSTANTS["TRACKING_URI"])
                
                # Vérification que l'URI est accessible
                try:
                    experiments = mlflow.search_experiments()
                    log_structured("INFO", f"MLflow connecté - {len(experiments)} expériences")
                except Exception as conn_error:
                    log_structured("WARNING", f"MLflow connecté mais erreur recherche: {str(conn_error)[:100]}")
                
                # Gestion expérience
                experiment = mlflow.get_experiment_by_name(MLFLOW_CONSTANTS["EXPERIMENT_NAME"])
                if experiment is None:
                    try:
                        mlflow.create_experiment(MLFLOW_CONSTANTS["EXPERIMENT_NAME"])
                        log_structured("INFO", f"Expérience créée: {MLFLOW_CONSTANTS['EXPERIMENT_NAME']}")
                    except Exception as create_error:
                        log_structured("ERROR", f"Échec création expérience: {str(create_error)[:100]}")
                else:
                    log_structured("INFO", f"Expérience existante: {experiment.name}")
                    
            except Exception as e:
                log_structured("ERROR", f"Échec configuration MLflow: {str(e)[:100]}")
                # Continuer sans MLflow

def monitor_ml_operation(func):
    """Décorateur de monitoring"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2) if psutil else 0
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            memory_delta = (psutil.virtual_memory().used / (1024**2) - start_memory) if psutil else 0
            log_structured("INFO", f"Performance {func.__name__}", {
                "duration_s": elapsed,
                "memory_delta_mb": memory_delta
            })
            if elapsed > TRAINING_CONSTANTS["MONITOR_TIME_THRESHOLD"]:
                log_structured("WARNING", f"Opération lente: {func.__name__} a pris {elapsed:.2f}s")
            if memory_delta > LOGGING_CONSTANTS["HIGH_MEMORY_THRESHOLD"]:
                log_structured("WARNING", f"Usage mémoire élevé: {func.__name__} a utilisé {memory_delta:.1f}MB")
            return result
        except Exception as e:
            log_structured("ERROR", f"Échec {func.__name__}: {str(e)[:100]}")
            raise
    return wrapper

def validate_clustering_features(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Valide que les features sont adaptées au clustering avec analyses avancées"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "valid_features": [],
        "suggested_features": [],
        "analysis": {}
    }
    
    try:
        # Analyse de variance et corrélation
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 1:
            # Matrice de corrélation
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Features peu corrélées (bonnes pour clustering)
            low_corr_features = []
            for col in numeric_cols:
                if col in features:
                    other_cols = [c for c in numeric_cols if c != col]
                    avg_corr = corr_matrix.loc[col, other_cols].mean()
                    if avg_corr < VALIDATION_CONSTANTS["FEATURE_CORR_THRESHOLD"]:
                        low_corr_features.append((col, avg_corr))
            
            # Trier par corrélation moyenne (plus bas = mieux)
            low_corr_features.sort(key=lambda x: x[1])
            validation["suggested_features"] = [col for col, _ in low_corr_features[:10]]
            validation["analysis"]["avg_correlation"] = float(np.mean(corr_matrix.values))
        
        # Validation individuelle des features
        for feature in features:
            if feature not in df.columns:
                validation["issues"].append(f"Colonne '{feature}' introuvable")
                continue
                
            # Vérification variance
            if df[feature].std() == 0:
                validation["warnings"].append(f"'{feature}' est constante")
                continue
                
            # Vérification valeurs manquantes
            missing_ratio = df[feature].isnull().mean()
            if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
                validation["warnings"].append(f"'{feature}' a {missing_ratio:.1%} de valeurs manquantes")
                continue
                
            # Vérification valeurs uniques
            if df[feature].nunique() == 1:
                validation["warnings"].append(f"'{feature}' n'a qu'une seule valeur unique")
                continue
            
            # Vérification outliers (pour clustering)
            if df[feature].dtype in ['float64', 'int64']:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))).sum()
                if outlier_count > len(df) * 0.05:  # Plus de 5% d'outliers
                    validation["warnings"].append(f"'{feature}' a {outlier_count} outliers")
            
            validation["valid_features"].append(feature)
        
        # Validation finale
        if len(validation["valid_features"]) < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables numériques requises")
        
        # Analyse de qualité des données
        if validation["valid_features"]:
            validation["analysis"]["n_valid_features"] = len(validation["valid_features"])
            validation["analysis"]["data_quality_score"] = len(validation["valid_features"]) / len(features) if features else 0
        
        log_structured("INFO", "Validation clustering features", {
            "n_valid": len(validation["valid_features"]),
            "n_issues": len(validation["issues"]),
            "n_warnings": len(validation["warnings"]),
            "quality_score": validation["analysis"].get("data_quality_score", 0)
        })
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)[:100]}")
        log_structured("ERROR", f"Validation clustering features échouée: {str(e)[:100]}")
    
    return validation

def estimate_training_time(df: pd.DataFrame, n_models: int, task_type: str, 
                         optimize_hp: bool, n_features: int, use_smote: bool = False) -> int:
    """Estime le temps d'entraînement"""
    try:
        n_samples = min(len(df), VISUALIZATION_CONSTANTS["MAX_SAMPLES"])
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
        result = max(30, min(int(estimated_seconds), TRAINING_CONSTANTS["MAX_TRAINING_TIME"]))
        log_structured("INFO", "Estimation temps entraînement", {"seconds": result})
        return result
    except Exception as e:
        log_structured("WARNING", f"Erreur estimation temps: {str(e)[:100]}")
        return 60

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système"""
    try:
        memory = psutil.virtual_memory()
        metrics = {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'timestamp': time.time()
        }
        log_structured("INFO", "Métriques système collectées", metrics)
        return metrics
    except Exception as e:
        log_structured("ERROR", f"Échec métriques système: {str(e)[:100]}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'cpu_percent': 0, 'timestamp': time.time()}

def check_system_resources(df: pd.DataFrame, n_models: int) -> Dict[str, Any]:
    """Vérifie les ressources système"""
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
            check_result["issues"].append(f"Mémoire insuffisante: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        elif estimated_needed > available_memory * VALIDATION_CONSTANTS["MEMORY_WARNING_THRESHOLD"]:
            check_result["warnings"].append(f"Mémoire limite: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > TRAINING_CONSTANTS["HIGH_CPU_THRESHOLD"]:
            check_result["warnings"].append(f"CPU élevé: {cpu_percent:.1f}%")
            
        log_structured("INFO", "Vérification ressources", check_result)
    except Exception as e:
        check_result["warnings"].append("Échec vérification ressources")
        log_structured("WARNING", f"Erreur vérification ressources: {str(e)[:100]}")
    
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
            validation["issues"].append(f"Insuffisant: {n_rows} lignes (minimum: {VALIDATION_CONSTANTS['MIN_ROWS_REQUIRED']})")
        
        if n_cols < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Insuffisant: {n_cols} colonnes (minimum: {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']})")
        
        missing_ratio = df.isnull().mean().max()
        if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop de valeurs manquantes: {missing_ratio:.1%}")
        elif missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"] * VALIDATION_CONSTANTS["MISSING_WARNING_THRESHOLD"]:
            validation["warnings"].append(f"Beaucoup de valeurs manquantes: {missing_ratio:.1%}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if len(constant_cols) == len(numeric_cols):
                validation["warnings"].append("Toutes les colonnes numériques sont constantes")
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        validation["stats"]["memory_mb"] = memory_usage
        if memory_usage > TRAINING_CONSTANTS["MEMORY_LIMIT_MB"] / 4:
            validation["warnings"].append(f"Dataset volumineux: {memory_usage:.1f} MB")
            
        log_structured("INFO", "Validation DataFrame", validation["stats"])
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)[:100]}")
        log_structured("ERROR", f"Erreur validation DataFrame: {str(e)[:100]}")
    
    return validation

@st.cache_data(ttl=300, max_entries=3)
def cached_auto_detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Cache la détection des types de colonnes"""
    try:
        result = auto_detect_column_types(df)
        log_structured("INFO", "Types de colonnes détectés", {"n_cols": len(df.columns)})
        return result
    except Exception as e:
        log_structured("ERROR", f"Échec détection types colonnes: {str(e)[:100]}")
        return {"numeric": [], "categorical": [], "identifier": []}

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
            'scaling_method': PREPROCESSING_CONSTANTS["SCALING_METHOD"],
            'smote_k_neighbors': 5,
            'smote_sampling_strategy': 'auto'
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
        'mlflow_runs': None,
        'warnings': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@monitor_ml_operation
def safe_get_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Détection sécurisée du type de tâche"""
    try:
        if not target_column or target_column not in df.columns:
            return {"task_type": "unknown", "n_classes": 0, "error": "Colonne cible invalide", "warnings": []}
        
        if df[target_column].nunique() == len(df):
            return {"task_type": "unknown", "n_classes": 0, "error": "Variable cible est un identifiant", "warnings": []}
        
        result_dict = get_target_and_task(df, target_column)
        task_type = result_dict.get("task", "unknown")
        target_type = result_dict.get("target_type", "unknown")
        
        warnings = []
        n_classes = 0
        if task_type == "classification":
            n_classes = df[target_column].nunique()
            if n_classes > TRAINING_CONSTANTS["MAX_CLASSES"]:
                return {"task_type": "unknown", "n_classes": n_classes, "error": f"Trop de classes ({n_classes})", "warnings": []}
            if n_classes < 2:
                warnings.append("Variable cible a moins de 2 classes")
        
        log_structured("INFO", "Type de tâche détecté", {"task_type": task_type, "n_classes": n_classes})
        return {"task_type": task_type, "target_type": target_type, "n_classes": n_classes, "error": None, "warnings": warnings}
    except Exception as e:
        log_structured("ERROR", f"Échec détection type tâche: {str(e)[:100]}")
        return {"task_type": "unknown", "n_classes": 0, "error": str(e)[:100], "warnings": []}

def get_task_specific_models(task_type: str) -> List[str]:
    """Retourne les modèles disponibles pour une tâche"""
    try:
        models = list(MODEL_CATALOG.get(task_type, {}).keys())
        log_structured("INFO", f"Modèles récupérés pour {task_type}", {"n_models": len(models)})
        return models
    except Exception as e:
        log_structured("ERROR", f"Erreur récupération modèles pour {task_type}: {str(e)[:100]}")
        return []

def get_default_models_for_task(task_type: str) -> List[str]:
    """Retourne les modèles par défaut pour une tâche"""
    default_models = {
        'classification': ['RandomForest', 'XGBoost', 'LogisticRegression'],
        'regression': ['RandomForest', 'XGBoost', 'LinearRegression'],
        'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture']
    }
    available_models = get_task_specific_models(task_type)
    models = [model for model in default_models.get(task_type, []) if model in available_models]
    log_structured("INFO", f"Modèles par défaut pour {task_type}", {"n_models": len(models)})
    return models

def process_training_results(results: List[Dict], task_type: str) -> Dict[str, Any]:
    """Traite et analyse les résultats d'entraînement de façon robuste"""
    analysis = {
        "successful_models": [],
        "failed_models": [],
        "best_model": None,
        "performance_summary": {},
        "warnings": [],
        "recommendations": []
    }
    
    try:
        # Séparation modèles réussis/échoués
        for result in results:
            if result.get('success', False) and not result.get('metrics', {}).get('error'):
                analysis["successful_models"].append(result)
            else:
                analysis["failed_models"].append(result)
        
        # Analyse des modèles réussis
        if analysis["successful_models"]:
            # Détermination métrique principale
            primary_metric = (
                'silhouette_score' if task_type == 'clustering' 
                else 'r2' if task_type == 'regression' 
                else 'accuracy'
            )
            
            # Recherche meilleur modèle
            valid_models = [
                m for m in analysis["successful_models"] 
                if m.get('metrics', {}).get(primary_metric) is not None
            ]
            
            if valid_models:
                analysis["best_model"] = max(
                    valid_models, 
                    key=lambda x: x['metrics'][primary_metric]
                )
            
            # Statistiques de performance
            metrics_data = {}
            for model in analysis["successful_models"]:
                for metric, value in model.get('metrics', {}).items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(value)
            
            analysis["performance_summary"] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                for metric, values in metrics_data.items()
            }
            
            # Recommandations
            if len(analysis["successful_models"]) == 0:
                analysis["recommendations"].append("Aucun modèle n'a réussi - vérifiez les données")
            elif len(analysis["successful_models"]) < len(results) / 2:
                analysis["recommendations"].append("Moins de la moitié des modèles ont réussi - optimisez la configuration")
            
            # Warnings spécifiques
            if analysis["performance_summary"].get(primary_metric, {}).get('std', 0) > 0.1:
                analysis["warnings"].append("Grande variance entre les modèles - données incohérentes")
        
        log_structured("INFO", "Analyse résultats entraînement", {
            "successful": len(analysis["successful_models"]),
            "failed": len(analysis["failed_models"]),
            "best_model": analysis["best_model"]["model_name"] if analysis["best_model"] else None
        })
        
    except Exception as e:
        log_structured("ERROR", f"Erreur analyse résultats: {str(e)[:100]}")
        analysis["warnings"].append(f"Erreur analyse: {str(e)[:100]}")
    
    return analysis

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
    st.session_state.warnings.extend(validation_result["warnings"])

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
    sys_metrics = get_system_metrics()
    color = "🟢" if sys_metrics["memory_percent"] < 70 else "🟡" if sys_metrics["memory_percent"] < TRAINING_CONSTANTS["HIGH_MEMORY_THRESHOLD"] else "🔴"
    st.metric(f"{color} RAM Sys", f"{sys_metrics['memory_percent']:.0f}%")

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
        st.session_state.warnings = []
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
            st.info(f"Classification: ≤{TRAINING_CONSTANTS['MAX_CLASSES']} valeurs uniques\nRégression: numérique, >{VALIDATION_CONSTANTS['MIN_UNIQUE_VALUES']} valeurs uniques")
            st.session_state.warnings.append("Aucune variable cible appropriée")
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
                    st.session_state.warnings.append(task_info["error"])
                else:
                    if selected_task_type == "classification":
                        st.success(f"✅ **Classification** ({task_info['n_classes']} classes)")
                        class_dist = df[target_column].value_counts()
                        if len(class_dist) <= 10:
                            st.bar_chart(class_dist, height=300, color=VISUALIZATION_CONSTANTS["BAR_CHART_COLOR"])
                            st.caption(f"Distribution des classes")
                        imbalance_info = detect_imbalance(df, target_column)
                        if imbalance_info.get("is_imbalanced", False):
                            st.warning(f"⚠️ Déséquilibre (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                            st.session_state.warnings.append(f"Déséquilibre classes (ratio: {imbalance_info['imbalance_ratio']:.2f})")
                        if task_info["warnings"]:
                            st.session_state.warnings.extend(task_info["warnings"])
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
                    column_types = cached_auto_detect_column_types(df)
                    recommended_features = column_types.get('numeric', []) + [
                        col for col in column_types.get('categorical', []) if df[col].nunique() <= VALIDATION_CONSTANTS["MAX_CATEGORICAL_UNIQUE"]
                    ]
                    recommended_features = [col for col in recommended_features if col != target_column and col in all_features][:TRAINING_CONSTANTS["MAX_FEATURES"]]
                    st.session_state.feature_list_for_ml_config = recommended_features
                    st.success(f"✅ {len(recommended_features)} features sélectionnées")
                    log_structured("INFO", "Features auto-sélectionnées", {"n_features": len(recommended_features)})
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
                    st.session_state.warnings.append("Nombre élevé de features")
                    st.info("Action: Réduisez le nombre de features ou activez PCA.")
            else:
                st.warning("⚠️ Aucune feature sélectionnée")
                st.info("Action: Sélectionnez au moins une variable.")
                st.session_state.warnings.append("Aucune feature sélectionnée")
        else:
            st.error("❌ Aucune feature disponible")
            st.info("Action: Vérifiez votre dataset.")
            st.session_state.warnings.append("Aucune feature disponible")
    
    else:  # Clustering
        st.session_state.target_column_for_ml_config = None
        st.success("✅ **Clustering Non Supervisé**")
        st.info("🔍 Le modèle identifiera des groupes naturels dans les données.")
        
        all_numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        if not all_numeric_features:
            st.error("❌ Aucune variable numérique pour le clustering")
            st.info("Action: Ajoutez des variables numériques au dataset.")
            st.session_state.warnings.append("Aucune variable numérique pour clustering")
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
                    st.session_state.warnings.extend(validation_result["warnings"])
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées")
                log_structured("INFO", "Variables clustering auto-sélectionnées", {"n_features": len(st.session_state.feature_list_for_ml_config)})
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
                    st.session_state.warnings.extend(validation_result["warnings"])
            
            if st.session_state.feature_list_for_ml_config:
                st.success(f"✅ {len(st.session_state.feature_list_for_ml_config)} variables sélectionnées")
                if len(st.session_state.feature_list_for_ml_config) < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
                    st.warning(f"⚠️ Minimum {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables pour clustering")
                    st.session_state.warnings.append(f"Moins de {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']} variables pour clustering")
                    st.info("Action: Ajoutez des variables numériques.")
                elif len(st.session_state.feature_list_for_ml_config) > TRAINING_CONSTANTS["MAX_FEATURES"]:
                    st.warning("⚠️ Nombre élevé de variables - risque de malédiction dimensionnelle")
                    st.session_state.warnings.append("Nombre élevé de variables pour clustering")
                    st.info("Action: Activez PCA ou réduisez les variables.")
                with st.expander("📈 Aperçu statistiques", expanded=False):
                    st.dataframe(df[st.session_state.feature_list_for_ml_config].describe().style.format("{:.3f}"), width="stretch")
            else:
                st.warning("⚠️ Aucune variable sélectionnée")
                st.info("Action: Sélectionnez des variables numériques.")
                st.session_state.warnings.append("Aucune variable sélectionnée pour clustering")

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
                column_types = cached_auto_detect_column_types(df)
                numeric_cols = df.select_dtypes(include='number').columns
                constant_cols = [col for col in numeric_cols if df[col].std() == 0] if len(numeric_cols) > 0 else []
                identifier_cols = [col for col in df.columns if df[col].nunique() == len(df)]
                if constant_cols or identifier_cols:
                    st.info(f"🧹 Nettoyage: {len(constant_cols)} colonnes constantes, {len(identifier_cols)} colonnes identifiantes détectées")
                    log_structured("INFO", "Colonnes à nettoyer détectées", {
                        "n_constant": len(constant_cols),
                        "n_identifier": len(identifier_cols)
                    })
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
                st.session_state.warnings.append("PCA recommandé pour nombre élevé de features")

        if task_type == 'clustering' and not st.session_state.preprocessing_choices.get('scale_features', True):
            st.error("❌ Normalisation critique pour le clustering!")
            st.info("Action: Activez la normalisation pour de meilleurs résultats.")
            st.session_state.warnings.append("Normalisation non activée pour clustering")
        
        if task_type == 'classification':
            st.subheader("⚖️ Déséquilibre")
            if st.session_state.target_column_for_ml_config:
                imbalance_info = detect_imbalance(df, st.session_state.target_column_for_ml_config)
                min_class_count = min(df[st.session_state.target_column_for_ml_config].value_counts())
                if imbalance_info.get("is_imbalanced", False):
                    st.warning(f"📉 Déséquilibre détecté (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})")
                    st.session_state.preprocessing_choices['use_smote'] = st.checkbox(
                        "Activer SMOTE",
                        value=st.session_state.preprocessing_choices.get('use_smote', True),
                        key="smote_checkbox",
                        help="Génère des données synthétiques pour équilibrer les classes minoritaires"
                    )
                    if st.session_state.preprocessing_choices['use_smote']:
                        with st.expander("⚙️ Paramètres SMOTE", expanded=False):
                            st.session_state.preprocessing_choices['smote_k_neighbors'] = st.number_input(
                                "Nombre de voisins (k)",
                                min_value=1,
                                max_value=min(20, min_class_count-1 if min_class_count > 1 else 1),
                                value=min(st.session_state.preprocessing_choices.get('smote_k_neighbors', 5), min_class_count-1 if min_class_count > 1 else 1),
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
                            if min_class_count < st.session_state.preprocessing_choices['smote_k_neighbors']:
                                st.warning(f"⚠️ Classe minoritaire trop petite ({min_class_count} samples) pour k={st.session_state.preprocessing_choices['smote_k_neighbors']}.")
                                st.session_state.warnings.append(f"Classe minoritaire trop petite pour SMOTE k={st.session_state.preprocessing_choices['smote_k_neighbors']}")
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
                                max_value=min(20, min_class_count-1 if min_class_count > 1 else 1),
                                value=min(st.session_state.preprocessing_choices.get('smote_k_neighbors', 5), min_class_count-1 if min_class_count > 1 else 1),
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
                            if min_class_count < st.session_state.preprocessing_choices['smote_k_neighbors']:
                                st.warning(f"⚠️ Classe minoritaire trop petite ({min_class_count} samples) pour k={st.session_state.preprocessing_choices['smote_k_neighbors']}.")
                                st.session_state.warnings.append(f"Classe minoritaire trop petite pour SMOTE k={st.session_state.preprocessing_choices['smote_k_neighbors']}")
            else:
                st.info("🔒 Variable cible requise pour activer SMOTE")
                st.session_state.preprocessing_choices['use_smote'] = False
                st.session_state.warnings.append("SMOTE désactivé: pas de cible")
        elif task_type == 'clustering':
            st.subheader("🔍 Clustering")
            st.session_state.preprocessing_choices['pca_preprocessing'] = st.checkbox(
                "Réduction dimension (PCA)",
                value=st.session_state.preprocessing_choices.get('pca_preprocessing', False),
                key="pca_preprocessing_checkbox",
                help="Réduit le bruit pour données haute dimension"
            )
            if st.session_state.preprocessing_choices['pca_preprocessing']:
                for model in st.session_state.selected_models_for_training:
                    if model in ['DBSCAN']:  # Modèles sensibles à PCA
                        st.warning(f"⚠️ PCA peut être incompatible avec {model}")
                        st.session_state.warnings.append(f"PCA potentiellement incompatible avec {model}")

# Étape 3: Sélection des Modèles
elif st.session_state.current_step == 3:
    st.header("🤖 Sélection des Modèles")
    task_type = st.session_state.get('task_type', 'classification')
    available_models = get_task_specific_models(task_type)
    
    if not available_models:
        st.error(f"❌ Aucun modèle disponible pour '{task_type}'")
        st.info("Action: Vérifiez le catalogue de modèles.")
        st.session_state.warnings.append(f"Aucun modèle pour {task_type}")
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
            if len(selected_models) > TRAINING_CONSTANTS["MAX_MODELS"]:
                st.warning(f"⚠️ Maximum {TRAINING_CONSTANTS['MAX_MODELS']} modèles recommandés")
                st.session_state.warnings.append(f"Trop de modèles sélectionnés ({len(selected_models)})")
                st.session_state.selected_models_for_training = selected_models[:TRAINING_CONSTANTS["MAX_MODELS"]]
            st.success(f"✅ {len(st.session_state.selected_models_for_training)} modèles sélectionnés")
            with st.expander("📋 Détails des modèles", expanded=False):
                for model_name in selected_models:
                    model_config = MODEL_CATALOG[task_type].get(model_name, {})
                    st.write(f"**{model_name}**")
                    st.caption(f"• {model_config.get('description', 'Description non disponible')}")
        else:
            st.warning("⚠️ Aucun modèle sélectionné")
            st.info("Action: Sélectionnez au moins un modèle.")
            st.session_state.warnings.append("Aucun modèle sélectionné")
            
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
            st.session_state.preprocessing_choices['optimization_method'] = st.selectbox(
                "Méthode",
                options=['Silhouette Score', 'Davies-Bouldin'] if task_type == 'clustering' else ['GridSearch', 'RandomSearch'],
                index=0,
                key="optimization_method_selector",
                help="Silhouette=qualité clusters, Davies-Bouldin=compacité, GridSearch=exhaustif, RandomSearch=rapide"
            )
        
        n_features = len(st.session_state.feature_list_for_ml_config)
        estimated_seconds = estimate_training_time(df, len(selected_models), task_type, optimize_hp, n_features, st.session_state.preprocessing_choices.get('use_smote', False))
        st.info(f"⏱️ Temps estimé: {max(1, estimated_seconds // 60)} minute(s)")
        
        if selected_models:
            resource_check = check_system_resources(df, len(selected_models))
            if not resource_check["has_enough_resources"]:
                st.error("❌ Ressources insuffisantes")
                for issue in resource_check["issues"]:
                    st.error(f"• {issue}")
                st.session_state.warnings.extend(resource_check["issues"])
            elif resource_check["warnings"]:
                st.warning("⚠️ Ressources limites")
                for warning in resource_check["warnings"]:
                    st.warning(f"• {warning}")
                st.session_state.warnings.extend(resource_check["warnings"])

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
            st.error("❌ Configuration incomplète:")
            for issue in config_issues:
                st.error(f"• {issue}")
                st.info("Action: Revenez aux étapes précédentes pour corriger.")
            st.session_state.warnings.extend(config_issues)
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
                'preprocessing_choices': st.session_state.preprocessing_choices,
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            try:
                status_text.text("📊 Préparation des données...")
                progress_bar.progress(10)
                
                n_models = len(st.session_state.selected_models_for_training)
                results = []
                
                def train_single_model(model_name, config):
                    """Entraîne un seul modèle sans duplication"""
                    try:
                        model_config = config.copy()
                        model_config['model_names'] = [model_name]
                        
                        model_results = safe_train_models(**model_config)
                        
                        # ✅ VALIDATION: Garantir que task_type est présent
                        if model_results:
                            for result in model_results:
                                if 'task_type' not in result or not result['task_type']:
                                    result['task_type'] = config.get('task_type', 'unknown')
                                    logger.log_structured("WARNING", f"task_type manquant, ajouté: {result['task_type']}")
                                
                                # ✅ VALIDATION BONUS: feature_names
                                if 'feature_names' not in result:
                                    result['feature_names'] = config.get('feature_list', [])
                        
                        return model_results
                        
                    except Exception as e:
                        log_structured("ERROR", f"Erreur entraînement {model_name}: {str(e)[:100]}")
                        return [{
                            "model_name": model_name,
                            "task_type": config.get('task_type', 'unknown'),  # ✅ DÉJÀ PRÉSENT
                            "metrics": {"error": str(e)[:100]},
                            "success": False,
                            "training_time": 0,
                            "X_sample": None,
                            "y_test": None,
                            "labels": None,
                            "feature_names": config.get('feature_list', []),
                            "warnings": []
                        }]

                # Dans la boucle d'entraînement principal, assurez-vous que chaque modèle n'est appelé qu'une fois
                if TRAINING_CONSTANTS["N_JOBS"] == -1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(st.session_state.selected_models_for_training)) as executor:
                        future_to_model = {
                            executor.submit(train_single_model, model_name, training_config): model_name 
                            for model_name in st.session_state.selected_models_for_training
                        }
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_model)):
                            model_name = future_to_model[future]
                            status_text.text(f"🔧 Entraînement {i+1}/{n_models}: {model_name}")
                            progress_bar.progress(10 + int((i / n_models) * 80))
                            try:
                                model_result = future.result()
                                if model_result:
                                    results.extend(model_result)
                            except Exception as e:
                                log_structured("ERROR", f"Échec entraînement parallèle {model_name}: {str(e)[:100]}")
                                results.append({
                                    "model_name": model_name,
                                    "metrics": {"error": str(e)[:100]},
                                    "success": False,
                                    "training_time": 0,
                                    "X_sample": None,
                                    "y_test": None,
                                    "labels": None,
                                    "feature_names": training_config['feature_list'],
                                    "warnings": []
                                })
                else:
                    # Version séquentielle
                    for i, model_name in enumerate(st.session_state.selected_models_for_training):
                        status_text.text(f"🔧 Entraînement {i+1}/{n_models}: {model_name}")
                        progress_bar.progress(10 + int((i / n_models) * 80))
                        model_result = train_single_model(model_name, training_config)
                        if model_result:
                            results.extend(model_result)
                
                status_text.text("✅ Finalisation...")
                progress_bar.progress(95)
                
                elapsed_time = time.time() - st.session_state.ml_last_training_time
                status_text.text(f"✅ Terminé en {elapsed_time:.1f}s")
                progress_bar.progress(100)
                
                st.session_state.ml_results = results
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = 0
                
                # Stocker runs MLflow
                try:
                    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(MLFLOW_CONSTANTS["EXPERIMENT_NAME"]).experiment_id])
                    st.session_state.mlflow_runs = [
                        {
                            "run_id": row["run_id"],
                            "metrics": {k.split(".")[1]: v for k, v in row.items() if k.startswith("metrics.") and pd.notnull(v)},
                            "tags": {k.split(".")[1]: v for k, v in row.items() if k.startswith("tags.") and pd.notnull(v)},
                            "start_time": row["start_time"],
                            "status": row["status"],
                            "model_name": row.get("tags.mlflow.runName", "Unknown")
                        } for _, row in runs.iterrows() if row.get("run_id") and row.get("tags.mlflow.runName")
                    ]
                    log_structured("INFO", f"{len(st.session_state.mlflow_runs)} runs MLflow stockés", {"n_runs": len(st.session_state.mlflow_runs)})
                    with mlflow.start_run(run_name=f"Training_{st.session_state.ml_session_id}", nested=True):
                        mlflow.log_metrics(get_system_metrics())
                        mlflow.log_param("warnings", json.dumps(st.session_state.warnings))
                except ImportError:
                    log_structured("INFO", "MLflow non installé, tracking désactivé")
                    st.session_state.mlflow_runs = []
                except Exception as e:
                    log_structured("ERROR", f"Échec récupération runs MLflow: {str(e)[:100]}")
                    st.session_state.mlflow_runs = []
                
                # Après l'entraînement dans l'étape 4
                successful_models = [r for r in results if r.get('success', False) and not r.get('metrics', {}).get('error')]

                # Analyse avancée des résultats
                results_analysis = process_training_results(results, task_type)

                with results_container.container():
                    if successful_models:
                        st.success(f"✅ {len(successful_models)}/{len(results)} modèles entraînés avec succès")
                        
                        # Affichage meilleur modèle
                        if results_analysis["best_model"]:
                            best_model = results_analysis["best_model"]
                            best_score = best_model['metrics'].get(
                                'silhouette_score' if task_type == 'clustering' else 'r2' if task_type == 'regression' else 'accuracy', 
                                0
                            )
                            st.info(f"🏆 **Meilleur modèle**: {best_model['model_name']} (Score: {best_score:.3f})")
                        
                        # Statistiques de performance
                        with st.expander("📊 Résultats détaillés", expanded=False):
                            for model in successful_models:
                                col1, col2, col3 = st.columns([2, 2, 1])
                                with col1:
                                    st.write(f"**{model['model_name']}**")
                                with col2:
                                    primary_metric = (
                                        'silhouette_score' if task_type == 'clustering' 
                                        else 'r2' if task_type == 'regression' 
                                        else 'accuracy'
                                    )
                                    score = model['metrics'].get(primary_metric, 'N/A')
                                    if isinstance(score, (int, float)):
                                        st.write(f"`{primary_metric}: {score:.3f}`")
                                with col3:
                                    st.write(f"`{model['training_time']:.1f}s`")
                        
                        # Recommandations
                        if results_analysis["recommendations"]:
                            with st.expander("💡 Recommandations", expanded=False):
                                for rec in results_analysis["recommendations"]:
                                    st.info(rec)
                    else:
                        st.error("❌ Aucun modèle n'a pu être entraîné")
                        if results_analysis["failed_models"]:
                            with st.expander("🔍 Détails des erreurs", expanded=True):
                                for model in results_analysis["failed_models"]:
                                    st.error(f"**{model['model_name']}**: {model.get('metrics', {}).get('error', 'Erreur inconnue')}")

                # Bouton de navigation vers les résultats
                if successful_models:
                    if st.button("📈 Voir l'analyse détaillée des résultats", type="primary", width='stretch'):
                        st.session_state.ml_results = results
                        st.session_state.results_analysis = results_analysis
                        st.switch_page("pages/3_evaluation.py")
                
                gc.collect()
                
            except Exception as e:
                st.session_state.ml_training_in_progress = False
                st.session_state.ml_error_count = st.session_state.get('ml_error_count', 0) + 1
                status_text.text("❌ Échec")
                progress_bar.progress(0)
                st.error(f"❌ Erreur: {str(e)[:100]}")
                st.info("Action: Vérifiez la configuration ou contactez le support.")
                log_structured("ERROR", f"Training échoué: {str(e)[:100]}", {"error": str(e)[:100]})
                st.session_state.warnings.append(f"Échec entraînement: {str(e)[:100]}")
    
    with col_reset:
        if st.button("🔄 Reset", width='stretch'):
            ml_keys_to_reset = [
                'target_column_for_ml_config', 'feature_list_for_ml_config',
                'selected_models_for_training', 'ml_results', 'task_type', 
                'previous_task_type', 'test_split_for_ml_config', 'optimize_hp_for_ml_config',
                'preprocessing_choices', 'ml_training_in_progress', 'ml_last_training_time', 
                'ml_error_count', 'mlflow_runs', 'model_performance_history', 'warnings'
            ]
            for key in ml_keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_step = 1
            st.success("Configuration réinitialisée")
            log_structured("INFO", "Configuration réinitialisée")
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
                    st.session_state.warnings.append("Variable cible non sélectionnée")
                elif not st.session_state.feature_list_for_ml_config:
                    st.error("Veuillez sélectionner au moins une variable")
                    st.session_state.warnings.append("Aucune variable explicative sélectionnée")
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
            "error_count": st.session_state.get('ml_error_count', 0),
            "warnings": st.session_state.get('warnings', [])
        })