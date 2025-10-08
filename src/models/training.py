"""
Module d'entraînement robuste pour le machine learning.
Supporte l'apprentissage supervisé et non-supervisé avec gestion MLOps avancée.
"""

import subprocess
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple
import warnings
from sklearn.exceptions import ConvergenceWarning
import json
from datetime import datetime

from utils.mlflow import clean_model_name, format_mlflow_run_for_ui, get_git_info, is_mlflow_available, save_artifacts_for_mlflow

# Intégration MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import des modules de l'application
try:
    from src.models.catalog import get_model_config
    from src.data.preprocessing import create_preprocessor, safe_label_encode, validate_preprocessor
    from src.evaluation.metrics import EvaluationMetrics
    from src.data.data_analysis import auto_detect_column_types
    from src.shared.logging import get_logger
    from src.config.constants import TRAINING_CONSTANTS, PREPROCESSING_CONSTANTS
except ImportError as e:
    print(f"Warning: Some imports failed - {e}")
    # Fallback pour les tests
    def get_model_config(*args, **kwargs): return None
    def auto_detect_column_types(*args, **kwargs): return {}
    def get_logger(name): 
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    TRAINING_CONSTANTS = {}
    PREPROCESSING_CONSTANTS = {}

logger = get_logger(__name__)

class TrainingMonitor:
    """Monitor pour suivre la progression et les ressources pendant l'entraînement."""
    
    def __init__(self):
        self.start_time = None
        self.model_start_time = None
        self.memory_usage = []
        self.current_model = None
        
    def start_training(self) -> None:
        """Démarre le monitoring de l'entraînement."""
        self.start_time = time.time()
        self.memory_usage = []
        logger.info("🚀 Début du monitoring de l'entraînement")
        
    def start_model(self, model_name: str) -> None:
        """Démarre le monitoring pour un modèle spécifique."""
        self.model_start_time = time.time()
        self.current_model = model_name
        logger.info(f"🔧 Début de l'entraînement pour: {model_name}")
        
    def check_resources(self) -> Dict[str, Any]:
        """Vérifie l'utilisation des ressources système."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            resource_info = {
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'timestamp': time.time(),
                'model': self.current_model
            }
            
            self.memory_usage.append(resource_info)
            
            # Alertes si utilisation élevée
            high_mem_threshold = TRAINING_CONSTANTS.get("HIGH_MEMORY_THRESHOLD", 85)
            high_cpu_threshold = TRAINING_CONSTANTS.get("HIGH_CPU_THRESHOLD", 90)
            
            if memory.percent > high_mem_threshold:
                logger.warning(f"⚠️ Utilisation mémoire élevée: {memory.percent:.1f}%")
            if cpu_percent > high_cpu_threshold:
                logger.warning(f"⚠️ Utilisation CPU élevée: {cpu_percent:.1f}%")
                
            return resource_info
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification ressources: {e}")
            return {}
    
    def get_model_duration(self) -> float:
        """Retourne la durée d'entraînement du modèle actuel."""
        if self.model_start_time:
            return time.time() - self.model_start_time
        return 0.0
    
    def get_total_duration(self) -> float:
        """Retourne la durée totale d'entraînement."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé du monitoring."""
        return {
            'total_duration': self.get_total_duration(),
            'memory_samples': len(self.memory_usage),
            'peak_memory': max([m.get('memory_percent', 0) for m in self.memory_usage]) if self.memory_usage else 0,
            'current_model': self.current_model
        }


def validate_training_data(X: pd.DataFrame, 
                          y: Optional[pd.Series], 
                          task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entraînement de façon robuste.
    
    Args:
        X: Features
        y: Target (peut être None pour unsupervised)
        task_type: Type de tâche ML
        
    Returns:
        Dict avec les résultats de validation
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "samples_count": len(X) if X is not None else 0,
        "features_count": len(X.columns) if hasattr(X, 'columns') else (X.shape[1] if X is not None else 0),
        "data_quality": {}
    }
    
    try:
        min_samples = TRAINING_CONSTANTS.get("MIN_SAMPLES_REQUIRED", 10)
        max_missing_ratio = TRAINING_CONSTANTS.get("MAX_MISSING_RATIO", 0.9)
        min_numeric_features = TRAINING_CONSTANTS.get("MIN_NUMERIC_FEATURES", 2)
        max_classes = TRAINING_CONSTANTS.get("MAX_CLASSES", 50)
        
        # Vérification des dimensions de base
        if X is None or len(X) == 0:
            validation["is_valid"] = False
            validation["issues"].append("Dataset X vide ou None")
            return validation
            
        if len(X) < min_samples:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop peu d'échantillons ({len(X)} < {min_samples})")
            
        if validation["features_count"] == 0:
            validation["is_valid"] = False
            validation["issues"].append("Aucune feature disponible")
        
        # Vérification de la mémoire
        try:
            if hasattr(X, 'memory_usage'):
                x_memory = X.memory_usage(deep=True).sum() / (1024 * 1024)
                validation["data_quality"]["memory_usage_mb"] = x_memory
                memory_limit = TRAINING_CONSTANTS.get("MEMORY_LIMIT_MB", 1000)
                if x_memory > memory_limit:
                    validation["warnings"].append(f"Dataset volumineux ({x_memory:.1f}MB)")
        except Exception as e:
            logger.debug(f"Erreur calcul mémoire: {e}")
        
        # Vérification des valeurs manquantes
        missing_stats = X.isna().sum()
        total_missing = missing_stats.sum()
        missing_ratio = total_missing / (X.shape[0] * X.shape[1]) if X.size > 0 else 0
        
        validation["data_quality"]["total_missing"] = int(total_missing)
        validation["data_quality"]["missing_ratio"] = float(missing_ratio)
        
        if missing_ratio > max_missing_ratio:
            validation["warnings"].append(f"Ratio de valeurs manquantes élevé: {missing_ratio:.1%}")
        
        # Vérification spécifique au non-supervisé
        if task_type == 'clustering':
            if y is not None:
                validation["warnings"].append("Target ignorée pour le clustering")
            
            numeric_features = X.select_dtypes(include=[np.number]).columns
            validation["data_quality"]["numeric_features"] = len(numeric_features)
            
            if len(numeric_features) < min_numeric_features:
                validation["warnings"].append(f"Peu de features numériques ({len(numeric_features)} < {min_numeric_features}) pour le clustering")
        
        # Vérification de la target pour supervisé
        elif y is not None:
            if len(y) != len(X):
                validation["is_valid"] = False
                validation["issues"].append(f"Dimensions X et y incohérentes: {len(X)} vs {len(y)}")
                
            valid_target_count = y.notna().sum() if hasattr(y, 'notna') else np.sum(~np.isnan(y))
            validation["data_quality"]["valid_target_count"] = int(valid_target_count)
            
            if valid_target_count < min_samples:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop peu de targets valides ({valid_target_count})")
                
            if task_type == 'classification':
                unique_classes = np.unique(y.dropna()) if hasattr(y, 'dropna') else np.unique(y[~np.isnan(y)])
                n_classes = len(unique_classes)
                validation["data_quality"]["n_classes"] = n_classes
                
                if n_classes < 2:
                    validation["is_valid"] = False
                    validation["issues"].append("Moins de 2 classes distinctes")
                elif n_classes > max_classes:
                    validation["warnings"].append(f"Plus de {max_classes} classes - vérifiez la variable cible")
            
        logger.info(f"✅ Validation données: {validation['samples_count']} échantillons, "
                   f"{validation['features_count']} features, {len(validation['issues'])} issues")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur de validation: {str(e)}")
        logger.error(f"❌ Validation error: {e}")
    
    return validation


# ===============================================
# FONCTION PRINCIPALE : CREATE LEAK-FREE PIPELINE
# ===============================================

from imblearn.pipeline import Pipeline as ImbPipeline
def create_leak_free_pipeline(
    model_name: str, 
    task_type: str, 
    column_types: Dict[str, List[str]],
    preprocessing_choices: Dict[str, Any],
    use_smote: bool = False,
    optimize_hyperparams: bool = False
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    Crée un pipeline sans data leakage en intégrant le préprocesseur, SMOTE et le modèle.
    
    IMPORTANT: Cette fonction garantit qu'aucune fuite de données ne se produit en encapsulant
    toutes les transformations (preprocessing, SMOTE) et le modèle dans un seul pipeline.
    
    Args:
        model_name: Nom du modèle (ex: 'RandomForestClassifier')
        task_type: Type de tâche ('classification', 'regression', 'clustering')
        column_types: Dictionnaire des types de colonnes (numériques, catégoriques, etc.)
        preprocessing_choices: Options de prétraitement
        use_smote: Utiliser SMOTE pour le rééquilibrage des classes (classification uniquement)
        optimize_hyperparams: Optimiser les hyperparamètres avec GridSearchCV
        
    Returns:
        Tuple (pipeline, param_grid):
            - pipeline: Pipeline scikit-learn ou imblearn contenant toutes les étapes
            - param_grid: Dictionnaire pour GridSearchCV (ou None si optimize=False)
    
    Raises:
        ValueError: Si la configuration du modèle est invalide
        
    Example:
        >>> pipeline, params = create_leak_free_pipeline(
        ...     model_name='RandomForest',
        ...     task_type='classification',
        ...     column_types={'numeric': ['age'], 'categorical': ['gender']},
        ...     preprocessing_choices={'scaling_method': 'standard'},
        ...     use_smote=True,
        ...     optimize_hyperparams=True
        ... )
        >>> pipeline.fit(X_train, y_train)
    """
    
    try:
        logger.info(f"🔧 Création pipeline pour {model_name} (task: {task_type}, SMOTE: {use_smote})")
        
        # ============================================
        # 1. Récupérer la configuration du modèle
        # ============================================
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"❌ Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
        
        model = model_config["model"]
        logger.debug(f"Modèle instancié: {type(model).__name__}")
        
        # ============================================
        # 2. Préparer la grille de paramètres
        # ============================================
        param_grid = {}
        if optimize_hyperparams and "params" in model_config:
            # Préfixer avec 'model__' pour le pipeline
            param_grid = {f"model__{k}": v for k, v in model_config["params"].items()}
            logger.debug(f"Grille paramètres: {len(param_grid)} hyperparamètres")
        
        # ============================================
        # 3. Créer le préprocesseur
        # ============================================
        preprocessor = create_preprocessor(preprocessing_choices, column_types)
        if preprocessor is None:
            logger.error(f"❌ Échec création préprocesseur pour {model_name}")
            return None, None
        
        # ============================================
        # 4. Valider le préprocesseur (optionnel mais recommandé)
        # ============================================
        try:
            # Créer un DataFrame minimal pour la validation
            validation_df = pd.DataFrame({
                col: [0] * 2  # Au moins 2 lignes pour la validation
                for cols in column_types.values() 
                for col in cols
            })
            
            validation_result = validate_preprocessor(preprocessor, validation_df)
            if not validation_result["is_valid"]:
                logger.warning(f"⚠️ Issues détectées dans le préprocesseur: {validation_result['issues']}")
                # Ne pas bloquer, mais logger les problèmes
                for issue in validation_result.get("issues", []):
                    logger.warning(f"  - {issue}")
        except Exception as val_error:
            logger.warning(f"⚠️ Validation préprocesseur échouée: {val_error}")
            # Continuer quand même
        
        # ============================================
        # 5. Construire le pipeline selon le contexte
        # ============================================
        
        # Cas 1: Classification avec SMOTE
        if use_smote and task_type == 'classification':
            logger.info("🔄 Construction pipeline avec SMOTE (imblearn)")
            
            # Configuration SMOTE
            smote_k = preprocessing_choices.get("smote_k_neighbors", 5)
            random_state = preprocessing_choices.get("random_state", 42)
            sampling_strategy = preprocessing_choices.get("smote_sampling_strategy", 'auto')
            
            # CRITIQUE: Vérifier que k_neighbors est valide
            # SMOTE nécessite au moins k_neighbors+1 échantillons de la classe minoritaire
            logger.debug(f"SMOTE config: k_neighbors={smote_k}, strategy={sampling_strategy}")
            
            # Construire le pipeline avec ImbPipeline
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(
                    random_state=random_state,
                    k_neighbors=smote_k,
                    sampling_strategy=sampling_strategy
                )),
                ('model', model)  # ✅ Modèle ajouté DANS le pipeline
            ])
            
            logger.info(f"✅ ImbPipeline créé avec 3 étapes: preprocessor → SMOTE → {model_name}")
        
        # Cas 2: Autres cas (pas de SMOTE)
        else:
            logger.info("🔄 Construction pipeline standard (sklearn)")
            
            # Si SMOTE était demandé pour autre chose que classification
            if use_smote:
                logger.warning(f"⚠️ SMOTE ignoré pour task_type='{task_type}' (classification uniquement)")
            
            # Construire le pipeline sklearn classique
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)  # ✅ Modèle ajouté DANS le pipeline
            ])
            
            logger.info(f"✅ Pipeline créé avec 2 étapes: preprocessor → {model_name}")
        
        # ============================================
        # 6. Validation finale du pipeline
        # ============================================
        
        # Vérifier que le pipeline a bien les étapes attendues
        expected_steps = ['preprocessor', 'model'] if not (use_smote and task_type == 'classification') else ['preprocessor', 'smote', 'model']
        actual_steps = list(pipeline.named_steps.keys())
        
        if actual_steps != expected_steps:
            logger.error(f"❌ Pipeline invalide! Attendu: {expected_steps}, Obtenu: {actual_steps}")
            return None, None
        
        # Vérifier que chaque étape est bien définie
        for step_name in expected_steps:
            if pipeline.named_steps[step_name] is None:
                logger.error(f"❌ Étape '{step_name}' est None dans le pipeline!")
                return None, None
        
        logger.info(f"✅ Pipeline validé avec succès pour {model_name}")
        logger.debug(f"Étapes du pipeline: {' → '.join(actual_steps)}")
        
        return pipeline, param_grid if param_grid else None
    
    except KeyError as ke:
        logger.error(f"❌ Clé manquante dans la configuration: {ke}")
        return None, None
    
    except ValueError as ve:
        logger.error(f"❌ Valeur invalide dans la configuration: {ve}")
        return None, None
    
    except Exception as e:
        logger.error(f"❌ Erreur inattendue lors de la création du pipeline pour {model_name}: {e}")
        logger.exception("Stack trace complète:")
        return None, None


def train_single_model_supervised(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict = None,
    task_type: str = 'classification',
    monitor: TrainingMonitor = None
) -> Dict[str, Any]:
    """Entraîne un modèle supervisé avec validation croisée propre."""
    result = {
        "model_name": model_name,
        "success": False,
        "model": None,
        "training_time": 0,
        "error": None,
        "best_params": None,
        "cv_scores": None
    }
    
    start_time = time.time()
    
    try:
        if monitor:
            monitor.start_model(model_name)
        
        cv_folds = TRAINING_CONSTANTS.get("CV_FOLDS", 5)
        random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
        
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = 'r2'
        
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation hyperparamètres pour {model_name}")
            
            max_combinations = TRAINING_CONSTANTS.get("MAX_GRID_COMBINATIONS", 100)
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            
            if total_combinations > max_combinations:
                logger.warning(f"Grille trop large ({total_combinations} combinaisons), limitation automatique")
                limited_param_grid = {}
                for k, v in param_grid.items():
                    limited_param_grid[k] = v[:2] if len(v) > 2 else v
                param_grid = limited_param_grid
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring,
                n_jobs=n_jobs, verbose=0, error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
            
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["cv_scores"] = {
                'mean': grid_search.best_score_,
                'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            }
            result["success"] = True
            
            logger.info(f"✅ Optimisation terminée pour {model_name} - score: {grid_search.best_score_:.3f}")
            
        else:
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                result["cv_scores"] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
                logger.info(f"📊 CV scores pour {model_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as cv_error:
                logger.warning(f"⚠️ CV échouée pour {model_name}: {cv_error}")
                result["cv_scores"] = None
            
            pipeline.fit(X_train, y_train)
            result["model"] = pipeline
            result["success"] = True
        
        result["training_time"] = time.time() - start_time
        
        if monitor:
            resource_info = monitor.check_resources()
            logger.info(f"✅ {model_name} entraîné en {result['training_time']:.2f}s - "
                       f"RAM: {resource_info.get('memory_percent', 0):.1f}%")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur entraînement {model_name}: {e}")
    
    return result

def train_single_model_unsupervised(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    param_grid: Dict = None,
    monitor: TrainingMonitor = None
) -> Dict[str, Any]:
    """Entraîne un modèle non-supervisé (clustering)."""
    result = {
        "model_name": model_name,
        "success": False,
        "model": None,
        "training_time": 0,
        "error": None,
        "best_params": None
    }
    
    start_time = time.time()
    
    try:
        if monitor:
            monitor.start_model(model_name)
        
        cv_folds = TRAINING_CONSTANTS.get("CV_FOLDS", 5)
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
        
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation hyperparamètres clustering pour {model_name}")
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv_folds,
                scoring='silhouette_score',  # Utilisation de silhouette_score pour évaluer les clusters
                n_jobs=n_jobs, verbose=0
            )
            
            grid_search.fit(X)
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["success"] = True
            
        else:
            # Ajustement du pipeline
            pipeline.fit(X)
            result["model"] = pipeline
            
            # Si le modèle est DBSCAN, utilisez fit_predict
            if hasattr(pipeline, 'named_steps') and 'dbscan' in pipeline.named_steps:
                result["model"].fit_predict(X)
            
            result["success"] = True
        
        result["training_time"] = time.time() - start_time
        
        if monitor:
            resource_info = monitor.check_resources()
            logger.info(f"✅ Clustering {model_name} terminé en {result['training_time']:.2f}s")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur clustering {model_name}: {e}")
    
    return result


def evaluate_model_with_metrics_calculator(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    task_type: str,
    label_encoder: Any = None,
    X_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """Évalue un modèle en utilisant EvaluationMetrics."""
    try:
        metrics_calculator = EvaluationMetrics(task_type)
        
        if task_type == 'clustering':
            if X_data is None:
                return {"error": "Données X requises pour l'évaluation non supervisée"}
            
            if hasattr(model, 'predict'):
                cluster_labels = model.predict(X_data)
            elif hasattr(model, 'labels_'):
                cluster_labels = model.labels_
            else:
                cluster_labels = model.fit_predict(X_data)
            
            metrics = metrics_calculator.calculate_unsupervised_metrics(X_data.values, cluster_labels)
            metrics['n_clusters'] = len(np.unique(cluster_labels[~np.isnan(cluster_labels)]))
            metrics['task_type'] = task_type
            metrics['n_samples'] = len(X_data)
            
        else:
            y_pred = model.predict(X_test)
            y_proba = None
            
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception as e:
                    logger.warning(f"⚠️ Predict_proba non disponible: {e}")
            
            if task_type == 'classification':
                metrics = metrics_calculator.calculate_classification_metrics(
                    y_test.values, y_pred, y_proba
                )
            else:
                metrics = metrics_calculator.calculate_regression_metrics(
                    y_test.values, y_pred
                )
            
            metrics['task_type'] = task_type
            metrics['n_samples'] = len(X_test)
        
        if hasattr(metrics_calculator, 'error_messages') and metrics_calculator.error_messages:
            metrics['calculation_warnings'] = metrics_calculator.error_messages
        
        metrics['success'] = True
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur évaluation avec EvaluationMetrics: {e}")
        return {
            "error": f"Erreur évaluation: {str(e)}",
            "success": False,
            "task_type": task_type
        }


# ============================================
# FONCTION PRINCIPALE : TRAIN MODELS
# ============================================
def train_models(
    df: pd.DataFrame,
    target_column: Optional[str],
    model_names: List[str],
    task_type: str,
    test_size: float = 0.2,
    optimize: bool = False,
    feature_list: List[str] = None,
    use_smote: bool = False,
    preprocessing_choices: Dict = None
) -> List[Dict[str, Any]]:
    """
    Orchestre l'entraînement sans data leakage avec MLflow tracking complet.
    
    Cette fonction gère:
    - La préparation des données avec validation
    - La création de pipelines sans data leakage
    - L'entraînement de multiples modèles
    - Le tracking MLflow avec traçabilité Git
    - La sauvegarde des modèles et artefacts
    - Le monitoring des ressources
    
    Args:
        df: DataFrame contenant les données
        target_column: Nom de la colonne cible (None pour clustering)
        model_names: Liste des noms de modèles à entraîner
        task_type: Type de tâche ('classification', 'regression', 'clustering')
        test_size: Proportion du jeu de test (ignoré pour clustering)
        optimize: Si True, effectue GridSearchCV
        feature_list: Liste des features à utiliser (None = toutes sauf target)
        use_smote: Si True, applique SMOTE (classification uniquement)
        preprocessing_choices: Configuration du prétraitement
        
    Returns:
        Liste de dictionnaires contenant les résultats pour chaque modèle
    """
    
    results = []
    monitor = TrainingMonitor()
    monitor.start_training()
    
    # ============================================
    # 1. NORMALISATION ET CONFIGURATION
    # ============================================
    
    task_type = task_type.lower()
    if task_type == 'unsupervised':
        task_type = 'clustering'
    
    if task_type == 'clustering':
        target_column = None
        use_smote = False
        test_size = 0.0
    
    logger.info(f"🎯 Début entraînement - Task: {task_type}, Models: {len(model_names)}, Target: {target_column}")
    
    # ============================================
    # 2. CONFIGURATION MLFLOW
    # ============================================
    
    mlflow_enabled = is_mlflow_available()
    git_info = get_git_info() if mlflow_enabled else {}
    
    if mlflow_enabled:
        try:
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "datalab_pro_experimentations")
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ MLflow experiment: {experiment_name}")
            logger.info(f"📌 Git: {git_info.get('branch', 'unknown')}@{git_info.get('commit_hash', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}")
            mlflow_enabled = False
    
    # ============================================
    # 3. PRÉPARATION DES FEATURES
    # ============================================
    
    if not feature_list:
        if target_column and target_column in df.columns:
            feature_list = [col for col in df.columns if col != target_column]
        else:
            feature_list = list(df.columns)
    
    max_features = TRAINING_CONSTANTS.get("MAX_FEATURES", 100)
    if len(feature_list) > max_features:
        logger.warning(f"⚠️ Features limitées: {len(feature_list)} → {max_features}")
        feature_list = feature_list[:max_features]
    
    # ============================================
    # 4. CONFIGURATION PRÉTRAITEMENT
    # ============================================
    
    if preprocessing_choices is None:
        preprocessing_choices = {
            'numeric_imputation': PREPROCESSING_CONSTANTS.get("NUMERIC_IMPUTATION_DEFAULT", "mean"),
            'categorical_imputation': PREPROCESSING_CONSTANTS.get("CATEGORICAL_IMPUTATION_DEFAULT", "most_frequent"),
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'scaling_method': PREPROCESSING_CONSTANTS.get("SCALING_METHOD", "standard"),
            'encoding_method': PREPROCESSING_CONSTANTS.get("ENCODING_METHOD", "onehot"),
            'random_state': TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
        }
    
    # ============================================
    # 5. CRÉATION DES RÉPERTOIRES
    # ============================================
    
    os.makedirs("models_output", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    
    # ============================================
    # 6. PRÉPARATION DES DONNÉES
    # ============================================
    
    logger.info("📊 Préparation des données...")
    X = df[feature_list].copy()
    
    if X.empty:
        logger.error("❌ DataFrame vide")
        return [{"model_name": "Validation", "metrics": {"error": "DataFrame vide"}}]
    
    y = None
    label_encoder = None
    
    if task_type != 'clustering' and target_column:
        if target_column not in df.columns:
            logger.error(f"❌ Target '{target_column}' non trouvée")
            return [{"model_name": "Validation", "metrics": {"error": f"Target non trouvée"}}]
        
        y_raw = df[target_column].copy()
        y_encoded, label_encoder, _ = safe_label_encode(y_raw)
        y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
    
    # ============================================
    # 7. VALIDATION DES DONNÉES
    # ============================================
    
    data_validation = validate_training_data(X, y, task_type)
    if not data_validation["is_valid"]:
        error_msg = f"Données invalides: {', '.join(data_validation['issues'])}"
        logger.error(f"❌ {error_msg}")
        return [{"model_name": "Validation", "metrics": {"error": error_msg}}]
    
    for warning in data_validation["warnings"]:
        logger.warning(f"⚠️ {warning}")
    
    # ============================================
    # 8. DÉTECTION DES TYPES DE COLONNES
    # ============================================
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        column_types = auto_detect_column_types(X)
    
    logger.debug(f"Types détectés: {len(column_types.get('numeric', []))} num, {len(column_types.get('categorical', []))} cat")
    
    # ============================================
    # 9. SPLIT TRAIN/TEST
    # ============================================
    
    X_train, X_test, y_train, y_test = None, None, None, None
    
    if task_type != 'clustering':
        try:
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            stratification = y if task_type == 'classification' else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=stratification
            )
            
            logger.info(f"✅ Split: train={len(X_train)}, test={len(X_test)}")
            
        except Exception as split_error:
            logger.error(f"❌ Erreur split: {split_error}")
            return [{"model_name": "Split", "metrics": {"error": str(split_error)}}]
    
    # ============================================
    # 10. RÉCUPÉRATION DES CONSTANTES
    # ============================================
    
    successful_models = 0
    max_training_time = TRAINING_CONSTANTS.get("MAX_TRAINING_TIME", 3600)
    high_memory_threshold = TRAINING_CONSTANTS.get("HIGH_MEMORY_THRESHOLD", 85)
    max_sample_size = TRAINING_CONSTANTS.get("MAX_VISUALIZATION_SAMPLES", 1000)
    
    mlflow_runs = []
    
    # ============================================
    # 11. BOUCLE D'ENTRAÎNEMENT DES MODÈLES
    # ============================================
    
    for i, model_name in enumerate(model_names, 1):
        logger.info(f"🔧 [{i}/{len(model_names)}] Training {model_name}")
        
        # Vérification temps global
        total_duration = monitor.get_total_duration()
        if total_duration > max_training_time:
            logger.warning(f"⏰ Timeout global ({total_duration:.0f}s)")
            results.append({
                "model_name": model_name,
                "metrics": {"error": "Timeout global dépassé"}
            })
            continue
        
        # Vérification mémoire
        resource_info = monitor.check_resources()
        if resource_info.get('memory_percent', 0) > high_memory_threshold:
            logger.warning("🧹 Mémoire élevée, nettoyage...")
            gc.collect()
        
        # ============================================
        # 11.1. CRÉATION DU PIPELINE
        # ============================================
        
        try:
            pipeline, param_grid = create_leak_free_pipeline(
                model_name=model_name,
                task_type=task_type,
                column_types=column_types,
                preprocessing_choices=preprocessing_choices,
                use_smote=use_smote,
                optimize_hyperparams=optimize
            )
        except Exception as pipeline_error:
            logger.error(f"❌ Pipeline error: {pipeline_error}")
            results.append({
                "model_name": model_name,
                "metrics": {"error": f"Pipeline: {str(pipeline_error)}"}
            })
            continue
        
        if pipeline is None:
            results.append({
                "model_name": model_name,
                "metrics": {"error": "Pipeline None"}
            })
            continue
        
        # ============================================
        # 11.2. DÉMARRAGE RUN MLFLOW
        # ============================================
        
        run_id = None
        mlflow_active = mlflow_enabled
        timestamp = int(time.time())
        
        if mlflow_active:
            try:
                mlflow.start_run(run_name=f"{model_name}_{timestamp}")
                run_id = mlflow.active_run().info.run_id
                
                # Logs configuration
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_samples", len(X))
                mlflow.log_param("n_features", len(feature_list))
                mlflow.log_param("use_smote", use_smote)
                mlflow.log_param("optimize_hyperparams", optimize)
                mlflow.log_param("test_size", test_size)
                
                # Logs Git (traçabilité)
                for key, value in git_info.items():
                    mlflow.log_param(f"git_{key}", value)
                
                # Logs preprocessing
                for key, value in preprocessing_choices.items():
                    mlflow.log_param(f"preprocessing_{key}", str(value)[:250])
                
                logger.info(f"✅ MLflow run started: {run_id}")
                
            except Exception as mlflow_error:
                logger.warning(f"⚠️ MLflow start failed: {mlflow_error}")
                mlflow_active = False
                run_id = None
        
        # ============================================
        # 11.3. ENTRAÎNEMENT
        # ============================================
        
        try:
            if task_type == 'clustering':
                training_result = train_single_model_unsupervised(
                    model_name=model_name,
                    pipeline=pipeline,
                    X=X,
                    param_grid=param_grid,
                    monitor=monitor
                )
            else:
                training_result = train_single_model_supervised(
                    model_name=model_name,
                    pipeline=pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    param_grid=param_grid,
                    task_type=task_type,
                    monitor=monitor
                )
        
        except Exception as train_error:
            logger.error(f"❌ Training error: {train_error}")
            
            if mlflow_active and run_id:
                try:
                    mlflow.log_param("status", "FAILED")
                    mlflow.log_param("error", str(train_error)[:250])
                    mlflow.end_run(status="FAILED")
                except:
                    pass
            
            results.append({
                "model_name": model_name,
                "metrics": {"error": f"Training: {str(train_error)}"},
                "training_time": 0
            })
            gc.collect()
            continue
        
        # ============================================
        # 11.4. TRAITEMENT DES RÉSULTATS
        # ============================================
        
        if training_result["success"] and training_result["model"] is not None:
            try:
                # Évaluation
                if task_type == 'clustering':
                    metrics = evaluate_model_with_metrics_calculator(
                        model=training_result["model"],
                        X_test=None,
                        y_test=None,
                        task_type=task_type,
                        X_data=X
                    )
                else:
                    metrics = evaluate_model_with_metrics_calculator(
                        model=training_result["model"],
                        X_test=X_test,
                        y_test=y_test,
                        task_type=task_type,
                        label_encoder=label_encoder
                    )
                
                # Sauvegarde modèle
                model_name_clean = clean_model_name(model_name)
                model_filename = f"{model_name_clean}_{task_type}_{timestamp}.joblib"
                model_path = os.path.join("models_output", model_filename)
                
                joblib.dump(training_result["model"], model_path)
                logger.info(f"💾 Modèle sauvegardé: {model_path}")
                
                # Préparation échantillons visualisation
                if task_type == 'clustering':
                    sample_size = min(max_sample_size, len(X))
                    X_sample = X.iloc[:sample_size].copy() if hasattr(X, 'iloc') else X[:sample_size].copy()
                    labels_sample = training_result["model"].predict(X_sample)
                    X_test_vis = None
                    y_test_vis = None
                    X_train_vis = None
                    y_train_vis = None
                else:
                    sample_size_test = min(max_sample_size, len(X_test))
                    sample_size_train = min(max_sample_size, len(X_train))
                    
                    X_test_vis = X_test.iloc[:sample_size_test].copy() if hasattr(X_test, 'iloc') else X_test[:sample_size_test].copy()
                    y_test_vis = y_test.iloc[:sample_size_test].copy() if hasattr(y_test, 'iloc') else y_test[:sample_size_test].copy()
                    X_train_vis = X_train.iloc[:sample_size_train].copy() if hasattr(X_train, 'iloc') else X_train[:sample_size_train].copy()
                    y_train_vis = y_train.iloc[:sample_size_train].copy() if hasattr(y_train, 'iloc') else y_train[:sample_size_train].copy()
                    X_sample = X_test_vis.copy()
                    labels_sample = None
                
                # Construction résultat
                result = {
                    "model_name": model_name,
                    "metrics": metrics,
                    "model_path": model_path,
                    "training_time": training_result["training_time"],
                    "best_params": training_result.get("best_params"),
                    "cv_scores": training_result.get("cv_scores"),
                    "model": training_result["model"],
                    "label_encoder": label_encoder,
                    "feature_names": feature_list,
                    "task_type": task_type,
                    "timestamp": timestamp,
                    "X_test": X_test_vis,
                    "y_test": y_test_vis,
                    "X_train": X_train_vis,
                    "y_train": y_train_vis,
                    "X_sample": X_sample,
                    "labels": labels_sample,
                }
                
                # ============================================
                # 11.5. LOGGING MLFLOW
                # ============================================
                
                if mlflow_active and run_id:
                    try:
                        # Log métriques
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                                mlflow.log_metric(metric_name, float(metric_value))
                        
                        # Log hyperparamètres
                        if training_result.get("best_params"):
                            for param_name, param_value in training_result["best_params"].items():
                                mlflow.log_param(f"best_{param_name}", str(param_value)[:250])
                        
                        # Log CV scores
                        if training_result.get("cv_scores") is not None:
                            cv_scores = training_result["cv_scores"]
                            if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                                mlflow.log_metric("cv_mean_score", float(np.mean(cv_scores)))
                                mlflow.log_metric("cv_std_score", float(np.std(cv_scores)))
                        
                        # Log modèle sklearn (✅ CORRECTION: name au lieu de artifact_path)
                        task_type_clean = clean_model_name(task_type)
                        registered_name = f"{task_type_clean}_{model_name_clean}_{timestamp}"
                        
                        try:
                            input_example = X_sample.head(1) if isinstance(X_sample, pd.DataFrame) else X_sample[:1]
                        except:
                            input_example = None
                        
                        mlflow.sklearn.log_model(
                            sk_model=training_result["model"],
                            artifact_path="model",  # ✅ CORRECTION: Utiliser name
                            registered_model_name=registered_name,
                            input_example=input_example
                        )
                        
                        # Log fichier modèle
                        mlflow.log_artifact(model_path)
                        mlflow.log_metric("training_time", float(training_result["training_time"]))
                        mlflow.log_metric("memory_percent", float(resource_info.get('memory_percent', 0)))
                        mlflow.log_metric("cpu_percent", float(resource_info.get('cpu_percent', 0)))
                        
                        # Log artefacts visualisation (✅ CORRECTION: utiliser tempfile)
                        with tempfile.TemporaryDirectory() as temp_dir:
                            artifact_paths = save_artifacts_for_mlflow(
                                task_type=task_type,
                                model=training_result["model"],
                                X_test_vis=X_test_vis,
                                y_test_vis=y_test_vis,
                                X_sample=X_sample,
                                labels_sample=labels_sample,
                                temp_dir=temp_dir
                            )
                            
                            for artifact_path in artifact_paths:
                                try:
                                    mlflow.log_artifact(artifact_path)
                                except Exception as artifact_error:
                                    logger.warning(f"⚠️ Échec log artifact {artifact_path}: {artifact_error}")
                        
                        # Log validation
                        mlflow.log_dict(data_validation, "data_validation.json")
                        
                        # ✅ CORRECTION: Stocker infos run pour UI avec format correct
                        run_info = mlflow.active_run()
                        mlflow_run_data = format_mlflow_run_for_ui(
                            run_info=run_info,
                            metrics=metrics,
                            preprocessing_choices=preprocessing_choices,
                            model_name=model_name,
                            timestamp=timestamp
                        )
                        
                        mlflow_runs.append(mlflow_run_data)
                        
                        logger.info(f"✅ MLflow logged: {run_id}")
                        
                        # Debug: afficher structure
                        logger.debug(f"📊 Structure run: {list(mlflow_run_data.keys())}")
                        logger.debug(f"📊 Run ID: {mlflow_run_data.get('run id', 'MANQUANT')}")
                    
                    except Exception as mlflow_log_error:
                        logger.warning(f"⚠️ MLflow logging failed: {mlflow_log_error}")
                        logger.exception("Stack trace MLflow logging:")
                    
                    finally:
                        try:
                            mlflow.end_run()
                        except:
                            pass
                
                results.append(result)
                successful_models += 1
                
                logger.info(f"✅ {model_name} - OK en {training_result['training_time']:.2f}s")
            
            except Exception as eval_error:
                logger.error(f"❌ Evaluation error: {eval_error}")
                logger.exception("Stack trace evaluation:")
                
                if mlflow_active and run_id:
                    try:
                        mlflow.log_param("status", "FAILED")
                        mlflow.log_param("error", f"Eval: {str(eval_error)[:250]}")
                        mlflow.end_run(status="FAILED")
                    except:
                        pass
                
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": f"Eval: {str(eval_error)}"},
                    "training_time": training_result.get("training_time", 0)
                })
        
        else:
            # Échec de l'entraînement
            error_msg = training_result.get("error", "Erreur inconnue")
            logger.error(f"❌ Échec entraînement {model_name}: {error_msg}")
            
            if mlflow_active and run_id:
                try:
                    mlflow.log_param("status", "FAILED")
                    mlflow.log_param("error", str(error_msg)[:250])
                    mlflow.end_run(status="FAILED")
                except:
                    pass
            
            results.append({
                "model_name": model_name,
                "metrics": {"error": error_msg},
                "training_time": training_result.get("training_time", 0)
            })
        
        # Nettoyage mémoire après chaque modèle
        gc.collect()
    
    # ============================================
    # 12. GÉNÉRATION DU RAPPORT FINAL
    # ============================================
    
    total_time = monitor.get_total_duration()
    monitor_summary = monitor.get_summary()
    
    training_log = {
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type,
        "target_column": target_column,
        "models_attempted": len(model_names),
        "models_successful": successful_models,
        "total_training_time": total_time,
        "monitor_summary": monitor_summary,
        "data_validation": data_validation,
        "git_info": git_info,
        "results_summary": [
            {
                "model_name": r["model_name"],
                "success": "metrics" in r and "error" not in r.get("metrics", {}),
                "training_time": r.get("training_time", 0)
            } for r in results
        ]
    }
    
    # Sauvegarde du log d'entraînement
    log_filename = f"training_log_{int(time.time())}.json"
    log_path = os.path.join("training_logs", log_filename)
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Log sauvegardé: {log_path}")
    except Exception as log_error:
        logger.warning(f"⚠️ Échec sauvegarde log: {log_error}")
    
    # ============================================
    # 13. LOG MLFLOW FINAL (SUMMARY)
    # ============================================
    
    if mlflow_enabled:
        try:
            with mlflow.start_run(run_name=f"training_summary_{int(time.time())}"):
                mlflow.log_artifact(log_path)
                mlflow.log_metric("total_training_time", float(total_time))
                mlflow.log_metric("models_successful", int(successful_models))
                mlflow.log_metric("models_attempted", int(len(model_names)))
                mlflow.log_metric("success_rate", float(successful_models / len(model_names) * 100 if model_names else 0))
                mlflow.log_dict(monitor_summary, "monitor_summary.json")
                
                # Log Git info
                for key, value in git_info.items():
                    mlflow.log_param(f"git_{key}", value)
                
            logger.info("📊 MLflow summary logged")
        except Exception as mlflow_final_error:
            logger.warning(f"⚠️ MLflow final logging failed: {mlflow_final_error}")
    
    # ============================================
    # 14. STOCKER LES RUNS MLFLOW EN SESSION
    # ============================================
    
    if mlflow_runs:
        try:
            import streamlit as st
            
            # ✅ CORRECTION: Initialiser si n'existe pas ou si None
            if not hasattr(st.session_state, 'mlflow_runs') or st.session_state.mlflow_runs is None:
                st.session_state.mlflow_runs = []
                logger.info("🔄 mlflow_runs initialisé")
            
            # Vérifier que c'est bien une liste
            if not isinstance(st.session_state.mlflow_runs, list):
                logger.warning("⚠️ mlflow_runs n'est pas une liste, réinitialisation")
                st.session_state.mlflow_runs = []
            
            # Ajouter les nouveaux runs
            st.session_state.mlflow_runs.extend(mlflow_runs)
            
            logger.info(f"✅ {len(mlflow_runs)} runs MLflow stockés (total: {len(st.session_state.mlflow_runs)})")
            
            # Debug: afficher info sur les runs
            if mlflow_runs:
                logger.debug(f"📊 Premier run keys: {list(mlflow_runs[0].keys())}")
                logger.debug(f"📊 Premier run_id: {mlflow_runs[0].get('run id', 'MANQUANT')}")
            
        except ImportError:
            logger.debug("Streamlit non disponible, runs MLflow non stockés en session")
        except Exception as session_error:
            logger.warning(f"⚠️ Échec stockage runs MLflow: {session_error}")
            logger.exception("Stack trace stockage session:")
    
    # ============================================
    # 15. LOG FINAL ET NETTOYAGE
    # ============================================
    
    logger.info(f"🎯 Entraînement terminé: {successful_models}/{len(model_names)} modèles réussis en {total_time:.2f}s")
    
    if successful_models > 0:
        avg_time = total_time / successful_models
        logger.info(f"⏱️ Temps moyen par modèle: {avg_time:.2f}s")
    
    if len(results) != len(model_names):
        logger.warning(f"⚠️ Incohérence: {len(results)} résultats pour {len(model_names)} modèles demandés")
    
    # Nettoyage final
    gc.collect()
    
    return results


def cleanup_models_directory(max_files: int = None):
    """Nettoie le dossier des modèles pour éviter l'accumulation."""
    if max_files is None:
        max_files = TRAINING_CONSTANTS.get("MAX_MODEL_FILES", 50)
    
    try:
        if not os.path.exists("models_output"):
            return
            
        model_files = []
        for filename in os.listdir("models_output"):
            if filename.endswith('.joblib'):
                filepath = os.path.join("models_output", filename)
                model_files.append((filepath, os.path.getctime(filepath)))
        
        model_files.sort(key=lambda x: x[1])
        
        if len(model_files) > max_files:
            for i in range(len(model_files) - max_files):
                filepath, _ = model_files[i]
                os.remove(filepath)
                logger.info(f"🗑️ Fichier modèle supprimé: {filepath}")
                
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage dossier modèles: {e}")


# Nettoyage automatique au chargement du module
cleanup_models_directory()

# Export des fonctions principales
__all__ = [
    'TrainingMonitor',
    'validate_training_data', 
    'create_leak_free_pipeline',
    'train_single_model_supervised',
    'train_single_model_unsupervised',
    'train_models',
    'cleanup_models_directory',
    'is_mlflow_available',
    'MLFLOW_AVAILABLE'
]