"""
Module d'entraînement robuste pour le machine learning.
Supporte l'apprentissage supervisé et non-supervisé avec gestion MLOps avancée.
"""

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


def is_mlflow_available() -> bool:
    """Vérifie si MLflow est disponible et fonctionnel."""
    return MLFLOW_AVAILABLE and mlflow is not None


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


def create_leak_free_pipeline(
    model_name: str, 
    task_type: str, 
    column_types: Dict[str, List[str]],
    preprocessing_choices: Dict[str, Any],
    use_smote: bool = False,
    optimize_hyperparams: bool = False
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    Crée un pipeline sans data leakage en intégrant le préprocesseur DANS le pipeline.
    
    Args:
        model_name: Nom du modèle
        task_type: Type de tâche
        column_types: Types de colonnes détectés
        preprocessing_choices: Options de prétraitement
        use_smote: Utiliser SMOTE (seulement si supervisé)
        optimize_hyperparams: Optimiser les hyperparamètres
        
    Returns:
        Tuple (pipeline, param_grid)
    """
    try:
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"❌ Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
            
        model = model_config["model"]
        param_grid = {}
        
        if optimize_hyperparams and "params" in model_config:
            param_grid = {f"model__{k}": v for k, v in model_config["params"].items()}
            logger.debug(f"Grille paramètres pour {model_name}: {len(param_grid)} combinaisons")
        
        preprocessor = create_preprocessor(preprocessing_choices, column_types)
        if preprocessor is None:
            logger.error(f"❌ Échec création préprocesseur pour {model_name}")
            return None, None
        
        validation_result = validate_preprocessor(preprocessor, pd.DataFrame({
            col: [0] for cols in column_types.values() for col in cols
        }))
        
        if not validation_result["is_valid"]:
            logger.warning(f"⚠️ Préprocesseur avec issues: {validation_result['issues']}")
        
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if use_smote and task_type == 'classification':
            smote_k = PREPROCESSING_CONSTANTS.get("SMOTE_K_NEIGHBORS", 5)
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            pipeline_steps.append(('smote', SMOTE(
                random_state=random_state,
                k_neighbors=smote_k
            )))
            logger.debug("SMOTE ajouté au pipeline")
        
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)
        
        logger.info(f"✅ Pipeline leak-free créé pour {model_name} avec {len(pipeline_steps)} étapes")
        return pipeline, param_grid
        
    except Exception as e:
        logger.error(f"❌ Erreur création pipeline pour {model_name}: {e}")
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
                scoring='silhouette_score' if hasattr(pipeline, 'fit_predict') else 'silhouette',
                n_jobs=n_jobs, verbose=0
            )
            
            grid_search.fit(X)
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["success"] = True
            
        else:
            pipeline.fit(X)
            result["model"] = pipeline
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
    """Orchestre l'entraînement sans data leakage pour tous types de tâches avec MLflow tracking."""
    
    results = []
    monitor = TrainingMonitor()
    monitor.start_training()
    
    task_type = task_type.lower()
    if task_type == 'unsupervised':
        task_type = 'clustering'
        
    if task_type == 'clustering':
        target_column = None
        use_smote = False
        test_size = 0.0
    
    logger.info(f"🎯 Début entraînement - Type: {task_type}, Modèles: {len(model_names)}, Target: {target_column}")
    
    # Configuration MLflow avec gestion sécurisée
    mlflow_enabled = is_mlflow_available()
    if mlflow_enabled:
        try:
            mlflow.set_experiment(f"{task_type}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            logger.info("✅ MLflow experiment initialized")
        except Exception as e:
            logger.warning(f"⚠️ MLflow experiment setup failed: {e}")
            mlflow_enabled = False
    
    # Détermination de la liste des caractéristiques
    if not feature_list:
        if target_column and target_column in df.columns:
            feature_list = [col for col in df.columns if col != target_column]
        else:
            feature_list = list(df.columns)
    
    # Limitation des caractéristiques
    max_features = TRAINING_CONSTANTS.get("MAX_FEATURES", 100)
    if len(feature_list) > max_features:
        logger.warning(f"⚠️ Trop de features ({len(feature_list)}), limitation à {max_features}")
        feature_list = feature_list[:max_features]
    
    # Configuration par défaut pour le prétraitement
    if preprocessing_choices is None:
        preprocessing_choices = {
            'numeric_imputation': PREPROCESSING_CONSTANTS.get("NUMERIC_IMPUTATION_DEFAULT", "mean"),
            'categorical_imputation': PREPROCESSING_CONSTANTS.get("CATEGORICAL_IMPUTATION_DEFAULT", "most_frequent"),
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'scaling_method': PREPROCESSING_CONSTANTS.get("SCALING_METHOD", "standard"),
            'encoding_method': PREPROCESSING_CONSTANTS.get("ENCODING_METHOD", "onehot")
        }
    
    os.makedirs("models_output", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)

    # Préparation des données
    logger.info("📊 Préparation des données...")
    X = df[feature_list].copy()
    
    if X.empty:
        logger.error("❌ DataFrame d'entrée vide ou None")
        return [{"model_name": "Validation", "metrics": {"error": "DataFrame d'entrée vide ou None"}}]
    
    y = None
    label_encoder = None
    
    if task_type != 'clustering' and target_column:
        if target_column not in df.columns:
            logger.error(f"❌ Colonne cible '{target_column}' non trouvée")
            return [{"model_name": "Validation", "metrics": {"error": f"Target '{target_column}' non trouvée"}}]
        
        y_raw = df[target_column].copy()
        y_encoded, label_encoder, _ = safe_label_encode(y_raw)
        y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
    
    # Validation des données
    data_validation = validate_training_data(X, y, task_type)
    if not data_validation["is_valid"]:
        error_msg = f"Données invalides: {', '.join(data_validation['issues'])}"
        logger.error(f"❌ {error_msg}")
        return [{"model_name": "Validation", "metrics": {"error": error_msg}}]
    
    for warning in data_validation["warnings"]:
        logger.warning(f"⚠️ {warning}")
    
    # Détection des types de colonnes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        column_types = auto_detect_column_types(X)
    
    # Séparation des données pour les tâches supervisées
    X_train, X_test, y_train, y_test = None, None, None, None
    
    if task_type != 'clustering':
        try:
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            stratification = y if task_type == 'classification' else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratification
            )
            logger.info(f"✅ Données splittées: train={len(X_train)}, test={len(X_test)}")
        except Exception as split_error:
            logger.error(f"❌ Erreur lors du split: {split_error}")
            return [{"model_name": "Split", "metrics": {"error": str(split_error)}}]
    
    successful_models = 0
    max_training_time = TRAINING_CONSTANTS.get("MAX_TRAINING_TIME", 3600)
    high_memory_threshold = TRAINING_CONSTANTS.get("HIGH_MEMORY_THRESHOLD", 85)
    
    for i, model_name in enumerate(model_names, 1):
        logger.info(f"🔧 Processing model {i}/{len(model_names)}: {model_name}")
        
        # Vérification du temps d'entraînement
        total_duration = monitor.get_total_duration()
        if total_duration > max_training_time:
            logger.warning(f"⏰ Temps d'entraînement dépassé ({total_duration:.0f}s > {max_training_time}s)")
            results.append({
                "model_name": model_name,
                "metrics": {"error": "Temps d'entraînement maximum dépassé"}
            })
            continue
        
        # Vérification des ressources
        resource_info = monitor.check_resources()
        if resource_info.get('memory_percent', 0) > high_memory_threshold:
            logger.warning("🧹 Mémoire élevée, nettoyage...")
            gc.collect()
        
        # Création du pipeline
        pipeline, param_grid = create_leak_free_pipeline(
            model_name=model_name, task_type=task_type, column_types=column_types,
            preprocessing_choices=preprocessing_choices, use_smote=use_smote, optimize_hyperparams=optimize
        )
        
        if pipeline is None:
            results.append({"model_name": model_name, "metrics": {"error": "Erreur création du pipeline"}})
            continue
        
        run_id = None
        if mlflow_enabled:
            try:
                with mlflow.start_run(run_name=f"{model_name}_{int(time.time())}"):
                    run_id = mlflow.active_run().info.run_id
                    mlflow.log_param("task_type", task_type)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("n_samples", len(X))
                    mlflow.log_param("n_features", len(feature_list))
                    mlflow.log_param("use_smote", use_smote)
                    mlflow.log_param("optimize_hyperparams", optimize)
                    mlflow.log_param("test_size", test_size)
                    for key, value in preprocessing_choices.items():
                        mlflow.log_param(f"preprocessing_{key}", value)
            except Exception as e:
                logger.warning(f"⚠️ MLflow run start failed for {model_name}: {e}")
        
        # Entraînement du modèle
        training_result = (train_single_model_unsupervised(
            model_name=model_name, pipeline=pipeline, X=X, param_grid=param_grid, monitor=monitor
        ) if task_type == 'clustering' else
        train_single_model_supervised(
            model_name=model_name, pipeline=pipeline, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, param_grid=param_grid, task_type=task_type, monitor=monitor
        ))
        
        if training_result["success"] and training_result["model"] is not None:
            try:
                metrics = (evaluate_model_with_metrics_calculator(
                    model=training_result["model"], X_test=None, y_test=None,
                    task_type=task_type, X_data=X
                ) if task_type == 'clustering' else
                evaluate_model_with_metrics_calculator(
                    model=training_result["model"], X_test=X_test, y_test=y_test,
                    task_type=task_type, label_encoder=label_encoder
                ))
                
                timestamp = int(time.time())
                model_filename = f"{model_name.replace(' ', '_').lower()}_{task_type}_{timestamp}.joblib"
                model_path = os.path.join("models_output", model_filename)
                
                joblib.dump(training_result["model"], model_path)
                
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
                }
                
                if mlflow_enabled and run_id:
                    try:
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                                mlflow.log_metric(metric_name, metric_value)
                        
                        if training_result.get("best_params"):
                            for param_name, param_value in training_result["best_params"].items():
                                mlflow.log_param(param_name, param_value)
                        
                        mlflow.sklearn.log_model(
                            training_result["model"], artifact_path="model",
                            registered_model_name=f"{task_type}_{model_name}",
                            #signature=signature, #input_example=input_example
                        )
                        
                        mlflow.log_artifact(model_path)
                        mlflow.log_metric("training_time", training_result["training_time"])
                        mlflow.log_metric("memory_percent", resource_info.get('memory_percent', 0))
                        mlflow.log_metric("cpu_percent", resource_info.get('cpu_percent', 0))
                        mlflow.log_dict(data_validation, "data_validation.json")
                        
                        logger.info(f"✅ MLflow logged for {model_name} - Run ID: {run_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ MLflow logging failed for {model_name}: {e}")
                
                results.append(result)
                successful_models += 1
                
                logger.info(f"✅ {model_name} - succès en {training_result['training_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Erreur évaluation {model_name}: {e}")
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": f"Erreur évaluation: {str(e)}"},
                    "training_time": training_result["training_time"]
                })
        else:
            error_msg = training_result.get("error", "Erreur inconnue lors de l'entraînement")
            logger.error(f"❌ Échec entraînement {model_name}: {error_msg}")
            results.append({
                "model_name": model_name,
                "metrics": {"error": error_msg},
                "training_time": training_result["training_time"]
            })
        
        gc.collect()
    
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
        "results_summary": [
            {
                "model_name": r["model_name"],
                "success": "metrics" in r and "error" not in r.get("metrics", {}),
                "training_time": r.get("training_time", 0)
            } for r in results
        ]
    }
    
    log_filename = f"training_log_{int(time.time())}.json"
    log_path = os.path.join("training_logs", log_filename)
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    if mlflow_enabled:
        try:
            mlflow.log_artifact(log_path)
            mlflow.log_metric("total_training_time", total_time)
            mlflow.log_metric("models_successful", successful_models)
            mlflow.log_dict(monitor_summary, "monitor_summary.json")
        except Exception as e:
            logger.warning(f"⚠️ MLflow final logging failed: {e}")
    
    logger.info(f"🎯 Entraînement terminé: {successful_models}/{len(model_names)} "
                f"modèles réussis en {total_time:.2f}s")
    
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