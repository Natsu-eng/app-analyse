"""
Module d'entraînement robuste pour le machine learning.
Supporte l'apprentissage supervisé et non-supervisé avec gestion MLOps avancée.
Version Production - Corrigée pour MLflow/Streamlit
"""
import traceback
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
import threading
from contextlib import contextmanager

from src.evaluation.metrics import evaluate_single_train_test_split
from utils.mlflow import _ensure_array_like, _safe_cluster_metrics, clean_model_name, format_mlflow_run_for_ui, get_git_info, is_mlflow_available

# Intégration MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from joblib import Parallel, delayed

# Imports conditionnels pour robustesse
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

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
    from src.config.constants import TRAINING_CONSTANTS, PREPROCESSING_CONSTANTS, LOGGING_CONSTANTS, MLFLOW_CONSTANTS
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

import logging
# Configuration logging structuré (JSON)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOGGING_CONSTANTS.get("DEFAULT_LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGGING_CONSTANTS.get("LOG_DIR", "logs"), LOGGING_CONSTANTS.get("LOG_FILE", "training.log"))),
        logging.StreamHandler() if LOGGING_CONSTANTS.get("CONSOLE_LOGGING", True) else logging.NullHandler()
    ]
)

# ===============================================
# CLASSES DE GESTION D'ÉTAT THREAD-SAFE - AMÉLIORÉES
# ===============================================

class MLflowRunCollector:
    """Collecteur thread-safe pour les runs MLflow."""
    
    def __init__(self):
        self._runs = []
        self._lock = threading.Lock()
    
    def add_run(self, run_data: Dict[str, Any]) -> None:
        """Ajoute un run de façon thread-safe avec validation."""
        with self._lock:
            if run_data and isinstance(run_data, dict) and run_data.get('run_id'):
                # Validation des données critiques
                required_keys = ['run_id', 'status', 'start_time']
                if all(key in run_data for key in required_keys):
                    self._runs.append(run_data)
                    logger.debug(f"Run MLflow ajouté: {run_data['run_id'][:8]}")
                else:
                    logger.warning(f"Run MLflow incomplet ignoré: {run_data.get('run_id', 'unknown')}")
    
    def get_runs(self) -> List[Dict[str, Any]]:
        """Retourne tous les runs collectés avec validation."""
        with self._lock:
            return [run for run in self._runs if self._is_valid_run(run)]
    
    def _is_valid_run(self, run: Dict) -> bool:
        """Valide la structure d'un run MLflow."""
        return (isinstance(run, dict) and 
                run.get('run_id') and 
                run.get('status') and 
                run.get('start_time'))
    
    def clear(self) -> None:
        """Vide le collecteur."""
        with self._lock:
            self._runs.clear()
    
    def count(self) -> int:
        """Retourne le nombre de runs collectés valides."""
        with self._lock:
            return len([run for run in self._runs if self._is_valid_run(run)])

class TrainingStateManager:
    """Gestionnaire d'état global pour l'entraînement."""
    
    def __init__(self):
        self.mlflow_collector = MLflowRunCollector()
        self._training_lock = threading.Lock()
        self._active_training = False
    
    @contextmanager
    def training_session(self):
        """Context manager pour une session d'entraînement."""
        with self._training_lock:
            self._active_training = True
            self.mlflow_collector.clear()
            try:
                yield self
            finally:
                self._active_training = False
    
    def is_training_active(self) -> bool:
        """Vérifie si un entraînement est en cours."""
        return self._active_training

# Instance globale
TRAINING_STATE = TrainingStateManager()

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
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            resource_info = {
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
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

# ===============================================
# FONCTIONS DE VALIDATION ROBUSTES - AMÉLIORÉES
# ===============================================

def validate_training_data(X: pd.DataFrame, 
                          y: Optional[pd.Series], 
                          task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entraînement de façon robuste.
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
            
        logger.info(f"✅ Validation données: {validation['samples_count']} échantillons, "
                   f"{validation['features_count']} features, {len(validation['issues'])} issues")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur de validation: {str(e)}")
        logger.error(f"❌ Validation error: {e}")
    
    return validation

# ===============================================
# FONCTIONS DE PIPELINE AVEC GESTION D'ERREURS - AMÉLIORÉES
# ===============================================

def create_leak_free_pipeline(
    model_name: str, 
    task_type: str, 
    column_types: Dict[str, List[str]],
    preprocessing_choices: Dict[str, Any],
    use_smote: bool = False,
    optimize_hyperparams: bool = False
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    Crée un pipeline sans data leakage avec gestion robuste des erreurs.
    """
    
    try:
        logger.info(f"🔧 Création pipeline pour {model_name} (task: {task_type}, SMOTE: {use_smote})")
        
        # Récupérer la configuration du modèle
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"❌ Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
        
        model = model_config["model"]
        
        # Préparer la grille de paramètres
        param_grid = {}
        if optimize_hyperparams and "params" in model_config:
            param_grid = {f"model__{k}": v for k, v in model_config["params"].items()}
        
        # Créer le préprocesseur
        preprocessor = create_preprocessor(preprocessing_choices, column_types)
        if preprocessor is None:
            logger.error(f"❌ Échec création préprocesseur pour {model_name}")
            return None, None
        
        # Construire le pipeline selon le contexte
        if use_smote and task_type == 'classification':
            logger.info("🔄 Construction pipeline avec SMOTE")
            
            smote_k = preprocessing_choices.get("smote_k_neighbors", 5)
            random_state = preprocessing_choices.get("random_state", 42)
            sampling_strategy = preprocessing_choices.get("smote_sampling_strategy", 'auto')
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(
                    random_state=random_state,
                    k_neighbors=smote_k,
                    sampling_strategy=sampling_strategy
                )),
                ('model', model)
            ])
            
            logger.info(f"✅ Pipeline créé avec 3 étapes: preprocessor → SMOTE → {model_name}")
        
        else:
            logger.info("🔄 Construction pipeline standard")
            
            if use_smote:
                logger.warning(f"⚠️ SMOTE ignoré pour task_type='{task_type}'")
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            logger.info(f"✅ Pipeline créé avec 2 étapes: preprocessor → {model_name}")
        
        # Validation finale du pipeline
        expected_steps = ['preprocessor', 'model'] if not (use_smote and task_type == 'classification') else ['preprocessor', 'smote', 'model']
        actual_steps = list(pipeline.named_steps.keys())
        
        if actual_steps != expected_steps:
            logger.error(f"❌ Pipeline invalide! Attendu: {expected_steps}, Obtenu: {actual_steps}")
            return None, None
        
        for step_name in expected_steps:
            if pipeline.named_steps[step_name] is None:
                logger.error(f"❌ Étape '{step_name}' est None dans le pipeline!")
                return None, None
        
        logger.info(f"✅ Pipeline validé avec succès pour {model_name}")
        return pipeline, param_grid if param_grid else None
    
    except Exception as e:
        logger.error(f"❌ Erreur création pipeline pour {model_name}: {e}")
        return None, None

# ===============================================
# FONCTIONS D'ENTRAÎNEMENT PAR MODÈLE - AMÉLIORÉES
# ===============================================

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
    """Entraîne un modèle supervisé avec validation croisée."""
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
                n_jobs=n_jobs, verbose=0
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
            except Exception as cv_error:
                logger.warning(f"⚠️ CV échouée pour {model_name}: {cv_error}")
                result["cv_scores"] = None
            
            pipeline.fit(X_train, y_train)
            result["model"] = pipeline
            result["success"] = True
        
        result["training_time"] = time.time() - start_time
        
        if monitor:
            resource_info = monitor.check_resources()
            logger.info(f"✅ {model_name} entraîné en {result['training_time']:.2f}s")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur entraînement {model_name}: {e}")
    
    return result

def train_single_model_unsupervised(
    model_name: str,
    pipeline,
    X,
    param_grid: Dict = None,
    monitor: Any = None
) -> Dict[str, Any]:
    """
    Entraîne un modèle non-supervisé de façon robuste.
    """
    result = {
        "model_name": model_name,
        "success": False,
        "model": None,
        "training_time": 0.0,
        "error": None,
        "best_params": None,
        "labels": None,
        "metrics": {}
    }

    start_time = time.time()

    try:
        X_arr, is_df, idx = _ensure_array_like(X)
        n_samples = X_arr.shape[0]
        if n_samples < 2:
            raise ValueError("Jeu de données insuffisant pour le clustering")

        # Identification de l'estimateur
        estimator = pipeline
        if hasattr(pipeline, "named_steps"):
            last_step = list(pipeline.named_steps.items())[-1]
            estimator = last_step[1]

        def _fit_predict_with_pipeline(pipeline_obj, X_in):
            if hasattr(pipeline_obj, "fit_predict"):
                return pipeline_obj.fit_predict(X_in)
            else:
                fitted = pipeline_obj.fit(X_in)
                if hasattr(fitted, "predict"):
                    return fitted.predict(X_in)
                if hasattr(fitted, "labels_"):
                    return getattr(fitted, "labels_")
                raise AttributeError("Estimator ne supporte ni fit_predict ni predict ni labels_.")

        if param_grid and isinstance(param_grid, dict) and len(param_grid) > 0:
            from itertools import product
            from sklearn.base import clone
            
            keys = list(param_grid.keys())
            values = [param_grid[k] if isinstance(param_grid[k], (list, tuple, np.ndarray)) else [param_grid[k]] for k in keys]
            combos = list(product(*values))

            best_score = -np.inf
            best_params = None

            for combo in combos:
                try:
                    params = dict(zip(keys, combo))
                    pipeline_candidate = clone(pipeline)
                    
                    try:
                        pipeline_candidate.set_params(**params)
                    except Exception:
                        if hasattr(pipeline_candidate, "named_steps"):
                            final_name = list(pipeline_candidate.named_steps.keys())[-1]
                            pipeline_candidate.named_steps[final_name].set_params(**params)

                    labels = _fit_predict_with_pipeline(pipeline_candidate, X)
                    labels = np.asarray(labels)
                    metrics = _safe_cluster_metrics(X, labels)
                    score = metrics.get("silhouette", np.nan)
                    
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_params = params
                        best_candidate = pipeline_candidate
                        best_labels = labels
                        best_metrics = metrics
                        
                except Exception:
                    continue

            if best_score == -np.inf:
                raise RuntimeError("Aucun jeu de paramètres valides trouvé")
                
            result["best_params"] = best_params
            result["model"] = best_candidate
            result["labels"] = best_labels
            result["metrics"] = best_metrics

        else:
            labels = _fit_predict_with_pipeline(pipeline, X)
            labels = np.asarray(labels)
            result["model"] = pipeline
            result["labels"] = labels
            result["metrics"] = _safe_cluster_metrics(X_arr, labels)

        result["training_time"] = time.time() - start_time
        result["success"] = True

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur clustering {model_name}: {e}")

    return result

# ===============================================
# FONCTION PRINCIPALE AMÉLIORÉE - CORRIGÉE
# ===============================================

def log_structured(level: str, message: str, extra: Dict = None):
    """Log structuré en JSON pour parsing en prod."""
    log_dict = {
        "timestamp": datetime.now().isoformat(), 
        "level": level, 
        "message": message,
        "module": "training"
    }
    if extra:
        log_dict.update(extra)
    
    log_message = json.dumps(log_dict, ensure_ascii=False)
    getattr(logger, level.lower())(log_message)

def _store_results_in_session(results: List[Dict], mlflow_runs: List[Dict]) -> bool:
    """
    Stocke les résultats dans session_state de manière ROBUSTE.
    Retourne True si succès, False sinon.
    """
    if not STREAMLIT_AVAILABLE or st is None:
        return False
        
    try:
        # Initialisation ROBUSTE des session states
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = []
        
        if 'mlflow_runs' not in st.session_state:
            st.session_state.mlflow_runs = []
        
        # Validation des données avant stockage
        valid_results = [r for r in results if isinstance(r, dict) and r.get('model_name')]
        valid_mlflow_runs = [r for r in mlflow_runs if isinstance(r, dict) and r.get('run_id')]
        
        # Stockage SÉCURISÉ avec vérification de type
        if isinstance(st.session_state.ml_results, list):
            st.session_state.ml_results.extend(valid_results)
        else:
            st.session_state.ml_results = valid_results
            
        if isinstance(st.session_state.mlflow_runs, list):
            st.session_state.mlflow_runs.extend(valid_mlflow_runs)
        else:
            st.session_state.mlflow_runs = valid_mlflow_runs
            
        log_structured("INFO", "Résultats stockés avec succès", {
            "n_results": len(valid_results),
            "n_mlflow_runs": len(valid_mlflow_runs),
            "total_results": len(st.session_state.ml_results),
            "total_runs": len(st.session_state.mlflow_runs)
        })
        
        return True
        
    except Exception as e:
        log_structured("ERROR", "Échec stockage session", {
            "error": str(e),
            "results_type": type(results) if results else None,
            "mlflow_runs_type": type(mlflow_runs) if mlflow_runs else None
        })
        return False

def train_single_model_with_mlflow(
    model_name: str,
    task_type: str,
    X_train: Optional[pd.DataFrame],
    y_train: Optional[pd.Series],
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
    X: Optional[pd.DataFrame],
    column_types: Dict,
    preprocessing_choices: Dict,
    use_smote: bool,
    optimize: bool,
    feature_list: List[str],
    git_info: Dict,
    label_encoder: Any,
    sample_metrics: bool,
    max_samples_metrics: int,
    monitor: TrainingMonitor,
    mlflow_enabled: bool
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Entraîne un seul modèle avec gestion MLflow complète.
    Retourne (result_dict, mlflow_run_data)
    """
    
    # Initialisation
    mlflow_run_data = None
    result = None
    
    try:
        # Création du pipeline
        pipeline, param_grid = create_leak_free_pipeline(
            model_name=model_name,
            task_type=task_type,
            column_types=column_types,
            preprocessing_choices=preprocessing_choices,
            use_smote=use_smote,
            optimize_hyperparams=optimize
        )
        
        if pipeline is None:
            log_structured("ERROR", f"Pipeline vide pour {model_name}")
            return None, None

        # Configuration MLflow
        run_id = None
        timestamp = int(time.time())
        
        if mlflow_enabled:
            try:
                run_name = f"{clean_model_name(model_name)}_{timestamp}"
                mlflow.start_run(run_name=run_name)
                
                # Log des paramètres de base
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("optimize_hyperparams", optimize)
                mlflow.log_param("use_smote", use_smote)
                mlflow.log_param("n_features", len(feature_list))
                
                # Log git info
                for k, v in git_info.items():
                    if v:  # Éviter les valeurs vides
                        mlflow.log_param(f"git_{k}", v)
                
                # Log preprocessing choices
                for k, v in preprocessing_choices.items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow.log_param(f"preprocessing_{k}", v)
                
                run_id = mlflow.active_run().info.run_id
                log_structured("INFO", f"Run MLflow démarré", {"run_id": run_id, "model": model_name})
                
            except Exception as e:
                log_structured("WARNING", f"Échec démarrage MLflow pour {model_name}: {str(e)}")
                mlflow_enabled = False

        # Entraînement
        training_result = None
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
        except Exception as e:
            log_structured("ERROR", f"Échec entraînement pour {model_name}: {str(e)}")
            if mlflow_enabled and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            return None, None

        # Évaluation
        metrics = {}
        warnings_list = []
        
        try:
            if training_result["success"]:
                if task_type == 'clustering':
                    metrics = training_result["metrics"]
                    metrics["success"] = True
                    metrics["warnings"] = metrics.get("warnings", [])
                else:
                    metrics = evaluate_single_train_test_split(
                        model=training_result["model"],
                        X_test=X_test,
                        y_test=y_test,
                        task_type=task_type,
                        label_encoder=label_encoder,
                        sample_metrics=sample_metrics,
                        max_samples_metrics=max_samples_metrics
                    )
            else:
                metrics = {"error": training_result.get("error", "Échec entraînement")}
        except Exception as e:
            log_structured("ERROR", f"Échec évaluation pour {model_name}: {str(e)}")
            metrics = {"error": f"Erreur évaluation: {str(e)}"}

        # Propagation des warnings
        warnings_list = metrics.pop("warnings", [])

        # Sauvegarde du modèle
        model_name_clean = clean_model_name(model_name)
        model_filename = f"{model_name_clean}_{task_type}_{timestamp}.joblib"
        model_path = os.path.join("models_output", model_filename)
        
        try:
            os.makedirs("models_output", exist_ok=True)
            joblib.dump(training_result["model"], model_path)
            log_structured("INFO", f"Modèle sauvegardé", {"path": model_path, "model": model_name})
        except Exception as e:
            log_structured("ERROR", f"Échec sauvegarde modèle {model_name}: {str(e)}")

        # Logging MLflow final
        if mlflow_enabled and training_result["success"]:
            try:
                # Log des métriques
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        mlflow.log_metric(k, float(v))
                
                # Log du temps d'entraînement
                mlflow.log_metric("training_time", training_result.get("training_time", 0.0))
                
                # Sauvegarde du modèle comme artifact
                mlflow.log_artifact(model_path)
                
                # Formatage des données pour l'UI
                mlflow_run_data = format_mlflow_run_for_ui(
                    run_info=mlflow.active_run(),
                    metrics=metrics,
                    preprocessing_choices=preprocessing_choices,
                    model_name=model_name,
                    timestamp=timestamp
                )
                
                log_structured("INFO", f"Run MLflow complété", {
                    "model": model_name, 
                    "run_id": run_id,
                    "metrics_count": len([k for k in metrics if isinstance(metrics[k], (int, float))])
                })
                
            except Exception as e:
                log_structured("WARNING", f"Échec logging MLflow pour {model_name}: {str(e)}")
            finally:
                if mlflow.active_run():
                    mlflow.end_run()

        # Construction du résultat final
        result = {
            "model_name": model_name,
            "task_type": task_type,
            "metrics": metrics,
            "training_time": training_result.get("training_time", 0),
            "model_path": model_path,
            "warnings": warnings_list,
            "success": training_result.get("success", False),
            "feature_names": feature_list
        }

        # Ajout des données spécifiques au type de tâche
        if task_type == 'clustering' and training_result.get("labels") is not None:
            result["labels"] = training_result["labels"]
            result["X_sample"] = X  # Pour les visualisations
        elif task_type in ['classification', 'regression']:
            # AJOUT: Données pour visualisations avancées
            result["X_train"] = X_train
            result["y_train"] = y_train
            result["X_test"] = X_test
            result["y_test"] = y_test
            result["model"] = training_result["model"]  # Pipeline complet
        
        # Échantillon pour SHAP (éviter surcharge mémoire)
        if X_test is not None and len(X_test) > 0:
            sample_size = min(1000, len(X_test))
            result["X_sample"] = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]

    except Exception as e:
        log_structured("ERROR", f"Erreur critique dans train_single_model_with_mlflow pour {model_name}: {str(e)}")
        if mlflow_enabled and mlflow.active_run():
            mlflow.end_run(status="FAILED")
        return None, None

    return result, mlflow_run_data

def train_models(
    df: pd.DataFrame,
    target_column: Optional[str],
    model_names: List[str],
    task_type: str,
    test_size: float = 0.2,
    optimize: bool = False,
    feature_list: List[str] = None,
    use_smote: bool = False,
    preprocessing_choices: Dict = None,
    sample_metrics: bool = True,
    max_samples_metrics: int = 100000,
    n_jobs: int = None
) -> List[Dict[str, Any]]:
    """
    Fonction principale d'entraînement - Version Production Corrigée.
    """
    
    if n_jobs is None:
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
    
    # Utilisation du gestionnaire d'état global
    with TRAINING_STATE.training_session() as state:
        results = []
        monitor = TrainingMonitor()
        monitor.start_training()

        # ============================================================
        # 1. CONFIGURATION DE BASE
        # ============================================================
        task_type = task_type.lower()
        if task_type not in ['classification', 'regression', 'clustering', 'unsupervised']:
            log_structured("ERROR", f"Type de tâche invalide: {task_type}")
            return [{"model_name": "Validation", "metrics": {"error": f"Type de tâche {task_type} non supporté"}, "warnings": []}]
        
        if task_type == 'unsupervised':
            task_type = 'clustering'

        if task_type == 'clustering':
            target_column = None
            use_smote = False
            test_size = 0.0

        log_structured("INFO", f"Début entraînement", {
            "task_type": task_type, 
            "n_models": len(model_names), 
            "target_column": target_column,
            "dataset_shape": df.shape
        })

        # Validation initiale
        min_samples = TRAINING_CONSTANTS.get("MIN_SAMPLES_REQUIRED", 10)
        if len(df) < min_samples:
            log_structured("ERROR", f"Nombre d'échantillons insuffisant: {len(df)} < {min_samples}")
            return [{"model_name": "Validation", "metrics": {"error": "Échantillons insuffisants"}, "warnings": []}]

        # ============================================================
        # 2. CONFIGURATION MLFLOW
        # ============================================================
        mlflow_enabled = MLFLOW_AVAILABLE and MLFLOW_CONSTANTS.get("AVAILABLE", False)
        git_info = get_git_info() if mlflow_enabled else {}

        if mlflow_enabled:
            try:
                mlflow.set_tracking_uri(MLFLOW_CONSTANTS.get("TRACKING_URI", "sqlite:///mlflow.db"))
                mlflow.set_experiment(MLFLOW_CONSTANTS.get("EXPERIMENT_NAME", "datalab_experiments"))
                log_structured("INFO", f"MLflow configuré", {
                    "experiment_name": MLFLOW_CONSTANTS.get("EXPERIMENT_NAME"),
                    "tracking_uri": MLFLOW_CONSTANTS.get("TRACKING_URI"),
                    "git_branch": git_info.get('branch', 'unknown')
                })
            except Exception as e:
                log_structured("WARNING", f"Échec configuration MLflow: {str(e)}")
                mlflow_enabled = False

        # ============================================================
        # 3. PRÉPARATION DES FEATURES
        # ============================================================
        if not feature_list:
            if target_column and target_column in df.columns:
                feature_list = [col for col in df.columns if col != target_column]
            else:
                feature_list = list(df.columns)

        max_features = TRAINING_CONSTANTS.get("MAX_FEATURES", 1000)
        if len(feature_list) > max_features:
            log_structured("WARNING", f"Trop de features ({len(feature_list)}) → limité à {max_features}")
            feature_list = feature_list[:max_features]

        # ============================================================
        # 4. CONFIGURATION PRÉTRAITEMENT
        # ============================================================
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

        # Création des répertoires
        os.makedirs("models_output", exist_ok=True)
        os.makedirs(LOGGING_CONSTANTS.get("LOG_DIR", "logs"), exist_ok=True)

        # ============================================================
        # 5. PRÉPARATION DES DONNÉES
        # ============================================================
        X = df[feature_list].copy()
        if X.empty:
            log_structured("ERROR", "DataFrame vide après sélection des features")
            return [{"model_name": "Validation", "metrics": {"error": "DataFrame vide"}, "warnings": []}]

        y, label_encoder = None, None
        if task_type != 'clustering' and target_column:
            y_raw = df[target_column].copy()
            y_encoded, label_encoder, warnings_enc = safe_label_encode(y_raw)
            y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
            if warnings_enc:
                log_structured("WARNING", f"Encodage des labels: {warnings_enc}")

        data_validation = validate_training_data(X, y, task_type)
        if not data_validation["is_valid"]:
            log_structured("ERROR", f"Validation des données échouée: {', '.join(data_validation['issues'])}")
            return [{"model_name": "Validation", "metrics": {"error": ', '.join(data_validation['issues'])}, "warnings": data_validation.get('warnings', [])}]

        column_types = auto_detect_column_types(X)

        # ============================================================
        # 6. SPLIT TRAIN/TEST
        # ============================================================
        X_train = X_test = y_train = y_test = None
        if task_type != 'clustering':
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            stratify = y if task_type == 'classification' else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=stratify
                )
                log_structured("INFO", "Split train/test effectué", {
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                })
            except Exception as e:
                log_structured("ERROR", f"Échec du split train/test: {str(e)}")
                return [{"model_name": "Validation", "metrics": {"error": f"Split train/test échoué: {str(e)}"}, "warnings": []}]

        # ============================================================
        # 7. ENTRAÎNEMENT PARALLÈLE AVEC COLLECTE MLFLOW
        # ============================================================
        successful_models = 0

        def train_model_wrapper(args):
            """Wrapper pour l'entraînement parallèle avec collecte MLflow."""
            i, model_name = args
            log_structured("INFO", f"Début entraînement modèle", {
                "model_index": i,
                "total_models": len(model_names),
                "model_name": model_name
            })
            
            result, mlflow_data = train_single_model_with_mlflow(
                model_name=model_name,
                task_type=task_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X=X,
                column_types=column_types,
                preprocessing_choices=preprocessing_choices,
                use_smote=use_smote,
                optimize=optimize,
                feature_list=feature_list,
                git_info=git_info,
                label_encoder=label_encoder,
                sample_metrics=sample_metrics,
                max_samples_metrics=max_samples_metrics,
                monitor=monitor,
                mlflow_enabled=mlflow_enabled
            )
            
            # Collecte thread-safe des données MLflow
            if mlflow_data and isinstance(mlflow_data, dict):
                state.mlflow_collector.add_run(mlflow_data)
            
            return result

        # Exécution parallèle
        try:
            # Préparation des arguments
            model_args = [(i, model_name) for i, model_name in enumerate(model_names, 1)]
            
            if n_jobs == 1 or len(model_names) == 1:
                # Mode séquentiel pour debug ou petits datasets
                log_structured("INFO", "Exécution en mode séquentiel")
                parallel_results = [train_model_wrapper(args) for args in model_args]
            else:
                # Mode parallèle
                log_structured("INFO", f"Exécution en mode parallèle (n_jobs={n_jobs})")
                parallel_results = Parallel(n_jobs=n_jobs)(
                    delayed(train_model_wrapper)(args) for args in model_args
                )
            
            # Filtrage des résultats valides
            results = [res for res in parallel_results if res is not None and res.get("success", False)]
            successful_models = len(results)
            
        except Exception as e:
            log_structured("ERROR", f"Échec de l'exécution parallèle: {str(e)} - Fallback séquentiel")
            # Fallback séquentiel
            results = []
            for i, model_name in enumerate(model_names, 1):
                try:
                    result = train_model_wrapper((i, model_name))
                    if result and result.get("success", False):
                        results.append(result)
                        successful_models += 1
                except Exception as model_error:
                    log_structured("ERROR", f"Échec sur modèle {model_name}: {str(model_error)}")

        # ============================================================
        # 8. FINALISATION ET STOCKAGE STREAMLIT ROBUSTE
        # ============================================================
        total_time = monitor.get_total_duration()
        
        # Récupération des runs MLflow collectés
        mlflow_runs = state.mlflow_collector.get_runs()
        
        log_structured("INFO", "Entraînement terminé", {
            "successful_models": successful_models,
            "total_models": len(model_names),
            "total_time": total_time,
            "mlflow_runs_collected": len(mlflow_runs)
        })

        # Stockage Streamlit ROBUSTE
        storage_success = _store_results_in_session(results, mlflow_runs)
        if not storage_success:
            log_structured("ERROR", "Échec critique du stockage Streamlit")

        # Vérification finale MLflow
        if mlflow_enabled:
            try:
                runs = mlflow.search_runs()
                log_structured("INFO", "Vérification MLflow", {
                    "runs_found": len(runs),
                    "successful_models": successful_models
                })
                
                if len(runs) < successful_models:
                    log_structured("WARNING", f"Incohérence MLflow: {len(runs)} runs vs {successful_models} modèles")
                    
            except Exception as e:
                log_structured("WARNING", f"Échec vérification MLflow: {str(e)}")

        # Nettoyage mémoire
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
    'TrainingStateManager',
    'MLflowRunCollector',
    'validate_training_data', 
    'create_leak_free_pipeline',
    'train_single_model_supervised',
    'train_single_model_unsupervised',
    'train_models',
    'cleanup_models_directory',
    'is_mlflow_available',
    'MLFLOW_AVAILABLE',
    'TRAINING_STATE'
]