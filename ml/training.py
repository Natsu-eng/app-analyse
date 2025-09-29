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

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import des modules de l'application
try:
    from ml.catalog import get_model_config
    from ml.data_preprocessing import create_preprocessor, safe_label_encode, validate_preprocessor
    from ml.evaluation.metrics_calculation import EvaluationMetrics
    from utils.data_analysis import auto_detect_column_types
    from utils.logging_config import get_logger
except ImportError as e:
    print(f"Warning: Some imports failed - {e}")
    # Fallback pour les tests
    def get_model_config(*args, **kwargs): return None
    def auto_detect_column_types(*args, **kwargs): return {}
    def get_logger(name): 
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = get_logger(__name__)

# Constantes de configuration
MAX_TRAINING_TIME = 3600  # 1 heure maximum par modèle
MEMORY_LIMIT_MB = 4000    # 4GB mémoire limite
MIN_SAMPLES_REQUIRED = 10
MAX_FEATURES = 1000       # Maximum de features pour éviter l'explosion dimensionnelle

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
            if memory.percent > 85:
                logger.warning(f"⚠️ Utilisation mémoire élevée: {memory.percent:.1f}%")
            if cpu_percent > 90:
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
        # Vérification des dimensions de base
        if X is None or len(X) == 0:
            validation["is_valid"] = False
            validation["issues"].append("Dataset X vide ou None")
            return validation
            
        if len(X) < MIN_SAMPLES_REQUIRED:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop peu d'échantillons ({len(X)} < {MIN_SAMPLES_REQUIRED})")
            
        if validation["features_count"] == 0:
            validation["is_valid"] = False
            validation["issues"].append("Aucune feature disponible")
        
        # Vérification de la mémoire
        try:
            if hasattr(X, 'memory_usage'):
                x_memory = X.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                validation["data_quality"]["memory_usage_mb"] = x_memory
                if x_memory > MEMORY_LIMIT_MB:
                    validation["warnings"].append(f"Dataset volumineux ({x_memory:.1f}MB)")
        except Exception as e:
            logger.debug(f"Erreur calcul mémoire: {e}")
        
        # Vérification des valeurs manquantes
        missing_stats = X.isna().sum()
        total_missing = missing_stats.sum()
        missing_ratio = total_missing / (X.shape[0] * X.shape[1]) if X.size > 0 else 0
        
        validation["data_quality"]["total_missing"] = int(total_missing)
        validation["data_quality"]["missing_ratio"] = float(missing_ratio)
        
        if missing_ratio > 0.5:
            validation["warnings"].append(f"Ratio de valeurs manquantes élevé: {missing_ratio:.1%}")
        
        # Vérification spécifique au non-supervisé
        if task_type == 'clustering':
            if y is not None:
                validation["warnings"].append("Target ignorée pour le clustering")
            
            # Pour le clustering, vérifier qu'on a assez de features numériques
            numeric_features = X.select_dtypes(include=[np.number]).shape[1]
            validation["data_quality"]["numeric_features"] = numeric_features
            
            if numeric_features < 2:
                validation["warnings"].append("Peu de features numériques pour le clustering")
        
        # Vérification de la target pour supervisé
        elif y is not None:
            if len(y) != len(X):
                validation["is_valid"] = False
                validation["issues"].append(f"Dimensions X et y incohérentes: {len(X)} vs {len(y)}")
                
            valid_target_count = y.notna().sum() if hasattr(y, 'notna') else np.sum(~np.isnan(y))
            validation["data_quality"]["valid_target_count"] = int(valid_target_count)
            
            if valid_target_count < MIN_SAMPLES_REQUIRED:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop peu de targets valides ({valid_target_count})")
                
            # Pour la classification, vérifier le nombre de classes
            if task_type == 'classification':
                unique_classes = np.unique(y.dropna()) if hasattr(y, 'dropna') else np.unique(y[~np.isnan(y)])
                n_classes = len(unique_classes)
                validation["data_quality"]["n_classes"] = n_classes
                
                if n_classes < 2:
                    validation["is_valid"] = False
                    validation["issues"].append("Moins de 2 classes distinctes")
                elif n_classes > 100:
                    validation["warnings"].append("Plus de 100 classes - vérifiez la variable cible")
            
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
        # Récupération de la configuration du modèle
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"❌ Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
            
        model = model_config["model"]
        param_grid = {}
        
        if optimize_hyperparams and "params" in model_config:
            # Préfixer les paramètres avec 'model__' pour le pipeline
            param_grid = {f"model__{k}": v for k, v in model_config["params"].items()}
            logger.debug(f"Grille paramètres pour {model_name}: {len(param_grid)} combinaisons")
        
        # Créer le préprocesseur - intégré dans le pipeline pour éviter le data leakage
        preprocessor = create_preprocessor(preprocessing_choices, column_types)
        if preprocessor is None:
            logger.error(f"❌ Échec création préprocesseur pour {model_name}")
            return None, None
        
        # Validation du préprocesseur
        validation_result = validate_preprocessor(preprocessor, pd.DataFrame({
            col: [0] for cols in column_types.values() for col in cols
        }))
        
        if not validation_result["is_valid"]:
            logger.warning(f"⚠️ Préprocesseur avec issues: {validation_result['issues']}")
        
        # Construction du pipeline SANS DATA LEAKAGE
        pipeline_steps = [('preprocessor', preprocessor)]
        
        # SMOTE seulement pour classification supervisée
        if use_smote and task_type == 'classification':
            pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=3)))
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
    """
    Entraîne un modèle supervisé avec validation croisée propre (sans data leakage).
    
    Args:
        model_name: Nom du modèle
        pipeline: Pipeline du modèle
        X_train, y_train: Données d'entraînement 
        X_test, y_test: Données de test
        param_grid: Grille d'hyperparamètres
        task_type: Type de tâche
        monitor: Monitor de progression
        
    Returns:
        Résultats de l'entraînement
    """
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
        
        # Configuration de la cross-validation adaptée
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_train))), shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:  # regression
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
        
        # Optimisation des hyperparamètres avec CV propre
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation hyperparamètres pour {model_name}")
            
            # Limiter le nombre de combinaisons pour les gros datasets
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            if total_combinations > 50:
                logger.warning(f"Grille trop large ({total_combinations} combinaisons), limitation automatique")
                # Prendre les 2 premières valeurs pour chaque paramètre
                limited_param_grid = {}
                for k, v in param_grid.items():
                    if len(v) > 2:
                        limited_param_grid[k] = v[:2]
                    else:
                        limited_param_grid[k] = v
                param_grid = limited_param_grid
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=1,  # Éviter parallélisme pour stabilité
                verbose=0,
                error_score='raise'
            )
            
            # Entraînement avec optimisation SUR TRAIN SEULEMENT
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
            # Entraînement simple avec validation croisée pour évaluation
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                result["cv_scores"] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std()
                }
                logger.info(f"📊 CV scores pour {model_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as cv_error:
                logger.warning(f"⚠️ CV échouée pour {model_name}: {cv_error}")
                result["cv_scores"] = None
            
            # Entraînement final sur toutes les données train
            pipeline.fit(X_train, y_train)
            result["model"] = pipeline
            result["success"] = True
        
        result["training_time"] = time.time() - start_time
        
        # Vérification des ressources
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
    """
    Entraîne un modèle non-supervisé (clustering).
    
    Args:
        model_name: Nom du modèle
        pipeline: Pipeline du modèle
        X: Données (pas de split train/test pour clustering)
        param_grid: Grille d'hyperparamètres
        monitor: Monitor de progression
        
    Returns:
        Résultats de l'entraînement
    """
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
        
        # Pour le clustering, pas de train/test split classique
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation hyperparamètres clustering pour {model_name}")
            
            # Pour le clustering, on utilise silhouette score pour optimiser
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,  # Cross-validation même pour clustering (pour robustesse)
                scoring='silhouette_score' if hasattr(pipeline, 'fit_predict') else 'silhouette',
                n_jobs=1,
                verbose=0
            )
            
            grid_search.fit(X)
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["success"] = True
            
        else:
            # Entraînement simple
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
    """
    Évalue un modèle en utilisant EvaluationMetrics pour une gestion robuste.
    
    Args:
        model: Modèle entraîné
        X_test: Données de test (None pour clustering)
        y_test: Labels de test (None pour clustering)
        task_type: Type de tâche
        label_encoder: Encodeur de labels
        X_data: Données complètes (pour clustering)
        
    Returns:
        Métriques d'évaluation
    """
    try:
        metrics_calculator = EvaluationMetrics(task_type)
        
        if task_type == 'clustering':
            # Pour le clustering
            if X_data is None:
                return {"error": "Données X requises pour l'évaluation non supervisée"}
            
            # Prédiction des clusters
            if hasattr(model, 'predict'):
                cluster_labels = model.predict(X_data)
            elif hasattr(model, 'labels_'):
                cluster_labels = model.labels_
            else:
                cluster_labels = model.fit_predict(X_data)
            
            # Calcul des métriques de clustering
            metrics = metrics_calculator.calculate_unsupervised_metrics(X_data.values, cluster_labels)
            metrics['n_clusters'] = len(np.unique(cluster_labels[~np.isnan(cluster_labels)]))
            metrics['task_type'] = task_type
            metrics['n_samples'] = len(X_data)
            
        else:
            # Pour les tâches supervisées
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
            else:  # regression
                metrics = metrics_calculator.calculate_regression_metrics(
                    y_test.values, y_pred
                )
            
            metrics['task_type'] = task_type
            metrics['n_samples'] = len(X_test)
        
        # Ajouter les messages d'avertissement si présents
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
    target_column: Optional[str],  # Peut être None pour clustering
    model_names: List[str],
    task_type: str,
    test_size: float = 0.2,
    optimize: bool = False,
    feature_list: List[str] = None,
    use_smote: bool = False,
    preprocessing_choices: Dict = None
) -> List[Dict[str, Any]]:
    """
    Orchestre l'entraînement sans data leakage pour tous types de tâches.
    
    Args:
        df: DataFrame contenant les données
        target_column: Colonne cible (None pour clustering)
        model_names: Liste des noms de modèles à entraîner
        task_type: Type de tâche ML ('classification', 'regression', 'clustering')
        test_size: Proportion des données de test (ignoré pour clustering)
        optimize: Optimiser les hyperparamètres
        feature_list: Liste des features à utiliser
        use_smote: Utiliser SMOTE (seulement pour classification)
        preprocessing_choices: Options de prétraitement
        
    Returns:
        Liste des résultats d'entraînement
    """
    # Initialisation
    results = []
    monitor = TrainingMonitor()
    monitor.start_training()
    
    # Validation et préparation des paramètres
    task_type = task_type.lower()
    if task_type == 'unsupervised':
        task_type = 'clustering'
        
    if task_type == 'clustering':
        target_column = None
        use_smote = False
        test_size = 0.0  # Pas de split pour clustering
    
    logger.info(f"🎯 Début entraînement - Type: {task_type}, Modèles: {len(model_names)}, "
               f"Target: {target_column}")
    
    # Préparation des features
    if not feature_list:
        if target_column and target_column in df.columns:
            feature_list = [col for col in df.columns if col != target_column]
        else:
            feature_list = list(df.columns)
    
    # Limiter le nombre de features pour éviter l'explosion dimensionnelle
    if len(feature_list) > MAX_FEATURES:
        logger.warning(f"⚠️ Trop de features ({len(feature_list)}), limitation à {MAX_FEATURES}")
        feature_list = feature_list[:MAX_FEATURES]
    
    # Configuration du preprocessing par défaut
    if not preprocessing_choices:
        preprocessing_choices = {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'scaling_method': 'standard',
            'encoding_method': 'onehot'
        }
    
    # Création du dossier de sortie
    os.makedirs("models_output", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    
    try:
        # Préparation des données
        logger.info("📊 Préparation des données...")
        
        X = df[feature_list].copy() if df is not None else None
        if X is None or len(X) == 0:
            logger.error("❌ DataFrame d'entrée vide ou None")
            return [{"model_name": "Validation", "metrics": {"error": "DataFrame d'entrée vide ou None"}}]
        
        # Gestion de la target selon le type de tâche
        y = None
        y_encoded = None
        label_encoder = None
        encoding_map = None
        
        if task_type != 'clustering' and target_column:
            if target_column not in df.columns:
                logger.error(f"❌ Colonne cible '{target_column}' non trouvée")
                return [{"model_name": "Validation", "metrics": {"error": f"Target '{target_column}' non trouvée"}}]
                
            y_raw = df[target_column].copy()
            y_encoded, label_encoder, encoding_map = safe_label_encode(y_raw)
            y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
        
        # Validation des données
        data_validation = validate_training_data(X, y, task_type)
        if not data_validation["is_valid"]:
            error_msg = f"Données invalides: {', '.join(data_validation['issues'])}"
            logger.error(f"❌ {error_msg}")
            return [{"model_name": "Validation", "metrics": {"error": error_msg}}]
        
        # Affichage des warnings
        for warning in data_validation["warnings"]:
            logger.warning(f"⚠️ {warning}")
        
        # Détection des types de colonnes AVANT le split (pas de leakage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            column_types = auto_detect_column_types(X)
        
        # Split train-test SEULEMENT pour supervisé
        X_train, X_test, y_train, y_test = None, None, None, None
        
        if task_type != 'clustering':
            try:
                stratification = y if task_type == 'classification' else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=42, 
                    stratify=stratification
                )
                logger.info(f"✅ Données splittées: train={len(X_train)}, test={len(X_test)}")
            except Exception as split_error:
                logger.error(f"❌ Erreur lors du split: {split_error}")
                return [{"model_name": "Split", "metrics": {"error": str(split_error)}}]
        else:
            logger.info(f"🔍 Mode clustering: utilisation de tout le dataset ({len(X)} échantillons)")
        
        # Entraînement de chaque modèle
        successful_models = 0
        
        for i, model_name in enumerate(model_names, 1):
            logger.info(f"🔧 Processing model {i}/{len(model_names)}: {model_name}")
            
            # Vérification du temps total
            total_duration = monitor.get_total_duration()
            if total_duration > MAX_TRAINING_TIME:
                logger.warning(f"⏰ Temps d'entraînement dépassé ({total_duration:.0f}s > {MAX_TRAINING_TIME}s)")
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": "Temps d'entraînement maximum dépassé"}
                })
                continue
            
            # Vérification des ressources
            resource_info = monitor.check_resources()
            if resource_info.get('memory_percent', 0) > 90:
                logger.warning("🧹 Mémoire élevée, nettoyage...")
                gc.collect()
            
            # Création du pipeline LEAK-FREE
            pipeline, param_grid = create_leak_free_pipeline(
                model_name=model_name,
                task_type=task_type,
                column_types=column_types,
                preprocessing_choices=preprocessing_choices,
                use_smote=use_smote,
                optimize_hyperparams=optimize
            )
            
            if pipeline is None:
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": "Erreur création du pipeline"}
                })
                continue
            
            # Entraînement selon le type de tâche
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
            
            # Traitement des résultats
            if training_result["success"] and training_result["model"] is not None:
                try:
                    # Évaluation robuste avec EvaluationMetrics
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
                    
                    # Sauvegarde du modèle
                    timestamp = int(time.time())
                    model_filename = f"{model_name.replace(' ', '_').lower()}_{task_type}_{timestamp}.joblib"
                    model_path = os.path.join("models_output", model_filename)
                    
                    joblib.dump(training_result["model"], model_path)
                    
                    # Résultat complet
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

                    # Ajout des données pour visualisations (avec gestion mémoire)
                    try:
                        max_samples = 1000
                        if task_type == 'clustering':
                            # Échantillonnage pour clustering
                            if X is not None and len(X) > 0:
                                X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)
                                if hasattr(training_result["model"], 'predict'):
                                    cluster_labels = training_result["model"].predict(X_sample)
                                else:
                                    cluster_labels = training_result["model"].fit_predict(X_sample)
                                result["X_sample"] = X_sample
                                result["labels"] = cluster_labels
                        else:
                            # Supervisé - échantillonnage des données de test
                            if X_test is not None and len(X_test) > 0:
                                X_sample = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)
                                result["X_sample"] = X_sample
                                # Ajout de X_test, y_test, X_train, y_train pour les visualisations
                                result["X_test"] = X_test
                                result["y_test"] = y_test
                                result["X_train"] = X_train
                                result["y_train"] = y_train
                                if task_type == 'classification' and label_encoder is not None:
                                    result["class_names"] = label_encoder.classes_.tolist()

                            logger.debug(f"📈 Données de visualisation ajoutées pour {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Erreur ajout données visualisation pour {model_name}: {e}")
                        result["metrics"]["visualization_error"] = str(e)

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
                # Échec de l'entraînement
                error_msg = training_result.get("error", "Erreur inconnue lors de l'entraînement")
                logger.error(f"❌ Échec entraînement {model_name}: {error_msg}")
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": error_msg},
                    "training_time": training_result["training_time"]
                })
            
            # Nettoyage mémoire après chaque modèle
            gc.collect()
        
        # Rapport final
        total_time = monitor.get_total_duration()
        monitor_summary = monitor.get_summary()
        
        # Sauvegarde du log d'entraînement
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
        with open(os.path.join("training_logs", log_filename), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        logger.info(f"🎯 Entraînement terminé: {successful_models}/{len(model_names)} "
                   f"modèles réussis en {total_time:.2f}s")
        
        # Nettoyage mémoire final
        gc.collect()
        
    except Exception as e:
        logger.error(f"💥 Erreur critique dans train_models: {e}", exc_info=True)
        results.append({
            "model_name": "Système",
            "metrics": {"error": f"Erreur critique: {str(e)}"}
        })
    
    return results

def cleanup_models_directory(max_files: int = 20):
    """
    Nettoie le dossier des modèles pour éviter l'accumulation.
    
    Args:
        max_files: Nombre maximum de fichiers à conserver
    """
    try:
        if not os.path.exists("models_output"):
            return
            
        model_files = []
        for filename in os.listdir("models_output"):
            if filename.endswith('.joblib'):
                filepath = os.path.join("models_output", filename)
                model_files.append((filepath, os.path.getctime(filepath)))
        
        # Trier par date de création (plus ancien en premier)
        model_files.sort(key=lambda x: x[1])
        
        # Supprimer les fichiers excédentaires
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
    'cleanup_models_directory'
]