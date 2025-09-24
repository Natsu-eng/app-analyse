import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from sklearn.exceptions import ConvergenceWarning

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import des modules de l'application
from ml.catalog import MODEL_CATALOG, get_model_config
from ml.data_preprocessing import create_preprocessor, safe_label_encode
from ml.evaluation import calculate_global_metrics, evaluate_single_train_test_split
from utils.data_analysis import auto_detect_column_types, get_target_and_task
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Constantes de configuration
MAX_TRAINING_TIME = 3600  # 1 heure maximum par modèle
MEMORY_LIMIT_MB = 4000    # 4GB mémoire limite
MIN_SAMPLES_REQUIRED = 10

class TrainingMonitor:
    """Monitor pour suivre la progression et les ressources pendant l'entraînement"""
    
    def __init__(self):
        self.start_time = None
        self.model_start_time = None
        self.memory_usage = []
        
    def start_training(self):
        """Démarre le monitoring de l'entraînement"""
        self.start_time = time.time()
        self.memory_usage = []
        logger.info("🚀 Début du monitoring de l'entraînement")
        
    def start_model(self, model_name: str):
        """Démarre le monitoring pour un modèle spécifique"""
        self.model_start_time = time.time()
        logger.info(f"🤖 Début de l'entraînement pour: {model_name}")
        
    def check_resources(self) -> Dict[str, Any]:
        """Vérifie l'utilisation des ressources"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            resource_info = {
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'timestamp': time.time()
            }
            
            self.memory_usage.append(resource_info)
            
            # Alerte si utilisation élevée
            if memory.percent > 85:
                logger.warning(f"⚠️ Utilisation mémoire élevée: {memory.percent:.1f}%")
            if cpu_percent > 90:
                logger.warning(f"⚠️ Utilisation CPU élevée: {cpu_percent:.1f}%")
                
            return resource_info
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des ressources: {e}")
            return {}
    
    def get_model_duration(self) -> float:
        """Retourne la durée d'entraînement du modèle actuel"""
        if self.model_start_time:
            return time.time() - self.model_start_time
        return 0.0
    
    def get_total_duration(self) -> float:
        """Retourne la durée totale d'entraînement"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

def validate_training_data(X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entraînement de façon robuste.
    
    Args:
        X: Features
        y: Target
        task_type: Type de tâche ML
    
    Returns:
        Dict avec les résultats de validation
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "samples_count": len(X),
        "features_count": len(X.columns) if hasattr(X, 'columns') else X.shape[1]
    }
    
    try:
        # Vérification des dimensions
        if len(X) == 0:
            validation["is_valid"] = False
            validation["issues"].append("Dataset vide")
            return validation
            
        if len(X) < MIN_SAMPLES_REQUIRED:
            validation["is_valid"] = False
            validation["issues"].append(f"Trop peu d'échantillons ({len(X)} < {MIN_SAMPLES_REQUIRED})")
            
        if validation["features_count"] == 0:
            validation["is_valid"] = False
            validation["issues"].append("Aucune feature disponible")
            
        # Vérification de la target
        if y is not None:
            if len(y) != len(X):
                validation["is_valid"] = False
                validation["issues"].append("Dimensions X et y incohérentes")
                
            if task_type in ['classification', 'regression']:
                valid_target_count = y.notna().sum() if hasattr(y, 'notna') else np.sum(~np.isnan(y))
                if valid_target_count < MIN_SAMPLES_REQUIRED:
                    validation["is_valid"] = False
                    validation["issues"].append(f"Trop peu de targets valides ({valid_target_count})")
                    
                # Pour la classification, vérifier le nombre de classes
                if task_type == 'classification':
                    unique_classes = np.unique(y.dropna()) if hasattr(y, 'dropna') else np.unique(y[~np.isnan(y)])
                    if len(unique_classes) < 2:
                        validation["is_valid"] = False
                        validation["issues"].append("Moins de 2 classes distinctes")
                    elif len(unique_classes) > 100:
                        validation["warnings"].append("Plus de 100 classes - vérifiez la variable cible")
                        
        # Vérification de la mémoire
        try:
            if hasattr(X, 'memory_usage'):
                x_memory = X.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                if x_memory > MEMORY_LIMIT_MB:
                    validation["warnings"].append(f"Dataset volumineux ({x_memory:.1f}MB)")
        except:
            pass
            
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur de validation: {str(e)}")
        logger.error(f"Validation error: {e}")
    
    return validation

def create_model_pipeline(
    model_name: str, 
    task_type: str, 
    preprocessor: Pipeline, 
    use_smote: bool = False,
    optimize_hyperparams: bool = False
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    Crée un pipeline de modèle avec gestion robuste des erreurs.
    
    Args:
        model_name: Nom du modèle
        task_type: Type de tâche
        preprocessor: Préprocesseur
        use_smote: Utiliser SMOTE pour l'équilibrage
        optimize_hyperparams: Optimiser les hyperparamètres
    
    Returns:
        Tuple (pipeline, param_grid)
    """
    try:
        # Récupération de la configuration du modèle
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
            
        model = model_config["model"]
        param_grid = model_config["params"] if optimize_hyperparams else {}
        
        # Extraction des étapes du préprocesseur
        preprocessor_steps = preprocessor.steps if hasattr(preprocessor, 'steps') else [('preprocessor', preprocessor)]
        
        # Construction du pipeline
        if use_smote and task_type == 'classification':
            # Pipeline avec SMOTE (nécessite imblearn)
            pipeline_steps = preprocessor_steps + [('smote', SMOTE(random_state=42))] + [('model', model)]
            pipeline = ImbPipeline(pipeline_steps)
        else:
            # Pipeline standard
            pipeline_steps = preprocessor_steps + [('model', model)]
            pipeline = Pipeline(pipeline_steps)
        
        logger.info(f"Pipeline créé pour {model_name} avec {len(pipeline_steps)} étapes")
        return pipeline, param_grid
        
    except Exception as e:
        logger.error(f"Erreur création pipeline pour {model_name}: {e}")
        return None, None

def train_single_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict = None,
    task_type: str = 'classification',
    monitor: TrainingMonitor = None
) -> Dict[str, Any]:
    """
    Entraîne un modèle unique avec gestion robuste des erreurs.
    
    Args:
        model_name: Nom du modèle
        pipeline: Pipeline du modèle
        X_train: Données d'entraînement
        y_train: Target d'entraînement
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
        "best_params": None
    }
    
    start_time = time.time()
    
    try:
        if monitor:
            monitor.start_model(model_name)
        
        # Optimisation des hyperparamètres si demandée
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation des hyperparamètres pour {model_name}")
            
            # Configuration de GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,  # Cross-validation réduite pour la performance
                scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=1,  # Éviter le parallélisme pour la stabilité
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["success"] = True
            
            logger.info(f"✅ Optimisation terminée pour {model_name}. Meilleurs paramètres: {grid_search.best_params_}")
            
        else:
            # Entraînement simple
            pipeline.fit(X_train, y_train)
            result["model"] = pipeline
            result["success"] = True
            logger.info(f"✅ Entraînement terminé pour {model_name}")
        
        result["training_time"] = time.time() - start_time
        
        # Vérification des ressources pendant l'entraînement
        if monitor:
            resource_info = monitor.check_resources()
            logger.info(f"⏱️ {model_name} entraîné en {result['training_time']:.2f}s - RAM: {resource_info.get('memory_percent', 0):.1f}%")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur entraînement {model_name}: {e}")
        
        # Tentative de fallback avec configuration simplifiée
        logger.info(f"🔄 Tentative de fallback pour {model_name}")
        try:
            # Réessayer avec des paramètres par défaut
            simple_pipeline = Pipeline([
                ('simple_imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('model', get_model_config(task_type, model_name)["model"])
            ])
            simple_pipeline.fit(X_train, y_train)
            result["model"] = simple_pipeline
            result["success"] = True
            result["error"] = None
            logger.info(f"✅ Fallback réussi pour {model_name}")
        except Exception as fallback_error:
            logger.error(f"❌ Fallback échoué pour {model_name}: {fallback_error}")
    
    return result

def train_models(
    df: pd.DataFrame,
    target_column: str,
    model_names: List[str],
    task_type: str,
    test_size: float = 0.2,
    optimize: bool = False,
    feature_list: List[str] = None,
    use_smote: bool = False,
    preprocessing_choices: Dict = None
) -> List[Dict[str, Any]]:
    """
    Orchestre l'entraînement et l'évaluation de plusieurs modèles.
    Version robuste avec gestion complète des erreurs.
    
    Args:
        df: DataFrame contenant les données
        target_column: Colonne cible
        model_names: Liste des noms de modèles à entraîner
        task_type: Type de tâche ML
        test_size: Proportion des données de test
        optimize: Optimiser les hyperparamètres
        feature_list: Liste des features à utiliser
        use_smote: Utiliser SMOTE pour l'équilibrage
        preprocessing_choices: Options de prétraitement
    
    Returns:
        Liste des résultats d'entraînement
    """
    # Initialisation
    results = []
    monitor = TrainingMonitor()
    monitor.start_training()
    
    # Validation des entrées
    if not feature_list:
        feature_list = [col for col in df.columns if col != target_column]
    
    if not preprocessing_choices:
        preprocessing_choices = {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'remove_constant_cols': True,
            'remove_identifier_cols': True
        }
    
    # Création du dossier de sortie
    os.makedirs("models_output", exist_ok=True)
    
    try:
        # Préparation des données
        logger.info("📊 Préparation des données...")
        
        X = df[feature_list].copy()
        y_raw = df[target_column].copy()
        
        # Validation des données
        data_validation = validate_training_data(X, y_raw, task_type)
        if not data_validation["is_valid"]:
            error_msg = f"Données invalides: {', '.join(data_validation['issues'])}"
            logger.error(error_msg)
            return [{"model_name": "Validation", "metrics": {"error": error_msg}}]
        
        # Encodage de la target si nécessaire
        y_encoded, label_encoder, encoding_map = safe_label_encode(y_raw)
        y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
        
        # Détection des types de colonnes pour le préprocesseur
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            column_types = auto_detect_column_types(X)
        
        # Création du préprocesseur
        preprocessor = create_preprocessor(preprocessing_choices, column_types)
        
        # Split train-test
        stratification = y if task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=stratification
        )
        
        logger.info(f"📁 Données splitées: train={len(X_train)}, test={len(X_test)}")
        
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
                logger.warning("🔄 Mémoire élevée, nettoyage...")
                gc.collect()
            
            # Création du pipeline
            pipeline, param_grid = create_model_pipeline(
                model_name, task_type, preprocessor, use_smote, optimize
            )
            
            if pipeline is None:
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": "Erreur création du pipeline"}
                })
                continue
            
            # Entraînement du modèle
            training_result = train_single_model(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                param_grid=param_grid,
                task_type=task_type,
                monitor=monitor
            )
            
            if training_result["success"] and training_result["model"] is not None:
                # Évaluation du modèle
                try:
                    # Prédictions
                    y_pred = training_result["model"].predict(X_test)
                    y_proba = None
                    if hasattr(training_result["model"], "predict_proba"):
                        y_proba = training_result["model"].predict_proba(X_test)
                    
                    # Calcul des métriques
                    metrics = calculate_global_metrics(
                        y_true_all=[y_test],
                        y_pred_all=[y_pred], 
                        y_proba_all=[y_proba] if y_proba is not None else [],
                        task_type=task_type,
                        label_encoder=label_encoder
                    )
                    
                    # Sauvegarde du modèle
                    model_filename = f"{model_name.replace(' ', '_').lower()}_{int(time.time())}.joblib"
                    model_path = os.path.join("models_output", model_filename)
                    
                    joblib.dump(training_result["model"], model_path)
                    
                    # Résultat complet
                    result = {
                        "model_name": model_name,
                        "metrics": metrics,
                        "model_path": model_path,
                        "training_time": training_result["training_time"],
                        "best_params": training_result.get("best_params"),
                        "model": training_result["model"],
                        "label_encoder": label_encoder,
                        "feature_names": feature_list,
                        "task_type": task_type
                    }
                    
                    results.append(result)
                    successful_models += 1
                    
                    logger.info(f"✅ {model_name} - entraînement réussi en {training_result['training_time']:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur évaluation {model_name}: {e}")
                    results.append({
                        "model_name": model_name,
                        "metrics": {"error": f"Erreur évaluation: {str(e)}"},
                        "training_time": training_result["training_time"]
                    })
            else:
                # Échec de l'entraînement
                results.append({
                    "model_name": model_name,
                    "metrics": {"error": training_result.get("error", "Erreur inconnue")},
                    "training_time": training_result["training_time"]
                })
        
        # Rapport final
        total_time = monitor.get_total_duration()
        logger.info(f"🎯 Entraînement terminé: {successful_models}/{len(model_names)} modèles réussis en {total_time:.2f}s")
        
        # Nettoyage mémoire
        gc.collect()
        
    except Exception as e:
        logger.error(f"❌ Erreur critique dans train_models: {e}", exc_info=True)
        results.append({
            "model_name": "Système",
            "metrics": {"error": f"Erreur critique: {str(e)}"}
        })
    
    return results

def cleanup_models_directory(max_files: int = 10):
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
                logger.info(f"🧹 Fichier modèle supprimé: {filepath}")
                
    except Exception as e:
        logger.error(f"Erreur nettoyage dossier modèles: {e}")

# Nettoyage automatique au chargement du module
cleanup_models_directory()