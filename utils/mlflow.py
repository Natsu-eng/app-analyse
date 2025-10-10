import os
import subprocess
import time
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from src.evaluation.metrics import MetricsLogger


# Intégration MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def is_mlflow_available() -> bool:
    """Vérifie si MLflow est disponible et configuré."""
    return MLFLOW_AVAILABLE


def get_git_info() -> Dict[str, str]:
    """
    Récupère les informations Git du projet pour la traçabilité.
    
    Returns:
        Dictionnaire contenant commit_hash, branch, et is_dirty
    """
    git_info = {
        "commit_hash": "unknown",
        "branch": "unknown",
        "is_dirty": "unknown"
    }
    
    try:
        # Commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()[:8]
        
        # Branche
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # État dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            git_info["is_dirty"] = "yes" if result.stdout.strip() else "no"
    
    except Exception as e:
        logger.debug(f"Impossible de récupérer les infos Git: {e}")
    
    return git_info


def clean_model_name(name: str) -> str:
    """
    Nettoie un nom de modèle pour MLflow (ASCII seulement, pas d'espaces).
    
    Args:
        name: Nom du modèle à nettoyer
        
    Returns:
        Nom nettoyé (lowercase, underscores, ASCII)
    """
    # Remplacer les espaces et tirets par underscores
    cleaned = name.replace(' ', '_').replace('-', '_').lower()
    
    # Garder seulement les caractères alphanumériques et underscores
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
    
    # Éviter les underscores multiples
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    
    return cleaned.strip('_')


def format_mlflow_run_for_ui(run_info, metrics, preprocessing_choices, model_name, timestamp):
    try:
        return {
            "run_id": run_info.info.run_id,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)},
            "tags": {
                "mlflow.runName": f"{model_name}_{timestamp}",
                "task_type": preprocessing_choices.get("task_type", "clustering"),
                **{f"preprocessing_{k}": str(v) for k, v in preprocessing_choices.items()}
            },
            "start_time": run_info.info.start_time,
            "status": run_info.info.status,
            "model_name": model_name
        }
    except Exception as e:
        MetricsLogger.log_structured("WARNING", f"Échec formatage MLflow run: {str(e)}")
        return None

def save_artifacts_for_mlflow(
    task_type: str,
    model,
    X_test_vis,
    y_test_vis,
    X_sample,
    labels_sample,
    temp_dir: str
) -> List[str]:
    """
    Sauvegarde les artefacts de visualisation dans des fichiers temporaires pour MLflow.
    
    Args:
        task_type: Type de tâche ('classification', 'regression', 'clustering')
        model: Modèle entraîné
        X_test_vis: Données de test pour visualisation
        y_test_vis: Labels de test pour visualisation
        X_sample: Échantillon de données
        labels_sample: Labels de clustering
        temp_dir: Répertoire temporaire pour sauvegarder les fichiers
        
    Returns:
        Liste des chemins de fichiers créés
    """
    artifact_paths = []
    
    try:
        if task_type == 'classification' and X_test_vis is not None and y_test_vis is not None:
            # Prédictions
            y_pred = model.predict(X_test_vis)
            
            # Confusion matrix data
            cm_path = os.path.join(temp_dir, "confusion_matrix_data.joblib")
            joblib.dump({"y_test": y_test_vis, "y_pred": y_pred}, cm_path)
            artifact_paths.append(cm_path)
            
            # ROC curve data (si predict_proba disponible)
            if hasattr(model, 'predict_proba') or (hasattr(model, 'named_steps') and hasattr(list(model.named_steps.values())[-1], 'predict_proba')):
                try:
                    y_proba = model.predict_proba(X_test_vis)
                    roc_path = os.path.join(temp_dir, "roc_data.joblib")
                    joblib.dump({"y_test": y_test_vis, "y_proba": y_proba}, roc_path)
                    artifact_paths.append(roc_path)
                except Exception as e:
                    logger.debug(f"Impossible de sauvegarder ROC data: {e}")
        
        elif task_type == 'regression' and X_test_vis is not None and y_test_vis is not None:
            # Prédictions
            y_pred = model.predict(X_test_vis)
            
            # Convertir en numpy arrays si nécessaire
            if isinstance(y_test_vis, pd.Series):
                y_test_np = y_test_vis.values
            else:
                y_test_np = np.array(y_test_vis)
            
            if isinstance(y_pred, pd.Series):
                y_pred_np = y_pred.values
            else:
                y_pred_np = np.array(y_pred)
            
            residuals = y_test_np - y_pred_np
            
            # Residuals data
            res_path = os.path.join(temp_dir, "residuals_data.joblib")
            joblib.dump({"y_pred": y_pred_np, "residuals": residuals}, res_path)
            artifact_paths.append(res_path)
            
            # Predictions vs Actual
            pred_path = os.path.join(temp_dir, "predictions_data.joblib")
            joblib.dump({"y_test": y_test_np, "y_pred": y_pred_np}, pred_path)
            artifact_paths.append(pred_path)
        
        elif task_type == 'clustering' and X_sample is not None and labels_sample is not None:
            # Cluster data
            cluster_path = os.path.join(temp_dir, "cluster_data.joblib")
            joblib.dump({"X_sample": X_sample, "labels": labels_sample}, cluster_path)
            artifact_paths.append(cluster_path)
    
    except Exception as e:
        logger.warning(f"⚠️ Erreur sauvegarde artefacts: {e}")
    
    return artifact_paths


def reset_mlflow_session():
    """Réinitialise proprement la session MLflow."""
    try:
        import streamlit as st
        st.session_state.mlflow_runs = []
        logger.info("✅ Session MLflow réinitialisée")
    except ImportError:
        logger.debug("Streamlit non disponible")
    except Exception as e:
        logger.error(f"❌ Erreur réinitialisation MLflow: {e}")


# ===============================================
# FONCTIONS INTERNES POUR MÉTRIQUES DE CLUSTERING
# ===============================================
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def _ensure_array_like(X, force_float: bool = False) -> Tuple[np.ndarray, bool, Any]:
    """
    Convertit les données en array numpy de façon robuste - VERSION AMÉLIORÉE.
    
    Args:
        X: Données d'entrée (DataFrame, array, liste, etc.)
        force_float: Forcer la conversion en float (déconseillé sauf nécessité)
    
    Returns:
        Tuple (array, is_dataframe, index)
    """
    try:
        if X is None:
            logger.warning("Données None fournies à _ensure_array_like")
            return np.array([]), False, np.array([])
        
        # Cas DataFrame pandas - CONSERVER les types d'origine
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(), True, X.index
        
        # Cas Series pandas
        if isinstance(X, pd.Series):
            values = X.to_numpy()
            # Reshape seulement si nécessaire pour clustering
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            return values, True, X.index
        
        # Conversion générique en array numpy SANS forcer le type
        try:
            # Essai de conservation des types d'origine
            arr = np.asarray(X)
            
            if arr.size == 0:
                logger.warning("Tableau vide après conversion")
                return arr, False, np.arange(0)
            
            # Vérification de la compatibilité numérique
            if force_float and not np.issubdtype(arr.dtype, np.number):
                try:
                    arr = arr.astype(float)
                except (ValueError, TypeError):
                    logger.warning("Conversion en float échouée, conservation du type original")
            
            # Reshape si 1D pour clustering (besoin 2D)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
                
            return arr, False, np.arange(arr.shape[0])
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Conversion directe échouée: {e}, tentative via DataFrame")
            # Fallback via DataFrame pour données complexes
            try:
                X_df = pd.DataFrame(X)
                return X_df.to_numpy(), True, X_df.index
            except Exception as df_error:
                logger.error(f"Échec conversion via DataFrame: {df_error}")
                return np.array([]), False, np.array([])
                
    except Exception as e:
        logger.error(f"Erreur critique dans _ensure_array_like: {e}")
        return np.array([]), False, np.array([])

def _safe_cluster_metrics(X, labels, min_samples_per_cluster: int = 5) -> Dict[str, Any]:
    """
    Calcule les métriques de clustering de façon ultra-robuste - VERSION AMÉLIORÉE.
    
    Args:
        X: Données features
        labels: Labels des clusters
        min_samples_per_cluster: Minimum d'échantillons par cluster pour calcul fiable
    
    Returns:
        Dict avec métriques et informations de diagnostic
    """
    metrics = {
        "silhouette_score": np.nan,
        "calinski_harabasz_score": np.nan, 
        "davies_bouldin_score": np.nan,
        "n_clusters": 0,
        "n_samples": 0,
        "n_noise": 0,
        "cluster_sizes": [],
        "is_valid": False,
        "warnings": [],
        "data_quality": "unknown"
    }
    
    try:
        # Validation des entrées
        if X is None or labels is None:
            metrics["warnings"].append("Données d'entrée None")
            metrics["data_quality"] = "invalid"
            return metrics
        
        # Conversion robuste SANS forcer le float
        X_arr, _, _ = _ensure_array_like(X, force_float=False)
        labels_arr = np.asarray(labels)
        
        # Validation post-conversion
        if X_arr.size == 0:
            metrics["warnings"].append("Données X vides après conversion")
            metrics["data_quality"] = "invalid"
            return metrics
            
        if labels_arr.size == 0:
            metrics["warnings"].append("Labels vides après conversion")
            metrics["data_quality"] = "invalid"
            return metrics
        
        if len(X_arr) != len(labels_arr):
            metrics["warnings"].append(f"Dimensions incohérentes: X={len(X_arr)}, labels={len(labels_arr)}")
            metrics["data_quality"] = "invalid"
            return metrics
        
        metrics["n_samples"] = len(labels_arr)
        
        # Analyse des clusters
        unique_labels = np.unique(labels_arr)
        valid_labels = unique_labels[unique_labels != -1]  # Exclure le bruit (-1)
        metrics["n_clusters"] = len(valid_labels)
        metrics["n_noise"] = np.sum(labels_arr == -1)
        
        # Calcul de la taille des clusters
        cluster_sizes = []
        for label in valid_labels:
            size = np.sum(labels_arr == label)
            cluster_sizes.append(size)
        metrics["cluster_sizes"] = cluster_sizes
        
        # NOUVEAU : Qualité des données avant calcul
        data_quality_issues = []
        
        # Vérification variance
        try:
            variances = np.var(X_arr, axis=0)
            zero_variance_features = np.sum(variances == 0)
            if zero_variance_features > 0:
                data_quality_issues.append(f"{zero_variance_features} features sans variance")
        except Exception:
            pass
        
        # Vérification NaN/Inf
        try:
            nan_count = np.sum(np.isnan(X_arr))
            inf_count = np.sum(np.isinf(X_arr))
            if nan_count > 0:
                data_quality_issues.append(f"{nan_count} valeurs NaN")
            if inf_count > 0:
                data_quality_issues.append(f"{inf_count} valeurs infinies")
        except Exception:
            pass
        
        # Conditions pour le calcul des métriques - CRITÈRES RENFORCÉS
        has_enough_clusters = metrics["n_clusters"] >= 2
        has_enough_samples = metrics["n_samples"] >= 10  # Augmenté de 3 à 10
        has_valid_data = len(X_arr) > 0 and X_arr.shape[1] > 0
        clusters_have_min_samples = all(size >= min_samples_per_cluster for size in cluster_sizes)
        
        if not has_enough_clusters:
            metrics["warnings"].append(f"Clusters insuffisants: {metrics['n_clusters']} < 2")
            metrics["data_quality"] = "poor"
            return metrics
            
        if not has_enough_samples:
            metrics["warnings"].append(f"Échantillons insuffisants: {metrics['n_samples']} < 10")
            metrics["data_quality"] = "poor"
            return metrics
            
        if not has_valid_data:
            metrics["warnings"].append("Données X invalides pour le clustering")
            metrics["data_quality"] = "invalid"
            return metrics
        
        if not clusters_have_min_samples:
            small_clusters = [size for size in cluster_sizes if size < min_samples_per_cluster]
            metrics["warnings"].append(f"{len(small_clusters)} clusters avec < {min_samples_per_cluster} échantillons")
            metrics["data_quality"] = "questionable"
        
        # Détermination qualité données
        if not data_quality_issues and clusters_have_min_samples:
            metrics["data_quality"] = "good"
        elif data_quality_issues and clusters_have_min_samples:
            metrics["data_quality"] = "acceptable"
        else:
            metrics["data_quality"] = "poor"
            metrics["warnings"].extend(data_quality_issues)
        
        # Filtrage des points valides (non bruit)
        valid_mask = labels_arr != -1
        valid_labels = labels_arr[valid_mask]
        valid_X = X_arr[valid_mask]
        
        n_valid_samples = len(valid_labels)
        n_valid_clusters = len(np.unique(valid_labels))
        
        if n_valid_samples < 10:  # Augmenté le minimum
            metrics["warnings"].append(f"Échantillons valides insuffisants: {n_valid_samples} < 10")
            return metrics
            
        if n_valid_clusters < 2:
            metrics["warnings"].append(f"Clusters valides insuffisants: {n_valid_clusters} < 2")
            return metrics
        
        # NOUVEAU : Conversion en float seulement si nécessaire et possible
        try:
            if not np.issubdtype(valid_X.dtype, np.number):
                logger.info("Conversion des données en float pour calcul métriques clustering")
                valid_X = valid_X.astype(float)
        except (ValueError, TypeError) as e:
            metrics["warnings"].append(f"Impossible de convertir en float: {e}")
            return metrics
        
        # Calcul des métriques avec gestion d'erreurs individuelle
        try:
            # Silhouette Score - nécessite au moins 2 échantillons par cluster
            if n_valid_samples >= 10 and n_valid_clusters >= 2:
                try:
                    # Vérification supplémentaire pour silhouette
                    cluster_sizes_valid = [np.sum(valid_labels == label) for label in np.unique(valid_labels)]
                    if all(size >= 2 for size in cluster_sizes_valid):  # Silhouette nécessite au moins 2 par cluster
                        metrics["silhouette_score"] = float(silhouette_score(valid_X, valid_labels))
                    else:
                        metrics["warnings"].append("Silhouette impossible: clusters avec < 2 échantillons")
                except Exception as e:
                    metrics["warnings"].append(f"Silhouette échoué: {str(e)[:100]}")
            
            # Calinski-Harabasz Score - plus tolérant
            if n_valid_samples >= n_valid_clusters and n_valid_clusters >= 2:
                try:
                    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(valid_X, valid_labels))
                except Exception as e:
                    metrics["warnings"].append(f"Calinski-Harabasz échoué: {str(e)[:100]}")
            
            # Davies-Bouldin Score  
            if n_valid_samples >= n_valid_clusters and n_valid_clusters >= 2:
                try:
                    metrics["davies_bouldin_score"] = float(davies_bouldin_score(valid_X, valid_labels))
                except Exception as e:
                    metrics["warnings"].append(f"Davies-Bouldin échoué: {str(e)[:100]}")
        
        except Exception as e:
            metrics["warnings"].append(f"Erreur générale calcul métriques: {str(e)[:100]}")
        
        # Validation finale des résultats
        valid_metrics = [
            m for m in [metrics["silhouette_score"], metrics["calinski_harabasz_score"], 
                       metrics["davies_bouldin_score"]]
            if not np.isnan(m)
        ]
        
        if len(valid_metrics) > 0:
            metrics["is_valid"] = True
            
        logger.debug(f"Métriques clustering calculées: {len(valid_metrics)}/3 valides, qualité: {metrics['data_quality']}")
        
    except Exception as e:
        logger.error(f"Erreur critique dans _safe_cluster_metrics: {e}")
        metrics["warnings"].append(f"Erreur critique: {str(e)[:100]}")
        metrics["data_quality"] = "error"
    
    return metrics