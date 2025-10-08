import os
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from src.shared.logging import get_logger
logger = get_logger(__name__)

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


def format_mlflow_run_for_ui(
    run_info,
    metrics: Dict[str, Any],
    preprocessing_choices: Dict[str, Any],
    model_name: str,
    timestamp: int
) -> Dict[str, Any]:
    """
    Formate les données d'un run MLflow pour l'interface UI.
    
    Args:
        run_info: Objet run MLflow actif
        metrics: Dict des métriques du modèle
        preprocessing_choices: Dict des choix de preprocessing
        model_name: Nom du modèle
        timestamp: Timestamp du run
        
    Returns:
        Dict formaté pour l'UI avec toutes les clés nécessaires
    """
    run_data = {
        'run id': run_info.info.run_id,  # ✅ Avec espace (pour UI)
        'run_id': run_info.info.run_id,  # Sans espace (pour API)
        'status': 'FINISHED',
        'start_time': run_info.info.start_time,
        'end_time': int(time.time() * 1000),
        'tags.mlflow.runName': f"{model_name}_{timestamp}",
    }
    
    # Ajouter les métriques avec préfixe
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            run_data[f'metrics.{k}'] = float(v)
    
    # Ajouter les paramètres avec préfixe
    for k, v in preprocessing_choices.items():
        run_data[f'params.preprocessing_{k}'] = str(v)[:100]
    
    return run_data


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