# E:\gemini\app-analyse\ml\evaluation.py

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def calculate_global_metrics(y_true_all, y_pred_all, y_proba_all, task_type, label_encoder=None, X_data=None):
    """
    Calcule les métriques de performance sur un ensemble agrégé de prédictions.
    """
    metrics = {}

    try:
        # Aplatir les listes de listes/arrays venant de la validation croisée
        y_true_flat = np.concatenate(y_true_all) if len(y_true_all) > 0 else np.array([])
        y_pred_flat = np.concatenate(y_pred_all) if len(y_pred_all) > 0 else np.array([])
    except (ValueError, TypeError):
        # Fallback pour des formats inattendus
        y_true_flat = np.array(y_true_all).flatten()
        y_pred_flat = np.array(y_pred_all).flatten()

    if task_type == "classification":
        if label_encoder is not None:
            y_true_original = label_encoder.inverse_transform(y_true_flat)
            y_pred_original = label_encoder.inverse_transform(y_pred_flat)
        else:
            y_true_original = y_true_flat
            y_pred_original = y_pred_flat

        if pd.Series(y_true_original).nunique() <= 1:
            return {"error": "Cible avec une seule classe."}

        metrics["accuracy"] = accuracy_score(y_true_original, y_pred_original)
        metrics["classification_report"] = classification_report(y_true_original, y_pred_original, output_dict=True, zero_division=0)

        if y_proba_all is not None and len(y_proba_all) > 0:
            y_proba_flat = np.concatenate(y_proba_all, axis=0)
            if y_proba_flat.ndim > 1 and y_proba_flat.shape[1] > 2:
                metrics["roc_auc"] = roc_auc_score(y_true_flat, y_proba_flat, multi_class='ovr', average='weighted')
            else:
                y_proba_binary = y_proba_flat[:, 1] if y_proba_flat.ndim > 1 else y_proba_flat
                metrics["roc_auc"] = roc_auc_score(y_true_flat, y_proba_binary)
        else:
            metrics["roc_auc"] = "N/A"

    elif task_type == "regression":
        metrics["mse"] = mean_squared_error(y_true_flat, y_pred_flat)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true_flat, y_pred_flat)
        metrics["r2"] = r2_score(y_true_flat, y_pred_flat)

    elif task_type == "unsupervised":
        # Pour non-supervisé, y_pred_all contient les labels de cluster
        labels = y_pred_flat
        if X_data is None:
            return {"error": "Les données X sont requises pour l'évaluation non supervisée."}
        
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            metrics["silhouette_score"] = silhouette_score(X_data, labels)
            metrics["davies_bouldin_score"] = davies_bouldin_score(X_data, labels)
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_data, labels)
        else:
            metrics["silhouette_score"] = "N/A (1 seul cluster)"
            metrics["davies_bouldin_score"] = "N/A (1 seul cluster)"
            metrics["calinski_harabasz_score"] = "N/A (1 seul cluster)"
        metrics["n_clusters"] = len(unique_labels)

    else:
        metrics["error"] = f"Type de tâche non supporté: {task_type}"

    return metrics

def evaluate_single_train_test_split(model, X_test, y_test, task_type="classification", label_encoder=None):
    """
    Évalue un modèle sur un unique jeu de test (utilisé en fallback si la CV échoue).
    """
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    return calculate_global_metrics([y_test], [y_pred], [y_proba], task_type, label_encoder, X_data=X_test)