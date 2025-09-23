# E:\gemini\app-analyse\utils\model_persistence.py

# CHANGE: Nouveau module pour la sauvegarde et le chargement des modèles.

import joblib
import os
import logging

logger = logging.getLogger(__name__)

def save_model_to_disk(model, model_name: str, save_dir="outputs/models"):
    """
    Sauvegarde un modèle entraîné sur le disque en utilisant joblib.

    Args:
        model: Le modèle (pipeline) à sauvegarder.
        model_name (str): Le nom du modèle, utilisé pour le nom de fichier.
        save_dir (str): Le répertoire où sauvegarder le modèle.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{model_name}.joblib")
        joblib.dump(model, file_path)
        logger.info(f"Modèle '{model_name}' sauvegardé avec succès à : {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle '{model_name}': {e}", exc_info=True)
        return None

def load_model_from_disk(file_path: str):
    """
    Charge un modèle depuis un fichier joblib.

    Args:
        file_path (str): Le chemin vers le fichier .joblib.

    Returns:
        Le modèle chargé, ou None si une erreur survient.
    """
    try:
        model = joblib.load(file_path)
        logger.info(f"Modèle chargé avec succès depuis : {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Fichier modèle non trouvé à : {file_path}")
        return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle depuis '{file_path}': {e}", exc_info=True)
        return None
