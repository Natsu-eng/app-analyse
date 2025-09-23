import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml.catalog import MODEL_CATALOG
from ml.data_preprocessing import create_preprocessor
from ml.evaluation import calculate_global_metrics

logger = logging.getLogger(__name__)

def train_models(df, target_column, model_names, task_type, test_size, optimize, feature_list, use_smote, preprocessing_choices):
    """
    Orchestre l'entraînement et l'évaluation de plusieurs modèles en fonction du type de tâche.
    """
    results = []
    os.makedirs("models_output", exist_ok=True)

    # --- LOGIQUE POUR TÂCHES SUPERVISÉES ---
    if task_type in ['classification', 'regression']:
        if not target_column or target_column not in df.columns:
            logger.error("Colonne cible non valide ou non fournie pour une tâche supervisée.")
            return []
        
        X = df[feature_list]
        y_raw = df[target_column]

        # Gérer l'encodage de la cible ici pour le passer aux résultats
        label_encoder = None
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_raw):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index, name=target_column)
        else:
            y = y_raw

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task_type == 'classification' else None)
        preprocessor = create_preprocessor(preprocessing_choices)

        for model_name in model_names:
            logger.info(f"--- Début du processus supervisé pour: {model_name} ---")
            try:
                model_config = MODEL_CATALOG[task_type][model_name]
                model = model_config["model"]

                # Extraire les étapes du préprocesseur
                preprocessor_steps = preprocessor.steps

                # Construire le pipeline complet en aplatissant les étapes du préprocesseur
                all_steps = preprocessor_steps + [('model', model)]

                if task_type == 'classification' and use_smote:
                    # Insère SMOTE avant le modèle dans un pipeline imblearn
                    # La pipeline imblearn doit être utilisée si SMOTE est activé
                    pipeline = ImbPipeline(all_steps[:-1] + [('sampler', SMOTE(random_state=42))] + [all_steps[-1]])
                else:
                    pipeline = Pipeline(all_steps)

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
                
                metrics = calculate_global_metrics(y_true_all=[y_test], y_pred_all=[y_pred], y_proba_all=[y_proba] if y_proba is not None else [], task_type=task_type)
                
                model_path = f"models_output/{model_name.replace(' ', '_')}.joblib"
                joblib.dump(pipeline, model_path)

                results.append({
                    "model_name": model_name, "metrics": metrics, "model_path": model_path,
                    'y_test_encoded': y_test.values, 'y_pred_encoded': y_pred, 'y_proba': y_proba,
                    'X_test_raw': X_test, 'X_test_processed': X_test_processed, 'model': pipeline,
                    'label_encoder': label_encoder
                })
            except Exception as e:
                logger.exception(f"Échec de l'entraînement supervisé pour {model_name}")
                results.append({"model_name": model_name, "metrics": {"error": str(e)}})

    # --- LOGIQUE POUR TÂCHES NON SUPERVISÉES ---
    elif task_type == 'unsupervised':
        X = df[feature_list]
        preprocessor = create_preprocessor(preprocessing_choices)

        for model_name in model_names:
            logger.info(f"--- Début du processus non supervisé pour: {model_name} ---")
            try:
                model_config = MODEL_CATALOG[task_type][model_name]
                model = model_config["model"]

                pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                
                y_pred = pipeline.fit_predict(X)
                X_processed = pipeline.named_steps['preprocessor'].transform(X)

                metrics = calculate_global_metrics(y_true_all=[], y_pred_all=[y_pred], y_proba_all=[], task_type=task_type, X_data=X_processed)

                model_path = f"models_output/{model_name.replace(' ', '_')}.joblib"
                joblib.dump(pipeline, model_path)

                results.append({
                    "model_name": model_name, "metrics": metrics, "model_path": model_path, 'model': pipeline
                })

            except Exception as e:
                logger.exception(f"Échec de l'entraînement non supervisé pour {model_name}")
                results.append({"model_name": model_name, "metrics": {"error": str(e)}})

    else:
        logger.error(f"Type de tâche inconnu : {task_type}")

    return results