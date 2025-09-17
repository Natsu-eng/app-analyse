# E:\gemini\app-analyse\models.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

# Preprocessing & Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

# --- Configuration centralisée des modèles et hyperparamètres ---

MODEL_CATALOG = {
    "regression": {
        "Linear Regression": {
            "model": LinearRegression,
            "params": {},
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
                "max_depth": {"type": "int", "min": 3, "max": 20, "default": 5},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
            },
        },
        "XGBoost Regressor": {
            "model": XGBRegressor,
            "params": {
                "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
                "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
                "max_depth": {"type": "int", "min": 3, "max": 10, "default": 3},
            },
        },
        "Support Vector Regressor (SVR)": {
            "model": SVR,
            "params": {
                "C": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
                "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
            },
        },
    },
    "classification": {
        "Logistic Regression": {
            "model": LogisticRegression,
            "params": {
                "C": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
                "solver": {"type": "select", "options": ["liblinear", "lbfgs"], "default": "liblinear"},
            },
        },
        "Random Forest Classifier": {
            "model": RandomForestClassifier,
            "params": {
                "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
                "max_depth": {"type": "int", "min": 3, "max": 20, "default": 5},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
            },
        },
        "XGBoost Classifier": {
            "model": XGBClassifier,
            "params": {
                "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
                "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": 0.1, "step": 0.01},
                "max_depth": {"type": "int", "min": 3, "max": 10, "default": 3},
            },
        },
        "Support Vector Classifier (SVC)": {
            "model": SVC,
            "params": {
                "C": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
                "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
                "probability": {"type": "static", "value": True},
            },
        },
    },
}

# --- Fonctions de calcul des métriques ---

def _calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }

def _calculate_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted"),
        "Matthews CorrCoef": matthews_corrcoef(y_true, y_pred),
    }
    # ROC-AUC est binaire ou multi-classe (One-vs-Rest)
    if y_proba.shape[1] == 2: # Binaire
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba[:, 1])
    else: # Multi-classe
        metrics["ROC-AUC (OvR)"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    return metrics

# --- Fonction principale d'entraînement ---

def train_and_evaluate(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_name: str,
    task_type: str,
    test_size: float,
    model_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fonction centrale pour le prétraitement, l'entraînement et l'évaluation d'un modèle.
    """
    X = df[feature_columns]
    y = df[target_column]

    # 1. Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task_type == "classification" else None)

    # 2. Création du pipeline de prétraitement
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    # 3. Instanciation du modèle avec les hyperparamètres
    model_class = MODEL_CATALOG[task_type][model_name]["model"]
    model = model_class(**model_params)

    # 4. Création du pipeline complet (prétraitement + modèle)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 5. Entraînement
    pipeline.fit(X_train, y_train)

    # 6. Prédictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if task_type == "classification" else None

    # 7. Évaluation
    if task_type == "regression":
        metrics = _calculate_regression_metrics(y_test, y_pred)
    else:
        metrics = _calculate_classification_metrics(y_test, y_pred, y_proba)

    # 8. Construire la liste des noms de features après transformation
    final_feature_names = numeric_features.copy()
    if categorical_features:
        try:
            ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            final_feature_names.extend(list(ohe_feature_names))
        except NotFittedError:
            # This case can happen if the categorical columns only contained NaNs and were dropped.
            pass

    # 9. Retourner les résultats
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": final_feature_names
    }