from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, 
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier

import logging
from typing import Dict, Any, Optional, List
import warnings

# Configuration
warnings.filterwarnings("ignore")

# Fallbacks pour les librairies optionnelles
XGBRegressor, XGBClassifier, LGBMRegressor, LGBMClassifier = None, None, None, None

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    logging.warning("XGBoost non disponible - installation: pip install xgboost")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    logging.warning("LightGBM non disponible - installation: pip install lightgbm")

# Catalogue complet des modèles avec hyperparamètres et descriptions
MODEL_CATALOG = {
    "regression": {
        "Régression Linéaire": {
            "model": LinearRegression(),
            "params": {
                "model__fit_intercept": [True, False],
                "model__normalize": [True, False]
            },
            "description": "Modèle linéaire simple, adapté aux relations linéaires entre variables."
        },
        "Régression Ridge": {
            "model": Ridge(random_state=42),
            "params": {
                "model__alpha": [0.1, 1.0, 10.0],
                "model__solver": ['auto', 'svd', 'cholesky']
            },
            "description": "Régression linéaire avec régularisation L2, robuste aux variables corrélées."
        },
        "Régression Lasso": {
            "model": Lasso(random_state=42),
            "params": {
                "model__alpha": [0.1, 1.0, 10.0],
                "model__selection": ['cyclic', 'random']
            },
            "description": "Régression linéaire avec régularisation L1, favorise la sparsité des features."
        },
        "Forêt Aléatoire": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2]
            },
            "description": "Ensemble d'arbres de décision, robuste aux données bruitées et hétérogènes."
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0]
            },
            "description": "Boosting d'arbres, performant pour les relations complexes."
        },
        "SVM Regression": {
            "model": SVR(),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ['linear', 'rbf'],
                "model__gamma": ['scale', 'auto']
            },
            "description": "Support Vector Machine pour régression, efficace pour les datasets de petite taille."
        },
        "K Plus Proches Voisins": {
            "model": KNeighborsRegressor(),
            "params": {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree']
            },
            "description": "Prédiction basée sur les voisins les plus proches, sensible à la normalisation."
        },
        "Arbre de Décision": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },
            "description": "Arbre unique, simple mais sujet au surapprentissage."
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0, 1.5]
            },
            "description": "Boosting d'arbres faibles, adapté aux datasets bruités."
        },
        "Réseau de Neurones": {
            "model": MLPRegressor(random_state=42, max_iter=1000),
            "params": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__activation": ['relu', 'tanh'],
                "model__alpha": [0.0001, 0.001, 0.01]
            },
            "description": "Réseau neuronal multicouche, puissant pour les données complexes."
        }
    },
    "classification": {
        "Régression Logistique": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ['l1', 'l2', 'elasticnet'],
                "model__solver": ['liblinear', 'saga']
            },
            "description": "Modèle linéaire interprétable, adapté aux classifications binaires."
        },
        "Forêt Aléatoire": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": [None, 'balanced']
            },
            "description": "Ensemble d'arbres, robuste aux données bruitées et hétérogènes."
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            },
            "description": "Boosting d'arbres, performant pour les classifications complexes."
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ['linear', 'rbf'],
                "model__gamma": ['scale', 'auto']
            },
            "description": "Support Vector Machine, efficace pour les datasets de petite taille."
        },
        "K Plus Proches Voisins": {
            "model": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree']
            },
            "description": "Classification basée sur les voisins, sensible à la normalisation."
        },
        "Arbre de Décision": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, 'balanced']
            },
            "description": "Arbre unique, simple mais sujet au surapprentissage."
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0, 1.5]
            },
            "description": "Boosting d'arbres faibles, adapté aux datasets bruités."
        },
        "Naive Bayes Gaussien": {
            "model": GaussianNB(),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7]
            },
            "description": "Modèle probabiliste simple, efficace pour les données continues."
        },
        "Analyse Discriminante Linéaire": {
            "model": LinearDiscriminantAnalysis(),
            "params": {
                "model__solver": ['svd', 'lsqr', 'eigen']
            },
            "description": "Classification linéaire, bonne pour les données linéairement séparables."
        },
        "Réseau de Neurones": {
            "model": MLPClassifier(random_state=42, max_iter=1000),
            "params": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__activation": ['relu', 'tanh'],
                "model__alpha": [0.0001, 0.001, 0.01]
            },
            "description": "Réseau neuronal multicouche, puissant pour les données complexes."
        }
    },
    "clustering": {
        "K-Means": {
            "model": KMeans(random_state=42, n_init=10),
            "params": {
                "model__n_clusters": [2, 3, 4, 5, 6, 7, 8],
                "model__init": ['k-means++', 'random'],
                "model__algorithm": ['lloyd', 'elkan']
            },
            "description": "Clustering en clusters sphériques de taille similaire."
        },
        "DBSCAN": {
            "model": DBSCAN(),
            "params": {
                "model__eps": [0.3, 0.5, 0.7, 1.0],
                "model__min_samples": [5, 10, 15]
            },
            "description": "Clustering robuste au bruit, détecte des clusters de forme arbitraire."
        },
        "Clustering Hiérarchique": {
            "model": AgglomerativeClustering(),
            "params": {
                "model__n_clusters": [2, 3, 4, 5],
                "model__linkage": ['ward', 'complete', 'average', 'single'],
                "model__metric": ['euclidean', 'manhattan', 'cosine']
            },
            "description": "Construit une hiérarchie de clusters, flexible pour différentes formes."
        },
        "ACP (PCA)": {
            "model": PCA(random_state=42),
            "params": {
                "model__n_components": [2, 3, 4, 0.95, 0.99],
                "model__svd_solver": ['auto', 'full', 'arpack', 'randomized']
            },
            "description": "Réduction dimensionnelle, conserve les composantes principales."
        },
        "SVD Tronqué": {
            "model": TruncatedSVD(random_state=42),
            "params": {
                "model__n_components": [2, 3, 4, 5],
                "model__algorithm": ['arpack', 'randomized']
            },
            "description": "Réduction dimensionnelle pour matrices creuses ou grandes."
        }
    }
}

# Ajout des modèles optionnels si disponibles
if XGBRegressor:
    MODEL_CATALOG["regression"]["XGBoost"] = {
        "model": XGBRegressor(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        },
        "description": "Boosting d'arbres optimisé, performant pour les données structurées."
    }

if XGBClassifier:
    MODEL_CATALOG["classification"]["XGBoost"] = {
        "model": XGBClassifier(random_state=42, eval_metric='logloss'),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        },
        "description": "Boosting d'arbres optimisé, performant pour les classifications complexes."
    }

if LGBMRegressor:
    MODEL_CATALOG["regression"]["LightGBM"] = {
        "model": LGBMRegressor(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__num_leaves": [31, 50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        },
        "description": "Boosting d'arbres rapide, efficace pour les grands datasets."
    }

if LGBMClassifier:
    MODEL_CATALOG["classification"]["LightGBM"] = {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__num_leaves": [31, 50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        },
        "description": "Boosting d'arbres rapide, efficace pour les classifications complexes."
    }

def get_model_config(task_type: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Récupère la configuration d'un modèle de façon sécurisée.  
    Args:
        task_type: Type de tâche
        model_name: Nom du modèle   
    Returns:
        Configuration du modèle ou None si non trouvé
    """
    try:
        if not isinstance(task_type, str) or not isinstance(model_name, str):
            logging.error("Les paramètres 'task_type' et 'model_name' doivent être des chaînes.")
            return None
        
        if task_type not in MODEL_CATALOG:
            logging.error(f"Type de tâche non supporté: {task_type}")
            return None
        
        if model_name not in MODEL_CATALOG[task_type]:
            logging.error(f"Modèle non trouvé: {model_name} pour {task_type}")
            return None
        
        model_config = MODEL_CATALOG[task_type][model_name]
        logging.info(f"✅ Configuration récupérée pour le modèle '{model_name}' de type '{task_type}'")
        return model_config
        
    except Exception as e:
        logging.error(f"Erreur récupération configuration modèle: {e}", exc_info=True)
        return None

def get_available_models(task_type: str) -> List[str]:
    """
    Retourne la liste des modèles disponibles pour un type de tâche.
    
    Args:
        task_type: Type de tâche
    
    Returns:
        Liste des noms de modèles
    """
    try:
        return list(MODEL_CATALOG.get(task_type, {}).keys())
    except Exception as e:
        logging.error(f"Erreur récupération modèles disponibles: {e}")
        return []

def validate_model_selection(task_type: str, model_names: List[str]) -> Dict[str, Any]:
    """
    Valide la sélection de modèles pour une tâche donnée.
    
    Args:
        task_type: Type de tâche
        model_names: Liste des noms de modèles
    
    Returns:
        Résultat de validation
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "available_models": get_available_models(task_type)
    }
    
    if task_type not in MODEL_CATALOG:
        validation["is_valid"] = False
        validation["errors"].append(f"Type de tâche non supporté: {task_type}")
        return validation
    
    if not model_names:
        validation["is_valid"] = False
        validation["errors"].append("Aucun modèle sélectionné")
        return validation
    
    available_models = set(validation["available_models"])
    selected_models = set(model_names)
    
    unavailable_models = selected_models - available_models
    if unavailable_models:
        validation["warnings"].append(f"Modèles non disponibles: {unavailable_models}")
        model_names = [m for m in model_names if m in available_models]
    
    if not model_names:
        validation["is_valid"] = False
        validation["errors"].append("Aucun modèle valide sélectionné")
    
    return validation