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

# Catalogue complet des modèles avec hyperparamètres optimisés
MODEL_CATALOG = {
    "regression": {
        "Régression Linéaire": {
            "model": LinearRegression(),
            "params": {
                "model__fit_intercept": [True, False],
                "model__normalize": [True, False]
            }
        },
        "Régression Ridge": {
            "model": Ridge(random_state=42),
            "params": {
                "model__alpha": [0.1, 1.0, 10.0],
                "model__solver": ['auto', 'svd', 'cholesky']
            }
        },
        "Régression Lasso": {
            "model": Lasso(random_state=42),
            "params": {
                "model__alpha": [0.1, 1.0, 10.0],
                "model__selection": ['cyclic', 'random']
            }
        },
        "Forêt Aléatoire": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0]
            }
        },
        "SVM Regression": {
            "model": SVR(),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ['linear', 'rbf'],
                "model__gamma": ['scale', 'auto']
            }
        },
        "K Plus Proches Voisins": {
            "model": KNeighborsRegressor(),
            "params": {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree']
            }
        },
        "Arbre de Décision": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0, 1.5]
            }
        },
        "Réseau de Neurones": {
            "model": MLPRegressor(random_state=42, max_iter=1000),
            "params": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__activation": ['relu', 'tanh'],
                "model__alpha": [0.0001, 0.001, 0.01]
            }
        }
    },
    "classification": {
        "Régression Logistique": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ['l1', 'l2', 'elasticnet'],
                "model__solver": ['liblinear', 'saga']
            }
        },
        "Forêt Aléatoire": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": [None, 'balanced']
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            }
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ['linear', 'rbf'],
                "model__gamma": ['scale', 'auto']
            }
        },
        "K Plus Proches Voisins": {
            "model": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ['uniform', 'distance'],
                "model__algorithm": ['auto', 'ball_tree', 'kd_tree']
            }
        },
        "Arbre de Décision": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, 'balanced']
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0, 1.5]
            }
        },
        "Naive Bayes Gaussien": {
            "model": GaussianNB(),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        },
        "Analyse Discriminante Linéaire": {
            "model": LinearDiscriminantAnalysis(),
            "params": {
                "model__solver": ['svd', 'lsqr', 'eigen']
            }
        },
        "Réseau de Neurones": {
            "model": MLPClassifier(random_state=42, max_iter=1000),
            "params": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__activation": ['relu', 'tanh'],
                "model__alpha": [0.0001, 0.001, 0.01]
            }
        }
    },
    "unsupervised": {
        "K-Means": {
            "model": KMeans(random_state=42, n_init=10),
            "params": {
                "model__n_clusters": [2, 3, 4, 5, 6, 7, 8],
                "model__init": ['k-means++', 'random'],
                "model__algorithm": ['lloyd', 'elkan']
            }
        },
        "DBSCAN": {
            "model": DBSCAN(),
            "params": {
                "model__eps": [0.3, 0.5, 0.7, 1.0],
                "model__min_samples": [5, 10, 15]
            }
        },
        "Clustering Hiérarchique": {
            "model": AgglomerativeClustering(),
            "params": {
                "model__n_clusters": [2, 3, 4, 5],
                "model__linkage": ['ward', 'complete', 'average', 'single'],
                "model__metric": ['euclidean', 'manhattan', 'cosine']
            }
        },
        "ACP (PCA)": {
            "model": PCA(random_state=42),
            "params": {
                "model__n_components": [2, 3, 4, 0.95, 0.99],
                "model__svd_solver": ['auto', 'full', 'arpack', 'randomized']
            }
        },
        "SVD Tronqué": {
            "model": TruncatedSVD(random_state=42),
            "params": {
                "model__n_components": [2, 3, 4, 5],
                "model__algorithm": ['arpack', 'randomized']
            }
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
        }
    }

if XGBClassifier:
    MODEL_CATALOG["classification"]["XGBoost"] = {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }
    }

if LGBMRegressor:
    MODEL_CATALOG["regression"]["LightGBM"] = {
        "model": LGBMRegressor(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__num_leaves": [31, 50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }
    }

if LGBMClassifier:
    MODEL_CATALOG["classification"]["LightGBM"] = {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__num_leaves": [31, 50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }
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
        if task_type not in MODEL_CATALOG:
            logging.error(f"Type de tâche non supporté: {task_type}")
            return None
        
        if model_name not in MODEL_CATALOG[task_type]:
            logging.error(f"Modèle non trouvé: {model_name} pour {task_type}")
            return None
        
        return MODEL_CATALOG[task_type][model_name]
        
    except Exception as e:
        logging.error(f"Erreur récupération configuration modèle: {e}")
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

def get_model_categories() -> Dict[str, List[str]]:
    """
    Retourne les catégories de modèles disponibles.
    
    Returns:
        Dictionnaire {catégorie: [modèles]}
    """
    categories = {}
    
    for task_type, models in MODEL_CATALOG.items():
        categories[task_type] = list(models.keys())
    
    return categories

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
    
    # Vérifier les modèles non disponibles
    unavailable_models = selected_models - available_models
    if unavailable_models:
        validation["warnings"].append(f"Modèles non disponibles: {unavailable_models}")
        # Retirer les modèles non disponibles
        model_names = [m for m in model_names if m in available_models]
    
    if not model_names:
        validation["is_valid"] = False
        validation["errors"].append("Aucun modèle valide sélectionné")
    
    return validation