from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Fallbacks for optional libraries
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor, XGBClassifier = None, None
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor, LGBMClassifier = None, None

# Model catalog with hyperparameter grids for GridSearchCV
# Keys ('regression', 'classification') MUST match the output of `get_task_type` in `utils/data_analysis.py`
MODEL_CATALOG = {
    "regression": {
        "Linear Regression": {
            "model": LinearRegression(), 
            "params": {}
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1]
            }
        }
    },
    "classification": {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "model__C": [0.1, 1.0, 10.0]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1]
            }
        }
    },
    "unsupervised": {
        "KMeans": {
            "model": KMeans(random_state=42, n_init='auto'),
            "params": {
                "model__n_clusters": [2, 3, 4, 5, 6]
            }
        },
        "DBSCAN": {
            "model": DBSCAN(),
            "params": {
                "model__eps": [0.3, 0.5, 0.7],
                "model__min_samples": [5, 10]
            }
        },
        "PCA": {
            "model": PCA(),
            "params": {
                "model__n_components": [2, 3, 4]
            }
        }
    }
}

# Add optional models if they were imported successfully
if XGBRegressor:
    MODEL_CATALOG["regression"]["XGBoost"] = {
        "model": XGBRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10],
            "model__learning_rate": [0.05, 0.1]
        }
    }
if XGBClassifier:
    MODEL_CATALOG["classification"]["XGBoost"] = {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10],
            "model__learning_rate": [0.05, 0.1]
        }
    }

if LGBMRegressor:
    MODEL_CATALOG["regression"]["LightGBM"] = {
        "model": LGBMRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__num_leaves": [31, 50],
            "model__learning_rate": [0.05, 0.1]
        }
    }
if LGBMClassifier:
    MODEL_CATALOG["classification"]["LightGBM"] = {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__num_leaves": [31, 50],
            "model__learning_rate": [0.05, 0.1]
        }
    }