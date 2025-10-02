"""
Configuration et fixtures pour les tests automatisés.
Support complet pour les tests de données, modèles et intégration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Supprimer les warnings pour les tests
warnings.filterwarnings("ignore")

@pytest.fixture(scope="session")
def test_data_dir():
    """Répertoire temporaire pour les données de test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_dataframe():
    """DataFrame d'exemple pour les tests."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'numeric_col': np.random.normal(100, 15, n_samples),
        'categorical_col': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'binary_col': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'target_classification': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'target_regression': np.random.normal(50, 10, n_samples),
        'missing_col': np.random.choice([1, 2, 3, np.nan], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    })

@pytest.fixture
def sample_series():
    """Série d'exemple pour les tests."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, 100), name='test_series')

@pytest.fixture
def imbalanced_dataframe():
    """DataFrame avec déséquilibre de classes."""
    np.random.seed(42)
    n_samples = 1000
    
    # 90% de classe 0, 10% de classe 1
    target = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.choice(['A', 'B'], n_samples),
        'target': target
    })

@pytest.fixture
def large_dataframe():
    """DataFrame volumineux pour tester les performances."""
    np.random.seed(42)
    n_samples = 10000
    
    return pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(10, 2, n_samples),
        'categorical': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })

@pytest.fixture
def corrupted_dataframe():
    """DataFrame avec des problèmes de qualité."""
    return pd.DataFrame({
        'mixed_types': [1, 'a', 2, 'b', 3],
        'all_nan': [np.nan, np.nan, np.nan],
        'constant': [1, 1, 1, 1],
        'duplicates': [1, 2, 1, 2, 1, 2],
        'target': [0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def model_training_data():
    """Données optimisées pour l'entraînement de modèles."""
    from sklearn.datasets import make_classification, make_regression
    
    # Données de classification
    X_class, y_class = make_classification(
        n_samples=500, n_features=10, n_classes=3, 
        n_redundant=2, random_state=42
    )
    
    classification_data = pd.DataFrame(
        X_class, columns=[f'feature_{i}' for i in range(10)]
    )
    classification_data['target'] = y_class
    
    # Données de régression
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=8, noise=0.1, random_state=42
    )
    
    regression_data = pd.DataFrame(
        X_reg, columns=[f'feature_{i}' for i in range(8)]
    )
    regression_data['target'] = y_reg
    
    return {
        'classification': classification_data,
        'regression': regression_data
    }

@pytest.fixture
def drift_test_data():
    """Données pour tester la détection de drift."""
    np.random.seed(42)
    
    # Données baseline
    baseline = pd.DataFrame({
        'normal_feature': np.random.normal(100, 10, 1000),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
        'binary_feature': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    # Données avec drift (moyenne différente)
    drifted = pd.DataFrame({
        'normal_feature': np.random.normal(110, 10, 1000),  # Moyenne différente
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.3, 0.5, 0.2]),  # Distribution différente
        'binary_feature': np.random.choice([0, 1], 1000, p=[0.5, 0.5])  # Distribution différente
    })
    
    return {
        'baseline': baseline,
        'drifted': drifted,
        'no_drift': baseline.sample(1000, random_state=42)  # Même distribution
    }

@pytest.fixture
def mock_model():
    """Modèle mock pour les tests."""
    class MockModel:
        def __init__(self):
            self.is_fitted = True
        
        def predict(self, X):
            np.random.seed(42)
            return np.random.choice([0, 1], len(X))
        
        def predict_proba(self, X):
            np.random.seed(42)
            prob_0 = np.random.uniform(0.3, 0.7, len(X))
            prob_1 = 1 - prob_0
            return np.column_stack([prob_0, prob_1])
        
        def fit(self, X, y):
            return self
    
    return MockModel()

@pytest.fixture
def mock_preprocessor():
    """Préprocesseur mock pour les tests."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['numeric_col', 'binary_col']),
        ('cat', categorical_transformer, ['categorical_col'])
    ])
    
    return preprocessor

@pytest.fixture
def sample_config():
    """Configuration d'exemple pour les tests."""
    return {
        'TRAINING_CONSTANTS': {
            'MIN_SAMPLES_REQUIRED': 10,
            'MAX_FEATURES': 50,
            'CV_FOLDS': 3,
            'RANDOM_STATE': 42
        },
        'PREPROCESSING_CONSTANTS': {
            'NUMERIC_IMPUTATION_DEFAULT': 'mean',
            'CATEGORICAL_IMPUTATION_DEFAULT': 'most_frequent',
            'SCALING_METHOD': 'standard'
        }
    }

@pytest.fixture
def mock_logger():
    """Logger mock pour les tests."""
    import logging
    
    class MockLogger:
        def __init__(self):
            self.messages = []
        
        def info(self, msg):
            self.messages.append(('INFO', msg))
        
        def warning(self, msg):
            self.messages.append(('WARNING', msg))
        
        def error(self, msg):
            self.messages.append(('ERROR', msg))
        
        def debug(self, msg):
            self.messages.append(('DEBUG', msg))
        
        def get_messages(self, level=None):
            if level:
                return [msg for lvl, msg in self.messages if lvl == level]
            return self.messages
        
        def clear(self):
            self.messages = []
    
    return MockLogger()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configuration automatique pour tous les tests."""
    # Définir les variables d'environnement pour les tests
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'  # Réduire les logs pendant les tests
    
    yield
    
    # Nettoyage après les tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def performance_benchmark():
    """Benchmark de performance pour les tests."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return PerformanceBenchmark()

# Fixtures pour les tests de données spécifiques
@pytest.fixture
def text_data():
    """Données texte pour les tests de NLP."""
    return pd.DataFrame({
        'text': [
            'This is a positive review about the product.',
            'Negative experience with customer service.',
            'Average quality, nothing special.',
            'Excellent product, highly recommended!',
            'Terrible quality, waste of money.'
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'category': ['product', 'service', 'product', 'product', 'product']
    })

@pytest.fixture
def time_series_data():
    """Données de séries temporelles pour les tests."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 10, 365).cumsum(),
        'trend': np.linspace(0, 10, 365),
        'seasonal': 5 * np.sin(2 * np.pi * np.arange(365) / 365),
        'noise': np.random.normal(0, 2, 365)
    })

@pytest.fixture
def high_cardinality_data():
    """Données avec haute cardinalité pour les tests."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'high_card_cat': [f'category_{i}' for i in np.random.randint(0, 1000, 100)],
        'normal_cat': np.random.choice(['A', 'B', 'C'], 100),
        'numeric': np.random.normal(0, 1, 100),
        'target': np.random.choice([0, 1], 100)
    })

# Utilitaires pour les tests
def assert_dataframe_quality(df, min_rows=1, min_cols=1, max_null_ratio=1.0):
    """Assertion pour la qualité des DataFrames."""
    assert len(df) >= min_rows, f"DataFrame trop petit: {len(df)} < {min_rows}"
    assert len(df.columns) >= min_cols, f"Pas assez de colonnes: {len(df.columns)} < {min_cols}"
    
    null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    assert null_ratio <= max_null_ratio, f"Trop de valeurs manquantes: {null_ratio:.2%} > {max_null_ratio:.2%}"

def assert_model_performance(metrics, min_accuracy=0.5):
    """Assertion pour les performances de modèle."""
    if 'accuracy' in metrics:
        assert metrics['accuracy'] >= min_accuracy, f"Accuracy trop faible: {metrics['accuracy']:.3f} < {min_accuracy}"
    
    if 'r2' in metrics:
        assert metrics['r2'] >= -1.0, f"R² invalide: {metrics['r2']:.3f}"

def assert_drift_detection_results(results, expected_drift=False):
    """Assertion pour les résultats de détection de drift."""
    assert isinstance(results, dict), "Les résultats doivent être un dictionnaire"
    
    for feature_name, result in results.items():
        assert hasattr(result, 'drift_detected'), f"Résultat manquant pour {feature_name}"
        assert hasattr(result, 'drift_score'), f"Score de drift manquant pour {feature_name}"
        assert 0 <= result.drift_score <= 1, f"Score de drift invalide: {result.drift_score}"

# Configuration pytest
def pytest_configure(config):
    """Configuration globale de pytest."""
    config.addinivalue_line(
        "markers", "slow: marque les tests lents"
    )
    config.addinivalue_line(
        "markers", "integration: marque les tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "data_quality: marque les tests de qualité des données"
    )

def pytest_collection_modifyitems(config, items):
    """Modifie la collection des tests."""
    for item in items:
        # Marquer automatiquement les tests basés sur le nom
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "quality" in item.nodeid or "validation" in item.nodeid:
            item.add_marker(pytest.mark.data_quality)
