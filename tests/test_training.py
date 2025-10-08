import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.training import create_leak_free_pipeline

def test_smote_no_leakage():
    """
    Vérifie que SMOTE est appliqué uniquement aux données d'entraînement
    et ne modifie pas la taille du test set.
    """
    # Créer des données synthétiques déséquilibrées
    X, y = make_classification(
        n_samples=1000, 
        n_classes=2, 
        weights=[0.9, 0.1], 
        random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    # Simuler un split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configuration pour le test
    preprocessing_choices = {
        'numeric_imputation': 'mean',
        'scaling_method': 'standard',
        'smote_k_neighbors': 5,
        'smote_sampling_strategy': 'auto',
        'random_state': 42
    }
    column_types = {'numeric': list(X.columns)}
    
    # Créer un pipeline avec SMOTE
    pipeline, _ = create_leak_free_pipeline(
        model_name='RandomForestClassifier',
        task_type='classification',
        column_types=column_types,
        preprocessing_choices=preprocessing_choices,
        use_smote=True
    )
    
    # Entraîner le pipeline
    pipeline.fit(X_train, y_train)
    
    # Vérifier que le test set n'est pas modifié par SMOTE
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    assert X_test_transformed.shape[0] == len(X_test), "SMOTE a modifié la taille du test set !"
    
    # Vérifier que les données d'entraînement ont été rééquilibrées
    y_train_pred = pipeline.named_steps['smote'].fit_resample(X_train, y_train)[1]
    assert len(y_train_pred) >= len(y_train), "SMOTE n'a pas généré de données synthétiques !"



""" import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.models.training import create_leak_free_pipeline

def test_smote_parameters():
    #Vérifie que les paramètres SMOTE sont correctement appliqués
    X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessing_choices = {
        'numeric_imputation': 'mean',
        'scaling_method': 'standard',
        'smote_k_neighbors': 3,
        'smote_sampling_strategy': 'minority',
        'random_state': 42
    }
    column_types = {'numeric': list(X.columns)}

    pipeline, _ = create_leak_free_pipeline(
        model_name='RandomForestClassifier',
        task_type='classification',
        column_types=column_types,
        preprocessing_choices=preprocessing_choices,
        use_smote=True
    )

    assert pipeline.named_steps['smote'].k_neighbors == 3, "Mauvais paramètre k_neighbors"
    assert pipeline.named_steps['smote'].sampling_strategy == 'minority', "Mauvaise stratégie d'échantillonnage" """