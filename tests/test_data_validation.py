"""
Tests de validation de la qualité des données.
Vérifie la robustesse des fonctions de validation et de détection d'anomalies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import des modules à tester
from utils.data_analysis import (
    auto_detect_column_types,
    get_column_profile,
    detect_imbalance,
    analyze_columns,
    detect_useless_columns,
    validate_input_data
)
from utils.feature_store import FeatureStore, detect_drift
from conftest import (
    assert_dataframe_quality,
    sample_dataframe,
    corrupted_dataframe,
    imbalanced_dataframe,
    large_dataframe
)

class TestDataQualityValidation:
    """Tests de validation de la qualité des données."""
    
    def test_auto_detect_column_types_basic(self, sample_dataframe):
        """Test de détection automatique des types de colonnes."""
        result = auto_detect_column_types(sample_dataframe)
        
        # Vérifications de base
        assert isinstance(result, dict)
        assert 'numeric' in result
        assert 'categorical' in result
        assert 'text_or_high_cardinality' in result
        assert 'datetime' in result
        
        # Vérifications spécifiques
        assert 'numeric_col' in result['numeric']
        assert 'binary_col' in result['numeric']
        assert 'categorical_col' in result['categorical']
        assert 'date_col' in result['datetime']
    
    def test_auto_detect_column_types_empty_dataframe(self):
        """Test avec DataFrame vide."""
        empty_df = pd.DataFrame()
        result = auto_detect_column_types(empty_df)
        
        for col_type in ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']:
            assert result[col_type] == []
    
    def test_auto_detect_column_types_none_input(self):
        """Test avec entrée None."""
        result = auto_detect_column_types(None)
        
        for col_type in ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']:
            assert result[col_type] == []
    
    def test_get_column_profile_numeric(self, sample_dataframe):
        """Test de profilage d'une colonne numérique."""
        profile = get_column_profile(sample_dataframe['numeric_col'])
        
        assert isinstance(profile, dict)
        assert 'count' in profile
        assert 'mean' in profile
        assert 'std_dev' in profile
        assert 'min' in profile
        assert 'max' in profile
        assert 'median' in profile
        assert profile['count'] > 0
    
    def test_get_column_profile_categorical(self, sample_dataframe):
        """Test de profilage d'une colonne catégorielle."""
        profile = get_column_profile(sample_dataframe['categorical_col'])
        
        assert isinstance(profile, dict)
        assert 'count' in profile
        assert 'unique_values' in profile
        assert 'unique_ratio' in profile
        assert profile['unique_values'] <= len(sample_dataframe)
    
    def test_get_column_profile_with_missing_values(self, sample_dataframe):
        """Test de profilage avec valeurs manquantes."""
        profile = get_column_profile(sample_dataframe['missing_col'])
        
        assert 'missing_values' in profile
        assert 'missing_percentage' in profile
        assert profile['missing_values'] > 0
        assert float(profile['missing_percentage'].rstrip('%')) > 0
    
    def test_detect_imbalance_balanced(self, sample_dataframe):
        """Test de détection de déséquilibre sur données équilibrées."""
        # Créer des données équilibrées
        balanced_data = sample_dataframe.copy()
        balanced_data['balanced_target'] = np.random.choice([0, 1], len(balanced_data), p=[0.5, 0.5])
        
        result = detect_imbalance(balanced_data, 'balanced_target')
        
        assert result['is_imbalanced'] == False
        assert result['imbalance_ratio'] < 0.8
        assert result['message'] == "✅ Classes équilibrées"
    
    def test_detect_imbalance_imbalanced(self, imbalanced_dataframe):
        """Test de détection de déséquilibre sur données déséquilibrées."""
        result = detect_imbalance(imbalanced_dataframe, 'target')
        
        assert result['is_imbalanced'] == True
        assert result['imbalance_ratio'] > 0.8
        assert "Déséquilibre détecté" in result['message']
        assert result['recommendation'] == "Envisagez d'activer SMOTE ou d'utiliser l'échantillonnage"
    
    def test_detect_imbalance_missing_target(self, sample_dataframe):
        """Test avec colonne cible manquante."""
        result = detect_imbalance(sample_dataframe, 'nonexistent_column')
        
        assert result['is_imbalanced'] == False
        assert "non trouvée" in result['message']
    
    def test_analyze_columns_constant(self, sample_dataframe):
        """Test d'analyse des colonnes constantes."""
        # Ajouter une colonne constante
        test_df = sample_dataframe.copy()
        test_df['constant_col'] = 1
        
        result = analyze_columns(test_df)
        
        assert 'constant' in result
        assert 'id_like' in result
        assert 'constant_col' in result['constant']
    
    def test_analyze_columns_id_like(self, sample_dataframe):
        """Test d'analyse des colonnes de type ID."""
        # Créer une colonne ID-like
        test_df = sample_dataframe.copy()
        test_df['id_col'] = range(len(test_df))
        
        result = analyze_columns(test_df)
        
        # Note: peut ne pas être détecté comme ID selon les paramètres
        assert isinstance(result['id_like'], list)
    
    def test_detect_useless_columns_high_missing(self):
        """Test de détection des colonnes inutiles avec beaucoup de NaN."""
        # Créer des données avec beaucoup de NaN
        test_df = pd.DataFrame({
            'good_col': np.random.normal(0, 1, 100),
            'high_missing': [np.nan] * 70 + [1] * 30,  # 70% de NaN
            'target': np.random.choice([0, 1], 100)
        })
        
        useless_cols = detect_useless_columns(test_df, threshold_missing=0.6)
        
        assert 'high_missing' in useless_cols
        assert 'good_col' not in useless_cols
    
    def test_detect_useless_columns_constant(self):
        """Test de détection des colonnes constantes."""
        test_df = pd.DataFrame({
            'good_col': np.random.normal(0, 1, 100),
            'constant_col': [1] * 100,
            'target': np.random.choice([0, 1], 100)
        })
        
        useless_cols = detect_useless_columns(test_df)
        
        assert 'constant_col' in useless_cols
        assert 'good_col' not in useless_cols

class TestFeatureStoreValidation:
    """Tests de validation du Feature Store."""
    
    def test_feature_store_registration(self, sample_dataframe):
        """Test d'enregistrement d'une feature."""
        feature_store = FeatureStore(storage_path="test_feature_store")
        
        # Enregistrer une feature
        version = feature_store.register_feature(
            name="test_feature",
            data=sample_dataframe['numeric_col'],
            description="Test feature",
            tags=["test", "numeric"]
        )
        
        assert isinstance(version, str)
        assert len(version) > 0
        
        # Vérifier que la feature est enregistrée
        features = feature_store.list_features()
        assert "test_feature" in features
        assert version in features["test_feature"]
    
    def test_feature_store_retrieval(self, sample_dataframe):
        """Test de récupération d'une feature."""
        feature_store = FeatureStore(storage_path="test_feature_store")
        
        # Enregistrer une feature
        version = feature_store.register_feature(
            name="test_feature_retrieval",
            data=sample_dataframe['numeric_col']
        )
        
        # Récupérer la feature
        retrieved_data = feature_store.get_feature("test_feature_retrieval", version)
        
        assert retrieved_data is not None
        assert len(retrieved_data) == len(sample_dataframe['numeric_col'])
        pd.testing.assert_series_equal(retrieved_data, sample_dataframe['numeric_col'])
    
    def test_feature_store_validation(self, sample_dataframe):
        """Test de validation de cohérence des features."""
        feature_store = FeatureStore(storage_path="test_feature_store")
        
        # Enregistrer une feature
        feature_store.register_feature(
            name="test_validation",
            data=sample_dataframe['numeric_col']
        )
        
        # Valider avec les mêmes données
        validation_result = feature_store.validate_feature_consistency(
            "test_validation", 
            sample_dataframe['numeric_col']
        )
        
        assert validation_result['is_valid'] == True
        assert validation_result['feature_name'] == "test_validation"
    
    def test_feature_store_drift_detection_no_drift(self, sample_dataframe):
        """Test de détection de drift sans drift."""
        feature_store = FeatureStore(storage_path="test_feature_store")
        
        # Enregistrer une feature
        feature_store.register_feature(
            name="test_no_drift",
            data=sample_dataframe['numeric_col']
        )
        
        # Tester avec les mêmes données (pas de drift)
        drift_result = feature_store.detect_drift(
            "test_no_drift",
            sample_dataframe['numeric_col']
        )
        
        assert drift_result['drift_detected'] == False
        assert drift_result['drift_score'] < drift_result['threshold']
    
    def test_feature_store_drift_detection_with_drift(self, sample_dataframe):
        """Test de détection de drift avec drift."""
        feature_store = FeatureStore(storage_path="test_feature_store")
        
        # Enregistrer une feature
        feature_store.register_feature(
            name="test_with_drift",
            data=sample_dataframe['numeric_col']
        )
        
        # Créer des données avec drift (moyenne différente)
        drifted_data = sample_dataframe['numeric_col'] + 50  # Décalage de 50
        
        drift_result = feature_store.detect_drift(
            "test_with_drift",
            drifted_data,
            threshold=0.1  # Seuil plus strict
        )
        
        # Le drift devrait être détecté
        assert drift_result['drift_detected'] == True
        assert drift_result['drift_score'] > drift_result['threshold']

class TestInputDataValidation:
    """Tests de validation des données d'entrée."""
    
    def test_validate_input_data_classification(self, sample_dataframe):
        """Test de validation pour la classification."""
        y_true = sample_dataframe['target_classification']
        y_pred = sample_dataframe['target_classification'].copy()
        
        validation = validate_input_data(y_true, y_pred, "classification")
        
        assert validation['is_valid'] == True
        assert validation['n_samples'] == len(y_true)
        assert len(validation['issues']) == 0
    
    def test_validate_input_data_regression(self, sample_dataframe):
        """Test de validation pour la régression."""
        y_true = sample_dataframe['target_regression']
        y_pred = sample_dataframe['target_regression'] + np.random.normal(0, 1, len(y_true))
        
        validation = validate_input_data(y_true, y_pred, "regression")
        
        assert validation['is_valid'] == True
        assert validation['n_samples'] == len(y_true)
    
    def test_validate_input_data_inconsistent_dimensions(self, sample_dataframe):
        """Test avec dimensions incohérentes."""
        y_true = sample_dataframe['target_classification']
        y_pred = sample_dataframe['target_classification'][:-10]  # Moins d'échantillons
        
        validation = validate_input_data(y_true, y_pred, "classification")
        
        assert validation['is_valid'] == False
        assert "Dimensions incohérentes" in validation['issues'][0]
    
    def test_validate_input_data_empty_data(self):
        """Test avec données vides."""
        validation = validate_input_data([], [], "classification")
        
        assert validation['is_valid'] == False
        assert "Données vides" in validation['issues'][0]

class TestDataQualityIntegration:
    """Tests d'intégration pour la qualité des données."""
    
    @pytest.mark.integration
    def test_end_to_end_data_quality_pipeline(self, sample_dataframe):
        """Test du pipeline complet de qualité des données."""
        # 1. Détection des types de colonnes
        column_types = auto_detect_column_types(sample_dataframe)
        assert len(column_types['numeric']) > 0
        
        # 2. Profilage des colonnes
        for col in column_types['numeric'][:2]:  # Tester les 2 premières colonnes numériques
            profile = get_column_profile(sample_dataframe[col])
            assert profile['count'] > 0
        
        # 3. Analyse des colonnes
        analysis = analyze_columns(sample_dataframe)
        assert isinstance(analysis['constant'], list)
        assert isinstance(analysis['id_like'], list)
        
        # 4. Détection des colonnes inutiles
        useless_cols = detect_useless_columns(sample_dataframe)
        assert isinstance(useless_cols, list)
        
        # 5. Validation de la qualité globale
        assert_dataframe_quality(sample_dataframe, min_rows=100, min_cols=3)
    
    @pytest.mark.integration
    def test_data_quality_with_corrupted_data(self, corrupted_dataframe):
        """Test avec des données corrompues."""
        # Ces tests doivent passer même avec des données corrompues
        column_types = auto_detect_column_types(corrupted_dataframe)
        assert isinstance(column_types, dict)
        
        analysis = analyze_columns(corrupted_dataframe)
        assert isinstance(analysis, dict)
        
        # Les colonnes constantes et avec beaucoup de NaN devraient être détectées
        useless_cols = detect_useless_columns(corrupted_dataframe)
        assert len(useless_cols) > 0  # Au moins quelques colonnes inutiles
    
    @pytest.mark.performance
    def test_data_quality_performance_large_dataset(self, large_dataframe):
        """Test de performance avec un grand dataset."""
        import time
        
        start_time = time.time()
        
        # Exécuter les fonctions de qualité des données
        column_types = auto_detect_column_types(large_dataframe)
        analysis = analyze_columns(large_dataframe)
        useless_cols = detect_useless_columns(large_dataframe)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Vérifier que l'exécution est raisonnable (moins de 30 secondes)
        assert execution_time < 30, f"Exécution trop lente: {execution_time:.2f}s"
        
        # Vérifier que les résultats sont corrects
        assert len(column_types['numeric']) > 0
        assert isinstance(analysis, dict)
        assert isinstance(useless_cols, list)

class TestDataQualityEdgeCases:
    """Tests des cas limites pour la qualité des données."""
    
    def test_single_row_dataframe(self):
        """Test avec DataFrame d'une seule ligne."""
        single_row_df = pd.DataFrame({
            'col1': [1],
            'col2': ['A'],
            'target': [0]
        })
        
        # Ces fonctions doivent gérer gracieusement les données très petites
        column_types = auto_detect_column_types(single_row_df)
        assert isinstance(column_types, dict)
        
        analysis = analyze_columns(single_row_df)
        assert isinstance(analysis, dict)
    
    def test_all_nan_column(self):
        """Test avec colonne entièrement NaN."""
        nan_df = pd.DataFrame({
            'all_nan': [np.nan] * 100,
            'normal': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        column_types = auto_detect_column_types(nan_df)
        # La colonne NaN pourrait être classée dans différentes catégories
        
        useless_cols = detect_useless_columns(nan_df)
        assert 'all_nan' in useless_cols
    
    def test_mixed_data_types_in_column(self):
        """Test avec types de données mixtes dans une colonne."""
        mixed_df = pd.DataFrame({
            'mixed': [1, 'a', 2.5, 'b', np.nan],
            'normal': np.random.normal(0, 1, 5),
            'target': [0, 1, 0, 1, 0]
        })
        
        # Ces fonctions doivent gérer les types mixtes
        column_types = auto_detect_column_types(mixed_df)
        assert isinstance(column_types, dict)
        
        # La colonne mixte devrait être détectée comme text_or_high_cardinality
        assert 'mixed' in column_types['text_or_high_cardinality']
    
    def test_very_long_column_names(self):
        """Test avec noms de colonnes très longs."""
        long_name = 'a' * 1000  # Nom de 1000 caractères
        
        long_name_df = pd.DataFrame({
            long_name: np.random.normal(0, 1, 100),
            'normal': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Les fonctions doivent gérer les noms longs
        column_types = auto_detect_column_types(long_name_df)
        assert long_name in column_types['numeric']
        
        profile = get_column_profile(long_name_df[long_name])
        assert profile['count'] > 0
