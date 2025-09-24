import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Any, Optional, Union
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class SafeColumnCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoyage robuste des colonnes avec gestion d'erreurs avancée.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cols_to_drop_ = []
        self.cols_to_keep_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Identifie les colonnes à supprimer"""
        self.cols_to_drop_ = []
        self.cols_to_keep_ = list(X.columns)
        
        try:
            # Suppression des colonnes constantes
            if self.config.get('remove_constant_cols', True):
                constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
                self.cols_to_drop_.extend(constant_cols)
                logger.info(f"Colonnes constantes identifiées: {constant_cols}")
            
            # Suppression des colonnes de type ID
            if self.config.get('remove_identifier_cols', True):
                id_cols = [col for col in X.columns if X[col].nunique() == len(X)]
                self.cols_to_drop_.extend(id_cols)
                logger.info(f"Colonnes ID identifiées: {id_cols}")
            
            # Suppression des colonnes avec trop de valeurs manquantes
            missing_threshold = self.config.get('missing_threshold', 0.8)
            if missing_threshold < 1.0:
                high_missing_cols = [
                    col for col in X.columns 
                    if X[col].isna().mean() > missing_threshold
                ]
                self.cols_to_drop_.extend(high_missing_cols)
                logger.info(f"Colonnes avec trop de valeurs manquantes: {high_missing_cols}")
            
            # Déduplication
            self.cols_to_drop_ = list(set(self.cols_to_drop_))
            
            # Colonnes à conserver
            self.cols_to_keep_ = [col for col in X.columns if col not in self.cols_to_drop_]
            
            logger.info(f"Nettoyage colonnes: {len(self.cols_to_drop_)} à supprimer, {len(self.cols_to_keep_)} à conserver")
            
        except Exception as e:
            logger.error(f"Erreur lors du fit de SafeColumnCleaner: {e}")
            # Fallback: conserver toutes les colonnes
            self.cols_to_keep_ = list(X.columns)
            self.cols_to_drop_ = []
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique la transformation de nettoyage"""
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X doit être un pandas.DataFrame")
            
            # Suppression des colonnes identifiées
            X_clean = X.drop(columns=self.cols_to_drop_, errors='ignore')
            
            # Vérification qu'il reste des colonnes
            if X_clean.shape[1] == 0:
                logger.warning("Aucune colonne après nettoyage - fallback sur colonnes originales")
                return X
            
            logger.info(f"Après nettoyage: {X_clean.shape[1]} colonnes restantes")
            return X_clean
            
        except Exception as e:
            logger.error(f"Erreur transformation SafeColumnCleaner: {e}")
            # Fallback: retourner les données originales
            return X

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Imputation intelligente adaptée aux types de données.
    """
    
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent', k=5):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.k = k
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Adapte l'imputation aux types de colonnes"""
        try:
            # Identification des types de colonnes
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Imputation numérique
            if self.numeric_strategy == 'knn' and len(self.numeric_cols_) > 0:
                self.numeric_imputer_ = KNNImputer(n_neighbors=min(self.k, len(X)))
            else:
                self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
            
            if len(self.numeric_cols_) > 0:
                self.numeric_imputer_.fit(X[self.numeric_cols_])
            
            # Imputation catégorielle
            self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
            if len(self.categorical_cols_) > 0:
                self.categorical_imputer_.fit(X[self.categorical_cols_])
                
        except Exception as e:
            logger.error(f"Erreur fit SmartImputer: {e}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique l'imputation"""
        try:
            X_transformed = X.copy()
            
            # Imputation numérique
            if len(self.numeric_cols_) > 0 and self.numeric_imputer_ is not None:
                numeric_imputed = self.numeric_imputer_.transform(X[self.numeric_cols_])
                X_transformed[self.numeric_cols_] = numeric_imputed
            
            # Imputation catégorielle
            if len(self.categorical_cols_) > 0 and self.categorical_imputer_ is not None:
                categorical_imputed = self.categorical_imputer_.transform(X[self.categorical_cols_])
                X_transformed[self.categorical_cols_] = categorical_imputed
                
            return X_transformed
            
        except Exception as e:
            logger.error(f"Erreur transformation SmartImputer: {e}")
            return X

def safe_label_encode(y: pd.Series) -> tuple:
    """
    Encodage sécurisé des labels avec gestion des erreurs.
    
    Args:
        y: Série à encoder
    
    Returns:
        Tuple (y_encoded, label_encoder, encoding_map)
    """
    try:
        if pd.api.types.is_numeric_dtype(y):
            # Déjà numérique
            return y.values, None, {}
        
        # Encodage des labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Mapping pour référence
        encoding_map = {i: label for i, label in enumerate(label_encoder.classes_)}
        
        return y_encoded, label_encoder, encoding_map
        
    except Exception as e:
        logger.error(f"Erreur encodage labels: {e}")
        # Fallback: encodage manuel simple
        unique_labels = y.unique()
        encoding_map = {i: label for i, label in enumerate(unique_labels)}
        reverse_map = {label: i for i, label in encoding_map.items()}
        y_encoded = y.map(reverse_map).values
        return y_encoded, None, encoding_map

def create_preprocessor(preprocessing_config: Dict, column_types: Dict = None) -> Pipeline:
    """
    Crée un pipeline de prétraitement robuste et adaptable.
    
    Args:
        preprocessing_config: Configuration du prétraitement
        column_types: Types de colonnes détectés
    
    Returns:
        Pipeline de prétraitement
    """
    try:
        # Configuration par défaut
        default_config = {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'missing_threshold': 0.8,
            'scaling_method': 'standard',
            'encoding_method': 'onehot'
        }
        
        config = {**default_config, **preprocessing_config}
        
        # Étape 1: Nettoyage des colonnes
        column_cleaner = SafeColumnCleaner(config)
        
        # Étape 2: Imputation intelligente
        imputer = SmartImputer(
            numeric_strategy=config['numeric_imputation'],
            categorical_strategy=config['categorical_imputation']
        )
        
        # Étape 3: Scaling et encoding (via ColumnTransformer)
        numeric_features = column_types.get('numeric', []) if column_types else []
        categorical_features = column_types.get('categorical', []) if column_types else []
        text_features = column_types.get('text_or_high_cardinality', []) if column_types else []
        
        # Sélection du scaler numérique
        if config['scaling_method'] == 'robust':
            numeric_scaler = RobustScaler()
        elif config['scaling_method'] == 'minmax':
            numeric_scaler = MinMaxScaler()
        else:
            numeric_scaler = StandardScaler()
        
        # Sélection de l'encodeur catégoriel
        if config['encoding_method'] == 'ordinal':
            categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Transformers pour chaque type de données
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=config['numeric_imputation'])),
                ('scaler', numeric_scaler)
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=config['categorical_imputation'])),
                ('encoder', categorical_encoder)
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Pour les colonnes texte/haute cardinalité, traitement spécifique
        if text_features:
            text_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=config['categorical_imputation'], fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('text', text_transformer, text_features))
        
        # ColumnTransformer principal
        if transformers:
            column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',  # Conserver les autres colonnes
                n_jobs=1  # Éviter le parallélisme pour la stabilité
            )
        else:
            # Fallback si aucun transformer n'est défini
            column_transformer = ColumnTransformer(
                transformers=[('passthrough', 'passthrough', make_column_selector(dtype_include=np.number))],
                remainder='passthrough'
            )
        
        # Pipeline complète
        preprocessor_pipeline = Pipeline(steps=[
            ('cleaner', column_cleaner),
            ('imputer', imputer),
            ('transformer', column_transformer)
        ])
        
        logger.info("✅ Pipeline de prétraitement créée avec succès")
        return preprocessor_pipeline
        
    except Exception as e:
        logger.error(f"❌ Erreur création préprocesseur: {e}")
        
        # Fallback: pipeline minimale
        fallback_pipeline = Pipeline(steps=[
            ('simple_imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        
        logger.info("🔄 Utilisation du pipeline de fallback")
        return fallback_pipeline

def validate_preprocessor(preprocessor: Pipeline, X_sample: pd.DataFrame) -> Dict[str, Any]:
    """
    Valide le fonctionnement d'un préprocesseur sur un échantillon.
    
    Args:
        preprocessor: Pipeline à valider
        X_sample: Échantillon de données
    
    Returns:
        Rapport de validation
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "transformed_shape": None,
        "execution_time": 0
    }
    
    try:
        import time
        
        start_time = time.time()
        X_transformed = preprocessor.fit_transform(X_sample)
        validation["execution_time"] = time.time() - start_time
        
        validation["transformed_shape"] = X_transformed.shape
        
        # Vérifications
        if np.any(np.isnan(X_transformed)):
            validation["warnings"].append("Valeurs NaN détectées après transformation")
        
        if np.any(np.isinf(X_transformed)):
            validation["warnings"].append("Valeurs infinies détectées après transformation")
        
        if X_transformed.shape[1] == 0:
            validation["is_valid"] = False
            validation["issues"].append("Aucune feature après transformation")
        
        logger.info(f"✅ Validation préprocesseur: {X_transformed.shape} en {validation['execution_time']:.2f}s")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"❌ Erreur validation préprocesseur: {e}")
    
    return validation