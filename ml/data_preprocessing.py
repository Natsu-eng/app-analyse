"""
Module de preprocessing robuste pour le machine learning.
Gère le preprocessing pour les tâches supervisées et non-supervisées.
Conforme aux standards MLOps et prêt pour la production.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from typing import Dict, Any, List, Optional, Union
import warnings
import pickle
import gc

# Configuration du logging
logger = logging.getLogger(__name__)

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SafeColumnCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoyage robuste des colonnes avec gestion d'erreurs avancée.
    Conforme aux standards scikit-learn pour l'intégration dans les pipelines.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration du nettoyage avec les clés:
                - remove_constant_cols: Supprimer les colonnes constantes
                - remove_identifier_cols: Supprimer les colonnes de type ID
                - missing_threshold: Seuil pour la suppression des colonnes avec valeurs manquantes
                - min_unique_ratio: Ratio minimum de valeurs uniques pour conserver une colonne
        """
        self.config = config
        self.cols_to_drop_ = []
        self.cols_to_keep_ = []
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y=None) -> 'SafeColumnCleaner':
        """
        Identifie les colonnes à supprimer basé sur la configuration.
        
        Args:
            X: DataFrame d'entraînement
            y: Target (optionnel)
            
        Returns:
            self: Instance ajustée
        """
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X doit être un pandas.DataFrame")
                
            self.cols_to_drop_ = []
            self.cols_to_keep_ = list(X.columns)
            
            logger.info(f"Début du nettoyage des colonnes sur {X.shape[1]} colonnes")
            
            # Suppression des colonnes constantes
            if self.config.get('remove_constant_cols', True):
                constant_cols = []
                for col in X.columns:
                    try:
                        if X[col].nunique(dropna=True) <= 1:
                            constant_cols.append(col)
                    except Exception as e:
                        logger.warning(f"Erreur analyse colonne {col}: {e}")
                        continue
                
                self.cols_to_drop_.extend(constant_cols)
                if constant_cols:
                    logger.info(f"Colonnes constantes identifiées: {constant_cols}")
            
            # Suppression des colonnes de type ID
            if self.config.get('remove_identifier_cols', True):
                id_cols = []
                for col in X.columns:
                    try:
                        if col not in self.cols_to_drop_ and X[col].nunique() == len(X):
                            id_cols.append(col)
                    except Exception as e:
                        logger.warning(f"Erreur analyse ID colonne {col}: {e}")
                        continue
                
                self.cols_to_drop_.extend(id_cols)
                if id_cols:
                    logger.info(f"Colonnes ID identifiées: {id_cols}")
            
            # Suppression des colonnes avec trop de valeurs manquantes
            missing_threshold = self.config.get('missing_threshold', 0.8)
            if missing_threshold < 1.0:
                high_missing_cols = []
                for col in X.columns:
                    if col not in self.cols_to_drop_:
                        try:
                            missing_ratio = X[col].isna().mean()
                            if missing_ratio > missing_threshold:
                                high_missing_cols.append(col)
                        except Exception as e:
                            logger.warning(f"Erreur calcul valeurs manquantes {col}: {e}")
                            continue
                
                self.cols_to_drop_.extend(high_missing_cols)
                if high_missing_cols:
                    logger.info(f"Colonnes avec >{missing_threshold:.0%} valeurs manquantes: {high_missing_cols}")
            
            # Suppression des colonnes avec faible variance
            min_unique_ratio = self.config.get('min_unique_ratio', 0.01)
            if min_unique_ratio > 0:
                low_variance_cols = []
                for col in X.columns:
                    if col not in self.cols_to_drop_ and pd.api.types.is_numeric_dtype(X[col]):
                        try:
                            unique_ratio = X[col].nunique() / len(X[col].dropna())
                            if unique_ratio < min_unique_ratio:
                                low_variance_cols.append(col)
                        except Exception as e:
                            logger.warning(f"Erreur calcul variance {col}: {e}")
                            continue
                
                self.cols_to_drop_.extend(low_variance_cols)
                if low_variance_cols:
                    logger.info(f"Colonnes à faible variance (<{min_unique_ratio:.1%}): {low_variance_cols}")
            
            # Déduplication et validation finale
            self.cols_to_drop_ = list(set(self.cols_to_drop_))
            self.cols_to_keep_ = [col for col in X.columns if col not in self.cols_to_drop_]
            
            # Validation qu'il reste des colonnes
            if not self.cols_to_keep_:
                logger.error("Aucune colonne à conserver après nettoyage")
                raise ValueError("Aucune colonne valide après le nettoyage")
            
            self.fitted_ = True
            logger.info(f"Nettoyage terminé: {len(self.cols_to_drop_)} colonnes supprimées, {len(self.cols_to_keep_)} conservées")
            
        except Exception as e:
            logger.error(f"Erreur lors du fit de SafeColumnCleaner: {e}")
            # Fallback stratégique
            self.cols_to_keep_ = list(X.columns)
            self.cols_to_drop_ = []
            self.fitted_ = True
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applique la transformation de nettoyage.
        
        Args:
            X: DataFrame à transformer
            
        Returns:
            DataFrame nettoyé
        """
        if not self.fitted_:
            raise RuntimeError("Le transformateur n'a pas été ajusté. Appelez fit() d'abord.")
            
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X doit être un pandas.DataFrame")
            
            # Appliquer la suppression des colonnes
            X_clean = X.drop(columns=self.cols_to_drop_, errors='ignore')
            
            # Vérification de cohérence
            missing_columns = set(self.cols_to_keep_) - set(X_clean.columns)
            if missing_columns:
                logger.warning(f"Colonnes attendues manquantes: {missing_columns}")
            
            logger.debug(f"Transform: {X_clean.shape[1]} colonnes après nettoyage")
            return X_clean
            
        except Exception as e:
            logger.error(f"Erreur transformation SafeColumnCleaner: {e}")
            # Fallback: retourner les données originales
            return X
    
    def get_feature_names(self) -> List[str]:
        """Retourne la liste des noms de features conservés."""
        return self.cols_to_keep_.copy()

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Imputation intelligente adaptée aux types de données.
    Supporte l'imputation numérique et catégorielle avec différentes stratégies.
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent', 
                 k: int = 5,
                 numeric_constant_value: Any = 0):
        """
        Args:
            numeric_strategy: Stratégie d'imputation numérique ('mean', 'median', 'constant', 'knn')
            categorical_strategy: Stratégie d'imputation catégorielle ('most_frequent', 'constant')
            k: Nombre de voisins pour KNN
            numeric_constant_value: Valeur constante pour l'imputation numérique
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.k = k
        self.numeric_constant_value = numeric_constant_value
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y=None) -> 'SmartImputer':
        """
        Adapte l'imputation aux types de colonnes.
        
        Args:
            X: DataFrame d'entraînement
            y: Target (optionnel)
            
        Returns:
            self: Instance ajustée
        """
        try:
            # Identification robuste des types de colonnes
            self.numeric_cols_ = X.select_dtypes(
                include=[np.number, 'number']
            ).columns.tolist()
            
            self.categorical_cols_ = X.select_dtypes(
                include=['object', 'category', 'string']
            ).columns.tolist()
            
            logger.info(f"Colonnes numériques détectées: {len(self.numeric_cols_)}")
            logger.info(f"Colonnes catégorielles détectées: {len(self.categorical_cols_)}")
            
            # Configuration de l'imputation numérique
            if self.numeric_cols_:
                if self.numeric_strategy == 'knn':
                    self.numeric_imputer_ = KNNImputer(
                        n_neighbors=min(self.k, max(1, len(X) - 1))
                    )
                else:
                    fill_value = (self.numeric_constant_value 
                                if self.numeric_strategy == 'constant' 
                                else None)
                    self.numeric_imputer_ = SimpleImputer(
                        strategy=self.numeric_strategy,
                        fill_value=fill_value
                    )
                
                # Adapter seulement sur les colonnes avec des valeurs manquantes
                cols_with_missing = [
                    col for col in self.numeric_cols_ 
                    if X[col].isna().any()
                ]
                if cols_with_missing:
                    self.numeric_imputer_.fit(X[cols_with_missing])
            
            # Configuration de l'imputation catégorielle
            if self.categorical_cols_:
                fill_value = 'missing' if self.categorical_strategy == 'constant' else None
                self.categorical_imputer_ = SimpleImputer(
                    strategy=self.categorical_strategy,
                    fill_value=fill_value
                )
                
                # Adapter seulement sur les colonnes avec des valeurs manquantes
                cols_with_missing = [
                    col for col in self.categorical_cols_ 
                    if X[col].isna().any()
                ]
                if cols_with_missing:
                    self.categorical_imputer_.fit(X[cols_with_missing])
            
            self.fitted_ = True
            logger.info("Imputation intelligente ajustée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du fit de SmartImputer: {e}")
            # Fallback: imputeurs basiques
            self.numeric_imputer_ = SimpleImputer(strategy='mean')
            self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
            self.fitted_ = True
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l'imputation.
        
        Args:
            X: DataFrame à transformer
            
        Returns:
            DataFrame imputé
        """
        if not self.fitted_:
            raise RuntimeError("Le transformateur n'a pas été ajusté. Appelez fit() d'abord.")
            
        try:
            X_transformed = X.copy()
            
            # Imputation numérique
            if self.numeric_cols_ and self.numeric_imputer_ is not None:
                cols_with_missing = [
                    col for col in self.numeric_cols_ 
                    if col in X.columns and X[col].isna().any()
                ]
                if cols_with_missing:
                    numeric_imputed = self.numeric_imputer_.transform(X[cols_with_missing])
                    X_transformed[cols_with_missing] = numeric_imputed
                    logger.debug(f"Imputation numérique appliquée sur {len(cols_with_missing)} colonnes")
            
            # Imputation catégorielle
            if self.categorical_cols_ and self.categorical_imputer_ is not None:
                cols_with_missing = [
                    col for col in self.categorical_cols_ 
                    if col in X.columns and X[col].isna().any()
                ]
                if cols_with_missing:
                    categorical_imputed = self.categorical_imputer_.transform(X[cols_with_missing])
                    X_transformed[cols_with_missing] = categorical_imputed
                    logger.debug(f"Imputation catégorielle appliquée sur {len(cols_with_missing)} colonnes")
            
            # Vérification des valeurs manquantes restantes
            remaining_missing = X_transformed.isna().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"{remaining_missing} valeurs manquantes restantes après imputation")
            
            return X_transformed
            
        except Exception as e:
            logger.error(f"Erreur transformation SmartImputer: {e}")
            return X

def safe_label_encode(y: pd.Series) -> tuple:
    """
    Encodage sécurisé des labels avec gestion des erreurs robuste.
    
    Args:
        y: Série à encoder
        
    Returns:
        Tuple (y_encoded, label_encoder, encoding_map)
    """
    try:
        if y is None or len(y) == 0:
            raise ValueError("Série y vide ou None")
            
        # Vérification du type
        if pd.api.types.is_numeric_dtype(y):
            logger.info("Target déjà numérique, pas d'encodage nécessaire")
            return y.values, None, {}
        
        # Nettoyage des valeurs manquantes
        y_clean = y.dropna()
        if len(y_clean) == 0:
            raise ValueError("Aucune valeur valide dans la target après nettoyage")
        
        # Encodage des labels avec scikit-learn
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_clean)
        
        # Reconstruction avec les NaN préservés
        y_final = np.full(len(y), np.nan)
        y_final[y_clean.index] = y_encoded
        
        # Mapping pour référence
        encoding_map = {
            int(i): str(label) for i, label in enumerate(label_encoder.classes_)
        }
        
        logger.info(f"Encodage label terminé: {len(encoding_map)} classes")
        return y_final, label_encoder, encoding_map
        
    except Exception as e:
        logger.error(f"Erreur encodage labels: {e}")
        # Fallback: encodage manuel simple
        try:
            unique_labels = y.dropna().unique()
            encoding_map = {i: str(label) for i, label in enumerate(unique_labels)}
            reverse_map = {str(label): i for i, label in encoding_map.items()}
            y_encoded = y.map(reverse_map).values
            logger.warning("Encodage fallback manuel utilisé")
            return y_encoded, None, encoding_map
        except Exception as fallback_error:
            logger.error(f"Échec encodage fallback: {fallback_error}")
            raise

def create_preprocessor(preprocessing_config: Dict[str, Any], 
                       column_types: Dict[str, List[str]] = None) -> ColumnTransformer:
    """
    Crée un préprocesseur (ColumnTransformer) compatible avec les pipelines.
    Conforme aux standards MLOps et prêt pour la production.
    
    Args:
        preprocessing_config: Configuration du preprocessing
        column_types: Dictionnaire des types de colonnes
        
    Returns:
        ColumnTransformer configuré
    """
    try:
        # Configuration par défaut robuste
        default_config = {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'missing_threshold': 0.8,
            'scaling_method': 'standard',
            'encoding_method': 'onehot',
            'handle_unknown': 'ignore',
            'min_unique_ratio': 0.01
        }
        
        # Fusion des configurations
        config = {**default_config, **preprocessing_config}
        
        # Extraction des types de colonnes avec fallback
        numeric_features = column_types.get('numeric', []) if column_types else []
        categorical_features = column_types.get('categorical', []) if column_types else []
        text_features = column_types.get('text_or_high_cardinality', []) if column_types else []
        
        logger.info(f"Configuration préprocesseur: {len(numeric_features)} num, "
                   f"{len(categorical_features)} cat, {len(text_features)} text")
        
        # Validation des entrées
        if not any([numeric_features, categorical_features, text_features]):
            logger.warning("Aucune feature détectée - vérifier la détection des types de colonnes")
        
        transformers = []
        
        # Pipeline pour les features numériques
        if numeric_features:
            # Choix du scaler
            scaling_method = config.get('scaling_method', 'standard')
            if scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=config['numeric_imputation'])),
                ('scaler', scaler)
            ])
            
            transformers.append(('num', numeric_pipeline, numeric_features))
            logger.debug(f"Pipeline numérique créé pour {len(numeric_features)} features")
        
        # Pipeline pour les features catégorielles
        if categorical_features:
            # Choix de l'encodeur
            encoding_method = config.get('encoding_method', 'onehot')
            handle_unknown = config.get('handle_unknown', 'ignore')
            
            if encoding_method == 'ordinal':
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                )
            else:
                encoder = OneHotEncoder(
                    handle_unknown=handle_unknown, 
                    sparse_output=False,
                    drop='first' if config.get('drop_first', True) else None
                )
            
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(
                    strategy=config['categorical_imputation'], 
                    fill_value='missing'
                )),
                ('encoder', encoder)
            ])
            
            transformers.append(('cat', categorical_pipeline, categorical_features))
            logger.debug(f"Pipeline catégoriel créé pour {len(categorical_features)} features")
        
        # Pipeline pour les features texte/haute cardinalité
        if text_features:
            text_pipeline = Pipeline([
                ('imputer', SimpleImputer(
                    strategy=config['categorical_imputation'], 
                    fill_value='missing'
                )),
                ('encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                ))
            ])
            
            transformers.append(('text', text_pipeline, text_features))
            logger.debug(f"Pipeline texte créé pour {len(text_features)} features")
        
        # Création du ColumnTransformer
        if transformers:
            column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='drop',  # Supprimer les colonnes non traitées
                n_jobs=1,  # Éviter les problèmes de parallélisme en production
                verbose_feature_names_out=False
            )
            
            logger.info("✅ Preprocessor ColumnTransformer créé avec succès")
            return column_transformer
        else:
            logger.error("❌ Aucun transformateur créé - vérifier la configuration")
            raise ValueError("Aucun transformateur configuré")
            
    except Exception as e:
        logger.error(f"❌ Erreur création préprocesseur: {e}")
        # Fallback minimal pour éviter les crashes
        return ColumnTransformer([
            ('num_fallback', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), numeric_features if numeric_features else [])
        ], remainder='drop')

def validate_preprocessor(preprocessor: Pipeline, 
                         X_sample: pd.DataFrame,
                         y_sample: pd.Series = None) -> Dict[str, Any]:
    """
    Valide le fonctionnement d'un préprocesseur sur un échantillon.
    Tests complets pour la robustesse en production.
    
    Args:
        preprocessor: Pipeline à valider
        X_sample: Échantillon de données features
        y_sample: Échantillon de données target (optionnel)
        
    Returns:
        Rapport de validation détaillé
    """
    import time
    
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "transformed_shape": None,
        "execution_time": 0,
        "memory_usage_mb": 0,
        "feature_names_out": [],
        "data_quality_checks": {}
    }
    
    try:
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        # Transformation de l'échantillon
        if y_sample is not None:
            X_transformed = preprocessor.fit_transform(X_sample, y_sample)
        else:
            X_transformed = preprocessor.fit_transform(X_sample)
            
        validation["execution_time"] = time.time() - start_time
        validation["memory_usage_mb"] = _get_memory_usage() - start_memory
        
        validation["transformed_shape"] = X_transformed.shape
        
        # Récupération des noms de features
        try:
            if hasattr(preprocessor, 'get_feature_names_out'):
                validation["feature_names_out"] = preprocessor.get_feature_names_out().tolist()
        except Exception as e:
            validation["warnings"].append(f"Impossible de récupérer les noms de features: {e}")
        
        # Vérifications de qualité des données
        data_checks = {}
        
        # Vérification des NaN
        if hasattr(X_transformed, 'isna'):
            nan_count = np.isnan(X_transformed).sum() if hasattr(X_transformed, 'isna') else 0
        else:
            nan_count = np.sum(np.isnan(X_transformed))
            
        data_checks["nan_count"] = int(nan_count)
        if nan_count > 0:
            validation["warnings"].append(f"{nan_count} valeurs NaN détectées après transformation")
        
        # Vérification des infinis
        if hasattr(X_transformed, 'values'):
            inf_count = np.sum(np.isinf(X_transformed.values))
        else:
            inf_count = np.sum(np.isinf(X_transformed))
            
        data_checks["inf_count"] = int(inf_count)
        if inf_count > 0:
            validation["issues"].append(f"{inf_count} valeurs infinies détectées")
            validation["is_valid"] = False
        
        # Vérification de la variance
        try:
            if hasattr(X_transformed, 'var'):
                zero_variance_features = np.sum(X_transformed.var(axis=0) == 0)
            else:
                zero_variance_features = np.sum(np.var(X_transformed, axis=0) == 0)
                
            data_checks["zero_variance_features"] = int(zero_variance_features)
            if zero_variance_features > 0:
                validation["warnings"].append(f"{zero_variance_features} features avec variance nulle")
        except Exception as e:
            validation["warnings"].append(f"Impossible de calculer la variance: {e}")
        
        # Vérification des dimensions
        if X_transformed.shape[1] == 0:
            validation["issues"].append("Aucune feature après transformation")
            validation["is_valid"] = False
        
        validation["data_quality_checks"] = data_checks
        
        # Validation de performance
        if validation["execution_time"] > 30:
            validation["warnings"].append("Temps d'exécution élevé pour le preprocessing")
        
        if validation["memory_usage_mb"] > 1000:
            validation["warnings"].append("Utilisation mémoire élevée pour le preprocessing")
        
        logger.info(f"✅ Validation préprocesseur: {X_transformed.shape} "
                   f"en {validation['execution_time']:.2f}s, "
                   f"{validation['memory_usage_mb']:.1f}MB utilisés")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)}")
        logger.error(f"❌ Erreur validation préprocesseur: {e}")
    
    return validation

def save_preprocessor(preprocessor: Pipeline, filepath: str) -> bool:
    """
    Sauvegarde le préprocesseur pour la production.
    
    Args:
        preprocessor: Préprocesseur à sauvegarder
        filepath: Chemin de sauvegarde
        
    Returns:
        Succès de la sauvegarde
    """
    try:
        # Création du dossier si nécessaire
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarde avec pickle
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        logger.info(f"✅ Préprocesseur sauvegardé: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde préprocesseur: {e}")
        return False

def load_preprocessor(filepath: str) -> Optional[Pipeline]:
    """
    Charge un préprocesseur depuis le stockage.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Préprocesseur chargé ou None en cas d'erreur
    """
    try:
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"✅ Préprocesseur chargé: {filepath}")
        return preprocessor
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement préprocesseur: {e}")
        return None

def _get_memory_usage() -> float:
    """
    Obtient l'utilisation mémoire actuelle en MB.
    
    Returns:
        Mémoire utilisée en MB
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0

# Export des classes et fonctions principales
__all__ = [
    'SafeColumnCleaner',
    'SmartImputer', 
    'safe_label_encode',
    'create_preprocessor',
    'validate_preprocessor',
    'save_preprocessor', 
    'load_preprocessor'
]