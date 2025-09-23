import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, config: Dict):
        self.config = config
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_to_drop_ = []
        if self.config.get('remove_identifier_cols', False):
            self.cols_to_drop_.extend([col for col in X.columns if X[col].nunique() == len(X)])
        if self.config.get('remove_constant_cols', False):
            self.cols_to_drop_.extend([col for col in X.columns if X[col].nunique() == 1])
        self.cols_to_drop_ = list(set(self.cols_to_drop_))
        logger.info(f"Colonnes à supprimer identifiées : {self.cols_to_drop_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un pandas.DataFrame dans ColumnCleaner.transform")
        result = X.drop(columns=self.cols_to_drop_, errors='ignore')
        logger.info(f"Après nettoyage : colonnes restantes = {result.columns.tolist()}")
        return result

def create_preprocessor(preprocessing_config: Dict) -> Pipeline:
    """
    Construit une pipeline avec ColumnCleaner avant ColumnTransformer pour gérer tout type de dataset.
    """
    # Étape 1 : Nettoyage des colonnes
    column_cleaner = ColumnCleaner(preprocessing_config)
    
    # Étapes de transformation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=preprocessing_config.get('numeric_imputation', 'mean'))),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=preprocessing_config.get('categorical_imputation', 'most_frequent'))),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=object))
        ],
        remainder='passthrough'
    )

    # Pipeline complète : Cleaner -> Transformer
    preprocessor_pipeline = Pipeline(steps=[
        ('cleaner', column_cleaner),
        ('transformer', column_transformer)
    ])

    return preprocessor_pipeline