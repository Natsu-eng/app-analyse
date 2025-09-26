import pandas as pd
import dask.dataframe as dd
import numpy as np
import logging
import time
from typing import Union, Tuple, Dict, Any, List, Optional
from scipy.stats import pointbiserialr, f_oneway
import streamlit as st
import psutil
import gc
from functools import wraps

logger = logging.getLogger(__name__)

# =============================
# Décorateurs de monitoring
# =============================

def monitor_performance(func):
    """Décorateur pour monitorer les performances des fonctions critiques"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"{func.__name__} - Duration: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            # Alertes pour performances dégradées
            if duration > 30:
                logger.warning(f"{func.__name__} took {duration:.2f}s - performance issue detected")
            if memory_delta > 500:
                logger.warning(f"{func.__name__} used {memory_delta:.1f}MB - memory issue detected")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
            
    return wrapper

def safe_execute(fallback_value=None, log_errors=True):
    """Décorateur pour l'exécution sécurisée avec fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Safe execution failed in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

# =============================
# Helpers
# =============================

def compute_if_dask(data: Any) -> Any:
    """
    Exécute .compute() si l'objet est un DataFrame, Series ou Scalar Dask.
    
    Args:
        data: Objet à évaluer (DataFrame, Series ou Scalar)
    
    Returns:
        Objet calculé (si Dask) ou inchangé (si Pandas)
    """
    if isinstance(data, (dd.DataFrame, dd.Series, dd.Scalar)):
        start = time.time()
        try:
            result = data.compute()
            elapsed = time.time() - start
            logger.info(f"Dask compute() terminé en {elapsed:.2f} sec")
            return result
        except Exception as e:
            logger.error(f"Dask compute() failed: {e}")
            raise
    return data

def is_dask_dataframe(df: Any) -> bool:
    """
    Vérifie si l'objet est un DataFrame Dask.
    
    Args:
        df: Objet à vérifier
    
    Returns:
        Booléen indiquant si c'est un DataFrame Dask
    """
    return isinstance(df, dd.DataFrame)

def optimize_dataframe(df: pd.DataFrame, memory_threshold_mb: float = 100.0) -> pd.DataFrame:
    """
    Optimise un DataFrame Pandas en convertissant les colonnes object en category.
    Version améliorée avec seuil de mémoire.
    
    Args:
        df: DataFrame Pandas
        memory_threshold_mb: Seuil de mémoire en MB pour déclencher l'optimisation
    
    Returns:
        DataFrame optimisé
    """
    if df.empty:
        return df
        
    # Vérifier l'utilisation mémoire actuelle
    current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    if current_memory < memory_threshold_mb:
        return df  # Pas d'optimisation nécessaire
        
    logger.info(f"Optimisation du DataFrame ({current_memory:.1f}MB)")
    
    try:
        df_copy = df.copy()
        optimized_columns = 0
        
        for col in df_copy.select_dtypes(include="object").columns:
            try:
                unique_ratio = df_copy[col].nunique() / len(df_copy[col].dropna())
                if unique_ratio < 0.5 and df_copy[col].nunique() < 10000:
                    df_copy[col] = df_copy[col].astype("category")
                    optimized_columns += 1
            except Exception as e:
                logger.debug(f"Failed to optimize column {col}: {e}")
                continue
                
        if optimized_columns > 0:
            new_memory = df_copy.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = current_memory - new_memory
            logger.info(f"DataFrame optimized: {optimized_columns} columns, {memory_saved:.1f}MB saved")
            
        return df_copy
        
    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}")
        return df

def safe_sample(df: Union[pd.DataFrame, dd.DataFrame], 
            sample_frac: float = 0.01, 
            max_rows: int = 10000,
            min_rows: int = 100) -> pd.DataFrame:
    """
    Échantillonnage sécurisé d'un DataFrame avec gestion d'erreurs.
    
    Args:
        df: DataFrame à échantillonner
        sample_frac: Fraction d'échantillonnage
        max_rows: Nombre maximum de lignes
        min_rows: Nombre minimum de lignes requises
    
    Returns:
        DataFrame échantillonné
    """
    try:
        is_dask = is_dask_dataframe(df)
        n_rows = len(df) if not is_dask else compute_if_dask(df.shape[0])
        
        if n_rows < min_rows:
            logger.warning(f"Dataset too small ({n_rows} rows), using entire dataset")
            return compute_if_dask(df) if is_dask else df
            
        # Calcul de la taille d'échantillon optimale
        target_size = min(max_rows, max(min_rows, int(n_rows * sample_frac)))
        
        if target_size >= n_rows:
            sample_df = df
        else:
            if is_dask:
                # Pour Dask, utiliser un échantillonnage par fraction
                actual_frac = min(0.1, target_size / n_rows)
                sample_df = df.sample(frac=actual_frac).head(target_size)
            else:
                # Pour Pandas, échantillonnage direct
                sample_df = df.sample(n=target_size, random_state=42)
                
        result_df = compute_if_dask(sample_df)
        result_df = optimize_dataframe(result_df)
        
        logger.debug(f"Sampled {len(result_df)} rows from {n_rows} total")
        return result_df
        
    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        # Fallback: retourner les premières lignes
        try:
            fallback_df = compute_if_dask(df.head(min_rows))
            return optimize_dataframe(fallback_df)
        except:
            logger.error("Complete sampling fallback failed")
            raise

# =============================
# Fonctions principales
# =============================

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []})
@monitor_performance
def auto_detect_column_types(
    df: Union[pd.DataFrame, dd.DataFrame], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, List[str]]:
    """
    Détecte automatiquement les types de colonnes (numérique, catégorielle, datetime, texte).
    Version robuste avec gestion d'erreurs améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les listes de colonnes par type
    """
    try:
        if df.empty or len(df.columns) == 0:
            logger.warning("DataFrame empty or no columns")
            return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}
        
        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            logger.warning("Sample DataFrame is empty")
            return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}

        # Initialisation du résultat
        result = {
            "numeric": [],
            "datetime": [],
            "categorical": [],
            "text_or_high_cardinality": []
        }

        # Détection des colonnes numériques (types natifs)
        try:
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            result["numeric"] = [col for col in numeric_cols if col in df.columns]
            logger.debug(f"Detected {len(result['numeric'])} numeric columns")
        except Exception as e:
            logger.warning(f"Numeric column detection failed: {e}")

        # Détection des colonnes datetime (types natifs)
        try:
            datetime_cols = sample_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
            result["datetime"] = [col for col in datetime_cols if col in df.columns]
            logger.debug(f"Detected {len(result['datetime'])} datetime columns")
        except Exception as e:
            logger.warning(f"Datetime column detection failed: {e}")

        # Analyse des colonnes object/category
        try:
            object_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
            
            for col in object_cols:
                try:
                    if col not in df.columns:
                        continue
                        
                    col_series = sample_df[col].dropna()
                    if len(col_series) == 0:
                        result["text_or_high_cardinality"].append(col)
                        continue
                        
                    unique_count = col_series.nunique()
                    total_count = len(col_series)
                    unique_ratio = unique_count / total_count if total_count > 0 else 1
                    
                    # Logique de classification améliorée
                    if unique_ratio < 0.5 and unique_count < 100:
                        result["categorical"].append(col)
                    else:
                        result["text_or_high_cardinality"].append(col)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing column {col}: {e}")
                    result["text_or_high_cardinality"].append(col)
                    
            logger.debug(f"Detected {len(result['categorical'])} categorical columns")
            logger.debug(f"Detected {len(result['text_or_high_cardinality'])} text/high cardinality columns")
            
        except Exception as e:
            logger.warning(f"Object column analysis failed: {e}")

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        total_detected = sum(len(cols) for cols in result.values())
        logger.info(f"Column type detection completed: {total_detected} columns classified")
        return result

    except Exception as e:
        logger.error(f"Critical error in auto_detect_column_types: {e}", exc_info=True)
        return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={})
@monitor_performance
def get_column_profile(
    series: Union[pd.Series, dd.Series], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, Any]:
    """
    Génère un profil statistique pour une colonne donnée.
    Version robuste avec gestion d'erreurs.
    
    Args:
        series: Série Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les statistiques de la colonne
    """
    try:
        if series.empty:
            return {"count": 0, "missing_values": 0, "missing_percentage": "100.00%"}
        
        # Échantillonnage pour les gros datasets
        is_dask = isinstance(series, dd.Series)
        n_rows = len(series) if not is_dask else compute_if_dask(series.shape[0])
        
        if is_dask or n_rows > max_rows:
            sample_size = min(max_rows, max(100, int(n_rows * sample_frac)))
            if is_dask:
                actual_frac = min(0.1, sample_size / n_rows)
                sample_series = series.sample(frac=actual_frac).head(sample_size)
            else:
                sample_series = series.sample(n=sample_size, random_state=42)
        else:
            sample_series = series
            
        sample_series = compute_if_dask(sample_series)

        # Statistiques de base
        total_count = len(sample_series)
        valid_count = sample_series.count()
        missing_count = total_count - valid_count
        missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0

        profile = {
            "count": valid_count,
            "missing_values": missing_count,
            "missing_percentage": f"{missing_percentage:.2f}%",
            "total_rows_analyzed": total_count
        }

        # Statistiques spécifiques au type
        if valid_count > 0:
            try:
                if pd.api.types.is_numeric_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    profile.update({
                        "mean": float(valid_series.mean()),
                        "std_dev": float(valid_series.std()),
                        "min": float(valid_series.min()),
                        "25%": float(valid_series.quantile(0.25)),
                        "median": float(valid_series.median()),
                        "75%": float(valid_series.quantile(0.75)),
                        "max": float(valid_series.max()),
                        "skewness": float(valid_series.skew()) if len(valid_series) > 1 else 0.0
                    })
                elif pd.api.types.is_datetime64_any_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    profile.update({
                        "min_date": str(valid_series.min()),
                        "max_date": str(valid_series.max()),
                        "unique_dates": int(valid_series.nunique())
                    })
                else:
                    # Colonnes catégorielles ou texte
                    valid_series = sample_series.dropna()
                    unique_count = valid_series.nunique()
                    profile.update({
                        "unique_values": int(unique_count),
                        "unique_ratio": unique_count / len(valid_series) if len(valid_series) > 0 else 0
                    })
                    
                    # Top valeurs pour les colonnes catégorielles
                    if unique_count <= 20:
                        try:
                            top_values = valid_series.value_counts().head(10).to_dict()
                            profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                        except Exception as e:
                            logger.debug(f"Failed to compute top values: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to compute advanced statistics for {series.name}: {e}")
                profile["computation_error"] = str(e)

        logger.debug(f"Profile computed for column {series.name}: {profile.get('count', 0)} valid values")
        return profile

    except Exception as e:
        logger.error(f"Critical error in get_column_profile for {getattr(series, 'name', 'unknown')}: {e}")
        return {"error": str(e), "count": 0, "missing_values": 0, "missing_percentage": "100.00%"}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={})
@monitor_performance
def get_data_profile(
    df: Union[pd.DataFrame, dd.DataFrame], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, Dict[str, Any]]:
    """
    Génère un profil global du dataset par colonne.
    Version optimisée avec traitement par batch.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les profils par colonne
    """
    try:
        if df.empty or len(df.columns) == 0:
            logger.warning("DataFrame is empty or has no columns")
            return {}

        profiles = {}
        total_columns = len(df.columns)
        
        logger.info(f"Computing data profile for {total_columns} columns")
        
        # Traitement par batch pour éviter la surcharge mémoire
        batch_size = 10
        
        for i in range(0, total_columns, batch_size):
            batch_cols = df.columns[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(total_columns + batch_size - 1)//batch_size}")
            
            for col in batch_cols:
                try:
                    profiles[col] = get_column_profile(df[col], sample_frac, max_rows)
                except Exception as e:
                    logger.warning(f"Failed to profile column {col}: {e}")
                    profiles[col] = {"error": str(e), "count": 0}
                    
            # Nettoyage mémoire périodique
            if i % (batch_size * 3) == 0:
                gc.collect()

        logger.info(f"Data profile completed for {len(profiles)} columns")
        return profiles

    except Exception as e:
        logger.error(f"Critical error in get_data_profile: {e}")
        return {}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={"constant": [], "id_like": []})
@monitor_performance
def analyze_columns(
    df: Union[pd.DataFrame, dd.DataFrame], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, List[str]]:
    """
    Détecte les colonnes constantes ou de type ID.
    Version optimisée avec gestion d'erreurs.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les colonnes constantes et ID-like
    """
    try:
        if df.empty or len(df.columns) == 0:
            return {"constant": [], "id_like": []}

        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            return {"constant": [], "id_like": []}

        constant_cols = []
        id_like_cols = []
        
        try:
            nunique = sample_df.nunique()
            n_rows = len(sample_df)
            
            for col in sample_df.columns:
                try:
                    unique_count = nunique.get(col, 0)
                    
                    # Colonnes constantes
                    if unique_count <= 1:
                        constant_cols.append(col)
                        
                    # Colonnes de type ID (unique pour chaque ligne)
                    elif unique_count == n_rows and n_rows > 10:
                        id_like_cols.append(col)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing column {col}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Column analysis failed: {e}")
            return {"constant": [], "id_like": []}

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        logger.info(f"Column analysis completed: {len(constant_cols)} constant, {len(id_like_cols)} ID-like")
        return {"constant": constant_cols, "id_like": id_like_cols}

    except Exception as e:
        logger.error(f"Critical error in analyze_columns: {e}")
        return {"constant": [], "id_like": []}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={"is_imbalanced": False, "imbalance_ratio": 1.0, "message": "Error in detection"})
@monitor_performance
def detect_imbalance(
    df: Union[pd.DataFrame, dd.DataFrame], 
    target_column: str, 
    threshold: float = 0.8,
    sample_frac: float = 0.1,
    max_rows: int = 10000
) -> Dict[str, Any]:
    """
    Détecte le déséquilibre des classes pour les problèmes de classification.
    Version robuste avec gestion d'erreurs.
    
    Args:
        df: DataFrame Pandas ou Dask
        target_column: Nom de la colonne cible
        threshold: Seuil de déséquilibre (0.8 = 80% dans une classe)
        sample_frac: Fraction d'échantillonnage
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les résultats de détection
    """
    try:
        # Validation des entrées
        if target_column not in df.columns:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": f"Colonne cible '{target_column}' non trouvée",
                "class_distribution": {}
            }
        
        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Échantillon vide",
                "class_distribution": {}
            }
        
        target_series = sample_df[target_column].dropna()
        
        if len(target_series) == 0:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Aucune valeur valide dans la colonne cible",
                "class_distribution": {}
            }
        
        # Calcul de la distribution des classes
        class_distribution = target_series.value_counts().to_dict()
        total_samples = len(target_series)
        
        if len(class_distribution) <= 1:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Une seule classe détectée",
                "class_distribution": class_distribution,
                "total_classes": len(class_distribution)
            }
        
        # Calcul du ratio de déséquilibre
        majority_class_count = max(class_distribution.values())
        imbalance_ratio = majority_class_count / total_samples
        
        # Détection du déséquilibre
        is_imbalanced = imbalance_ratio > threshold
        
        # Calcul d'metrics supplémentaires
        minority_class_count = min(class_distribution.values())
        balance_ratio = minority_class_count / majority_class_count if majority_class_count > 0 else 0
        
        result = {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": float(imbalance_ratio),
            "balance_ratio": float(balance_ratio),
            "threshold_used": float(threshold),
            "total_samples": total_samples,
            "total_classes": len(class_distribution),
            "class_distribution": class_distribution,
            "majority_class": {
                "class": max(class_distribution, key=class_distribution.get),
                "count": majority_class_count,
                "percentage": float(majority_class_count / total_samples * 100)
            },
            "minority_class": {
                "class": min(class_distribution, key=class_distribution.get),
                "count": minority_class_count,
                "percentage": float(minority_class_count / total_samples * 100)
            }
        }
        
        # Messages explicatifs
        if is_imbalanced:
            result["message"] = f"Déséquilibre détecté : {result['majority_class']['percentage']:.1f}% dans la classe majoritaire"
            result["recommendation"] = "Envisagez d'activer SMOTE ou d'utiliser l'échantillonnage"
        else:
            result["message"] = "Classes équilibrées"
            result["recommendation"] = "Aucune action nécessaire"
        
        logger.info(f"Analyse déséquilibre terminée : {result['message']}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur dans detect_imbalance : {e}")
        return {
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "message": f"Erreur d'analyse : {str(e)}",
            "class_distribution": {}
        }

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={"target_type": "unknown", "task": "unknown"})
def get_target_and_task(
    df: Union[pd.DataFrame, dd.DataFrame],
    target: Optional[str]
) -> Dict[str, str]:
    """
    Détecte le type de tâche ML selon la colonne cible.
    Version robuste avec validation.
    
    Args:
        df: DataFrame Pandas ou Dask
        target: Nom de la colonne cible ou None pour non supervisé
    
    Returns:
        Dictionnaire avec le type de cible et la tâche ML
    """
    try:
        # Cas non supervisé
        if target is None:
            return {"target_type": "unsupervised", "task": "unsupervised"}

        if target not in df.columns:
            logger.warning(f"Target column '{target}' not found in DataFrame")
            return {"target_type": "unknown", "task": "unknown"}
            
        # Échantillonnage pour l'analyse
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=20000)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"target_type": "unknown", "task": "unknown"}
            
        target_series = sample_df[target].dropna()
        
        if len(target_series) == 0:
            logger.warning(f"Target column '{target}' has no valid values")
            return {"target_type": "unknown", "task": "unknown"}
            
        unique_vals = target_series.nunique()
        total_vals = len(target_series)

        # Logique de détection améliorée
        if pd.api.types.is_numeric_dtype(target_series):
            # Pour les variables numériques
            unique_ratio = unique_vals / total_vals
            
            if unique_vals <= 20 or unique_ratio < 0.05:
                # Peu de valeurs uniques -> classification
                return {"target_type": "classification", "task": "classification"}
            else:
                # Beaucoup de valeurs uniques -> régression
                return {"target_type": "regression", "task": "regression"}
        else:
            # Pour les variables non-numériques -> toujours classification
            return {"target_type": "classification", "task": "classification"}
            
    except Exception as e:
        logger.error(f"Error in get_target_and_task for target '{target}': {e}")
        return {"target_type": "unknown", "task": "unknown"}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value={"numeric": [], "categorical": []})
@monitor_performance
def get_relevant_features(
    df: Union[pd.DataFrame, dd.DataFrame],
    target: str,
    sample_frac: float = 0.05,
    max_rows: int = 20000
) -> Dict[str, List[str]]:
    """
    Sélectionne les features pertinentes via corrélation/ANOVA.
    Version robuste avec validation améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        target: Nom de la colonne cible
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
    
    Returns:
        Dictionnaire avec les features numériques et catégorielles pertinentes
    """
    try:
        if target not in df.columns:
            logger.warning(f"Target column '{target}' not found")
            return {"numeric": [], "categorical": []}

        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"numeric": [], "categorical": []}

        target_series = sample_df[target].dropna()
        
        if len(target_series) < 10:  # Minimum pour les tests statistiques
            logger.warning(f"Insufficient data for feature relevance analysis ({len(target_series)} valid target values)")
            return {"numeric": [], "categorical": []}

        features = {"numeric": [], "categorical": []}
        
        # Aligner les indices pour éviter les erreurs de correspondance
        valid_indices = target_series.index
        sample_df_aligned = sample_df.loc[valid_indices]

        # Analyse des features numériques
        numeric_cols = sample_df_aligned.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target]
        
        for col in numeric_cols:
            try:
                feature_series = sample_df_aligned[col].fillna(0)  # Imputation simple
                
                if feature_series.nunique() > 1:  # Éviter les colonnes constantes
                    try:
                        # Corrélation de Pearson pour variables continues
                        if pd.api.types.is_numeric_dtype(target_series):
                            corr_coef = np.corrcoef(target_series, feature_series)[0, 1]
                            if not np.isnan(corr_coef) and abs(corr_coef) > 0.1:
                                features["numeric"].append(col)
                        else:
                            # Point-biserial pour target catégorielle
                            corr, p_val = pointbiserialr(target_series.astype('category').cat.codes, 
                                                    feature_series)
                            if not np.isnan(corr) and abs(corr) > 0.1 and p_val < 0.05:
                                features["numeric"].append(col)
                                
                    except Exception as e:
                        logger.debug(f"Correlation analysis failed for {col}: {e}")
                        
            except Exception as e:
                logger.debug(f"Error processing numeric feature {col}: {e}")
                continue

        # Analyse des features catégorielles
        categorical_cols = sample_df_aligned.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [col for col in categorical_cols if col != target]
        
        for col in categorical_cols:
            try:
                feature_series = sample_df_aligned[col].dropna()
                
                if len(feature_series) < 10 or feature_series.nunique() <= 1:
                    continue
                    
                # Éviter les colonnes avec trop de catégories
                if feature_series.nunique() > min(50, len(feature_series) * 0.5):
                    continue
                
                try:
                    # ANOVA pour tester la différence de moyennes entre groupes
                    groups = []
                    for value in feature_series.unique():
                        if pd.notna(value):
                            group_data = target_series[sample_df_aligned[col] == value].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                    
                    if len(groups) > 1 and all(len(g) > 0 for g in groups):
                        # Vérifier que les groupes ont des données numériques pour ANOVA
                        if pd.api.types.is_numeric_dtype(target_series):
                            _, p_val = f_oneway(*groups)
                            if not np.isnan(p_val) and p_val < 0.05:
                                features["categorical"].append(col)
                        else:
                            # Pour target catégorielle, utiliser chi2 ou autre test
                            # Pour simplifier, on garde si plus de 2 groupes avec données suffisantes
                            if len(groups) >= 2:
                                features["categorical"].append(col)
                                
                except Exception as e:
                    logger.debug(f"ANOVA failed for {col}: {e}")
                    
            except Exception as e:
                logger.debug(f"Error processing categorical feature {col}: {e}")
                continue

        # Nettoyage mémoire
        del sample_df, sample_df_aligned
        gc.collect()

        total_features = len(features["numeric"]) + len(features["categorical"])
        logger.info(f"Feature relevance analysis completed: {total_features} relevant features found")
        
        return features

    except Exception as e:
        logger.error(f"Critical error in get_relevant_features: {e}")
        return {"numeric": [], "categorical": []}

@st.cache_data(ttl=300, show_spinner=False)
@safe_execute(fallback_value=[])
@monitor_performance
def detect_useless_columns(
    df: Union[pd.DataFrame, dd.DataFrame],
    threshold_missing: float = 0.6,
    min_unique_ratio: float = 0.0001,   # variabilité minimale acceptée
    max_sample_size: int = 50000
) -> List[str]:
    """
    Détecte les colonnes inutiles :
    - Trop de valeurs manquantes
    - Constantes ou quasi-constantes
    Optimisé pour production.
    """
    try:
        if df.empty or len(df.columns) == 0:
            logger.warning("DataFrame is empty or has no columns")
            return []

        # --- Échantillonnage unique ---
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=max_sample_size, min_rows=200)
        if sample_df.empty:
            return []

        useless_columns = []

        # --- 1. Colonnes avec trop de NaN ---
        try:
            missing_ratios = sample_df.isna().mean()
            high_missing = missing_ratios[missing_ratios > threshold_missing].index.tolist()
            useless_columns.extend(high_missing)
        except Exception as e:
            logger.warning(f"Missing ratio check failed: {e}")

        # --- 2. Colonnes constantes ou quasi-constantes ---
        try:
            nunique_counts = sample_df.nunique(dropna=True)
            constant_cols = nunique_counts[nunique_counts <= 1].index.tolist()
            useless_columns.extend(constant_cols)
        except Exception as e:
            logger.warning(f"Constant value check failed: {e}")

        # --- 3. Colonnes numériques avec variance quasi nulle ---
        try:
            numeric_cols = sample_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                desc = sample_df[numeric_cols].describe().T  # mean, std dispo
                low_var_cols = desc.index[
                    (desc["std"].fillna(0) == 0) |
                    ((desc["mean"].replace(0, np.nan)).notna() &
                     (abs(desc["std"] / desc["mean"].replace(0, np.nan)) < min_unique_ratio))
                ].tolist()
                useless_columns.extend(low_var_cols)
        except Exception as e:
            logger.warning(f"Low variance check failed: {e}")

        # Nettoyage final
        useless_columns = list(set(useless_columns))
        valid_useless_columns = [col for col in useless_columns if col in df.columns]

        logger.info(f"Useless column detection completed: {len(valid_useless_columns)} columns")
        return valid_useless_columns

    except Exception as e:
        logger.error(f"Critical error in detect_useless_columns: {e}")
        return []

# =============================
# Fonctions de monitoring système
# =============================

def get_system_health() -> Dict[str, Any]:
    """
    Retourne des métriques de santé du système.
    
    Returns:
        Dictionnaire avec les métriques système
    """
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {"error": str(e), "timestamp": time.time()}

def cleanup_memory():
    """
    Force le nettoyage mémoire et du cache Streamlit.
    """
    try:
        # Nettoyage du garbage collector Python
        collected = gc.collect()
        
        # Nettoyage du cache Streamlit si disponible
        if hasattr(st, 'cache_data') and hasattr(st.cache_data, 'clear'):
            st.cache_data.clear()
            
        logger.info(f"Memory cleanup completed: {collected} objects collected")
        return collected
        
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return 0