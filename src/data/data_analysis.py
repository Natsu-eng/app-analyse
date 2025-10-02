"""
Module d'analyse de données robuste pour le machine learning.
Optimisé pour la production avec gestion mémoire avancée et monitoring.
"""

import pandas as pd
import numpy as np
import time

from typing import Union, Tuple, Dict, Any, List, Optional
import gc
from functools import wraps
import os

# Configuration du logging
from src.shared.logging import get_logger
logger = get_logger(__name__)

# Tentative d'import de dépendances optionnelles
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask non disponible, utilisation de Pandas uniquement")

try:
    from scipy.stats import pointbiserialr, f_oneway
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy non disponible, certaines analyses statistiques limitées")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible, monitoring mémoire limité")

# =============================
# Décorateurs de monitoring
# =============================

def monitor_performance(func):
    """Décorateur pour monitorer les performances des fonctions critiques"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PSUTIL_AVAILABLE:
            return func(*args, **kwargs)
            
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.debug(f"{func.__name__} - Duration: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            # Alertes pour performances dégradées
            if duration > 30:
                logger.warning(f"⏰ {func.__name__} took {duration:.2f}s - performance issue")
            if memory_delta > 500:
                logger.warning(f"💾 {func.__name__} used {memory_delta:.1f}MB - memory issue")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {str(e)}", exc_info=True)
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
                    logger.error(f"❌ Safe execution failed in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

def conditional_cache(use_cache=True):
    """Décorateur de cache conditionnel pour Streamlit"""
    def decorator(func):
        if STREAMLIT_AVAILABLE and use_cache:
            return st.cache_data(ttl=300, show_spinner=False)(func)
        return func
    return decorator

# =============================
# Helpers
# =============================

def is_dask_dataframe(df: Any) -> bool:
    """
    Vérifie si l'objet est un DataFrame Dask.
    
    Args:
        df: Objet à vérifier
        
    Returns:
        Booléen indiquant si c'est un DataFrame Dask
    """
    return DASK_AVAILABLE and isinstance(df, dd.DataFrame)

def compute_if_dask(data: Any) -> Any:
    """
    Exécute .compute() si l'objet est un DataFrame, Series ou Scalar Dask.
    
    Args:
        data: Objet à évaluer (DataFrame, Series ou Scalar)
        
    Returns:
        Objet calculé (si Dask) ou inchangé (si Pandas)
    """
    if is_dask_dataframe(data) or (DASK_AVAILABLE and isinstance(data, (dd.Series, dd.Scalar))):
        start = time.time()
        try:
            result = data.compute()
            elapsed = time.time() - start
            logger.debug(f"Dask compute() terminé en {elapsed:.2f} sec")
            return result
        except Exception as e:
            logger.error(f"❌ Dask compute() failed: {e}")
            raise
    return data

def optimize_dataframe(df: pd.DataFrame, 
                      memory_threshold_mb: float = 100.0,
                      downcast_int: bool = True,
                      downcast_float: bool = True) -> pd.DataFrame:
    """
    Optimise un DataFrame Pandas en réduisant l'utilisation mémoire.
    Version améliorée avec seuil de mémoire et downcasting.
    
    Args:
        df: DataFrame Pandas
        memory_threshold_mb: Seuil de mémoire pour déclencher l'optimisation
        downcast_int: Downcaster les entiers
        downcast_float: Downcaster les floats
        
    Returns:
        DataFrame optimisé
    """
    if df.empty:
        return df
        
    try:
        # Vérifier l'utilisation mémoire actuelle
        current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if current_memory < memory_threshold_mb:
            return df  # Pas d'optimisation nécessaire
            
        logger.info(f"🔧 Optimisation du DataFrame ({current_memory:.1f}MB)")
        
        df_copy = df.copy()
        memory_saved = 0
        
        # Downcasting numérique
        if downcast_int or downcast_float:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_type = df_copy[col].dtype
                
                # Downcast entiers
                if downcast_int and np.issubdtype(col_type, np.integer):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
                
                # Downcast floats
                elif downcast_float and np.issubdtype(col_type, np.floating):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
        
        # Conversion object en category
        object_cols = df_copy.select_dtypes(include=['object']).columns
        optimized_columns = 0
        
        for col in object_cols:
            try:
                unique_ratio = df_copy[col].nunique() / len(df_copy[col].dropna())
                if unique_ratio < 0.5 and df_copy[col].nunique() < 10000:
                    df_copy[col] = df_copy[col].astype("category")
                    optimized_columns += 1
            except Exception as e:
                logger.debug(f"Failed to optimize column {col}: {e}")
                continue
        
        # Calcul mémoire économisée
        if optimized_columns > 0:
            new_memory = df_copy.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = current_memory - new_memory
            logger.info(f"✅ DataFrame optimisé: {optimized_columns} colonnes, {memory_saved:.1f}MB économisés")
            
        return df_copy
        
    except Exception as e:
        logger.warning(f"⚠️ DataFrame optimization failed: {e}")
        return df

def safe_sample(df: Union[pd.DataFrame, 'dd.DataFrame'], 
                sample_frac: float = 0.01, 
                max_rows: int = 10000,
                min_rows: int = 100,
                random_state: int = 42) -> pd.DataFrame:
    """
    Échantillonnage sécurisé d'un DataFrame avec gestion d'erreurs.
    
    Args:
        df: DataFrame à échantillonner
        sample_frac: Fraction d'échantillonnage
        max_rows: Nombre maximum de lignes
        min_rows: Nombre minimum de lignes requises
        random_state: Seed pour la reproductibilité
        
    Returns:
        DataFrame échantillonné
    """
    try:
        is_dask = is_dask_dataframe(df)
        n_rows = len(df) if not is_dask else compute_if_dask(df.shape[0])
        
        if n_rows < min_rows:
            logger.warning(f"⚠️ Dataset trop petit ({n_rows} rows), utilisation complète")
            return compute_if_dask(df) if is_dask else df
            
        # Calcul de la taille d'échantillon optimale
        target_size = min(max_rows, max(min_rows, int(n_rows * sample_frac)))
        
        if target_size >= n_rows:
            sample_df = df
        else:
            if is_dask:
                # Pour Dask, utiliser un échantillonnage par fraction
                actual_frac = min(0.1, target_size / n_rows)
                sample_df = df.sample(frac=actual_frac, random_state=random_state)
                sample_df = sample_df.head(target_size, compute=False)
            else:
                # Pour Pandas, échantillonnage direct
                sample_df = df.sample(n=target_size, random_state=random_state)
                
        result_df = compute_if_dask(sample_df)
        result_df = optimize_dataframe(result_df)
        
        logger.debug(f"📊 Échantillonnage: {len(result_df)} lignes sur {n_rows} total")
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Sampling failed: {e}")
        # Fallback: retourner les premières lignes
        try:
            fallback_size = min(min_rows, len(df) if not is_dask else compute_if_dask(df.shape[0]))
            fallback_df = compute_if_dask(df.head(fallback_size))
            return optimize_dataframe(fallback_df)
        except Exception as fallback_error:
            logger.error(f"❌ Complete sampling fallback failed: {fallback_error}")
            raise

# =============================
# Fonctions principales
# =============================

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []})
@monitor_performance
def auto_detect_column_types(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000,
    high_cardinality_threshold: int = 100
) -> Dict[str, List[str]]:
    """
    Détecte automatiquement les types de colonnes (numérique, catégorielle, datetime, texte).
    Version robuste avec gestion d'erreurs améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        high_cardinality_threshold: Seuil pour la cardinalité élevée
        
    Returns:
        Dictionnaire avec les listes de colonnes par type
    """
    try:
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}
        
        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            logger.warning("⚠️ Échantillon DataFrame vide")
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
            logger.debug(f"🔢 Colonnes numériques détectées: {len(result['numeric'])}")
        except Exception as e:
            logger.warning(f"⚠️ Détection colonnes numériques échouée: {e}")

        # Détection des colonnes datetime (types natifs et conversion)
        try:
            # Types datetime natifs
            datetime_cols = sample_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
            
            # Tentative de conversion des colonnes objet qui ressemblent à des dates
            object_cols = sample_df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                if col not in datetime_cols and col not in result["numeric"]:
                    try:
                        # Essayer de convertir en datetime
                        converted = pd.to_datetime(sample_df[col], errors='coerce')
                        if converted.notna().mean() > 0.8:  # Si >80% de conversion réussie
                            datetime_cols.append(col)
                    except:
                        pass
            
            result["datetime"] = [col for col in datetime_cols if col in df.columns]
            logger.debug(f"📅 Colonnes datetime détectées: {len(result['datetime'])}")
        except Exception as e:
            logger.warning(f"⚠️ Détection colonnes datetime échouée: {e}")

        # Analyse des colonnes object/category
        try:
            object_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
            # Exclure les colonnes déjà classées comme datetime
            object_cols = [col for col in object_cols if col not in result["datetime"]]
            
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
                    if unique_ratio < 0.5 and unique_count <= high_cardinality_threshold:
                        result["categorical"].append(col)
                    else:
                        result["text_or_high_cardinality"].append(col)
                        
                except Exception as e:
                    logger.debug(f"❌ Erreur analyse colonne {col}: {e}")
                    result["text_or_high_cardinality"].append(col)
                    
            logger.debug(f"🏷️ Colonnes catégorielles détectées: {len(result['categorical'])}")
            logger.debug(f"📝 Colonnes texte/haute cardinalité: {len(result['text_or_high_cardinality'])}")
            
        except Exception as e:
            logger.warning(f"⚠️ Analyse colonnes object échouée: {e}")

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        total_detected = sum(len(cols) for cols in result.values())
        logger.info(f"✅ Détection types colonnes terminée: {total_detected} colonnes classifiées")
        return result

    except Exception as e:
        logger.error(f"❌ Erreur critique dans auto_detect_column_types: {e}", exc_info=True)
        return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def get_column_profile(
    series: Union[pd.Series, 'dd.Series'], 
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
        if series is None or len(series) == 0:
            return {"count": 0, "missing_values": 0, "missing_percentage": "100.00%"}
        
        # Échantillonnage pour les gros datasets
        is_dask = is_dask_dataframe(series) if hasattr(series, 'dtype') else False
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
            "total_rows_analyzed": total_count,
            "dtype": str(sample_series.dtype)
        }

        # Statistiques spécifiques au type
        if valid_count > 0:
            try:
                if pd.api.types.is_numeric_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    stats = {
                        "mean": float(valid_series.mean()),
                        "std_dev": float(valid_series.std()),
                        "min": float(valid_series.min()),
                        "25%": float(valid_series.quantile(0.25)),
                        "median": float(valid_series.median()),
                        "75%": float(valid_series.quantile(0.75)),
                        "max": float(valid_series.max()),
                    }
                    
                    # Skewness seulement si assez de données
                    if len(valid_series) > 1:
                        stats["skewness"] = float(valid_series.skew())
                    
                    profile.update(stats)
                    
                elif pd.api.types.is_datetime64_any_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    profile.update({
                        "min_date": str(valid_series.min()),
                        "max_date": str(valid_series.max()),
                        "unique_dates": int(valid_series.nunique()),
                        "date_range_days": (valid_series.max() - valid_series.min()).days
                    })
                else:
                    # Colonnes catégorielles ou texte
                    valid_series = sample_series.dropna()
                    unique_count = valid_series.nunique()
                    profile.update({
                        "unique_values": int(unique_count),
                        "unique_ratio": float(unique_count / len(valid_series)) if len(valid_series) > 0 else 0
                    })
                    
                    # Top valeurs pour les colonnes catégorielles
                    if unique_count <= 20:
                        try:
                            top_values = valid_series.value_counts().head(10).to_dict()
                            profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                        except Exception as e:
                            logger.debug(f"❌ Calcul top valeurs échoué: {e}")
                            
            except Exception as e:
                logger.warning(f"⚠️ Calcul statistiques avancées échoué pour {series.name}: {e}")
                profile["computation_error"] = str(e)

        logger.debug(f"✅ Profil calculé pour colonne {series.name}: {profile.get('count', 0)} valeurs valides")
        return profile

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_column_profile for {getattr(series, 'name', 'unknown')}: {e}")
        return {"error": str(e), "count": 0, "missing_values": 0, "missing_percentage": "100.00%"}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def get_data_profile(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
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
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return {}

        profiles = {}
        total_columns = len(df.columns)
        
        logger.info(f"📊 Calcul profil données pour {total_columns} colonnes")
        
        # Traitement par batch pour éviter la surcharge mémoire
        batch_size = min(10, total_columns)
        
        for i in range(0, total_columns, batch_size):
            batch_cols = df.columns[i:i + batch_size]
            logger.debug(f"🔧 Traitement batch {i//batch_size + 1}/{(total_columns + batch_size - 1)//batch_size}")
            
            for col in batch_cols:
                try:
                    profiles[col] = get_column_profile(df[col], sample_frac, max_rows)
                except Exception as e:
                    logger.warning(f"⚠️ Profilage colonne {col} échoué: {e}")
                    profiles[col] = {"error": str(e), "count": 0}
                    
            # Nettoyage mémoire périodique
            if i % (batch_size * 3) == 0:
                gc.collect()

        logger.info(f"✅ Profil données terminé pour {len(profiles)} colonnes")
        return profiles

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_data_profile: {e}")
        return {}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"constant": [], "id_like": []})
@monitor_performance
def analyze_columns(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
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
        if df is None or df.empty or len(df.columns) == 0:
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
                    logger.debug(f"❌ Erreur analyse colonne {col}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Analyse colonnes échouée: {e}")
            return {"constant": [], "id_like": []}

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        logger.info(f"✅ Analyse colonnes terminée: {len(constant_cols)} constantes, {len(id_like_cols)} ID-like")
        return {"constant": constant_cols, "id_like": id_like_cols}

    except Exception as e:
        logger.error(f"❌ Erreur critique dans analyze_columns: {e}")
        return {"constant": [], "id_like": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"is_imbalanced": False, "imbalance_ratio": 1.0, "message": "Error in detection"})
@monitor_performance
def detect_imbalance(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
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
        
        # Calcul de métriques supplémentaires
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
            result["message"] = f"🚨 Déséquilibre détecté : {result['majority_class']['percentage']:.1f}% dans la classe majoritaire"
            result["recommendation"] = "Envisagez d'activer SMOTE ou d'utiliser l'échantillonnage"
        else:
            result["message"] = "✅ Classes équilibrées"
            result["recommendation"] = "Aucune action nécessaire"
        
        logger.info(f"✅ Analyse déséquilibre terminée : {result['message']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur dans detect_imbalance : {e}")
        return {
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "message": f"Erreur d'analyse : {str(e)}",
            "class_distribution": {}
        }

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"target_type": "unknown", "task": "unknown"})
def get_target_and_task(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
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
            return {"target_type": "unsupervised", "task": "clustering"}

        if target not in df.columns:
            logger.warning(f"⚠️ Colonne cible '{target}' non trouvée dans le DataFrame")
            return {"target_type": "unknown", "task": "unknown"}
            
        # Échantillonnage pour l'analyse
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=20000)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"target_type": "unknown", "task": "unknown"}
            
        target_series = sample_df[target].dropna()
        
        if len(target_series) == 0:
            logger.warning(f"⚠️ Colonne cible '{target}' sans valeurs valides")
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
        logger.error(f"❌ Erreur dans get_target_and_task pour target '{target}': {e}")
        return {"target_type": "unknown", "task": "unknown"}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"numeric": [], "categorical": []})
@monitor_performance
def get_relevant_features(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
    target: str,
    sample_frac: float = 0.05,
    max_rows: int = 20000,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05
) -> Dict[str, List[str]]:
    """
    Sélectionne les features pertinentes via corrélation/ANOVA.
    Version robuste avec validation améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        target: Nom de la colonne cible
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        correlation_threshold: Seuil de corrélation minimum
        p_value_threshold: Seuil de p-value maximum
        
    Returns:
        Dictionnaire avec les features numériques et catégorielles pertinentes
    """
    try:
        if target not in df.columns:
            logger.warning(f"⚠️ Colonne cible '{target}' non trouvée")
            return {"numeric": [], "categorical": []}

        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"numeric": [], "categorical": []}

        target_series = sample_df[target].dropna()
        
        if len(target_series) < 10:  # Minimum pour les tests statistiques
            logger.warning(f"⚠️ Données insuffisantes pour analyse features ({len(target_series)} valeurs target valides)")
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
                            if not np.isnan(corr_coef) and abs(corr_coef) > correlation_threshold:
                                features["numeric"].append(col)
                        else:
                            # Point-biserial pour target catégorielle
                            if SCIPY_AVAILABLE:
                                corr, p_val = pointbiserialr(target_series.astype('category').cat.codes, 
                                                        feature_series)
                                if not np.isnan(corr) and abs(corr) > correlation_threshold and p_val < p_value_threshold:
                                    features["numeric"].append(col)
                            else:
                                # Fallback sans scipy
                                corr_coef = np.corrcoef(target_series.astype('category').cat.codes, feature_series)[0, 1]
                                if not np.isnan(corr_coef) and abs(corr_coef) > correlation_threshold:
                                    features["numeric"].append(col)
                                
                    except Exception as e:
                        logger.debug(f"❌ Analyse corrélation échouée pour {col}: {e}")
                        
            except Exception as e:
                logger.debug(f"❌ Erreur traitement feature numérique {col}: {e}")
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
                    if SCIPY_AVAILABLE and pd.api.types.is_numeric_dtype(target_series):
                        groups = []
                        for value in feature_series.unique():
                            if pd.notna(value):
                                group_data = target_series[sample_df_aligned[col] == value].dropna()
                                if len(group_data) > 0:
                                    groups.append(group_data)
                        
                        if len(groups) > 1 and all(len(g) > 0 for g in groups):
                            _, p_val = f_oneway(*groups)
                            if not np.isnan(p_val) and p_val < p_value_threshold:
                                features["categorical"].append(col)
                    else:
                        # Pour target catégorielle ou sans scipy, utiliser une heuristique simple
                        if feature_series.nunique() >= 2:
                            features["categorical"].append(col)
                                
                except Exception as e:
                    logger.debug(f"❌ ANOVA échouée pour {col}: {e}")
                    
            except Exception as e:
                logger.debug(f"❌ Erreur traitement feature catégorielle {col}: {e}")
                continue

        # Nettoyage mémoire
        del sample_df, sample_df_aligned
        gc.collect()

        total_features = len(features["numeric"]) + len(features["categorical"])
        logger.info(f"✅ Analyse pertinence features terminée: {total_features} features pertinentes trouvées")
        
        return features

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_relevant_features: {e}")
        return {"numeric": [], "categorical": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value=[])
@monitor_performance
def detect_useless_columns(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
    threshold_missing: float = 0.6,
    min_unique_ratio: float = 0.0001,
    max_sample_size: int = 50000
) -> List[str]:
    """
    Détecte les colonnes inutiles :
    - Trop de valeurs manquantes
    - Constantes ou quasi-constantes
    Optimisé pour production.
    """
    try:
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return []

        # Échantillonnage unique
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=max_sample_size, min_rows=200)
        if sample_df.empty:
            return []

        useless_columns = []

        # 1. Colonnes avec trop de NaN
        try:
            missing_ratios = sample_df.isna().mean()
            high_missing = missing_ratios[missing_ratios > threshold_missing].index.tolist()
            useless_columns.extend(high_missing)
        except Exception as e:
            logger.warning(f"⚠️ Vérification ratios NaN échouée: {e}")

        # 2. Colonnes constantes ou quasi-constantes
        try:
            nunique_counts = sample_df.nunique(dropna=True)
            constant_cols = nunique_counts[nunique_counts <= 1].index.tolist()
            useless_columns.extend(constant_cols)
        except Exception as e:
            logger.warning(f"⚠️ Vérification valeurs constantes échouée: {e}")

        # 3. Colonnes numériques avec variance quasi nulle
        try:
            numeric_cols = sample_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                desc = sample_df[numeric_cols].describe().T
                low_var_cols = desc.index[
                    (desc["std"].fillna(0) == 0) |
                    ((desc["mean"].replace(0, np.nan)).notna() &
                     (abs(desc["std"] / desc["mean"].replace(0, np.nan)) < min_unique_ratio))
                ].tolist()
                useless_columns.extend(low_var_cols)
        except Exception as e:
            logger.warning(f"⚠️ Vérification faible variance échouée: {e}")

        # Nettoyage final
        useless_columns = list(set(useless_columns))
        valid_useless_columns = [col for col in useless_columns if col in df.columns]

        logger.info(f"✅ Détection colonnes inutiles terminée: {len(valid_useless_columns)} colonnes")
        return valid_useless_columns

    except Exception as e:
        logger.error(f"❌ Erreur critique dans detect_useless_columns: {e}")
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
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil non disponible", "timestamp": time.time()}
            
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
        logger.error(f"❌ Vérification santé système échouée: {e}")
        return {"error": str(e), "timestamp": time.time()}

def cleanup_memory() -> int:
    """
    Force le nettoyage mémoire et du cache Streamlit.
    
    Returns:
        Nombre d'objets collectés
    """
    try:
        # Nettoyage du garbage collector Python
        collected = gc.collect()
        
        # Nettoyage du cache Streamlit si disponible
        if STREAMLIT_AVAILABLE and hasattr(st, 'cache_data') and hasattr(st.cache_data, 'clear'):
            st.cache_data.clear()
            
        logger.info(f"🧹 Nettoyage mémoire terminé: {collected} objets collectés")
        return collected
        
    except Exception as e:
        logger.error(f"❌ Nettoyage mémoire échoué: {e}")
        return 0

# Export des fonctions principales
__all__ = [
    'auto_detect_column_types',
    'get_column_profile',
    'get_data_profile',
    'analyze_columns',
    'detect_imbalance',
    'get_target_and_task',
    'get_relevant_features',
    'detect_useless_columns',
    'get_system_health',
    'cleanup_memory',
    'safe_sample',
    'optimize_dataframe'
]