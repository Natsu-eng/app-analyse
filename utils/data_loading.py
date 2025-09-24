import pandas as pd
import dask.dataframe as dd
import logging
import streamlit as st
from typing import Dict, Any, Tuple, Union, Optional
import os
import re
import time
import psutil
import gc
from functools import wraps
import pyarrow.parquet as pq

# Configuration du logging pour production
logger = logging.getLogger(__name__)

# Extensions de fichiers supportées
SUPPORTED_EXTENSIONS = {'csv', 'parquet', 'xlsx', 'xls', 'json'}
# Taille maximale du fichier en Mo (1 Go)
MAX_FILE_SIZE_MB = 1024

# Décorateurs de monitoring
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
            if duration > 30:  # Plus de 30 secondes
                logger.warning(f"{func.__name__} took {duration:.2f}s - performance issue detected")
            if memory_delta > 500:  # Plus de 500MB
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

def validate_file_integrity(file_path, file_extension: str) -> Dict[str, Any]:
    """
    Valide l'intégrité d'un fichier avant le chargement.
    
    Args:
        file_path: Chemin du fichier
        file_extension: Extension du fichier
    
    Returns:
        Dictionnaire avec le statut de validation
    """
    validation_report = {"is_valid": True, "issues": [], "warnings": []}
    
    try:
        if file_extension == 'csv':
            # Vérifier les premières lignes pour détecter les problèmes d'encoding
            try:
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline() for _ in range(5)]
                else:
                    # Fichier uploadé via Streamlit
                    first_lines = []
                    for _ in range(5):
                        line = file_path.readline()
                        if not line:
                            break
                        first_lines.append(line.decode('utf-8') if isinstance(line, bytes) else line)
                    file_path.seek(0)  # Reset position
                    
                if not any(line.strip() for line in first_lines):
                    validation_report["issues"].append("Fichier CSV vide ou mal formaté")
                    validation_report["is_valid"] = False
                    
            except UnicodeDecodeError:
                validation_report["warnings"].append("Possible problème d'encodage UTF-8")
                
        elif file_extension == 'parquet':
            # Test de lecture rapide des métadonnées
            try:
                if isinstance(file_path, str):
                    pq.read_metadata(file_path)
                else:
                    validation_report["warnings"].append("Validation Parquet limitée pour fichiers uploadés")
            except Exception as e:
                validation_report["issues"].append(f"Fichier Parquet corrompu: {e}")
                validation_report["is_valid"] = False
                
        elif file_extension in ['xlsx', 'xls']:
            # Vérification basique pour Excel
            try:
                if isinstance(file_path, str):
                    # Test rapide de lecture des métadonnées Excel
                    pd.read_excel(file_path, nrows=0)
                else:
                    validation_report["warnings"].append("Validation Excel limitée pour fichiers uploadés")
            except Exception as e:
                validation_report["issues"].append(f"Fichier Excel corrompu: {e}")
                validation_report["is_valid"] = False
                
    except Exception as e:
        validation_report["issues"].append(f"Erreur de validation: {e}")
        validation_report["is_valid"] = False
        logger.error(f"File validation error: {e}")
        
    return validation_report

@monitor_performance
def intelligent_type_coercion(df: Union[pd.DataFrame, dd.DataFrame], use_dask: bool, max_sample_size: int = 100000) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Dict[str, str]]:
    """
    Applique une coercion de type intelligente sur les colonnes pour éviter les erreurs de type mixte.
    Version optimisée pour la production.
    
    Args:
        df: DataFrame Pandas ou Dask
        use_dask: Booléen indiquant si Dask est utilisé
        max_sample_size: Taille maximale de l'échantillon pour l'analyse
    
    Returns:
        Tuple contenant le DataFrame modifié et un dictionnaire des changements de type
    """
    changes = {}
    
    if use_dask:
        logger.info("Coercion de type limitée pour Dask. Inspectez les types manuellement.")
        return df, changes

    df_copy = df.copy()
    
    # Échantillonnage intelligent pour gros datasets
    if len(df_copy) > max_sample_size:
        sample_df = df_copy.sample(n=max_sample_size, random_state=42)
        logger.info(f"Échantillonnage de {max_sample_size} lignes pour l'analyse des types")
    else:
        sample_df = df_copy

    # Formats de date optimisés par fréquence d'utilisation
    date_formats = [
        '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y',
        '%Y/%m/%d', '%d-%m-%Y', '%Y%m%d', '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
    ]

    # Expression régulière optimisée pour détecter les formats de date
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}(?::\d{2})?)?$|^\d{2}/\d{2}/\d{4}(?: \d{2}:\d{2}(?::\d{2})?)?$|^\d{8}$')

    object_columns = df_copy.select_dtypes(include=['object']).columns
    total_columns = len(object_columns)
    
    for idx, col in enumerate(object_columns, 1):
        try:
            # Logging du progrès pour les gros datasets
            if total_columns > 10 and idx % 10 == 0:
                logger.info(f"Type coercion progress: {idx}/{total_columns} columns processed")
            
            sample_col = sample_df[col].dropna()
            if sample_col.empty:
                continue
            
            # Test numérique optimisé
            try:
                numeric_col = pd.to_numeric(sample_col, errors='coerce')
                numeric_success_rate = numeric_col.notna().mean()
                
                if numeric_success_rate > 0.95:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    changes[col] = f"object -> numeric (success rate: {numeric_success_rate:.1%})"
                    logger.debug(f"Colonne '{col}' convertie en numérique.")
                    continue
                    
            except (ValueError, TypeError, OverflowError) as e:
                logger.debug(f"Numeric conversion failed for {col}: {e}")

            # Test de date optimisé
            first_values = sample_col.head(100).astype(str)
            date_like_ratio = sum(1 for val in first_values 
                                if any(char.isdigit() for char in val) and 
                                   any(sep in val for sep in ['-', '/', 'T', ' '])) / len(first_values)

            if date_like_ratio > 0.8:
                datetime_converted = False
                
                # Essayer les formats les plus courants d'abord
                for fmt in date_formats[:6]:  # Top 6 formats les plus courants
                    try:
                        datetime_col = pd.to_datetime(sample_col, format=fmt, errors='coerce')
                        datetime_success_rate = datetime_col.notna().mean()
                        
                        if datetime_success_rate > 0.9:
                            df_copy[col] = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                            changes[col] = f"object -> datetime64[ns] (format: {fmt}, success rate: {datetime_success_rate:.1%})"
                            logger.debug(f"Colonne '{col}' convertie en datetime avec format {fmt}.")
                            datetime_converted = True
                            break
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"DateTime conversion failed for {col} with format {fmt}: {e}")
                        continue

                # Fallback avec infer_datetime_format si pas de format exact trouvé
                if not datetime_converted:
                    try:
                        datetime_col = pd.to_datetime(sample_col, errors='coerce', infer_datetime_format=True)
                        datetime_success_rate = datetime_col.notna().mean()
                        
                        if datetime_success_rate > 0.9:
                            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', infer_datetime_format=True)
                            changes[col] = f"object -> datetime64[ns] (inferred, success rate: {datetime_success_rate:.1%})"
                            logger.debug(f"Colonne '{col}' convertie en datetime via infer_datetime_format.")
                            datetime_converted = True
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"DateTime infer conversion failed for {col}: {e}")

                if datetime_converted:
                    continue

            # Optimisation catégorie vs string
            unique_count = sample_col.nunique()
            total_count = len(sample_col)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            if unique_ratio < 0.5 and unique_count < 1000:
                try:
                    df_copy[col] = df_copy[col].astype('category')
                    changes[col] = f"object -> category (unique ratio: {unique_ratio:.1%})"
                    logger.debug(f"Colonne '{col}' convertie en 'category'.")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Category conversion failed for {col}: {e}")
                    changes[col] = f"object -> object (category conversion failed)"
            else:
                # Garder comme object mais s'assurer de l'homogénéité
                try:
                    df_copy[col] = df_copy[col].astype(str)
                    changes[col] = f"object -> str (homogenization, unique ratio: {unique_ratio:.1%})"
                except Exception as e:
                    logger.debug(f"String conversion failed for {col}: {e}")
                    changes[col] = f"object -> object (no change, conversion failed)"

        except Exception as e:
            logger.warning(f"Type coercion error for column '{col}': {str(e)}")
            changes[col] = f"object -> object (error: {type(e).__name__})"
            continue

    # Nettoyage mémoire
    del sample_df
    gc.collect()
    
    logger.info(f"Type coercion completed: {len(changes)} columns processed")
    return df_copy, changes

@safe_execute(fallback_value=(None, {"actions": ["Erreur critique lors du chargement"]}, None))
@monitor_performance
def load_data(
    file_path: str,
    force_dtype: Dict[str, Any] = None,
    sanitize_for_display: bool = True,
    size_threshold_mb: float = 100.0,
    blocksize: str = "64MB"
) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Dict[str, Any], Union[pd.DataFrame, dd.DataFrame]]:
    """
    Charge les données depuis un fichier et décide automatiquement d'utiliser Pandas ou Dask.
    Version optimisée pour la production avec validation et monitoring.
    
    Args:
        file_path: Chemin du fichier à charger (str ou file-like object)
        force_dtype: Dictionnaire des types forcés pour les colonnes
        sanitize_for_display: Si True, applique la coercion intelligente
        size_threshold_mb: Seuil en Mo pour basculer vers Dask
        blocksize: Taille des blocs pour Dask
    
    Returns:
        Tuple contenant le DataFrame, un rapport d'actions, et le DataFrame brut
    """
    report = {"actions": [], "changes": {}, "warnings": []}
    
    try:
        # Détermination du nom et extension du fichier
        if isinstance(file_path, str):
            file_name = os.path.basename(file_path)
        else:
            file_name = getattr(file_path, 'name', 'fichier_uploadé')
            
        file_extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
        
        # Validation de l'extension
        if file_extension not in SUPPORTED_EXTENSIONS:
            error_msg = f"Extension de fichier non supportée : {file_extension}. Extensions valides : {', '.join(SUPPORTED_EXTENSIONS)}"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Validation de la taille du fichier
        if isinstance(file_path, str):
            if not os.path.exists(file_path):
                error_msg = f"Fichier non trouvé : {file_path}"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        else:
            file_size_mb = getattr(file_path, 'size', 0) / (1024 * 1024)
            
        if file_size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"Taille du fichier ({file_size_mb:.2f} Mo) dépasse la limite de {MAX_FILE_SIZE_MB} Mo"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Validation de l'intégrité du fichier
        validation_result = validate_file_integrity(file_path, file_extension)
        if not validation_result["is_valid"]:
            error_msg = f"Fichier corrompu ou invalide : {'; '.join(validation_result['issues'])}"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None
            
        # Ajouter les avertissements au rapport
        if validation_result["warnings"]:
            report["warnings"].extend(validation_result["warnings"])

        logger.info(f"Chargement du fichier : {file_name} (extension: {file_extension}, taille: {file_size_mb:.2f} Mo)")

        # Décision Pandas vs Dask
        use_dask = file_size_mb > size_threshold_mb
        logger.info(f"Utilisation de {'Dask' if use_dask else 'Pandas'} (seuil: {size_threshold_mb} Mo)")

        # Préparation des paramètres de chargement
        load_params = {}
        if force_dtype:
            load_params['dtype'] = force_dtype

        # Chargement des données selon le type de fichier et l'engine
        df = None
        
        if use_dask:
            if file_extension == 'csv':
                load_params['blocksize'] = blocksize
                df = dd.read_csv(file_path, **load_params)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Dask (blocksize: {blocksize}).")
            elif file_extension == 'parquet':
                df = dd.read_parquet(file_path, **{k: v for k, v in load_params.items() if k != 'blocksize'})
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Dask.")
            elif file_extension == 'json':
                df = dd.read_json(file_path, **{k: v for k, v in load_params.items() if k != 'blocksize'})
                report["actions"].append(f"Chargement du fichier JSON '{file_name}' avec Dask.")
            else:
                error_msg = f"Extension de fichier '{file_extension}' non supportée pour Dask"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None
        else:
            if file_extension == 'csv':
                load_params['low_memory'] = False
                df = pd.read_csv(file_path, **load_params)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Pandas (low_memory=False).")
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path)
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Pandas.")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                report["actions"].append(f"Chargement du fichier Excel '{file_name}' avec Pandas.")
            elif file_extension == 'json':
                df = pd.read_json(file_path)
                report["actions"].append(f"Chargement du fichier JSON '{file_name}' avec Pandas.")
            else:
                error_msg = f"Extension de fichier '{file_extension}' non supportée"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None

        if df is None or (not use_dask and df.empty) or (use_dask and df.npartitions == 0):
            error_msg = "Le fichier chargé est vide"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Sauvegarde du DataFrame brut avant toute modification
        if use_dask:
            df_raw = df.copy()
        else:
            df_raw = df.copy()

        # Calcul du nombre de lignes initial
        initial_rows = len(df) if not use_dask else "inconnu (Dask, calcul paresseux)"

        # Suppression des doublons
        try:
            if use_dask:
                df = df.drop_duplicates()
                report["actions"].append("Suppression des lignes dupliquées (opération Dask paresseuse).")
            else:
                duplicates_count = df.duplicated().sum()
                if duplicates_count > 0:
                    df = df.drop_duplicates().reset_index(drop=True)
                    final_rows = len(df)
                    report["actions"].append(f"{duplicates_count} lignes dupliquées supprimées.")
                else:
                    report["actions"].append("Aucune ligne dupliquée détectée.")
                    
        except Exception as e:
            logger.warning(f"Erreur lors de la suppression des doublons : {e}")
            report["warnings"].append(f"Suppression des doublons échouée : {e}")

        # Coercion intelligente des types (seulement pour Pandas)
        if sanitize_for_display and not use_dask:
            try:
                df, changes = intelligent_type_coercion(df, use_dask)
                report["changes"] = changes
                if changes:
                    report["actions"].append(f"Standardisation des types : {len(changes)} colonnes modifiées.")
            except Exception as e:
                logger.error(f"Erreur lors de la coercion de types : {e}")
                report["warnings"].append(f"Coercion de types échouée : {e}")

        # Sauvegarde dans st.session_state
        try:
            st.session_state.df = df
            st.session_state.df_raw = df_raw
            logger.info("DataFrames sauvegardés dans session_state")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde session_state : {e}")

        # Statistiques finales
        final_rows = len(df) if not use_dask else "inconnu (Dask)"
        final_cols = len(df.columns)
        
        logger.info(f"Données chargées avec succès : {final_rows} lignes et {final_cols} colonnes")
        report["actions"].append(f"Dataset final : {final_rows} lignes × {final_cols} colonnes")
        
        return df, report, df_raw

    except Exception as e:
        error_msg = f"Erreur critique lors du chargement du fichier : {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, {"actions": [error_msg]}, None