import pandas as pd
import dask.dataframe as dd
import logging
from typing import Dict, Any, Tuple, Union

# Configuration du logging pour production
logger = logging.getLogger(__name__)

def intelligent_type_coercion(df: Union[pd.DataFrame, dd.DataFrame], use_dask: bool) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Dict[str, str]]:
    """
    Applique une coercion de type intelligente sur les colonnes pour éviter les erreurs de type mixte.
    """
    changes = {}
    
    if use_dask:
        logger.info("La coercion de type intelligente pour Dask est limitée. Inspectez les types manuellement.")
        return df, changes

    df_copy = df.copy()
    date_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M', '%d-%m-%Y', '%Y%m%d'
    ]

    for col in df_copy.select_dtypes(include=['object']).columns:
        try:
            # Tentative de conversion en numérique
            numeric_col = pd.to_numeric(df_copy[col], errors='coerce')
            if numeric_col.notna().sum() / len(df_copy[col].dropna()) > 0.9:
                if df_copy[col].dtype != numeric_col.dtype:
                    df_copy[col] = numeric_col
                    changes[col] = f"object -> {numeric_col.dtype}"
                    logger.info(f"Colonne '{col}' convertie en numérique (type: {numeric_col.dtype}).")
                continue

            # Tentative de conversion en datetime avec formats explicites
            for fmt in date_formats:
                try:
                    datetime_col = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                    if datetime_col.notna().sum() / len(df_copy[col].dropna()) > 0.9:
                        if df_copy[col].dtype != datetime_col.dtype:
                            df_copy[col] = datetime_col
                            changes[col] = f"object -> datetime64[ns]"
                            logger.info(f"Colonne '{col}' convertie en datetime avec format {fmt}.")
                        break
                except Exception:
                    continue
            else:
                # Si aucun format de date ne fonctionne, essayer dateutil comme secours
                datetime_col = pd.to_datetime(df_copy[col], errors='coerce', format=None)
                if datetime_col.notna().sum() / len(df_copy[col].dropna()) > 0.9:
                    if df_copy[col].dtype != datetime_col.dtype:
                        df_copy[col] = datetime_col
                        changes[col] = f"object -> datetime64[ns] (via dateutil)"
                        logger.info(f"Colonne '{col}' convertie en datetime via dateutil.")
                    continue

            # Si cardinalité faible, convertir en catégorie
            if df_copy[col].nunique() / len(df_copy[col].dropna()) < 0.5:
                df_copy[col] = df_copy[col].astype('category')
                changes[col] = "object -> category"
                logger.info(f"Colonne '{col}' convertie en 'category' pour optimiser la mémoire.")
            else:
                df_copy[col] = df_copy[col].astype(str)
                changes[col] = "object -> str"
                logger.info(f"Colonne '{col}' convertie en 'str' pour assurer un type homogène.")

        except Exception as e:
            logger.warning(f"Erreur lors de la coercion de type pour la colonne '{col}': {str(e)}")
            df_copy[col] = df_copy[col].astype(str)
            changes[col] = "object -> str (erreur)"
            logger.info(f"Colonne '{col}' convertie en 'str' suite à une erreur.")

    return df_copy, changes

def load_data(
    file_path: str,
    use_dask: bool = False,
    force_dtype: Dict[str, Any] = None,
    sanitize_for_display: bool = True
) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Dict[str, Any]]:
    """
    Charge les données depuis un fichier CSV de manière robuste.
    """
    report = {"actions": [], "changes": {}}
    
    try:
        file_name = file_path if isinstance(file_path, str) else getattr(file_path, 'name', 'fichier_uploadé')
        file_extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
        logger.info(f"Chargement du fichier : {file_name} (extension: {file_extension})")

        if use_dask:
            if file_extension == 'csv':
                df = dd.read_csv(file_path, dtype=force_dtype)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Dask.")
            elif file_extension == 'parquet':
                df = dd.read_parquet(file_path, dtype=force_dtype)
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Dask.")
            else:
                logger.error(f"Extension de fichier non supportée pour Dask : {file_extension}")
                return None, {"actions": [f"Erreur : Extension de fichier '{file_extension}' non supportée pour Dask."]}
        else:
            if file_extension == 'csv':
                df = pd.read_csv(file_path, dtype=force_dtype, low_memory=False)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Pandas (low_memory=False).")
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path)
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Pandas.")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                report["actions"].append(f"Chargement du fichier Excel '{file_name}' avec Pandas.")
            else:
                logger.error(f"Extension de fichier non supportée : {file_extension}")
                return None, {"actions": [f"Erreur : Extension de fichier '{file_extension}' non supportée."]}

        initial_rows = len(df) if not use_dask else "inconnu (Dask)"
        if use_dask:
            df = df.drop_duplicates()
            report["actions"].append("Les lignes dupliquées seront supprimées (opération Dask paresseuse).")
        else:
            df = df.drop_duplicates().reset_index(drop=True)
            final_rows = len(df)
            if initial_rows != final_rows:
                report["actions"].append(f"{initial_rows - final_rows} lignes dupliquées ont été supprimées.")

        if sanitize_for_display and not use_dask:
            df, changes = intelligent_type_coercion(df, use_dask)
            report["changes"] = changes
            if changes:
                report["actions"].append("Certaines colonnes avec des types de données mixtes ont été standardisées.")

        logger.info(f"Données chargées avec {len(df) if not use_dask else 'inconnu (Dask)'} lignes et {len(df.columns)} colonnes.")
        return df, report

    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_name}: {e}")
        return None, {"actions": [f"Erreur lors du chargement : {str(e)}"]}