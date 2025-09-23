import pandas as pd
import dask.dataframe as dd
import numpy as np
import logging
from typing import Union, Tuple, Dict, Any, List
from scipy.stats import pointbiserialr, f_oneway, chi2_contingency
import time

# Added a comment to force re-parsing
logger = logging.getLogger(__name__)

def compute_if_dask(data: Any) -> Any:
    """Exécute .compute() si l'objet est un Dask DataFrame, Series, ou Scalar."""
    if isinstance(data, (dd.DataFrame, dd.Series, dd.Scalar)):
        return data.compute()
    return data

def is_dask_dataframe(df: Any) -> bool:
    """Vérifie si l'objet est un DataFrame Dask."""
    try:
        import dask.dataframe as dd
        return isinstance(df, dd.DataFrame)
    except ImportError:
        return False

def sanitize_column_types_for_display(df: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    is_dask = is_dask_dataframe(df)
    df_display = df.head(500) if is_dask else df.copy()
    columns_changed = {}
    for col in df_display.select_dtypes(include=['object']).columns:
        try:
            inferred_types = df_display[col].apply(type).unique()
            if len(inferred_types) > 1:
                df_display[col] = df_display[col].astype(str)
                columns_changed[col] = f"{inferred_types} -> str"
        except Exception as e:
            df_display[col] = df_display[col].astype(str)
            columns_changed[col] = f"mixed -> str"
    return df_display, columns_changed

def auto_detect_column_types(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    potential_categorical = []
    high_cardinality = []
    for col in object_cols:
        nunique = compute_if_dask(df[col].nunique())
        total_count = compute_if_dask(df[col].count())
        if total_count > 0 and nunique / total_count < 0.5 and nunique < 100:
            potential_categorical.append(col)
        else:
            high_cardinality.append(col)
    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": potential_categorical,
        "text_or_high_cardinality": high_cardinality
    }

def get_column_profile(series: Union[pd.Series, dd.Series]) -> Dict[str, Any]:
    is_dask = is_dask_dataframe(series)
    base_profile = {
        "count": compute_if_dask(series.count()),
        "missing_values": compute_if_dask(series.isna().sum()),
        "missing_percentage": f"{compute_if_dask(series.isna().mean()) * 100:.2f}%"
    }
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric_profile = {
            "mean": compute_if_dask(series.mean()),
            "std_dev": compute_if_dask(series.std()),
            "min": compute_if_dask(series.min()),
            "25%": compute_if_dask(series.quantile(0.25)),
            "median": compute_if_dask(series.median()),
            "75%": compute_if_dask(series.quantile(0.75)),
            "max": compute_if_dask(series.max())
        }
        base_profile.update(numeric_profile)
    else:
        base_profile["unique_values"] = compute_if_dask(series.nunique())
    return base_profile

def get_data_profile(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    return {col: get_column_profile(df[col]) for col in df.columns}

def analyze_columns(df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, List[str]]:
    constant_cols = []
    id_like_cols = []
    df_computed = compute_if_dask(df)
    for col in df_computed.columns:
        nunique = df_computed[col].nunique()
        if nunique == 1:
            constant_cols.append(col)
        if nunique == len(df_computed):
            id_like_cols.append(col)
    return {"constant": constant_cols, "id_like": id_like_cols}

def get_target_and_task(df: pd.DataFrame, column_types: Dict) -> Tuple[str, str]:
    if column_types['categorical']:
        target = min(column_types['categorical'], key=lambda c: df[c].nunique())
        nunique = df[target].nunique()
        if nunique == 2:
            return target, "Classification Binaire"
        else:
            return target, "Classification Multiclasse"
    elif column_types['numeric']:
        return column_types['numeric'][0], "Régression"
    return df.columns[-1], "Inconnu"

def get_imbalance_details(df: pd.DataFrame, target: str, task_type: str) -> Dict:
    if "Classification" not in task_type or target not in df.columns:
        return None
    class_counts = df[target].value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    ratio = min_class_count / max_class_count if max_class_count > 0 else 0
    return {
        "is_imbalanced": ratio < 0.1,
        "min_class_ratio": ratio,
        "min_class_name": class_counts.idxmin(),
        "min_class_count": min_class_count
    }

def detect_useless_columns(df: Union[pd.DataFrame, dd.DataFrame]) -> List[str]:
    analysis = analyze_columns(df)
    return analysis['constant'] + analysis['id_like']

def calculate_cramer_v(x: pd.Series, y: pd.Series):
    """
    Calcule le coefficient de Cramer V pour deux variables catégoriques.
    """
    try:
        # Convertir en catégorie pour optimiser la mémoire
        x = x.astype('category')
        y = y.astype('category')
        # Vérifier le nombre de catégories uniques
        max_categories = 100
        if x.nunique() > max_categories or y.nunique() > max_categories:
            logger.warning(f"Nombre de catégories élevées ({x.nunique()} vs {y.nunique()}) pour {x.name} vs {y.name}. Saut du calcul.")
            return 0.0
        start_time = time.time()
        contingency_table = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        result = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        logger.debug(f"Cramer V pour {x.name} vs {y.name} calculé en {time.time() - start_time:.2f} secondes : {result:.2f}")
        return result
    except Exception as e:
        logger.warning(f"Erreur dans Cramer V pour {x.name} vs {y.name}: {str(e)}")
        return 0.0

def calculate_anova_score(df: pd.DataFrame, numeric_col: str, categorical_col: str):
    """
    Calcule un score basé sur ANOVA pour une variable numérique et une catégorique.
    """
    try:
        df[categorical_col] = df[categorical_col].astype('category')
        max_categories = 100
        if df[categorical_col].nunique() > max_categories:
            logger.warning(f"Nombre de catégories élevées ({df[categorical_col].nunique()}) pour {categorical_col}. Saut du calcul.")
            return 0.0
        start_time = time.time()
        groups = [df[numeric_col][df[categorical_col] == cat] for cat in df[categorical_col].unique()]
        groups = [g for g in groups if len(g) > 1]  # Ignorer les groupes avec moins de 2 observations
        if len(groups) > 1:
            f_stat, _ = f_oneway(*groups)
            result = min(f_stat / (f_stat + 1), 1.0)  # Normaliser entre 0 et 1
            logger.debug(f"ANOVA pour {numeric_col} vs {categorical_col} calculé en {time.time() - start_time:.2f} secondes : {result:.2f}")
            return result
        logger.debug(f"ANOVA pour {numeric_col} vs {categorical_col} ignoré : pas assez de groupes valides.")
        return 0.0
    except Exception as e:
        logger.warning(f"Erreur dans ANOVA pour {numeric_col} vs {categorical_col}: {str(e)}")
        return 0.0

def get_task_type(series: pd.Series) -> Tuple[str, int]:
    """
    Détermine le type de tâche (classification ou régression) et le nombre de classes/valeurs uniques.
    """
    n_unique = series.nunique()
    if pd.api.types.is_numeric_dtype(series):
        # Si numérique et peu de valeurs uniques, c'est une classification numérique
        if n_unique <= 20 and n_unique / len(series) < 0.1: # Heuristique pour éviter les ID numériques
            return "classification", n_unique
        else:
            return "regression", n_unique
    else: # Catégorique ou autre type non numérique
        return "classification", n_unique

def auto_detect_target(df: pd.DataFrame) -> str:
    """
    Tente de détecter automatiquement la colonne cible.
    """
    potential_targets = [col for col in df.columns if col.lower() in ['target', 'label', 'y']]
    if potential_targets:
        return potential_targets[0]
    return df.columns[-1] # Par défaut, la dernière colonne

def get_relevant_features(df: pd.DataFrame, target_column: str, top_n: int = 10) -> List[str]:
    """
    Sélectionne les features les plus pertinentes en fonction de leur corrélation avec la cible.
    """
    if target_column not in df.columns:
        return df.columns.tolist() # Retourne toutes les colonnes si la cible est introuvable

    target_series = df[target_column]
    feature_scores = {}
    
    target_is_numeric = pd.api.types.is_numeric_dtype(target_series)

    for col in df.columns:
        if col == target_column: continue
        if df[col].isnull().any(): continue # Ignorer les colonnes avec des NaN pour le calcul de pertinence

        feature_is_numeric = pd.api.types.is_numeric_dtype(df[col])

        try:
            if target_is_numeric and feature_is_numeric:
                # Pearson correlation
                score = abs(df[col].corr(target_series, method='pearson'))
            elif target_is_numeric and not feature_is_numeric:
                # ANOVA score (numérique vs catégorique)
                score = calculate_anova_score(df, col, target_column)
            elif not target_is_numeric and feature_is_numeric:
                # Point-biserial correlation (catégorique binaire vs numérique) ou ANOVA
                if target_series.nunique() == 2:
                    score = abs(pointbiserialr(df[col], target_series)[0])
                else:
                    score = calculate_anova_score(df, col, target_column)
            else: # Both categorical
                # Cramer's V
                score = calculate_cramer_v(df[col], target_series)
            feature_scores[col] = score
        except Exception as e:
            logger.warning(f"Impossible de calculer la pertinence pour {col} vs {target_column}: {e}")
            feature_scores[col] = 0 # Assign a low score on error

    # Trier et retourner les top_n features
    sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    return [f[0] for f in sorted_features[:top_n]]

def detect_imbalance(df: pd.DataFrame, target_column: str, threshold: float = 0.1) -> bool:
    """
    Détecte si la colonne cible est déséquilibrée pour une tâche de classification.
    """
    if target_column not in df.columns:
        return False
    
    counts = df[target_column].value_counts(normalize=True)
    if counts.empty:
        return False
    
    # Si la plus petite classe est inférieure à un certain seuil
    return counts.min() < threshold