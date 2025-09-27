from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, f_oneway, chi2_contingency
import logging
import time
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
from . import templates  # Fichier de templates

# Configuration du logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constantes de configuration
MAX_CATEGORIES = 100
HEATMAP_TIMEOUT = 30
MAX_THREADS = 4
PLOT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}

class PlottingError(Exception):
    """Exception personnalisée pour les erreurs de plotting"""
    pass

def _validate_dataframe(df: pd.DataFrame) -> bool:
    """Valide le DataFrame pour le plotting"""
    if not isinstance(df, pd.DataFrame):
        logger.error("L'argument n'est pas un DataFrame pandas")
        return False
    if df.empty:
        logger.warning("Le DataFrame est vide")
        return False
    return True

@st.cache_data(ttl=3600, max_entries=50)  # Cache plus agressif
def plot_overview_metrics(profile: Dict) -> Optional[go.Figure]:
    """Affiche les métriques clés du dataset sous forme de subplot."""
    try:
        if not profile or not isinstance(profile, dict):
            logger.error("Profil invalide ou vide")
            return None

        required_keys = ['n_rows', 'n_cols', 'missing_percentage', 'duplicate_rows', 'memory_usage']
        if not all(key in profile for key in required_keys):
            logger.error("Profil incomplet, clés manquantes")
            return None

        metrics_config = [
            {'value': profile['n_rows'], 'title': "Lignes", 'suffix': ""},
            {'value': profile['n_cols'], 'title': "Colonnes", 'suffix': ""},
            {'value': round(profile['missing_percentage'], 2), 'title': "Données Manquantes", 'suffix': " %"},
            {'value': profile['duplicate_rows'], 'title': "Lignes Dupliquées", 'suffix': ""},
            {'value': round(profile['memory_usage'], 2), 'title': "Taille en Mémoire", 'suffix': " MB"}
        ]

        fig = make_subplots(
            rows=1, cols=5,
            specs=[[{'type': 'indicator'}] * 5],
            vertical_spacing=0.1
        )

        for i, metric in enumerate(metrics_config, 1):
            fig.add_trace(go.Indicator(
                mode="number",
                value=metric['value'],
                number={'suffix': metric['suffix']},
                title={"text": metric['title']}
            ), row=1, col=i)

        fig.update_layout(
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            template=templates.DATALAB_TEMPLATE
        )
        
        logger.debug("Métriques clés générées avec succès")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des métriques: {str(e)}", exc_info=True)
        return None

@st.cache_data(ttl=1800, max_entries=20)
def plot_missing_values_overview(_df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique à barres du nombre de valeurs manquantes par colonne."""
    try:
        if not _validate_dataframe(_df):
            return None

        missing = _df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if missing.empty:
            logger.info("Aucune valeur manquante détectée")
            return None

        fig = px.bar(
            x=missing.index, 
            y=missing.values,
            labels={'x': 'Colonnes', 'y': 'Nombre de valeurs manquantes'},
            title="Vue d'ensemble des valeurs manquantes"
        )
        fig.update_layout(
            template=templates.DATALAB_TEMPLATE,
            xaxis_tickangle=45
        )
        logger.debug("Graphique des valeurs manquantes généré")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur génération valeurs manquantes: {str(e)}", exc_info=True)
        return None

@st.cache_data(ttl=1800, max_entries=20)
def plot_cardinality_overview(_df: pd.DataFrame, column_types: Dict[str, List]) -> Optional[go.Figure]:
    """Crée un graphique à barres de la cardinalité pour les variables catégoriques et textuelles."""
    try:
        if not _validate_dataframe(_df):
            return None

        cat_text_cols = [
            col for col in (column_types.get('categorical', []) + column_types.get('text', [])) 
            if col in _df.columns and _df[col].nunique() <= MAX_CATEGORIES
        ]
        
        if not cat_text_cols:
            logger.info("Aucune colonne catégorique/textuelle valide")
            return None

        cardinality = _df[cat_text_cols].nunique().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=cardinality.index, y=cardinality.values, ax=ax)
        ax.set_title("Cardinalité des Variables Non-Numériques")
        ax.set_xlabel("Colonnes")
        ax.set_ylabel("Nombre de Valeurs Uniques")
        ax.tick_params(axis='x', rotation=45)
        
        logger.debug("Graphique de cardinalité généré")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur génération cardinalité: {str(e)}", exc_info=True)
        return None

@st.cache_data(ttl=1200, max_entries=30)
def plot_distribution(series: pd.Series, name: str) -> Optional[go.Figure]:
    """Crée un histogramme et un box plot pour une variable numérique."""
    try:
        if not isinstance(series, pd.Series) or series.empty or series.isna().all():
            logger.warning(f"Série {name} invalide ou vide")
            return None

        # Nettoyage des valeurs aberrantes pour une meilleure visualisation
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if clean_series.empty:
            logger.warning(f"Série {name} ne contient que des valeurs non finies")
            return None

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Distribution de {name}', 'Box Plot')
        )
        
        fig.add_trace(go.Histogram(x=clean_series, name='Distribution', nbinsx=50), row=1, col=1)
        fig.add_trace(go.Box(x=clean_series, name='Box Plot', showlegend=False), row=2, col=1)
        
        fig.update_layout(
            showlegend=False,
            height=500,
            template=templates.DATALAB_TEMPLATE
        )
        
        logger.debug(f"Distribution de {name} générée")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur génération distribution {name}: {str(e)}", exc_info=True)
        return None

@st.cache_data(ttl=900, max_entries=50)
def plot_bivariate_analysis(
    df: pd.DataFrame, 
    col1: str, 
    col2: str, 
    type1: str, 
    type2: str
) -> Optional[go.Figure]:
    """Génère le graphique approprié pour une analyse bivariée."""
    try:
        if not _validate_dataframe(df) or col1 not in df.columns or col2 not in df.columns:
            return None

        # Vérification des données disponibles
        subset = df[[col1, col2]].dropna()
        if subset.empty:
            logger.warning(f"Pas de données disponibles pour {col1} vs {col2}")
            return None

        title = f"{col1} vs. {col2}"
        fig = None

        if type1 == 'numeric' and type2 == 'numeric':
            fig = px.scatter(subset, x=col1, y=col2, title=title, trendline="ols")
        elif (type1 == 'numeric' and type2 == 'categorical') or (type1 == 'categorical' and type2 == 'numeric'):
            num_col, cat_col = (col1, col2) if type1 == 'numeric' else (col2, col1)
            
            # Limiter le nombre de catégories affichées
            if subset[cat_col].nunique() > 20:
                top_categories = subset[cat_col].value_counts().head(20).index
                subset = subset[subset[cat_col].isin(top_categories)]
                
            fig = px.box(subset, x=cat_col, y=num_col, title=title, color=cat_col)
        elif type1 == 'categorical' and type2 == 'categorical':
            # Échantillonnage pour éviter les matrices trop grandes
            if len(subset) > 1000:
                subset = subset.sample(n=1000, random_state=42)
                
            contingency_table = pd.crosstab(subset[col1], subset[col2])
            fig = px.imshow(contingency_table, text_auto=True, title=title,
                          labels=dict(x=col2, y=col1, color="Count"))
        else:
            logger.warning(f"Combinaison de types non supportée: {type1} vs {type2}")
            return None

        if fig:
            fig.update_layout(
                template=templates.DATALAB_TEMPLATE,
                height=500
            )
            logger.debug(f"Analyse bivariée {col1} vs {col2} générée")
            return fig

    except Exception as e:
        logger.error(f"Erreur analyse bivariée {col1} vs {col2}: {str(e)}", exc_info=True)
        return None

def _calculate_correlation_pair(args: Tuple) -> Tuple[str, str, float]:
    """Calcule la corrélation pour une paire de colonnes (pour parallélisation)"""
    col1, col2, df, numeric_cols, categorical_cols = args
    
    if col1 == col2:
        return col1, col2, 1.0
        
    try:
        if col1 in numeric_cols and col2 in numeric_cols:
            corr = df[col1].corr(df[col2], method='pearson')
        elif (col1 in numeric_cols and col2 in categorical_cols):
            corr = _calculate_anova_score(df, col1, col2)
        elif (col1 in categorical_cols and col2 in numeric_cols):
            corr = _calculate_anova_score(df, col2, col1)
        else:
            corr = _calculate_cramer_v(df[col1], df[col2])
            
        return col1, col2, corr if not np.isnan(corr) else 0.0
        
    except Exception as e:
        logger.debug(f"Erreur calcul corrélation {col1}-{col2}: {str(e)}")
        return col1, col2, 0.0

def _calculate_cramer_v(x: pd.Series, y: pd.Series) -> float:
    """Calcule le coefficient de Cramer V optimisé"""
    try:
        if x.nunique() > MAX_CATEGORIES or y.nunique() > MAX_CATEGORIES:
            return 0.0
            
        contingency_table = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        
    except Exception:
        return 0.0

def _calculate_anova_score(df: pd.DataFrame, numeric_col: str, categorical_col: str) -> float:
    """Calcule un score basé sur ANOVA optimisé"""
    try:
        if df[categorical_col].nunique() > MAX_CATEGORIES:
            return 0.0
            
        groups = [group for _, group in df.groupby(categorical_col)[numeric_col] if len(group) > 1]
        if len(groups) > 1:
            f_stat, _ = f_oneway(*groups)
            return min(f_stat / (f_stat + 1), 1.0)
        return 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=3600, max_entries=5)  # Cache plus long, moins d'entrées
def plot_correlation_heatmap(
    df: pd.DataFrame, 
    target_column: Optional[str] = None, 
    task_type: str = "classification",
    timeout: int = 60  # Timeout augmenté
) -> Optional[go.Figure]:
    """Crée un heatmap interactif des corrélations avec optimisations."""
    start_time = time.time()
    
    try:
        # Validation basique du DataFrame
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("DataFrame vide pour le heatmap")
            return None

        logger.info(f"Début génération heatmap - {len(df.columns)} colonnes, {len(df)} lignes")

        # Sélection des colonnes numériques uniquement pour commencer simple
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Exclusion des colonnes problématiques
        exclude_patterns = ['id', 'product', 'sku', 'code', 'index', 'unique']
        cols_to_exclude = [
            col for col in numeric_cols 
            if (any(pattern in col.lower() for pattern in exclude_patterns) or 
                df[col].nunique() > 1000)  # Seuil plus élevé
        ]
        
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        # Gestion de la colonne cible
        if target_column and target_column in df.columns:
            if target_column not in numeric_cols:
                # Convertir la cible si elle est catégorielle
                try:
                    if task_type == "classification":
                        df[target_column] = pd.Categorical(df[target_column]).codes
                        numeric_cols.append(target_column)
                except Exception as e:
                    logger.warning(f"Impossible de convertir la cible {target_column}: {e}")
        
        # Limiter le nombre de colonnes pour la performance
        max_cols = 30  # Réduit pour la performance
        if len(numeric_cols) > max_cols:
            # Garder les colonnes les plus intéressantes (avec moins de valeurs manquantes)
            missing_rates = df[numeric_cols].isnull().mean()
            numeric_cols = missing_rates.nsmallest(max_cols).index.tolist()
            logger.info(f"Limite à {max_cols} colonnes sur {len(missing_rates)}")
        
        if len(numeric_cols) < 2:
            logger.warning("Pas assez de colonnes numériques pour le heatmap")
            return None

        # Nettoyer les données avant calcul
        df_clean = df[numeric_cols].dropna()
        
        if len(df_clean) < 10:  # Minimum de données
            logger.warning("Pas assez de données après nettoyage")
            return None

        # Calcul de corrélation simple et robuste
        try:
            corr_matrix = df_clean.corr(method='pearson', numeric_only=True)
            
            # Vérifier que la matrice n'est pas vide
            if corr_matrix.empty or corr_matrix.isna().all().all():
                logger.warning("Matrice de corrélation vide")
                return None
                
            # Remplacer les NaN par 0 pour l'affichage
            corr_matrix = corr_matrix.fillna(0)
            
        except Exception as e:
            logger.error(f"Erreur calcul corrélation: {e}")
            return None

        # Création du heatmap simplifié
        try:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                hoverinfo='z',
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corrélation: %{z:.3f}<extra></extra>",
                showscale=True,
                text=corr_matrix.round(2).values,      # <-- Texte ajouté
                texttemplate="%{text}"   
            ))

            title = f"Corrélations entre {len(numeric_cols)} variables"
            if target_column:
                title += f" (Cible: {target_column})"

            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center'),
                xaxis_title="Variables",
                yaxis_title="Variables",
                height=600,
                width=800,
                template="plotly_white",  # Template simple
                xaxis_tickangle=45,
                margin=dict(l=50, r=50, t=80, b=50)
            )

            # Améliorer la lisibilité
            fig.update_xaxes(tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))

            elapsed = time.time() - start_time
            logger.info(f"Heatmap généré en {elapsed:.2f}s - {len(numeric_cols)} variables")
            return fig, numeric_cols

        except Exception as e:
            logger.error(f"Erreur création figure heatmap: {e}")
            return None

    except Exception as e:
        logger.error(f"Erreur critique génération heatmap: {str(e)}")
        return None