import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, f_oneway, chi2_contingency
import logging
import time
import streamlit as st
from . import templates  # Fichier de templates

logger = logging.getLogger(__name__)

def plot_overview_metrics(profile: dict):
    """Affiche les métriques clés du dataset sous forme de subplot."""
    try:
        fig = make_subplots(
            rows=1, cols=5,
            specs=[[{'type': 'indicator'}] * 5],
            vertical_spacing=0.1
        )
        fig.add_trace(go.Indicator(
            mode="number", value=profile['n_rows'], title={"text": "Lignes"}), row=1, col=1)
        fig.add_trace(go.Indicator(
            mode="number", value=profile['n_cols'], title={"text": "Colonnes"}), row=1, col=2)
        fig.add_trace(go.Indicator(
            mode="number", number={"suffix": " %"}, value=round(profile['missing_percentage'], 2), title={"text": "Données Manquantes"}), row=1, col=3)
        fig.add_trace(go.Indicator(
            mode="number", value=profile['duplicate_rows'], title={"text": "Lignes Dupliquées"}), row=1, col=4)
        fig.add_trace(go.Indicator(
            mode="number", number={"suffix": " MB"}, value=round(profile['memory_usage'], 2), title={"text": "Taille en Mémoire"}), row=1, col=5)
        
        fig.update_layout(
            height=150, margin=dict(l=10, r=10, t=30, b=10),
            template=templates.DATALAB_TEMPLATE
        )
        logger.info("Métriques clés du dataset générées avec succès.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de la génération des métriques clés : {str(e)}")
        return None

def plot_missing_values_overview(_df: pd.DataFrame):
    """Crée un graphique à barres du nombre de valeurs manquantes par colonne."""
    try:
        if _df.empty or not _df.columns.tolist():
            logger.info("Aucun DataFrame ou colonnes disponibles pour le graphique des valeurs manquantes.")
            return None
            
        missing = _df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if missing.empty:
            logger.info("Aucune valeur manquante détectée dans le dataset.")
            return None
            
        fig = px.bar(missing, x=missing.index, y=missing.values, labels={'x': 'Colonnes', 'y': 'Nombre de valeurs manquantes'},
                     title="Vue d'ensemble des valeurs manquantes")
        fig.update_layout(template=templates.DATALAB_TEMPLATE)
        logger.info("Graphique des valeurs manquantes généré avec succès.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de la génération du graphique des valeurs manquantes : {str(e)}")
        return None

def plot_cardinality_overview(_df: pd.DataFrame, column_types: dict):
    """Crée un graphique à barres de la cardinalité pour les variables catégoriques et textuelles."""
    try:
        cat_text_cols = [col for col in (column_types.get('categorical', []) + column_types.get('text', [])) if col in _df.columns]
        if not cat_text_cols:
            logger.info("Aucune colonne catégorique ou textuelle valide pour le graphique de cardinalité.")
            return None
            
        cardinality = _df[cat_text_cols].nunique().sort_values(ascending=False)
        fig = px.bar(cardinality, x=cardinality.index, y=cardinality.values, labels={'x': 'Colonnes', 'y': 'Nombre de valeurs uniques'},
                     title="Cardinalité des variables non-numériques")
        fig.update_layout(template=templates.DATALAB_TEMPLATE)
        logger.info("Graphique de cardinalité généré avec succès.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de la génération du graphique de cardinalité : {str(e)}")
        return None

def plot_distribution(series: pd.Series, name: str):
    """Crée un histogramme et un box plot pour une variable numérique."""
    try:
        # Vérifier si la série est vide ou ne contient que des NaN
        if series.empty or series.isna().all():
            logger.info(f"Aucune donnée valide pour la distribution de {name} (série vide ou toutes valeurs NaN).")
            return None
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.8, 0.2])
        
        fig.add_trace(go.Histogram(x=series, name='Distribution'), row=1, col=1)
        fig.add_trace(go.Box(x=series, name='Box Plot'), row=2, col=1)
        
        fig.update_layout(
            title_text=f"Distribution de {name}",
            showlegend=False,
            height=400,
            template=templates.DATALAB_TEMPLATE
        )
        logger.info(f"Distribution de {name} générée avec succès.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la distribution pour {name} : {str(e)}")
        return None

def plot_bivariate_analysis(df: pd.DataFrame, col1: str, col2: str, type1: str, type2: str):
    """Génère le graphique approprié pour une analyse bivariée."""
    try:
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Une ou plusieurs colonnes ({col1}, {col2}) ne sont pas dans le DataFrame.")
            return None
            
        title = f"{col1} vs. {col2}"
        if type1 == 'numeric' and type2 == 'numeric':
            fig = px.scatter(df, x=col1, y=col2, title=title, trendline="ols", trendline_color_override="red")
        elif type1 == 'numeric' and type2 == 'categorical':
            fig = px.box(df, x=col2, y=col1, title=title, color=col2)
        elif type1 == 'categorical' and type2 == 'numeric':
            fig = px.box(df, x=col1, y=col2, title=title, color=col1)
        elif type1 == 'categorical' and type2 == 'categorical':
            contingency_table = pd.crosstab(df[col1], df[col2])
            fig = px.imshow(contingency_table, text_auto=True, title=title,
                            labels=dict(x=col2, y=col1, color="Count"))
        else:
            logger.warning(f"Type de colonnes non supporté pour l'analyse bivariée : {type1} vs {type2}")
            return None
            
        fig.update_layout(template=templates.DATALAB_TEMPLATE)
        logger.info(f"Analyse bivariée pour {col1} vs {col2} générée avec succès.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse bivariée pour {col1} vs {col2} : {str(e)}")
        return None

def calculate_cramer_v(x: pd.Series, y: pd.Series):
    """
    Calcule le coefficient de Cramer V pour deux variables catégoriques.
    """
    try:
        x = x.astype('category')
        y = y.astype('category')
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
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1:
            f_stat, _ = f_oneway(*groups)
            result = min(f_stat / (f_stat + 1), 1.0)
            logger.debug(f"ANOVA pour {numeric_col} vs {categorical_col} calculé en {time.time() - start_time:.2f} secondes : {result:.2f}")
            return result
        logger.debug(f"ANOVA pour {numeric_col} vs {categorical_col} ignoré : pas assez de groupes valides.")
        return 0.0
    except Exception as e:
        logger.warning(f"Erreur dans ANOVA pour {numeric_col} vs {categorical_col}: {str(e)}")
        return 0.0

@st.cache_data
def plot_correlation_heatmap(df: pd.DataFrame, target_column: str = None, task_type: str = "classification", timeout: int = 30):
    """
    Crée un heatmap interactif des corrélations pour un dataset supervisé ou non supervisé.
    """
    start_time = time.time()
    try:
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        cols_to_exclude = [col for col in df.columns if col.lower() in ['product id', 'id'] or df[col].nunique() > 100 or df[col].nunique() == len(df)]
        if cols_to_exclude:
            logger.info(f"Colonnes exclues du heatmap (identifiants ou cardinalité élevée) : {cols_to_exclude}")
            numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
            categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
        
        if target_column and target_column in df.columns:
            if target_column in categorical_cols:
                df[target_column] = pd.Categorical(df[target_column]).codes
                if target_column not in numeric_cols:
                    numeric_cols.append(target_column)
                    categorical_cols = [c for c in categorical_cols if c != target_column]
            cols_to_analyze = numeric_cols + categorical_cols
        else:
            cols_to_analyze = numeric_cols + categorical_cols
            task_type = "unsupervised"
        
        if not cols_to_analyze:
            logger.error("Aucune colonne à analyser pour le heatmap.")
            return None

        corr_matrix = pd.DataFrame(np.zeros((len(cols_to_analyze), len(cols_to_analyze))),
                                  index=cols_to_analyze, columns=cols_to_analyze)

        for i, col1 in enumerate(cols_to_analyze):
            for j, col2 in enumerate(cols_to_analyze):
                if time.time() - start_time > timeout:
                    logger.error(f"Timeout atteint ({timeout} secondes) lors du calcul des corrélations.")
                    return None
                if i > j:
                    continue
                logger.debug(f"Calcul de la corrélation entre {col1} et {col2}")
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                elif col1 in numeric_cols and col2 in numeric_cols:
                    corr_matrix.loc[col1, col2] = df[col1].corr(df[col2], method='pearson')
                elif (col1 in numeric_cols and col2 in categorical_cols) or (col1 in categorical_cols and col2 in numeric_cols):
                    num_col = col1 if col1 in numeric_cols else col2
                    cat_col = col2 if col1 in numeric_cols else col1
                    corr_matrix.loc[col1, col2] = calculate_anova_score(df, num_col, cat_col)
                else:
                    corr_matrix.loc[col1, col2] = calculate_cramer_v(df[col1], df[col2])
                corr_matrix.loc[col2, col1] = corr_matrix.loc[col1, col2]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"
        ))

        title = f"Heatmap des corrélations (Cible: {target_column or 'Aucune'})"
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600,
            width=800,
            template=templates.DATALAB_TEMPLATE,
            xaxis_tickangle=45
        )

        logger.info(f"Heatmap des corrélations généré avec succès en {time.time() - start_time:.2f} secondes.")
        return fig
    except Exception as e:
        logger.error(f"Erreur lors de la génération du heatmap : {str(e)}")
        return None