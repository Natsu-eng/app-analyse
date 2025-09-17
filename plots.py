from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def barplot_counts(df: pd.DataFrame, column: str) -> go.Figure:
	counts = df[column].astype("category").value_counts(dropna=False)
	fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": column, "y": "Effectifs"})
	fig.update_layout(bargap=0.2)
	return fig


def pie_proportions(df: pd.DataFrame, column: str) -> go.Figure:
	counts = df[column].astype("category").value_counts(dropna=False)
	fig = px.pie(values=counts.values, names=counts.index.astype(str), hole=0.3)
	return fig


def histogram_with_density(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
	series = pd.to_numeric(df[column], errors="coerce").dropna()
	fig = go.Figure()
	fig.add_trace(go.Histogram(x=series, nbinsx=bins, histnorm="probability density", name="Histogramme"))
	# KDE
	if len(series) > 1:
		kde = stats.gaussian_kde(series)
		xs = np.linspace(series.min(), series.max(), 200)
		fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name="Densité"))
	fig.update_layout(xaxis_title=column, yaxis_title="Densité")
	return fig


def boxplot(df: pd.DataFrame, column: str) -> go.Figure:
	return px.box(df, y=column, points="outliers")


def qqplot(df: pd.DataFrame, column: str) -> go.Figure:
	series = pd.to_numeric(df[column], errors="coerce").dropna()
	osm, osr = stats.probplot(series, dist="norm", sparams=(), fit=False)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Données"))
	min_v = min(min(osm), min(osr))
	max_v = max(max(osm), max(osr))
	fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Droite théorique"))
	fig.update_layout(xaxis_title="Quantiles théoriques", yaxis_title="Quantiles observés")
	return fig


def scatterplot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
	fig = px.scatter(df, x=x, y=y, color=color, trendline="ols")
	return fig


def boxplot_bivariate(df: pd.DataFrame, x_categ: str, y_numeric: str) -> go.Figure:
	return px.box(df, x=x_categ, y=y_numeric, points="outliers")

# --- Nouvelles fonctions pour le Machine Learning ---

import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def plot_predictions_vs_actual(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot des valeurs réelles vs. prédites pour la régression."""
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Valeurs réelles", "y": "Valeurs prédites"},
        title="Réel vs. Prédit",
        trendline="ols",
        trendline_color_override="red",
    )
    fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(color="gray", dash="dash"))
    return fig

def plot_residuals_distribution(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Histogramme de la distribution des résidus."""
    residuals = y_true - y_pred
    fig = px.histogram(
        residuals,
        title="Distribution des résidus",
        labels={"value": "Résidus"},
    )
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    return fig

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, labels: List) -> go.Figure:
    """Matrice de confusion interactive."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matrice de confusion",
    )
    return fig

def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, labels: List) -> go.Figure:
    """Courbe ROC pour la classification binaire ou multi-classe (OvR)."""
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    if y_proba.shape[1] == 2: # Binaire
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=labels[1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Classe '{labels[1]}' (AUC={auc:.3f})", mode="lines"))
    else: # Multi-classe
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=label)
            auc = roc_auc_score((y_true == label), y_proba[:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Classe '{label}' (AUC={auc:.3f})", mode="lines"))

    fig.update_layout(
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        title="Courbe ROC",
    )
    return fig

def plot_feature_importance(pipeline, feature_names: List[str]) -> go.Figure:
    """Graphique de l'importance des features."""
    try:
        importances = pipeline.named_steps["model"].feature_importances_
    except AttributeError:
        try:
            importances = pipeline.named_steps["model"].coef_[0]
        except AttributeError:
            return go.Figure().update_layout(title="Importance des features non disponible pour ce modèle")

    feature_importance = pd.DataFrame({"feature": feature_names, "importance": np.abs(importances)}).sort_values("importance", ascending=True)
    
    fig = px.bar(
        feature_importance.tail(20), # Affiche les 20 plus importantes
        x="importance",
        y="feature",
        orientation="h",
        title="Importance des features",
    )
    return fig