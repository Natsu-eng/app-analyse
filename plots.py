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
