import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio

from utils import load_data, summarize_missing_values, handle_missing_values, auto_detect_variable_types
from plots import barplot_counts, pie_proportions, histogram_with_density, boxplot, qqplot, scatterplot, boxplot_bivariate
from analyses import univariate_stats, correlations, contingency_and_tests, group_numeric_by_category, tests_cat_vs_num
from models import fit_linear_model, regression_diagnostics, predict_with_model

st.set_page_config(page_title="Data Science App", layout="wide")
pio.templates.default = "plotly_white"

# ===== Header =====
st.markdown("""
<style>
.kpi-card {background: #FFFFFF; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb;}
.kpi-title {font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em;}
.kpi-value {font-size: 22px; font-weight: 700; color: #111827;}
.section-title {margin-top: 0;}
</style>
""", unsafe_allow_html=True)

st.title("Data Science")
st.caption("Import, EDA, Modélisation et Prédictions — propulsé par Streamlit & Plotly")

with st.sidebar:
	st.header("1. Import des données")
	upload = st.file_uploader("Fichier (CSV, Excel, Parquet, JSON)", type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"]) 
	st.divider()
	st.header("2. Valeurs manquantes")
	missing_strategy = st.selectbox("Stratégie", [
		"aucun", "supprimer_lignes", "remplir_moyenne", "remplir_mediane", "remplir_mode", "remplir_constante"
	])
	fill_value = None
	if missing_strategy == "remplir_constante":
		fill_value = st.number_input("Valeur constante", value=0.0)

if upload is None:
	st.info("Importez un fichier pour démarrer l’analyse.")
	st.stop()

try:
	df_raw = load_data(upload.getvalue(), upload.name)
except Exception as e:
	st.error(f"Erreur de chargement: {e}")
	st.stop()

# KPI row
missing_summary = summarize_missing_values(df_raw)
nb_rows, nb_cols = df_raw.shape
nb_missing_cells = int(missing_summary["nb_nan"].sum())
memory_mb = float(df_raw.memory_usage(deep=True).sum()) / (1024 ** 2)

c1, c2, c3, c4 = st.columns(4)
with c1:
	st.markdown('<div class="kpi-card"><div class="kpi-title">Lignes</div><div class="kpi-value">{:,}</div></div>'.format(nb_rows), unsafe_allow_html=True)
with c2:
	st.markdown('<div class="kpi-card"><div class="kpi-title">Colonnes</div><div class="kpi-value">{:,}</div></div>'.format(nb_cols), unsafe_allow_html=True)
with c3:
	st.markdown('<div class="kpi-card"><div class="kpi-title">Cellules manquantes</div><div class="kpi-value">{:,}</div></div>'.format(nb_missing_cells), unsafe_allow_html=True)
with c4:
	st.markdown('<div class="kpi-card"><div class="kpi-title">Mémoire</div><div class="kpi-value">{:.2f} Mo</div></div>'.format(memory_mb), unsafe_allow_html=True)

# Manage missing
df = handle_missing_values(df_raw, strategy=missing_strategy, fill_constant=fill_value, columns=list(df_raw.columns))
qual_cols, quant_cols = auto_detect_variable_types(df)

# Tabs
onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs([
	"Aperçu", "Univarié", "Bivarié", "Régression linéaire", "Prédictions"
])

with onglet1:
	st.subheader("Aperçu des données")
	st.write("Aperçu (100 premières lignes):")
	st.dataframe(df.head(100), use_container_width=True)
	st.write("Valeurs manquantes (par colonne):")
	st.dataframe(missing_summary, use_container_width=True)

with onglet2:
	st.subheader("Analyse univariée")
	col = st.selectbox("Variable", options=list(df.columns))
	if col:
		if col in quant_cols:
			stats_dict = univariate_stats(df, col)
			mc1, mc2 = st.columns([1,1])
			with mc1:
				st.metric("Moyenne", f"{stats_dict.get('moyenne', np.nan):.3f}")
				st.metric("Écart-type", f"{stats_dict.get('ecart_type', np.nan):.3f}")
			with mc2:
				st.metric("Médiane", f"{stats_dict.get('mediane', np.nan):.3f}")
				st.metric("Skewness", f"{stats_dict.get('skewness', np.nan):.3f}")
			c1, c2 = st.columns(2)
			with c1:
				st.plotly_chart(histogram_with_density(df, col), use_container_width=True)
				st.plotly_chart(boxplot(df, col), use_container_width=True)
			with c2:
				st.plotly_chart(qqplot(df, col), use_container_width=True)
		else:
			c1, c2 = st.columns(2)
			with c1:
				st.plotly_chart(barplot_counts(df, col), use_container_width=True)
			with c2:
				st.plotly_chart(pie_proportions(df, col), use_container_width=True)

with onglet3:
	st.subheader("Analyses bivariées")
	left, right = st.columns(2)
	with left:
		x = st.selectbox("Variable X", options=list(df.columns), index=0)
	with right:
		y = st.selectbox("Variable Y", options=list(df.columns), index=1 if len(df.columns) > 1 else 0)
	if x and y and x != y:
		if x in quant_cols and y in quant_cols:
			st.plotly_chart(scatterplot(df, x, y), use_container_width=True)
			st.expander("Coefficients de corrélation").json(correlations(df, x, y))
		elif x in qual_cols and y in qual_cols:
			res = contingency_and_tests(df, x, y)
			st.expander("Tableau et tests").json(res)
			# Show contingency table
			st.dataframe(pd.crosstab(df[x], df[y]), use_container_width=True)
		else:
			cat = x if x in qual_cols else y
			num = y if x in qual_cols else x
			st.plotly_chart(boxplot_bivariate(df, cat, num), use_container_width=True)
			st.dataframe(group_numeric_by_category(df, cat, num), use_container_width=True)
			st.expander("Tests").json(tests_cat_vs_num(df, cat, num))

with onglet4:
	st.subheader("Régression linéaire")
	if not quant_cols:
		st.info("Aucune variable quantitative disponible.")
	else:
		y = st.selectbox("Variable cible (Y)", options=quant_cols)
		X_candidates = [c for c in quant_cols if c != y]
		X_sel = st.multiselect("Variables explicatives (X)", options=X_candidates, default=X_candidates[:1])
		if y and X_sel:
			model = fit_linear_model(df, y, tuple(X_sel))
			st.code(model.summary().as_text())
			diag = regression_diagnostics(model, df, y, tuple(X_sel))
			m1, m2 = st.columns(2)
			with m1:
				st.metric("BP p-value", f"{diag.get('breusch_pagan_p', np.nan):.3f}")
				st.metric("Shapiro p-value", f"{diag.get('shapiro_p', np.nan):.3f}")
			with m2:
				st.write("VIF:")
				st.json(diag.get("vif_json", {}))

with onglet5:
	st.subheader("Prédictions")
	st.write("Utilisez le modèle entraîné pour prédire de nouvelles données.")
	if 'model_state' not in st.session_state:
		st.session_state['model_state'] = {}
	if 'last_model' not in st.session_state['model_state']:
		st.session_state['model_state']['last_model'] = None
		st.session_state['model_state']['X_cols'] = []
	if 'model' in locals():
		st.session_state['model_state']['last_model'] = model
		st.session_state['model_state']['X_cols'] = tuple(X_sel)

	model_obj = st.session_state['model_state']['last_model']
	X_cols = st.session_state['model_state']['X_cols']
	if model_obj is None:
		st.info("Ajustez un modèle dans l'onglet Régression pour activer les prédictions.")
	else:
		upload_new = st.file_uploader("Nouvelles données (mêmes variables X)", type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"], key="newdata")
		if upload_new is not None:
			try:
				df_new = load_data(upload_new.getvalue(), upload_new.name)
				y_pred = predict_with_model(model_obj, df_new, tuple(X_cols))
				res = df_new.copy()
				res["prediction"] = y_pred
				st.dataframe(res.head(100), use_container_width=True)
			except Exception as e:
				st.error(f"Erreur de prédiction: {e}")
