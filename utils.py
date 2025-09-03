import io
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"}


def _infer_sep(file_name: str) -> str:
	lower = file_name.lower()
	if lower.endswith(".tsv"):
		return "\t"
	return ","


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
	"""Load a dataframe from raw bytes using extension-based parsing. Cached by Streamlit."""
	name = file_name.lower()
	if name.endswith((".csv", ".tsv", ".txt")):
		sep = _infer_sep(name)
		return pd.read_csv(io.BytesIO(file_bytes), sep=sep, low_memory=False)
	if name.endswith((".xlsx", ".xls")):
		return pd.read_excel(io.BytesIO(file_bytes))
	if name.endswith(".parquet"):
		return pd.read_parquet(io.BytesIO(file_bytes))
	if name.endswith(".json"):
		return pd.read_json(io.BytesIO(file_bytes), lines=False)
	raise ValueError("Extension de fichier non supporte. Formats: CSV/TSV/TXT, Excel, Parquet, JSON")


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Return missingness summary per column."""
	missing_count = df.isna().sum()
	missing_pct = (missing_count / len(df)) * 100 if len(df) else 0
	return pd.DataFrame({
		"colonne": missing_count.index,
		"nb_nan": missing_count.values,
		"pct_nan": missing_pct.values,
	})


def handle_missing_values(
	df: pd.DataFrame,
	strategy: str = "aucun",
	fill_constant: Optional[float] = None,
	columns: Optional[List[str]] = None,
) -> pd.DataFrame:
	"""Handle missing values with different strategies.

	- "aucun": no change
	- "supprimer_lignes": drop rows with any NaN in selected columns
	- "remplir_moyenne": fill numeric cols with mean
	- "remplir_mediane": fill numeric cols with median
	- "remplir_mode": fill each column with its mode
	- "remplir_constante": fill all selected columns with a constant
	"""
	if columns is None:
		columns = list(df.columns)

	result = df.copy()
	subset = [c for c in columns if c in result.columns]
	if strategy == "aucun":
		return result
	if strategy == "supprimer_lignes":
		return result.dropna(subset=subset)
	if strategy == "remplir_moyenne":
		numeric_cols = [c for c in subset if pd.api.types.is_numeric_dtype(result[c])]
		for c in numeric_cols:
			result[c] = result[c].fillna(result[c].mean())
		return result
	if strategy == "remplir_mediane":
		numeric_cols = [c for c in subset if pd.api.types.is_numeric_dtype(result[c])]
		for c in numeric_cols:
			result[c] = result[c].fillna(result[c].median())
		return result
	if strategy == "remplir_mode":
		for c in subset:
			mode_vals = result[c].mode(dropna=True)
			if not mode_vals.empty:
				result[c] = result[c].fillna(mode_vals.iloc[0])
		return result
	if strategy == "remplir_constante":
		if fill_constant is None:
			raise ValueError("Veuillez fournir une valeur constante pour le remplissage.")
		for c in subset:
			result[c] = result[c].fillna(fill_constant)
		return result

	raise ValueError("Stratégie de gestion des valeurs manquantes inconnue")


def auto_detect_variable_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	"""Return (qualitative_columns, quantitative_columns)."""
	qualitative_cols: List[str] = []
	quantitative_cols: List[str] = []
	for col in df.columns:
		series = df[col]
		if pd.api.types.is_numeric_dtype(series):
			# Numeric but low cardinality could be categorical, keep numeric here
			quantitative_cols.append(col)
		else:
			qualitative_cols.append(col)
	return qualitative_cols, quantitative_cols
