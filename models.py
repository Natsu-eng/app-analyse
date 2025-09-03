from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


def fit_linear_model(df: pd.DataFrame, y: str, X_cols: Tuple[str, ...]):
	clean = df[[y] + list(X_cols)].dropna()
	y_vec = clean[y].astype(float)
	X_mat = sm.add_constant(clean[list(X_cols)].astype(float), has_constant="add")
	model = sm.OLS(y_vec, X_mat).fit()
	return model


def regression_diagnostics(model, df: pd.DataFrame, y: str, X_cols: Tuple[str, ...]) -> Dict[str, float]:
	clean = df[[y] + list(X_cols)].dropna()
	y_vec = clean[y].astype(float)
	X_mat = sm.add_constant(clean[list(X_cols)].astype(float), has_constant="add")
	residuals = y_vec - model.predict(X_mat)
	# Normality
	shapiro_stat, shapiro_p = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 else (np.nan, np.nan)
	# Homoscedasticity
	bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_mat)
	# VIF
	vif = {col: float(variance_inflation_factor(X_mat.values, i)) for i, col in enumerate(["const"] + list(X_cols))}
	return {
		"shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else np.nan,
		"shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
		"breusch_pagan_stat": float(bp_stat),
		"breusch_pagan_p": float(bp_p),
		"vif_json": {k: float(v) for k, v in vif.items()},
	}


def predict_with_model(model, df_new: pd.DataFrame, X_cols: Tuple[str, ...]) -> pd.Series:
	X_mat = sm.add_constant(df_new[list(X_cols)].astype(float), has_constant="add")
	return model.predict(X_mat)
