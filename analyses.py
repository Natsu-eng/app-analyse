from typing import Dict, Tuple
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, tukey_hsd  # Ajouté pour post-hoc

@st.cache_data
def univariate_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
    ser = pd.to_numeric(df[column], errors="coerce")
    ser = ser.dropna()
    if ser.empty:
        return {}
    q = ser.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    shapiro_stat, shapiro_p = stats.shapiro(ser) if 3 <= len(ser) <= 5000 else (np.nan, np.nan)
    ks_stat, ks_p = stats.kstest((ser - ser.mean()) / ser.std(ddof=1), "norm") if ser.std(ddof=1) > 0 else (np.nan, np.nan)
    return {
        "moyenne": float(ser.mean()),
        "mediane": float(ser.median()),
        "ecart_type": float(ser.std(ddof=1)),
        "skewness": float(stats.skew(ser, bias=False)),
        "kurtosis": float(stats.kurtosis(ser, fisher=True, bias=False)),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else np.nan,
        "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "ks_stat": float(ks_stat) if not np.isnan(ks_stat) else np.nan,
        "ks_p": float(ks_p) if not np.isnan(ks_p) else np.nan,
    }

@st.cache_data
def correlations(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    xs = pd.to_numeric(df[x], errors="coerce").dropna()
    ys = pd.to_numeric(df[y], errors="coerce").dropna()
    aligned = pd.concat([xs, ys], axis=1).dropna()
    if aligned.shape[0] < 3:
        return {}
    pearson_r, pearson_p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    spearman_r, spearman_p = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    kendall_r, kendall_p = stats.kendalltau(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return {
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "kendall_r": float(kendall_r), "kendall_p": float(kendall_p),
    }

@st.cache_data
def contingency_and_tests(df: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    tab = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = chi2_contingency(tab)
    # Cramer's V and Tschuprow's T
    n = tab.values.sum()
    phi2 = chi2 / max(n, 1)
    r, k = tab.shape
    cramers_v = np.sqrt(phi2 / max(min(k - 1, r - 1), 1))
    tschuprow_t = np.sqrt(phi2 / np.sqrt(max((k - 1) * (r - 1), 1)))
    return {
        "chi2": float(chi2), "p_value": float(p), "ddl": int(dof),
        "cramers_v": float(cramers_v), "tschuprow_t": float(tschuprow_t),
        "n": int(n)
    }

@st.cache_data
def group_numeric_by_category(df: pd.DataFrame, cat: str, num: str) -> pd.DataFrame:
    grouped = df.groupby(cat)[num]
    return pd.DataFrame({
        "moyenne": grouped.mean(),
        "mediane": grouped.median(),
        "ecart_type": grouped.std(ddof=1),
        "taille": grouped.count(),
    })

@st.cache_data
def tests_cat_vs_num(df: pd.DataFrame, cat: str, num: str) -> Dict[str, float]:
    """Return appropriate test results depending on number of modalities.
    - 2 levels: Student and Wilcoxon
    - >2: ANOVA, Kruskal-Wallis, Tukey HSD post-hoc
    """
    clean = df[[cat, num]].dropna()
    if clean.empty:
        return {}
    levels = clean[cat].astype("category").cat.categories
    groups = [pd.to_numeric(clean.loc[clean[cat] == lvl, num], errors="coerce").dropna() for lvl in levels]
    res: Dict[str, float] = {}
    if len(groups) == 2:
        t_stat, t_p = stats.ttest_ind(groups[0], groups[1], equal_var=True)
        w_stat, w_p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        res.update({"t_stat": float(t_stat), "t_p": float(t_p), "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p)})
    elif len(groups) > 2:
        anova_stat, anova_p = stats.f_oneway(*groups)
        kw_stat, kw_p = stats.kruskal(*groups)
        res.update({
            "anova_stat": float(anova_stat), "anova_p": float(anova_p),
            "kruskal_stat": float(kw_stat), "kruskal_p": float(kw_p),
        })
        # Ajouté Tukey HSD post-hoc si ANOVA significatif
        if anova_p < 0.05:
            tukey_res = tukey_hsd(*groups)
            res["tukey_pvals"] = tukey_res.pvalue.tolist()  # Liste de p-values pairwise
    return res