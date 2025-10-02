import os
import streamlit as st
import pandas as pd
import time
import logging
import re
import psutil
import gc
import numpy as np
from functools import wraps
from typing import Dict, List, Any

from src.data.data_analysis import (
    compute_if_dask,
    is_dask_dataframe,
    auto_detect_column_types,
    detect_useless_columns,
    cleanup_memory
)
from src.evaluation.exploratory_plots import (
    plot_overview_metrics,
    plot_missing_values_overview,
    plot_cardinality_overview,
    plot_distribution,
    plot_bivariate_analysis,
    plot_correlation_heatmap
)

# Configuration du logging avec rotation
import logging.handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('logs/dashboard.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

from src.shared.logging import get_logger
logger = get_logger(__name__)

# Configuration Streamlit pour la production
st.set_page_config(
    page_title="Dashboard Analytics", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constantes de configuration
class Config:
    MAX_PREVIEW_ROWS = 100
    MAX_SAMPLE_SIZE = 15000
    MAX_BIVARIATE_SAMPLE = 10000
    MEMORY_CHECK_INTERVAL = 180  # 3 minutes
    CACHE_TTL = 600  # 10 minutes
    TIMEOUT_THRESHOLD = 30
    MEMORY_WARNING = 85
    MEMORY_CRITICAL = 90

# Configuration production
def setup_production_environment():
    """Configuration optimisée pour l'environnement de production"""
    if 'production_setup_done' not in st.session_state:
        st.session_state.production_setup_done = True
        
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            .stAlert > div {padding: 0.5rem;}
            .element-container {margin: 0.5rem 0;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_production_environment()

# Décorateur de monitoring avancé
def monitor_performance(operation_name: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if elapsed > 5:
                    logger.warning(f"SLOW: {operation_name or func.__name__} took {elapsed:.2f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"ERROR in {operation_name or func.__name__}: {str(e)} after {elapsed:.2f}s")
                raise
        return wrapper
    return decorator

st.title("📊 Dashboard Exploratoire")

# Gestion d'état centralisée et robuste
class StateManager:
    """Gestion centralisée de l'état du dashboard"""
    
    REQUIRED_KEYS = {
        'column_types': (dict, type(None)),
        'rename_list': list,
        'columns_to_drop': list,
        'useless_candidates': list,
        'dataset_hash': (str, type(None)),
        'last_memory_check': (int, float),
        'dashboard_version': int,
        'selected_univar_col': (str, type(None)),
        'selected_bivar_col1': (str, type(None)),
        'selected_bivar_col2': (str, type(None))
    }
    
    @classmethod
    def initialize(cls):
        """Initialise l'état avec validation"""
        defaults = {
            'column_types': None,
            'rename_list': [],
            'columns_to_drop': [],
            'useless_candidates': [],
            'dataset_hash': None,
            'last_memory_check': 0,
            'dashboard_version': 1,
            'selected_univar_col': None,
            'selected_bivar_col1': None,
            'selected_bivar_col2': None
        }
        
        for key, expected_type in cls.REQUIRED_KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = defaults[key]
            elif not isinstance(st.session_state[key], expected_type):
                logger.warning(f"Invalid type for {key}, resetting")
                st.session_state[key] = defaults[key]
    
    @classmethod
    def reset_selections(cls):
        """Reset les sélections pour éviter les erreurs"""
        selection_keys = ['selected_univar_col', 'selected_bivar_col1', 'selected_bivar_col2']
        for key in selection_keys:
            st.session_state[key] = None

StateManager.initialize()

# Monitoring système optimisé
class SystemMonitor:
    """Monitoring système avec seuils configurables"""
    
    @classmethod
    @monitor_performance("system_check")
    def check_resources(cls):
        """Vérifie les ressources système"""
        current_time = time.time()
        if current_time - st.session_state.last_memory_check > Config.MEMORY_CHECK_INTERVAL:
            try:
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                if memory_percent > Config.MEMORY_CRITICAL:
                    st.error(f"🚨 Mémoire critique: {memory_percent:.1f}%")
                    cls._emergency_cleanup()
                elif memory_percent > Config.MEMORY_WARNING:
                    st.warning(f"⚠️ Mémoire élevée: {memory_percent:.1f}%")
                    cls._show_cleanup_option()
                
                st.session_state.last_memory_check = current_time
                
            except Exception as e:
                logger.error(f"System check failed: {e}")
    
    @staticmethod
    def _emergency_cleanup():
        """Nettoyage d'urgence"""
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧹 Nettoyer", type="primary", key="emergency_clean"):
                cleanup_memory()
                st.cache_data.clear()
                gc.collect()
                st.success("Mémoire nettoyée")
                st.rerun()
    
    @staticmethod
    def _show_cleanup_option():
        """Option de nettoyage standard"""
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🧹", help="Nettoyer mémoire", key="normal_clean"):
                cleanup_memory()
                st.success("✅ Nettoyé")
                st.rerun()

SystemMonitor.check_resources()

# Validation des données
class DataValidator:
    """Validation et gestion des données"""
    
    @staticmethod
    @monitor_performance("validate_dataframe")
    def validate_dataframe() -> pd.DataFrame:
        """Valide le DataFrame avec gestion d'erreurs"""
        if 'df' not in st.session_state or st.session_state.df is None:
            st.error("📊 Aucun dataset chargé")
            st.info("Chargez un dataset depuis la page d'accueil pour commencer l'analyse.")
            if st.button("🏠 Retour à l'accueil"):
                st.switch_page("app.py")
            st.stop()
        
        df = st.session_state.df
        
        try:
            if hasattr(df, 'empty') and df.empty:
                st.error("Le dataset est vide")
                st.stop()
                
            if len(df.columns) == 0:
                st.error("Le dataset n'a pas de colonnes")
                st.stop()
                
            return df
            
        except Exception as e:
            logger.error(f"Dataframe validation error: {e}")
            st.error(f"Erreur de validation: {str(e)[:200]}")
            st.stop()
    
    @staticmethod
    def is_valid_column_name(name: str) -> bool:
        """Valide un nom de colonne"""
        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            return False
        
        name = name.strip()
        if len(name) > 50 or len(name) < 1:
            return False
        
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name))

df = DataValidator.validate_dataframe()

# Gestion du cache et hachage
@monitor_performance("dataset_hashing")
def get_dataset_hash(df) -> str:
    """Génère un hash stable du dataset"""
    try:
        if is_dask_dataframe(df):
            return f"dask_{hash(tuple(sorted(df.columns)))}_{df.npartitions}_{st.session_state.dashboard_version}"
        else:
            shape_hash = hash((df.shape[0], df.shape[1]))
            return f"pandas_{hash(tuple(sorted(df.columns)))}_{shape_hash}_{st.session_state.dashboard_version}"
    except Exception as e:
        logger.warning(f"Hash calculation fallback: {e}")
        return f"fallback_{int(time.time())}"

# Vérification du changement de dataset
current_hash = get_dataset_hash(df)
if st.session_state.dataset_hash != current_hash:
    logger.info(f"Dataset changed: {current_hash}")
    st.session_state.dataset_hash = current_hash
    st.session_state.column_types = None
    StateManager.reset_selections()

# Cache optimisé
@st.cache_data(ttl=Config.CACHE_TTL, max_entries=20, show_spinner=False)
@monitor_performance("global_metrics")
def compute_global_metrics(_df) -> Dict[str, Any]:
    """Calcule les métriques globales avec gestion robuste"""
    try:
        n_rows = compute_if_dask(_df.shape[0]) if hasattr(_df, 'shape') else len(_df)
        n_cols = _df.shape[1] if hasattr(_df, 'shape') else 0
        
        # Valeurs manquantes
        try:
            total_missing = compute_if_dask(_df.isna().sum().sum())
            missing_percentage = (total_missing / (n_rows * n_cols)) * 100 if (n_rows * n_cols) > 0 else 0
        except Exception:
            total_missing = 0
            missing_percentage = 0
        
        # Doublons
        try:
            duplicate_rows = compute_if_dask(_df.duplicated().sum())
        except Exception:
            duplicate_rows = 0
        
        # Mémoire
        memory_usage = "N/A"
        try:
            if not is_dask_dataframe(_df):
                memory_bytes = compute_if_dask(_df.memory_usage(deep=True).sum())
                memory_usage = memory_bytes / (1024**2)
            else:
                memory_usage = f"Dask ({_df.npartitions} partitions)"
        except Exception:
            pass
        
        return {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'memory_usage': memory_usage
        }
    except Exception as e:
        logger.error(f"Global metrics error: {e}")
        return {'n_rows': 0, 'n_cols': 0, 'missing_percentage': 0, 'duplicate_rows': 0, 'memory_usage': 'Error'}

@st.cache_data(ttl=Config.CACHE_TTL, max_entries=10)
@monitor_performance("column_detection")
def cached_auto_detect_column_types(_df) -> Dict[str, List]:
    """Cache la détection des types de colonnes"""
    try:
        result = auto_detect_column_types(_df)
        required_keys = ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']
        for key in required_keys:
            if key not in result:
                result[key] = []
        return result
    except Exception as e:
        logger.error(f"Column type detection failed: {e}")
        return {key: [] for key in ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']}

# Échantillonnage optimisé
class DataSampler:
    """Gestion optimisée de l'échantillonnage"""
    
    @staticmethod
    @monitor_performance("data_sampling")
    def get_sample(df, max_rows: int = Config.MAX_SAMPLE_SIZE, random_state: int = 42) -> pd.DataFrame:
        """Retourne un échantillon optimisé"""
        try:
            total_rows = compute_if_dask(df.shape[0])
            
            if total_rows <= max_rows:
                if is_dask_dataframe(df):
                    return compute_if_dask(df.head(max_rows))
                else:
                    return df.copy()
            
            sample_fraction = min(0.1, max_rows / total_rows)
            
            if is_dask_dataframe(df):
                sample = df.sample(frac=sample_fraction, random_state=random_state).head(max_rows)
                return compute_if_dask(sample)
            else:
                return df.sample(n=max_rows, random_state=random_state, replace=False)
                
        except Exception as e:
            logger.error(f"Sampling error: {e}")
            fallback_size = min(1000, total_rows) if 'total_rows' in locals() else 1000
            if is_dask_dataframe(df):
                return compute_if_dask(df.head(fallback_size))
            else:
                return df.head(fallback_size)

# Calculer les types de colonnes si nécessaire
if st.session_state.column_types is None:
    with st.spinner("🔍 Analyse des types de colonnes..."):
        st.session_state.column_types = cached_auto_detect_column_types(df)

column_types = st.session_state.column_types

# Vue d'ensemble
st.header("📋 Vue d'ensemble du jeu de données")

try:
    overview_metrics = compute_global_metrics(df)
    fig = plot_overview_metrics(overview_metrics)
    if fig:
        st.plotly_chart(fig, width='stretch', config={'responsive': True})
    else:
        st.info("📊 Métriques globales non disponibles")
except Exception as e:
    st.error(f"Erreur métriques: {str(e)[:100]}")

# Info colonnes optimisée
col_count = len(df.columns)
if col_count > 8:
    col_info = f"**{col_count} colonnes** : {', '.join(list(df.columns)[:6])}... +{col_count-6}"
else:
    col_info = f"**{col_count} colonnes** : {', '.join(list(df.columns))}"
st.info(col_info)

# Onglets avec gestion stable
tabs = st.tabs(["📈 Qualité", "🔬 Variables", "🔗 Relations", "🌐 Corrélations", "📄 Aperçu", "🗑️ Nettoyage"])

# Onglet 1: Qualité des données
with tabs[0]:
    st.subheader("📊 Qualité des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            missing_fig = plot_missing_values_overview(df)
            if missing_fig:
                st.plotly_chart(missing_fig, width='stretch', config={'responsive': True})
            else:
                st.success("✅ Aucune valeur manquante")
        except Exception as e:
            st.error("Erreur valeurs manquantes")
            logger.error(f"Missing values plot: {e}")
    
    with col2:
        try:
            cardinality_fig = plot_cardinality_overview(df, column_types)
            if cardinality_fig:
                st.plotly_chart(cardinality_fig, width='stretch', config={'responsive': True})
            else:
                st.info("📊 Cardinalité uniforme")
        except Exception as e:
            st.error("Erreur cardinalité")
            logger.error(f"Cardinality plot: {e}")

# Onglet 2: Analyse univariée
with tabs[1]:
    st.subheader("🔍 Analyse Univariée")
    
    available_columns = list(df.columns)
    if not available_columns:
        st.warning("Aucune colonne disponible")
    else:
        # Sélecteur stable avec état persistant
        if not st.session_state.selected_univar_col or st.session_state.selected_univar_col not in available_columns:
            st.session_state.selected_univar_col = available_columns[0]
        
        selected_col = st.selectbox(
            "Variable à analyser :",
            options=available_columns,
            index=available_columns.index(st.session_state.selected_univar_col),
            format_func=lambda x: f"{x} ({'Numérique' if x in column_types.get('numeric', []) else 'Catégorielle'})",
            key="univar_selector"
        )
        
        # Mise à jour de l'état seulement si nécessaire
        if selected_col != st.session_state.selected_univar_col:
            st.session_state.selected_univar_col = selected_col
        
        if selected_col:
            try:
                sample_df = DataSampler.get_sample(df)
                col_data = sample_df[selected_col].dropna()
                
                if col_data.empty:
                    st.warning(f"Aucune donnée valide pour **{selected_col}**")
                else:
                    if selected_col in column_types.get('numeric', []):
                        # Statistiques numériques
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            st.metric("Moyenne", f"{col_data.mean():.3f}")
                        with stats_cols[1]:
                            st.metric("Médiane", f"{col_data.median():.3f}")
                        with stats_cols[2]:
                            st.metric("Écart-type", f"{col_data.std():.3f}")
                        with stats_cols[3]:
                            st.metric("Uniques", f"{col_data.nunique():,}")
                        
                        # Graphique
                        with st.spinner("📊 Génération du graphique..."):
                            fig = plot_distribution(col_data, selected_col)
                            if fig:
                                st.plotly_chart(fig, width='stretch', config={'responsive': True})
                            else:
                                st.info("Graphique non disponible")
                    else:
                        # Variable catégorielle
                        value_counts = col_data.value_counts().head(20)
                        
                        if not value_counts.empty:
                            col_chart, col_table = st.columns([2, 1])
                            
                            with col_table:
                                df_display = value_counts.reset_index()
                                df_display.columns = ['Valeur', 'Count']
                                st.dataframe(df_display, height=400, width='stretch')
                            
                            with col_chart:
                                import plotly.express as px
                                fig = px.bar(
                                    x=value_counts.index.astype(str),
                                    y=value_counts.values,
                                    labels={'x': selected_col, 'y': 'Fréquence'},
                                    title=f"Distribution de {selected_col}"
                                )
                                fig.update_layout(template="plotly_white", height=400)
                                st.plotly_chart(fig, width='stretch', config={'responsive': True})
                        else:
                            st.info("Aucune donnée à afficher")
                            
            except Exception as e:
                st.error(f"Erreur analyse de {selected_col}")
                logger.error(f"Univariate analysis error for {selected_col}: {e}")

# Onglet 3: Relations bivariées
with tabs[2]:
    st.subheader("🔗 Relations entre Variables")
    
    available_columns = list(df.columns)
    
    if len(available_columns) >= 2:
        # États persistants pour sélections
        if not st.session_state.selected_bivar_col1 or st.session_state.selected_bivar_col1 not in available_columns:
            st.session_state.selected_bivar_col1 = available_columns[0]
        
        if not st.session_state.selected_bivar_col2 or st.session_state.selected_bivar_col2 not in available_columns:
            st.session_state.selected_bivar_col2 = available_columns[1] if len(available_columns) > 1 else available_columns[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox(
                "Variable 1",
                options=available_columns,
                index=available_columns.index(st.session_state.selected_bivar_col1),
                key="bivar_var1"
            )
            if var1 != st.session_state.selected_bivar_col1:
                st.session_state.selected_bivar_col1 = var1
        
        with col2:
            var2 = st.selectbox(
                "Variable 2",
                options=available_columns,
                index=available_columns.index(st.session_state.selected_bivar_col2),
                key="bivar_var2"
            )
            if var2 != st.session_state.selected_bivar_col2:
                st.session_state.selected_bivar_col2 = var2
        
        if var1 != var2:
            try:
                type1 = 'numeric' if var1 in column_types.get('numeric', []) else 'categorical'
                type2 = 'numeric' if var2 in column_types.get('numeric', []) else 'categorical'
                
                sample_df = DataSampler.get_sample(df, Config.MAX_BIVARIATE_SAMPLE)
                
                if not sample_df[[var1, var2]].empty:
                    with st.spinner("📊 Génération de l'analyse bivariée..."):
                        biv_fig = plot_bivariate_analysis(sample_df, var1, var2, type1, type2)
                        if biv_fig:
                            st.plotly_chart(biv_fig, width='stretch', config={'responsive': True})
                        else:
                            st.info("Graphique non disponible pour cette combinaison")
                else:
                    st.warning("Données insuffisantes")
                    
            except Exception as e:
                st.error("Erreur analyse bivariée")
                logger.error(f"Bivariate analysis error: {e}")
        else:
            st.warning("Sélectionnez deux variables différentes")
    else:
        st.warning("Au moins 2 colonnes nécessaires")

# Avant l'onglet qui suivent c'est une fonction de fallback pour afficher la corrélation 
# Plus simple si dans pltos/exploratory ça plante
import plotly.express as px
def create_simple_correlation_heatmap(df, max_cols=20):
    """Version ultra-simple du heatmap pour les cas problématiques"""
    try:
        # Prendre seulement les colonnes numériques
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) > max_cols:
            # Prendre les colonnes avec le moins de valeurs manquantes
            missing_rates = df[numeric_cols].isnull().mean()
            numeric_cols = missing_rates.nsmallest(max_cols).index.tolist()
        
        if len(numeric_cols) < 2:
            return None
            
        # Calcul de corrélation simple
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title=f"Matrice de corrélation ({len(numeric_cols)} variables)"
        )
        
        fig.update_layout(height=600)
        return fig, numeric_cols
        
    except Exception as e:
        logger.error(f"Simple heatmap failed: {e}")
        return None

# Onglet 4: Corrélations - VERSION CORRIGÉE
with tabs[3]:
    st.subheader("🌐 Matrice de Corrélations")
    
    # Options de configuration simplifiées
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        # Sélecteur de colonnes numériques seulement
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            target_col = st.selectbox(
                "Variable cible (optionnelle)", 
                options=[None] + numeric_cols,
                key="corr_target_select"
            )
        else:
            st.warning("Aucune variable numérique disponible")
            target_col = None
    
    with col_config2:
        # Option pour forcer le mode simple
        use_simple_mode = st.checkbox("Mode simple (recommandé)", value=True, key="simple_mode")
    
    if st.button("🔄 Générer la matrice", type="primary", key="generate_corr"):
        with st.spinner("📊 Calcul des corrélations..."):
            try:
                # Échantillonner pour les performances
                sample_df = DataSampler.get_sample(df, max_rows=3000)  # Réduit la taille
                
                if use_simple_mode:
                    # Utiliser la version simple
                    corr_fig, used_cols = create_simple_correlation_heatmap(sample_df, max_cols=15)
                else:
                    # Utiliser la version complète
                    corr_fig, used_cols = plot_correlation_heatmap(
                        sample_df, 
                        target_column=target_col,
                        task_type="classification" if target_col else None
                    )
                
                if corr_fig:
                    st.plotly_chart(corr_fig, width='stretch', config={'responsive': True})
                    st.success(f"✅ Matrice générée avec {len(used_cols)} variables")
                else:
                    st.warning("❌ Impossible de générer la matrice")
                    # Fallback ultime
                    st.info("Essayez avec le mode simple activé")
                    
            except Exception as e:
                st.error("Erreur lors du calcul des corrélations")
                logger.error(f"Correlation error: {e}")
                
                # Fallback direct avec Plotly Express
                try:
                    st.info("Tentative avec méthode alternative...")
                    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = sample_df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Pas assez de variables numériques")
                except Exception as fallback_e:
                    st.error("Échec de toutes les méthodes")

# Onglet 5: Aperçu données brutes - COMPLÉTÉ
with tabs[4]:
    st.subheader("📄 Aperçu des Données Brutes")
    
    try:
        # Utiliser df_raw si disponible, sinon df
        raw_df = st.session_state.get('df_raw', df)
        total_rows = compute_if_dask(raw_df.shape[0])
        
        # Configuration de l'aperçu
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            preview_size = st.slider(
                "Nombre de lignes à afficher",
                min_value=10,
                max_value=min(500, total_rows),
                value=min(Config.MAX_PREVIEW_ROWS, total_rows),
                key="preview_size_slider"
            )
        
        with col_config2:
            show_from_start = st.radio("Affichage", ["Début", "Échantillon"], key="preview_type")
        
        with col_config3:
            show_dtypes = st.checkbox("Afficher les types", value=False, key="show_dtypes")
        
        # Préparation des données à afficher
        if show_from_start == "Début":
            display_df = compute_if_dask(raw_df.head(preview_size))
        else:
            display_df = DataSampler.get_sample(raw_df, preview_size)
        
        # Gestion des colonnes avec beaucoup de texte
        display_df_truncated = display_df.copy()
        for col in display_df_truncated.select_dtypes(include=['object']).columns:
            display_df_truncated[col] = display_df_truncated[col].astype(str).apply(
                lambda x: x[:50] + "..." if len(str(x)) > 50 else x
            )
        
        # Affichage principal
        st.dataframe(
            display_df_truncated,
            height=400,
            width='stretch'
        )
        
        # Affichage des types de données si demandé
        if show_dtypes:
            with st.expander("📊 Types de données des colonnes"):
                dtypes_info = []
                for col in display_df.columns:
                    dtype = str(display_df[col].dtype)
                    non_null_count = display_df[col].count()
                    total_count = len(display_df[col])
                    null_percentage = ((total_count - non_null_count) / total_count * 100) if total_count > 0 else 0
                    
                    dtypes_info.append({
                        'Colonne': col,
                        'Type': dtype,
                        'Non-null': non_null_count,
                        'Null (%)': f"{null_percentage:.1f}%"
                    })
                
                dtypes_df = pd.DataFrame(dtypes_info)
                st.dataframe(dtypes_df, width='stretch')
        
        # Informations complémentaires
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.caption(f"📊 {len(display_df)} lignes affichées sur {total_rows:,} total")
        
        with info_col2:
            st.caption(f"📋 {len(display_df.columns)} colonnes")
        
        with info_col3:
            if len(display_df) > 0 and len(display_df.columns) > 0:
                missing_pct = (display_df.isnull().sum().sum() / (len(display_df) * len(display_df.columns))) * 100
                st.caption(f"🕳️ {missing_pct:.1f}% valeurs manquantes")
            else:
                st.caption("🕳️ Données insuffisantes")
        
        # Bouton de téléchargement de l'échantillon
        st.markdown("---")
        col_download1, col_download2 = st.columns([3, 1])
        
        with col_download2:
            if st.button("💾 Télécharger l'échantillon", key="download_sample"):
                try:
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"echantillon_donnees_{time.strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                except Exception as download_error:
                    st.error("❌ Erreur lors de la préparation du téléchargement")
                    logger.error(f"Download error: {download_error}")
    
    except Exception as e:
        st.error("❌ Erreur lors de l'affichage de l'aperçu des données")
        logger.error(f"Data preview error: {e}")
        
        # Fallback simple
        try:
            st.info("🔄 Tentative de chargement simplifié...")
            fallback_df = compute_if_dask(df.head(50))
            st.dataframe(fallback_df, height=300, width='stretch')
            st.caption(f"📊 Affichage de secours: 50 premières lignes sur {compute_if_dask(df.shape[0]):,} total")
        except Exception as fallback_error:
            st.error("🚨 Impossible d'afficher les données")
            logger.error(f"Fallback preview also failed: {fallback_error}")

# Onglet 6: Nettoyage des données 
with tabs[5]:
    st.subheader("🗑️ Nettoyage des Données")
    
    st.markdown("### 🔍 Détection des colonnes inutiles")
    
    # Choix du mode : automatique ou manuel
    mode = st.radio(
        "Mode de sélection des colonnes à supprimer :",
        options=["Automatique", "Manuelle"],
        horizontal=True
    )
    
    cols_to_remove = []

    # --- Mode automatique ---
    if mode == "Automatique":
        col_detect, col_action = st.columns([2, 1])
        with col_detect:
            if st.button("🔎 Analyser les colonnes inutiles", key="analyze_useless"):
                with st.spinner("Analyse en cours..."):
                    try:
                        useless_cols = detect_useless_columns(df, threshold_missing=0.7)
                        st.session_state.useless_candidates = useless_cols
                        
                        if useless_cols:
                            st.success(f"✅ {len(useless_cols)} colonne(s) potentiellement inutile(s) détectée(s)")
                            st.write(
                                "**Colonnes détectées:**",
                                ", ".join(useless_cols[:5]) + ("..." if len(useless_cols) > 5 else "")
                            )
                            cols_to_remove = useless_cols
                        else:
                            st.info("🎉 Aucune colonne inutile détectée")
                    except Exception as e:
                        st.error("❌ Erreur lors de l'analyse")
                        logger.error(f"Useless columns detection error: {e}")
        
        with col_action:
            if st.session_state.useless_candidates:
                if st.button("🗑️ Supprimer les colonnes inutiles", type="primary"):
                    try:
                        valid_cols = [col for col in st.session_state.useless_candidates if col in df.columns]
                        if valid_cols:
                            if is_dask_dataframe(df):
                                new_df = df.drop(columns=valid_cols).persist()
                            else:
                                new_df = df.drop(columns=valid_cols)
                            st.session_state.df = new_df
                            st.session_state.useless_candidates = []
                            st.session_state.column_types = None
                            st.session_state.dashboard_version += 1
                            st.success(f"✅ {len(valid_cols)} colonne(s) supprimée(s)")
                            st.rerun()
                        else:
                            st.warning("⚠️ Aucune colonne valide à supprimer")
                    except Exception as e:
                        st.error("❌ Erreur lors de la suppression")
                        logger.error(f"Column removal error: {e}")

    # --- Mode manuel ---
    else:
        all_cols = df.columns.tolist()
        cols_to_remove = st.multiselect(
            "Sélectionnez les colonnes à supprimer",
            options=all_cols,
            default=[]
        )
        if cols_to_remove:
            if st.button("🗑️ Supprimer les colonnes sélectionnées", type="primary"):
                try:
                    valid_cols = [col for col in cols_to_remove if col in df.columns]
                    if valid_cols:
                        if is_dask_dataframe(df):
                            new_df = df.drop(columns=valid_cols).persist()
                        else:
                            new_df = df.drop(columns=valid_cols)
                        st.session_state.df = new_df
                        st.session_state.column_types = None
                        st.session_state.dashboard_version += 1
                        st.success(f"✅ {len(valid_cols)} colonne(s) supprimée(s)")
                        st.rerun()
                    else:
                        st.warning("⚠️ Aucune colonne valide à supprimer")
                except Exception as e:
                    st.error("❌ Erreur lors de la suppression")
                    logger.error(f"Manual column removal error: {e}")

    st.markdown("---")
    st.markdown("### ✏️ Renommage des colonnes")
    
    col_rename1, col_rename2 = st.columns(2)
    
    with col_rename1:
        if df.columns.tolist():
            col_to_rename = st.selectbox(
                "Colonne à renommer",
                options=df.columns.tolist(),
                key="rename_select"
            )
        else:
            col_to_rename = None
            st.warning("Aucune colonne disponible")
    
    with col_rename2:
        new_name = st.text_input(
            "Nouveau nom",
            placeholder="Nouveau nom de colonne",
            key="rename_input"
        )
    
    col_add, col_clear = st.columns(2)
    
    with col_add:
        if st.button("➕ Ajouter au plan de renommage", key="add_rename"):
            if col_to_rename and new_name and DataValidator.is_valid_column_name(new_name):
                if new_name not in df.columns:
                    if (col_to_rename, new_name) not in st.session_state.rename_list:
                        st.session_state.rename_list.append((col_to_rename, new_name))
                        st.success(f"✅ {col_to_rename} → {new_name} ajouté")
                    else:
                        st.warning("⚠️ Ce renommage est déjà planifié")
                else:
                    st.error("❌ Ce nom de colonne existe déjà")
            else:
                st.error("❌ Nom invalide ou vide")
    
    with col_clear:
        if st.button("🗑️ Vider la liste", key="clear_renames"):
            st.session_state.rename_list = []
            st.success("✅ Liste vidée")
    
    # Affichage des renommages planifiés
    if st.session_state.rename_list:
        st.markdown("**📋 Renommages planifiés:**")
        rename_df = pd.DataFrame(st.session_state.rename_list, columns=["Ancien nom", "Nouveau nom"])
        st.dataframe(rename_df, width='stretch')
        
        if st.button("✅ Appliquer tous les renommages", type="primary", key="apply_renames"):
            try:
                rename_dict = dict(st.session_state.rename_list)
                valid_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
                
                if valid_renames:
                    if is_dask_dataframe(df):
                        new_df = df.rename(columns=valid_renames).persist()
                    else:
                        new_df = df.rename(columns=valid_renames)
                    st.session_state.df = new_df
                    st.session_state.rename_list = []
                    st.session_state.column_types = None
                    st.session_state.dashboard_version += 1
                    st.success(f"✅ {len(valid_renames)} colonne(s) renommée(s)")
                    st.rerun()
                else:
                    st.warning("⚠️ Aucun renommage valide à appliquer")
            except Exception as e:
                st.error("❌ Erreur lors du renommage")
                logger.error(f"Rename error: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Actions de maintenance")
    
    col_maint1, col_maint2 = st.columns(2)
    
    with col_maint1:
        if st.button("🧹 Nettoyer la mémoire", key="cleanup_mem"):
            try:
                cleanup_memory()
                st.success("✅ Mémoire nettoyée")
            except Exception as e:
                st.error("❌ Erreur de nettoyage")
    
    with col_maint2:
        if st.button("🔄 Rafraîchir l'analyse", key="refresh_analysis"):
            try:
                st.session_state.column_types = None
                st.cache_data.clear()
                st.success("✅ Analyse rafraîchie")
                st.rerun()
            except Exception as e:
                st.error("❌ Erreur de rafraîchissement")

# Footer final avec informations système
st.markdown("---")
footer_cols = st.columns(4)

with footer_cols[0]:
    try:
        n_rows = compute_if_dask(df.shape[0])
        n_cols = df.shape[1]
        st.caption(f"📊 {n_rows:,} × {n_cols}")
    except:
        st.caption("📊 Données non disponibles")

with footer_cols[1]:
    try:
        if not is_dask_dataframe(df):
            memory_mb = compute_if_dask(df.memory_usage(deep=True).sum()) / (1024**2)
            st.caption(f"💾 {memory_mb:.1f} MB")
        else:
            st.caption(f"💾 {df.npartitions} partitions")
    except:
        st.caption("💾 N/A")

with footer_cols[2]:
    try:
        sys_mem = psutil.virtual_memory().percent
        status = "🔴" if sys_mem > Config.MEMORY_CRITICAL else "🟡" if sys_mem > Config.MEMORY_WARNING else "🟢"
        st.caption(f"{status} {sys_mem:.0f}% RAM")
    except:
        st.caption("🔧 RAM: N/A")

with footer_cols[3]:
    st.caption(f"🕒 {time.strftime('%H:%M:%S')}")

# Nettoyage final
gc.collect()