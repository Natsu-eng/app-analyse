import os
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import time
import logging
import re
import psutil
import gc
from functools import wraps
from utils.data_analysis import (
    compute_if_dask,
    is_dask_dataframe,
    auto_detect_column_types,
    detect_useless_columns,
    get_data_profile,
    cleanup_memory
)
from plots.exploratory import (
    plot_overview_metrics,
    plot_missing_values_overview,
    plot_cardinality_overview,
    plot_distribution,
    plot_bivariate_analysis
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Streamlit
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Configuration production
def setup_production_dashboard():
    """Configuration pour l'environnement de production du dashboard"""
    if os.getenv('STREAMLIT_ENV') == 'production':
        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        .stAlert > div {padding: 0.5rem;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_production_dashboard()

# Décorateur de monitoring pour les fonctions critiques
def monitor_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 5:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

st.title("📊 Dashboard Exploratoire")

# Initialisation stable des états avec pattern défensif
def initialize_dashboard_state():
    """Initialise les variables de session de manière défensive et stable"""
    defaults = {
        'column_types': None,
        'rename_list': [],
        'columns_to_drop': [],
        'useless_candidates': [],
        'dataset_hash': None,
        'last_memory_check': 0,
        'dashboard_cache_version': 1
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_dashboard_state()

# Monitoring système périodique
def check_system_resources():
    """Vérifie les ressources système de manière non-bloquante"""
    current_time = time.time()
    if current_time - st.session_state.last_memory_check > 120:  # Check tous les 2 minutes
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                st.warning(f"⚠️ Utilisation mémoire élevée: {memory_percent:.1f}%")
                col_warn, col_clean = st.columns([3, 1])
                with col_clean:
                    if st.button("🧹 Libérer", help="Nettoyer la mémoire"):
                        cleanup_memory()
                        st.success("✅ Nettoyé")
                        st.rerun()
            st.session_state.last_memory_check = current_time
        except Exception as e:
            logger.error(f"System check failed: {e}")

check_system_resources()

# Validation robuste des noms de colonnes
def is_valid_column_name(name: str) -> bool:
    """Vérifie la validité d'un nom de colonne avec validation stricte"""
    if not name or not isinstance(name, str) or len(name.strip()) == 0:
        return False
    name = name.strip()
    # Autoriser lettres, chiffres, underscores, et tirets
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', name)) and len(name) <= 50

# Vérification et validation du DataFrame
@monitor_execution
def validate_dataframe():
    """Valide la présence et l'intégrité du DataFrame"""
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("📊 Aucun dataset chargé")
        st.info("Veuillez d'abord charger un jeu de données depuis la page d'accueil.")
        if st.button("🏠 Retour à l'accueil"):
            st.switch_page("app.py")
        st.stop()
    
    df = st.session_state.df
    
    # Vérifications de base
    try:
        if hasattr(df, 'empty') and df.empty:
            st.error("Le dataset est vide")
            st.stop()
        if len(df.columns) == 0:
            st.error("Le dataset n'a pas de colonnes")
            st.stop()
    except Exception as e:
        st.error(f"Erreur de validation du dataset: {e}")
        st.stop()
    
    return df

df = validate_dataframe()

# Fonction de hash stable et robuste
@monitor_execution
def get_dataset_hash(df):
    """Génère un hash stable basé sur la structure du dataset"""
    try:
        if is_dask_dataframe(df):
            return f"dask_{hash(tuple(df.columns))}_{df.npartitions}_{st.session_state.dashboard_cache_version}"
        else:
            return f"pandas_{hash(tuple(df.columns))}_{df.shape[0]}_{st.session_state.dashboard_cache_version}"
    except Exception as e:
        logger.error(f"Hash calculation error: {e}")
        return f"fallback_{time.time()}"

# Vérification du changement de dataset avec mécanisme de cache intelligent
current_hash = get_dataset_hash(df)
if st.session_state.dataset_hash != current_hash:
    st.session_state.dataset_hash = current_hash
    st.session_state.column_types = None
    logger.info(f"Dataset changed, new hash: {current_hash}")

# Cache optimisé pour les métriques globales
@st.cache_data(ttl=300, max_entries=10)
@monitor_execution
def compute_global_metrics(_df):
    """Calcule les métriques globales avec gestion d'erreurs robuste"""
    try:
        start_time = time.time()
        
        # Calculs de base avec fallbacks
        try:
            n_rows = compute_if_dask(_df.shape[0])
        except Exception:
            n_rows = len(_df) if hasattr(_df, '__len__') else 0
            
        n_cols = _df.shape[1] if hasattr(_df, 'shape') else 0
        
        # Valeurs manquantes avec gestion d'erreur
        try:
            total_missing = compute_if_dask(_df.isna().sum().sum())
            missing_percentage = (total_missing / (n_rows * n_cols)) * 100 if (n_rows * n_cols) > 0 else 0
        except Exception as e:
            logger.warning(f"Missing values calculation failed: {e}")
            total_missing = 0
            missing_percentage = 0
        
        # Doublons avec gestion d'erreur
        try:
            duplicate_rows = compute_if_dask(_df.duplicated().sum())
        except Exception as e:
            logger.warning(f"Duplicates calculation failed: {e}")
            duplicate_rows = 0
        
        # Usage mémoire
        memory_usage = "N/A"
        try:
            if not is_dask_dataframe(_df):
                memory_usage = compute_if_dask(_df.memory_usage(deep=True).sum()) / (1024**2)
            else:
                memory_usage = f"Dask ({_df.npartitions} partitions)"
        except Exception as e:
            logger.debug(f"Memory calculation failed: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Global metrics computed in {elapsed_time:.2f}s")
        
        return {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'memory_usage': memory_usage
        }
    except Exception as e:
        logger.error(f"Critical error in compute_global_metrics: {e}")
        return {
            'n_rows': 0, 'n_cols': 0, 'missing_percentage': 0,
            'duplicate_rows': 0, 'memory_usage': 'Error'
        }

# Cache pour la détection des types de colonnes
@st.cache_data(ttl=300, max_entries=5)
@monitor_execution
def cached_auto_detect_column_types(_df):
    """Cache la détection des types avec fallback sécurisé"""
    try:
        result = auto_detect_column_types(_df)
        # Validation du résultat
        required_keys = ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']
        for key in required_keys:
            if key not in result:
                result[key] = []
        return result
    except Exception as e:
        logger.error(f"Column type detection failed: {e}")
        return {'numeric': [], 'categorical': [], 'text_or_high_cardinality': [], 'datetime': []}

# Calculer ou récupérer les types de colonnes
if st.session_state.column_types is None:
    with st.spinner("🔍 Analyse des types de colonnes..."):
        st.session_state.column_types = cached_auto_detect_column_types(df)

column_types = st.session_state.column_types

# Validation et nettoyage des types de colonnes
required_keys = ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']
for key in required_keys:
    if key not in column_types or not isinstance(column_types[key], list):
        column_types[key] = []

# Vue d'ensemble avec gestion d'erreurs
st.header("📋 Vue d'ensemble du jeu de données")

try:
    overview_metrics = compute_global_metrics(df)
    fig = plot_overview_metrics(overview_metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Graphique des métriques non disponible")
except Exception as e:
    st.error(f"Erreur lors du calcul des métriques: {e}")
    logger.error(f"Overview metrics error: {e}")

# Information sur les colonnes avec truncature pour l'affichage
col_display = list(df.columns)
if len(col_display) > 10:
    col_info = f"Colonnes ({len(col_display)}): {', '.join(col_display[:8])}, ... +{len(col_display)-8} autres"
else:
    col_info = f"Colonnes ({len(col_display)}): {', '.join(col_display)}"
st.info(col_info)

# Onglets avec clés stables et uniques
tab_overview, tab_univariate, tab_bivariate, tab_preview, tab_cleaning = st.tabs([
    "📈 Qualité", "🔬 Variables", "🔗 Relations", "📄 Aperçu", "🗑️ Nettoyage"
])

# Onglet Qualité des données
with tab_overview:
    st.subheader("📊 Valeurs Manquantes et Cardinalité")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            missing_fig = plot_missing_values_overview(df)
            if missing_fig:
                st.plotly_chart(missing_fig, use_container_width=True)
            else:
                st.info("✅ Aucune valeur manquante détectée")
        except Exception as e:
            st.warning(f"Erreur valeurs manquantes: {str(e)[:100]}")
            logger.error(f"Missing values plot error: {e}")
    
    with col2:
        try:
            cardinality_fig = plot_cardinality_overview(df, column_types)
            if cardinality_fig:
                st.plotly_chart(cardinality_fig, use_container_width=True)
            else:
                st.info("📊 Cardinalité uniforme")
        except Exception as e:
            st.warning(f"Erreur cardinalité: {str(e)[:100]}")
            logger.error(f"Cardinality plot error: {e}")

# Onglet Analyse univariée avec sélection stable
with tab_univariate:
    st.subheader("🔍 Analyse d'une Variable")
    available_columns = list(df.columns)
    
    if available_columns:
        # Utilisation d'un sélecteur stable basé sur l'index
        try:
            # État persistent pour la sélection
            if 'selected_col_index_univar' not in st.session_state:
                st.session_state.selected_col_index_univar = 0
            
            selected_index = st.selectbox(
                "Variable à analyser",
                options=range(len(available_columns)),
                format_func=lambda x: f"{available_columns[x]} ({'num' if available_columns[x] in column_types.get('numeric', []) else 'cat'})",
                index=st.session_state.selected_col_index_univar,
                key="univar_selector"
            )
            
            # Mise à jour de l'état seulement si changement
            if selected_index != st.session_state.selected_col_index_univar:
                st.session_state.selected_col_index_univar = selected_index
            
            selected_col = available_columns[selected_index]
            
            if selected_col in df.columns:
                col_type = 'numeric' if selected_col in column_types.get('numeric', []) else 'categorical'
                
                # Échantillonnage sécurisé et optimisé
                try:
                    max_sample = min(30000, compute_if_dask(df.shape[0]))
                    
                    if is_dask_dataframe(df):
                        sample_df = df.sample(frac=min(0.1, max_sample / compute_if_dask(df.shape[0]))).head(max_sample)
                        sample_df = compute_if_dask(sample_df)
                    else:
                        sample_df = df.head(max_sample) if len(df) <= max_sample else df.sample(n=max_sample, random_state=42)
                    
                    if col_type == 'numeric':
                        col_data = sample_df[selected_col].dropna()
                        if len(col_data) == 0:
                            st.warning(f"❌ Aucune donnée valide pour {selected_col}")
                        else:
                            # Statistiques rapides
                            stats_col1, stats_col2, stats_col3 = st.columns(3)
                            with stats_col1:
                                st.metric("Moyenne", f"{col_data.mean():.3f}")
                            with stats_col2:
                                st.metric("Médiane", f"{col_data.median():.3f}")
                            with stats_col3:
                                st.metric("Écart-type", f"{col_data.std():.3f}")
                            
                            # Graphique de distribution
                            fig = plot_distribution(col_data, selected_col)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f"**Variable catégorielle**: `{selected_col}`")
                        value_counts = sample_df[selected_col].value_counts().head(15)
                        if not value_counts.empty:
                            df_display = value_counts.reset_index()
                            df_display.columns = ['Catégorie', 'Comptage']
                            st.dataframe(df_display, use_container_width=True)
                        else:
                            st.info("Aucune donnée valide")
                            
                except Exception as e:
                    st.error(f"Erreur analyse univariée: {str(e)[:100]}")
                    logger.error(f"Univariate analysis error: {e}")
        except Exception as e:
            st.error(f"Erreur sélection variable: {e}")
    else:
        st.warning("Aucune colonne disponible")

# Onglet Analyse bivariée avec sélecteurs stables
with tab_bivariate:
    st.subheader("🔗 Relations entre Variables")
    available_columns = list(df.columns)
    
    if len(available_columns) >= 2:
        col1, col2 = st.columns(2)
        
        # États persistants pour les sélections bivariées
        if 'bivar_var1_idx' not in st.session_state:
            st.session_state.bivar_var1_idx = 0
        if 'bivar_var2_idx' not in st.session_state:
            st.session_state.bivar_var2_idx = min(1, len(available_columns) - 1)
        
        with col1:
            var1_idx = st.selectbox(
                "Variable 1",
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                index=st.session_state.bivar_var1_idx,
                key="bivar_var1"
            )
            if var1_idx != st.session_state.bivar_var1_idx:
                st.session_state.bivar_var1_idx = var1_idx
            var1 = available_columns[var1_idx]
        
        with col2:
            var2_idx = st.selectbox(
                "Variable 2",
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                index=st.session_state.bivar_var2_idx,
                key="bivar_var2"
            )
            if var2_idx != st.session_state.bivar_var2_idx:
                st.session_state.bivar_var2_idx = var2_idx
            var2 = available_columns[var2_idx]
        
        if var1 != var2 and var1 in df.columns and var2 in df.columns:
            try:
                type1 = 'numeric' if var1 in column_types.get('numeric', []) else 'categorical'
                type2 = 'numeric' if var2 in column_types.get('numeric', []) else 'categorical'
                
                # Échantillonnage optimisé pour analyse bivariée
                max_sample = min(15000, compute_if_dask(df.shape[0]))
                
                if is_dask_dataframe(df):
                    sample_df = df.sample(frac=min(0.05, max_sample / compute_if_dask(df.shape[0]))).head(max_sample)
                    sample_df = compute_if_dask(sample_df)
                else:
                    sample_df = df.head(max_sample) if len(df) <= max_sample else df.sample(n=max_sample, random_state=42)
                
                if not sample_df[[var1, var2]].empty:
                    biv_fig = plot_bivariate_analysis(sample_df, var1, var2, type1, type2)
                    if biv_fig:
                        st.plotly_chart(biv_fig, use_container_width=True)
                    else:
                        st.info("Graphique non disponible pour cette combinaison")
                else:
                    st.warning("Données insuffisantes")
                    
            except Exception as e:
                st.error(f"Erreur analyse bivariée: {str(e)[:100]}")
                logger.error(f"Bivariate analysis error: {e}")
        else:
            st.warning("Sélectionnez deux variables différentes")
    else:
        st.warning("Au moins 2 colonnes nécessaires")

# Onglet Aperçu avec limitation de performance
with tab_preview:
    st.subheader("📄 Aperçu des Données")
    try:
        raw_df = st.session_state.get('df_raw', df)
        
        # Contrôle intelligent de la taille d'aperçu
        total_rows = compute_if_dask(raw_df.shape[0])
        preview_size = min(50, total_rows)  # Limité à 50 lignes pour performance
        
        raw_df_sample = compute_if_dask(raw_df.head(preview_size))
        
        # Affichage avec hauteur limitée
        st.dataframe(raw_df_sample, height=350, use_container_width=True)
        st.caption(f"📊 Aperçu: {preview_size} lignes sur {total_rows:,} total")
        
    except Exception as e:
        st.error(f"Erreur aperçu: {str(e)[:100]}")
        logger.error(f"Preview error: {e}")

# Onglet Nettoyage avec états stables et validation
with tab_cleaning:
    st.subheader("🧹 Nettoyage des Colonnes")
    
    # Section Renommage avec validation renforcée
    st.markdown("### ✏️ Renommage")
    available_columns = list(df.columns)
    
    if available_columns:
        col_rename1, col_rename2 = st.columns([1, 1])
        
        with col_rename1:
            # État persistent pour la colonne à renommer
            if 'rename_col_idx' not in st.session_state:
                st.session_state.rename_col_idx = 0
            
            rename_idx = st.selectbox(
                "Colonne à renommer",
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                index=st.session_state.rename_col_idx,
                key="rename_selector"
            )
            col_to_rename = available_columns[rename_idx]
        
        with col_rename2:
            new_name = st.text_input(
                "Nouveau nom",
                value="",
                key="rename_input",
                placeholder=f"nouveau_{col_to_rename}",
                max_chars=50
            )
        
        col_add, col_clear = st.columns([1, 1])
        
        with col_add:
            if st.button("➕ Ajouter", key="add_rename", use_container_width=True):
                new_name_clean = new_name.strip()
                if not new_name_clean:
                    st.error("❌ Nom vide")
                elif new_name_clean in df.columns:
                    st.error("❌ Nom déjà existant")
                elif new_name_clean == col_to_rename:
                    st.warning("⚠️ Nom identique")
                elif not is_valid_column_name(new_name_clean):
                    st.error("❌ Nom invalide (alphanumerique, _, - uniquement)")
                elif any(old == col_to_rename for old, new in st.session_state.rename_list):
                    st.warning(f"⚠️ {col_to_rename} déjà planifié")
                else:
                    st.session_state.rename_list.append((col_to_rename, new_name_clean))
                    st.success(f"✅ Ajouté: {col_to_rename} → {new_name_clean}")
                    st.rerun()
        
        with col_clear:
            if st.button("🗑️ Vider liste", key="clear_renames", use_container_width=True):
                st.session_state.rename_list = []
                st.success("✅ Liste vidée")
                st.rerun()
        
        # Affichage et application des renommages
        if st.session_state.rename_list:
            st.markdown("**📋 Renommages planifiés:**")
            rename_df = pd.DataFrame(st.session_state.rename_list, columns=["Ancien", "Nouveau"])
            st.dataframe(rename_df, use_container_width=True)
            
            if st.button("✅ Appliquer tous", key="apply_renames", type="primary"):
                try:
                    rename_dict = dict(st.session_state.rename_list)
                    valid_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
                    
                    if valid_renames:
                        if is_dask_dataframe(df):
                            df_renamed = df.rename(columns=valid_renames).persist()
                        else:
                            df_renamed = df.rename(columns=valid_renames)
                        
                        # Mise à jour atomique des états
                        st.session_state.df = df_renamed
                        st.session_state.df_raw = df_renamed.copy() if not is_dask_dataframe(df_renamed) else df_renamed
                        st.session_state.column_types = None
                        st.session_state.rename_list = []
                        st.session_state.dataset_hash = get_dataset_hash(df_renamed)
                        st.session_state.dashboard_cache_version += 1
                        
                        logger.info(f"Columns renamed: {valid_renames}")
                        st.success(f"✅ {len(valid_renames)} colonnes renommées!")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.warning("⚠️ Aucune colonne valide")
                        
                except Exception as e:
                    st.error(f"❌ Erreur renommage: {str(e)[:100]}")
                    logger.error(f"Rename error: {e}")

    # Section Suppression avec validation robuste
    st.markdown("### 🗑️ Suppression")
    
    col_detect, col_info = st.columns([1, 2])
    
    with col_detect:
        if st.button("🔍 Détecter inutiles", key="detect_useless", use_container_width=True):
            with st.spinner("🔍 Détection..."):
                try:
                    useless_candidates = detect_useless_columns(df, threshold_missing=0.7)
                    useless_candidates = [col for col in useless_candidates if col in df.columns]
                    st.session_state.useless_candidates = useless_candidates
                    
                    if useless_candidates:
                        st.success(f"✅ {len(useless_candidates)} trouvées")
                    else:
                        st.info("ℹ️ Aucune colonne inutile")
                        
                except Exception as e:
                    st.error(f"❌ Erreur détection: {str(e)[:100]}")
                    logger.error(f"Useless detection error: {e}")
    
    with col_info:
        if st.session_state.useless_candidates:
            candidates_display = st.session_state.useless_candidates[:3]
            display_text = ", ".join(candidates_display)
            if len(st.session_state.useless_candidates) > 3:
                display_text += f" ... +{len(st.session_state.useless_candidates)-3}"
            st.info(f"🎯 Détectées: {display_text}")
    
    # Formulaire de suppression sécurisé
    with st.form("drop_form", clear_on_submit=False):
        st.markdown("**Sélection pour suppression:**")
        
        cols_to_drop = []
        
        # Colonnes détectées
        if st.session_state.useless_candidates:
            auto_cols = st.multiselect(
                "🤖 Auto-détectées",
                options=[col for col in st.session_state.useless_candidates if col in df.columns],
                default=[],
                key="auto_drop_multi",
                help="Colonnes avec trop de valeurs manquantes ou constantes"
            )
            cols_to_drop.extend(auto_cols)
        
        # Sélection manuelle
        remaining = [col for col in df.columns if col not in cols_to_drop]
        manual_cols = st.multiselect(
            "👤 Manuelle",
            options=remaining,
            default=[],
            key="manual_drop_multi"
        )
        cols_to_drop.extend(manual_cols)
        
        # Récapitulatif
        if cols_to_drop:
            total_cols = len(df.columns)
            remaining_cols = total_cols - len(cols_to_drop)
            
            st.markdown(f"**📋 {len(cols_to_drop)} colonnes à supprimer**")
            if len(cols_to_drop) <= 5:
                st.write(f"🎯 {', '.join(cols_to_drop)}")
            else:
                st.write(f"🎯 {', '.join(cols_to_drop[:3])} ... +{len(cols_to_drop)-3} autres")
            
            if remaining_cols == 0:
                st.error("❌ Suppression de toutes les colonnes impossible!")
            elif len(cols_to_drop) > total_cols * 0.7:
                st.warning("⚠️ Plus de 70% des colonnes seront supprimées!")
        
        # Boutons d'action
        col_submit, col_cancel = st.columns([1, 1])
        
        with col_submit:
            submit_drop = st.form_submit_button(
                "🗑️ Supprimer",
                type="primary" if cols_to_drop and len(df.columns) - len(cols_to_drop) > 0 else "secondary",
                disabled=not cols_to_drop or len(df.columns) - len(cols_to_drop) == 0,
                use_container_width=True
            )
        
        with col_cancel:
            if st.form_submit_button("❌ Annuler", use_container_width=True):
                st.session_state.useless_candidates = []
                st.success("✅ Annulé")
                st.rerun()
        
        # Traitement sécurisé de la suppression
        if submit_drop and cols_to_drop:
            with st.spinner(f"🗑️ Suppression de {len(cols_to_drop)} colonne(s)..."):
                try:
                    valid_cols = [col for col in cols_to_drop if col in df.columns]
                    
                    if valid_cols and len(df.columns) - len(valid_cols) > 0:
                        # Suppression effective
                        if is_dask_dataframe(df):
                            df_cleaned = df.drop(columns=valid_cols).persist()
                        else:
                            df_cleaned = df.drop(columns=valid_cols)
                        
                        # Mise à jour atomique des états
                        st.session_state.df = df_cleaned
                        st.session_state.df_raw = df_cleaned.copy() if not is_dask_dataframe(df_cleaned) else df_cleaned
                        st.session_state.column_types = None
                        st.session_state.useless_candidates = []
                        st.session_state.columns_to_drop = []
                        st.session_state.dataset_hash = get_dataset_hash(df_cleaned)
                        st.session_state.dashboard_cache_version += 1
                        
                        logger.info(f"Columns dropped successfully: {valid_cols}")
                        st.success(f"✅ {len(valid_cols)} colonne(s) supprimée(s)!")
                        
                        remaining_cols = list(df_cleaned.columns)
                        if len(remaining_cols) <= 8:
                            st.info(f"📊 Colonnes restantes: {', '.join(remaining_cols)}")
                        else:
                            st.info(f"📊 {len(remaining_cols)} colonnes restantes: {', '.join(remaining_cols[:5])} ... +{len(remaining_cols)-5} autres")
                        
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Aucune colonne valide à supprimer")
                        
                except Exception as e:
                    st.error(f"❌ Erreur suppression: {str(e)[:100]}")
                    logger.error(f"Drop columns error: {e}")

# Footer avec monitoring système discret mais informatif
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    n_rows = len(df) if not is_dask_dataframe(df) else "Dask"
    if isinstance(n_rows, int):
        st.caption(f"📊 {n_rows:,} × {df.shape[1]} colonnes")
    else:
        st.caption(f"📊 {n_rows} × {df.shape[1]} colonnes")

with footer_col2:
    if not is_dask_dataframe(df):
        try:
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"💾 {memory_mb:.1f} MB")
        except:
            st.caption("💾 N/A")
    else:
        st.caption(f"💾 {df.npartitions} partitions")

with footer_col3:
    try:
        sys_mem = psutil.virtual_memory().percent
        color = "🔴" if sys_mem > 85 else "🟡" if sys_mem > 70 else "🟢"
        st.caption(f"{color} RAM: {sys_mem:.0f}%")
    except:
        st.caption("🔧 RAM: N/A")

with footer_col4:
    st.caption(f"⏰ {time.strftime('%H:%M:%S')}")

# Actions d'urgence et debug
action_col1, action_col2 = st.columns(2)

with action_col1:
    if st.button("🔄 Reset Dashboard", help="Réinitialise le cache et l'état", key="emergency_reset"):
        try:
            # Nettoyage sélectif pour éviter les problèmes
            keys_to_clear = ['column_types', 'useless_candidates', 'rename_list', 'columns_to_drop']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.cache_data.clear()
            cleanup_memory()
            st.session_state.dashboard_cache_version += 1
            
            st.success("✅ Dashboard réinitialisé")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erreur reset: {e}")

with action_col2:
    # Debug conditionnel basé sur variable d'environnement
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    if debug_mode and st.button("🔧 Debug Info", help="Informations de débogage"):
        with st.expander("🔍 Informations de débogage", expanded=True):
            st.json({
                "dataset_hash": st.session_state.dataset_hash,
                "cache_version": st.session_state.dashboard_cache_version,
                "columns_count": len(df.columns),
                "dataframe_type": "Dask" if is_dask_dataframe(df) else "Pandas",
                "column_types_available": st.session_state.column_types is not None,
                "system_memory_percent": get_system_metrics()['memory_percent']
            })

# Surveillance continue des performances (non-bloquante)
if hasattr(st.session_state, 'last_perf_check'):
    if time.time() - st.session_state.last_perf_check > 300:  # 5 minutes
        try:
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 90:
                st.error("🚨 Mémoire critique! Redémarrage recommandé.")
        except:
            pass
        st.session_state.last_perf_check = time.time()
else:
    st.session_state.last_perf_check = time.time()

def get_system_metrics():
    """Métriques système pour le monitoring"""
    try:
        return {
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }
    except:
        return {'memory_percent': 0, 'timestamp': time.time()}