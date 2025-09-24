import os
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import time
import logging
import re
import psutil
import gc
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
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Streamlit
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard Exploratoire")

# Initialisation stable des états
def initialize_session_state():
    """Initialise les variables de session de manière stable"""
    if 'widget_key_counter' not in st.session_state:
        st.session_state.widget_key_counter = 0
    if 'column_types' not in st.session_state:
        st.session_state.column_types = None
    if 'rename_list' not in st.session_state:
        st.session_state.rename_list = []
    if 'columns_to_drop' not in st.session_state:
        st.session_state.columns_to_drop = []
    if 'useless_candidates' not in st.session_state:
        st.session_state.useless_candidates = []
    if 'dataset_hash' not in st.session_state:
        st.session_state.dataset_hash = None

initialize_session_state()

# Validation des noms de colonnes
def is_valid_column_name(name: str) -> bool:
    """
    Vérifie si un nom de colonne est valide (alphanumérique, underscores, pas de caractères spéciaux).
    
    Args:
        name: Nom de la colonne à valider
    
    Returns:
        Booléen indiquant si le nom est valide
    """
    if not name or not isinstance(name, str):
        return False
    return bool(re.match(r'^[a-zA-Z0-9_]+$', name))

# Vérification du chargement du DataFrame
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Veuillez d'abord charger un jeu de données depuis la page d'accueil.")
    st.page_link("app.py", label="Retour à l'accueil", icon="🏠")
    st.stop()

df = st.session_state.df

# Fonction pour générer un hash stable du dataset
def get_dataset_hash(df):
    """Génère un hash stable basé sur la structure du dataset"""
    try:
        if is_dask_dataframe(df):
            return f"{tuple(df.columns)}_{df.npartitions}_{str(df.dtypes.to_dict())}"
        else:
            return f"{tuple(df.columns)}_{df.shape[0]}_{str(df.dtypes.to_dict())}"
    except Exception as e:
        logger.error(f"Erreur lors du calcul du hash: {e}")
        return str(time.time())

# Vérifier si le dataset a changé
current_hash = get_dataset_hash(df)
if st.session_state.dataset_hash != current_hash:
    st.session_state.dataset_hash = current_hash
    st.session_state.column_types = None  # Forcer la recalculation
    logger.info(f"Dataset changé, nouveau hash: {current_hash}")

# -----------------------
# Métriques globales avec cache optimisé
# -----------------------
@st.cache_data(
    hash_funcs={
        pd.DataFrame: lambda x: get_dataset_hash(x),
        dd.DataFrame: lambda x: get_dataset_hash(x)
    },
    ttl=300  # Cache pendant 5 minutes
)
def compute_global_metrics(_df):
    """Calcule les métriques globales du dataset."""
    try:
        start_time = time.time()
        n_rows = compute_if_dask(_df.shape[0])
        n_cols = _df.shape[1]
        
        # Calcul sécurisé des valeurs manquantes
        try:
            total_missing = compute_if_dask(_df.isna().sum().sum())
        except Exception as e:
            logger.warning(f"Failed to compute missing values: {e}")
            total_missing = 0
            
        missing_percentage = (total_missing / (n_rows * n_cols)) * 100 if (n_rows * n_cols) > 0 else 0
        
        # Calcul sécurisé des doublons
        try:
            duplicate_rows = compute_if_dask(_df.duplicated().sum())
        except Exception as e:
            logger.warning(f"Failed to compute duplicates: {e}")
            duplicate_rows = 0
        
        if not is_dask_dataframe(_df):
            try:
                memory_usage = compute_if_dask(_df.memory_usage(deep=True).sum()) / (1024**2)
            except:
                memory_usage = 0
        else:
            memory_usage = "inconnu (Dask)"
            
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
        logger.error(f"Erreur dans compute_global_metrics: {e}")
        return {
            'n_rows': 0,
            'n_cols': 0,
            'missing_percentage': 0,
            'duplicate_rows': 0,
            'memory_usage': 0
        }

# -----------------------
# Détection automatique des types avec cache optimisé
# -----------------------
@st.cache_data(
    hash_funcs={
        pd.DataFrame: lambda x: get_dataset_hash(x),
        dd.DataFrame: lambda x: get_dataset_hash(x)
    },
    ttl=300
)
def cached_auto_detect_column_types(_df):
    """Met en cache la détection des types de colonnes."""
    try:
        start_time = time.time()
        result = auto_detect_column_types(_df)
        elapsed_time = time.time() - start_time
        logger.info(f"Column types detected in {elapsed_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Erreur dans cached_auto_detect_column_types: {e}")
        return {'numeric': [], 'categorical': [], 'text_or_high_cardinality': [], 'datetime': []}

# Calculer les types de colonnes si nécessaire
if st.session_state.column_types is None:
    with st.spinner("Analyse des types de colonnes..."):
        st.session_state.column_types = cached_auto_detect_column_types(df)

column_types = st.session_state.column_types

# Assurer que toutes les clés nécessaires existent
required_keys = ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']
for key in required_keys:
    if key not in column_types:
        column_types[key] = []

# -----------------------
# Vue d'ensemble
# -----------------------
st.header("Vue d'ensemble du jeu de données")

try:
    overview_metrics = compute_global_metrics(df)
    fig = plot_overview_metrics(overview_metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Impossible d'afficher les métriques globales")
except Exception as e:
    st.error(f"Erreur lors du calcul des métriques: {e}")
    logger.error(f"Erreur métriques globales: {e}")

# Créer les onglets avec des clés stables
tab_overview, tab_univariate, tab_bivariate, tab_preview, tab_cleaning = st.tabs([
    "📈 Qualité des Données", 
    "🔬 Analyse par Variable", 
    "🔗 Relations Bivariées", 
    "📄 Aperçu des Données Brutes", 
    "🗑️ Nettoyage des Colonnes"
])

# -----------------------
# Qualité des données
# -----------------------
with tab_overview:
    st.subheader("Valeurs Manquantes et Cardinalité")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            missing_fig = plot_missing_values_overview(df)
            if missing_fig:
                st.plotly_chart(missing_fig, use_container_width=True)
            else:
                st.info("Aucune valeur manquante détectée.")
        except Exception as e:
            st.warning(f"Erreur lors de l'affichage des valeurs manquantes: {e}")
            logger.error(f"Erreur plot missing values: {e}")
    
    with col2:
        try:
            cardinality_fig = plot_cardinality_overview(df, column_types)
            if cardinality_fig:
                st.plotly_chart(cardinality_fig, use_container_width=True)
            else:
                st.info("Aucun graphique de cardinalité disponible.")
        except Exception as e:
            st.warning(f"Erreur lors de l'affichage de la cardinalité: {e}")
            logger.error(f"Erreur plot cardinality: {e}")

# -----------------------
# Analyse univariée avec clés stables
# -----------------------
with tab_univariate:
    st.subheader("Analyse Approfondie d'une Variable")
    available_columns = list(df.columns)
    
    if available_columns:
        selected_col_index = st.selectbox(
            "Choisissez une variable à analyser",
            options=range(len(available_columns)),
            format_func=lambda x: available_columns[x],
            key="univariate_column_selector"
        )
        selected_col = available_columns[selected_col_index]
        
        if selected_col and selected_col in df.columns:
            try:
                col_type = 'numeric' if selected_col in column_types.get('numeric', []) else 'categorical'
                
                # Échantillonnage intelligent et sécurisé
                sample_size = min(50000, compute_if_dask(df.shape[0]))
                if is_dask_dataframe(df) and df.npartitions > 1:
                    try:
                        sample_df = df.sample(frac=min(0.1, sample_size / compute_if_dask(df.shape[0]))).head(sample_size)
                    except:
                        sample_df = df.head(sample_size)
                else:
                    sample_df = df.head(sample_size)
                
                sample_df = compute_if_dask(sample_df)

                if col_type == 'numeric':
                    if sample_df[selected_col].empty or sample_df[selected_col].isna().all():
                        st.warning(f"Aucune donnée valide pour la colonne {selected_col}.")
                    else:
                        fig = plot_distribution(sample_df[selected_col], selected_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Impossible d'afficher le graphique de distribution.")
                else:
                    st.write(f"Analyse de la variable catégorielle: **{selected_col}**")
                    try:
                        value_counts = sample_df[selected_col].value_counts().head(20)
                        if not value_counts.empty:
                            value_counts_df = value_counts.reset_index()
                            value_counts_df.columns = ['Catégorie', 'Comptage']
                            st.dataframe(value_counts_df, use_container_width=True)
                        else:
                            st.info("Aucune donnée à afficher pour cette variable.")
                    except Exception as e:
                        st.warning(f"Erreur lors de l'analyse catégorielle: {e}")
                        
            except Exception as e:
                st.error(f"Erreur lors de l'analyse univariée: {e}")
                logger.error(f"Erreur analyse univariée: {e}")
    else:
        st.warning("Aucune colonne disponible pour l'analyse.")

# -----------------------
# Analyse bivariée avec clés stables
# -----------------------
with tab_bivariate:
    st.subheader("Analyse des Relations entre Deux Variables")
    available_columns = list(df.columns)
    
    if len(available_columns) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            var1_index = st.selectbox(
                "Variable 1", 
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                key="bivariate_var1"
            )
            var1 = available_columns[var1_index]
            
        with col2:
            var2_index = st.selectbox(
                "Variable 2", 
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                key="bivariate_var2",
                index=1 if len(available_columns) > 1 else 0
            )
            var2 = available_columns[var2_index]
        
        if var1 != var2:
            try:
                type1 = 'numeric' if var1 in column_types.get('numeric', []) else 'categorical'
                type2 = 'numeric' if var2 in column_types.get('numeric', []) else 'categorical'
                
                # Échantillonnage pour l'analyse bivariée
                sample_size = min(20000, compute_if_dask(df.shape[0]))
                if is_dask_dataframe(df) and df.npartitions > 1:
                    try:
                        sample_df = df.sample(frac=min(0.05, sample_size / compute_if_dask(df.shape[0]))).head(sample_size)
                    except:
                        sample_df = df.head(sample_size)
                else:
                    sample_df = df.head(sample_size)
                
                sample_df = compute_if_dask(sample_df)
                
                if not sample_df[[var1, var2]].empty:
                    biv_fig = plot_bivariate_analysis(sample_df, var1, var2, type1, type2)
                    if biv_fig:
                        st.plotly_chart(biv_fig, use_container_width=True)
                    else:
                        st.info("Aucun graphique bivarié disponible pour cette combinaison.")
                else:
                    st.warning(f"Données insuffisantes pour l'analyse bivariée de {var1} et {var2}.")
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse bivariée: {e}")
                logger.error(f"Erreur analyse bivariée: {e}")
        else:
            st.warning("Veuillez sélectionner deux variables différentes.")
    else:
        st.warning("Au moins deux colonnes sont nécessaires pour l'analyse bivariée.")

# -----------------------
# Aperçu des données brutes
# -----------------------
with tab_preview:
    st.subheader("Aperçu des Données Brutes")
    try:
        raw_df = st.session_state.get('df_raw', df)
        preview_size = min(100, compute_if_dask(raw_df.shape[0]))
        raw_df_sample = compute_if_dask(raw_df.head(preview_size))
        
        st.dataframe(raw_df_sample, height=400, use_container_width=True)
        st.caption(f"Affichage des {preview_size} premières lignes sur {compute_if_dask(raw_df.shape[0])} au total.")
        
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'aperçu: {e}")
        logger.error(f"Erreur aperçu données: {e}")

# -----------------------
# Nettoyage et renommage avec gestion d'état améliorée
# -----------------------
with tab_cleaning:
    st.subheader("Nettoyage des Colonnes")
    
    # Section Renommage
    st.markdown("### 📝 Renommage des Colonnes")
    available_columns = list(df.columns)
    
    if available_columns:
        col_rename1, col_rename2 = st.columns([1, 1])
        
        with col_rename1:
            col_to_rename_index = st.selectbox(
                "Sélectionnez une colonne à renommer", 
                options=range(len(available_columns)),
                format_func=lambda x: available_columns[x],
                key="rename_column_selector"
            )
            col_to_rename = available_columns[col_to_rename_index]
            
        with col_rename2:
            new_name = st.text_input(
                "Nouveau nom", 
                value="",
                key="new_column_name_input",
                placeholder=f"nouveau_nom_pour_{col_to_rename}"
            )
        
        col_add_btn, col_clear_btn = st.columns([1, 1])
        
        with col_add_btn:
            if st.button("➕ Ajouter le renommage", key="add_rename_btn"):
                if new_name.strip():
                    new_name = new_name.strip()
                    if new_name in df.columns:
                        st.error("Ce nom de colonne existe déjà.")
                    elif new_name == col_to_rename:
                        st.warning("Le nouveau nom est identique à l'ancien.")
                    elif not is_valid_column_name(new_name):
                        st.error("Le nom de colonne doit être alphanumérique ou contenir des underscores (ex.: 'colonne_1').")
                    else:
                        existing_renames = [old for old, new in st.session_state.rename_list]
                        if col_to_rename in existing_renames:
                            st.warning(f"La colonne {col_to_rename} est déjà prévue pour renommage.")
                        else:
                            st.session_state.rename_list.append((col_to_rename, new_name))
                            st.success(f"Renommage ajouté: {col_to_rename} → {new_name}")
                            st.rerun()
                else:
                    st.error("Veuillez saisir un nouveau nom de colonne.")
        
        with col_clear_btn:
            if st.button("🗑️ Vider la liste", key="clear_rename_list"):
                st.session_state.rename_list = []
                st.success("Liste de renommage vidée.")
                st.rerun()
        
        # Afficher la liste des renommages prévus
        if st.session_state.rename_list:
            st.markdown("**Renommages prévus:**")
            rename_df = pd.DataFrame(st.session_state.rename_list, columns=["Ancien Nom", "Nouveau Nom"])
            st.dataframe(rename_df, use_container_width=True)
            
            if st.button("✅ Appliquer tous les renommages", key="apply_all_renames"):
                try:
                    rename_dict = dict(st.session_state.rename_list)
                    valid_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
                    
                    if valid_renames:
                        if is_dask_dataframe(df):
                            df_renamed = df.rename(columns=valid_renames).persist()
                        else:
                            df_renamed = df.rename(columns=valid_renames)
                        
                        # Mise à jour des états de session
                        st.session_state.df = df_renamed
                        if 'df_raw' in st.session_state:
                            st.session_state.df_raw = df_renamed.copy() if not is_dask_dataframe(df_renamed) else df_renamed
                        
                        # Recalculer les types de colonnes et mettre à jour le hash
                        st.session_state.column_types = None
                        st.session_state.rename_list = []
                        st.session_state.dataset_hash = get_dataset_hash(df_renamed)
                        
                        logger.info(f"Colonnes renommées avec succès: {valid_renames}")
                        st.success(f"✅ {len(valid_renames)} colonnes renommées avec succès!")
                        
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("Aucune colonne valide à renommer.")
                        
                except Exception as e:
                    st.error(f"Erreur lors du renommage: {e}")
                    logger.error(f"Erreur renommage colonnes: {e}")

    # Section Suppression des colonnes
    st.markdown("### 🗑️ Suppression des Colonnes")
    
    col_detect, col_info = st.columns([1, 2])
    
    with col_detect:
        if st.button("🔍 Détecter les colonnes inutiles", key="detect_useless_btn"):
            with st.spinner("Détection en cours..."):
                try:
                    useless_candidates = detect_useless_columns(df, threshold_missing=0.6)
                    useless_candidates = [col for col in useless_candidates if col in df.columns]
                    st.session_state.useless_candidates = useless_candidates
                    
                    if useless_candidates:
                        st.success(f"✅ {len(useless_candidates)} colonnes inutiles détectées.")
                        logger.info(f"Colonnes inutiles détectées: {useless_candidates}")
                    else:
                        st.info("Aucune colonne inutile détectée.")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la détection: {e}")
                    logger.error(f"Erreur détection colonnes inutiles: {e}")
    
    with col_info:
        if st.session_state.useless_candidates:
            st.info(f"Colonnes détectées: {', '.join(st.session_state.useless_candidates[:5])}{'...' if len(st.session_state.useless_candidates) > 5 else ''}")
    
    # Formulaire de suppression des colonnes
    with st.form(key="drop_columns_form"):
        st.markdown("**Sélection des colonnes à supprimer:**")
        
        available_columns = list(df.columns)
        cols_to_drop = []
        
        # Colonnes détectées automatiquement
        if st.session_state.useless_candidates:
            auto_cols = st.multiselect(
                "🤖 Colonnes détectées automatiquement (recommandées)",
                options=[col for col in st.session_state.useless_candidates if col in available_columns],
                default=[],
                key="auto_drop_selection",
                help="Colonnes avec trop de valeurs manquantes ou constantes"
            )
            cols_to_drop.extend(auto_cols)
        
        # Sélection manuelle
        remaining_cols = [col for col in available_columns if col not in cols_to_drop]
        manual_cols = st.multiselect(
            "👤 Sélection manuelle",
            options=remaining_cols,
            default=[],
            key="manual_drop_selection",
            help="Sélectionnez d'autres colonnes à supprimer"
        )
        cols_to_drop.extend(manual_cols)
        
        # Résumé des colonnes à supprimer
        if cols_to_drop:
            st.markdown(f"**📋 Résumé: {len(cols_to_drop)} colonne(s) seront supprimées**")
            st.write(", ".join(cols_to_drop))
            
            if len(cols_to_drop) > len(available_columns) * 0.5:
                st.warning("⚠️ Attention: Vous supprimez plus de la moitié des colonnes!")
        
        # Boutons de soumission
        col_submit, col_cancel = st.columns([1, 1])
        
        with col_submit:
            submit_drop = st.form_submit_button(
                "🗑️ Supprimer les colonnes sélectionnées",
                type="primary" if cols_to_drop else "secondary"
            )
        
        with col_cancel:
            if st.form_submit_button("❌ Annuler"):
                st.session_state.useless_candidates = []
                st.success("Sélection annulée.")
                st.rerun()
        
        # Traitement de la suppression
        if submit_drop and cols_to_drop:
            with st.spinner(f"Suppression de {len(cols_to_drop)} colonne(s)..."):
                try:
                    valid_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
                    
                    if valid_cols_to_drop:
                        if df.shape[1] - len(valid_cols_to_drop) == 0:
                            st.error("❌ Impossible de supprimer toutes les colonnes!")
                            logger.error("Tentative de suppression de toutes les colonnes")
                        else:
                            # Effectuer la suppression
                            if is_dask_dataframe(df):
                                df_cleaned = df.drop(columns=valid_cols_to_drop).persist()
                            else:
                                df_cleaned = df.drop(columns=valid_cols_to_drop)
                            
                            # Mise à jour des états de session
                            st.session_state.df = df_cleaned
                            if 'df_raw' in st.session_state:
                                st.session_state.df_raw = df_cleaned.copy() if not is_dask_dataframe(df_cleaned) else df_cleaned
                            
                            # Réinitialiser les états liés aux colonnes
                            st.session_state.column_types = None
                            st.session_state.useless_candidates = []
                            st.session_state.columns_to_drop = []
                            st.session_state.dataset_hash = get_dataset_hash(df_cleaned)
                            
                            logger.info(f"Colonnes supprimées avec succès: {valid_cols_to_drop}")
                            st.success(f"✅ {len(valid_cols_to_drop)} colonne(s) supprimée(s) avec succès!")
                            st.info(f"📊 Colonnes restantes: {', '.join(list(df_cleaned.columns)[:10])}{'...' if len(df_cleaned.columns) > 10 else ''}")
                            
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.warning("⚠️ Aucune colonne valide à supprimer.")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la suppression: {e}")
                    logger.error(f"Erreur suppression colonnes: {e}")
        
        elif submit_drop and not cols_to_drop:
            st.warning("⚠️ Aucune colonne sélectionnée pour la suppression.")

# Footer avec monitoring système
st.markdown("---")
col_footer1, col_footer2, col_footer3, col_footer4 = st.columns(4)

with col_footer1:
    n_rows = len(df) if not is_dask_dataframe(df) else "Dask"
    st.caption(f"📊 Dataset: {n_rows:,} lignes × {df.shape[1]} colonnes" if isinstance(n_rows, int) else f"📊 Dataset: {n_rows} × {df.shape[1]} colonnes")

with col_footer2:
    if not is_dask_dataframe(df):
        try:
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.caption(f"💾 Mémoire: {memory_mb:.1f} MB")
        except:
            st.caption("💾 Mémoire: N/A")
    else:
        st.caption(f"💾 Partitions Dask: {df.npartitions}")

with col_footer3:
    try:
        system_memory = psutil.virtual_memory().percent
        color = "🔴" if system_memory > 85 else "🟡" if system_memory > 70 else "🟢"
        st.caption(f"{color} RAM système: {system_memory:.0f}%")
    except:
        st.caption("🔧 RAM: N/A")

with col_footer4:
    st.caption(f"⏱️ Session: {time.strftime('%H:%M:%S')}")

# Bouton d'urgence pour réinitialisation complète
if st.button("🔄 Réinitialiser le dashboard", help="Efface le cache et recharge", key="emergency_reset"):
    try:
        st.cache_data.clear()
        cleanup_memory()
        for key in ['column_types', 'useless_candidates', 'rename_list', 'columns_to_drop']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Dashboard réinitialisé!")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la réinitialisation: {e}")
        logger.error(f"Reset error: {e}")

# Mode debug (seulement si activé)
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
if DEBUG_MODE:
    with st.expander("🔧 Mode débogage", expanded=False):
        st.write(f"**Hash du dataset:** {st.session_state.dataset_hash}")
        st.write(f"**Nombre de colonnes:** {df.shape[1]}")
        st.write(f"**Type de DataFrame:** {'Dask' if is_dask_dataframe(df) else 'Pandas'}")
        st.write(f"**Types de colonnes:** {st.session_state.column_types}")
        if is_dask_dataframe(df):
            st.write(f"**Partitions Dask:** {df.npartitions}")
        
        # Métriques système détaillées
        try:
            memory_info = psutil.virtual_memory()
            st.write(f"**Mémoire disponible:** {memory_info.available / (1024**3):.1f} GB")
            st.write(f"**CPU usage:** {psutil.cpu_percent()}%")
        except:
            st.write("**Métriques système:** Non disponibles")