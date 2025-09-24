import streamlit as st
import pandas as pd
import logging
import warnings
import os
import time
import psutil
from utils.data_loading import load_data
from utils.logging_config import setup_logging
from typing import Dict, Any
import gc

# --- Configuration Production ---
def setup_production_environment():
    """Configuration pour l'environnement de production"""
    
    # Configuration des warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')
    warnings.filterwarnings("ignore", category=UserWarning, module='streamlit')
    
    # Configuration des logs pour production
    logging.getLogger().setLevel(logging.INFO)
    
    # Configuration Streamlit pour la production
    if 'production_setup_done' not in st.session_state:
        st.session_state.production_setup_done = True
        
        # Masquer les éléments Streamlit en production
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            .stAlert > div  {
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
            }
            .main > div {
                padding-top: 1rem;
            }
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Initialisation ---
setup_production_environment()
setup_logging()
logger = logging.getLogger(__name__)

# --- Constantes de l'Application ---
DEFAULT_TASK_TYPE = "classification"
DEFAULT_N_SPLITS = 3
SUPPORTED_EXTENSIONS = {'csv', 'parquet', 'xlsx', 'xls', 'json'}
MAX_FILE_SIZE_MB = 1024  # 1 Go maximum
MEMORY_WARNING_THRESHOLD = 85  # Pourcentage d'utilisation mémoire

# --- Fonctions de Monitoring ---
def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système actuelles"""
    try:
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def check_system_health():
    """Vérifie la santé du système et affiche des alertes si nécessaire"""
    metrics = get_system_metrics()
    
    if metrics['memory_percent'] > MEMORY_WARNING_THRESHOLD:
        st.warning(f"⚠️ Utilisation mémoire élevée: {metrics['memory_percent']:.1f}%")
        logger.warning(f"High memory usage detected: {metrics['memory_percent']:.1f}%")
        
        # Suggestion de nettoyage automatique
        if metrics['memory_percent'] > 90:
            if st.button("🧹 Nettoyer la mémoire", help="Libère la mémoire et vide les caches"):
                cleanup_memory()
                st.success("Nettoyage mémoire effectué")
                st.rerun()

def cleanup_memory():
    """Nettoyage mémoire avec logging"""
    try:
        # Garbage collection Python
        collected = gc.collect()
        
        # Nettoyage cache Streamlit
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
            
        logger.info(f"Memory cleanup: {collected} objects collected")
        return collected
        
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return 0

# --- Fonctions de Gestion d'État ---
def initialize_session():
    """Initialise l'état de base de la session de façon robuste"""
    required_keys = {
        'df': None,
        'df_raw': None,
        'uploaded_file_name': None,
        'target_column_for_ml_config': None,
        'task_type': DEFAULT_TASK_TYPE,
        'config': None,
        'model_name': None,
        'model_params': {},
        'preprocessing': {},
        'n_splits': DEFAULT_N_SPLITS,
        'model': None,
        'metrics_summary': None,
        'preprocessor': None,
        'ml_results': [],
        'last_system_check': 0,
        'error_count': 0
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Vérification périodique du système (toutes les 5 minutes)
    current_time = time.time()
    if current_time - st.session_state.last_system_check > 300:
        check_system_health()
        st.session_state.last_system_check = current_time

def reset_app_state():
    """
    Réinitialise toutes les variables de session liées à un jeu de données.
    Version robuste avec validation et logging.
    """
    logger.info("Réinitialisation de l'état de l'application pour un nouveau fichier")
    
    try:
        # Sauvegarde des métriques avant reset
        old_error_count = st.session_state.get('error_count', 0)
        
        # Réinitialisation des variables d'état principales
        reset_keys = [
            'df', 'df_raw', 'uploaded_file_name', 'target_column_for_ml_config',
            'task_type', 'config', 'model_name', 'model_params', 'preprocessing',
            'n_splits', 'model', 'metrics_summary', 'preprocessor', 'ml_results'
        ]
        
        for key in reset_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Réinitialisation avec valeurs par défaut
        st.session_state.df = None
        st.session_state.df_raw = None
        st.session_state.uploaded_file_name = None
        st.session_state.target_column_for_ml_config = None
        st.session_state.task_type = DEFAULT_TASK_TYPE
        st.session_state.config = None
        st.session_state.model_name = None
        st.session_state.model_params = {}
        st.session_state.preprocessing = {}
        st.session_state.n_splits = DEFAULT_N_SPLITS
        st.session_state.model = None
        st.session_state.metrics_summary = None
        st.session_state.preprocessor = None
        st.session_state.ml_results = []
        
        # Conservation des métriques système
        st.session_state.error_count = old_error_count
        
        # Nettoyage mémoire
        cleanup_memory()
        
        logger.info("État de l'application réinitialisé avec succès")
        st.toast("Application réinitialisée pour le nouveau fichier", icon="🔄")
        
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation : {e}")
        st.error(f"Erreur lors de la réinitialisation : {e}")

def validate_session_state() -> bool:
    """
    Valide l'intégrité de l'état de la session.
    
    Returns:
        True si l'état est valide, False sinon
    """
    try:
        # Vérifications de base
        if 'df' not in st.session_state:
            return False
            
        df = st.session_state.df
        if df is not None:
            # Vérification de l'intégrité du DataFrame
            if not hasattr(df, 'columns') or len(df.columns) == 0:
                logger.warning("DataFrame in session_state is corrupted")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Session state validation failed: {e}")
        return False

# --- Interface Principale ---
st.set_page_config(
    page_title="DataLab Pro | Accueil",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialisation robuste de la session
try:
    initialize_session()
    
    # Validation de l'état de session
    if not validate_session_state():
        logger.warning("Invalid session state detected, resetting...")
        reset_app_state()
        
except Exception as e:
    logger.error(f"Session initialization failed: {e}")
    st.error("Erreur d'initialisation de la session. Veuillez recharger la page.")
    st.stop()

# Header avec informations système
col_title, col_system = st.columns([3, 1])

with col_title:
    st.title("🧪 DataLab Pro")
    st.markdown("Plateforme d'analyse de données et de Machine Learning automatisé")

with col_system:
    # Affichage discret des métriques système
    metrics = get_system_metrics()
    if metrics['memory_percent'] > 0:
        color = "🔴" if metrics['memory_percent'] > 85 else "🟡" if metrics['memory_percent'] > 70 else "🟢"
        st.caption(f"{color} RAM: {metrics['memory_percent']:.0f}%")

st.markdown("---")

# Section principale de chargement
st.header("📂 Importation des données")

# Informations sur les formats supportés
with st.expander("ℹ️ Formats supportés et limites", expanded=False):
    st.markdown(f"""
    **Formats acceptés :** {', '.join(SUPPORTED_EXTENSIONS).upper()}
    
    **Limites :**
    - Taille maximale : {MAX_FILE_SIZE_MB:,} MB
    - Automatiquement optimisé selon la taille (Pandas ≤ 100MB, Dask > 100MB)
    - Validation d'intégrité avant chargement
    
    **Fonctionnalités automatiques :**
    - Détection et suppression des doublons
    - Conversion intelligente des types de données
    - Optimisation mémoire pour les gros datasets
    """)

# Widget de téléchargement avec validation
uploaded_file = st.file_uploader(
    "Choisissez votre fichier de données",
    type=list(SUPPORTED_EXTENSIONS),
    key="file_uploader",
    help=f"Formats supportés: {', '.join(SUPPORTED_EXTENSIONS).upper()} • Maximum {MAX_FILE_SIZE_MB}MB"
)

# Traitement du fichier uploadé
if uploaded_file is not None:
    try:
        # Validation de la taille
        file_size_mb = uploaded_file.size / (1024 * 1024) if hasattr(uploaded_file, 'size') else 0
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ Fichier trop volumineux: {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            logger.error(f"File too large: {file_size_mb:.1f}MB")
            st.stop()
        
        # Vérification si c'est un nouveau fichier
        if st.session_state.uploaded_file_name != uploaded_file.name:
            logger.info(f"New file detected: {uploaded_file.name}")
            
            # Reset de l'état pour le nouveau fichier
            reset_app_state()
            
            # Interface de chargement avec progress
            progress_container = st.container()
            with progress_container:
                st.info(f"📥 Chargement de **{uploaded_file.name}** ({file_size_mb:.1f}MB)...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Étapes de chargement
                progress_bar.progress(20)
                status_text.text("Validation du fichier...")
                time.sleep(0.5)
                
                progress_bar.progress(40)
                status_text.text("Chargement des données...")
                
                # Chargement des données avec gestion d'erreur
                try:
                    df, report, df_raw = load_data(
                        file_path=uploaded_file,
                        blocksize="64MB",
                        sanitize_for_display=True
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("Finalisation...")
                    
                except Exception as load_error:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Erreur lors du chargement: {str(load_error)}")
                    logger.error(f"Data loading failed: {load_error}")
                    st.session_state.error_count += 1
                    st.stop()
                
                progress_bar.progress(100)
                status_text.text("Terminé!")
                time.sleep(0.5)
                
                # Nettoyage de l'interface de progress
                progress_container.empty()
            
            # Traitement des résultats
            if df is not None:
                # Sauvegarde dans la session
                st.session_state.df = df
                st.session_state.df_raw = df_raw
                st.session_state.uploaded_file_name = uploaded_file.name
                
                logger.info(f"File loaded successfully: {uploaded_file.name}")
                
                # Affichage du rapport de chargement
                if report and report.get("actions"):
                    st.success("✅ Fichier chargé avec succès!")
                    
                    # Rapport détaillé dans un expander
                    with st.expander("📋 Rapport de chargement", expanded=False):
                        for action in report["actions"]:
                            st.write(f"• {action}")
                            
                        # Changements de types si applicable
                        if report.get("changes"):
                            st.subheader("🔧 Conversions de types automatiques")
                            changes_df = pd.DataFrame([
                                {"Colonne": col, "Conversion": change}
                                for col, change in report["changes"].items()
                            ])
                            st.dataframe(changes_df, use_container_width=True)
                            
                        # Avertissements si présents
                        if report.get("warnings"):
                            st.subheader("⚠️ Avertissements")
                            for warning in report["warnings"]:
                                st.warning(warning)
                
                # Rafraîchir la page pour mettre à jour l'interface
                st.rerun()
                
            else:
                # Échec du chargement
                error_messages = report.get("actions", ["Erreur inconnue"]) if report else ["Erreur inconnue"]
                st.error(f"❌ Échec du chargement: {error_messages[0]}")
                logger.error(f"Data loading failed: {error_messages[0]}")
                st.session_state.error_count += 1
                
                # Suggestions de résolution
                st.markdown("""
                **Suggestions pour résoudre le problème:**
                - Vérifiez le format du fichier
                - Assurez-vous que le fichier n'est pas corrompu
                - Essayez avec un fichier plus petit
                - Vérifiez l'encodage (UTF-8 recommandé)
                """)
                
    except Exception as e:
        st.error(f"❌ Erreur inattendue: {str(e)}")
        logger.error(f"Unexpected error during file processing: {e}", exc_info=True)
        st.session_state.error_count += 1

# --- Affichage de l'état actuel ---
if st.session_state.df is not None:
    try:
        df = st.session_state.df
        
        # Informations sur le dataset chargé
        st.success(f"✅ Dataset **{st.session_state.uploaded_file_name}** prêt pour l'analyse")
        
        # Métriques du dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_rows = len(df) if not hasattr(df, 'npartitions') else "Dask"
            st.metric("Lignes", f"{n_rows:,}" if isinstance(n_rows, int) else n_rows)
            
        with col2:
            st.metric("Colonnes", f"{len(df.columns)}")
            
        with col3:
            if not hasattr(df, 'npartitions'):
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Mémoire", f"{memory_mb:.1f} MB")
            else:
                st.metric("Partitions", f"{df.npartitions}")
                
        with col4:
            df_type = "Dask" if hasattr(df, 'npartitions') else "Pandas"
            st.metric("Type", df_type)
        
        # Aperçu des données avec gestion d'erreurs
        st.subheader("Aperçu des données")
        try:
            # Limitation de l'aperçu pour les performances
            preview_rows = min(100, len(df) if not hasattr(df, 'npartitions') else 100)
            
            if hasattr(df, 'npartitions'):
                # Pour Dask
                df_preview = df.head(preview_rows).compute()
            else:
                # Pour Pandas
                df_preview = df.head(preview_rows)
            
            st.dataframe(df_preview, use_container_width=True, height=300)
            
            if len(df_preview) == preview_rows:
                st.caption(f"Affichage des {preview_rows} premières lignes")
                
        except Exception as preview_error:
            st.warning(f"⚠️ Erreur d'aperçu: {preview_error}")
            logger.error(f"Preview error: {preview_error}")
            
            # Fallback avec conversion string
            try:
                df_fallback = df.head(50).astype(str)
                if hasattr(df_fallback, 'compute'):
                    df_fallback = df_fallback.compute()
                st.dataframe(df_fallback, use_container_width=True)
                st.caption("Aperçu avec conversion forcée en texte")
            except:
                st.error("Impossible d'afficher l'aperçu des données")
        
        # Navigation vers les autres pages
        st.markdown("---")
        st.subheader("🚀 Étapes suivantes")
        
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        
        with col_nav1:
            st.markdown("""
            **📊 Dashboard**
            - Vue d'ensemble des données
            - Analyse des valeurs manquantes
            - Distribution des variables
            """)
            
        with col_nav2:
            st.markdown("""
            **🤖 AutoML**
            - Configuration automatique
            - Entraînement de modèles
            - Évaluation des performances
            """)
            
        with col_nav3:
            st.markdown("""
            **📈 Résultats**
            - Métriques détaillées
            - Visualisations
            - Export des modèles
            """)
        
        st.info("💡 Utilisez la barre latérale pour naviguer entre les pages")
        
    except Exception as display_error:
        st.error(f"❌ Erreur d'affichage: {display_error}")
        logger.error(f"Display error: {display_error}", exc_info=True)
        st.session_state.error_count += 1
        
        # Option de récupération
        if st.button("🔄 Réinitialiser l'application"):
            reset_app_state()
            st.rerun()

else:
    # État initial - aucun fichier chargé
    st.info("📁 Chargez un fichier pour commencer l'analyse des données")
    
    # Exemples et conseils
    with st.expander("💡 Conseils pour de meilleurs résultats", expanded=False):
        st.markdown("""
        **Préparation des données:**
        - Nettoyez vos données avant le chargement si possible
        - Utilisez des noms de colonnes clairs et sans espaces
        - Évitez les caractères spéciaux dans les noms de colonnes
        
        **Performance:**
        - Les fichiers > 100MB utiliseront automatiquement Dask
        - Format Parquet recommandé pour les gros volumes
        - CSV avec séparateurs standards (virgule, point-virgule)
        
        **Formats recommandés:**
        - **CSV**: Simple et universel
        - **Parquet**: Optimal pour gros volumes
        - **Excel**: Pratique mais plus lent
        """)

# Footer avec informations de debug et actions utiles
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if st.session_state.get('error_count', 0) > 0:
        st.caption(f"⚠️ Erreurs: {st.session_state.error_count}")

with footer_col2:
    current_time = time.strftime("%H:%M:%S")
    st.caption(f"⏰ Session: {current_time}")

with footer_col3:
    if st.button("🧹 Nettoyer cache", help="Libère la mémoire et vide les caches"):
        cleanup_memory()
        st.success("Cache nettoyé")
        st.rerun()

# Gestion d'erreurs globales non capturées
if 'last_error_check' not in st.session_state:
    st.session_state.last_error_check = time.time()

# Vérification périodique des erreurs (toutes les 10 minutes)
if time.time() - st.session_state.last_error_check > 600:
    if st.session_state.get('error_count', 0) > 10:
        st.warning("⚠️ Plusieurs erreurs détectées. Considérez recharger l'application.")
        if st.button("🔄 Recharger l'application"):
            st.session_state.clear()
            st.rerun()
    st.session_state.last_error_check = time.time()