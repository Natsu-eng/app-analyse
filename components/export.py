import streamlit as st
import os
import pandas as pd
from reports.pdf_generator import generate_report_pdf
from io import BytesIO

def show_export():
    """Affiche la section d'exportation des modèles et rapports"""
    st.header("💾 Export & Rapport")
    st.markdown("Téléchargez les modèles entraînés et un rapport d'expérimentation complet.")
    
    results = st.session_state.get('results')
    if not results:
        st.warning("Aucun résultat à exporter. Veuillez d'abord lancer une expérimentation.")
        return

    col1, col2 = st.columns(2)

    # --- Export des Modèles ---
    with col1:
        with st.container(border=True):
            st.subheader("📦 Exporter les Modèles")
            st.info("Téléchargez les modèles au format .joblib pour les réutiliser.")
            for result in results:
                model_path = result.get('model_path')
                model_name = result.get('model_name', 'modèle inconnu')
                try:
                    if model_path and os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            st.download_button(
                                label=f"Télécharger {model_name}",
                                data=f,
                                file_name=os.path.basename(model_path),
                                mime="application/octet-stream",
                                use_container_width=True,
                                key=f"download_{model_name}" # Clé unique
                            )
                    else:
                        st.warning(f"Le fichier du modèle {model_name} est introuvable.")
                except IOError as e:
                    st.error(f"Erreur de lecture pour {model_name}: {e}")
    
    # --- Génération du Rapport PDF ---
    with col2:
        with st.container(border=True):
            st.subheader("📄 Générer le Rapport PDF")
            st.info("Obtenez un rapport PDF complet de l'analyse et des performances des modèles.")
            
            target_column = st.session_state.get('target_column_for_ml_config')
            if not target_column:
                st.error("La colonne cible n'est pas définie. Impossible de générer le rapport.")
                return

            if st.button("Générer le Rapport", type="primary", use_container_width=True):
                with st.spinner("Génération du rapport PDF..."):
                    try:
                        pdf_bytes = generate_report_pdf(
                            results,
                            st.session_state.get('task_type', 'Inconnu'),
                            st.session_state.get('df', pd.DataFrame()).shape,
                            target_column
                        )
                        
                        # Le bouton de téléchargement doit être hors de la condition du premier bouton
                        st.session_state.pdf_report_bytes = pdf_bytes
                        st.session_state.report_ready = True
                        st.success("Rapport PDF généré avec succès ! Cliquez ci-dessous pour télécharger.")

                    except Exception as e:
                        st.error(f"Erreur lors de la génération du rapport : {str(e)}")
                        st.session_state.report_ready = False

            # Afficher le bouton de téléchargement si le rapport est prêt
            if st.session_state.get('report_ready'):
                st.download_button(
                    label="📥 Télécharger le Rapport PDF",
                    data=st.session_state.pdf_report_bytes,
                    file_name="rapport_datalab.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )