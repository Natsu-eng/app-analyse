import contextlib
import io
import tempfile
import numpy as np
import plotly.io as pio
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.units import inch
from reportlab.lib import colors
from ml.evaluation.visualization import ModelEvaluationVisualizer
import logging
from datetime import datetime
import gc
import os
import pandas as pd
from typing import Dict, Any, Optional
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

# --- Styles personnalisés modernisés ---
def _create_custom_styles():
    """Crée des styles modernes pour le PDF"""
    styles = getSampleStyleSheet()
    
    # Vérifier et ajouter les styles seulement s'ils n'existent pas déjà
    custom_styles = {}
    
    if 'MainTitle' not in styles:
        styles.add(ParagraphStyle(
            name='MainTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
    
    if 'SubTitle' not in styles:
        styles.add(ParagraphStyle(
            name='SubTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=15,
            fontName='Helvetica-Bold'
        ))
    
    if 'MetricHighlight' not in styles:
        styles.add(ParagraphStyle(
            name='MetricHighlight',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#d62728'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
    
    # Utiliser un nom unique pour éviter les conflits
    if 'BodyTextCustom' not in styles:
        styles.add(ParagraphStyle(
            name='BodyTextCustom',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            fontName='Helvetica'
        ))
    
    return styles

CUSTOM_STYLES = _create_custom_styles()

# --- En-tête et pied de page modernisés ---
def _header(canvas, doc, model_name: str = ""):
    """En-tête moderne avec informations contextuelles"""
    canvas.saveState()
    
    # Fond d'en-tête
    canvas.setFillColor(colors.HexColor('#f8f9fa'))
    canvas.rect(0, doc.height + doc.topMargin - 0.7*inch, doc.width + 2*inch, 0.7*inch, fill=1, stroke=0)
    
    # Texte
    canvas.setFont('Helvetica-Bold', 10)
    canvas.setFillColor(colors.HexColor('#1f77b4'))
    canvas.drawString(inch, doc.height + doc.topMargin - 0.4*inch, 
                     f"Rapport ML - {model_name}" if model_name else "Rapport d'Évaluation ML")
    
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawRightString(doc.width + inch, doc.height + doc.topMargin - 0.4*inch, 
                          f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    canvas.restoreState()

def _footer(canvas, doc):
    """Pied de page professionnel"""
    canvas.saveState()
    
    # Ligne de séparation
    canvas.setStrokeColor(colors.HexColor('#e9ecef'))
    canvas.setLineWidth(0.5)
    canvas.line(inch, 0.7*inch, doc.width + inch, 0.7*inch)
    
    # Texte
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(inch, 0.5*inch, f"Page {doc.page}")
    canvas.drawRightString(doc.width + inch, 0.5*inch, "Document confidentiel - Plateforme ML")
    
    canvas.restoreState()

# --- Conversion graphiques Plotly optimisée ---
def _plotly_fig_to_pdf_image(fig, width=6*inch, height=4*inch, dpi=150) -> Image:
    """Convertit une figure Plotly en image PDF avec gestion d'erreur robuste"""
    try:
        if fig is None:
            return Paragraph("Graphique non disponible", CUSTOM_STYLES['BodyText'])
        
        # Conversion haute qualité
        img_bytes = pio.to_image(
            fig, 
            format='png', 
            width=int(width / inch * dpi),
            height=int(height / inch * dpi),
            scale=2,
            engine='kaleido'
        )
        
        return Image(BytesIO(img_bytes), width=width, height=height)
        
    except Exception as e:
        logger.error(f"Erreur conversion graphique: {e}")
        error_msg = f"Erreur affichage graphique: {str(e)[:100]}..."
        return Paragraph(error_msg, CUSTOM_STYLES['BodyText'])

# --- Page de couverture moderne ---
def _build_cover_page(story, model_result: Dict[str, Any]):
    """Page de couverture professionnelle"""
    model_name = model_result.get('model_name', 'Modèle Inconnu')
    task_type = model_result.get('task_type', 'inconnu')
    metrics = model_result.get('metrics', {})
    
    # Titre principal
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("RAPPORT D'ÉVALUATION", CUSTOM_STYLES['MainTitle']))
    story.append(Paragraph("PLATEFORME MACHINE LEARNING", CUSTOM_STYLES['SubTitle']))
    story.append(Spacer(1, 0.5*inch))
    
    # Carte d'information moderne
    info_data = [
        ["🔮 Modèle", model_name],
        ["🎯 Type de tâche", task_type.upper()],
        ["📅 Date", datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ["⏱️ Temps d'entraînement", f"{model_result.get('training_time', 0):.2f}s"],
    ]
    
    # Ajout métrique principale selon le type
    if task_type == 'classification' and 'accuracy' in metrics:
        accuracy = metrics.get('accuracy', 0)
        info_data.append(["📊 Accuracy", f"{accuracy:.3f}"])
    elif task_type == 'regression' and 'r2' in metrics:
        r2 = metrics.get('r2', 0)
        info_data.append(["📈 Score R²", f"{r2:.3f}"])
    elif task_type == 'clustering' and 'silhouette_score' in metrics:
        silhouette = metrics.get('silhouette_score', 0)
        info_data.append(["🎨 Score Silhouette", f"{silhouette:.3f}"])
    
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f8f9fa')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(info_table)
    story.append(PageBreak())

# --- Section métriques modernisée ---
def _add_metrics_section(story, model_result: Dict[str, Any]):
    """Section métriques avec design moderne"""
    task_type = model_result.get('task_type', 'inconnu')
    metrics = model_result.get('metrics', {})
    
    story.append(Paragraph("📊 MÉTRIQUES DE PERFORMANCE", CUSTOM_STYLES['SubTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    if task_type == 'classification':
        _add_classification_metrics(story, metrics)
    elif task_type == 'regression':
        _add_regression_metrics(story, metrics)
    elif task_type == 'clustering':
        _add_clustering_metrics(story, metrics)
    else:
        story.append(Paragraph("Type de tâche non reconnu", CUSTOM_STYLES['BodyText']))
    
    story.append(Spacer(1, 0.3*inch))

def _add_classification_metrics(story, metrics: Dict[str, Any]):
    """Métriques de classification avec indicateurs visuels"""
    metric_data = [
        ['Métrique', 'Valeur', 'Interprétation'],
        ['Accuracy', f"{metrics.get('accuracy', 0):.3f}", _get_interpretation('accuracy', metrics.get('accuracy', 0))],
        ['F1-Score', f"{metrics.get('f1_score', 0):.3f}", _get_interpretation('f1_score', metrics.get('f1_score', 0))],
        ['Précision', f"{metrics.get('precision', 0):.3f}", _get_interpretation('precision', metrics.get('precision', 0))],
        ['Rappel', f"{metrics.get('recall', 0):.3f}", _get_interpretation('recall', metrics.get('recall', 0))],
    ]
    
    if 'roc_auc' in metrics:
        metric_data.append(['AUC ROC', f"{metrics.get('roc_auc', 0):.3f}", _get_interpretation('roc_auc', metrics.get('roc_auc', 0))])
    
    metric_table = Table(metric_data, colWidths=[1.8*inch, 1*inch, 2.2*inch])
    metric_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#343a40')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#adb5bd')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ]))
    story.append(metric_table)

def _add_regression_metrics(story, metrics: Dict[str, Any]):
    """Métriques de régression avec indicateurs"""
    metric_data = [
        ['Métrique', 'Valeur', 'Interprétation'],
        ['R²', f"{metrics.get('r2', 0):.3f}", _get_interpretation('r2', metrics.get('r2', 0))],
        ['MAE', f"{metrics.get('mae', 0):.3f}", _get_interpretation('mae', metrics.get('mae', 0))],
        ['RMSE', f"{metrics.get('rmse', 0):.3f}", _get_interpretation('rmse', metrics.get('rmse', 0))],
    ]
    
    metric_table = Table(metric_data, colWidths=[1.8*inch, 1*inch, 2.2*inch])
    metric_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#198754')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#adb5bd')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ]))
    story.append(metric_table)

def _add_clustering_metrics(story, metrics: Dict[str, Any]):
    """Métriques de clustering modernisées"""
    metric_data = [
        ['Métrique', 'Valeur', 'Interprétation'],
        ['Silhouette', f"{metrics.get('silhouette_score', 0):.3f}", _get_interpretation('silhouette_score', metrics.get('silhouette_score', 0))],
        ['Clusters', f"{metrics.get('n_clusters', 0)}", _get_interpretation('n_clusters', metrics.get('n_clusters', 0))],
    ]
    
    if 'n_outliers' in metrics:
        metric_data.append(['Outliers', f"{metrics.get('n_outliers', 0)}", 'Points de bruit détectés'])
    
    metric_table = Table(metric_data, colWidths=[1.8*inch, 1*inch, 2.2*inch])
    metric_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6f42c1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#adb5bd')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ]))
    story.append(metric_table)

def _get_interpretation(metric: str, value: float) -> str:
    """Retourne l'interprétation d'une métrique"""
    interpretations = {
        'accuracy': {
            (0.9, 1.0): 'Excellent',
            (0.8, 0.9): 'Très bon',
            (0.7, 0.8): 'Bon',
            (0.6, 0.7): 'Moyen',
            (0.0, 0.6): 'À améliorer'
        },
        'f1_score': {
            (0.9, 1.0): 'Excellent',
            (0.8, 0.9): 'Très bon',
            (0.7, 0.8): 'Bon',
            (0.6, 0.7): 'Moyen',
            (0.0, 0.6): 'À améliorer'
        },
        'r2': {
            (0.9, 1.0): 'Excellente explication',
            (0.7, 0.9): 'Bonne explication',
            (0.5, 0.7): 'Explication modérée',
            (0.3, 0.5): 'Faible explication',
            (0.0, 0.3): 'Très faible explication'
        },
        'silhouette_score': {
            (0.7, 1.0): 'Clusters excellents',
            (0.5, 0.7): 'Clusters raisonnables',
            (0.3, 0.5): 'Clusters faibles',
            (0.0, 0.3): 'Clusters douteux',
            (-1.0, 0.0): 'Clusters inadéquats'
        }
    }
    
    if metric in interpretations:
        for range_val, interpretation in interpretations[metric].items():
            if range_val[0] <= value <= range_val[1]:
                return interpretation
    
    return 'Non évalué'

# --- Section configuration du modèle ---
def _add_model_configuration(story, model_result: Dict[str, Any]):
    """Section configuration du modèle"""
    story.append(Paragraph("⚙️ CONFIGURATION DU MODÈLE", CUSTOM_STYLES['SubTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    config_data = [
        ["Paramètre", "Valeur"],
        ["Nom du modèle", model_result.get('model_name', 'N/A')],
        ["Type de tâche", model_result.get('task_type', 'N/A')],
        ["Temps d'entraînement", f"{model_result.get('training_time', 0):.2f} secondes"],
    ]
    
    # Paramètres optimisés
    best_params = model_result.get('best_params', {})
    if best_params:
        for param, value in list(best_params.items())[:8]:  # Limite à 8 paramètres
            config_data.append([f"🔧 {param}", str(value)[:50] + '...' if len(str(value)) > 50 else str(value)])
    
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c757d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ]))
    
    story.append(config_table)
    story.append(Spacer(1, 0.3*inch))

# --- Section visualisations adaptative ---
def _add_visualizations_section(story, model_result: Dict[str, Any], evaluator: ModelEvaluationVisualizer):
    """Section visualisations adaptée au type de tâche"""
    story.append(PageBreak())
    story.append(Paragraph("📈 VISUALISATIONS", CUSTOM_STYLES['SubTitle']))
    story.append(Spacer(1, 0.3*inch))
    
    task_type = model_result.get('task_type', 'inconnu')
    
    try:
        if task_type == 'clustering':
            _add_clustering_visualizations(story, model_result, evaluator)
        elif task_type == 'classification':
            _add_classification_visualizations(story, model_result, evaluator)
        elif task_type == 'regression':
            _add_regression_visualizations(story, model_result, evaluator)
        else:
            story.append(Paragraph("Visualisations non disponibles pour ce type de tâche.", CUSTOM_STYLES['BodyText']))
    except Exception as e:
        logger.error(f"Erreur génération visualisations: {e}")
        story.append(Paragraph(f"❌ Erreur lors de la génération des visualisations: {str(e)}", CUSTOM_STYLES['BodyText']))

def _add_clustering_visualizations(story, model_result: Dict[str, Any], evaluator: ModelEvaluationVisualizer):
    """Visualisations pour le clustering"""
    # Graphique des clusters
    story.append(Paragraph("🎯 Visualisation des Clusters", CUSTOM_STYLES['BodyText']))
    try:
        cluster_fig = evaluator.create_cluster_scatter_plot(model_result)
        if cluster_fig:
            story.append(_plotly_fig_to_pdf_image(cluster_fig, width=6*inch, height=4*inch))
        else:
            story.append(Paragraph("Graphique des clusters non disponible", CUSTOM_STYLES['BodyText']))
    except Exception as e:
        logger.error(f"Erreur graphique clusters: {e}")
        story.append(Paragraph("❌ Erreur génération graphique clusters", CUSTOM_STYLES['BodyText']))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Graphique silhouette
    story.append(Paragraph("📊 Analyse Silhouette", CUSTOM_STYLES['BodyText']))
    try:
        silhouette_fig = evaluator.create_silhouette_plot(model_result)
        if silhouette_fig:
            story.append(_plotly_fig_to_pdf_image(silhouette_fig, width=6*inch, height=4*inch))
        else:
            story.append(Paragraph("Graphique silhouette non disponible", CUSTOM_STYLES['BodyText']))
    except Exception as e:
        logger.error(f"Erreur graphique silhouette: {e}")
        story.append(Paragraph("❌ Erreur génération graphique silhouette", CUSTOM_STYLES['BodyText']))

def _add_classification_visualizations(story, model_result: Dict[str, Any], evaluator: ModelEvaluationVisualizer):
    """Visualisations pour la classification"""
    story.append(Paragraph("📊 Métriques de Classification", CUSTOM_STYLES['BodyText']))
    # Ajouter d'autres visualisations spécifiques à la classification si nécessaire
    story.append(Paragraph("Visualisations détaillées disponibles dans l'interface web.", CUSTOM_STYLES['BodyText']))

def _add_regression_visualizations(story, model_result: Dict[str, Any], evaluator: ModelEvaluationVisualizer):
    """Visualisations pour la régression"""
    story.append(Paragraph("📈 Métriques de Régression", CUSTOM_STYLES['BodyText']))
    # Ajouter d'autres visualisations spécifiques à la régression si nécessaire
    story.append(Paragraph("Visualisations détaillées disponibles dans l'interface web.", CUSTOM_STYLES['BodyText']))

# --- Section conclusion ---
def _add_conclusion_section(story, model_result: Dict[str, Any]):
    """Section conclusion avec recommandations"""
    story.append(PageBreak())
    story.append(Paragraph("🎯 CONCLUSION ET RECOMMANDATIONS", CUSTOM_STYLES['SubTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    task_type = model_result.get('task_type', 'inconnu')
    metrics = model_result.get('metrics', {})
    model_name = model_result.get('model_name', 'Modèle')
    
    # Évaluation globale
    evaluation_text = f"Le modèle {model_name} a démontré des performances "
    
    if task_type == 'classification':
        accuracy = metrics.get('accuracy', 0)
        if accuracy >= 0.9:
            evaluation_text += "excellentes en classification avec une précision très élevée."
        elif accuracy >= 0.8:
            evaluation_text += "satisfaisantes en classification."
        else:
            evaluation_text += "nécessitant des améliorations en classification."
    
    elif task_type == 'regression':
        r2 = metrics.get('r2', 0)
        if r2 >= 0.8:
            evaluation_text += "excellentes en régression avec une forte capacité explicative."
        elif r2 >= 0.6:
            evaluation_text += "correctes en régression."
        else:
            evaluation_text += "limitées en régression."
    
    elif task_type == 'clustering':
        silhouette = metrics.get('silhouette_score', 0)
        if silhouette >= 0.7:
            evaluation_text += "excellentes en clustering avec une séparation nette des groupes."
        elif silhouette >= 0.5:
            evaluation_text += "satisfaisantes en clustering."
        else:
            evaluation_text += "nécessitant une optimisation en clustering."
    
    story.append(Paragraph(evaluation_text, CUSTOM_STYLES['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommandations
    story.append(Paragraph("💡 Recommandations:", CUSTOM_STYLES['BodyText']))
    recommendations = [
        "• Vérifier la qualité des données d'entraînement",
        "• Expérimenter avec d'autres algorithmes",
        "• Optimiser les hyperparamètres",
        "• Valider sur de nouvelles données",
        "• Surveiller les performances en production"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, CUSTOM_STYLES['BodyText']))

# --- Gestionnaire de contexte pour les fichiers temporaires ---
@contextlib.contextmanager
def _temp_directory_manager():
    """Gestionnaire de répertoire temporaire"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Répertoire temporaire nettoyé")
        except Exception as e:
            logger.warning(f"Impossible de nettoyer le répertoire temporaire: {e}")

# --- Fonction principale de génération de PDF ---
def generate_pdf_report(model_result: Dict[str, Any]) -> Optional[bytes]:
    """
    Génère un rapport PDF professionnel pour un modèle ML
    
    Args:
        model_result: Résultats du modèle au format dictionnaire
        
    Returns:
        bytes: Contenu du PDF ou None en cas d'erreur
    """
    if not model_result or not isinstance(model_result, dict):
        logger.error("Données du modèle invalides pour la génération du PDF")
        return None
    
    try:
        # Initialisation du buffer
        buffer = io.BytesIO()
        
        # Configuration du document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.7*inch,
            bottomMargin=0.7*inch,
            leftMargin=0.7*inch,
            rightMargin=0.7*inch
        )
        
        story = []
        model_name = model_result.get('model_name', 'Modèle')
        
        # Initialisation de l'évaluateur
        evaluator = ModelEvaluationVisualizer([model_result])
        
        # Construction du rapport
        _build_cover_page(story, model_result)
        _add_metrics_section(story, model_result)
        _add_model_configuration(story, model_result)
        _add_visualizations_section(story, model_result, evaluator)
        _add_conclusion_section(story, model_result)
        
        # Génération du PDF
        logger.info(f"Génération du PDF pour {model_name} avec {len(story)} éléments")
        
        doc.build(
            story,
            onFirstPage=lambda c, d: _header(c, d, model_name),
            onLaterPages=lambda c, d: _header(c, d, model_name)
        )
        
        # Récupération des bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        logger.info(f"✅ PDF généré avec succès pour {model_name} ({len(pdf_bytes)} bytes)")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de la génération du PDF: {e}")
        try:
            if 'buffer' in locals():
                buffer.close()
        except:
            pass
        return None

# --- Version simplifiée pour tests ---
def generate_basic_pdf_report(model_result: Dict[str, Any]) -> Optional[bytes]:
    """Version simplifiée pour tests rapides"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        story.append(Paragraph("Rapport ML Simplifié", CUSTOM_STYLES['MainTitle']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Modèle: {model_result.get('model_name', 'N/A')}", CUSTOM_STYLES['BodyText']))
        story.append(Paragraph(f"Type: {model_result.get('task_type', 'N/A')}", CUSTOM_STYLES['BodyText']))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Erreur PDF basique: {e}")
        return None

# --- Fonction utilitaire pour prévisualisation ---
def get_report_summary(model_result: Dict[str, Any]) -> Dict[str, Any]:
    """Retourne un résumé du rapport"""
    if not model_result:
        return {"error": "Données manquantes"}
    
    metrics = model_result.get('metrics', {})
    task_type = model_result.get('task_type', 'inconnu')
    
    summary = {
        "model_name": model_result.get('model_name', 'Inconnu'),
        "task_type": task_type,
        "training_time": model_result.get('training_time', 0),
        "main_metric": None,
        "metric_value": 0,
        "status": "inconnu"
    }
    
    # Détermination de la métrique principale
    if task_type == 'classification':
        summary["main_metric"] = "accuracy"
        summary["metric_value"] = metrics.get('accuracy', 0)
    elif task_type == 'regression':
        summary["main_metric"] = "r2"
        summary["metric_value"] = metrics.get('r2', 0)
    elif task_type == 'clustering':
        summary["main_metric"] = "silhouette_score"
        summary["metric_value"] = metrics.get('silhouette_score', 0)
    
    # Statut de performance
    if summary["metric_value"] is not None:
        if summary["metric_value"] > 0.8:
            summary["status"] = "excellent"
        elif summary["metric_value"] > 0.6:
            summary["status"] = "bon"
        else:
            summary["status"] = "à améliorer"
    
    return summary