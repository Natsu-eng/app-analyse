import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
STYLES = getSampleStyleSheet()

# --- Styles personnalisés ---
def _create_custom_styles():
    styles = STYLES
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        textColor=colors.darkblue,
        spaceAfter=30,
    ))
    styles.add(ParagraphStyle(
        name='MetricValue',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.darkgreen,
    ))
    return styles

CUSTOM_STYLES = _create_custom_styles()

# --- En-tête et pied de page ---
def _header(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(inch, doc.height + doc.topMargin - 0.5*inch, "Rapport d'Évaluation de Modèle - ML Platform")
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(doc.width + inch, doc.height + doc.topMargin - 0.5*inch, 
                          f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    canvas.restoreState()

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawString(inch, 0.5*inch, f"Page {doc.page}")
    canvas.drawRightString(doc.width + inch, 0.5*inch, "Document confidentiel")
    canvas.restoreState()

# --- Conversion graphiques ---
def _plot_to_image(fig, width=6*inch, height=4*inch):
    """Convertit une figure matplotlib en image pour PDF"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return Image(buf, width=width, height=height)
    except Exception as e:
        logger.error(f"Erreur conversion graphique: {e}")
        return Paragraph(f"Erreur génération graphique: {e}", CUSTOM_STYLES['Normal'])

# --- Page de couverture ---
def _build_cover_page(story, model_name, task_type, metrics):
    story.append(Paragraph("RAPPORT D'ÉVALUATION DE MODÈLE", CUSTOM_STYLES['Title']))
    story.append(Spacer(1, inch))
    
    story.append(Paragraph("Informations du Modèle", CUSTOM_STYLES['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    info_data = [
        ["Nom du modèle:", model_name],
        ["Type de tâche:", task_type.upper()],
        ["Date de génération:", datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
    ]
    
    # Ajouter métrique principale selon le type
    if task_type == 'classification' and 'accuracy' in metrics:
        info_data.append(["Accuracy:", f"{metrics['accuracy']:.3f}"])
    elif task_type == 'regression' and 'r2' in metrics:
        info_data.append(["Score R²:", f"{metrics['r2']:.3f}"])
    
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(PageBreak())

# --- Métriques et configuration ---
def _add_metrics_and_config(story, result):
    story.append(Paragraph("1. MÉTRIQUES DE PERFORMANCE", CUSTOM_STYLES['Heading1']))
    story.append(Spacer(1, 0.3*inch))
    
    metrics = result.get('metrics', {})
    task_type = result.get('task_type', 'inconnu')
    
    if task_type == 'classification':
        _add_classification_metrics(story, metrics)
    elif task_type == 'regression':
        _add_regression_metrics(story, metrics)
    elif task_type == 'unsupervised':
        _add_clustering_metrics(story, metrics)
    
    # Configuration du modèle
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("2. CONFIGURATION DU MODÈLE", CUSTOM_STYLES['Heading1']))
    
    config_data = [
        ["Temps d'entraînement:", f"{result.get('training_time', 0):.2f} secondes"],
        ["Modèle:", result.get('model_name', 'N/A')],
    ]
    
    best_params = result.get('best_params', {})
    if best_params:
        for param, value in list(best_params.items())[:5]:  # Limiter aux 5 premiers
            config_data.append([f"Paramètre {param}:", str(value)])
    
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    story.append(config_table)

def _add_classification_metrics(story, metrics):
    metric_data = [
        ['Métrique', 'Valeur'],
        ['Accuracy', f"{metrics.get('accuracy', 0):.3f}"],
        ['F1-Score', f"{metrics.get('f1_score', 0):.3f}"],
        ['Précision', f"{metrics.get('precision', 0):.3f}"],
        ['Rappel', f"{metrics.get('recall', 0):.3f}"],
    ]
    
    if 'roc_auc' in metrics:
        metric_data.append(['AUC ROC', f"{metrics['roc_auc']:.3f}"])
    
    metric_table = Table(metric_data, colWidths=[1.5*inch, 1*inch])
    metric_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(metric_table)

def _add_regression_metrics(story, metrics):
    metric_data = [
        ['Métrique', 'Valeur'],
        ['R²', f"{metrics.get('r2', 0):.3f}"],
        ['MAE', f"{metrics.get('mae', 0):.3f}"],
        ['RMSE', f"{metrics.get('rmse', 0):.3f}"],
        ['MSE', f"{metrics.get('mse', 0):.3f}"],
    ]
    
    metric_table = Table(metric_data)
    metric_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(metric_table)

def _add_clustering_metrics(story, metrics):
    metric_data = [
        ['Métrique', 'Valeur'],
        ['Score de silhouette', f"{metrics.get('silhouette_score', 0):.3f}"],
        ['Nombre de clusters', f"{metrics.get('n_clusters', 0)}"],
    ]
    
    metric_table = Table(metric_data)
    story.append(metric_table)

# --- Visualisations ---
def _add_visualizations(story, result):
    story.append(PageBreak())
    story.append(Paragraph("3. VISUALISATIONS", CUSTOM_STYLES['Heading1']))
    
    task_type = result.get("task_type")
    metrics = result.get("metrics", {})
    
    # Pour l'instant, on se contente des métriques textuelles
    # Les graphiques nécessitent les données d'évaluation (y_true, y_pred, etc.)
    # qui ne sont pas forcément disponibles dans le résultat
    
    story.append(Paragraph("Graphiques non disponibles dans cette version", CUSTOM_STYLES['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Vous pourriez ajouter ici la génération de graphiques si les données sont disponibles
    # _add_classification_plots(story, result.get('eval_data', {}), metrics)

# --- Génération principale ---
def generate_pdf_report(model_result: dict) -> bytes:
    """Génère un rapport PDF professionnel et complet."""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            title=f"Rapport Modèle {model_result.get('model_name', '')}"
        )
        
        story = []
        
        # Page de couverture
        _build_cover_page(
            story, 
            model_result.get("model_name", "Modèle"), 
            model_result.get("task_type", "inconnu"),
            model_result.get("metrics", {})
        )
        
        # Métriques et configuration
        _add_metrics_and_config(story, model_result)
        
        # Visualisations
        _add_visualizations(story, model_result)
        
        # Génération du PDF
        doc.build(story, onFirstPage=_header, onLaterPages=_header)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF généré pour {model_result.get('model_name')}")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Erreur génération PDF: {e}")
        raise

# Version simplifiée pour test
def generate_basic_pdf_report(model_result: dict) -> bytes:
    """Version simplifiée pour tester rapidement"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        story.append(Paragraph("Rapport Modèle ML", CUSTOM_STYLES['Title']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Modèle: {model_result.get('model_name', 'N/A')}", CUSTOM_STYLES['Heading2']))
        story.append(Paragraph(f"Type: {model_result.get('task_type', 'N/A')}", CUSTOM_STYLES['Heading2']))
        
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Erreur PDF basique: {e}")
        raise