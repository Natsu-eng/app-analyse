import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepInFrame
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
STYLES = getSampleStyleSheet()

# --- Fonctions de dessin pour l'en-tête et le pied de page ---
def _header(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, doc.height + doc.topMargin - 0.5*inch, "Rapport d'Audit de Modèle - DataLab Pro")
    canvas.restoreState()

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 0.75 * inch, f"Page {doc.page}")
    canvas.drawRightString(doc.width + doc.leftMargin - inch, 0.75 * inch, f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    canvas.restoreState()

# --- Fonctions utilitaires pour les graphiques ---
def _plot_to_image(fig, width=7*inch, height=5*inch):
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)

# --- Sections du rapport ---
def _build_cover_page(story, model_name, task_type):
    story.append(Paragraph("Rapport d'Audit de Modèle", STYLES['h1']))
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(f"Modèle: {model_name}", STYLES['h2']))
    story.append(Paragraph(f"Type de Tâche: {task_type.capitalize()}", STYLES['h2']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Date du rapport: {datetime.now().strftime('%Y-%m-%d')}", STYLES['Normal']))
    story.append(PageBreak())

def _add_metrics_and_config(story, result):
    story.append(Paragraph("1. Résumé et Configuration", STYLES['h2']))
    # ... (Code pour les métriques et la config)

def _add_visualizations(story, result):
    story.append(PageBreak())
    story.append(Paragraph("2. Visualisations de Performance", STYLES['h2']))
    task_type = result.get("task_type")
    eval_data = result.get("eval_data", {})

    if task_type == 'classification':
        _add_classification_plots(story, eval_data, result['metrics'])
    elif task_type == 'regression':
        _add_regression_plots(story, eval_data)
    elif task_type == 'unsupervised':
        _add_clustering_plots(story, eval_data)

def _add_classification_plots(story, eval_data, metrics):
    # Matrice de confusion
    story.append(Paragraph("Matrice de Confusion", STYLES['h3']))
    cm_data = metrics.get('confusion_matrix')
    class_names = list(metrics.get('class_metrics', {}).keys())
    if cm_data and class_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(np.array(cm_data), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        story.append(_plot_to_image(fig, width=6*inch, height=4.5*inch))
    story.append(Spacer(1, 0.25*inch))

    # Courbes ROC et PR
    y_true, y_proba = eval_data.get('y_true'), eval_data.get('y_proba')
    if y_true is not None and y_proba is not None and y_proba.shape[1] > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (aire = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_title('Courbe ROC')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.legend(loc="lower right")
        story.append(_plot_to_image(fig, width=5*inch, height=3.5*inch))

def _add_regression_plots(story, eval_data):
    y_true, y_pred = eval_data.get('y_true'), eval_data.get('y_pred')
    if y_true is None or y_pred is None: return

    # Prédictions vs Vraies valeurs
    story.append(Paragraph("Prédictions vs Vraies Valeurs", STYLES['h3']))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Vraies Valeurs')
    ax.set_ylabel('Prédictions')
    ax.set_title('Prédictions vs Vraies Valeurs')
    story.append(_plot_to_image(fig, width=5*inch, height=4*inch))
    story.append(Spacer(1, 0.25*inch))

    # Distribution des résidus
    residuals = y_true - y_pred
    story.append(Paragraph("Distribution des Résidus", STYLES['h3']))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Erreur de prédiction (Résidus)')
    ax.set_title('Distribution des Résidus')
    story.append(_plot_to_image(fig, width=5*inch, height=4*inch))

def _add_clustering_plots(story, eval_data):
    X, labels = eval_data.get('X_processed'), eval_data.get('labels')
    if X is None or labels is None: return

    story.append(Paragraph("Visualisation des Clusters (PCA)", STYLES['h3']))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    ax.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels).tolist(), title="Clusters")
    story.append(_plot_to_image(fig, width=6*inch, height=4.5*inch))

def _add_feature_importance(story, result):
    # ... (Code de la fonction _add_feature_importance_plot précédente)
    pass

def generate_pdf_report(model_result: dict) -> bytes:
    """Génère un rapport PDF professionnel et complet."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=inch, bottomMargin=inch)
    
    story = []
    _build_cover_page(story, model_result.get("model_name", "N/A"), model_result.get("task_type", "N/A"))
    _add_metrics_and_config(story, model_result)
    _add_visualizations(story, model_result)
    # _add_feature_importance(story, model_result)

    doc.build(story, onFirstPage=_header, onLaterPages=_header)
    buffer.seek(0)
    return buffer.read()
