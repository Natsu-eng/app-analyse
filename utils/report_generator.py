# E:\gemini\app-analyse\utils\report_generator.py

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

STYLES = getSampleStyleSheet()

def _plot_to_image_buffer(fig):
    """Convertit une figure matplotlib en buffer d'image pour ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def _add_title(story, model_name, task_type):
    story.append(Paragraph("Rapport d'Analyse du Modèle", STYLES['h1']))
    story.append(Paragraph(f"Modèle : {model_name}", STYLES['h2']))
    story.append(Paragraph(f"Type de Tâche : {task_type.capitalize()}", STYLES['h2']))
    story.append(Spacer(1, 0.25 * inch))

def _add_summary_metrics(story, metrics, task_type):
    story.append(Paragraph("Métriques de Performance Globales", STYLES['h3']))
    data = [['Métrique', 'Score']]
    
    # Métriques formatées pour une meilleure lisibilité
    if task_type == "classification":
        data.append(['Accuracy', f"{metrics.get('accuracy', 0):.4f}"])
        roc_auc = metrics.get('roc_auc', 'N/A')
        data.append(['ROC AUC', f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else roc_auc])
        data.append(['Precision (pondérée)', f"{metrics.get('precision', 0):.4f}"])
        data.append(['Recall (pondéré)', f"{metrics.get('recall', 0):.4f}"])
        data.append(['F1-Score (pondéré)', f"{metrics.get('f1_score', 0):.4f}"])
    elif task_type == "regression":
        data.append(['R² Score', f"{metrics.get('r2', 0):.4f}"])
        data.append(['RMSE', f"{metrics.get('rmse', 0):.4f}"])
        data.append(['MAE', f"{metrics.get('mae', 0):.4f}"])
    elif task_type == "unsupervised":
        data.append(["Nombre de Clusters", f"{metrics.get('n_clusters', 'N/A'):.0f}"])
        data.append(["Score de Silhouette", f"{metrics.get('silhouette_score', 0):.4f}"])
        data.append(["Score Davies-Bouldin", f"{metrics.get('davies_bouldin_score', 0):.4f}"])
        data.append(["Score Calinski-Harabasz", f"{metrics.get('calinski_harabasz_score', 0):.2f}"])

    table = Table(data, colWidths=[2.5 * inch, 1.5 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

def _add_confusion_matrix(story, oof_preds, label_encoder):
    story.append(PageBreak())
    story.append(Paragraph("Matrice de Confusion (Out-of-Fold)", STYLES['h3']))
    
    y_true = oof_preds['y_true']
    y_pred = oof_preds['y_pred']
    
    labels = np.unique(np.concatenate((y_true, y_pred)))
    class_names = label_encoder.inverse_transform(labels) if label_encoder else [str(l) for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Vraies valeurs")
    ax.set_title("Matrice de Confusion")
    
    img_buffer = _plot_to_image_buffer(fig)
    story.append(Image(img_buffer, width=6 * inch, height=4.5 * inch))
    story.append(Spacer(1, 0.25 * inch))

def _add_cluster_plot(story, X_processed, labels):
    story.append(PageBreak())
    story.append(Paragraph("Visualisation des Clusters (PCA)", STYLES['h3']))
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    ax.set_title("Clusters projetés sur les 2 premières composantes PCA")
    ax.grid(True)
    
    # Ajouter une légende
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    img_buffer = _plot_to_image_buffer(fig)
    story.append(Image(img_buffer, width=6 * inch, height=4.5 * inch))

def generate_pdf_report(model_result: dict) -> bytes:
    """Génère un rapport PDF complet à partir des résultats du pipeline."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []

    model_name = model_result.get("model_name", "Inconnu")
    task_type = model_result.get("config", {}).get("task_type", "classification")
    metrics = model_result.get("metrics_summary", {}).get("global_metrics", {})

    _add_title(story, model_name, task_type)

    if not metrics:
        story.append(Paragraph("Aucune métrique d'évaluation n'a pu être calculée.", STYLES['h3']))
    else:
        _add_summary_metrics(story, metrics, task_type)

    # --- Ajout des graphiques spécifiques à la tâche ---
    if task_type == "classification":
        oof_preds = model_result.get("oof_predictions")
        if oof_preds and oof_preds.get('y_true') is not None:
            _add_confusion_matrix(story, oof_preds, model_result.get("label_encoder"))
    
    elif task_type == "unsupervised":
        X_processed = model_result.get("X_processed")
        labels = model_result.get("labels")
        if X_processed is not None and labels is not None:
            _add_cluster_plot(story, X_processed, labels)

            # --- Visualisations --- 
            elements.append(Paragraph("Visualisations Clés", styles['h2']))

            # Matrice de Confusion
            if y_test_encoded is not None and y_pred_encoded is not None:
                cm_fig = plot_confusion_matrix(y_test_encoded, y_pred_encoded, class_labels=label_encoder.classes_.tolist() if label_encoder else None)
                elements.append(Image(io.BytesIO(cm_fig.to_image(format="png")), width=400, height=300))
                elements.append(Paragraph("Matrice de Confusion", styles['Normal']))

            # Courbes ROC et Précision-Rappel
            if y_test_encoded is not None and y_proba is not None:
                roc_fig = plot_roc_curve(y_test_encoded, y_proba, class_labels=label_encoder.classes_.tolist() if label_encoder else None)
                pr_fig = plot_precision_recall_curve(y_test_encoded, y_proba, class_labels=label_encoder.classes_.tolist() if label_encoder else None)
                elements.append(Image(io.BytesIO(roc_fig.to_image(format="png")), width=400, height=300))
                elements.append(Paragraph("Courbe ROC", styles['Normal']))
                elements.append(Image(io.BytesIO(pr_fig.to_image(format="png")), width=400, height=300))
                elements.append(Paragraph("Courbe Précision-Rappel", styles['Normal']))

            # SHAP Plotting
            if X_test_processed is not None and model is not None and task_type != 'unsupervised':
                try:
                    # Calcul des valeurs SHAP
                    model_to_explain = model.named_steps['model'] if 'model' in model.named_steps else model # Gérer si le modèle n'est pas dans un pipeline nommé
                    
                    # S'assurer que X_test_processed est un DataFrame Pandas pour SHAP
                    if not isinstance(X_test_processed, pd.DataFrame):
                        X_test_processed = pd.DataFrame(X_test_processed)

                    # Sélectionner uniquement les colonnes numériques pour SHAP si elles existent
                    X_shap = X_test_processed.select_dtypes(include=np.number)
                    if X_shap.empty:
                        logger.warning("Aucune colonne numérique pour SHAP dans le rapport PDF.")
                    else:
                        # Utiliser un explainer approprié
                        if "XGB" in model_to_explain.__class__.__name__ or "LGBM" in model_to_explain.__class__.__name__ or "RandomForest" in model_to_explain.__class__.__name__ or "GradientBoosting" in model_to_explain.__class__.__name__:
                            explainer = shap.TreeExplainer(model_to_explain)
                        else:
                            # Pour les autres modèles, utiliser KernelExplainer avec un échantillon
                            # Attention: KernelExplainer peut être très lent
                            X_sample = shap.sample(X_shap, min(100, len(X_shap)))
                            explainer = shap.KernelExplainer(model_to_explain.predict, X_sample)

                        shap_values = explainer(X_shap)
                        
                        # Générer le graphique SHAP (retourne une figure matplotlib)
                        shap_fig_mpl = plot_shap_summary(shap_values, X_shap)
                        
                        # Convertir la figure matplotlib en image pour ReportLab
                        buf = io.BytesIO()
                        shap_fig_mpl.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0);
                        elements.append(Image(buf, width=400, height=300))
                        elements.append(Paragraph("Interprétation SHAP (Impact des Features)", styles['Normal']))
                        plt.close(shap_fig_mpl) # Fermer la figure pour libérer la mémoire

                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique SHAP pour le rapport PDF : {e}", exc_info=True)
                    elements.append(Paragraph(f"Impossible de générer le graphique SHAP : {e}", styles['Normal']))

            # --- Notes ---

    doc.build(story)
    buffer.seek(0)
    return buffer.read()