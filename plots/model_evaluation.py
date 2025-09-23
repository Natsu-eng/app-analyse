import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import shap # Assuming shap is installed for shap plots
import streamlit as st # For st.cache_data if needed, but not directly in plot functions
import matplotlib.pyplot as plt # For SHAP plots which often use matplotlib
from typing import List # <--- ADDED THIS IMPORT

from . import templates # Assuming templates module exists and defines DATALAB_TEMPLATE

def plot_class_distribution(y_train, y_test, class_labels=None):
    """Affiche la distribution des classes dans les jeux d'entraînement et de test."""
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    df_train = pd.DataFrame({'labels': train_counts.index, 'count': train_counts.values, 'dataset': 'Train'})
    df_test = pd.DataFrame({'labels': test_counts.index, 'count': test_counts.values, 'dataset': 'Test'})
    
    df_plot = pd.concat([df_train, df_test])

    if class_labels is not None:
        df_plot['labels'] = df_plot['labels'].apply(lambda x: class_labels[x] if isinstance(x, int) else x)

    fig = px.bar(df_plot, x='labels', y='count', color='dataset', barmode='group',
                 title="Distribution des Classes (Train vs. Test)",
                 labels={'labels': 'Classe', 'count': 'Nombre d\'échantillons'})
    fig.update_layout(template=templates.DATALAB_TEMPLATE)
    return fig

def plot_metrics_comparison(metrics_df: pd.DataFrame):
    """Affiche un bar chart comparant les métriques de plusieurs modèles."""
    # This function's implementation was not provided, keeping as placeholder
    st.warning("plot_metrics_comparison not implemented yet.")
    return go.Figure() # Return empty figure to avoid errors

def plot_regression_results(y_test, y_pred):
    """Affiche les résultats pour une tâche de régression."""
    # This function's implementation was not provided, keeping as placeholder
    st.warning("plot_regression_results not implemented yet.")
    return go.Figure() # Return empty figure to avoid errors

def plot_classification_results(y_test, y_pred, y_proba=None, class_labels=None):
    """Affiche les résultats graphiques pour une tâche de classification."""
    # This function's implementation was not provided, keeping as placeholder
    st.warning("plot_classification_results not implemented yet.")
    return go.Figure() # Return empty figure to avoid errors

def plot_feature_importance(feature_importance: pd.Series):
    """
    Affiche un bar chart de l'importance des features.
    Args:
        feature_importance: Une Series Pandas avec les noms des features comme index et leurs importances comme valeurs.
    """
    if feature_importance.empty:
        st.warning("Aucune importance de feature à afficher.")
        return go.Figure()

    fig = px.bar(feature_importance.sort_values(ascending=True),
                 orientation='h',
                 title='Importance des Features',
                 labels={'value': 'Importance', 'index': 'Feature'})
    fig.update_layout(template=templates.DATALAB_TEMPLATE)
    return fig

# --- New Plotting Functions ---
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_labels: List[str] = None):
    """
    Plots a confusion matrix.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_labels: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    if class_labels is None:
        class_labels = np.unique(y_true).astype(str).tolist()

    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=class_labels, y=class_labels,
                    text_auto=True, # Display values in cells
                    color_continuous_scale="Viridis")
    fig.update_layout(title_text='**Matrice de Confusion**', template=templates.DATALAB_TEMPLATE)
    return fig

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, class_labels: List[str] = None):
    """
    Plots the ROC curve for binary or multi-class classification.
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        class_labels: List of class names.
    """
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

    if len(np.unique(y_true)) == 2: # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
    else: # Multi-class classification (One-vs-Rest)
        if class_labels is None:
            class_labels = np.unique(y_true).astype(str).tolist()
        
        for i, class_name in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=i)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve for {class_name}'))

    fig.update_layout(title_text='**Courbe ROC**', 
                      xaxis_title='False Positive Rate', 
                      yaxis_title='True Positive Rate', 
                      yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis=dict(constrain='domain'),
                      template=templates.DATALAB_TEMPLATE)
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, class_labels: List[str] = None):
    """
    Plots the Precision-Recall curve for binary or multi-class classification.
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        class_labels: List of class names.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = go.Figure()

    if len(np.unique(y_true)) == 2: # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (AP={avg_precision:.2f})'))
    else: # Multi-class classification (One-vs-Rest)
        if class_labels is None:
            class_labels = np.unique(y_true).astype(str).tolist()

        for i, class_name in enumerate(class_labels):
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, i], pos_label=i)
            avg_precision = average_precision_score(y_true, y_proba[:, i], pos_label=i)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve for {class_name} (AP={avg_precision:.2f})'))

    fig.update_layout(title_text='**Courbe Précision-Rappel**', 
                      xaxis_title='Recall', 
                      yaxis_title='Precision', 
                      yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis=dict(constrain='domain'),
                      template=templates.DATALAB_TEMPLATE)
    return fig

def plot_shap_summary(shap_values: np.ndarray, features_df: pd.DataFrame):
    """
    Plots a SHAP summary plot.
    Args:
        shap_values: SHAP values array.
        features_df: DataFrame of features (used for feature names).
    """
    # SHAP plots often use matplotlib, so we need to handle that
    shap.summary_plot(shap_values, features_df, show=False)
    st.pyplot(plt.gcf()) # Get current matplotlib figure and display it in Streamlit
    plt.clf() # Clear the figure to prevent overlap

def plot_shap_dependence(shap_values: np.ndarray, features_df: pd.DataFrame, feature_name: str):
    """
    Plots a SHAP dependence plot for a specific feature.
    Args:
        shap_values: SHAP values array.
        features_df: DataFrame of features.
        feature_name: Name of the feature to plot.
    """
    shap.dependence_plot(feature_name, shap_values, features_df, show=False)
    st.pyplot(plt.gcf())
    plt.clf()