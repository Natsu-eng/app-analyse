import pytest
import numpy as np
import pandas as pd
from src.evaluation.model_plots import ModelEvaluationVisualizer

def test_create_learning_curve_plot():
    model_result = {
        "model_name": "Test Model",
        "task_type": "classification",
        "model": None,  # Remplacer par un modèle réel pour les tests
        "X_train": pd.DataFrame(np.random.rand(100, 5)),
        "y_train": pd.Series(np.random.randint(0, 2, 100)),
        "X_test": pd.DataFrame(np.random.rand(20, 5)),
        "y_test": pd.Series(np.random.randint(0, 2, 20))
    }
    visualizer = ModelEvaluationVisualizer([model_result])
    fig = visualizer.create_learning_curve_plot(model_result)
    assert fig is not None, "La courbe d'apprentissage doit retourner une figure Plotly"