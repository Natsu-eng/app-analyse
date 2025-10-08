import pytest
from src.models.catalog import get_available_models, get_model_config

def test_model_config_descriptions():
    """Vérifie que chaque modèle dans MODEL_CATALOG a une description."""
    tasks = ["regression", "classification", "clustering"]
    for task in tasks:
        for model_name in get_available_models(task):
            config = get_model_config(task, model_name)
            assert config is not None, f"Configuration manquante pour {model_name} ({task})"
            assert "description" in config, f"Description manquante pour {model_name} ({task})"
            assert isinstance(config["description"], str) and len(config["description"]) > 0, f"Description invalide pour {model_name} ({task})"