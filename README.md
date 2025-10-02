# DataLab Pro 🧪

**DataLab Pro** est une plateforme d'analyse de données et de Machine Learning automatisé construite avec Streamlit. Elle permet de charger, d'explorer, de prétraiter des données, ainsi que d'entraîner et d'évaluer des modèles de classification, de régression et de clustering.

## Architecture

Le projet suit une architecture modulaire pour une séparation claire des responsabilités :

```
app-analyse/
├── src/
│   ├── app/          # Interface utilisateur Streamlit
│   ├── config/       # Configuration de l'application
│   ├── data/         # Chargement et prétraitement des données
│   ├── models/       # Logique d'entraînement et catalogue de modèles
│   ├── evaluation/   # Calcul des métriques et visualisations
│   ├── monitoring/   # Détection de dérive et surveillance
│   └── shared/       # Modules partagés (état, logging)
├── .env            # Fichier pour les variables d'environnement
├── requirements.txt  # Dépendances Python
├── Dockerfile        # Fichier de build Docker
└── docker-compose.yml # Orchestration des services
```

## 🚀 Démarrage Rapide

### 1. Prérequis

- Python 3.11+
- Docker & Docker Compose
- Un client PostgreSQL (optionnel, pour MLflow)

### 2. Installation Locale

1.  **Clonez le projet :**
    ```bash
    git clone <repository_url>
    cd app-analyse
    ```

2.  **Créez un environnement virtuel et installez les dépendances :**
    ```bash
    python -m venv env
    source env/bin/activate  # sur Windows: env\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configurez l'environnement :**
    - Créez un fichier `.env` à la racine du projet.
    - Ajoutez vos configurations, notamment pour MLflow si vous l'utilisez :
      ```env
      MLFLOW_TRACKING_URI=postgresql+psycopg2://user:password@host:port/dbname
      ```

4.  **Lancez l'application :**
    ```bash
    streamlit run src/app/main.py
    ```

### 3. Démarrage avec Docker

Cette méthode est recommandée pour un environnement de production reproductible.

1.  **Assurez-vous que votre fichier `.env` est configuré.** Le `docker-compose.yml` l'utilisera.

2.  **Lancez les services :**
    - Pour lancer l'application, la base de données et MLflow :
      ```bash
      docker-compose up --build
      ```
    - L'application sera disponible sur `http://localhost:8501`.
    - L'interface MLflow sera sur `http://localhost:5000`.

## Utilisation de l'Application

1.  **Accueil** : Chargez votre jeu de données (CSV, Parquet, Excel).
2.  **Dashboard** : Explorez les données via les onglets (qualité, analyse univariée, corrélations, etc.).
3.  **Entraînement** : Configurez votre expérimentation ML (cible, features, modèles) et lancez l'entraînement.
4.  **Évaluation** : Comparez les modèles, analysez les métriques et visualisez les résultats détaillés.
