# Application Data Science avec Streamlit

Une application moderne, modulaire et évolutive construite avec Streamlit pour :

- **l'importation et l'exploration de données**
- **l'analyse exploratoire univariée et bivariée**
- **la modélisation (classification et régression)**
- **la génération de rapports PDF**

Les graphiques interactifs sont basés sur **Plotly**.

## ⚙️ Installation et configuration

### 1) Créer et activer un environnement virtuel
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell : .venv\Scripts\Activate.ps1
source .venv/bin/activate  # Linux / macOS
```

### 2) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3) Exécuter l’application
```bash
streamlit run app.py
```

## ✨ Fonctionnalités principales

### 📥 Importation des données
- **Formats supportés** : CSV, Excel, Parquet, JSON
- **Nettoyage automatique** : détection des types mixtes, suppression des doublons.
- **Mise en cache** des données pour de meilleures performances
- **Aperçu interactif** et sécurisé avec AgGrid

### 📊 Analyse exploratoire
- Détection automatique du **type de variable** (catégoriel, numérique, date, etc.)
- Profil détaillé de chaque variable.
- Analyse bivariée avec tests statistiques appropriés (Pearson, Chi², ANOVA).
- Visualisations interactives : histogrammes, boxplots, scatterplots.

### 🤖 Modélisation et Évaluation
- Pipeline d'entraînement **robuste** pour la classification.
- **Gestion automatique** des cas limites : stratification, sur-échantillonnage (SMOTE).
- Calcul **sécurisé** des métriques (Accuracy, F1-Score, AUC ROC) avec gestion des erreurs.
- Tableau de bord d'évaluation complet avec graphiques (matrice de confusion, courbe ROC) et interprétabilité (SHAP).

### 📄 Rapports
- Génération de **rapports PDF** professionnels et complets.
- Inclusion des métriques, graphiques et notes d'évaluation automatiques.

## 🧱 Structure du projet
- `app.py` : Point d'entrée principal de l'application.
- `pages/` : Contient les différentes pages de l'application Streamlit.
- `components/` : Modules Streamlit réutilisables (sidebar, dashboard, evaluation...).
- `utils/` : Fonctions utilitaires (chargement de données, analyse, prétraitement).
- `ml/` : Logique du pipeline de machine learning (entraînement, narration).
- `plots/` : Fonctions dédiées à la création des graphiques Plotly.
- `assets/` : Fichiers statiques (CSS).

## ⭐ Points forts
- Interface **moderne et responsive** (style dashboard)
- Visualisations **interactives** et **dynamiques** (Plotly)
- Architecture **modulaire** et **évolutive**
- Gestion de **gros volumes** de données grâce au **caching** et au support (partiel) de **Dask**.
- Code **lisible**, **maintenable** et **extensible**

## 🚀 Améliorations et Robustesse
Cette version intègre des améliorations significatives pour garantir la stabilité et une expérience utilisateur de qualité professionnelle :

- **Chargement de Données Sécurisé** : Le chargement des fichiers prévient les erreurs de type mixte (`DtypeWarning`) en inspectant et en convertissant intelligemment les colonnes problématiques. L'utilisateur est informé de ces conversions.
- **Pipeline de ML Fiable** : L'entraînement des modèles est protégé contre les erreurs courantes :
  - La **stratification** est automatiquement désactivée si une classe a trop peu d'échantillons.
  - Le sur-échantillonnage **SMOTE** est remplacé par une méthode plus sûre (`RandomOverSampler`) lorsque la classe minoritaire est trop petite, évitant ainsi les crashs.
  - Le calcul de l'**AUC ROC** est ignoré proprement si une seule classe est présente dans les données de test.
- **Feedback Utilisateur Clair** : Des messages et des notes d'évaluation sont affichés dans l'interface et dans les rapports PDF pour expliquer toutes les décisions automatiques prises par l'application.
- **Affichage Stable** : Les tableaux de données (`st.dataframe`, `AgGrid`) sont protégés contre les erreurs `pyarrow` en forçant une conversion en `str` des colonnes à types mixtes avant l'affichage.

## 📌 Remarques
Ce projet constitue une base solide pour développer un tableau de bord de data science, adapté aussi bien à l’entreprise qu’à la recherche ou l’enseignement.