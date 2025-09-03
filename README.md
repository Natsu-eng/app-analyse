# Application Data Science avec Streamlit

Une application moderne, modulaire et évolutive construite avec Streamlit pour :

- **l'importation et l'exploration de données**
- **l'analyse exploratoire univariée et bivariée**
- **la régression linéaire avec diagnostics**
- **la réalisation de prédictions**

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
- **Gestion des valeurs manquantes (NaN)** avec plusieurs stratégies
- **Mise en cache** des données pour de meilleures performances
- **Aperçu** des 100 premières lignes

### 📊 Analyse exploratoire univariée
- Détection automatique du **type de variable** (qualitative / quantitative)
- Qualitatives : effectifs, proportions, barplots, pie charts interactifs
- Quantitatives : moyenne, médiane, écart-type, quantiles, skewness, kurtosis
- Visualisations : histogrammes, courbes de densité, boxplots, QQplots
- Tests de normalité : **Shapiro-Wilk**, **Kolmogorov–Smirnov**

### 🔁 Analyse bivariée
- Quantitative vs quantitative : scatterplots interactifs, corrélations (**Pearson**, **Spearman**, **Kendall**)
- Qualitative vs qualitative : tableaux de contingence, **Khi²**, **V de Cramer**, **Tschuprow**
- Qualitative vs quantitative : boxplots, tableaux de moyennes/médianes, tests adaptés :
  - 2 modalités → **Student**, **Wilcoxon**
  - >2 modalités → **ANOVA**, **Welch**, **Kruskal–Wallis**

### 📈 Régression linéaire
- Régression **simple** et **multiple**
- Vérification des hypothèses : résidus, **normalité**, **homoscédasticité**, **multicolinéarité (VIF)**
- Résultats textuels et graphiques interactifs

### 🤖 Prédictions
- Importation de **nouvelles données**
- Génération des **prédictions** avec le modèle entraîné
- Résultats affichés sous forme de **tableaux** et prêts pour la visualisation

## 🧱 Structure du projet
- `app.py` : point d’entrée principal (interface Streamlit)
- `utils.py` : fonctions utilitaires (import, nettoyage, typage)
- `plots.py` : création des graphiques Plotly
- `analyses.py` : calculs statistiques et tests analytiques
- `models.py` : modèles de régression et prédictions
- `.streamlit/config.toml` : thème de l’application

## ⭐ Points forts
- Interface **moderne et responsive** (style dashboard)
- Visualisations **interactives** et **dynamiques** (Plotly)
- Architecture **modulaire** et **évolutive**
- Gestion de **gros volumes** de données grâce au **caching**
- Code **lisible**, **maintenable** et **extensible**

## 📌 Remarques
Ce projet constitue une base solide pour développer un tableau de bord de data science, adapté aussi bien à l’entreprise qu’à la recherche ou l’enseignement.
