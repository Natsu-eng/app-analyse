# 📊 Application Data Science avec Streamlit

Une application **moderne, modulaire et évolutive** construite avec **Streamlit** pour :  
- l’importation et l’exploration de données,  
- l’analyse exploratoire univariée et bivariée,  
- la régression linéaire avec diagnostics,  
- la réalisation de prédictions,  
avec des graphiques interactifs basés sur **Plotly**.

---

## ⚙️ Installation et configuration

### 1. Créer et activer un environnement virtuel
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell : .venv\Scripts\Activate.ps1
source .venv/bin/activate  # Linux / macOS

### 2. Installer les dépendances
pip install -r requirements.txt

### 3. Exécution de l’application
streamlit run app.py

# ✨ Fonctionnalités principales
#📥 Importation des données

-Support des formats : CSV, Excel, Parquet, JSON.
-Gestion des valeurs manquantes (NaN).
-Mise en cache des données pour de meilleures performances.
-Aperçu des 100 premières lignes.

## 📊 Analyse exploratoire univariée

Détection automatique du type de variable.
Qualitatives : effectifs, proportions, barplots, pie charts interactifs
Quantitatives : moyenne, médiane, écart-type, quantiles, skewness, kurtosis.
Visualisations : histogrammes, courbes de densité, boxplots, QQplots.
Tests de normalité : Shapiro-Wilk, Kolmogorov-Smirnov.

 ## Analyse bivariée

Quantitative vs quantitative : scatterplots interactifs, corrélations (Pearson, Spearman, Kendall).
Qualitative vs qualitative : tableaux de contingence, Khi², V de Cramer, Tschuprow.
Qualitative vs quantitative : boxplots, tableaux de moyennes/médianes, tests statistiques adaptés :
Student, Wilcoxon.
ANOVA, Welch, Kruskal-Wallis.

📈 ## Régression linéaire

Régression simple et multiple.
Vérification des hypothèses : résidus, normalité, homoscédasticité, multicolinéarité (VIF).
Graphiques interactifs des résultats.

###Prédictions

Importation de nouvelles données.
Génération des prédictions.
Résultats affichés sous forme de tableaux et graphiques interactifs.

## Structure du projet

app.py : point d’entrée principal (interface Streamlit).
utils.py : fonctions utilitaires (import, nettoyage, typage).
plots.py : création des graphiques Plotly.
nalyses.py : calculs statistiques et tests analytiques.
models.py : modèles de régression et prédictions.

## Points forts

Interface moderne et responsive.
Visualisations interactives et dynamiques.
Architecture modulaire et évolutive.
Gestion de gros volumes de données grâce au caching.
Code lisible, maintenable et extensible.

📌 ## Remarques

Ce projet constitue une base solide pour développer un tableau de bord de data science adapté aussi bien à l’entreprise qu’à la recherche ou l’enseignement.