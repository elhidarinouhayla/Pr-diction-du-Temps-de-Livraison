# Prediction du Temps de Livraison



 ## 📝 Concept du Projet

Ce projet vise à développer un modèle de Machine Learning capable de prédire le temps total de livraison d’une commande, depuis sa préparation jusqu’à sa réception par le client.

Cette solution vise à aider l’entreprise de logistique à :

- Anticiper les retards,
- Informer les clients en temps réel,
- Optimiser l’organisation des tournées de livraison.

Le modèle prédit la variable cible DeliveryTime à partir de plusieurs variables :
Distance_km, Traffic_Level, Vehicle_Type, Time_of_Day, Courier_Experience_yrs, Weather, Preparation_Time_min.




## Installation

1. Cloner le projet :  

git clone  https://github.com/elhidarinouhayla/Pr-diction-du-Temps-de-Livraison.git



2. Installer les dépendances :

Avant de lancer le projet, il faut installer les bibliothèques nécessaires (pandas, numpy, scikit-learn, matplotlib, seaborn...).
Pour cela, ouvrez le terminal dans le dossier du projet et tapez la commande suivante :
pip install -r requirements.txt
Cette commande va installer automatiquement toutes les librairies dont le projet a besoin.



## 📂 Structure du Projet


- 'EDA.ipynb' : Exploration des données et visualisations
- 'pipeline.py' : Fonctions de préparation des données, encodage, split train/test, normalisation et        selectKBest
-' test_pipeline.py' : Tests unitaires pour valider les fonctions du pipeline
- 'data.csv : Dataset utilisé
-' README.md' : Documentation du projet

 

### 🧠 Exploration des données :

Étapes principales :

- Analyse des donnees
- visualisation 
- Pipeline sklearn


### ⚙️ Pipeline :  

Étapes principales :

- Nettoyage des données : Traitement des valeurs manquantes
- Encodage : utilisation de OneHotEncoder pour les variables catégorielles
- Split des données : train_test_split
- Normalisation :   StandardScaler sur les variables numériques
- selectKBest :  Sélection de features


### 🧪 Tests unitaires :
Des tests unitaires ont été ajoutés pour assurer la fiabilité du projet :
- Vérification du format et des dimensions des données après le prétraitement,
- Vérification que la MAE maximale des modèles ne dépasse pas un seuil défini (7),
- Exécution automatisée avec pytest.

### 🤖 Modélisation 

Deux modèles de régression ont été testés et comparés :

- RandomForestRegressor

- SVR (Support Vector Regression)


### 📈 Résultats

Deux modèles de régression ont été testés et comparés à l’aide de la validation croisée (GridSearchCV) et évalués selon les métriques MAE et R².

| Modèle                             |     R²    |    MAE   |
| :--------------------------------- | :-------: | :------: |
| **SVR (Support Vector Regressor)** |   0.806   |   5.54   |
| **RandomForestRegressor**          |   0.756   |   6.58   |

🏆 Meilleur modèle : SVR
Le modèle SVR a obtenu les meilleures performances avec un R² = 0.80 et une MAE = 5.54, ce qui signifie qu’il explique environ 80 % de la variance du temps de livraison et présente une erreur moyenne faible.

Ce modèle a donc été retenu comme modèle final pour la prédiction du temps de livraison, car il offre :

- Une meilleure précision,

- Une meilleure généralisation,

Et une erreur absolue moyenne inférieure à celle du modèle RandomForest.




## Auteur

Nom : El hidari Nouhayla  








 

