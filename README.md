# Prédiction des Prix des Maisons en Île-de-France

## Description du Projet

Ce projet vise à développer un modèle de machine learning capable de prédire les prix des maisons en Île-de-France en utilisant les données de la Demande de Valeurs Foncières (DVF). Le modèle utilise un RandomForestRegressor pour faire des prédictions basées sur des caractéristiques telles que la surface réelle bâtie, le nombre de pièces principales, le code postal, le type de local et la surface du terrain.

## Contexte et Motivation

Chez **Prestige Property Solution**, nous sommes passionnés par la fourniture de solutions immobilières de qualité supérieure qui répondent aux besoins uniques de nos clients. En tant qu'experts dans le domaine de l'immobilier, nous comprenons que l'estimation précise des prix des propriétés est essentielle pour prendre des décisions éclairées, que ce soit pour l'achat, la vente ou l'investissement immobilier.

**Prestige Property Solution** s'engage à utiliser des technologies avancées pour améliorer l'expérience de nos clients et optimiser nos services. Dans ce contexte, nous avons entrepris un projet ambitieux visant à développer un modèle de machine learning capable de prédire avec précision les prix des maisons en Île-de-France. Cette initiative s'inscrit dans notre volonté constante d'innovation et de perfectionnement de nos offres.

Les objectifs de ce projet sont multiples :
- **Aider nos clients** à obtenir des estimations de prix précises et rapides, leur permettant de prendre des décisions immobilières informées.
- **Optimiser nos processus internes** en automatisant l'évaluation des propriétés, réduisant ainsi les coûts et le temps nécessaires pour effectuer des estimations.
- **Renforcer notre position** en tant que leader du marché immobilier en offrant des services basés sur des technologies de pointe.

En combinant notre expertise en immobilier avec les puissantes capacités des technologies de machine learning, nous visons à transformer la manière dont les prix des propriétés sont estimés, offrant ainsi à nos clients un avantage compétitif significatif.

## Données Utilisées

Les données utilisées dans ce projet proviennent de la Demande de Valeurs Foncières (DVF) pour l'Île-de-France en 2020. Elles incluent des informations détaillées sur les transactions immobilières telles que la surface réelle bâtie, le nombre de pièces principales, le code postal, le type de local et la surface du terrain.

## Structure du Projet

- `download_clean_and_visualize.py` : Script principal pour le téléchargement, le nettoyage, l'encodage, l'entraînement du modèle et la visualisation des résultats.
- `data/` : Dossier pour stocker les données nettoyées.
- `output/` : Dossier pour stocker les visualisations générées.

## Instructions d'Installation

1. **Cloner le dépôt** :

```
git clone https://github.com/your-repository.git
cd your-repository
```
## Créer et activer un environnement virtuel
```
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

## Installer les dépendances
```
pip install -r requirements.txt
```

## Exécuter le script principal
```
python download_data.py
```
