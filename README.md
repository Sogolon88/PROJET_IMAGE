# Coin Recognition
Système de reconnaissance et denombrement automatique de pièces de monnaie (euro) en utilisant des algorithmes de traitement d'image.

## Description
Ce projet permet de détecter, identifier et compter le nombre de pièces de monnaie à partir d'images, en utilisant des techniques de traitement d'image classiques avec OpenCV. Il est capable de segmenter les pièces, d'extraire leurs caractéristiques visuelles (forme, taille, couleur) et de les compter.

## Technologies

- Python 3.11
- OpenCV — traitement d'image et détection de contours
- NumPy — manipulation des tableaux et calculs numériques

## Structure du projet

```
coin-recognition/
├── src/
│   ├── detection.py       # Détection et segmentation des pièces
│   ├── comptage.py        # Classification par caractéristiques
│   └── utils.py           # Fonctions utilitaires
├── data/
│   ├── base_valid/        
|   |   ├──image_raw/      # images de validations
|   |   ├──labels/         # resultats annotés
|   |
|   ├──base_test/          # images de validations
|   └── /image_raw/        # images de test
|   └── /labels/           # resultats annotés
├── requirements.txt
└── README.md
```

## Installation:

Cloner le dépôt : `git clone https://github.com/Sogolon88/PROJET_IMAGE.git`
Créer un environnement virtuel et installer les dépendances :
bashpython -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
🚀 Utilisation
bashpython src/detection.py --image images/samples/piece.jpg

```
## 🔍 Fonctionnement

1. **Prétraitement** — conversion en niveaux de gris, flou gaussien, seuillage.
2. **Détection** — détection de contours circulaires via les algorithmes de Canny et la transformé de Hough
3. **Extraction** — calcul du rayon, de la couleur moyenne et du ratio de forme
4. **Classification** — comparaison aux caractéristiques de référence par valeur faciale

# Objectif : un algo non supervisé pour compter le nombre de pieces dans l'image et pkus faire 
preparation des données(images pour la verité terrain Labelme).

# etapes pretraitement: Image brute

reduction de bruit local
Image en niveau de gris
flu gaussian pour la reducction du bruit
     ↓
Gaussien
     ↓
     Otsu # pour le moment non utilisé
     ↓
Canny
     ↓
Hough          → détecte les cercles (position + rayon)
     ↓
Classification → identifie chaque pièce via le rayon
     ↓
Comptage + Somme
Base de validation et base de test
```


##  Dépendances
opencv-python
numpy

## Equipe de projet:

- KEITA Fode Laye
