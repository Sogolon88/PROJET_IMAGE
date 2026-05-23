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
projet_Image/
├── src/
│   ├── algorithme.py        # Pipeline de traitement : HSV, CLAHE, Hough, NMS, filtres couleur
│   ├── classification.py    # Classification des pièces par couleur et ratio de taille
│   ├── evaluation.py        # Métriques d'évaluation (TP, FP, FN, F1, MSE)
│   └── utiles.py            # Chargement des images et des labels
├── data/
│   ├── base_validation/
│   │   ├── images/          # 150 images de validation (image1.png … image150.png)
│   │   └── labels/          # Annotations JSON (vérité terrain)
│   ├── base_test/
│   │   ├── images/          # 50 images de test (image151.png … image200.png)
│   │   └── labels/          # Annotations JSON (vérité terrain)
│   └── images/              # Images brutes diverses
├── tests/
│   ├── test_class.py        # Debug et évaluation de la classification (base validation)
│   └── test_final.py        # Évaluation finale sur la base de test (à lancer une seule fois)
├── main.py                  # Point d'entrée principal — pipeline de détection
├── optimize.py              # Optimisation des paramètres avec Optuna
├── train.py                 # Script d'entraînement
├── interface.py             # Interface graphique de visualisation
├── coinvision_optuna.db     # Base de données des essais Optuna
├── output.log               # Logs de la dernière exécution
├── result.txt               # Résultats de la dernière exécution
└── README.md
```

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/Sogolon88/PROJET_IMAGE.git
```

Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Utilisation

```bash
python main.py
```

## 🔍 Pipeline de traitement

L'algorithme principal (`main.py`) traite chaque image selon les étapes suivantes :

```
Image brute (RGB)
     ↓
Redimensionnement (max 800px)        # mise à l'échelle si nécessaire
     ↓
Conversion BGR → HSV                 # extraction du canal V (luminosité)
     ↓
CLAHE                                # égalisation adaptative du contraste (clipLimit=2.5, tile 8×8)
     ↓
Flou gaussien (11×11, σ=2.5)         # réduction du bruit
     ↓
Morphologie (OPEN + CLOSE)           # suppression du bruit résiduel et fermeture des contours
     ↓
Transformée de Hough (HoughCircles)  # détection des cercles (dp=1, param1=50, param2=60)
     ↓
NMS — Non-Maximum Suppression        # suppression des cercles redondants (overlap > 0.8)
     ↓
Comptage des pièces détectées
     ↓
Évaluation (régression)              # comparaison aux labels de vérité terrain
```

### Détail des étapes

1. **Redimensionnement** — les images dont le côté le plus grand dépasse 800 px sont redimensionnées proportionnellement afin d'uniformiser le traitement.
2. **Espace colorimétrique HSV** — seul le canal V (valeur/luminosité) est utilisé, ce qui rend la détection robuste aux variations de couleur des pièces.
3. **CLAHE** — l'égalisation adaptative du contraste améliore la lisibilité des bords sur les images surexposées ou sous-éclairées.
4. **Flou gaussien** — atténue le bruit haute fréquence avant la détection de cercles.
5. **Morphologie (OPEN + CLOSE)** — l'ouverture supprime les petits artefacts, la fermeture réunit les contours brisés.
6. **Transformée de Hough** — détecte les cercles dans l'image prétraitée. Les rayons acceptés vont de 25 à 140 px, la distance minimale entre deux centres est de 30 px.
7. **NMS (Non-Maximum Suppression)** — filtre les cercles en double en supprimant ceux dont le chevauchement dépasse 80 %.
8. **Évaluation** — les prédictions sont comparées aux annotations de la base de validation via une métrique de régression.


## Optimisation des paramètres avec Optuna

La pipeline de détection repose sur **11 paramètres** (taille du noyau gaussien, seuils de Hough, rayon min/max, seuil NMS...). Plutôt que de les régler à la main, on a utilisé **Optuna**, un framework d'optimisation bayésienne, pour trouver automatiquement la meilleure combinaison.

Le principe : Optuna lance des centaines d'essais en faisant varier les paramètres, et pour chaque essai il calcule le MSE sur la base de validation. Il retient la combinaison qui minimise l'erreur. Après **10 383 essais**, le meilleur résultat a été obtenu au trial #10040 avec un MSE de 14.99.

Les paramètres retenus sont fixés directement dans `main.py` :

```python
PARAMS = {
    "kernel_size": 15, "sigma": 4.711, "clip_limit": 1.316,
    "dp": 1, "param1": 32, "param2": 29,
    "minRadius": 40, "maxRadius": 100, "minDist": 75,
    "overlap_thresh": 0.986, "uniformite_kernel": 19,
}
```

### Relancer l'optimisation

Si on veut ré-optimiser (par exemple sur de nouvelles images), il suffit de lancer :

```bash
python optimize.py
```

Les résultats sont sauvegardés dans `coinvision_optuna.db` (base SQLite). On peut visualiser les essais avec le dashboard Optuna :

```bash
optuna-dashboard sqlite:///coinvision_optuna.db
```

> **Attention** : relancer l'optimisation prend plusieurs heures. Les paramètres actuels dans `main.py` sont déjà les meilleurs trouvés.

##  Dépendances
opencv-python
numpy

## Equipe de projet:

- KEITA Fode Laye