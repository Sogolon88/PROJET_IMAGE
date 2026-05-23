"""
test_final.py - evaluation sur la base de test (50 images)

Ce script est a lancer une seule fois quand on est satisfait des resultats
sur la base de validation. Il mesure la performance reelle du modele sur
des images qu'il n'a jamais vues pendant le developpement.

Lancer depuis la racine du projet : python tests/test_final.py
"""

import cv2
import numpy as np
import os
import sys

# on remonte d'un niveau pour acceder aux modules du projet
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.algorithme import hough_transform, nms_circles
from src.utiles import load_all_labels
from src.evaluation import evaluate_regression

# meilleurs parametres trouves avec Optuna apres optimisation sur la validation
PARAMS = {
    "kernel_size":        15,
    "sigma":              4.711,
    "clip_limit":         1.316,
    "dp":                 1,
    "param1":             32,
    "param2":             29,
    "minRadius":          40,
    "maxRadius":          100,
    "minDist":            75,
    "overlap_thresh":     0.986,
    "uniformite_kernel":  19,
}
MAX_SIDE = 1025


def detecter_image(img_bgr):
    """
    Applique toute la pipeline de detection sur une image et retourne
    le nombre de pieces trouvees. Meme pipeline que dans main.py.
    """
    # on redimensionne pour garder des tailles de cercles coherentes
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    # on travaille sur le canal de luminosite en HSV
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = hsv[:, :, 2]

    # amelioration du contraste local avec CLAHE
    clahe    = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8, 8))
    combined = clahe.apply(combined)

    # flou gaussien pour lisser les petits details parasites
    k        = PARAMS["kernel_size"]
    img_blur = cv2.GaussianBlur(combined, (k, k), PARAMS["sigma"])

    # morphologie pour nettoyer le bruit et fermer les contours
    kernel    = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_blur,  cv2.MORPH_OPEN,  kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    # detection des cercles avec la transformee de Hough
    circles = hough_transform(
        image=img_clean,
        dp=PARAMS["dp"], param1=PARAMS["param1"], param2=PARAMS["param2"],
        minRadius=PARAMS["minRadius"], minDist=PARAMS["minDist"],
        maxRadius=PARAMS["maxRadius"],
    )

    circles_list = []
    if len(circles) != 0:
        circles_list = [(c[0], c[1], c[2]) for c in circles[0]]

    # suppression des doublons et faux positifs
    circles_filtres = nms_circles(
        circles_list, combined,
        overlap_thresh=PARAMS["overlap_thresh"],
        img_bgr=img_bgr,
        uniformite_kernel=PARAMS["uniformite_kernel"],
    )

    return len(circles_filtres)


if __name__ == "__main__":
    IMAGES_DIR = os.path.join(ROOT_DIR, "data", "base_test", "images")
    LABELS_DIR = os.path.join(ROOT_DIR, "data", "base_test", "labels")

    labels = load_all_labels(LABELS_DIR)

    # tri numerique pour avoir image151, image152, ... dans le bon ordre
    fichiers = sorted(
        [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda f: int(os.path.splitext(f)[0].replace("image", ""))
    )

    predictions = {}
    print(f"{'='*60}")
    print(f"  BASE DE TEST — {len(fichiers)} images")
    print(f"{'='*60}")

    for fichier in fichiers:
        img_bgr = cv2.imread(os.path.join(IMAGES_DIR, fichier))
        if img_bgr is None:
            print(f"  [!] Impossible de lire : {fichier}")
            continue

        key    = os.path.splitext(fichier)[0]
        nb     = detecter_image(img_bgr)
        predictions[key] = nb

        gt     = labels.get(f"{key}.json", "?")
        statut = "Oui" if nb == gt else "Non"
        print(f"  {statut} {key}: predit={nb}, reel={gt}")

    print(f"\n{'─'*60}")
    evaluate_regression(predictions, labels)
