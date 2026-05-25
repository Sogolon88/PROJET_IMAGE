"""
test_final.py - evaluation sur la base de test (50 images)

Ce script est a lancer une seule fois quand on est satisfait des resultats
sur la base de validation. Il mesure la performance reelle du modele sur
des images qu'il n'a jamais vues pendant le developpement.

Lancer depuis la racine du projet : python tests/test_final.py
"""

import cv2
import os
import sys

# on remonte d'un niveau pour acceder aux modules du projet
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# on importe directement depuis main pour garantir la meme pipeline
from main import detecter_image
from src.utiles import load_all_labels, load_all_boxes
from src.evaluation import evaluate_regression, evaluate_iou


if __name__ == "__main__":
    IMAGES_DIR = os.path.join(ROOT_DIR, "data", "base_test", "images")
    LABELS_DIR = os.path.join(ROOT_DIR, "data", "base_test", "labels")

    labels = load_all_labels(LABELS_DIR)

    fichiers = sorted(
        [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda f: int(os.path.splitext(f)[0].replace("image", ""))
    )

    predictions         = {}
    predictions_circles = {}

    print(f"{'='*60}")
    print(f"  BASE DE TEST — {len(fichiers)} images")
    print(f"{'='*60}")

    for fichier in fichiers:
        img_bgr = cv2.imread(os.path.join(IMAGES_DIR, fichier))
        if img_bgr is None:
            print(f"  [!] Impossible de lire : {fichier}")
            continue

        key = os.path.splitext(fichier)[0]
        _, nb, circles = detecter_image(img_bgr)
        predictions[key]         = nb
        predictions_circles[key] = circles

        gt     = labels.get(f"{key}.json", "?")
        statut = "✓" if nb == gt else "✗"
        print(f"  {statut} {key}: predit={nb}, reel={gt}")

    print(f"\n{'─'*60}")
    evaluate_regression(predictions, labels)
    gt_boxes = load_all_boxes(LABELS_DIR)
    evaluate_iou(predictions_circles, gt_boxes)
