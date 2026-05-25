"""
test_full_classif.py - evaluation complete detection + classification

Lance le pipeline complet sur toutes les images de validation (150 images),
meme celles ou la detection est incorrecte. Affiche la somme predite vs reelle
et les statistiques globales (exact%, MAE, MSE).

Lancer depuis la racine : python tests/test_full_classif.py
"""

import os, sys, json, cv2

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "images")
LABELS_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "labels")
sys.path.insert(0, ROOT_DIR)

from main import detecter_image
from src.classification import classifier_pieces
from src.utiles import load_labels

VALEURS = {
    "1_cent": 0.01, "2_cent": 0.02, "5_cent": 0.05,
    "10_cent": 0.10, "20_cent": 0.20, "50_cent": 0.50,
    "1_euro": 1.00, "2_euro": 2.00,
}

fichiers = sorted([f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))])

erreurs   = []
nb_ok     = 0
nb_traite = 0

print("=" * 70)
print("  PIPELINE COMPLET - BASE VALIDATION (detection + classification)")
print("=" * 70)

for fichier in fichiers:
    stem = os.path.splitext(fichier)[0]
    label_path = os.path.join(LABELS_DIR, stem + ".json")
    if not os.path.exists(label_path):
        continue

    img_bgr = cv2.imread(os.path.join(IMAGES_DIR, fichier))
    if img_bgr is None:
        continue

    # somme reelle
    with open(label_path) as f:
        data = json.load(f)
    somme_reelle = round(sum(VALEURS.get(s["label"], 0.0)
                             for s in data["shapes"] if s["label"] in VALEURS), 2)
    nb_reel = load_labels(label_path)

    # detection
    _, nb_pred, cercles = detecter_image(img_bgr)

    # classification
    resultats, somme_pred = classifier_pieces(cercles, img_bgr)

    erreur = round(somme_pred - somme_reelle, 2)
    statut = "OK" if erreur == 0 else "XX"
    det_ok = "det=OK" if nb_pred == nb_reel else "det=XX(%d/%d)" % (nb_pred, nb_reel)

    print("%s %s  |  %s  |  predit=%.2f  reel=%.2f  erreur=%+.2f" % (
        statut, fichier, det_ok, somme_pred, somme_reelle, erreur))

    erreurs.append(abs(erreur))
    if erreur == 0:
        nb_ok += 1
    nb_traite += 1

print("\n" + "=" * 70)
print("  BILAN SUR %d IMAGES" % nb_traite)
print("=" * 70)
mae = sum(erreurs) / nb_traite
mse = sum(e**2 for e in erreurs) / nb_traite
print("  Exact (somme juste) : %d/%d  (%.1f%%)" % (nb_ok, nb_traite, 100*nb_ok/nb_traite))
print("  MAE                 : %.3f EUR" % mae)
print("  MSE                 : %.3f" % mse)
print("=" * 70)
