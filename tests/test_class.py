"""
test_class.py - debug et evaluation de la classification complete

Ce script evalue le pipeline groupement + vote sur les images de validation
dont la detection est correcte. Il affiche pour chaque image les denominations
predites vs reelles, les groupes, les features (diff_b, mean_a) et la somme.
A la fin, il compare plusieurs configurations d'epsilons pour le vote.

Lancer depuis la racine : python tests/test_class.py
"""

import os, sys, json, cv2

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "images")
LABELS_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "labels")
sys.path.insert(0, ROOT_DIR)

from main import detecter_image, MAX_SIDE
from src.classification import classifier_pieces, _extraire_features
from src.utiles import load_labels

VALEURS = {
    "1_cent": 0.01, "2_cent": 0.02, "5_cent": 0.05,
    "10_cent": 0.10, "20_cent": 0.20, "50_cent": 0.50,
    "1_euro": 1.00, "2_euro": 2.00,
}


def charger_somme_reelle(label_path):
    with open(label_path) as f:
        data = json.load(f)
    return round(sum(VALEURS.get(s["label"], 0.0) for s in data["shapes"]
                     if s["label"] in VALEURS), 2)


def charger_detail_reel(label_path):
    with open(label_path) as f:
        data = json.load(f)
    return [s["label"] for s in data["shapes"] if s["label"] in VALEURS]


# on ne garde que les images ou la detection est correcte
fichiers = sorted([f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))])

images_valides = []
for fichier in fichiers:
    stem = os.path.splitext(fichier)[0]
    label_path = os.path.join(LABELS_DIR, stem + ".json")
    if not os.path.exists(label_path):
        continue
    img_bgr = cv2.imread(os.path.join(IMAGES_DIR, fichier))
    if img_bgr is None:
        continue
    nb_reel = load_labels(label_path)
    _, nb_pred, cercles = detecter_image(img_bgr)
    if nb_pred != nb_reel:
        continue
    images_valides.append((fichier, img_bgr, cercles, charger_somme_reelle(label_path)))

print("Images avec detection correcte : %d\n" % len(images_valides))

EPSILONS_DETAIL = [0.02, 0.05, 0.10]

print("=" * 70)
print("  DETAIL DE CLASSIFICATION  (epsilons=%s)" % EPSILONS_DETAIL)
print("=" * 70)

for fichier, img_bgr, cercles, somme_reelle in images_valides:
    stem = os.path.splitext(fichier)[0]
    label_path = os.path.join(LABELS_DIR, stem + ".json")

    resultats, somme_pred = classifier_pieces(cercles, img_bgr, epsilons=EPSILONS_DETAIL)
    erreur = round(somme_pred - somme_reelle, 2)
    statut = "OK" if erreur == 0 else "XX"

    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    img_sc = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr
    features = [_extraire_features(img_sc, cx, cy, r) for cx, cy, r in cercles]

    print("\n%s %s  |  predit=%.2f  reel=%.2f  erreur=%+.2f" % (
        statut, fichier, somme_pred, somme_reelle, erreur))
    print("   Reel    : %s" % charger_detail_reel(label_path))
    print("   Classes : %s" % [r["classe"] for r in resultats])
    print("   Groupes : %s" % [r["groupe"] for r in resultats])
    line = "   Features: " + "  ".join(
        "p%d[dB=%.2f a=%.1f]" % (i+1, db, ma) for i, (db, ma) in enumerate(features))
    print(line)

# comparaison de plusieurs configurations d'epsilons
print("\n\n" + "=" * 60)
configs = [
    [0.02, 0.05, 0.10],
    [0.03, 0.06, 0.10],
    [0.03, 0.07, 0.12],
    [0.05, 0.10, 0.15],
]
print("%-30s | %7s | %7s | %7s" % ("Epsilons", "Exact%", "MAE", "MSE"))
print("-" * 60)
for epsilons in configs:
    erreurs = []
    nb_ok   = 0
    for fichier, img_bgr, cercles, somme_reelle in images_valides:
        resultats, somme_pred = classifier_pieces(cercles, img_bgr, epsilons=epsilons)
        erreur = round(somme_pred - somme_reelle, 2)
        erreurs.append(abs(erreur))
        if erreur == 0:
            nb_ok += 1
    n   = len(erreurs)
    mae = round(sum(erreurs) / n, 3)
    mse = round(sum(e**2 for e in erreurs) / n, 3)
    pct = 100 * nb_ok // n
    print("%-30s | %6d%% | %7.3f | %7.3f" % (str(epsilons), pct, mae, mse))
