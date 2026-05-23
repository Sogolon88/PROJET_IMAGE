"""
test_class.py - script de debug pour la classification des pieces

Ce script sert a analyser en detail comment se comporte la classification
sur les images de la base de validation dont la detection est correcte.
Il affiche pour chaque image les pieces predites vs reelles, les groupes
de couleur detectes et les features extraites (teinte et diff saturation).

A la fin il compare plusieurs configurations d'epsilons pour le vote.

Lancer depuis la racine du projet : python tests/test_class.py
"""

import os, sys, json, cv2

# on remonte d'un niveau pour acceder aux modules du projet
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "images")
LABELS_DIR = os.path.join(ROOT_DIR, "data", "base_validation", "labels")
sys.path.insert(0, ROOT_DIR)

from main import detecter_image
from src.classification import classifier_pieces, _reduire_reflets, _extraire_features
from src.utiles import load_labels

# correspondance label -> valeur en euros
VALEURS = {
    "1_cent": 0.01, "2_cent": 0.02, "5_cent": 0.05,
    "10_cent": 0.10, "20_cent": 0.20, "50_cent": 0.50,
    "1_euro": 1.00, "2_euro": 2.00,
}


def charger_somme_reelle(label_path):
    """Calcule la somme reelle d'une image a partir de son fichier JSON."""
    with open(label_path, "r") as f:
        data = json.load(f)
    return round(sum(VALEURS.get(s["label"], 0.0) for s in data["shapes"]
                     if s["label"] in VALEURS), 2)


def charger_detail_reel(label_path):
    """Retourne la liste des labels de pieces dans une image."""
    with open(label_path, "r") as f:
        data = json.load(f)
    return [s["label"] for s in data["shapes"] if s["label"] in VALEURS]


# on charge uniquement les images ou la detection a trouve le bon nombre de pieces
# pour ne pas melanger les erreurs de detection et de classification
fichiers = sorted([f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))])

images_valides = []
for fichier in fichiers:
    stem       = os.path.splitext(fichier)[0]
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

# affichage detaille de chaque image pour comprendre les erreurs
EPSILONS_DETAIL = [0.02, 0.05, 0.10]

print("=" * 70)
print("  DETAIL DE CLASSIFICATION  (epsilons=%s)" % EPSILONS_DETAIL)
print("=" * 70)

for fichier, img_bgr, cercles, somme_reelle in images_valides:
    stem       = os.path.splitext(fichier)[0]
    label_path = os.path.join(LABELS_DIR, stem + ".json")

    resultats, somme_pred = classifier_pieces(cercles, img_bgr, epsilons=EPSILONS_DETAIL)
    erreur = round(somme_pred - somme_reelle, 2)
    statut = "OK" if erreur == 0 else "XX"

    pieces_pred = [r["classe"] for r in resultats]
    pieces_reel = charger_detail_reel(label_path)

    # on recalcule les features sur l'image redimensionnee pour l'affichage
    MAX_SIDE = 1025
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    img_sc  = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr
    img_cor  = _reduire_reflets(img_sc)
    features = [_extraire_features(img_cor, cx, cy, r) for cx, cy, r in cercles]

    print("\n%s %s  |  predit=%.2f  reel=%.2f  erreur=%+.2f" % (
        statut, fichier, somme_pred, somme_reelle, erreur))
    print("   Reel    : %s" % pieces_reel)
    print("   Classes : %s" % pieces_pred)
    print("   Groupes : %s" % [r['groupe'] for r in resultats])
    line = "   Features: " + "  ".join(
        "p%d[H=%.1f dS=%.3f]" % (i+1, mH, dS) for i, (mH, dS) in enumerate(features))
    print(line)

# comparaison de plusieurs configurations d'epsilons pour voir laquelle est la meilleure
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
