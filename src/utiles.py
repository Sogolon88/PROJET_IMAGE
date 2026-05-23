import os
import json
import cv2
import numpy as np
from PIL import Image


def load_images(dossier="data/images"):
    """
    Charge toutes les images d'un dossier et les retourne en format PIL RGB.
    On utilise cv2 pour la lecture car il gere mieux les formats corrompus,
    puis on convertit en PIL pour la compatibilite avec le reste du code.
    Le tri numerique evite d'avoir image10 avant image2.
    """
    images = []

    def tri_numerique(nom):
        base = os.path.splitext(nom)[0]
        chiffres = ''.join(filter(str.isdigit, base))
        return int(chiffres) if chiffres else 0

    for fichier in sorted(os.listdir(dossier), key=tri_numerique):
        if fichier.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            chemin = os.path.join(dossier, fichier)
            img_bgr = cv2.imread(chemin)
            if img_bgr is None:
                print(f"[load_images] Impossible de lire : {chemin}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(img_rgb))

    return images


def load_labels(fichier="data/image1.json"):
    """
    Lit un fichier JSON d'annotation et retourne le nombre de pieces presentes.
    On compte uniquement les formes dont le label correspond a une piece euro connue.
    """
    with open(fichier, "r") as f:
        labels = json.load(f)
    LABELS_PIECES = ["1_cent", "2_cent", "5_cent", "10_cent", "20_cent",
                     "50_cent", "1_euro", "2_euro"]
    return sum(1 for s in labels["shapes"] if s["label"] in LABELS_PIECES)


def load_all_labels(dossier):
    """
    Charge tous les fichiers JSON d'un dossier et retourne un dictionnaire
    avec le nom du fichier comme cle et le nombre de pieces comme valeur.
    Utile pour comparer d'un coup toutes les predictions avec la verite terrain.
    """
    labels_dict = {}
    for file in os.listdir(dossier):
        if file.endswith(".json"):
            path = os.path.join(dossier, file)
            labels_dict[file] = load_labels(path)
    return labels_dict
