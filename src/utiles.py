"""
utiles.py

Fonctions utilitaires pour charger les images et les annotations depuis le disque.
Utilise par main.py et par les scripts d'evaluation.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image


def load_images(dossier="data/images"):
    """
    Charge toutes les images d'un dossier et les retourne sous forme de liste PIL.

    Les fichiers sont tries par ordre numerique (image1, image2 ... image150)
    plutot qu'alphabetique, pour eviter que image10 se retrouve avant image2.
    Les images sont converties de BGR (OpenCV) en RGB (PIL) avant d'etre retournees.
    """
    def tri_numerique(nom):
        base     = os.path.splitext(nom)[0]
        chiffres = "".join(filter(str.isdigit, base))
        return int(chiffres) if chiffres else 0

    images = []
    for fichier in sorted(os.listdir(dossier), key=tri_numerique):
        if fichier.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            chemin  = os.path.join(dossier, fichier)
            img_bgr = cv2.imread(chemin)
            if img_bgr is None:
                print(f"[load_images] Impossible de lire : {chemin}")
                continue
            images.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    return images


def load_labels(fichier="data/image1.json"):
    """
    Compte le nombre de pieces annotees dans un fichier JSON Labelme.

    On ne compte que les formes dont le label correspond a une denomination
    euro connue, pour ignorer les annotations parasites ou les labels errones.
    """
    LABELS_PIECES = ["1_cent", "2_cent", "5_cent", "10_cent",
                     "20_cent", "50_cent", "1_euro", "2_euro"]
    with open(fichier, "r") as f:
        labels = json.load(f)
    return sum(1 for s in labels["shapes"] if s["label"] in LABELS_PIECES)


def load_all_labels(dossier):
    """
    Charge tous les fichiers JSON d'un dossier et retourne un dictionnaire
    { nom_fichier.json : nombre_de_pieces }.

    Utilise par evaluate_regression pour comparer predictions et verite terrain.
    """
    labels_dict = {}
    for file in os.listdir(dossier):
        if file.endswith(".json"):
            labels_dict[file] = load_labels(os.path.join(dossier, file))
    return labels_dict


def load_all_boxes(dossier):
    """
    Charge les bounding boxes des JSON et les convertit en cercles inscrits.

    Chaque annotation Labelme stocke un rectangle (deux coins opposes). On en
    deduit le cercle inscrit en prenant le centre du rectangle et la moitie du
    plus petit cote comme rayon. C'est une approximation raisonnable puisque
    les pieces sont rondes et que le rectangle les entoure de pres.

    Les coordonnees sont dans l'espace de l'image originale, avant tout
    redimensionnement. La mise a l'echelle est faite dans evaluate_iou.

    Retourne { nom_image : {"circles": [(cx,cy,r),...], "img_h": h, "img_w": w} }.
    """
    LABELS_PIECES = ["1_cent", "2_cent", "5_cent", "10_cent",
                     "20_cent", "50_cent", "1_euro", "2_euro"]
    boxes_dict = {}
    for file in os.listdir(dossier):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(dossier, file), "r") as f:
            data = json.load(f)
        img_h = data.get("imageHeight", 1)
        img_w = data.get("imageWidth",  1)
        cercles = []
        for shape in data["shapes"]:
            if shape["label"] in LABELS_PIECES and shape["shape_type"] == "rectangle":
                (x1, y1), (x2, y2) = shape["points"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                r  = min(abs(x2 - x1), abs(y2 - y1)) / 2
                cercles.append((cx, cy, r))
        boxes_dict[os.path.splitext(file)[0]] = {
            "circles": cercles, "img_h": img_h, "img_w": img_w
        }
    return boxes_dict
