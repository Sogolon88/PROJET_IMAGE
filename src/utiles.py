import os
import json
from PIL import Image

def load_images(dossier="data/images"):
    images = []
    for fichier in os.listdir(dossier):
        if fichier.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(dossier, fichier))
            images.append(img)
    return images

def load_labels(fichier="data/image1.json"):
    with open(fichier, "r") as f:
        labels = json.load(f)
        LABELS_PIECES = ["1_cent", "2_cent", "5_cent", "10_cent", "20_cent", "50_cent", "1_euro", "2_euro"]
        nb_pieces = sum(1 for s in labels["shapes"] if s["label"] in LABELS_PIECES)
    return nb_pieces

def load_all_labels(dossier):
    """Charge les labels de tous les fichiers JSON dans un dictionnaire"""
    labels_dict = {}
    for file in os.listdir(dossier):
        if file.endswith(".json"):
            path = os.path.join(dossier, file)
            labels_dict[file] = load_labels(path)
    return labels_dict