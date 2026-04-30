import cv2
import numpy as np
import os
from itertools import product
from src.utiles import load_images, load_all_labels
from src.algorithme import hough_transform, nms_circles


def compute_metrics(predictions, labels, nb):
    vrai_positifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) > 0
                        and labels.get(f"image{i}.json", 0) > 0)
    faux_positifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) > 0
                        and labels.get(f"image{i}.json", 0) == 0)
    faux_negatifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) == 0
                        and labels.get(f"image{i}.json", 0) > 0)

    precision = vrai_positifs / (vrai_positifs + faux_positifs) if (vrai_positifs + faux_positifs) > 0 else 0
    rappel    = vrai_positifs / (vrai_positifs + faux_negatifs) if (vrai_positifs + faux_negatifs) > 0 else 0
    f1        = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0

    mse = sum((predictions.get(f"image{i}", 0) - labels.get(f"image{i}.json", 0))**2
              for i in range(1, nb+1)) / nb

    score = f1 - 0.1 * mse
    return f1, mse, score


if __name__ == "__main__":
    MAX_SIDE = 800
    nb = 150

    images_raw = load_images("data/base_validation/images")
    labels = load_all_labels("data/base_validation/labels")

    # Pré-traiter les images une seule fois (resize + extraction canaux V et S)
    print("Pré-traitement des images...")
    images_preprocessed = []
    for img in images_raw[:nb]:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        images_preprocessed.append((v, s))
    print(f"{len(images_preprocessed)} images pré-traitées.\n")

    # Grille de paramètres (kernel_size doit être impair)
    param_grid = {
        "kernel_size":    [5, 9],
        "sigma":          [2.0, 2.5, 3.0],
        "poids_v":        [0.85, 0.9, 0.95],
        "dp":             [1.2],
        "param1":         [100, 120],
        "param2":         [80, 90, 100],
        "minRadius":      [18, 20],
        "maxRadius":      [150],
        "minDist":        [25, 30, 35],
        "clip_limit":     [1.5, 2.0, 2.5],
        "overlap_thresh": [0.4, 0.5, 0.6],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"Nombre total de combinaisons : {total}\n")

    best_score = -999
    best_params = None
    kernel_morph = np.ones((3, 3), np.uint8)

    for idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))
        predictions = {}

        clahe = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=(8, 8))

        for i, (v_ch, s_ch) in enumerate(images_preprocessed):
            # Combinaison pondérée V + S
            combined = cv2.addWeighted(v_ch, params["poids_v"], s_ch, 1.0 - params["poids_v"], 0)

            # CLAHE
            combined = clahe.apply(combined)

            # Flou gaussien
            k = params["kernel_size"]
            img_blur = cv2.GaussianBlur(combined, (k, k), params["sigma"])

            # Morphologie
            img_clean = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel_morph)
            img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel_morph)

            # Détection Hough
            circles = hough_transform(
                image=img_clean,
                dp=params["dp"],
                param1=params["param1"],
                param2=params["param2"],
                minRadius=params["minRadius"],
                minDist=params["minDist"],
                maxRadius=params["maxRadius"]
            )

            # NMS pour filtrer les faux cercles
            circles_list = []
            if len(circles) != 0:
                circles_list = [(c[0], c[1], c[2]) for c in circles[0]]
            circles_filtres = nms_circles(circles_list, combined, overlap_thresh=params["overlap_thresh"])

            predictions[f"image{i+1}"] = len(circles_filtres)

        f1, mse, score = compute_metrics(predictions, labels, nb)

        if score > best_score:
            best_score = score
            best_params = params
            print(f"[{idx+1}/{total}] Nouveau meilleur -> F1={f1:.3f}, MSE={mse:.2f}, score={score:.3f}")
            print(f"  Params: {params}\n")
            with open("result.txt", "a") as f :
                f.write(f"[{idx+1}/{total}] Nouveau meilleur -> F1={f1:.3f}, MSE={mse:.2f}, score={score:.3f}")
                f.write(f"  Params: {params}\n")

    print("===== RÉSULTAT FINAL =====")
    print(f"Meilleur score : {best_score:.3f}")
    print(f"Meilleurs paramètres : {best_params}")
    with open("result.txt", "a") as f:
        f.write("\n===== RÉSULTAT FINAL =====\n")
        f.write(f"Meilleur score : {best_score:.3f}\n")
        f.write(f"Meilleurs paramètres : {best_params}\n")
    