import cv2
import numpy as np
import os
import sys
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from src.utiles import load_images, load_all_labels
from src.algorithme import hough_transform, nms_circles


def setup_logging(log_file="output.log"):
    """Redirige print() et les erreurs vers le fichier log ET la console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

def log(msg):
    logging.info(msg)


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


def evaluer_combo(args):
    """Évalue une combinaison de paramètres — exécutée dans un worker."""
    idx, total, params, images_preprocessed, labels, nb = args

    kernel_morph = np.ones((3, 3), np.uint8)
    clahe = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=(8, 8))
    predictions = {}

    for i, (v_ch, s_ch) in enumerate(images_preprocessed):
        # Canal V uniquement (poids_v=1.0 fixé)
        combined = v_ch.copy()

        # CLAHE
        combined = clahe.apply(combined)

        # Flou gaussien
        k = params["kernel_size"]
        img_blur = cv2.GaussianBlur(combined, (k, k), params["sigma"])

        # Morphologie
        img_clean = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN,  kernel_morph)
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

        circles_list = [(c[0], c[1], c[2]) for c in circles[0]] if len(circles) != 0 else []
        circles_filtres = nms_circles(circles_list, combined, overlap_thresh=params["overlap_thresh"])
        predictions[f"image{i+1}"] = len(circles_filtres)

    f1, mse, score = compute_metrics(predictions, labels, nb)
    return idx, params, f1, mse, score


if __name__ == "__main__":
    setup_logging("output.log")
    MAX_SIDE = 800
    nb = 80
    NB_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # laisse 1 cœur libre

    images_raw = load_images("data/base_validation/images")
    labels = load_all_labels("data/base_validation/labels")

    # Pré-traitement des images une seule fois
    log("Pré-traitement des images...")
    images_preprocessed = []
    for img in images_raw[:nb]:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images_preprocessed.append((hsv[:, :, 2], hsv[:, :, 1]))
    log(f"{len(images_preprocessed)} images pré-traitées.")

    # Grille de paramètres — Phase 3 : exploration des paramètres Hough
    # Paramètres fixés (optimisés phases 1 & 2) :
    #   kernel_size=11, sigma=2.5, poids_v=1.0, dp=1,
    #   clip_limit=2.5, overlap_thresh=1.0
    # On explore param1, param2, minRadius, maxRadius, minDist
    param_grid = {
        "kernel_size":    [7, 9, 11],
        "sigma":          [1, 2, 2.5, 3],
        "poids_v":        [1.0],
        "dp":             [1, 1.2],
        "clip_limit":     [2, 2.5],
        "overlap_thresh": [1.0],
        "param1":         [40, 45, 50, 55, 60],
        "param2":         [50, 55, 60, 65, 70],
        "minRadius":      [20, 25, 30],
        "maxRadius":      [130, 140, 150],
        "minDist":        [25, 30, 35],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    total  = 1
    for v in values:
        total *= len(v)

    log(f"Nombre total de combinaisons : {total}")
    log(f"Workers parallèles           : {NB_WORKERS}")

    # Préparer toutes les tâches
    tasks = [
        (idx, total, dict(zip(keys, combo)), images_preprocessed, labels, nb)
        for idx, combo in enumerate(product(*values))
    ]

    best_score  = -999
    best_params = None
    done        = 0

    with ProcessPoolExecutor(max_workers=NB_WORKERS) as executor:
        futures = {executor.submit(evaluer_combo, t): t[0] for t in tasks}

        for future in as_completed(futures):
            idx, params, f1, mse, score = future.result()
            done += 1

            if score > best_score:
                best_score  = score
                best_params = params
                log(f"[{done}/{total}] Nouveau meilleur -> F1={f1:.3f}, MSE={mse:.2f}, score={score:.3f}")
                log(f"  Params: {params}")
                with open("result.txt", "a") as f:
                    f.write(f"[{done}/{total}] Nouveau meilleur -> F1={f1:.3f}, MSE={mse:.2f}, score={score:.3f}\n")
                    f.write(f"  Params: {params}\n")

            # Progression toutes les 50 combinaisons
            if done % 50 == 0:
                log(f"  ... {done}/{total} combinaisons traitées")

    log("===== RÉSULTAT FINAL =====")
    log(f"Meilleur score      : {best_score:.3f}")
    log(f"Meilleurs paramètres: {best_params}")
    with open("result.txt", "a") as f:
        f.write("\n===== RÉSULTAT FINAL =====\n")
        f.write(f"Meilleur score      : {best_score:.3f}\n")
        f.write(f"Meilleurs paramètres: {best_params}\n")
    