"""
optimize.py  Recherche automatique des meilleurs paramètres via Optuna
Lancer : python optimize.py
Résultats sauvegardés dans coinvision_optuna.db (reprise possible si interruption)
"""

import cv2
import numpy as np
import optuna
import os
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression
from src.algorithme import hough_transform, nms_circles, est_couleur_piece

# ── Config ──────────────────────────────────────────
N_TRIALS   = 10_000      # nombre d'essais max (laisse tourner jusqu'au bout)
N_JOBS     = 4           # cœurs parallèles (mettre -1 pour tous les utiliser)
TIMEOUT    = None        # pas de limite de temps — tourne jusqu'à N_TRIALS
N_IMAGES   = 150         # nombre d'images sur lesquelles évaluer chaque essai
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "base_validation", "images")
LABELS_DIR = os.path.join(os.path.dirname(__file__), "data", "base_validation", "labels")
DB_PATH    = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'coinvision_optuna.db')}"
MAX_SIDE   = 800


# ── Pipeline de détection paramétrée ────────────────

def detecter_image_params(img_bgr, params):
    """Même pipeline que detecter_image() mais accepte un dict params explicite."""
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = hsv[:, :, 2]

    clahe    = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=(8, 8))
    combined = clahe.apply(combined)

    k        = params["kernel_size"]
    img_blur = cv2.GaussianBlur(combined, (k, k), params["sigma"])

    kernel    = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_blur,  cv2.MORPH_OPEN,  kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    circles = hough_transform(
        image=img_clean,
        dp=params["dp"],
        param1=params["param1"],
        param2=params["param2"],
        minRadius=params["minRadius"],
        minDist=params["minDist"],
        maxRadius=params["maxRadius"],
    )

    circles_list = []
    if len(circles) != 0:
        circles_list = [(c[0], c[1], c[2]) for c in circles[0]]

    circles_filtres = nms_circles(
        circles_list, combined,
        overlap_thresh=params["overlap_thresh"],
        img_bgr=img_bgr,
        uniformite_kernel=params["uniformite_kernel"],
    )

    return len(circles_filtres)


# ── Chargement des données (une seule fois) ──────────

print("Chargement des images...")
_images_pil = load_images(IMAGES_DIR)
_images_bgr = []
for img in _images_pil[:N_IMAGES]:
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _images_bgr.append(img)

_labels = load_all_labels(LABELS_DIR)
print(f"{len(_images_bgr)} images chargées.\n")


# ── Fonction objectif ────────────────────────────────

def objective(trial):
    params = {
        "kernel_size":       trial.suggest_int("kernel_size",      3,  21, step=2),
        "sigma":             trial.suggest_float("sigma",           0.5, 5.0),
        "clip_limit":        trial.suggest_float("clip_limit",      1.0, 5.0),
        "dp":                trial.suggest_int("dp",                1,   2),
        "param1":            trial.suggest_int("param1",            20,  120),
        "param2":            trial.suggest_int("param2",            20,  100),
        "minRadius":         trial.suggest_int("minRadius",         10,  60),
        "maxRadius":         trial.suggest_int("maxRadius",         80,  200),
        "minDist":           trial.suggest_int("minDist",           10,  80),
        "overlap_thresh":    trial.suggest_float("overlap_thresh",  0.3, 1.0),
        "uniformite_kernel": trial.suggest_int("uniformite_kernel", 3,  21, step=2),
    }

    # contrainte : minRadius < maxRadius
    if params["minRadius"] >= params["maxRadius"]:
        raise optuna.exceptions.TrialPruned()

    predictions = {}
    for i, img_bgr in enumerate(_images_bgr):
        nb = detecter_image_params(img_bgr, params)
        predictions[f"image{i+1}"] = nb

    stats = evaluate_regression(predictions, _labels)
    return stats["mse"]


# ── Lancement ────────────────────────────────────────

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="minimize",
        storage=DB_PATH,
        study_name="coinvision",
        load_if_exists=True,    # reprend là où on s'est arrêté si relance
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    print(f"Démarrage — {N_TRIALS} essais max, pas de timeout")
    print(f"Base de données : {DB_PATH}\n")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )

    # ── Résultats ────────────────────────────────────
    best = study.best_params
    print("\n" + "="*50)
    print(f"Meilleur MSE : {study.best_value:.4f}")
    print("="*50)
    print("\nPARAMS optimaux à copier dans main.py :\n")
    print("PARAMS = {")
    for k, v in best.items():
        if isinstance(v, float):
            print(f'    "{k}": {v:.4f},')
        else:
            print(f'    "{k}": {v},')
    print("}")

    # Sauvegarde aussi dans un fichier texte
    out_path = os.path.join(os.path.dirname(__file__), "best_params.txt")
    with open(out_path, "w") as f:
        f.write(f"Meilleur MSE : {study.best_value:.4f}\n\n")
        f.write("PARAMS = {\n")
        for k, v in best.items():
            if isinstance(v, float):
                f.write(f'    "{k}": {v:.4f},\n')
            else:
                f.write(f'    "{k}": {v},\n')
        f.write("}\n")
    print(f"\nRésultats sauvegardés dans : {out_path}")
