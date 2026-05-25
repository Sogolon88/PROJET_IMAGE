"""
main.py

Point d'entrée principal du système CoinVision.

Ce fichier orchestre le pipeline complet de détection des pièces euro :
  1. Redimensionnement de l'image pour normaliser les tailles.
  2. Extraction du canal de luminosité (V en HSV) pour travailler sur une image
     monochrome robuste aux variations de couleur.
  3. CLAHE pour améliorer le contraste localement.
  4. Flou gaussien pour atténuer le bruit avant Hough.
  5. Morphologie (ouverture + fermeture) pour nettoyer l'image.
  6. Transformée de Hough circulaire pour détecter les pièces.
  7. Évaluation des résultats sur la base de validation.

Les paramètres ci-dessous ont été trouvés automatiquement par Optuna après
plus de 10 000 essais d'optimisation bayésienne sur la base de validation.
"""

from src.algorithme import *
from src.algorithme import nms_circles
from data.images import *
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression, evaluate_iou
from src.utiles import load_all_boxes
from matplotlib import pyplot as plt

# Meilleurs paramètres trouvés par Optuna (trial #10040, MSE = 14.99).
# Ne pas modifier sans relancer une optimisation complète.
PARAMS = {
    "kernel_size":       15,      # taille du noyau gaussien (doit être impair)
    "sigma":             4.711,   # écart-type du flou gaussien
    "clip_limit":        1.316,   # limite de contraste pour CLAHE
    "dp":                1,       # résolution de l'accumulateur Hough
    "param1":            32,      # seuil haut de Canny interne à Hough
    "param2":            29,      # seuil de l'accumulateur Hough (plus bas = plus de détections)
    "minRadius":         40,      # rayon minimum des cercles cherchés (en pixels)
    "maxRadius":         100,     # rayon maximum
    "minDist":           75,      # distance minimale entre deux centres détectés
    "overlap_thresh":    0.986,   # seuil de chevauchement pour la NMS
    "uniformite_kernel": 19,      # taille du noyau pour le score d'uniformité NMS
    "morph_kernel":      9,       # taille du noyau morphologique
}
MAX_SIDE = 1025  # côté maximum de l'image après redimensionnement


def detecter_image(img_bgr):
    """
    Applique le pipeline complet de détection sur une image BGR.

    Retourne l'image annotée avec les cercles, le nombre de pièces détectées
    et la liste des cercles sous forme (cx, cy, r). Cette fonction est appelée
    aussi bien par run_aldo() que par l'interface graphique.

    On travaille en niveaux de gris (canal V de HSV) plutôt qu'en couleur parce
    que la transformée de Hough ne traite qu'une seule couche, et le canal V
    (luminosité) est celui qui capture le mieux les bords des pièces métalliques
    quel que soit leur couleur.

    Le CLAHE (Contrast Limited Adaptive Histogram Equalization) améliore le contraste
    de façon locale : il divise l'image en petits blocs et égalise l'histogramme
    dans chaque bloc séparément, ce qui évite de sur-amplifier le bruit dans les
    zones déjà bien contrastées (d'où la limite "clip_limit").

    L'ouverture morphologique supprime les petits artefacts (poussières, reflets
    ponctuels) et la fermeture réunit les bords brisés, rendant les contours
    des pièces plus nets pour Hough.

"""
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    img_original = img_bgr.copy()

    hsv         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    canal_v     = hsv[:, :, 2]

    clahe       = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8, 8))
    apres_clahe = clahe.apply(canal_v)

    k        = PARAMS["kernel_size"]
    img_blur = cv2.GaussianBlur(apres_clahe, (k, k), PARAMS["sigma"])

    mk         = PARAMS["morph_kernel"]
    kernel     = np.ones((mk, mk), np.uint8)
    img_clean  = cv2.morphologyEx(img_blur,  cv2.MORPH_OPEN,  kernel)
    img_clean  = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    circles = hough_transform(
        image=img_clean,
        dp=PARAMS["dp"],
        param1=PARAMS["param1"],
        param2=PARAMS["param2"],
        minRadius=PARAMS["minRadius"],
        minDist=PARAMS["minDist"],
        maxRadius=PARAMS["maxRadius"],
    )

    circles_list = []
    if len(circles) != 0:
        circles_list = [(c[0], c[1], c[2]) for c in circles[0]]

    circles_filtres = nms_circles(
         circles_list, apres_clahe,
         overlap_thresh=PARAMS["overlap_thresh"],
         img_bgr=img_bgr,
         uniformite_kernel=PARAMS["uniformite_kernel"],
     )

    img_result = img_bgr.copy()
    for (cx, cy, r) in circles_filtres:
        cv2.circle(img_result, (cx, cy), r, (0, 200, 80), 2)
        cv2.circle(img_result, (cx, cy), 5, (0, 200, 80), -1)

    return img_result, len(circles_filtres), circles_filtres


def run_aldo(image_path: str = "data/base_validation/images"):
    """
    Lance la détection sur toutes les images d'un dossier et évalue les résultats.

    On traite les 150 premières images dans l'ordre numérique, on collecte le
    nombre de pièces détectées par image, puis on appelle les deux fonctions
    d'évaluation : evaluate_regression (comptage pur) et evaluate_iou (localisation).

    Le compteur faux_negatifs suit le nombre d'images où aucune pièce n'a été
    trouvée alors qu'il y en avait, ce qui est utile pour le débogage.
    """
    images              = load_images(image_path)
    faux_negatifs       = 0
    predictions         = {}
    predictions_circles = {}

    import os as _os

    def _tri_num(nom):
        base   = _os.path.splitext(nom)[0]
        digits = ''.join(filter(str.isdigit, base))
        return int(digits) if digits else 0

    fichiers = sorted(
        [f for f in _os.listdir(image_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=_tri_num
    )

    for i, (img, fichier) in enumerate(zip(images[:150], fichiers[:150])):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        _, nb, circles = detecter_image(img)

        key = _os.path.splitext(fichier)[0]
        print(f"******** {key} ********")
        if nb == 0:
            print("Aucune pièce détectée")
            faux_negatifs += 1
            predictions[key] = 0
        else:
            print(f"Nombre de pièces détectées : {nb}")
            predictions[key] = nb
            print(f"Faux négatifs cumulés : {faux_negatifs}\n")
        predictions_circles[key] = circles

    print("\n")
    label_path = "data/base_validation/labels"
    stats      = evaluate_regression(predictions, load_all_labels(label_path))
    gt_boxes   = load_all_boxes(label_path)
    stats_iou  = evaluate_iou(predictions_circles, gt_boxes)

    return {
        "predictions":         predictions,
        "predictions_circles": predictions_circles,
        "stats":               stats,
        "stats_iou":           stats_iou,
    }


if __name__ == "__main__":
    resultats = run_aldo()
