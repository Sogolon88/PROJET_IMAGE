from src.algorithme import *
from src.algorithme import nms_circles
from data.images import *
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression
from matplotlib import pyplot as plt

# meilleurs parametres trouves avec Optuna apres optimisation
PARAMS = {
    "kernel_size":        15,
    "sigma":              4.711,
    "clip_limit":         1.316,
    "dp":                 1,
    "param1":             32,
    "param2":             29,
    "minRadius":          40,
    "maxRadius":          100,
    "minDist":            75,
    "overlap_thresh":     0.986,
    "uniformite_kernel":  19,
}
MAX_SIDE = 1025


def detecter_image(img_bgr):
    """
    Fonction principale de detection : prend une image BGR et retourne
    l'image annotee, le nombre de pieces detectees et la liste des cercles.
    Utilisee par l'interface graphique et par run_aldo().
    """
    # on redimensionne pour garder des cercles a des tailles coherentes
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)

    # on extrait le canal de luminosite (V) depuis l'espace HSV
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = hsv[:, :, 2]

    # CLAHE pour ameliorer le contraste de facon locale
    clahe    = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8, 8))
    combined = clahe.apply(combined)

    # flou gaussien pour attenuer le bruit avant la detection
    k        = PARAMS["kernel_size"]
    img_blur = cv2.GaussianBlur(combined, (k, k), PARAMS["sigma"])

    # morphologie pour supprimer les petits artefacts et fermer les contours
    kernel    = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_blur,  cv2.MORPH_OPEN,  kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    # detection des cercles avec la transformee de Hough circulaire
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

    # on supprime les doublons et les detections sur des zones non-pieces
    circles_filtres = nms_circles(
        circles_list, combined,
        overlap_thresh=PARAMS["overlap_thresh"],
        img_bgr=img_bgr,
        uniformite_kernel=PARAMS["uniformite_kernel"],
    )

    # on dessine les cercles detectes sur l'image pour la visualisation
    img_result = img_bgr.copy()
    for (cx, cy, r) in circles_filtres:
        cv2.circle(img_result, (cx, cy), r,  (0, 200, 80), 2)
        cv2.circle(img_result, (cx, cy), 5,  (0, 200, 80), -1)

    return img_result, len(circles_filtres), circles_filtres


def run_aldo(image_path: str = "data/base_validation/images"):
    """
    Lance la detection sur toutes les images d'un dossier et affiche
    les resultats image par image avec les statistiques finales.
    """
    images        = load_images(image_path)
    faux_negatifs = 0
    predictions   = {}

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

        _, nb, _ = detecter_image(img)

        key = _os.path.splitext(fichier)[0]
        print(f"******** {key} ********")
        if nb == 0:
            print("Aucune piece detectee")
            faux_negatifs += 1
            predictions[key] = 0
        else:
            print(f"Nombre de pieces detectees : {nb}")
            predictions[key] = nb
            print(f"Faux negatifs cumules : {faux_negatifs}\n")

    print("\n")
    stats = evaluate_regression(predictions, load_all_labels("data/base_validation/labels"))

    return {
        "predictions": predictions,
        "stats":       stats,
    }


if __name__ == "__main__":
    resultats = run_aldo()
