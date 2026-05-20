from src.algorithme import *
from src.algorithme import nms_circles
from data.images import *
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression
from matplotlib import pyplot as plt

# ── Paramètres calibrés
PARAMS = {
    "kernel_size":   11,
    "sigma":         2.5,
    "clip_limit":    2.5,
    "dp":            1,
    "param1":        50,
    "param2":        60,
    "minRadius":     25,
    "maxRadius":     140,
    "minDist":       30,
    "overlap_thresh":1.0,
}
MAX_SIDE = 800


def detecter_image(img_bgr):
    """
    Applique la pipeline complète de détection sur une image BGR.
    Retourne (img_annotée, nb_pièces, liste_cercles).
    Utilisée par l'interface graphique ET par run_aldo().
    """
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)

    # Canal V (luminosité HSV)
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = hsv[:, :, 2]

    # CLAHE — amélioration du contraste adaptatif
    clahe    = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8, 8))
    combined = clahe.apply(combined)

    # Flou gaussien
    k        = PARAMS["kernel_size"]
    img_blur = cv2.GaussianBlur(combined, (k, k), PARAMS["sigma"])

    # Morphologie : OPEN supprime le bruit, CLOSE ferme les contours brisés
    kernel    = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_blur,  cv2.MORPH_OPEN,  kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    # Transformée de Hough
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

    # NMS + filtre couleur
    circles_filtres = nms_circles(
        circles_list, combined,
        overlap_thresh=PARAMS["overlap_thresh"],
        img_bgr=img_bgr,
    )

    # Dessin des cercles sur l'image originale
    img_result = img_bgr.copy()
    for (cx, cy, r) in circles_filtres:
        cv2.circle(img_result, (cx, cy), r,  (0, 200, 80), 2)
        cv2.circle(img_result, (cx, cy), 5,  (0, 200, 80), -1)

    return img_result, len(circles_filtres), circles_filtres


def run_aldo(image_path: str = "data/base_validation/images"):
    images        = load_images(image_path)
    faux_negatifs = 0
    predictions   = {}

    for i, img in enumerate(images[:150]):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        _, nb, _ = detecter_image(img)

        print(f"******** Image {i + 1} ********")
        if nb == 0:
            print("Aucune pièce détectée")
            faux_negatifs += 1
            predictions[f"image{i+1}"] = 0
        else:
            print(f"Nombre de pièces détectées : {nb}")
            predictions[f"image{i+1}"] = nb
            print(f"Faux négatifs cumulés : {faux_negatifs}\n")

    print("\n")
    stats = evaluate_regression(predictions, load_all_labels("data/base_validation/labels"))

    return {
        "predictions": predictions,
        "stats":       stats,
    }


if __name__ == "__main__":
    resultats = run_aldo()

