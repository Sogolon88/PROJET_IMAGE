import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import json
import math
import os

"""
Ce fichier python regroupe les fonctions de traitement des images,
tels que Otsu, canny, fermeture etc
"""


def polygon_to_mask(points, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(poly1, poly2, image_shape):
    mask1 = polygon_to_mask(poly1, image_shape)
    mask2 = polygon_to_mask(poly2, image_shape)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0
    return intersection / union

def image_gray(image_path):
    """
    Convertit une image couleur en niveaux de gris en utilisant la formule de luminance.
    """
    img = mplimg.imread(image_path)

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    gray_image = np.dot(img[..., :3], [0.299, 0.587, 0.114])

    return gray_image.astype(np.uint8)


def algo_otsu(image_path):
    """
    Applique l'algorithme de seuillage d'Otsu sur une image en niveaux de gris.
    """
    img = image_gray(image_path)

    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    total_pixels = img.size
    prob = hist / total_pixels

    cumulative_sum = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * np.arange(256))
    global_mean = cumulative_mean[-1]

    max_variance = 0
    optimal_threshold = 0

    for t in range(1, 256):
        weight_bg = cumulative_sum[t]
        weight_fg = 1 - weight_bg
        mean_bg = cumulative_mean[t] / weight_bg if weight_bg != 0 else 0
        mean_fg = (cumulative_mean[-1] - cumulative_mean[t]) / weight_fg if weight_fg != 0 else 0
        variance_inter_class = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance_inter_class > max_variance:
            max_variance = variance_inter_class
            optimal_threshold = t

    binary_image = img > optimal_threshold
    binary_image = binary_image.astype(np.uint8) * 255  # Pour affichage OpenCV

    return binary_image, img  # On retourne aussi l'image en niveaux de gris

def detect_circle(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gaussien = cv2.GaussianBlur(gray,(7,7),0)
    edges = cv2.Canny(gaussien,50,150)

    circles = cv2.HoughCircles(gaussien, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=100, param2=28, minRadius=25, maxRadius=130)

    return circles

def count_gt(json_path):
    if not os.path.exists(json_path):
        return 0
    
    with open(json_path) as f:
        data = json.load(f)
    
    return len(data.get("shapes", []))

def circle_to_polygon(cx, cy, r , n_points= 50):
    points = []
    for i in range (n_points):
        angle = 2 * np.pi * i / n_points
        x = int (cx + r * np.cos(angle))
        y = int (cy + r * np.sin(angle))
        points.append ([x,y])
    return points

def draw_hough_circles(img, circles, color=(0, 255, 0), thickness=2):
    """
        Dessine les segments de lignes détectés par HoughLinesP sur l'image.

        Paramètres :
            img (ndarray) : Image d'origine, en niveaux de gris ou en couleur.
            lignes (ndarray) : Résultat retourné par HoughLinesP.
            couleur (tuple) : Couleur des lignes tracées (par défaut : rouge en BGR).
            epaisseur (int) : Épaisseur des lignes (par défaut : 2).

        Retour :
            img_avec_cercles (ndarray) : Copie de l'image avec les lignes tracées.
    """
   
    img_with_circles = img.copy()

    if circles is not None:
        """La liste des cercles"""
        for circle in circles [0,:]:
            x,y,r = circle  
            cv2.circle(img_with_circles, (x, y), r ,color, thickness)

    return img_with_circles

    
# =========================
#         TRAITER IMAGE VERSION GAUSSIAN
# =========================


def traiter_dossier_v1(dossier_images, dossier_annotations,valid_ext):
    os.makedirs(dossier_annotations, exist_ok=True)


    fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith(valid_ext)])
    errors = []
    for fichier in fichiers:
        image_path = os.path.join(dossier_images, fichier)
        nom_base = os.path.splitext(fichier)[0]
        json_path = os.path.join(dossier_annotations, f"{nom_base}.json")

        print(f"Traitement de {fichier}...")

        binary_image, gray_img = algo_otsu(image_path)
        
        img = cv2.imread(image_path)
        # === PIPELINE VISUEL ===

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussien = cv2.GaussianBlur(gray, (7,7), 0)

        # Otsu
        _, otsu = cv2.threshold(
        gaussien, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Canny
        edges = cv2.Canny(gaussien, 50, 150)

        #HoughCircles
        circles = cv2.HoughCircles(
        gaussien,  
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60  ,
        param1=100,
        param2=28,
        minRadius=25,
        maxRadius=130
    )

        # === AFFICHAGE ===
        cv2.imshow("Gray", gray)
        cv2.imshow("Gaussian", gaussien)
        cv2.imshow("Otsu", otsu)
        cv2.imshow("Canny", edges)

        img_hough = img.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for c in circles[0]:
                x_c, y_c, r_c = c
                cv2.circle(img_hough, (x_c, y_c), r_c, (0,255,0), 2)

        cv2.imshow("HoughCircles", img_hough)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    
        #Conversion en polygone
        if circles is None :
            print(f"[!] Aucun cercle détecté ")
            all_polygons = []
        else:
            circles = np.uint16(np.around(circles))
            all_polygons =[]

            for c in circles[0]:
                x,y,r = c
                poly = circle_to_polygon (x ,y , r)
                all_polygons.append(poly)
        print(f"Nombre de pièces détectées : {len(all_polygons)}")
         #Sauvagarde JSON    
        annotation = {
            "shapes" : [ 
                {
                    "label" : "piece",
                    "points" : poly,
                       "shape_type" : "polygon"
                }
                   for poly in all_polygons                        ]
        }
        with open(json_path, 'w') as f:
               json.dump(annotation, f, indent=4)
        print(f"Annotation sauvegardé  : {json_path}")

        
        # === COMPARAISON AVEC GT ===
        json_gt_path = os.path.join("Annotations_GT", f"{nom_base}.json")

        nb_detected = len(all_polygons)
        nb_gt = count_gt(json_gt_path)

        error = abs(nb_detected - nb_gt)
        print(f"GT: {nb_gt} | Détecté: {nb_detected} | Erreur: {error}")


# === Lancer le traitement ===
if __name__ == "__main__":
      
    dossier_images = "data/base_validation/images"
    dossier_annotations = "data/base_validation/labels"

    valid_ext = ('.jpg', '.jpeg', '.png')
    traiter_dossier_v1(dossier_images, dossier_annotations,valid_ext)
    #traiter_dossier_v2(dossier_images, dossier_annotations,valid_ext)
    print("Fini")

