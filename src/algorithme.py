"""
algorithme.py

Ce fichier regroupe les briques de bas niveau du pipeline de detection :
conversion en niveaux de gris, filtre gaussien maison, transformee de Hough
pour trouver les cercles, et NMS pour eliminer les doublons.
"""

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np


def image_gray(image):
    """
    Convertit une image RGB en tableau uint8 monochrome.

    On utilise les coefficients ITU-R BT.601 (0.298, 0.587, 0.114) plutot
    qu'une simple moyenne des trois canaux, parce que l'oeil humain est bien
    plus sensible au vert qu'au rouge ou au bleu. Si l'image est en flottant
    (0..1), on la normalise vers 0..255 avant de la retourner.
    """
    img = np.array(image)
    if img.ndim == 3:
        image_gray = np.dot(img[..., :3], [0.298, 0.587, 0.114])
    else:
        image_gray = img
    if img.dtype == np.float32 or img.dtype == np.float64:
        image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return image_gray.astype(np.uint8)


def gaussian_kernel(kernel_size, sigma):
    """
    Construit un noyau gaussien 2D de taille kernel_size x kernel_size.

    On evalue la gaussienne e^(-(x^2+y^2) / 2*sigma^2) sur une grille centree
    en zero, puis on normalise pour que la somme vaille 1. Ca garantit que le
    filtre ne modifie pas la luminosite globale, il ne fait que lisser.
    """
    center = kernel_size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * np.pi * sigma ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def convolution_2d(image, kernel):
    """
    Applique une convolution 2D sur l'image avec le noyau donne.

    On fait glisser le noyau sur chaque pixel et on calcule la somme ponderee
    de son voisinage. Les bords sont geres en mode 'reflect' pour eviter les
    artefacts noirs. Cette implementation naive est a but pedagogique ; en
    pratique on utilise cv2.filter2D qui est beaucoup plus rapide.
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    pad_h, pad_w = ker_h // 2, ker_w // 2
    img_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(img_h):
        for j in range(img_w):
            region = img_padded[i:i + ker_h, j:j + ker_w]
            output[i, j] = np.sum(region * kernel)
    return output


def filtre_gaussian(image, kernel_size=5, sigma=1):
    """
    Lisse l'image avec un filtre gaussien.

    On s'assure que kernel_size est impair (un noyau pair n'a pas de centre
    exact). Le resultat est clippe entre 0 et 255. Ce flou attenuer le bruit
    haute frequence avant la detection de cercles, sinon les petites variations
    de luminosite genereront de faux bords.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel(kernel_size, sigma)
    result = convolution_2d(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Detecte les contours de l'image avec l'algorithme de Canny.

    Canny fonctionne en quatre etapes : flou gaussien, calcul du gradient,
    suppression des non-maxima (on ne garde que le pixel le plus fort le long
    d'un bord), puis seuillage par hysteresis avec deux seuils. Un pixel est
    retenu comme bord fort si son gradient depasse high_threshold, comme bord
    faible s'il est entre les deux seuils et connecte a un bord fort.
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def hough_transform(image, dp=1, minDist=20, param1=150, param2=30,
                    minRadius=20, maxRadius=100):
    """
    Detecte les cercles dans l'image avec la transformee de Hough circulaire.

    Pour chaque point de contour, on vote dans un espace d'accumulation a trois
    dimensions (cx, cy, r). Les cellules qui recoivent beaucoup de votes
    correspondent a des cercles reels dans l'image.

    param1 est le seuil haut de Canny utilise en interne. param2 est le seuil
    de l'accumulateur : plus il est bas, plus on detecte de cercles (y compris
    des faux positifs). minDist evite de detecter plusieurs fois le meme cercle.
    """
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    if circles is None:
        return []
    circles = np.uint16(np.around(circles))
    return circles


def est_couleur_piece(img_bgr, cx, cy, r, seuil_ratio=0.45):
    """
    Verifie si la zone d'un cercle ressemble visuellement a une piece.

    On travaille en HSV parce que la saturation et la valeur separent mieux
    la couleur de la luminosite qu'en RGB. On definit trois familles metalliques :
    doree (teinte orange-jaune), argentee (faible saturation, valeur elevee) et
    bimetallique (teinte plus large). Si au moins 45% des pixels du disque
    appartiennent a une de ces familles, on considere que c'est une piece.
    """
    masque = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(masque, (cx, cy), int(r * 0.75), 255, -1)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[masque > 0]
    if len(pixels) == 0:
        return False
    h = pixels[:, 0].astype(float)
    s = pixels[:, 1].astype(float)
    v = pixels[:, 2].astype(float)
    if v.mean() < 55:
        return False
    doree    = (h >= 5)  & (h <= 35) & (s >= 40) & (v >= 50)
    argentee = (s < 80)  & (v >= 100)
    bimetall = (h >= 5)  & (h <= 90) & (s >= 20) & (v >= 100)
    ratio = (doree | argentee | bimetall).sum() / len(pixels)
    return ratio >= seuil_ratio


def scorer_cercle(img_gray, cx, cy, r, uniformite_kernel=9):
    """
    Attribue un score de qualite a un cercle detecte.

    Le score combine deux criteres : la force du gradient sur le bord du cercle
    (un vrai bord de piece produit une transition nette) et l'uniformite de
    l'interieur (une piece a une surface assez homogene). On multiplie les deux
    pour favoriser les cercles qui ont a la fois un bord net et un interieur lisse.
    """
    masque_bord = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.circle(masque_bord, (cx, cy), r, 255, 3)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_bord = np.sqrt(sobelx ** 2 + sobely ** 2)[masque_bord > 0].mean()

    k = uniformite_kernel if uniformite_kernel % 2 == 1 else uniformite_kernel + 1
    img_lisse = cv2.GaussianBlur(img_gray, (k, k), 0)
    masque_int = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.circle(masque_int, (cx, cy), int(r * 0.7), 255, -1)
    uniformite = 1.0 / (img_lisse[masque_int > 0].std() + 1)

    return gradient_bord * uniformite


def nms_circles(circles, img_gray, overlap_thresh=0.5, img_bgr=None, uniformite_kernel=9):
    """
    Non-Maximum Suppression adaptee aux cercles.

    Apres Hough, plusieurs cercles proches sont souvent detectes pour la meme
    piece. La NMS resout ca en trois etapes :
    1. Filtre couleur : on elimine les cercles dont la zone ne ressemble pas a
       une piece metallique (si l'image BGR est fournie).
    2. Scoring : on attribue un score a chaque cercle (gradient x uniformite).
    3. Suppression : on trie par score decroissant et on rejette tout cercle
       trop proche d'un cercle deja retenu. "Trop proche" signifie que la
       distance entre centres est inferieure a (r1 + r2) * overlap_thresh.
    """
    if len(circles) == 0:
        return []

    if img_bgr is not None:
        circles = [(cx, cy, r) for (cx, cy, r) in circles
                   if est_couleur_piece(img_bgr, cx, cy, r)]
    if len(circles) == 0:
        return []

    cercles_scores = sorted(
        [(cx, cy, r, scorer_cercle(img_gray, cx, cy, r, uniformite_kernel))
         for (cx, cy, r) in circles],
        key=lambda c: c[3], reverse=True
    )

    final = []
    for (cx, cy, r, score) in cercles_scores:
        trop_proche = any(
            np.sqrt((int(cx) - int(cx2))**2 + (int(cy) - int(cy2))**2) < (r + r2) * overlap_thresh
            for (cx2, cy2, r2, _) in final
        )
        if not trop_proche:
            final.append((cx, cy, r, score))

    return [(cx, cy, r) for (cx, cy, r, _) in final]
