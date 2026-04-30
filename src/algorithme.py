from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np

def image_gray(image):
    """
    Convertit une image en niveaux de gris
    """
    img = np.array(image)
    if img.ndim == 3:  
        image_gray = np.dot(img[...,:3], [0.298, 0.587, 0.114])
    else:
        image_gray = img

    if img.dtype == np.float32 or img.dtype == np.float64:
        image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return image_gray.astype(np.uint8)
    
def gaussian_kernel(kernel_size, sigma):
    """
    Génère un noyau gaussien 2D
    """
    # Centre du kernel
    center = kernel_size // 2
    
    # Grille de coordonnées
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # Formule gaussienne 2D
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)
    
    # Normalisation : la somme des poids doit valoir 1
    kernel = kernel / kernel.sum()
    
    return kernel


def convolution_2d(image, kernel):
    """
    Applique une convolution 2D sur une image
    """
    img_h, img_w     = image.shape
    ker_h, ker_w     = kernel.shape
    pad_h, pad_w     = ker_h // 2, ker_w // 2

    # Zero-padding pour conserver la taille de l'image
    img_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    output = np.zeros_like(image, dtype=np.float64)

    # Convolution pixel par pixel
    for i in range(img_h):
        for j in range(img_w):
            region = img_padded[i:i+ker_h, j:j+ker_w]
            output[i, j] = np.sum(region * kernel)

    return output


def filtre_gaussian(image, kernel_size=5, sigma=1):
    """
    Applique un filtre gaussien à une image sans cv2
    """
    # kernel_size doit être impair
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel(kernel_size, sigma)

    # Appliquer la convolution 2D
    result = convolution_2d(image, kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def hough_transform(image, dp=1, minDist=20, param1=150, param2=30, minRadius=20, maxRadius=100):
    
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

def scorer_cercle(img_gray, cx, cy, r):
    """Score de crédibilité d'un cercle : plus il est élevé, plus c'est probablement une pièce"""
    
    # 1. Gradient moyen sur le bord (fort = bord net = pièce probable)
    masque_bord = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.circle(masque_bord, (cx, cy), r, 255, 3)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient_bord = gradient[masque_bord > 0].mean()

    # 2. Uniformité intérieure (une pièce a un intérieur relativement uniforme)
    masque_interieur = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.circle(masque_interieur, (cx, cy), int(r * 0.7), 255, -1)  # rempli
    pixels_interieur = img_gray[masque_interieur > 0]
    uniformite = 1.0 / (pixels_interieur.std() + 1)  # plus std est faible, plus c'est uniforme

    score = gradient_bord * uniformite
    return score

def nms_circles(circles, img_gray, overlap_thresh=0.5):
    if len(circles) == 0:
        return []
    
    # Scorer chaque cercle
    cercles_scores = [(cx, cy, r, scorer_cercle(img_gray, cx, cy, r)) 
                      for (cx, cy, r) in circles]
    
    # Trier par score décroissant (le plus crédible en premier)
    cercles_scores = sorted(cercles_scores, key=lambda c: c[3], reverse=True)
    
    final = []
    for (cx, cy, r, score) in cercles_scores:
        trop_proche = False
        for (cx2, cy2, r2, _) in final:
            dist = np.sqrt((int(cx) - int(cx2))**2 + (int(cy) - int(cy2))**2)
            if dist < (r + r2) * overlap_thresh:
                trop_proche = True
                break
        if not trop_proche:
            final.append((cx, cy, r, score))
    
    return [(cx, cy, r) for (cx, cy, r, _) in final]