from src.algorithme import * 
from src.algorithme import nms_circles
from data.images import *
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression
from matplotlib import pyplot as plt

if __name__ == "__main__":
    images = load_images("data/base_validation/images")
    MAX_SIDE = 800
    faux_negatifs = 0
    predictions = {}

    for i, img in enumerate(images[:150]):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Canaux HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        combined = hsv[:, :, 2] # cv2.addWeighted(v, 1, s, 0., 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        combined = clahe.apply(combined)

        # Flou gaussien
        img_blur = cv2.GaussianBlur(combined, (11, 11), 2.5)

        # Morphologie : OPEN supprime le bruit, CLOSE ferme les contours brisés
        kernel = np.ones((3, 3), np.uint8)
        img_clean = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel)
        img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

        circles = hough_transform(
            image=img_clean,     
            dp=1,
            param1=50,
            param2=60,            
            minRadius=25,         
            minDist=30,           
            maxRadius=150        
        )

        circles_list = []
        if len(circles) != 0:
            circles_list = [(c[0], c[1], c[2]) for c in circles[0]]
    
        circles_filtres = nms_circles(circles_list, combined, overlap_thresh=0.6)

        print(f"******** Image {i + 1} ********")
        if len(circles_filtres) == 0:
            print("Aucune pièce détectée")
            faux_negatifs += 1
            predictions[f"image{i+1}"] = 0
        else:
            print(f"Nombre de pièces détectées : {len(circles_filtres)}")
            predictions[f"image{i+1}"] = len(circles_filtres)

            img_result = cv2.cvtColor(img_clean, cv2.COLOR_GRAY2BGR)
            for (cx, cy, r) in circles_filtres:
                cv2.circle(img_result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(img_result, (cx, cy), 2, (0, 0, 255), 3)
            print(f"Faux négatifs cumulés : {faux_negatifs}\n")

            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.title(f"Image {i+1} — {len(circles_filtres)} pièce(s)")
            plt.axis('off')
            plt.show()

    # Evaluation une seule fois à la fin
    print("\n")
    evaluate_regression(predictions, load_all_labels("data/base_validation/labels"))