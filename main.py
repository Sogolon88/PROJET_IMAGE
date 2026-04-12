from src.algorithme import * 
from data.images import *
from src.utiles import load_images, load_all_labels
from src.evaluation import evaluate_regression
from matplotlib import pyplot as plt

if __name__ == "__main__":
    images = load_images("data/base_validation/images")
    TARGET_SIZE = (640, 480)
    faux_negatifs = 0
    predictions = {}

    for i, img in enumerate(images[19:30]):
        img = cv2.resize(np.array(img), TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        img_gray = image_gray(img)
        #img_gray = cv2.equalizeHist(img_gray)

        # Flou modéré pour préserver les contours circulaires
        img_blur = filtre_gaussian(img_gray, kernel_size=11, sigma=2)
        #img_blur = cv2.GaussianBlur(img_gray, (7, 7), 2)

        #img_otsu = cv2.threshold(
        #    img_blur, 0, 255,
        #    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        #)

        circles = hough_transform(
            image=img_blur,     
            dp=1.2,
            param1=80,
            param2=40,            
            minRadius=30,         
            minDist=60,           
            maxRadius=100        
        )

        print(f"******** Image {i + 1} ********")
        if len(circles) == 0:
            print("Aucune pièce détectée")
            faux_negatifs += 1
            predictions[f"image{i+1}"] = 0
        else:
            print(f"Nombre de pièces détectées : {len(circles[0])}")
            predictions[f"image{i+1}"] = len(circles[0])

            img_result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for circle in circles[0, :]:
                cx, cy, r = circle[0], circle[1], circle[2]
                cv2.circle(img_result, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(img_result, (cx, cy), 2, (0, 0, 255), 3)
            print(f"Faux négatifs cumulés : {faux_negatifs}\n")

            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.title(f"Image {i+1} — {len(circles[0])} pièce(s)")
            plt.axis('off')
            plt.show()
        
    print("\n")
    evaluate_regression(predictions, load_all_labels("data/base_validation/labels"))