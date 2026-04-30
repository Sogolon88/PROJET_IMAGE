import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from src.algorithme import hough_transform, nms_circles

# ── Couleurs ──
BG      = "#f8f9fa"
HDR_BG  = "#3b82f6"
HDR_FG  = "white"
CARD_BG = "#ffffff"
CARD_BD = "#dee2e6"
BTN_BLUE   = "#3b82f6"
BTN_GREEN  = "#22c55e"
TEXT    = "#1e293b"
MUTED   = "#64748b"
SUCCESS = "#16a34a"
DANGER  = "#dc2626"

PARAMS = {
    "kernel_size": 5, "sigma": 2.0, "poids_v": 0.9,
    "dp": 1.2, "param1": 100, "param2": 80,
    "minRadius": 20, "maxRadius": 150, "minDist": 30,
    "clip_limit": 2.0, "overlap_thresh": 0.5,
}
MAX_SIDE = 800
IMG_SIZE = 280


def detecter(img_bgr):
    h, w = img_bgr.shape[:2]
    sc = MAX_SIDE / max(h, w)
    if sc < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = cv2.addWeighted(hsv[:,:,2], PARAMS["poids_v"], hsv[:,:,1], 1-PARAMS["poids_v"], 0)
    combined = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8,8)).apply(combined)
    k = PARAMS["kernel_size"]
    blur = cv2.GaussianBlur(combined, (k,k), PARAMS["sigma"])
    km = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(cv2.morphologyEx(blur, cv2.MORPH_OPEN, km), cv2.MORPH_CLOSE, km)
    circles = hough_transform(image=clean, dp=PARAMS["dp"], param1=PARAMS["param1"],
                              param2=PARAMS["param2"], minRadius=PARAMS["minRadius"],
                              minDist=PARAMS["minDist"], maxRadius=PARAMS["maxRadius"])
    lst = [(c[0],c[1],c[2]) for c in circles[0]] if len(circles) else []
    filtres = nms_circles(lst, combined, overlap_thresh=PARAMS["overlap_thresh"])
    res = img_bgr.copy()
    for (cx,cy,r) in filtres:
        cv2.circle(res, (cx,cy), r, (34,197,94), 2)
        cv2.circle(res, (cx,cy), 4, (220,38,38), -1)
    return res, len(filtres)


def bgr_to_tk(img_bgr, size=IMG_SIZE):
    h, w = img_bgr.shape[:2]
    sc = min(size/w, size/h)
    img = cv2.resize(img_bgr, (int(w*sc), int(h*sc)))
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de pièces d'euro")
        self.root.configure(bg=BG)
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        self.img_bgr = None
        self._build()

    def _build(self):
        # ── En-tête ──
        hdr = tk.Frame(self.root, bg=HDR_BG, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Détecteur de pièces d'euro",
                 font=("Arial", 17, "bold"), bg=HDR_BG, fg=HDR_FG).pack()
        tk.Label(hdr, text="Hough Circle Transform",
                 font=("Arial", 9), bg=HDR_BG, fg="#bfdbfe").pack()

        # ── Bouton charger ──
        self._btn(self.root, "Charger une image", self.load_image, BTN_BLUE).pack(pady=12)

        # ── Zone des deux images côte à côte ──
        zone = tk.Frame(self.root, bg=BG)
        zone.pack(padx=20)

        self.lbl_orig = self._card(zone, "Image originale", 0)
        self.lbl_res  = self._card(zone, "Résultat détection", 1)

        # ── Bouton détecter ──
        self._btn(self.root, "Lancer la détection", self.detect, BTN_GREEN).pack(pady=12)

        # ── Label résultat ──
        self.lbl_count = tk.Label(self.root, text="",
                                  font=("Arial", 13, "bold"), bg=BG, fg=TEXT)
        self.lbl_count.pack()

        # ── Barre de statut ──
        self.lbl_statut = tk.Label(self.root, text="Prêt",
                                   font=("Arial", 9), bg="#e2e8f0",
                                   fg=MUTED, anchor="w", padx=10, pady=3)
        self.lbl_statut.pack(fill="x", side="bottom")

    def _card(self, parent, titre, col):
        outer = tk.Frame(parent, bg=CARD_BD, padx=1, pady=1)
        outer.grid(row=0, column=col, padx=10)
        inner = tk.Frame(outer, bg=CARD_BG, padx=6, pady=6)
        inner.pack()
        tk.Label(inner, text=titre, font=("Arial", 10, "bold"),
                 bg=CARD_BG, fg=MUTED).pack(pady=(2,4))
        lbl = tk.Label(inner, bg="#f1f5f9", width=IMG_SIZE, height=IMG_SIZE,
                       text="—", fg=MUTED, font=("Arial", 12))
        lbl.pack()
        return lbl

    def _btn(self, parent, texte, cmd, bg):
        r, g, b = int(bg[1:3],16), int(bg[3:5],16), int(bg[5:7],16)
        hover = "#{:02x}{:02x}{:02x}".format(max(0,r-25), max(0,g-25), max(0,b-25))
        btn = tk.Button(parent, text=texte, command=cmd,
                        font=("Arial", 11, "bold"), bg=bg, fg="white",
                        activebackground=hover, activeforeground="white",
                        relief="flat", padx=20, pady=8, cursor="hand2", bd=0)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not path:
            return
        self.img_bgr = cv2.imread(path)
        if self.img_bgr is None:
            self.lbl_statut.config(text="Erreur : impossible de lire l'image")
            return
        tk_img = bgr_to_tk(self.img_bgr)
        self.lbl_orig.config(image=tk_img, text="", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_orig.image = tk_img
        self.lbl_res.config(image="", text="—", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_count.config(text="")
        self.lbl_statut.config(text=f"Image chargée : {path.split('/')[-1]}")

    def detect(self):
        if self.img_bgr is None:
            self.lbl_statut.config(text="Veuillez d'abord charger une image")
            return
        self.lbl_statut.config(text="Détection en cours...")
        self.root.update()
        res, nb = detecter(self.img_bgr)
        tk_res = bgr_to_tk(res)
        self.lbl_res.config(image=tk_res, text="", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_res.image = tk_res
        if nb == 0:
            self.lbl_count.config(text="Aucune pièce détectée", fg=DANGER)
        else:
            self.lbl_count.config(text=f"{nb} pièce(s) détectée(s)", fg=SUCCESS)
        self.lbl_statut.config(text="Détection terminée")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
