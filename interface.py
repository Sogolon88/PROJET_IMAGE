import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
from src.algorithme import hough_transform, nms_circles

# ── Palette ──
BG        = "#f0f2f8"
SURFACE   = "#1e2235"
SURFACE2  = "#ffffff"
ACCENT    = "#6366f1"
ACCENT2   = "#0ea5e9"
SUCCESS   = "#10b981"
DANGER    = "#ef4444"
WARNING   = "#f59e0b"
TEXT      = "#1e293b"
MUTED     = "#94a3b8"
BORDER    = "#e2e8f0"

PARAMS = {
    "kernel_size": 11, "sigma": 2.5, "poids_v": 1.0,
    "dp": 1, "param1": 50, "param2": 60,
    "minRadius": 25, "maxRadius": 150, "minDist": 30,
    "clip_limit": 2.5, "overlap_thresh": 0.8,
}
MAX_SIDE = 800
IMG_SIZE = 300
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Dossiers des labels (vérité terrain)
LABELS_DIRS = [
    os.path.join(os.path.dirname(__file__), "data", "base_validation", "labels"),
    os.path.join(os.path.dirname(__file__), "data", "base_test", "labels"),
]


def charger_verite_terrain(image_path):
    """Cherche le fichier JSON de vérité terrain correspondant à l'image chargée."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for labels_dir in LABELS_DIRS:
        json_path = os.path.join(labels_dir, stem + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data.get("shapes", []))
    return None


def detecter(img_bgr):
    h, w = img_bgr.shape[:2]
    sc = MAX_SIDE / max(h, w)
    if sc < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)

    # Canal V (luminosité) uniquement — identique à main.py
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = hsv[:, :, 2]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=PARAMS["clip_limit"], tileGridSize=(8, 8))
    combined = clahe.apply(combined)

    # Flou gaussien
    k = PARAMS["kernel_size"]
    blur = cv2.GaussianBlur(combined, (k, k), PARAMS["sigma"])

    # Morphologie OPEN + CLOSE
    km = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(blur,  cv2.MORPH_OPEN,  km)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, km)

    # Transformée de Hough
    circles = hough_transform(
        image=clean,
        dp=PARAMS["dp"],
        param1=PARAMS["param1"],
        param2=PARAMS["param2"],
        minRadius=PARAMS["minRadius"],
        minDist=PARAMS["minDist"],
        maxRadius=PARAMS["maxRadius"]
    )

    # NMS
    lst = [(c[0], c[1], c[2]) for c in circles[0]] if len(circles) else []
    filtres = nms_circles(lst, combined, overlap_thresh=PARAMS["overlap_thresh"], img_bgr=img_bgr)

    # Dessin des cercles détectés
    res = img_bgr.copy()
    for (cx, cy, r) in filtres:
        cv2.circle(res, (cx, cy), r, (34, 197, 94), 2)
        cv2.circle(res, (cx, cy), 5, (34, 197, 94), -1)
    return res, len(filtres)


def bgr_to_tk(img_bgr, size=IMG_SIZE):
    h, w = img_bgr.shape[:2]
    sc = min(size/w, size/h)
    img = cv2.resize(img_bgr, (int(w*sc), int(h*sc)))
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Coin Recognition")
        self.root.configure(bg=BG)
        self.root.geometry("980x640")
        self.root.resizable(True, True)
        self.root.minsize(900, 580)
        self.img_bgr   = None
        self.img_path  = None
        # Mode dossier
        self.folder_images = []   # liste des chemins d'images du dossier
        self.folder_index  = 0   # index courant
        self._build()

    def _build(self):
        # ── Barre latérale gauche ──
        sidebar = tk.Frame(self.root, bg=SURFACE, width=210)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Logo / titre
        tk.Label(sidebar, text="◉", font=("Arial", 28), bg=SURFACE,
                 fg=ACCENT).pack(pady=(30, 4))
        tk.Label(sidebar, text="Coin\nRecognition", font=("Arial", 13, "bold"),
                 bg=SURFACE, fg="#f1f5f9", justify="center").pack()
        tk.Label(sidebar, text="Hough Circle Transform",
                 font=("Arial", 8), bg=SURFACE, fg="#94a3b8").pack(pady=(2, 20))

        # Séparateur
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

        # ── Boutons charger ──
        self._sidebar_btn(sidebar, "  ＋  Charger image", self.load_image, ACCENT).pack(
            padx=20, pady=4, fill="x")
        self._sidebar_btn(sidebar, "  ▶  Lancer détection", self.detect, SUCCESS).pack(
            padx=20, pady=4, fill="x")

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

        self._sidebar_btn(sidebar, "  📁  Charger dossier", self.load_folder, "#7c3aed").pack(
            padx=20, pady=4, fill="x")

        # Navigation dossier — conteneur fixe toujours présent dans le layout
        # On masque les widgets internes plutôt que le frame lui-même
        self.nav_frame = tk.Frame(sidebar, bg=SURFACE, height=60)
        self.nav_frame.pack(fill="x", padx=20, pady=4)
        self.nav_frame.pack_propagate(False)

        nav_btns = tk.Frame(self.nav_frame, bg=SURFACE)
        nav_btns.pack(fill="x")

        self.btn_prev = tk.Button(nav_btns, text="◀", command=self.prev_image,
                                  font=("Arial", 11, "bold"), bg="#374151", fg="white",
                                  relief="flat", padx=12, pady=6, cursor="hand2", bd=0)
        self.btn_prev.pack(side="left", expand=True, fill="x")

        self.btn_next = tk.Button(nav_btns, text="▶", command=self.next_image,
                                  font=("Arial", 11, "bold"), bg="#374151", fg="white",
                                  relief="flat", padx=12, pady=6, cursor="hand2", bd=0)
        self.btn_next.pack(side="right", expand=True, fill="x")

        self.lbl_nav = tk.Label(self.nav_frame, text="", font=("Arial", 8),
                                bg=SURFACE, fg="#94a3b8")
        self.lbl_nav.pack(pady=(4, 0))

        # Masquer les boutons au départ (pas le frame)
        self.btn_prev.pack_forget()
        self.btn_next.pack_forget()
        self.lbl_nav.pack_forget()

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=16)

        # ── Section Résultats ──
        tk.Label(sidebar, text="RÉSULTATS", font=("Arial", 8, "bold"),
                 bg=SURFACE, fg="#94a3b8").pack(padx=20, anchor="w", pady=(0, 8))

        self._stat_row(sidebar, "Détecté",   "—", "detected")
        self._stat_row(sidebar, "Réel (GT)", "—", "ground_truth")
        self._stat_row(sidebar, "Erreur",    "—", "error")
        self._stat_row(sidebar, "MSE",       "—", "mse")

        # Compteur principal en bas
        self.lbl_count = tk.Label(sidebar, text="", font=("Arial", 28, "bold"),
                                  bg=SURFACE, fg=SUCCESS)
        self.lbl_count.pack(side="bottom", pady=(10, 20))
        self.lbl_count_label = tk.Label(sidebar, text="", font=("Arial", 9),
                                        bg=SURFACE, fg="#94a3b8")
        self.lbl_count_label.pack(side="bottom")
        tk.Frame(sidebar, bg=BORDER, height=1).pack(side="bottom", fill="x", padx=20, pady=6)

        # ── Zone principale ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # Titre zone
        top_bar = tk.Frame(main, bg=BG)
        top_bar.pack(fill="x", padx=24, pady=(20, 10))
        self.lbl_titre = tk.Label(top_bar, text="Visualisation", font=("Arial", 14, "bold"),
                                  bg=BG, fg=TEXT)
        self.lbl_titre.pack(side="left")
        self.lbl_statut = tk.Label(top_bar, text="● Prêt", font=("Arial", 9),
                                   bg=BG, fg=MUTED)
        self.lbl_statut.pack(side="right")

        # Zone images
        imgs_frame = tk.Frame(main, bg=BG)
        imgs_frame.pack(fill="both", expand=True, padx=24, pady=4)
        imgs_frame.columnconfigure(0, weight=1)
        imgs_frame.columnconfigure(1, weight=1)
        imgs_frame.rowconfigure(0, weight=1)

        self.lbl_orig = self._img_card(imgs_frame, "ORIGINAL", 0)
        self.lbl_res  = self._img_card(imgs_frame, "DÉTECTION", 1)

    # ── Helpers UI ──

    def _stat_row(self, parent, label, value, key):
        """Crée une ligne de stat dans la sidebar et mémorise le widget valeur."""
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=20, pady=3)
        tk.Label(row, text=label, font=("Arial", 9), bg=SURFACE,
                 fg="#94a3b8", anchor="w").pack(side="left")
        val_lbl = tk.Label(row, text=value, font=("Arial", 9, "bold"),
                           bg=SURFACE, fg=ACCENT2, anchor="e")
        val_lbl.pack(side="right")
        setattr(self, f"stat_{key}", val_lbl)

    def _img_card(self, parent, titre, col):
        card = tk.Frame(parent, bg=SURFACE2, bd=0)
        card.grid(row=0, column=col, padx=8, sticky="nsew")
        tk.Label(card, text=titre, font=("Arial", 8, "bold"),
                 bg=SURFACE2, fg=MUTED, pady=8).pack()
        border_frame = tk.Frame(card, bg=BORDER, padx=1, pady=1)
        border_frame.pack(padx=12, pady=(0, 12), fill="both", expand=True)
        lbl = tk.Label(border_frame, bg="#e8edf5",
                       width=IMG_SIZE, height=IMG_SIZE,
                       text="Aucune image", fg=MUTED, font=("Arial", 10))
        lbl.pack(fill="both", expand=True)
        return lbl

    def _sidebar_btn(self, parent, texte, cmd, color):
        btn = tk.Button(parent, text=texte, command=cmd,
                        font=("Arial", 10, "bold"), bg=color, fg="white",
                        activebackground=color, activeforeground="white",
                        relief="flat", pady=10, cursor="hand2", bd=0, anchor="w")
        darker = self._darken(color)
        btn.bind("<Enter>", lambda e: btn.config(bg=darker))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def _darken(self, hex_color, amount=30):
        r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
        return "#{:02x}{:02x}{:02x}".format(
            max(0, r-amount), max(0, g-amount), max(0, b-amount))

    def _reset_stats(self):
        for key in ("detected", "ground_truth", "error", "mse"):
            getattr(self, f"stat_{key}").config(text="—", fg=ACCENT2)

    def _set_status(self, msg, color):
        self.lbl_statut.config(text=msg, fg=color)

    # ── Chargement image unique ──

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not path:
            return
        # Quitter le mode dossier — masquer les boutons de navigation
        self.folder_images = []
        self.btn_prev.pack_forget()
        self.btn_next.pack_forget()
        self.lbl_nav.pack_forget()
        self._afficher_image(path)

    def _afficher_image(self, path):
        # Lecture robuste via numpy pour contourner le bug OpenCV
        # sur les chemins Windows contenant des espaces ou caractères accentués
        try:
            with open(path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            self._set_status("● Erreur : impossible de lire l'image", DANGER)
            return
        self.img_bgr  = img
        self.img_path = path
        tk_img = bgr_to_tk(self.img_bgr)
        self.lbl_orig.config(image=tk_img, text="", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_orig.image = tk_img
        self.lbl_res.config(image="", text="Aucune image", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_count.config(text="")
        self.lbl_count_label.config(text="")
        self._reset_stats()
        name = os.path.basename(path)
        self.lbl_titre.config(text=name)
        self._set_status(f"● {name} chargée", ACCENT2)

    # ── Chargement dossier ──

    def load_folder(self):
        folder = filedialog.askdirectory(title="Choisir un dossier d'images")
        if not folder:
            return
        images = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if not images:
            self._set_status("● Aucune image trouvée dans ce dossier", DANGER)
            return
        self.folder_images = images
        self.folder_index  = 0
        # Afficher les boutons de navigation
        self.btn_prev.pack(side="left", expand=True, fill="x")
        self.btn_next.pack(side="right", expand=True, fill="x")
        self.lbl_nav.pack(pady=(4, 0))
        self._charger_image_dossier()

    def _charger_image_dossier(self):
        path = self.folder_images[self.folder_index]
        self._afficher_image(path)
        total = len(self.folder_images)
        self.lbl_nav.config(
            text=f"Image {self.folder_index + 1} / {total}")
        # Détecter automatiquement
        self.detect()

    def prev_image(self):
        if not self.folder_images:
            return
        self.folder_index = (self.folder_index - 1) % len(self.folder_images)
        self._charger_image_dossier()

    def next_image(self):
        if not self.folder_images:
            return
        self.folder_index = (self.folder_index + 1) % len(self.folder_images)
        self._charger_image_dossier()

    # ── Détection ──

    def detect(self):
        if self.img_bgr is None:
            self._set_status("● Chargez d'abord une image", DANGER)
            return
        self._set_status("● Détection en cours...", ACCENT)
        self.root.update()
        res, nb_detecte = detecter(self.img_bgr)

        # Affichage image résultat
        tk_res = bgr_to_tk(res)
        self.lbl_res.config(image=tk_res, text="", width=IMG_SIZE, height=IMG_SIZE)
        self.lbl_res.image = tk_res

        # Compteur principal
        if nb_detecte == 0:
            self.lbl_count.config(text="0", fg=DANGER)
            self.lbl_count_label.config(text="pièce détectée", fg="#94a3b8")
            self._set_status("● Aucune pièce trouvée", DANGER)
        else:
            self.lbl_count.config(text=str(nb_detecte), fg=SUCCESS)
            self.lbl_count_label.config(
                text="pièce détectée" if nb_detecte == 1 else "pièces détectées",
                fg="#727d8d")
            self._set_status(f"● {nb_detecte} pièce(s) détectée(s)", SUCCESS)

        # ── Mise à jour des stats ──
        self.stat_detected.config(text=str(nb_detecte), fg=ACCENT2)

        # Vérité terrain
        nb_reel = charger_verite_terrain(self.img_path) if self.img_path else None
        if nb_reel is not None:
            self.stat_ground_truth.config(text=str(nb_reel), fg=ACCENT2)
            erreur = nb_detecte - nb_reel
            # Erreur absolue
            if erreur == 0:
                self.stat_error.config(text="0  ✓", fg=SUCCESS)
            elif erreur > 0:
                self.stat_error.config(text=f"+{erreur}", fg=WARNING)
            else:
                self.stat_error.config(text=str(erreur), fg=DANGER)
            # MSE (sur une seule image = erreur²)
            mse = erreur ** 2
            mse_color = SUCCESS if mse == 0 else (WARNING if mse <= 4 else DANGER)
            self.stat_mse.config(text=f"{mse:.2f}", fg=mse_color)
        else:
            self.stat_ground_truth.config(text="N/A", fg=MUTED)
            self.stat_error.config(text="N/A", fg=MUTED)
            self.stat_mse.config(text="N/A", fg=MUTED)

    def _set_status(self, msg, color):
        self.lbl_statut.config(text=msg, fg=color)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
