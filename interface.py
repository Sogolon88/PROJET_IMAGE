import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
from main import detecter_image

# ── Thème CustomTkinter ──
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Palette ──
BG         = "#eaeaee"   # droite  : gris perle clair
SIDEBAR    = "#2e2e36"   # gauche  : gris charbon profond
CARD       = "#f4f4f6"   # cartes droite : blanc cassé très doux
CARD2      = "#3c3c46"   # badges stats sidebar
SEP        = "#44444e"   # séparateurs sidebar
ACCENT     = "#b0a2e0"   # violet moyen (lisible sur charbon)
ACCENT2    = "#d4c6f8"   # lilas clair (valeurs stats — bon contraste sur CARD2)
SUCCESS    = "#72c99a"   # vert menthe
DANGER     = "#e08090"   # rouge rosé doux
WARNING    = "#dba060"   # ocre chaud

# Textes sidebar (fond sombre)
TEXT_DARK  = "#f5f3fa"   # principal — blanc cassé très doux
SUB_DARK   = "#cdc9de"   # sous-titre / tagline  — gris lavande clair
MUTED_DARK = "#a09cb4"   # secondaire / placeholders — plus lisible sur charbon

# Textes droite (fond clair)
TEXT_LIGHT = "#1e1a30"   # principal — quasi-noir profond
SUB_LIGHT  = "#44406a"   # sous-titres / labels des cartes
MUTED_LIGHT= "#6e6a88"   # placeholders / statut inactif

BORDER     = "#d4d0dc"   # bordure cartes côté clair
BTN_GRAY   = "#484854"   # boutons sidebar
BTN_GRAY_H = "#5c5c6a"   # hover boutons sidebar

IMG_SIZE = 260
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

LABELS_DIRS = [
    os.path.join(os.path.dirname(__file__), "data", "base_validation", "labels"),
    os.path.join(os.path.dirname(__file__), "data", "base_test", "labels"),
]

DEFAULT_IMAGES_DIR = os.path.join(
    os.path.dirname(__file__), "data", "base_test", "images"
)


# ══════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════

def charger_verite_terrain(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for labels_dir in LABELS_DIRS:
        json_path = os.path.join(labels_dir, stem + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data.get("shapes", []))
    return None


def detecter(img_bgr):
    img_result, nb, _ = detecter_image(img_bgr)
    return img_result, nb


def bgr_to_ctk(img_bgr, size=IMG_SIZE):
    h, w = img_bgr.shape[:2]
    sc   = min(size / w, size / h)
    nw, nh = int(w * sc), int(h * sc)
    img  = cv2.resize(img_bgr, (nw, nh))
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))


def _make_blank_ctk(size=IMG_SIZE):
    """Image vide transparente — remplace image=None pour CTkLabel."""
    pil = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(size, size))


# ══════════════════════════════════════════════
#  Application principale
# ══════════════════════════════════════════════

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CoinVision")
        self.geometry("780x600")
        self.minsize(780, 540)
        self.configure(fg_color=BG)
        self.resizable(True, True)

        self.img_bgr       = None
        self.img_path      = None
        self.folder_images = []
        self.folder_index  = 0
        # image vide réutilisable — évite de passer image=None à CTkLabel
        self._blank        = _make_blank_ctk(IMG_SIZE)
        self._ctk_orig     = self._blank
        self._ctk_res      = self._blank
        self._last_res     = None   # numpy BGR de la dernière détection

        self._build()

        # Chargement automatique du dossier de validation par défaut
        self.after(100, self._load_default_folder)

    # ─── Construction ──────────────────────────

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    # ── Sidebar ────────────────────────────────

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=230, corner_radius=0,
                          fg_color=SIDEBAR, border_width=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(20, weight=1)   # espace flexible avant compteur

        row = 0

        # ── Logo ──
        logo_frame = ctk.CTkFrame(sb, fg_color="transparent")
        logo_frame.grid(row=row, column=0, pady=(28, 4), padx=20, sticky="ew"); row += 1

        # Cercles concentriques via Canvas CTk
        from tkinter import Canvas
        logo_c = Canvas(logo_frame, width=48, height=48,
                        bg=SIDEBAR, highlightthickness=0)
        logo_c.pack()
        logo_c.create_oval(3, 3, 45, 45, fill=ACCENT,   outline="")
        logo_c.create_oval(13, 13, 35, 35, fill=SIDEBAR, outline="")
        logo_c.create_oval(19, 19, 29, 29, fill=ACCENT,  outline="")

        ctk.CTkLabel(sb, text="CoinVision",
                     font=ctk.CTkFont("Segoe UI", 16, "bold"),
                     text_color=TEXT_DARK, fg_color="transparent"
                     ).grid(row=row, column=0, pady=(2, 0)); row += 1

        ctk.CTkLabel(sb, text="Hough Circle Detection",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=SUB_DARK, fg_color="transparent"
                     ).grid(row=row, column=0, pady=(0, 12)); row += 1

        # ── Séparateur ──
        ctk.CTkFrame(sb, height=1, fg_color=SEP
                     ).grid(row=row, column=0, sticky="ew", padx=20, pady=8); row += 1

        # ── Boutons principaux ──
        self._btn(sb, "📷  Charger image",    self.load_image,  BTN_GRAY
                  ).grid(row=row, column=0, padx=16, pady=4, sticky="ew"); row += 1

        self._btn(sb, "🔍  Lancer détection", self.detect,      BTN_GRAY
                  ).grid(row=row, column=0, padx=16, pady=4, sticky="ew"); row += 1

        ctk.CTkFrame(sb, height=1, fg_color=SEP
                     ).grid(row=row, column=0, sticky="ew", padx=20, pady=8); row += 1

        self._btn(sb, "📁  Charger dossier",  self.load_folder, BTN_GRAY
                  ).grid(row=row, column=0, padx=16, pady=4, sticky="ew"); row += 1

        # ── Navigation dossier ──
        self.nav_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.nav_frame.grid(row=row, column=0, padx=16, pady=2, sticky="ew"); row += 1
        self.nav_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_prev = ctk.CTkButton(
            self.nav_frame, text="◀", width=70, height=32,
            command=self.prev_image,
            fg_color=CARD2, hover_color=BTN_GRAY_H,
            text_color=TEXT_DARK, corner_radius=8, font=ctk.CTkFont("Segoe UI", 11))
        self.btn_next = ctk.CTkButton(
            self.nav_frame, text="▶", width=70, height=32,
            command=self.next_image,
            fg_color=CARD2, hover_color=BTN_GRAY_H,
            text_color=TEXT_DARK, corner_radius=8, font=ctk.CTkFont("Segoe UI", 11))
        self.lbl_nav = ctk.CTkLabel(
            self.nav_frame, text="",
            font=ctk.CTkFont("Segoe UI", 11), text_color=MUTED_DARK,
            fg_color="transparent")

        # masqués au départ
        ctk.CTkFrame(sb, height=1, fg_color=SEP
                     ).grid(row=row, column=0, sticky="ew", padx=20, pady=(4, 2)); row += 1

        # ── Section Résultats (compteur + grille 2×2) ──
        res_frame = ctk.CTkFrame(sb, fg_color=CARD2, corner_radius=12)
        res_frame.grid(row=row, column=0, padx=14, pady=(2, 8), sticky="ew"); row += 1
        res_frame.grid_columnconfigure((0, 1), weight=1)

        # En-tête : label à gauche, grand compteur à droite
        ctk.CTkLabel(res_frame, text="RÉSULTATS",
                     font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color=SUB_DARK, fg_color="transparent"
                     ).grid(row=0, column=0, padx=(12, 0), pady=(10, 2), sticky="w")

        count_box = ctk.CTkFrame(res_frame, fg_color="transparent")
        count_box.grid(row=0, column=1, padx=(0, 12), pady=(8, 2), sticky="e")

        self.lbl_count = ctk.CTkLabel(
            count_box, text="",
            font=ctk.CTkFont("Segoe UI", 30, "bold"),
            text_color=SUCCESS, fg_color="transparent")
        self.lbl_count.pack(side="left")

        self.lbl_count_label = ctk.CTkLabel(
            count_box, text="",
            font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED_DARK, fg_color="transparent",
            wraplength=38, justify="left")
        self.lbl_count_label.pack(side="left", padx=(4, 0))

        # Séparateur interne
        ctk.CTkFrame(res_frame, height=1, fg_color=SEP
                     ).grid(row=1, column=0, columnspan=2,
                            sticky="ew", padx=10, pady=0)

        # Grille stats : 3 cellules sur une ligne
        self.stat_detected     = self._stat_cell(res_frame, "Détecté",  2, 0)
        self.stat_ground_truth = self._stat_cell(res_frame, "Réel (GT)",2, 1)
        self.stat_error        = self._stat_cell(res_frame, "Erreur",   2, 2)
        res_frame.grid_columnconfigure(2, weight=1)

        # espace flexible
        sb.grid_rowconfigure(row, weight=1); row += 1

    # ── Zone principale ────────────────────────

    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0,
                            fg_color=BG, border_width=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # Top bar
        top = ctk.CTkFrame(main, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=1,
                 padx=28, pady=(22, 10), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        self.lbl_titre = ctk.CTkLabel(
            top, text="Visualisation",
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
            text_color=TEXT_LIGHT, fg_color="transparent")
        self.lbl_titre.grid(row=0, column=0, sticky="w")

        # Badge statut
        status_frame = ctk.CTkFrame(top, fg_color="transparent")
        status_frame.grid(row=0, column=1, sticky="e")

        self.dot_lbl = ctk.CTkLabel(
            status_frame, text="●", width=16,
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=MUTED_LIGHT, fg_color="transparent")
        self.dot_lbl.grid(row=0, column=0, padx=(0, 4))

        self.lbl_statut = ctk.CTkLabel(
            status_frame, text="Prêt",
            font=ctk.CTkFont("Segoe UI", 9),
            text_color=MUTED_LIGHT, fg_color="transparent")
        self.lbl_statut.grid(row=0, column=1)

        # Carte détection uniquement — pleine largeur
        self.lbl_orig = None
        self.lbl_res  = self._img_card(main, "DÉTECTION", 0)

    # ─── Widgets helpers ───────────────────────

    def _btn(self, parent, text, cmd, color):
        return ctk.CTkButton(
            parent, text=text, command=cmd,
            fg_color=color, hover_color=BTN_GRAY_H,
            text_color=TEXT_DARK, corner_radius=10,
            height=40, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            anchor="w")

    def _stat_cell(self, parent, label, row, col):
        """Cellule compacte pour la grille 2×2 des stats."""
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        cell.grid(row=row, column=col, padx=(10, 6), pady=(4, 8), sticky="ew")

        ctk.CTkLabel(cell, text=label,
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=MUTED_DARK, fg_color="transparent"
                     ).pack(anchor="w")

        val = ctk.CTkLabel(cell, text="—",
                           font=ctk.CTkFont("Segoe UI", 14, "bold"),
                           text_color=ACCENT2, fg_color="transparent")
        val.pack(anchor="w")
        return val

    def _stat_row(self, parent, label, row):
        frame = ctk.CTkFrame(parent, fg_color=CARD2, corner_radius=8)
        frame.grid(row=row, column=0, padx=16, pady=3, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=label,
                     font=ctk.CTkFont("Segoe UI", 9),
                     text_color=SUB_DARK, fg_color="transparent"
                     ).grid(row=0, column=0, padx=(10, 4), pady=6, sticky="w")

        val = ctk.CTkLabel(frame, text="—",
                           font=ctk.CTkFont("Segoe UI", 9, "bold"),
                           text_color=ACCENT2, fg_color="transparent")
        val.grid(row=0, column=1, padx=(0, 10), pady=6, sticky="e")
        return val

    def _img_card(self, parent, titre, col):
        card = ctk.CTkFrame(parent, corner_radius=14,
                            fg_color=CARD, border_width=1,
                            border_color=BORDER)
        card.grid(row=1, column=col, columnspan=1,
                  padx=20, pady=(0, 20), sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=titre,
                     font=ctk.CTkFont("Segoe UI", 9, "bold"),
                     text_color=SUB_LIGHT, fg_color="transparent"
                     ).grid(row=0, column=0, pady=(12, 4))

        lbl = ctk.CTkLabel(card, text="Aucune image\n\n📂",
                           font=ctk.CTkFont("Segoe UI", 10),
                           text_color=MUTED_LIGHT,
                           fg_color=BG,
                           corner_radius=8)
        lbl.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        # Redimensionne l'image quand la carte change de taille
        card.bind("<Configure>", lambda e: self._on_card_resize(e, lbl))
        return lbl

    def _on_card_resize(self, event, lbl):
        """Redimensionne l'image affichée pour remplir le conteneur."""
        w = event.width  - 24   # padding gauche+droite
        h = event.height - 52   # titre + padding haut+bas
        if w < 10 or h < 10:
            return
        # Redessine seulement si une détection a eu lieu
        if self._last_res is not None:
            # Adapte l'image à la taille réelle de la carte (en gardant le ratio)
            img_h, img_w = self._last_res.shape[:2]
            sc   = min(w / img_w, h / img_h)
            nw   = max(1, int(img_w * sc))
            nh   = max(1, int(img_h * sc))
            pil  = Image.fromarray(cv2.cvtColor(
                cv2.resize(self._last_res, (nw, nh)), cv2.COLOR_BGR2RGB))
            self._ctk_res = ctk.CTkImage(
                light_image=pil, dark_image=pil, size=(nw, nh))
            lbl.configure(image=self._ctk_res, text="", width=w, height=h)
        else:
            lbl.configure(width=w, height=h)

    # ─── Statut ────────────────────────────────

    def _set_status(self, msg, color):
        self.lbl_statut.configure(text=msg, text_color=color)
        self.dot_lbl.configure(text_color=color)

    def _reset_stats(self):
        for w in (self.stat_detected, self.stat_ground_truth, self.stat_error):
            w.configure(text="—", text_color=ACCENT2)

    # ─── Chargement image ──────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        self.folder_images = []
        self._hide_nav()
        self._afficher_image(path)

    def _afficher_image(self, path):
        try:
            with open(path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            self._set_status("Erreur : impossible de lire l'image", DANGER)
            return

        self.img_bgr   = img
        self.img_path  = path
        self._last_res = None   # efface l'ancienne détection

        self._ctk_orig = bgr_to_ctk(self.img_bgr)

        # Réinitialise la carte résultat avec l'image vide
        self._ctk_res = self._blank
        self.lbl_res.configure(image=self._ctk_res,
                               text="Aucune image\n\n🔍",
                               width=IMG_SIZE, height=IMG_SIZE)

        self.lbl_count.configure(text="")
        self.lbl_count_label.configure(text="")
        self._reset_stats()

        name = os.path.basename(path)
        self.lbl_titre.configure(text=name)
        self._set_status(f"{name} chargée", ACCENT2)

    # ─── Chargement dossier ────────────────────

    def _load_default_folder(self):
        """Charge le dossier base_validation/images au démarrage si il existe."""
        folder = DEFAULT_IMAGES_DIR
        if not os.path.isdir(folder):
            return
        def _tri_num(p):
            base   = os.path.splitext(os.path.basename(p))[0]
            digits = ''.join(filter(str.isdigit, base))
            return int(digits) if digits else 0

        images = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ], key=_tri_num)
        if not images:
            return
        self.folder_images = images
        self.folder_index  = 0
        self._show_nav()
        self._charger_image_dossier()

    def load_folder(self):
        folder = filedialog.askdirectory(title="Choisir un dossier d'images")
        if not folder:
            return

        def _tri_num(p):
            base   = os.path.splitext(os.path.basename(p))[0]
            digits = ''.join(filter(str.isdigit, base))
            return int(digits) if digits else 0

        images = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ], key=_tri_num)
        if not images:
            self._set_status("Aucune image trouvée dans ce dossier", DANGER)
            return
        self.folder_images = images
        self.folder_index  = 0
        self._show_nav()
        self._charger_image_dossier()

    def _charger_image_dossier(self):
        self._afficher_image(self.folder_images[self.folder_index])
        total = len(self.folder_images)
        self.lbl_nav.configure(
            text=f"Image {self.folder_index + 1} / {total}")
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

    def _show_nav(self):
        self.btn_prev.grid(row=0, column=0, padx=(0, 4), pady=4, sticky="ew")
        self.btn_next.grid(row=0, column=1, padx=(4, 0), pady=4, sticky="ew")
        self.lbl_nav.grid(row=1, column=0, columnspan=2, pady=(2, 0))

    def _hide_nav(self):
        self.btn_prev.grid_remove()
        self.btn_next.grid_remove()
        self.lbl_nav.grid_remove()

    # ─── Détection ────────────────────────────

    def detect(self):
        if self.img_bgr is None:
            self._set_status("Chargez d'abord une image", DANGER)
            return
        self._set_status("Détection en cours...", WARNING)
        self.update()

        res, nb = detecter(self.img_bgr)

        self._last_res = res   # gardé pour _on_card_resize
        # Calcule la taille disponible dans la carte courante
        card_w = self.lbl_res.winfo_width()
        card_h = self.lbl_res.winfo_height()
        size = max(min(card_w, card_h), IMG_SIZE)   # au moins IMG_SIZE
        self._ctk_res = bgr_to_ctk(res, size)
        self.lbl_res.configure(image=self._ctk_res, text="",
                               width=card_w if card_w > 10 else IMG_SIZE,
                               height=card_h if card_h > 10 else IMG_SIZE)

        if nb == 0:
            self.lbl_count.configure(text="0", text_color=DANGER)
            self.lbl_count_label.configure(text="pièce détectée")
            self._set_status("Aucune pièce trouvée", DANGER)
        else:
            self.lbl_count.configure(text=str(nb), text_color=SUCCESS)
            self.lbl_count_label.configure(
                text="pièce détectée" if nb == 1 else "pièces détectées")
            self._set_status(f"{nb} pièce(s) détectée(s)", SUCCESS)

        self.stat_detected.configure(text=str(nb), text_color=ACCENT2)

        nb_reel = charger_verite_terrain(self.img_path) if self.img_path else None
        if nb_reel is not None:
            self.stat_ground_truth.configure(text=str(nb_reel), text_color=ACCENT2)
            erreur = nb - nb_reel
            if erreur == 0:
                self.stat_error.configure(text="0 ✓", text_color=SUCCESS)
            elif erreur > 0:
                self.stat_error.configure(text=f"+{erreur}", text_color=WARNING)
            else:
                self.stat_error.configure(text=str(erreur), text_color=DANGER)
        else:
            for w in (self.stat_ground_truth, self.stat_error):
                w.configure(text="N/A", text_color=MUTED_DARK)


if __name__ == "__main__":
    app = App()
    app.mainloop()
