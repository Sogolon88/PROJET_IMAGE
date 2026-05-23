"""
classification.py

Ce fichier permet de reconnaitre chaque piece euro detectee dans une image
et de calculer la somme totale.

Le principe general :
  1. On reduit les reflets sur la piece (ils faussent la couleur)
  2. On calcule deux caracteristiques de couleur par piece
  3. On determine si la piece est en cuivre, en or, ou bimetallique
  4. On compare les rayons des pieces entre elles pour voter la denomination
  5. Si le vote ne donne rien, on utilise une valeur par defaut selon le groupe
"""

import cv2
import numpy as np

# dimensions reelles des pieces en euros (rayon en millimetres)
PIECES_EURO = {
    "1ct":  {"valeur": 0.01, "rayon_mm": 8.125},
    "2ct":  {"valeur": 0.02, "rayon_mm": 9.375},
    "5ct":  {"valeur": 0.05, "rayon_mm": 10.625},
    "10ct": {"valeur": 0.10, "rayon_mm": 9.875},
    "20ct": {"valeur": 0.20, "rayon_mm": 11.125},
    "50ct": {"valeur": 0.50, "rayon_mm": 12.125},
    "1e":   {"valeur": 1.00, "rayon_mm": 11.625},
    "2e":   {"valeur": 2.00, "rayon_mm": 12.875},
}

# on regroupe les pieces selon leur couleur visuelle
GROUPES = {
    "cuivre":       [("1ct",  8.125), ("2ct",  9.375), ("5ct",  10.625)],
    "or":           [("10ct", 9.875), ("20ct", 11.125), ("50ct", 12.125)],
    "bimetallique": [("1e",   11.625), ("2e",  12.875)],
}

# valeur par defaut si on n'arrive pas a identifier la piece
FALLBACK_GROUPE = {
    "cuivre":       "5ct",
    "or":           "50ct",
    "bimetallique": "1e",
}

# seuils reglés manuellement apres tests sur la base de validation
SEUIL_BIMETAL_ABSOLU = 0.05   # difference de saturation centre/anneau au dessus de laquelle -> bimetallique
SEUIL_H_ABSOLU       = 17.0   # teinte en dessous de laquelle -> cuivre, au dessus -> or
SEUIL_H_RANGE_MIN    = 8.0    # si les teintes sont trop proches on n'utilise pas Otsu


# ── Otsu 1D ──────────────────────────────────────────────────────────────────

def _otsu_1d(valeurs):
    """
    Trouve le meilleur seuil pour separer une liste de valeurs en deux groupes.
    C'est une version simplifiee de la methode d'Otsu adaptee a une seule dimension.
    """
    vals = np.sort(np.array(valeurs, dtype=float))
    n = len(vals)
    if n < 2 or vals.max() == vals.min():
        return None
    mu_total   = vals.mean()
    best_var   = -1.0
    best_seuil = None
    for i in range(1, n):
        g1, g2 = vals[:i], vals[i:]
        w1, w2 = len(g1) / n, len(g2) / n
        var_inter = w1 * (g1.mean() - mu_total)**2 + w2 * (g2.mean() - mu_total)**2
        if var_inter > best_var:
            best_var   = var_inter
            best_seuil = (vals[i - 1] + vals[i]) / 2.0
    return best_seuil


# ── Reduction des reflets ─────────────────────────────────────────────────────

def _reduire_reflets(img_bgr, seuil_v=230):
    """
    Les zones tres lumineuses sur les pieces (reflets metalliques) faussent
    la detection de couleur. On les remplace par une version floutee de l'image
    pour lisser ces zones sans perdre le reste.
    """
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_cor = img_bgr.copy()
    floue   = cv2.GaussianBlur(img_bgr, (15, 15), 0)
    img_cor[hsv[:, :, 2] > seuil_v] = floue[hsv[:, :, 2] > seuil_v]
    return img_cor


# ── Extraction des caracteristiques couleur ───────────────────────────────────

def _extraire_features(img_cor, cx, cy, r):
    """
    Pour chaque piece on calcule deux valeurs :

    - mean_H : la teinte moyenne des pixels les plus satures (30% les plus vifs).
      On prend les plus satures plutot que tous pour eviter que les zones grises
      ou surexposees ne faussent le resultat.

    - diff_S : la difference de saturation entre le centre et l'anneau de la piece.
      Les pieces bimetalliques (1e, 2e) ont deux metaux differents donc une grande
      difference entre les deux zones. Les pieces normales sont uniformes.
    """
    hsv = cv2.cvtColor(img_cor, cv2.COLOR_BGR2HSV)

    # on definit trois zones circulaires dans la piece
    m_centre = np.zeros(img_cor.shape[:2], dtype=np.uint8)
    cv2.circle(m_centre, (int(cx), int(cy)), max(1, int(r * 0.50)), 255, -1)

    m_anneau = np.zeros(img_cor.shape[:2], dtype=np.uint8)
    cv2.circle(m_anneau, (int(cx), int(cy)), max(1, int(r * 0.85)), 255, -1)
    cv2.circle(m_anneau, (int(cx), int(cy)), max(1, int(r * 0.50)),   0, -1)

    m_glob = np.zeros(img_cor.shape[:2], dtype=np.uint8)
    cv2.circle(m_glob, (int(cx), int(cy)), max(1, int(r * 0.75)), 255, -1)

    px_c = hsv[m_centre > 0]
    px_a = hsv[m_anneau > 0]
    px_g = hsv[m_glob   > 0]

    if len(px_g) == 0:
        return SEUIL_H_ABSOLU + 1, 0.0

    H_g = px_g[:, 0].astype(float)
    S_g = px_g[:, 1].astype(float)
    V_g = px_g[:, 2].astype(float)

    # on ignore les pixels trop sombres (bords, ombres)
    mask_vis = V_g > 50
    if mask_vis.sum() == 0:
        return SEUIL_H_ABSOLU + 1, 0.0

    # teinte moyenne sur les 30% pixels les plus satures
    k       = max(1, int(0.30 * mask_vis.sum()))
    top_idx = np.argsort(S_g[mask_vis])[-k:]
    mean_H  = float(H_g[mask_vis][top_idx].mean())

    # difference de saturation entre centre et anneau
    def _mean_S_vis(pixels):
        if len(pixels) < 5:
            return None
        S = pixels[:, 1].astype(float)
        V = pixels[:, 2].astype(float)
        m = V > 50
        return float(S[m].mean()) if m.sum() > 0 else None

    ms_c   = _mean_S_vis(px_c)
    ms_a   = _mean_S_vis(px_a)
    diff_S = abs(ms_a - ms_c) / 255.0 if (ms_c is not None and ms_a is not None) else 0.0

    return mean_H, diff_S


# ── Groupement par couleur ────────────────────────────────────────────────────

def _grouper(cercles, img_cor):
    """
    Assigne chaque piece a un groupe de couleur : cuivre, or ou bimetallique.

    On utilise d'abord la difference de saturation entre zones pour reperer
    les bimetalliques. Ensuite on utilise la teinte pour separer cuivre et or.

    Si toutes les pieces ont des teintes tres proches, on evite d'utiliser Otsu
    car il va les separer arbitrairement (par exemple deux pieces identiques
    se retrouveraient dans deux groupes differents).
    """
    n        = len(cercles)
    features = [_extraire_features(img_cor, cx, cy, r) for cx, cy, r in cercles]
    groupes  = {}

    # cas particulier : une seule piece, on utilise les seuils directement
    if n == 1:
        mean_H, diff_S = features[0]
        if diff_S > SEUIL_BIMETAL_ABSOLU:
            groupes[0] = "bimetallique"
        elif mean_H < SEUIL_H_ABSOLU:
            groupes[0] = "cuivre"
        else:
            groupes[0] = "or"
        return groupes

    # etape 1 : separation bimetallique / reste avec Otsu sur diff_S
    diff_S_vals   = [f[1] for f in features]
    seuil_bimetal = _otsu_1d(diff_S_vals)
    if seuil_bimetal is None or seuil_bimetal < SEUIL_BIMETAL_ABSOLU:
        seuil_bimetal = SEUIL_BIMETAL_ABSOLU

    idx_bimetal, idx_autres = [], []
    for i, ds in enumerate(diff_S_vals):
        if ds > seuil_bimetal:
            idx_bimetal.append(i)
            groupes[i] = "bimetallique"
        else:
            idx_autres.append(i)

    # etape 2 : separation cuivre / or avec Otsu sur la teinte
    if len(idx_autres) >= 2:
        mean_Hs = [features[i][0] for i in idx_autres]
        h_range = max(mean_Hs) - min(mean_Hs)
        # si les teintes sont trop proches Otsu n'est pas fiable, on prend le seuil fixe
        if h_range < SEUIL_H_RANGE_MIN:
            seuil_H = SEUIL_H_ABSOLU
        else:
            seuil_H = _otsu_1d(mean_Hs) or SEUIL_H_ABSOLU
        for i in idx_autres:
            groupes[i] = "cuivre" if features[i][0] < seuil_H else "or"
    elif len(idx_autres) == 1:
        i = idx_autres[0]
        groupes[i] = "cuivre" if features[i][0] < SEUIL_H_ABSOLU else "or"

    return groupes


# ── Vote par ratios de rayons ─────────────────────────────────────────────────

def _vote_contraint(cercles, candidats_par_idx, epsilons=(0.03, 0.06, 0.10)):
    """
    Pour identifier chaque piece, on compare les rayons detectes entre toutes
    les pieces de l'image. Si le ratio entre deux rayons correspond au ratio
    reel entre deux denominations connues (a epsilon pres), on vote pour ces
    denominations. Plusieurs pieces peuvent avoir la meme denomination.

    On essaie d'abord avec un epsilon faible (plus strict) et on elargit
    progressivement si des pieces restent non identifiees.
    """
    n = len(cercles)
    classes_finale = {i: "inconnu" for i in range(n)}

    for epsilon in epsilons:
        votes = {i: {} for i in range(n)}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                alpha     = cercles[i][2] / cercles[j][2]
                alpha_inv = cercles[j][2] / cercles[i][2]

                for nom_i, r_i in candidats_par_idx[i]:
                    for nom_j, r_j in candidats_par_idx[j]:
                        ratio_theo = r_i / r_j
                        ratio_inv  = r_j / r_i
                        if (abs(alpha - ratio_theo) <= epsilon and
                                abs(alpha_inv - ratio_inv) <= epsilon):
                            votes[i][nom_i] = votes[i].get(nom_i, 0) + 1

        for i in range(n):
            if votes[i] and classes_finale[i] == "inconnu":
                best = max(votes[i], key=votes[i].get)
                classes_finale[i] = best

        if all(v != "inconnu" for v in classes_finale.values()):
            break

    return classes_finale


# ── Fonction principale ───────────────────────────────────────────────────────

def classifier_pieces(cercles, img_bgr, epsilons=(0.02, 0.05, 0.10), max_side=1025):
    """
    Fonction principale : prend les cercles detectes et l'image originale,
    retourne la liste des pieces identifiees avec leur denomination et valeur,
    ainsi que la somme totale.
    """
    if len(cercles) == 0:
        return [], 0.0

    # on redimensionne l'image pour etre coherent avec les coordonnees des cercles
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    img_cor = _reduire_reflets(img_bgr)
    groupes = _grouper(cercles, img_cor)

    candidats_par_idx = {i: GROUPES[g] for i, g in groupes.items()}
    classes = _vote_contraint(cercles, candidats_par_idx, epsilons)

    # pour les pieces toujours non identifiees, on prend la valeur par defaut du groupe
    for i in range(len(cercles)):
        if classes.get(i, "inconnu") == "inconnu":
            groupe = groupes.get(i, "or")
            classes[i] = FALLBACK_GROUPE.get(groupe, "inconnu")

    resultats = []
    for i, (cx, cy, r) in enumerate(cercles):
        nom    = classes.get(i, "inconnu")
        valeur = PIECES_EURO[nom]["valeur"] if nom in PIECES_EURO else None
        resultats.append({
            "cercle": (cx, cy, r),
            "groupe": groupes.get(i, "inconnu"),
            "classe": nom,
            "valeur": valeur,
        })

    somme = round(sum(r["valeur"] for r in resultats if r["valeur"] is not None), 2)
    return resultats, somme
