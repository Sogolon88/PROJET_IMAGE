"""
classification.py

Ce module identifie chaque piece euro detectee et calcule la somme totale.
Il prend en entree les cercles trouves par Hough et retourne une denomination
(1ct, 2ct ... 2e) pour chacun, en deux etapes :

1. Groupement par couleur via deux Otsu successifs.
   On extrait deux features depuis l'image originale (pas celle traitee par
   le pipeline de detection qui a ete floutee et modifiee).
   - diff_b : ecart de b* entre le centre et l'anneau de la piece en Lab.
     Les pieces bimetalliques (1e, 2e) melent deux metaux de teintes
     differentes, ce qui produit un ecart centre/anneau eleve. Les pieces
     monocolores restent uniformes quelle que soit la luminosite.
   - mean_a : moyenne du canal a* sur la zone globale en Lab.
     Le cuivre est orange-rouge (a* eleve), l'or est jaune (a* modere).
     On prefere Lab a HSV parce que a* reste stable meme sur les metaux
     peu satures, la ou la teinte HSV devient instable.

2. Vote par ratios de rayons.
   Les dimensions reelles des pieces euro sont connues en mm. Le rapport
   entre deux rayons detectes doit correspondre au rapport reel. On parcourt
   toutes les paires et on vote pour la denomination la plus coherente.
"""

import cv2
import numpy as np

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

# Candidats par groupe de couleur.
GROUPES = {
    "cuivre":       [("1ct",  8.125), ("2ct",  9.375), ("5ct",  10.625)],
    "or":           [("10ct", 9.875), ("20ct", 11.125), ("50ct", 12.125)],
    "bimetallique": [("1e",   11.625), ("2e",  12.875)],
}

# Denomination retenue si le vote echoue completement.
FALLBACK_GROUPE = {
    "cuivre":       "5ct",
    "or":           "50ct",
    "bimetallique": "1e",
}

# Seuil absolu de diff_b pour classifier une piece seule en bimetallique.
SEUIL_BIMETAL_ABS = 4.0

# Ecart minimum entre les diff_b pour que l'Otsu soit utile (n > 1).
SEUIL_MIN_DIFFB = 3.0

# Ecart minimum entre les mean_a pour que l'Otsu soit utile.
SEUIL_MIN_A = 3.0

# Seuil fixe de a* utilise quand l'Otsu sur mean_a n'est pas fiable.
SEUIL_A_CUIVRE = 136.0


def _otsu_1d(valeurs):
    """
    Trouve le seuil qui separe le mieux une liste de valeurs en deux groupes.

    C'est une adaptation de la methode d'Otsu en 1D : on cherche le seuil t
    qui maximise la variance inter-classe, c'est-a-dire l'ecart entre les
    moyennes des deux groupes pondere par leurs tailles. Plus les deux groupes
    sont distincts, plus la variance inter-classe est elevee.
    """
    vals = np.sort(np.array(valeurs, dtype=float))
    n = len(vals)
    if n < 2 or vals.max() == vals.min():
        return None
    mu_total = vals.mean()
    best_var, best_seuil = -1.0, None
    for i in range(1, n):
        g1, g2 = vals[:i], vals[i:]
        w1, w2 = len(g1) / n, len(g2) / n
        var_inter = w1 * (g1.mean() - mu_total)**2 + w2 * (g2.mean() - mu_total)**2
        if var_inter > best_var:
            best_var, best_seuil = var_inter, (vals[i-1] + vals[i]) / 2.0
    return best_seuil


def _extraire_features(img_bgr, cx, cy, r):
    """
    Calcule diff_b et mean_a pour une piece depuis l'image originale.

    On travaille sur trois zones concentriques definies par le rayon r :
    - centre : disque de rayon 0.50*r
    - anneau : couronne entre 0.50*r et 0.85*r
    - globale : disque de rayon 0.75*r

    Les pixels trop sombres (L < 50) sont ignores pour eviter que les bords
    et les ombres ne biaisent les valeurs de couleur.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    m_centre = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(m_centre, (int(cx), int(cy)), max(1, int(r * 0.50)), 255, -1)

    m_anneau = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(m_anneau, (int(cx), int(cy)), max(1, int(r * 0.85)), 255, -1)
    cv2.circle(m_anneau, (int(cx), int(cy)), max(1, int(r * 0.50)),   0, -1)

    m_glob = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(m_glob, (int(cx), int(cy)), max(1, int(r * 0.75)), 255, -1)

    def mean_b_vis(masque):
        px = lab[masque > 0]
        if len(px) < 5:
            return None
        L = px[:, 0].astype(float)
        b = px[:, 2].astype(float)
        m = L > 50
        return float(b[m].mean()) if m.sum() > 0 else None

    mb_c   = mean_b_vis(m_centre)
    mb_a   = mean_b_vis(m_anneau)
    diff_b = abs(mb_a - mb_c) if (mb_c is not None and mb_a is not None) else 0.0

    px_g = lab[m_glob > 0]
    if len(px_g) == 0:
        return diff_b, float(SEUIL_A_CUIVRE)
    L_g = px_g[:, 0].astype(float)
    a_g = px_g[:, 1].astype(float)
    mask_vis = L_g > 50
    mean_a = float(a_g[mask_vis].mean()) if mask_vis.sum() > 0 else float(SEUIL_A_CUIVRE)

    return diff_b, mean_a


def _grouper(cercles, img_bgr):
    """
    Assigne chaque piece a un groupe cuivre / or / bimetallique.

    Premier Otsu sur diff_b : si l'ecart entre les valeurs est suffisant,
    les pieces avec le diff_b le plus eleve sont classees bimetalliques.
    Si l'ecart est trop faible, toutes les pieces sont monocolores et on
    passe directement a l'etape suivante.

    Deuxieme Otsu sur mean_a : sur les pieces restantes, on separe cuivre
    et or. Si les valeurs sont trop proches, on bascule sur le seuil fixe.

    Pour n=1 (piece seule), l'Otsu n'a pas de sens donc on compare
    directement aux seuils absolus.
    """
    n = len(cercles)
    features    = [_extraire_features(img_bgr, cx, cy, r) for cx, cy, r in cercles]
    diff_b_vals = [f[0] for f in features]
    mean_a_vals = [f[1] for f in features]
    groupes = {}

    if n == 1:
        diff_b, mean_a = features[0]
        if diff_b > SEUIL_BIMETAL_ABS:
            groupes[0] = "bimetallique"
        elif mean_a > SEUIL_A_CUIVRE:
            groupes[0] = "cuivre"
        else:
            groupes[0] = "or"
        return groupes

    # etape 1 : detection des bimetalliques via diff_b
    plage_db = max(diff_b_vals) - min(diff_b_vals)
    idx_bimetal, idx_autres = [], []

    if plage_db >= SEUIL_MIN_DIFFB:
        seuil_db = _otsu_1d(diff_b_vals)
        if seuil_db is not None and seuil_db > SEUIL_MIN_DIFFB:
            for i, db in enumerate(diff_b_vals):
                if db > seuil_db:
                    idx_bimetal.append(i)
                    groupes[i] = "bimetallique"
                else:
                    idx_autres.append(i)
        else:
            idx_autres = list(range(n))
    else:
        idx_autres = list(range(n))

    # etape 2 : separation cuivre / or via mean_a
    if len(idx_autres) == 0:
        pass
    elif len(idx_autres) == 1:
        i = idx_autres[0]
        groupes[i] = "cuivre" if mean_a_vals[i] > SEUIL_A_CUIVRE else "or"
    else:
        vals_a  = [mean_a_vals[i] for i in idx_autres]
        plage_a = max(vals_a) - min(vals_a)
        seuil_a = _otsu_1d(vals_a) if plage_a >= SEUIL_MIN_A else None
        if seuil_a is None:
            seuil_a = SEUIL_A_CUIVRE
        for i in idx_autres:
            groupes[i] = "cuivre" if mean_a_vals[i] > seuil_a else "or"

    return groupes


def _vote_contraint(cercles, candidats_par_idx, epsilons=(0.03, 0.06, 0.10)):
    """
    Identifie la denomination de chaque piece par comparaison des ratios de rayons.

    Pour chaque paire de pieces (i, j), le rapport r_i / r_j mesure doit
    correspondre au rapport de leurs rayons reels (en mm). On vote pour les
    denominations dont le ratio theorique est proche du ratio mesure, a
    epsilon pres. On commence avec un epsilon strict et on l'elargit
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
                classes_finale[i] = max(votes[i], key=votes[i].get)
        if all(v != "inconnu" for v in classes_finale.values()):
            break

    return classes_finale


def classifier_pieces(cercles, img_bgr, epsilons=(0.02, 0.05, 0.10), max_side=1025):
    """
    Fonction principale de classification.

    Prend les cercles detectes et l'image originale BGR, retourne la liste
    des pieces identifiees avec leur denomination et valeur, ainsi que la
    somme totale en euros.

    On redimensionne l'image a max_side pour etre coherent avec les
    coordonnees des cercles issus de detecter_image(). On travaille sur
    cette image sans aucun traitement supplementaire pour conserver les
    vraies couleurs des pieces.
    """
    if len(cercles) == 0:
        return [], 0.0

    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    groupes           = _grouper(cercles, img_bgr)
    candidats_par_idx = {i: GROUPES[g] for i, g in groupes.items()}
    classes           = _vote_contraint(cercles, candidats_par_idx, epsilons)

    for i in range(len(cercles)):
        if classes.get(i, "inconnu") == "inconnu":
            groupe     = groupes.get(i, "or")
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
