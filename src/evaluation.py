"""
evaluation.py

Fonctions d'evaluation des performances du systeme de detection.

On distingue deux niveaux d'evaluation :
- evaluate_regression : compare le nombre de pieces predit au nombre reel,
  image par image. C'est une metrique de comptage pur.
- evaluate_iou : va plus loin en verifiant que chaque cercle detecte
  correspond spatialement a une piece reelle, via l'IoU circulaire.
"""

import math

MAX_SIDE   = 1025
IOU_THRESH = 0.3


def circle_iou(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'IoU (Intersection over Union) entre deux cercles.

    L'IoU est le rapport entre l'aire de l'intersection et l'aire de l'union.
    Un IoU de 1 signifie superposition parfaite, 0 signifie aucun contact.

    On distingue trois cas geometriques : cercles disjoints (d >= r1+r2,
    IoU=0), l'un contenu dans l'autre (d <= |r1-r2|, on retourne le rapport
    des surfaces), ou chevauchement partiel (formule analytique combinant deux
    secteurs circulaires moins le triangle qu'ils forment).
    """
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        small_r = min(r1, r2)
        large_r = max(r1, r2)
        return (small_r ** 2) / (large_r ** 2)
    a1 = r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    a3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection = a1 + a2 - a3
    union = math.pi * r1 ** 2 + math.pi * r2 ** 2 - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_iou(predictions_circles, gt_boxes, iou_thresh=IOU_THRESH):
    """
    Evalue la detection en comparant spatialement chaque cercle predit a la verite terrain.

    Pour chaque image, on remet a l'echelle les cercles de reference (annotes
    dans l'espace de l'image originale) vers l'espace redimensionne a MAX_SIDE
    utilise par le detecteur. On associe ensuite chaque prediction au cercle de
    reference le plus proche en IoU, si cet IoU depasse iou_thresh.

    Un TP est une prediction qui correspond a une piece reelle. Un FP est une
    detection sans correspondance. Un FN est une piece reelle non detectee.
    Le mIoU est la moyenne des IoU des vrais positifs : il mesure non seulement
    si on trouve les pieces, mais aussi si les cercles sont bien places.
    """
    tp_iou   = 0
    fp_iou   = 0
    fn_iou   = 0
    iou_list = []

    for key, pred_circles in predictions_circles.items():
        if key not in gt_boxes:
            fp_iou += len(pred_circles)
            continue
        info  = gt_boxes[key]
        scale = min(1.0, MAX_SIDE / max(info["img_h"], info["img_w"]))

        gt_circles_scaled = [
            (cx * scale, cy * scale, r * scale)
            for (cx, cy, r) in info["circles"]
        ]

        matched_gt = set()
        for (pcx, pcy, pr) in pred_circles:
            best_iou, best_idx = 0.0, -1
            for i, (gcx, gcy, gr) in enumerate(gt_circles_scaled):
                if i in matched_gt:
                    continue
                iou = circle_iou(pcx, pcy, pr, gcx, gcy, gr)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_thresh and best_idx >= 0:
                tp_iou += 1
                matched_gt.add(best_idx)
                iou_list.append(best_iou)
            else:
                fp_iou += 1

        fn_iou += len(gt_circles_scaled) - len(matched_gt)

    rappel_iou    = tp_iou / (tp_iou + fn_iou) if (tp_iou + fn_iou) > 0 else 0
    precision_iou = tp_iou / (tp_iou + fp_iou) if (tp_iou + fp_iou) > 0 else 0
    f1_iou        = (2 * precision_iou * rappel_iou / (precision_iou + rappel_iou)
                     if (precision_iou + rappel_iou) > 0 else 0)
    mean_iou      = sum(iou_list) / len(iou_list) if iou_list else 0.0

    print(f"\n  Metriques IoU (seuil={iou_thresh})")
    print(f"  TP-IoU : {tp_iou}  |  FP-IoU : {fp_iou}  |  FN-IoU : {fn_iou}")
    print(f"  Rappel-IoU    : {rappel_iou * 100:.2f}%")
    print(f"  Precision-IoU : {precision_iou * 100:.2f}%")
    print(f"  F1-IoU        : {f1_iou * 100:.2f}%")
    print(f"  mIoU (moyenne IoU des TP) : {mean_iou:.4f}")
    print("=" * 60)

    return {
        "tp_iou": tp_iou, "fp_iou": fp_iou, "fn_iou": fn_iou,
        "rappel_iou": rappel_iou, "precision_iou": precision_iou,
        "f1_iou": f1_iou, "mean_iou": mean_iou,
    }


def evaluate_regression(predictions, labels):
    """
    Evalue le systeme sur le seul comptage : combien de pieces y a-t-il ?

    Pour chaque image, on compare le nombre predit au nombre reel. On prend
    min(predit, reel) comme TP car on ne peut pas avoir plus de vrais positifs
    que de pieces reelles. L'excedent devient des FP, le manque des FN.

    Le taux de succes exact compte les images ou le nombre predit est juste.
    Cette metrique est plus permissive que l'IoU car elle ne verifie pas la
    localisation des cercles, seulement le total.
    """
    tp_total    = 0
    fp_total    = 0
    fn_total    = 0
    mse         = 0.0
    taux_succes = 0

    for key, pred in predictions.items():
        label = labels.get(f"{key}.json", 0)
        tp_total    += min(pred, label)
        fp_total    += max(0, pred - label)
        fn_total    += max(0, label - pred)
        if pred == label:
            taux_succes += 1
        mse += (pred - label) ** 2

    nb  = len(predictions) if predictions else 1
    mse /= nb
    taux_succes /= nb

    rappel    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    f1        = (2 * precision * rappel / (precision + rappel)
                 if (precision + rappel) > 0 else 0)

    print("=" * 60)
    print("  STATISTIQUES D'EVALUATION - DETECTION")
    print("=" * 60)
    print(f"  Pieces correctement detectees (TP) : {tp_total}")
    print(f"  Pieces detectees en trop       (FP) : {fp_total}")
    print(f"  Pieces manquees                (FN) : {fn_total}")
    print(f"  Rappel    : {rappel * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  F1-score  : {f1 * 100:.2f}%")
    print(f"  MSE       : {mse:.4f}")
    print(f"  Taux de succes (comptage exact) : {taux_succes * 100:.2f}%")
    print("=" * 60)

    return {
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "rappel": rappel, "precision": precision, "f1": f1,
        "mse": mse, "taux_succes": taux_succes,
    }
