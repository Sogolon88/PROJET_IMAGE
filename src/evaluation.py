def evaluate_regression(predictions, labels):
    """
    Calcule les métriques de detection a partir des predictions et des vrais labels.

    Pour chaque image on compare le nombre de pieces detecees au nombre reel :
    - TP : pieces correctement detectees (le minimum entre predit et reel)
    - FP : pieces detectees en trop (fausses alarmes)
    - FN : pieces manquees (non detectees)

    On calcule ensuite rappel, precision, F1 et MSE pour avoir une vue globale.
    """
    tp_total    = 0
    fp_total    = 0
    fn_total    = 0
    mse         = 0.0
    taux_succes = 0

    for key, pred in predictions.items():
        label = labels.get(f"{key}.json", 0)

        tp_total += min(pred, label)
        fp_total += max(0, pred - label)
        fn_total += max(0, label - pred)

        if pred == label:
            taux_succes += 1
        mse += (pred - label) ** 2

    nb  = len(predictions) if predictions else 1
    mse /= nb
    taux_succes /= nb

    # rappel : parmi toutes les pieces reelles, combien on en a trouvées
    rappel    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    # precision : parmi toutes les pieces detecees, combien sont vraiment des pieces
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    # F1 : moyenne harmonique entre rappel et precision
    f1        = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) > 0 else 0

    print("=" * 60)
    print("  STATISTIQUES D'EVALUATION — DETECTION")
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
