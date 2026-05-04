from src.utiles import load_all_labels

def evaluate_regression(predictions, labels):

    nb = len(predictions)

    faux_negatifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) == 0
                        and labels.get(f"image{i}.json", 0) > 0)

    faux_positifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) > 0
                        and labels.get(f"image{i}.json", 0) == 0)

    vrai_positifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) > 0
                        and labels.get(f"image{i}.json", 0) > 0)

    vrai_negatifs = sum(1 for i in range(1, nb+1)
                        if predictions.get(f"image{i}", 0) == 0
                        and labels.get(f"image{i}.json", 0) == 0)

    mse = 0
    taux_succes = 0
    for i in range(1, nb + 1):
        pred = predictions.get(f"image{i}", 0)
        label = labels.get(f"image{i}.json", 0)

        if pred == label:
            taux_succes += 1

        mse += (pred - label) ** 2

    mse /= nb if nb > 0 else 1
    taux_succes = taux_succes / nb if nb > 0 else 1

    total_avec_pieces = vrai_positifs + faux_negatifs
    rappel = vrai_positifs / total_avec_pieces if total_avec_pieces > 0 else 0

    stats = {
        "faux_negatifs":  faux_negatifs,
        "faux_positifs":  faux_positifs,
        "vrai_positifs":  vrai_positifs,
        "vrai_negatifs":  vrai_negatifs,
        "rappel":         rappel,
        "mse":            mse,
        "taux_succes":    taux_succes,
    }

    print("***************** statistiques d'évaluation de la regression*****************")
    print(f"Faux négatifs : {faux_negatifs}")
    print(f"Faux positifs : {faux_positifs}")
    print(f"Vrais positifs : {vrai_positifs}")
    print(f"Vrais négatifs : {vrai_negatifs}")
    print(f"Rappel (détection) : {rappel * 100:.2f}%")
    print(f"Mesure de taux d'erreur MSE pour la regression : {mse:.2f}")
    print(f"Taux de succès : {taux_succes*100:.2f}%")

    return stats



