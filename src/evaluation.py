from src.utiles import load_all_labels

def evaluate_regression(predictions, labels):

    faux_negatifs = sum(1 for pred, label in zip(predictions.values(), labels.values()) if pred == 0 and label > 0)
    faux_positifs = sum(1 for pred, label in zip(predictions.values(), labels.values()) if pred > 0 and label == 0)
    vrai_positifs = sum(1 for pred, label in zip(predictions.values(), labels.values()) if pred > 0 and label > 0)
    vrai_negatifs = sum(1 for pred, label in zip(predictions.values(), labels.values()) if pred == 0 and label == 0)

    mse = 0
    nb = len(predictions)
    taux_succes = 0
    for i in range(1, nb + 1):
        pred = predictions.get(f"image{i}", 0)
        label = labels.get(f"image{i}.json", 0)

        if pred == label:
            taux_succes += 1
        
        mse += (pred - label) ** 2

    mse /= nb if nb > 0 else 1
    taux_succes = taux_succes / nb if nb > 0 else 1

    print("***************** statistiques d'évaluation de la regression*****************")
    print(f"Faux négatifs : {faux_negatifs}")
    print(f"Faux positifs : {faux_positifs}")
    print(f"Vrais positifs : {vrai_positifs}")
    print(f"Vrais négatifs : {vrai_negatifs}")
    print(f"taux de succes pour la detection:{(nb - faux_negatifs) / nb * 100:.2f}%")
    print(f"Mesure de taux d'erreur MSE pour la regression : {mse:.2f}")
    print(f"Taux de succès : {taux_succes*100:.2f}%")



