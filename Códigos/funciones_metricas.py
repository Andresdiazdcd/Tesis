import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def ari(labels1, labels2):
    return adjusted_rand_score(labels1, labels2)

def nmi(labels1, labels2):
    return normalized_mutual_info_score(labels1, labels2, average_method='arithmetic')

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    return -sum((c/n) * np.log(c/n) for c in counts.values())


def mutual_info(labels1, labels2):
    n = len(labels1)
    joint = Counter(zip(labels1, labels2))
    c1 = Counter(labels1)
    c2 = Counter(labels2)

    mi = 0
    for (i, j), count in joint.items():
        p_ij = count / n
        p_i = c1[i] / n
        p_j = c2[j] / n
        mi += p_ij * np.log(p_ij / (p_i * p_j))

    return mi


def variation_of_information(labels1, labels2):
    H1 = entropy(labels1)
    H2 = entropy(labels2)
    MI = mutual_info(labels1, labels2)
    return H1 + H2 - 2 * MI

def matriz_metricas(mapas, metrica="ari"):
    nombres = sorted(mapas.keys())
    n = len(nombres)
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            labels_i = mapas[nombres[i]]
            labels_j = mapas[nombres[j]]

            if metrica == "ari":
                M[i, j] = adjusted_rand_score(labels_i, labels_j)
            elif metrica == "nmi":
                M[i, j] = normalized_mutual_info_score(labels_i, labels_j, average_method="arithmetic")
            elif metrica == "vi":
                M[i, j] = variation_of_information(labels_i, labels_j)

    return pd.DataFrame(M, index=nombres, columns=nombres)