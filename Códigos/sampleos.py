import random
import re
import numpy as np
from itertools import islice, accumulate
import bisect


# elements = lista con nombre de comunas
# probabilities = lista con probabilidades para cada comuna
# los índices de las listas anteriores deben referir a la misma comuna

# k = número de elementos que queremos como output en el sampleo

def systematic_sampling(elements, probabilities, k):

    # Mezclar manteniendo correspondencia
    combined = list(zip(elements, probabilities))
    random.shuffle(combined)
    elements, probabilities = zip(*combined)

    probabilities = [0] + list(probabilities)

    n = len(elements)

    shift = random.uniform(0, 1)

    partial_sum_t = [x + shift for x in list(accumulate(probabilities))]

    sampleo = []

    for i in range(1, k+1):
        for l in range(n):
            if partial_sum_t[l+1] > i and partial_sum_t[l] <= i:
                sampleo.append(elements[l])

    return sampleo

def systematic_sampling_n(n, k, comunas, probabilidades):
    systematic_n = []
    for i in range(n):
        seleccionados = systematic_sampling(comunas, probabilidades, k)
        systematic_n.append(seleccionados)

def pivotal_step(elements, probabilities):
    """
    Aplica una sola iteración del método pivotal.
    Retorna:
        - elemento seleccionado si su probabilidad llega a 1,
        - None si ninguno llega a 1
    """
    n = len(elements)
    if n < 2:
        return None

    # Selección aleatoria de dos índices distintos
    i, j = random.sample(range(n), 2)

    pi_i = probabilities[i]
    pi_j = probabilities[j]

    if pi_i + pi_j > 1:
        lambda_ij = (1 - pi_j) / (2 - pi_i - pi_j)
        if random.random() < lambda_ij:
            probabilities[i] = 1
            probabilities[j] = pi_i + pi_j - 1
        else:
            probabilities[i] = pi_i + pi_j - 1
            probabilities[j] = 1
    else:
        # pi_i + pi_j <= 1:
        lambda_ij = pi_i / (pi_i + pi_j)
        if random.random() < lambda_ij:
            probabilities[i] = pi_i + pi_j
            probabilities[j] = 0
        else:
            probabilities[i] = 0
            probabilities[j] = pi_i + pi_j

    # Revisamos si alguno llegó a 1 (entra al sampleo) o a 0 (se elimina)
    selected_element = None
    for idx in sorted([i, j], reverse=True):  # de mayor a menor para tener cuidado con el .pop()
        if probabilities[idx] == 1:
            selected_element = elements[idx]
            elements.pop(idx)
            probabilities.pop(idx)
        elif probabilities[idx] == 0:
            elements.pop(idx)
            probabilities.pop(idx)

    return selected_element

def pivotal_sampling(elements, probabilities):
    """
    Aplica el método pivotal completo.
    Devuelve la muestra final: elementos con probabilidad 1.
    Como sum_i(pi_i) = k, el output son k elementos
    """
    # Copias para no modificar las originales
    elements_iter = elements.copy()
    probabilities_iter = probabilities.copy()

    muestra = []

    while len(elements_iter) >= 2:
        elegido = pivotal_step(elements_iter, probabilities_iter)
        if elegido is not None:
            muestra.append(elegido)

    # Si queda solo 1 elemento, revisar si su probabilidad es 1
    # si no rendondeamos hay problemas
    # -> los pesos dados por gurobi suman 5.9999999999+
    for i in range(len(elements_iter)):
        if round(probabilities_iter[i], 10) == 1:
            muestra.append(elements_iter[i])

    return muestra

def weighted_choice(elements, weights):
    total = sum(weights)
    u = random.uniform(0, total)

    acumulado = 0
    for e, w in zip(elements, weights):
        acumulado += w
        if u <= acumulado:
            return e

    return elements[-1]

def build_weighted_sampler(weights):
    total = sum(weights)
    cumsum = list(accumulate(weights))
    return cumsum, total

def weighted_choice_precomputed(elements, cumsum, total):
    r = random.random() * total
    idx = bisect.bisect_left(cumsum, r)
    return elements[idx]


def sampford_sampling_base(elements, probabilities, k):

    while True:

        sampleo = []

        # Paso 1: elegir i1 proporcional a p_i
        i1 = weighted_choice(elements, probabilities)
        sampleo.append(i1)

        # Paso 2: pesos p_i / (1 - p_i)
        pesos = []
        for p in probabilities:
            if round(1 - p, 12) == 0:
                pesos.append(float("inf"))
            else:
                pesos.append(p / (1 - p))

        # Seleccionar k-1 con reemplazo
        for _ in range(k - 1):
            elegido = weighted_choice(elements, pesos)
            sampleo.append(elegido)

        # Paso 3: aceptar si todos distintos
        if len(set(sampleo)) == k:
            return sampleo
        

def sampford_sampling(elements, probabilities, k):
    n = len(elements)
    indices = list(range(n))

    pesos = []
    for p in probabilities:
        if round(1 - p, 12) == 0:
            raise ValueError("Hay un p_i = 1; Sampford con esta implementación no lo maneja bien.")
        pesos.append(p / (1 - p))

    cumsum_p, total_p = build_weighted_sampler(probabilities)
    cumsum_w, total_w = build_weighted_sampler(pesos)

    while True:
        usados = set()
        sampleo_idx = []

        i1 = weighted_choice_precomputed(indices, cumsum_p, total_p)
        usados.add(i1)
        sampleo_idx.append(i1)

        valido = True
        for _ in range(k - 1):
            elegido = weighted_choice_precomputed(indices, cumsum_w, total_w)
            if elegido in usados:
                valido = False
                break
            usados.add(elegido)
            sampleo_idx.append(elegido)

        if valido:
            return [elements[i] for i in sampleo_idx]