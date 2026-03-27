import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time, json, os, re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

from funciones import (obtener_comunas, dist, obtener_region, resultados_sampleo, ensure_dir,
                       safe_attr, parse_x_name, extraer_y_guardar_modelo, matriz_X_desde_modelo,
                       promedio_X, comparar_con_baseline, build_matrices_from_gurobi, delta_b_from_eps)

from DataChile.chile_data import regiones

from modelos import modelo_con_limite, modelo_centros_fijos_con_limite, modelo_relajado, modelo_relajado_2, modelo_sin_limite


from sampleos import systematic_sampling, pivotal_sampling, sampford_sampling

from funciones import extraer_prob_centros

from itertools import combinations
from tqdm import tqdm
import time

comunas = pd.read_excel('DataChile/comunas.xlsx')
distancias = pd.read_excel('DataChile/distancias.xlsx')


# R ES UNA LISTA DE COMUNAS, LAS COMUNAS QUE SERÁN PARTE SE DEFINEN A TRAVÉS DE SI 
# LA REGIÓN A LA QUE PERTENECE ESTARÁ INCLUIDA O NO
R = sum([obtener_comunas(comunas, region) for region in regiones], [])

# SE CREA dict_ady_com. ES UN DICCIONARIO DONDE ESTA CADA UNA DE LAS COMUNAS CON SUS ADYACENCIAS
with open('DataChile/adyacencia_comunas.txt', 'r') as dict_file:
    text_dicc = dict_file.read()
    dict_ady_com = eval(text_dicc)

# SE CREA EL GRAFO DE ADYACENCIA A PARTIR DEL DICCIONARIO
grafo_adyacencia = nx.Graph()
id_local = []
for comuna, adyacencia in dict_ady_com.items():
  id_local.append(comuna)
  for comuna_ady in adyacencia:
    grafo_adyacencia.add_edge(comuna,comuna_ady, weight=dist(distancias, comuna,comuna_ady))

# SE TRABAJA EN UNA MATRIZ CON UN 1 SI LAS COMUNAS SON ADYACENTES, 0 SI NO.
matrix_adyacencia = nx.adjacency_matrix(grafo_adyacencia)
m_ady=np.matrix(matrix_adyacencia.toarray())
matriz_ady=np.matrix.round(m_ady)


# SE CARGA EL PREPROCESAMIENTO DE S COMO DICCIONARIO,
# DE FORMA QUE DESPUÉS SOLO SE DEBA ACCEDER A LOS VALORES Y NO GENERARLOS EN CADA ITERACIÓN
with open('DataChile/s_nuevo.txt', 'r') as dict_file:
    dict_text = dict_file.read()
    dict_s = eval(dict_text)

# SE CREA UN DICCIONARIO PARA SABER LAS COMUNAS DE CADA UNA DE LAS REGIONES.
comunas_por_region = {}
# Iterar sobre la lista de comunas y asignarlas a las regiones correspondientes
for comuna in R:
    region = obtener_region(comunas, comuna)
    # Verificar si la región ya está en el diccionario, si no, agregarla
    if region not in comunas_por_region:
        comunas_por_region[region] = []
    # Agregar la comuna a la lista de comunas de la región
    comunas_por_region[region].append(comuna)

modelo_con_limite = modelo_con_limite(0.8398, R, 28, dict_s, comunas)

count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(modelo_con_limite, 28)

def centros_fijados(modelo):
    centros = []
    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and v.x == 1.0:
            texto_1 = v.VarName
            comuna = texto_1[texto_1.find('[')+1 : texto_1.find(']')]
            centros.append(comuna)
    return centros

centros_fijados = centros_fijados(modelo_con_limite)

t_mapas = 8000

comunas_t, probabilidades = zip(*centros_frac)
comunas_t = list(comunas_t)
probabilidades = list(probabilidades)


TIEMPO_MAX = 20 * 60 * 60   # 20 horas en segundos
t_inicio = time.time()

modelos_factibles = []
centros_factibles = []

centros_i_unicos = list(combinations(comunas_t, 6))
print(f"Total de combinaciones a evaluar: {len(centros_i_unicos)}")

epsilon_1 = 0.9

for centros_i in tqdm(centros_i_unicos, desc="Buscando modelos factibles"):

    # chequeo de tiempo
    tiempo_transcurrido = time.time() - t_inicio
    if tiempo_transcurrido >= TIEMPO_MAX:
        print(" Límite de 3 horas alcanzado.")
        break

    centros_total_i = centros_fijados + list(centros_i)

    modelo_i = modelo_centros_fijos_con_limite(
        epsilon_1, R, centros_total_i, dict_s, comunas, verbose=False
    )

    # almacenar solo si es factible
    if modelo_i:
        modelos_factibles.append(modelo_i)
        centros_factibles.append(centros_total_i)
        print(f'Factibles encontrados hasta ahora: {len(modelos_factibles)}')

tiempo_total = time.time() - t_inicio
horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("Cantidad de modelos factibles encontrados:", len(modelos_factibles))
print(f"Tiempo total de ejecución: {horas}h {minutos}m {segundos}s")

# guardar centros factibles en .txt
ruta_salida = f"centros_factibles_epsilon_{epsilon_1}.txt"

with open(ruta_salida, "w", encoding="utf-8") as f:
    for centros in centros_factibles:
        f.write(",".join(centros) + "\n")

print(f"Centros guardados en {ruta_salida}")