import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time, json, os, re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import sys
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
from funciones_guardado import guardar_resultado_factible
import math

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

K_centros = 28

TIPO_MODELO = "CL" # CON LIMITE
METODO = "SYS" # Systematic
modelo_con_limite = modelo_con_limite(0.8398, R, K_centros, dict_s, comunas)

count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(modelo_con_limite, 28)

print(f"Centros fijados por modelo: {count_centros_fijados}")

def centros_fijados(modelo):
    centros = []
    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and v.x == 1.0:
            texto_1 = v.VarName
            comuna = texto_1[texto_1.find('[')+1 : texto_1.find(']')]
            centros.append(comuna)
    return centros

centros_fijados = centros_fijados(modelo_con_limite)

print(f"Centros fijados por modelo check 1: {len(centros_fijados)}")

comunas_t, probabilidades = zip(*centros_frac)
comunas_t = list(comunas_t)
probabilidades = list(probabilidades)

HORAS = 20
TIEMPO_MAX = HORAS * 60 * 60   # 20 horas en segundos
t_inicio = time.time()

epsilon_1 = 0.89
t_mapas = 100
k_sampleo = K_centros - len(centros_fijados)


k_sampleo = K_centros - len(centros_fijados)
n = len(comunas_t)

total_combinaciones = math.comb(len(comunas_t), k_sampleo)

print(f"Centros a samplear: {k_sampleo}")
print(f"Meta de mapas factibles a guardar: {t_mapas}")

BASE_RESULTADOS = "resultados"
os.makedirs(BASE_RESULTADOS, exist_ok=True)

base_resultados = os.path.join(
    BASE_RESULTADOS,
    f"resultados_{TIPO_MODELO}_{METODO}_epsilon_{epsilon_1}"
)
os.makedirs(base_resultados, exist_ok=True)

ruta_centros_factibles = os.path.join(base_resultados, "centros_factibles.txt")
ruta_centros_infactibles = os.path.join(base_resultados, "centros_infactibles.txt")
ruta_log = os.path.join(base_resultados, "log_corrida.txt")

centros_factibles = []
centros_infactibles = []

centros_factibles_set = set()
centros_infactibles_set = set()

intentos = 0
repetidos = 0
errores_gurobi = 0
errores_otro_tipo = 0

LOG_CADA = 10

with open(ruta_log, "w", encoding="utf-8") as f_log:
    f_log.write(f"Inicio corrida: {time.ctime()}\n")
    f_log.write(f"TIPO_MODELO = {TIPO_MODELO}\n")
    f_log.write(f"epsilon_1 = {epsilon_1}\n")
    f_log.write(f"K_centros = {K_centros}\n")
    f_log.write(f"centros_fijados = {len(centros_fijados)}\n")
    f_log.write(f"centros_a_samplear = {k_sampleo}\n")
    f_log.write(f"t_mapas_objetivo = {t_mapas}\n\n")

while len(centros_factibles) < t_mapas:

    if len(centros_factibles_set) + len(centros_infactibles_set) == total_combinaciones:
        print("Se exploraron todas las combinaciones posibles.")
        break

    tiempo_transcurrido = time.time() - t_inicio
    if tiempo_transcurrido >= TIEMPO_MAX:
        mensaje = f"Límite de {HORAS} horas alcanzado."
        print(mensaje)
        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")
            f_log.write(f"Límite de tiempo alcanzado en intento {intentos}\n")
        break

    intentos += 1

    try:
        centros_i = systematic_sampling(comunas_t, probabilidades, k_sampleo)
        centros_i_key = tuple(sorted(centros_i))

        # skip si ya fue visto antes, sea factible o infactible
        if centros_i_key in centros_factibles_set or centros_i_key in centros_infactibles_set:
            repetidos += 1

            if intentos % LOG_CADA == 0:
                mensaje = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"intentos={intentos} | "
                    f"factibles={len(centros_factibles)} | "
                    f"infactibles={len(centros_infactibles)} | "
                    f"repetidos={repetidos}"
                )
                print(mensaje)
                with open(ruta_log, "a", encoding="utf-8") as f_log:
                    f_log.write(mensaje + "\n")

            continue

        centros_total_i = centros_fijados + list(centros_i)

        if len(centros_total_i) != K_centros:
            mensaje = f"[ERROR] Número de centros incorrecto en intento {intentos}: {len(centros_total_i)}"
            print(mensaje)
            with open(ruta_log, "a", encoding="utf-8") as f_log:
                f_log.write(mensaje + "\n")
            continue

        modelo_i = modelo_centros_fijos_con_limite(
            epsilon_1, R, centros_total_i, dict_s, comunas, verbose=False
        )

        # si el modelo retorna None / False / no factible
        if not modelo_i:
            centros_infactibles.append(centros_total_i)
            centros_infactibles_set.add(centros_i_key)

            with open(ruta_centros_infactibles, "a", encoding="utf-8") as f:
                f.write(",".join(centros_total_i) + "\n")

            mensaje = (
                f"[INFACITBLE] intento={intentos} | "
                f"factibles={len(centros_factibles)} | "
                f"infactibles={len(centros_infactibles)} | "
                f"repetidos={repetidos}"
            )
            print(mensaje)

            with open(ruta_log, "a", encoding="utf-8") as f_log:
                f_log.write(mensaje + "\n")

            if intentos % LOG_CADA == 0:
                mensaje_log = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"intentos={intentos} | "
                    f"factibles={len(centros_factibles)} | "
                    f"infactibles={len(centros_infactibles)} | "
                    f"repetidos={repetidos}"
                )
                print(mensaje_log)
                with open(ruta_log, "a", encoding="utf-8") as f_log:
                    f_log.write(mensaje_log + "\n")

            continue

        # si llegó acá, el modelo fue factible
        centros_factibles.append(centros_total_i)
        centros_factibles_set.add(centros_i_key)

        with open(ruta_centros_factibles, "a", encoding="utf-8") as f:
            f.write(",".join(centros_total_i) + "\n")

        nombre_resultado = f"t_{len(centros_factibles):03d}"

        metadata = {
            "epsilon": epsilon_1,
            "K_centros": K_centros,
            "cantidad_centros": len(centros_total_i),
            "intento": intentos,
            "metodo_sampleo": "systematic"
        }

        guardar_resultado_factible(
            base_resultados,
            nombre_resultado,
            modelo_i,
            centros_total_i,
            metadata=metadata
        )

        mensaje = (
            f"[OK] {nombre_resultado} | "
            f"intento={intentos} | "
            f"factibles={len(centros_factibles)} | "
            f"infactibles={len(centros_infactibles)} | "
            f"repetidos={repetidos}"
        )
        print(mensaje)

        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")

        if intentos % LOG_CADA == 0:
            mensaje_log = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"intentos={intentos} | "
                f"factibles={len(centros_factibles)} | "
                f"infactibles={len(centros_infactibles)} | "
                f"repetidos={repetidos}"
            )
            print(mensaje_log)
            with open(ruta_log, "a", encoding="utf-8") as f_log:
                f_log.write(mensaje_log + "\n")

    except gp.GurobiError as e:
        errores_gurobi += 1
        mensaje = f"[GUROBI ERROR] intento={intentos} | código={getattr(e, 'errno', 'NA')} | mensaje={str(e)}"
        print(mensaje)
        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")
        continue

    except Exception as e:
        errores_otro_tipo += 1
        mensaje = f"[ERROR GENERAL] intento={intentos} | mensaje={str(e)}"
        print(mensaje)
        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")
        continue

tiempo_total = time.time() - t_inicio
horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("\nResumen final")
print(f"Mapas factibles guardados: {len(centros_factibles)}")
print(f"Combinaciones infactibles detectadas: {len(centros_infactibles)}")
print(f"Intentos totales: {intentos}")
print(f"Repetidos saltados: {repetidos}")
print(f"Errores Gurobi: {errores_gurobi}")
print(f"Otros errores: {errores_otro_tipo}")
print(f"Tiempo total de ejecución: {horas}h {minutos}m {segundos}s")
print(f"Resultados guardados en {base_resultados}")

print(f"Total combinaciones posibles: {total_combinaciones}")
print(f"Exploradas: {len(centros_factibles_set) + len(centros_infactibles_set)}")

with open(ruta_log, "a", encoding="utf-8") as f_log:
    f_log.write("\nResumen final\n")
    f_log.write(f"Mapas factibles guardados: {len(centros_factibles)}\n")
    f_log.write(f"Combinaciones infactibles detectadas: {len(centros_infactibles)}\n")
    f_log.write(f"Intentos totales: {intentos}\n")
    f_log.write(f"Repetidos saltados: {repetidos}\n")
    f_log.write(f"Errores Gurobi: {errores_gurobi}\n")
    f_log.write(f"Otros errores: {errores_otro_tipo}\n")
    f_log.write(f"Tiempo total: {horas}h {minutos}m {segundos}s\n")
    f_log.write(f"Fin corrida: {time.ctime()}\n")