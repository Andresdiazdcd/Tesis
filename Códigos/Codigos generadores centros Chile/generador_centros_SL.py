import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time, json, os, re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import sys
from funciones import (obtener_comunas, dist, obtener_region)

from DataChile.chile_data import regiones

from modelos import modelo_sin_limite, modelo_centros_fijos_sin_limite


from sampleos import systematic_sampling, pivotal_sampling

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
with open('DataChile/s_new.txt', 'r') as dict_file:
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

TIPO_MODELO = "SL" # Sin LIMITE
METODO = "PIVOTAL" # Systematic
modelo_sin_limite = modelo_sin_limite(0.0390625, R, K_centros, dict_s, comunas)

count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(modelo_sin_limite, 28)

print(f"Centros fijados por modelo: {count_centros_fijados}")

def centros_fijados(modelo):
    centros = []
    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and v.x == 1.0:
            texto_1 = v.VarName
            comuna = texto_1[texto_1.find('[')+1 : texto_1.find(']')]
            centros.append(comuna)
    return centros

centros_fijados = centros_fijados(modelo_sin_limite)

print(f"Centros fijados por modelo check 1: {len(centros_fijados)}")

comunas_t, probabilidades = zip(*centros_frac)
comunas_t = list(comunas_t)
probabilidades = list(probabilidades)

HORAS = 20
TIEMPO_MAX = HORAS * 60 * 60
t_inicio = time.time()

epsilon_1 = 0.75
t_mapas = 100
k_sampleo = K_centros - len(centros_fijados)

n = len(comunas_t)
total_combinaciones = math.comb(len(comunas_t), k_sampleo)

print(f"Centros a samplear: {k_sampleo}")
print(f"Cantidad de comunas fraccionarias: {n}")
print(f"Total combinaciones teóricas: {total_combinaciones}")
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

# todas las combinaciones distintas observadas por el sampleo
centros_observados_set = set()

intentos = 0
repetidos = 0
repetidos_consecutivos = 0
errores_gurobi = 0
errores_otro_tipo = 0

LOG_CADA = 50
MAX_REPETIDOS_CONSECUTIVOS = 5000
PREMUESTREO_MUESTRAS = 10000


def log_mensaje(mensaje, consola=True, archivo=True, overwrite=False):
    if consola:
        if overwrite:
            print(mensaje, end="\r", flush=True)
        else:
            print(mensaje, flush=True)

    if archivo:
        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")


def estado_actual():
    exploradas = len(centros_factibles_set) + len(centros_infactibles_set)
    porcentaje_explorado = 100 * exploradas / total_combinaciones if total_combinaciones > 0 else 0
    porcentaje_observado = 100 * len(centros_observados_set) / total_combinaciones if total_combinaciones > 0 else 0

    return (
        f"[{time.strftime('%H:%M:%S')}] "
        f"intentos={intentos} | "
        f"obs_distintas={len(centros_observados_set)}/{total_combinaciones} ({porcentaje_observado:.2f}%) | "
        f"exploradas={exploradas}/{total_combinaciones} ({porcentaje_explorado:.2f}%) | "
        f"factibles={len(centros_factibles)} | "
        f"infactibles={len(centros_infactibles)} | "
        f"repetidos={repetidos} | "
        f"rep_cons={repetidos_consecutivos}"
    )


with open(ruta_log, "w", encoding="utf-8") as f_log:
    f_log.write(f"Inicio corrida: {time.ctime()}\n")
    f_log.write(f"TIPO_MODELO = {TIPO_MODELO}\n")
    f_log.write(f"METODO = {METODO}\n")
    f_log.write(f"epsilon_1 = {epsilon_1}\n")
    f_log.write(f"K_centros = {K_centros}\n")
    f_log.write(f"centros_fijados = {len(centros_fijados)}\n")
    f_log.write(f"centros_a_samplear = {k_sampleo}\n")
    f_log.write(f"comunas_fraccionarias = {n}\n")
    f_log.write(f"total_combinaciones_teoricas = {total_combinaciones}\n")
    f_log.write(f"t_mapas_objetivo = {t_mapas}\n\n")

# ---------------------------------
# PREMUESTREO DIAGNÓSTICO
# ---------------------------------
vistos_premuestreo = set()

for _ in range(PREMUESTREO_MUESTRAS):
    centros_i = pivotal_sampling(comunas_t, probabilidades)#, k_sampleo)
    vistos_premuestreo.add(tuple(sorted(centros_i)))

mensaje = (
    f"Premuestreo {METODO}: {len(vistos_premuestreo)} combinaciones distintas "
    f"observadas en {PREMUESTREO_MUESTRAS} sampleos "
    f"(sobre {total_combinaciones} teóricas)"
)
log_mensaje(mensaje, consola=True, archivo=True, overwrite=False)

# ---------------------------------
# LOOP PRINCIPAL
# ---------------------------------
while len(centros_factibles) < t_mapas:

    exploradas = len(centros_factibles_set) + len(centros_infactibles_set)

    if exploradas == total_combinaciones:
        print()
        log_mensaje("Se exploraron todas las combinaciones posibles.", consola=True, archivo=True)
        break

    tiempo_transcurrido = time.time() - t_inicio
    if tiempo_transcurrido >= TIEMPO_MAX:
        print()
        log_mensaje(f"Límite de {HORAS} horas alcanzado.", consola=True, archivo=True)
        log_mensaje(f"Límite de tiempo alcanzado en intento {intentos}", consola=False, archivo=True)
        break

    if repetidos_consecutivos >= MAX_REPETIDOS_CONSECUTIVOS:
        print()
        log_mensaje(
            f"Se detiene por estancamiento: {repetidos_consecutivos} repetidos consecutivos.",
            consola=True,
            archivo=True
        )
        break

    intentos += 1

    try:
        centros_i = pivotal_sampling(comunas_t, probabilidades)#, k_sampleo)
        centros_i_key = tuple(sorted(centros_i))

        centros_observados_set.add(centros_i_key)

        # skip si ya fue visto antes y ya fue evaluado
        if centros_i_key in centros_factibles_set or centros_i_key in centros_infactibles_set:
            repetidos += 1
            repetidos_consecutivos += 1

            if intentos % LOG_CADA == 0:
                mensaje_estado = estado_actual()
                log_mensaje(mensaje_estado, consola=True, archivo=True, overwrite=True)

            continue

        repetidos_consecutivos = 0

        centros_total_i = centros_fijados + list(centros_i)

        if len(centros_total_i) != K_centros:
            print()
            log_mensaje(
                f"[ERROR] Número de centros incorrecto en intento {intentos}: {len(centros_total_i)}",
                consola=True,
                archivo=True
            )
            continue

        modelo_i = modelo_centros_fijos_sin_limite(
            epsilon_1, R, centros_total_i, dict_s, comunas, verbose=False
        )

        # INFATIBLE
        if not modelo_i:
            centros_infactibles.append(centros_total_i)
            centros_infactibles_set.add(centros_i_key)

            with open(ruta_centros_infactibles, "a", encoding="utf-8") as f:
                f.write(",".join(centros_total_i) + "\n")

            if intentos % LOG_CADA == 0:
                mensaje_estado = estado_actual()
                log_mensaje(mensaje_estado, consola=True, archivo=True, overwrite=True)

            continue

        # FACTIBLE
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
            "metodo_sampleo": {METODO}
        }

        guardar_resultado_factible(
            base_resultados,
            nombre_resultado,
            modelo_i,
            centros_total_i,
            metadata=metadata
        )

        print()
        log_mensaje(
            f"[OK] {nombre_resultado} | intento={intentos} | "
            f"factibles={len(centros_factibles)} | "
            f"infactibles={len(centros_infactibles)} | "
            f"obs_distintas={len(centros_observados_set)} | "
            f"repetidos={repetidos}",
            consola=True,
            archivo=True
        )

        if intentos % LOG_CADA == 0:
            mensaje_estado = estado_actual()
            log_mensaje(mensaje_estado, consola=True, archivo=True, overwrite=True)

    except gp.GurobiError as e:
        errores_gurobi += 1
        print()
        log_mensaje(
            f"[GUROBI ERROR] intento={intentos} | código={getattr(e, 'errno', 'NA')} | mensaje={str(e)}",
            consola=True,
            archivo=True
        )
        continue

    except Exception as e:
        errores_otro_tipo += 1
        print()
        log_mensaje(
            f"[ERROR GENERAL] intento={intentos} | mensaje={str(e)}",
            consola=True,
            archivo=True
        )
        continue

# limpiar línea dinámica
print()

tiempo_total = time.time() - t_inicio
horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

exploradas = len(centros_factibles_set) + len(centros_infactibles_set)
porcentaje_explorado = 100 * exploradas / total_combinaciones if total_combinaciones > 0 else 0
porcentaje_observado = 100 * len(centros_observados_set) / total_combinaciones if total_combinaciones > 0 else 0

print("Resumen final")
print(f"Mapas factibles guardados: {len(centros_factibles)}")
print(f"Combinaciones infactibles detectadas: {len(centros_infactibles)}")
print(f"Combinaciones distintas observadas por el sampleo: {len(centros_observados_set)}")
print(f"Intentos totales: {intentos}")
print(f"Repetidos saltados: {repetidos}")
print(f"Repetidos consecutivos al cierre: {repetidos_consecutivos}")
print(f"Errores Gurobi: {errores_gurobi}")
print(f"Otros errores: {errores_otro_tipo}")
print(f"Tiempo total de ejecución: {horas}h {minutos}m {segundos}s")
print(f"Resultados guardados en {base_resultados}")
print(f"Total combinaciones teóricas: {total_combinaciones}")
print(f"Porcentaje observado por sampleo: {porcentaje_observado:.2f}%")
print(f"Porcentaje explorado: {porcentaje_explorado:.2f}%")

with open(ruta_log, "a", encoding="utf-8") as f_log:
    f_log.write("\nResumen final\n")
    f_log.write(f"Mapas factibles guardados: {len(centros_factibles)}\n")
    f_log.write(f"Combinaciones infactibles detectadas: {len(centros_infactibles)}\n")
    f_log.write(f"Combinaciones distintas observadas por el sampleo: {len(centros_observados_set)}\n")
    f_log.write(f"Intentos totales: {intentos}\n")
    f_log.write(f"Repetidos saltados: {repetidos}\n")
    f_log.write(f"Repetidos consecutivos al cierre: {repetidos_consecutivos}\n")
    f_log.write(f"Errores Gurobi: {errores_gurobi}\n")
    f_log.write(f"Otros errores: {errores_otro_tipo}\n")
    f_log.write(f"Tiempo total: {horas}h {minutos}m {segundos}s\n")
    f_log.write(f"Total combinaciones teóricas: {total_combinaciones}\n")
    f_log.write(f"Porcentaje observado por sampleo: {porcentaje_observado:.2f}%\n")
    f_log.write(f"Porcentaje explorado: {porcentaje_explorado:.2f}%\n")
    f_log.write(f"Fin corrida: {time.ctime()}\n")