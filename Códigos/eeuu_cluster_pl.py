import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time, json, os, re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import pickle
from collections import defaultdict

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time, json, os, re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import json

from funciones import (obtener_comunas, dist, obtener_region, resultados_sampleo, ensure_dir,
                       safe_attr, parse_x_name, extraer_y_guardar_modelo, matriz_X_desde_modelo,
                       promedio_X, comparar_con_baseline, build_matrices_from_gurobi, delta_b_from_eps,
                       extraer_prob_centros)

from modelos import (modelo_con_limite_sparse_v2)

from data_chile_distrito_censal.chile_data import regiones

print("\n==============================")
print("LEYENDO DATOS CHILE CENSAL")
print("==============================")

comunas = pd.read_excel('data_chile_distrito_censal/comunas_chile_2024_caso_B_conectado.xlsx')
print("[OK] comunas_chile.xlsx leído")
print("shape comunas:", comunas.shape)
print("columnas comunas:", comunas.columns.tolist())
print(comunas.head())

distancias = pd.read_excel('data_chile_distrito_censal/distancias_chile_2024_caso_B_conectado.xlsx')
print("\n[OK] distancias_chile.xlsx leído")
print("shape distancias:", distancias.shape)
print("primeras columnas distancias:", distancias.columns[:8].tolist())

print("\nLeyendo s_nuevo...")
with open(
    "data_chile_distrito_censal/s_nuevo_chile_2024_caso_B_conectado.pkl",
    "rb"
) as f:
    dict_s_base = pickle.load(f)

print("[OK] s_nuevo_chile.txt leído")

print("Evaluando dict_s...")
dict_s = defaultdict(lambda: [[]], dict_s_base)
print("[OK] s_nuevo_chile_sparse.pkl leído")
print("Creando defaultdict...")

print("\nChequeando regiones disponibles...")
print("Regiones en comunas:")
print(sorted(comunas["region"].unique()))

print("\nRegiones esperadas:")
print(regiones)

R_por_region = {}

for region in sorted(comunas["region"].unique()):
    r_region = obtener_comunas(comunas, region)
    R_por_region[region] = r_region
    print(f"{region}: {len(r_region)} unidades")

R = sum(R_por_region.values(), [])

print("\nTotal R:", len(R))
print("Total comunas archivo:", len(comunas))

faltan_en_R = set(comunas["comuna"]) - set(R)
print("Unidades no incluidas en R:", len(faltan_en_R))

if faltan_en_R:
    print(list(faltan_en_R)[:30])

print("==============================")
print("FIN LECTURA")
print("==============================\n")


# # ============================================================
# # BÚSQUEDA DE EPSILON FACTIBLE
# # ============================================================

epsilon_inicial = 0.6
epsilon_max = 0.8
paso = 0.1

K = 28

ensure_dir("datos_modelo")

epsilon = epsilon_inicial
modelo_factible = None
epsilon_factible = None

while epsilon <= epsilon_max + 1e-12:

    print(f"\nProbando epsilon = {epsilon:.5f}")

    modelo = modelo_con_limite_sparse_v2(
        epsilon=epsilon,
        R=R,
        K=28,
        dict_s=dict_s,
        comunas=comunas,
        distancias=distancias,
        M=260,
        max_iter_cierre=80
    )
    if modelo is None:
        print(f"epsilon={epsilon:.5f} infactible/no óptimo")
        epsilon += paso
        continue

    status = modelo.Status

    if status == GRB.OPTIMAL:

        print(f"[OK] Modelo factible con epsilon = {epsilon:.5f}")

        modelo_factible = modelo
        epsilon_factible = epsilon

        ruta_lp = (
            f"datos_modelo/modelo_chile_censal_eps_"
            f"{epsilon:.5f}_B_cl_v2.lp"
        )

        modelo.write(ruta_lp)

        valores = {
            v.VarName: v.X
            for v in modelo.getVars()
        }

        ruta_json = (
            f"datos_modelo/valores_chile_censal_eps_"
            f"{epsilon:.5f}_B_cl_v2.json"
        )

        with open(ruta_json, "w") as f:
            json.dump(valores, f)

        with open(
            "datos_modelo/epsilon_factible_chile_censal_B_cl_v2.txt",
            "w"
        ) as f:
            f.write(str(epsilon_factible))

        print(f"[OK] LP guardado en: {ruta_lp}")
        print(f"[OK] Valores guardados en: {ruta_json}")
        print("[FIN] Se detiene la búsqueda.")

        break

    elif status == GRB.INFEASIBLE:
        print(f"[NO] Infactible con epsilon = {epsilon:.5f}")

    else:
        print(f"[WARN] Status inesperado: {status}")

    epsilon += paso

if modelo_factible is None:
    print(
        f"\n[FIN] No se encontró factibilidad "
        f"hasta epsilon = {epsilon_max}"
    )
else:
    print(
        f"\nEpsilon factible encontrado: "
        f"{epsilon_factible:.5f}"
    )