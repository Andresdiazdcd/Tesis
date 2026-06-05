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

from modelos import (modelo_con_limite, modelo_centros_fijos_sin_limite,
                     modelo_sin_limite, modelo_IP)

comunas = pd.read_excel('Dict_eeuu/data_eeuu_procesada_county_muni/comunas_pa.xlsx')
distancias = pd.read_excel('Dict_eeuu/data_eeuu_procesada_county_muni/distancias_pa.xlsx')

with open('Dict_eeuu/data_eeuu_procesada_county_muni/s_nuevo_pa.txt', 'r') as dict_file:
    dict_text = dict_file.read()
    dict_s = eval(dict_text)

R = obtener_comunas(comunas, "pennsylvania")

# ============================================================
# BÚSQUEDA DE EPSILON FACTIBLE
# ============================================================

epsilon_inicial = 0.00001
epsilon_max = 0.5
paso = 0.005  

K = 17

ensure_dir("datos_modelo")

epsilon = epsilon_inicial
modelo_factible = None
epsilon_factible = None

while epsilon <= epsilon_max + 1e-12:

    print(f"\nProbando epsilon = {epsilon:.5f}")

    modelo = modelo_sin_limite(epsilon, R, K, dict_s, comunas)

    status = modelo.Status

    if status == GRB.OPTIMAL:
        print(f"[OK] Modelo factible con epsilon = {epsilon:.5f}")

        modelo_factible = modelo
        epsilon_factible = epsilon

        # Guardar PL
        ruta_lp = f"datos_modelo/modelo_pa_county_muni_eps_{epsilon:.5f}.lp"
        modelo.write(ruta_lp)

        # Guardar valores
        valores_ia = {v.VarName: v.X for v in modelo.getVars()}

        ruta_json = f"datos_modelo/valores_pa_county_muni_eps_{epsilon:.5f}.json"
        with open(ruta_json, "w") as f:
            json.dump(valores_ia, f)

        # Guardar epsilon encontrado
        with open("datos_modelo/epsilon_factible_pa_county_muni.txt", "w") as f:
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
    print(f"\n[FIN] No se encontró factibilidad hasta epsilon = {epsilon_max}")
else:
    print(f"\nEpsilon factible encontrado: {epsilon_factible:.5f}")