import os
import ast
import json
import pandas as pd
from collections import defaultdict

from gurobipy import GRB

from funciones import ensure_dir
from modelos import modelo_sin_limite_sparse_v2


# ============================================================
# RUTAS PENNSYLVANIA
# ============================================================

BASE_PA = "DataEEUU2/data_eeuu_procesada_county_muni_pa"

RUTA_COMUNAS = os.path.join(
    BASE_PA,
    "comunas_pa.xlsx"
)

RUTA_DISTANCIAS = os.path.join(
    BASE_PA,
    "distancias_pa.xlsx"
)

RUTA_S_NUEVO = os.path.join(
    BASE_PA,
    "s_nuevo_pa.txt"
)


# ============================================================
# LECTURA DE DATOS
# ============================================================

print("\n==============================")
print("LEYENDO DATOS PENNSYLVANIA")
print("==============================")

comunas = pd.read_excel(RUTA_COMUNAS)

print("[OK] comunas_pa.xlsx leído")
print("shape comunas:", comunas.shape)
print("columnas comunas:", comunas.columns.tolist())
print(comunas.head())


distancias = pd.read_excel(RUTA_DISTANCIAS)

print("\n[OK] distancias_pa.xlsx leído")
print("shape distancias:", distancias.shape)
print(
    "primeras columnas distancias:",
    distancias.columns[:8].tolist()
)


# ============================================================
# LECTURA DE S_NUEVO DESDE TXT
# ============================================================

print("\nLeyendo s_nuevo_pa.txt...")

with open(RUTA_S_NUEVO, "r", encoding="utf-8") as f:
    contenido = f.read()

dict_s_base = ast.literal_eval(contenido)

dict_s = defaultdict(
    lambda: [[]],
    dict_s_base
)

print("[OK] s_nuevo_pa.txt leído")
print("Cantidad de entradas:", len(dict_s_base))


# ============================================================
# CONSTRUCCIÓN DE R
# ============================================================
# En Estados Unidos no se impone límite regional.
# Por eso R contiene directamente todas las unidades.
# ============================================================

if "comuna" not in comunas.columns:
    raise ValueError(
        "comunas_pa.xlsx no contiene la columna 'comuna'."
    )

comunas["comuna"] = comunas["comuna"].astype(str)

R = comunas["comuna"].tolist()

print("\nTotal R:", len(R))
print("Unidades únicas:", comunas["comuna"].nunique())

if len(R) != comunas["comuna"].nunique():
    duplicadas = comunas.loc[
        comunas["comuna"].duplicated(keep=False),
        "comuna"
    ].tolist()

    print("[WARN] Hay unidades duplicadas:")
    print(duplicadas[:30])

print("==============================")
print("FIN LECTURA")
print("==============================\n")


# ============================================================
# BÚSQUEDA DE EPSILON FACTIBLE
# ============================================================

epsilon_inicial = 0.01
epsilon_max = 0.1
paso = 0.01

K = 17

M = 250
MAX_ITER_CIERRE = 80

ensure_dir("datos_modelo")

epsilon = epsilon_inicial
modelo_factible = None
epsilon_factible = None


while epsilon <= epsilon_max + 1e-12:

    print(
        f"\nProbando epsilon = {epsilon:.5f}",
        flush=True
    )

    modelo = modelo_sin_limite_sparse_v2(
        epsilon=epsilon,
        R=R,
        K=K,
        dict_s=dict_s,
        comunas=comunas,
        distancias=distancias,
        M=M,
        max_iter_cierre=MAX_ITER_CIERRE
    )

    if modelo is None:
        print(
            f"epsilon={epsilon:.5f} infactible/no óptimo",
            flush=True
        )

        epsilon += paso
        continue

    status = modelo.Status

    if status == GRB.OPTIMAL:

        print(
            f"[OK] Modelo factible con epsilon={epsilon:.5f}",
            flush=True
        )

        modelo_factible = modelo
        epsilon_factible = epsilon

        ruta_lp = (
            "datos_modelo/"
            f"modelo_pa_eps_{epsilon:.5f}_v2.lp"
        )

        modelo.write(ruta_lp)

        valores = {
            variable.VarName: variable.X
            for variable in modelo.getVars()
        }

        ruta_json = (
            "datos_modelo/"
            f"valores_pa_eps_{epsilon:.5f}_v2.json"
        )

        with open(
            ruta_json,
            "w",
            encoding="utf-8"
        ) as archivo:
            json.dump(valores, archivo)

        ruta_epsilon = (
            "datos_modelo/"
            "epsilon_factible_pa_v2.txt"
        )

        with open(
            ruta_epsilon,
            "w",
            encoding="utf-8"
        ) as archivo:
            archivo.write(str(epsilon_factible))

        print(f"[OK] LP guardado en: {ruta_lp}")
        print(f"[OK] Valores guardados en: {ruta_json}")
        print(f"[OK] Epsilon guardado en: {ruta_epsilon}")
        print("[FIN] Se detiene la búsqueda.")

        break

    elif status == GRB.INFEASIBLE:
        print(
            f"[NO] Infactible con epsilon={epsilon:.5f}"
        )

    else:
        print(
            f"[WARN] Status inesperado: {status}"
        )

    epsilon += paso


# ============================================================
# RESUMEN
# ============================================================

if modelo_factible is None:
    print(
        "\n[FIN] No se encontró factibilidad "
        f"hasta epsilon={epsilon_max}"
    )
else:
    print(
        "\nEpsilon factible encontrado: "
        f"{epsilon_factible:.5f}"
    )