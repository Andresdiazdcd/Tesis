import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time
import json
import os
import re
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import pickle

from funciones import (
    obtener_comunas,
    dist,
    obtener_region,
    resultados_sampleo,
    ensure_dir,
    safe_attr,
    parse_x_name,
    extraer_y_guardar_modelo,
    matriz_X_desde_modelo,
    promedio_X,
    comparar_con_baseline,
    build_matrices_from_gurobi,
    delta_b_from_eps,
    extraer_prob_centros
)

from modelos import modelo_sin_limite_opti #_sparse_v2

from data_chile_distrito_censal.chile_data import regiones


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_COMUNAS = (
    "data_chile_distrito_censal/"
    "comunas_chile_2024_caso_A_principal.xlsx"
)

RUTA_DISTANCIAS = (
    "data_chile_distrito_censal/"
    "distancias_chile_2024_caso_A_principal.xlsx"
)

RUTA_S_NUEVO = (
    "data_chile_distrito_censal/"
    "s_nuevo_chile_2024_caso_A_principal.pkl"
)

CARPETA_RESULTADOS = (
    "datos_modelo/"
    "chile_censal_reducido"
)


# Regiones desde la IV Región hasta La Araucanía.
REGIONES_UTILIZADAS = [
    "coquimbo",
    "valparaiso",
    "metropolitana_de_santiago",
    "libertador_general_bernardo_ohiggins",
    "maule",
    "nuble",
    "biobio",
    #"la_araucania"
]


# El modelo nacional utiliza K=28.

K_NACIONAL = 28


# Parámetros de búsqueda de epsilon.
epsilon_inicial = 0.05
epsilon_max = 0.1
paso = 0.01


# Parámetros sparse v2.
M = 280
MAX_ITER_CIERRE = 80


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def detectar_columna_poblacion(df):
    """
    Detecta la columna que contiene la población.
    """

    columnas_posibles = [
        "poblacion2024",
        "poblacion_2024",
        "poblacion2017",
        "poblacion_2017",
        "poblacion",
        "pop"
    ]

    for columna in columnas_posibles:
        if columna in df.columns:
            return columna

    raise ValueError(
        "No se encontró una columna de población.\n"
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def guardar_informacion_subconjunto(
    comunas_filtradas,
    regiones_utilizadas,
    columna_poblacion,
    carpeta_salida,
    k
):
    """
    Guarda la información del subconjunto geográfico utilizado.
    """

    ensure_dir(carpeta_salida)

    poblacion_total = int(
        comunas_filtradas[columna_poblacion]
        .fillna(0)
        .sum()
    )

    resumen_regiones = (
        comunas_filtradas
        .groupby("region", as_index=False)
        .agg(
            cantidad_unidades=("comuna", "count"),
            poblacion=(columna_poblacion, "sum")
        )
        .sort_values(
            "poblacion",
            ascending=False
        )
        .reset_index(drop=True)
    )

    resumen_regiones["porcentaje_poblacion"] = (
        resumen_regiones["poblacion"]
        / poblacion_total
    )

    poblacion_objetivo = (
        poblacion_total / k
    )

    informacion = {
        "regiones_utilizadas": regiones_utilizadas,
        "cantidad_regiones": len(regiones_utilizadas),
        "cantidad_unidades": int(len(comunas_filtradas)),
        "columna_poblacion": columna_poblacion,
        "poblacion_total": poblacion_total,
        "K": int(k),
        "poblacion_objetivo_por_distrito": float(
            poblacion_objetivo
        )
    }

    ruta_json = os.path.join(
        carpeta_salida,
        "informacion_subconjunto.json"
    )

    with open(
        ruta_json,
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            informacion,
            archivo,
            ensure_ascii=False,
            indent=4
        )

    ruta_regiones_txt = os.path.join(
        carpeta_salida,
        "regiones_utilizadas.txt"
    )

    with open(
        ruta_regiones_txt,
        "w",
        encoding="utf-8"
    ) as archivo:

        for region in regiones_utilizadas:
            archivo.write(f"{region}\n")

    ruta_resumen_csv = os.path.join(
        carpeta_salida,
        "resumen_regiones.csv"
    )

    resumen_regiones.to_csv(
        ruta_resumen_csv,
        index=False,
        encoding="utf-8-sig"
    )

    ruta_unidades_csv = os.path.join(
        carpeta_salida,
        "unidades_utilizadas.csv"
    )

    columnas_unidades = [
        columna
        for columna in [
            "comuna",
            "region",
            columna_poblacion
        ]
        if columna in comunas_filtradas.columns
    ]

    comunas_filtradas[
        columnas_unidades
    ].to_csv(
        ruta_unidades_csv,
        index=False,
        encoding="utf-8-sig"
    )

    ruta_comunas_excel = os.path.join(
        carpeta_salida,
        "comunas_filtradas.xlsx"
    )

    comunas_filtradas.to_excel(
        ruta_comunas_excel,
        index=False
    )

    print("\n[OK] Información del subconjunto guardada:")

    print(f"  JSON: {ruta_json}")
    print(f"  regiones: {ruta_regiones_txt}")
    print(f"  resumen: {ruta_resumen_csv}")
    print(f"  unidades: {ruta_unidades_csv}")
    print(f"  comunas filtradas: {ruta_comunas_excel}")

    return informacion, resumen_regiones


# ============================================================
# LECTURA DE DATOS
# ============================================================

print("\n==============================")
print("LEYENDO DATOS CHILE CENSAL")
print("==============================")


# ------------------------------------------------------------
# Comunas o distritos censales
# ------------------------------------------------------------

comunas_completas = pd.read_excel(
    RUTA_COMUNAS
)

print(f"[OK] Archivo leído: {RUTA_COMUNAS}")
print("Shape comunas completas:", comunas_completas.shape)
print("Columnas comunas:", comunas_completas.columns.tolist())
print(comunas_completas.head())


if "comuna" not in comunas_completas.columns:
    raise ValueError(
        "El archivo de comunas no contiene la columna 'comuna'."
    )


if "region" not in comunas_completas.columns:
    raise ValueError(
        "El archivo de comunas no contiene la columna 'region'."
    )


columna_poblacion = detectar_columna_poblacion(
    comunas_completas
)

print(
    "\nColumna de población detectada:",
    columna_poblacion
)


# ------------------------------------------------------------
# Distancias
# ------------------------------------------------------------

distancias = pd.read_excel(
    RUTA_DISTANCIAS
)

print(f"\n[OK] Archivo leído: {RUTA_DISTANCIAS}")
print("Shape distancias:", distancias.shape)

print(
    "Primeras columnas distancias:",
    distancias.columns[:8].tolist()
)


# ------------------------------------------------------------
# Caminos de contigüidad
# ------------------------------------------------------------

print("\nLeyendo s_nuevo...")

with open(
    RUTA_S_NUEVO,
    "rb"
) as archivo:
    dict_s_base = pickle.load(archivo)

print(f"[OK] Archivo leído: {RUTA_S_NUEVO}")


print("\nCreando defaultdict para dict_s...")

dict_s = defaultdict(
    lambda: [[]],
    dict_s_base
)

print("[OK] dict_s creado")


# ============================================================
# FILTRAR DESDE COQUIMBO HASTA LA ARAUCANÍA
# ============================================================

print("\n==========================================")
print("FILTRANDO REGIONES UTILIZADAS")
print("==========================================")


regiones_disponibles = sorted(
    comunas_completas["region"]
    .dropna()
    .unique()
    .tolist()
)

print("\nRegiones disponibles en el archivo:")

for region in regiones_disponibles:
    print(f"  - {region}")


regiones_no_encontradas = [
    region
    for region in REGIONES_UTILIZADAS
    if region not in regiones_disponibles
]


if regiones_no_encontradas:

    raise ValueError(
        "\nNo se encontraron las siguientes regiones "
        "en el archivo de comunas:\n"
        f"{regiones_no_encontradas}\n\n"
        "Revise que los nombres coincidan exactamente con "
        "los valores de la columna 'region'."
    )


comunas = comunas_completas[
    comunas_completas["region"].isin(
        REGIONES_UTILIZADAS
    )
].copy()


comunas.reset_index(
    drop=True,
    inplace=True
)


if comunas.empty:
    raise ValueError(
        "El filtrado por regiones produjo un DataFrame vacío."
    )


regiones_en_subconjunto = sorted(
    comunas["region"]
    .dropna()
    .unique()
    .tolist()
)


print("\nRegiones incluidas en el modelo:")

for region in regiones_en_subconjunto:
    print(f"  - {region}")


print(
    "\nCantidad de regiones:",
    len(regiones_en_subconjunto)
)

print(
    "Cantidad de unidades seleccionadas:",
    len(comunas)
)


poblacion_nacional_archivo = (
    comunas_completas[columna_poblacion]
    .fillna(0)
    .sum()
)

poblacion_subconjunto = (
    comunas[columna_poblacion]
    .fillna(0)
    .sum()
)

# ============================================================
# CÁLCULO AUTOMÁTICO DE K SEGÚN POBLACIÓN
# ============================================================

poblacion_objetivo_nacional = (
    poblacion_nacional_archivo / K_NACIONAL
)

k_proporcional = (
    poblacion_subconjunto / poblacion_objetivo_nacional
)

K = round(k_proporcional)

print("\n==========================================")
print("CÁLCULO DE K SEGÚN POBLACIÓN")
print("==========================================")

print(
    "Población nacional:",
    f"{poblacion_nacional_archivo:,.0f}"
)

print(
    "Población del subconjunto:",
    f"{poblacion_subconjunto:,.0f}"
)

print(
    "Población objetivo nacional por distrito:",
    f"{poblacion_objetivo_nacional:,.2f}"
)

print(
    "K proporcional:",
    f"{k_proporcional:.4f}"
)

print(
    "K utilizado:",
    K
)

print(
    "Población objetivo del subconjunto:",
    f"{poblacion_subconjunto / K:,.2f}"
)

# ============================================================
# GUARDAR EL SUBCONJUNTO
# ============================================================

ensure_dir("datos_modelo")
ensure_dir(CARPETA_RESULTADOS)


informacion_subconjunto, resumen_regiones = (
    guardar_informacion_subconjunto(
        comunas_filtradas=comunas,
        regiones_utilizadas=REGIONES_UTILIZADAS,
        columna_poblacion=columna_poblacion,
        carpeta_salida=CARPETA_RESULTADOS,
        k=K
    )
)


print("\nResumen por región:")

print(
    resumen_regiones.to_string(
        index=False
    )
)


# ============================================================
# CONSTRUCCIÓN DE R
# ============================================================

print("\n==========================================")
print("CONSTRUYENDO R")
print("==========================================")


R_por_region = {}


for region in REGIONES_UTILIZADAS:

    r_region = obtener_comunas(
        comunas,
        region
    )

    R_por_region[region] = r_region

    poblacion_region = (
        comunas.loc[
            comunas["region"] == region,
            columna_poblacion
        ]
        .fillna(0)
        .sum()
    )

    print(
        f"{region}: "
        f"{len(r_region)} unidades, "
        f"población={poblacion_region:,.0f}"
    )


R = sum(
    R_por_region.values(),
    []
)


print("\nTotal R:", len(R))
print("Total filas en comunas filtradas:", len(comunas))


# ============================================================
# VERIFICACIONES DE CONSISTENCIA
# ============================================================

faltan_en_R = (
    set(comunas["comuna"])
    - set(R)
)

sobran_en_R = (
    set(R)
    - set(comunas["comuna"])
)


print(
    "Unidades del DataFrame que no están en R:",
    len(faltan_en_R)
)

print(
    "Unidades de R que no están en el DataFrame:",
    len(sobran_en_R)
)


if faltan_en_R:

    print("\nPrimeras unidades faltantes en R:")

    for unidad in list(faltan_en_R)[:30]:
        print(f"  - {unidad}")


if sobran_en_R:

    print("\nPrimeras unidades sobrantes en R:")

    for unidad in list(sobran_en_R)[:30]:
        print(f"  - {unidad}")


if len(R) != len(set(R)):

    duplicados_R = pd.Series(R)[
        pd.Series(R).duplicated()
    ].tolist()

    raise ValueError(
        "Existen unidades duplicadas en R.\n"
        f"Primeros duplicados: {duplicados_R[:30]}"
    )


if faltan_en_R or sobran_en_R:

    raise ValueError(
        "R no coincide con las unidades del subconjunto. "
        "Revise obtener_comunas() y los nombres de las regiones."
    )


print(
    "\n[OK] R coincide exactamente con las unidades "
    "del subconjunto."
)


print("\n==============================")
print("FIN LECTURA Y FILTRADO")
print("==============================\n")


# ============================================================
# BÚSQUEDA DE EPSILON FACTIBLE
# ============================================================

epsilon = epsilon_inicial

modelo_factible = None
epsilon_factible = None


while epsilon <= epsilon_max + 1e-12:

    print("\n==========================================")
    print(f"PROBANDO EPSILON = {epsilon:.5f}")
    print("==========================================")

    print("Cantidad de unidades:", len(R))
    print("Cantidad de regiones:", len(REGIONES_UTILIZADAS))
    print("K:", K)
    print("M:", M)
    print("Máximo cierre:", MAX_ITER_CIERRE)

    #modelo = modelo_sin_limite_sparse_v2(
    #    epsilon=epsilon,
    #    R=R,
    #    K=K,
    #    dict_s=dict_s,
    #    comunas=comunas,
    #    distancias=distancias,
    #    M=M,
    #    max_iter_cierre=MAX_ITER_CIERRE
    #)

    modelo = modelo_sin_limite_opti(
        epsilon=epsilon,
        R=R,
        K=K,
        dict_s=dict_s,
        comunas=comunas
    )



    if modelo is None:

        print(
            f"[NO] epsilon={epsilon:.5f}: "
            "modelo infactible o no óptimo"
        )

        epsilon = round(
            epsilon + paso,
            10
        )

        continue


    status = modelo.Status


    if status == GRB.OPTIMAL:

        print(
            f"\n[OK] Modelo factible con "
            f"epsilon={epsilon:.5f}"
        )

        modelo_factible = modelo
        epsilon_factible = epsilon


        # ----------------------------------------------------
        # Guardar LP
        # ----------------------------------------------------

        ruta_lp = os.path.join(
            CARPETA_RESULTADOS,
            (
                "modelo_chile_censal_"
                "coquimbo_araucania_"
                f"K_{K}_eps_{epsilon:.5f}_"
                "B_cl_v2.lp"
            )
        )

        modelo.write(
            ruta_lp
        )


        # ----------------------------------------------------
        # Guardar valores de las variables
        # ----------------------------------------------------

        valores = {
            variable.VarName: variable.X
            for variable in modelo.getVars()
        }


        ruta_valores_json = os.path.join(
            CARPETA_RESULTADOS,
            (
                "valores_chile_censal_"
                "coquimbo_araucania_"
                f"K_{K}_eps_{epsilon:.5f}_"
                "B_cl_v2.json"
            )
        )


        with open(
            ruta_valores_json,
            "w",
            encoding="utf-8"
        ) as archivo:

            json.dump(
                valores,
                archivo,
                ensure_ascii=False
            )


        # ----------------------------------------------------
        # Guardar epsilon factible
        # ----------------------------------------------------

        ruta_epsilon = os.path.join(
            CARPETA_RESULTADOS,
            (
                "epsilon_factible_chile_censal_"
                "coquimbo_araucania_"
                f"K_{K}_B_cl_v2.txt"
            )
        )


        with open(
            ruta_epsilon,
            "w",
            encoding="utf-8"
        ) as archivo:

            archivo.write(
                str(epsilon_factible)
            )


        # ----------------------------------------------------
        # Guardar información completa del experimento
        # ----------------------------------------------------

        ruta_experimento = os.path.join(
            CARPETA_RESULTADOS,
            (
                "configuracion_experimento_"
                f"K_{K}_eps_{epsilon:.5f}.json"
            )
        )


        configuracion_experimento = {
            "epsilon_factible": float(
                epsilon_factible
            ),
            "epsilon_inicial": float(
                epsilon_inicial
            ),
            "epsilon_max": float(
                epsilon_max
            ),
            "paso": float(
                paso
            ),
            "K": int(
                K
            ),
            "M": int(
                M
            ),
            "max_iter_cierre": int(
                MAX_ITER_CIERRE
            ),
            "regiones_utilizadas": (
                REGIONES_UTILIZADAS
            ),
            "cantidad_unidades": int(
                len(R)
            ),
            "poblacion_total": float(
                poblacion_subconjunto
            ),
            "poblacion_objetivo": float(
                poblacion_subconjunto / K
            ),
            "archivo_comunas_original": (
                RUTA_COMUNAS
            ),
            "archivo_distancias": (
                RUTA_DISTANCIAS
            ),
            "archivo_s_nuevo": (
                RUTA_S_NUEVO
            )
        }


        with open(
            ruta_experimento,
            "w",
            encoding="utf-8"
        ) as archivo:

            json.dump(
                configuracion_experimento,
                archivo,
                ensure_ascii=False,
                indent=4
            )


        print(f"\n[OK] LP guardado en:\n{ruta_lp}")

        print(
            "\n[OK] Valores guardados en:\n"
            f"{ruta_valores_json}"
        )

        print(
            "\n[OK] Epsilon guardado en:\n"
            f"{ruta_epsilon}"
        )

        print(
            "\n[OK] Configuración guardada en:\n"
            f"{ruta_experimento}"
        )

        print(
            "\n[FIN] Se encontró una solución óptima. "
            "Se detiene la búsqueda."
        )

        break


    elif status == GRB.INFEASIBLE:

        print(
            f"[NO] Modelo infactible con "
            f"epsilon={epsilon:.5f}"
        )


    elif status == GRB.INF_OR_UNBD:

        print(
            f"[WARN] Modelo infactible o no acotado "
            f"con epsilon={epsilon:.5f}"
        )


    elif status == GRB.TIME_LIMIT:

        print(
            f"[WARN] Se alcanzó el límite de tiempo "
            f"con epsilon={epsilon:.5f}"
        )


    elif status == GRB.INTERRUPTED:

        print(
            f"[WARN] Modelo interrumpido "
            f"con epsilon={epsilon:.5f}"
        )


    else:

        print(
            f"[WARN] Status inesperado: {status}"
        )


    epsilon = round(
        epsilon + paso,
        10
    )


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n==========================================")
print("RESULTADO FINAL")
print("==========================================")


if modelo_factible is None:

    print(
        "\n[FIN] No se encontró un modelo óptimo "
        f"entre epsilon={epsilon_inicial:.5f} "
        f"y epsilon={epsilon_max:.5f}."
    )

else:

    print(
        "\nEpsilon factible encontrado:",
        f"{epsilon_factible:.5f}"
    )

    print("K utilizado:", K)

    print(
        "Cantidad de unidades:",
        len(R)
    )

    print(
        "Población total:",
        f"{poblacion_subconjunto:,.0f}"
    )

    print(
        "Población objetivo por distrito:",
        f"{poblacion_subconjunto / K:,.2f}"
    )

    print("\nRegiones utilizadas:")

    for region in REGIONES_UTILIZADAS:
        print(f"  - {region}")

    print(
        "\nCarpeta de resultados:",
        CARPETA_RESULTADOS
    )