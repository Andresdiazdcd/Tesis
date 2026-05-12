import pandas as pd
import networkx as nx
import numpy as np
import time, json, os, math, csv
import gurobipy as gp

from funciones import (
    obtener_comunas, dist, obtener_region,
    extraer_prob_centros
)
from DataChile.chile_data import regiones
from modelos import modelo_centros_fijos_con_limite
from sampleos import systematic_sampling, pivotal_sampling, sampford_sampling
from funciones_guardado import guardar_resultado_factible

# ============================================================
# CONFIGURACION
# ============================================================
K_centros = 28
TIPO_MODELO = "CL"
T_MAPAS = 100
HORAS_POR_CORRIDA = 168
TIEMPO_MAX = HORAS_POR_CORRIDA * 60 * 60

EPSILON_INICIO = 0.87
EPSILON_FIN = 1.50
EPSILON_PASO = 0.03
EPSILONS = [round(x, 4) for x in np.arange(EPSILON_INICIO, EPSILON_FIN + 1e-9, EPSILON_PASO)]

METODOS = ["sys", "samp", "pivotal"]
LOG_CADA = 50
BASE_RESULTADOS = f"resultados_barrido_epsilon_{TIPO_MODELO}"
os.makedirs(BASE_RESULTADOS, exist_ok=True)

# ============================================================
# DATA Y MODELO BASE
# ============================================================
comunas = pd.read_excel("DataChile/comunas.xlsx")
distancias = pd.read_excel("DataChile/distancias.xlsx")

R = sum([obtener_comunas(comunas, region) for region in regiones], [])

with open("DataChile/adyacencia_comunas.txt", "r") as dict_file:
    dict_ady_com = eval(dict_file.read())

grafo_adyacencia = nx.Graph()
for comuna, adyacencia in dict_ady_com.items():
    for comuna_ady in adyacencia:
        grafo_adyacencia.add_edge(
            comuna, comuna_ady,
            weight=dist(distancias, comuna, comuna_ady)
        )

matrix_adyacencia = nx.adjacency_matrix(grafo_adyacencia)
matriz_ady = np.matrix.round(np.matrix(matrix_adyacencia.toarray()))

with open("DataChile/s_nuevo.txt", "r") as dict_file:
    dict_s = eval(dict_file.read())

comunas_por_region = {}
for comuna in R:
    region = obtener_region(comunas, comuna)
    comunas_por_region.setdefault(region, []).append(comuna)

modelo_con_limite = gp.read("datos_modelo/modelo.lp")
with open("datos_modelo/valores.json", "r") as f:
    valores_raw = json.load(f)

valores = {k.replace(" ", "_"): v for k, v in valores_raw.items()}
print("Modelo leido", flush=True)


def extraer_centros_fijados(modelo, valores):
    centros = []
    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and valores.get(v.VarName) == 1.0:
            texto = v.VarName
            comuna = texto[texto.find("[") + 1:texto.find("]")].replace("_", " ")
            centros.append(comuna)
    return centros


count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(
    modelo_con_limite, K_centros, valores
)
centros_fijados = extraer_centros_fijados(modelo_con_limite, valores)

comunas_t, probabilidades = zip(*centros_frac)
comunas_t = [c.replace("_", " ") for c in comunas_t]
probabilidades = list(probabilidades)

k_sampleo = K_centros - len(centros_fijados)
n = len(comunas_t)
total_combinaciones = math.comb(n, k_sampleo) if 0 <= k_sampleo <= n else None

print(f"Centros fijados por modelo: {len(centros_fijados)}", flush=True)
print(f"Centros a samplear para sys/samp: {k_sampleo}", flush=True)
print(f"Comunas fraccionarias: {n}", flush=True)
print(f"Epsilons: {EPSILONS}", flush=True)

# ============================================================
# HELPERS
# ============================================================

def log_mensaje(ruta_log, mensaje, consola=True, archivo=True, overwrite=False):
    if consola:
        if overwrite:
            print(mensaje, end="\r", flush=True)
        else:
            print(mensaje, flush=True)
    if archivo:
        with open(ruta_log, "a", encoding="utf-8") as f_log:
            f_log.write(mensaje + "\n")


def samplear_centros(metodo):
    if metodo == "sys":
        return systematic_sampling(comunas_t, probabilidades, k_sampleo)

    elif metodo == "samp":
        return sampford_sampling(comunas_t, probabilidades, k_sampleo)

    elif metodo == "pivotal":
        return pivotal_sampling(comunas_t, probabilidades)


def estado_actual(
    intentos, centros_factibles, centros_infactibles,
    centros_factibles_set, centros_infactibles_set,
    centros_observados_set, repetidos, repetidos_consecutivos
):
    exploradas = len(centros_factibles_set) + len(centros_infactibles_set)
    if total_combinaciones:
        porcentaje_explorado = 100 * exploradas / total_combinaciones
        porcentaje_observado = 100 * len(centros_observados_set) / total_combinaciones
        total_txt = f"/{total_combinaciones}"
        pct_txt = f"({porcentaje_observado:.2f}%)"
        pct_exp_txt = f"({porcentaje_explorado:.2f}%)"
    else:
        total_txt = ""
        pct_txt = ""
        pct_exp_txt = ""

    return (
        f"[{time.strftime('%H:%M:%S')}] "
        f"intentos={intentos} | "
        f"obs_distintas={len(centros_observados_set)}{total_txt} {pct_txt} | "
        f"exploradas={exploradas}{total_txt} {pct_exp_txt} | "
        f"factibles={len(centros_factibles)} | "
        f"infactibles={len(centros_infactibles)} | "
        f"repetidos={repetidos} | "
        f"rep_cons={repetidos_consecutivos}"
    )


def correr_metodo_epsilon(metodo, epsilon_1):
    t_inicio = time.time()

    base_resultados = os.path.join(
        BASE_RESULTADOS,
        f"resultados_{TIPO_MODELO}_{metodo}_epsilon_{epsilon_1}"
    )
    os.makedirs(base_resultados, exist_ok=True)

    ruta_centros_factibles = os.path.join(base_resultados, "centros_factibles.txt")
    ruta_centros_infactibles = os.path.join(base_resultados, "centros_infactibles.txt")
    ruta_log = os.path.join(base_resultados, "log_corrida.txt")

    centros_factibles = []
    centros_infactibles = []
    centros_factibles_set = set()
    centros_infactibles_set = set()
    centros_observados_set = set()

    intentos = 0
    repetidos = 0
    repetidos_consecutivos = 0
    errores_gurobi = 0
    errores_otro_tipo = 0
    muestras_tamano_incorrecto = 0

    with open(ruta_log, "w", encoding="utf-8") as f_log:
        f_log.write(f"Inicio corrida: {time.ctime()}\n")
        f_log.write(f"TIPO_MODELO = {TIPO_MODELO}\n")
        f_log.write(f"METODO = {metodo}\n")
        f_log.write(f"epsilon_1 = {epsilon_1}\n")
        f_log.write(f"K_centros = {K_centros}\n")
        f_log.write(f"centros_fijados = {len(centros_fijados)}\n")
        if metodo != "pivotal":
            f_log.write(f"centros_a_samplear = {k_sampleo}\n")
        else:
            f_log.write("centros_a_samplear = NA; pivotal no usa k_sampleo\n")
        f_log.write(f"comunas_fraccionarias = {n}\n")
        f_log.write(f"total_combinaciones_teoricas = {total_combinaciones}\n")
        f_log.write(f"t_mapas_objetivo = {T_MAPAS}\n\n")

    print("\n" + "=" * 70, flush=True)
    print(f"METODO={metodo} | epsilon_1={epsilon_1}", flush=True)
    print("=" * 70, flush=True)

    # LOOP PRINCIPAL
    # -----------------------------
    while len(centros_factibles) < T_MAPAS:
        exploradas = len(centros_factibles_set) + len(centros_infactibles_set)
        if total_combinaciones is not None and metodo != "pivotal" and exploradas == total_combinaciones:
            log_mensaje(ruta_log, "Se exploraron todas las combinaciones posibles.")
            break

        if time.time() - t_inicio >= TIEMPO_MAX:
            log_mensaje(ruta_log, f"Limite de {HORAS_POR_CORRIDA} horas alcanzado.")
            break

        intentos += 1

        try:
            centros_i = samplear_centros(metodo)
            centros_i = [c.replace("_", " ") for c in centros_i]
            centros_i_key = tuple(sorted(centros_i))
            centros_observados_set.add(centros_i_key)

            if centros_i_key in centros_factibles_set or centros_i_key in centros_infactibles_set:
                repetidos += 1
                repetidos_consecutivos += 1
                if intentos % LOG_CADA == 0:
                    log_mensaje(
                        ruta_log,
                        estado_actual(
                            intentos, centros_factibles, centros_infactibles,
                            centros_factibles_set, centros_infactibles_set,
                            centros_observados_set, repetidos, repetidos_consecutivos
                        ),
                        consola=True,
                        archivo=True,
                        overwrite=True
                    )
                continue

            repetidos_consecutivos = 0
            centros_total_i = centros_fijados + list(centros_i)

            # El modelo requiere exactamente K_centros.
            # En pivotal esto sirve como chequeo, porque no se le pasa k_sampleo.
            if len(centros_total_i) != K_centros:
                muestras_tamano_incorrecto += 1
                log_mensaje(
                    ruta_log,
                    f"[SKIP TAMANO] intento={intentos} | centros_total={len(centros_total_i)} | "
                    f"esperado={K_centros} | metodo={metodo}",
                    consola=False,
                    archivo=True
                )
                centros_infactibles_set.add(centros_i_key)
                continue

            modelo_i = modelo_centros_fijos_con_limite(
                epsilon_1, R, centros_total_i, dict_s, comunas, verbose=False
            )

            if not modelo_i:
                centros_infactibles.append(centros_total_i)
                centros_infactibles_set.add(centros_i_key)
                with open(ruta_centros_infactibles, "a", encoding="utf-8") as f:
                    f.write(",".join(centros_total_i) + "\n")
            else:
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
                    "metodo_sampleo": metodo
                }

                guardar_resultado_factible(
                    base_resultados,
                    nombre_resultado,
                    modelo_i,
                    centros_total_i,
                    metadata=metadata
                )

                log_mensaje(
                    ruta_log,
                    f"[OK] {nombre_resultado} | intento={intentos} | "
                    f"factibles={len(centros_factibles)} | "
                    f"infactibles={len(centros_infactibles)} | "
                    f"obs_distintas={len(centros_observados_set)} | "
                    f"repetidos={repetidos}",
                    consola=True,
                    archivo=True
                )

            if intentos % LOG_CADA == 0:
                log_mensaje(
                    ruta_log,
                    estado_actual(
                        intentos, centros_factibles, centros_infactibles,
                        centros_factibles_set, centros_infactibles_set,
                        centros_observados_set, repetidos, repetidos_consecutivos
                    ),
                    consola=True,
                    archivo=True,
                    overwrite=True
                )

        except gp.GurobiError as e:
            errores_gurobi += 1
            log_mensaje(
                ruta_log,
                f"[GUROBI ERROR] intento={intentos} | codigo={getattr(e, 'errno', 'NA')} | mensaje={str(e)}",
                consola=True,
                archivo=True
            )
            continue

        except Exception as e:
            errores_otro_tipo += 1
            log_mensaje(
                ruta_log,
                f"[ERROR GENERAL] intento={intentos} | mensaje={str(e)}",
                consola=True,
                archivo=True
            )
            continue

    tiempo_total = time.time() - t_inicio
    exploradas = len(centros_factibles_set) + len(centros_infactibles_set)
    porcentaje_observado = 100 * len(centros_observados_set) / total_combinaciones if total_combinaciones else None
    porcentaje_explorado = 100 * exploradas / total_combinaciones if total_combinaciones else None

    resumen = {
        "tipo_modelo": TIPO_MODELO,
        "metodo": metodo,
        "epsilon_1": epsilon_1,
        "mapas_factibles": len(centros_factibles),
        "centros_infactibles": len(centros_infactibles),
        "combinaciones_observadas": len(centros_observados_set),
        "combinaciones_exploradas": exploradas,
        "intentos": intentos,
        "repetidos": repetidos,
        "errores_gurobi": errores_gurobi,
        "errores_otro_tipo": errores_otro_tipo,
        "muestras_tamano_incorrecto": muestras_tamano_incorrecto,
        "tiempo_segundos": round(tiempo_total, 2),
        "porcentaje_observado": porcentaje_observado,
        "porcentaje_explorado": porcentaje_explorado,
        "ruta_resultados": base_resultados
    }

    with open(ruta_log, "a", encoding="utf-8") as f_log:
        f_log.write("\nResumen final\n")
        for k, v in resumen.items():
            f_log.write(f"{k}: {v}\n")
        f_log.write(f"Fin corrida: {time.ctime()}\n")

    print("\nResumen final corrida")
    for k, v in resumen.items():
        print(f"{k}: {v}")

    return resumen

# ============================================================
# BARRIDO COMPLETO
# ============================================================

ruta_resumen_global = os.path.join(BASE_RESULTADOS, "resumen_barrido_epsilon.csv")

if os.path.exists(ruta_resumen_global):
    df_res = pd.read_csv(ruta_resumen_global)
    resumenes = df_res.to_dict("records")
else:
    resumenes = []

for metodo in METODOS:
    for epsilon_1 in EPSILONS:

        # saltar si ya se hizo
        ya_hecho = any(
            r["metodo"] == metodo and r["epsilon_1"] == epsilon_1
            for r in resumenes
        )

        if ya_hecho:
            print(f"Saltando {metodo} epsilon={epsilon_1} (ya existe)")
            continue

        resumen = correr_metodo_epsilon(metodo, epsilon_1)
        resumenes.append(resumen)

        # Guardado incremental
        df_res = pd.DataFrame(resumenes)
        df_res.to_csv(ruta_resumen_global, index=False, encoding="utf-8")

print("\nBarrido terminado")
print(f"Resumen global guardado en: {ruta_resumen_global}")