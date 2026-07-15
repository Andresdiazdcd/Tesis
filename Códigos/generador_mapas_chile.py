import os
import json
import time
import math
import pickle
import pandas as pd
import gurobipy as gp
from collections import defaultdict

from funciones import obtener_comunas, extraer_prob_centros
from modelos import modelo_centros_fijos_sin_limite
from sampleos import systematic_sampling, pivotal_sampling, sampford_sampling
from funciones_guardado import guardar_resultado_factible
from data_chile_distrito_censal.chile_data import regiones


# ============================================================
# CONFIG
# ============================================================
# Configuración general para Chile.
# K=28 corresponde al número total de distritos/centros.
# El modelo LP y el JSON vienen del PL ya resuelto.
# Desde ahí se extraen:
#   - centros con y_j = 1
#   - centros fraccionarios 0 < y_j < 1
# ============================================================

CONFIG_CHILE = {
    "region": "chile",
    "K": 28,
    "comunas": "data_chile_distrito_censal/comunas_chile_2024_caso_B_conectado.xlsx",
    "s_nuevo": "data_chile_distrito_censal/s_nuevo_chile_2024_caso_B_conectado.pkl",
    "modelo_lp": "datos_modelo/modelo_chile_censal_eps_0.30000_B_sl_v2.lp",
    "valores_json": "datos_modelo/valores_chile_censal_eps_0.30000_B_sl_v2.json",
}

# Método de sampleo a usar.
# Puedes cambiar a ["pivotal"], ["sampford"] o varios métodos.
METODOS = ["sys"]

# Epsilons que se probarán para resolver los IP con centros fijos.
EPSILONS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

# Número de mapas factibles que quieres generar.
T_MAPAS = 100

# Tiempo máximo total de la fase 2.
HORAS = 60

# Frecuencia de impresión/log.
LOG_CADA = 50

BASE_RESULTADOS = "resultados_chile_censal"
os.makedirs(BASE_RESULTADOS, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def cargar_dict_s(path):
    """
    Carga el diccionario de caminos/intermedios usado por las restricciones
    de contigüidad.

    Se envuelve en defaultdict para que, si falta una llave, no reviente.
    """
    with open(path, "rb") as f:
        dict_s_base = pickle.load(f)

    return defaultdict(lambda: [[]], dict_s_base)


def cargar_valores(path):
    """
    Carga el JSON con los valores de la solución del PL.

    Importante:
    Se reemplazan espacios por "_" para que los nombres coincidan con
    los nombres de variables leídos desde el .lp por Gurobi.
    """
    with open(path, "r", encoding="utf-8") as f:
        valores_raw = json.load(f)

    return {k.replace(" ", "_"): v for k, v in valores_raw.items()}


def centros_fijados_desde_modelo(modelo, valores):
    """
    Extrae los centros que el PL dejó fijados con y_j = 1.

    En tu caso, las variables se llaman:
        centros_j[dist_...]

    Estos centros ya están decididos por el PL y no entran al sampleo.
    """
    centros = []

    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and valores.get(v.VarName, 0) == 1.0:
            comuna = v.VarName[v.VarName.find("[") + 1:v.VarName.find("]")]
            centros.append(comuna)

    return centros


def samplear_centros(metodo, comunas_t, probabilidades, k_sampleo):
    """
    Realiza el sampleo de los centros fraccionarios.

    comunas_t:
        lista de comunas candidatas con 0 < y_j < 1.

    probabilidades:
        valores y_j fraccionarios.

    k_sampleo:
        cantidad de centros restantes a seleccionar.
        En tu caso típico:
            K = 28
            centros_fijados = 24
            k_sampleo = 4
    """

    if metodo == "sys":
        return systematic_sampling(comunas_t, probabilidades, k_sampleo)

    if metodo == "sampford":
        return sampford_sampling(comunas_t, probabilidades, k_sampleo)

    if metodo == "pivotal":
        out = pivotal_sampling(comunas_t, probabilidades)

        # Algunos códigos de pivotal devuelven vector 0/1.
        # Si ocurre eso, se transforma a lista de comunas seleccionadas.
        if len(out) == len(comunas_t) and all(x in [0, 1] for x in out):
            return [c for c, z in zip(comunas_t, out) if z == 1]

        return out

    raise ValueError(f"Método no reconocido: {metodo}")


def regiones_saturadas_por_epsilon(comunas, centros_fijados, K, eps):
    df = comunas.copy()
    df["comuna"] = df["comuna"].astype(str)

    phat = df["poblacion2017"].sum() / K
    lb = phat * (1 - eps)

    df_fijos = df[df["comuna"].isin(centros_fijados)]

    pob_region = df.groupby("region")["poblacion2017"].sum()
    centros_region = df_fijos.groupby("region")["comuna"].count()

    res = pd.DataFrame({
        "poblacion": pob_region,
        "centros_fijos": centros_region
    }).fillna(0)

    res["centros_fijos"] = res["centros_fijos"].astype(int)
    res["max_centros"] = (res["poblacion"] // lb).astype(int)
    res["cupos_restantes"] = res["max_centros"] - res["centros_fijos"]

    saturadas = set(res[res["cupos_restantes"] <= 0].index)

    return saturadas, res


def sampleo_cae_en_region_saturada(centros_i, comunas, regiones_saturadas):
    df = comunas.copy()
    df["comuna"] = df["comuna"].astype(str)

    regiones_sampleo = set(
        df[df["comuna"].isin(centros_i)]["region"]
    )

    malas = regiones_sampleo & regiones_saturadas

    return len(malas) > 0, malas


def buscar_epsilon_minimo_chile(
    R,
    centros_fijados,
    comunas_t,
    probabilidades,
    k_sampleo,
    dict_s,
    comunas,
    metodo,
    max_intentos_por_eps=10
):
    for eps in EPSILONS:
        print(f"\nBuscando factibilidad con epsilon = {eps}", flush=True)

        vistos = set()

        regiones_saturadas, tabla_saturacion = regiones_saturadas_por_epsilon(
            comunas,
            centros_fijados,
            K=len(centros_fijados) + k_sampleo,
            eps=eps
        )

        descartes_region = 0

        print("Regiones saturadas:", sorted(regiones_saturadas), flush=True)

        intento = 0              # sampleos totales (incluyendo descartes)
        intentos_ip = 0          # solo los que llegan al IP

        while intentos_ip < max_intentos_por_eps:

            intento += 1

            centros_i = samplear_centros(
                metodo,
                comunas_t,
                probabilidades,
                k_sampleo
            )

            centros_i = list(centros_i)
            key = tuple(sorted(centros_i))

            cae_saturada, malas = sampleo_cae_en_region_saturada(
                centros_i,
                comunas,
                regiones_saturadas
            )

            if cae_saturada:
                descartes_region += 1
                continue

            if key in vistos:
                continue

            vistos.add(key)

            centros_total = centros_fijados + centros_i

            if len(centros_total) != len(centros_fijados) + k_sampleo:
                continue
            
            intentos_ip += 1
            modelo = modelo_centros_fijos_sin_limite(
                eps,
                R,
                centros_total,
                dict_s,
                comunas,
                verbose=False
            )

            if modelo:
                print(
                    f"[EPSILON FIJO ENCONTRADO] epsilon={eps} | intento={intento}",
                    flush=True
                )
                print(
                    f"IP resueltos={intentos_ip} | "
                    f"descartes_region={descartes_region}",
                    flush=True
                )
                return eps, centros_total, modelo

        print(
            f"No hubo factibilidad con epsilon={eps}. "
            f"Sampleos totales={intento} | "
            f"IP resueltos={intentos_ip} | "
            f"Descartes región={descartes_region}",
            flush=True
        )

    return None, None, None


# ============================================================
# CORRIDA CHILE
# ============================================================

def correr_chile_metodo(config, metodo):

    print("\n" + "=" * 70)
    print(f"CHILE | MÉTODO: {metodo}")
    print("=" * 70)

    comunas = pd.read_excel(config["comunas"])
    dict_s = cargar_dict_s(config["s_nuevo"])

    R_por_region = {}
    for region in sorted(comunas["region"].unique()):
        R_por_region[region] = obtener_comunas(comunas, region)

    R = sum(R_por_region.values(), [])
    K_centros = config["K"]

    modelo_pl = gp.read(config["modelo_lp"])
    valores = cargar_valores(config["valores_json"])

    _, centros_frac, _ = extraer_prob_centros(modelo_pl, K_centros, valores)
    centros_fijados = centros_fijados_desde_modelo(modelo_pl, valores)

    comunas_t, probabilidades = zip(*centros_frac)
    comunas_t = list(comunas_t)
    probabilidades = list(probabilidades)

    k_sampleo = K_centros - len(centros_fijados)

    if k_sampleo < 0:
        raise ValueError("Hay más centros fijados que K.")

    print(f"Nodos R: {len(R)}")
    print(f"K: {K_centros}")
    print(f"Centros fijados: {len(centros_fijados)}")
    print(f"Centros a samplear: {k_sampleo}")
    print(f"Centros fraccionarios: {len(comunas_t)}")

    base_resultados = os.path.join(
        BASE_RESULTADOS,
        f"chile_{metodo}"
    )
    os.makedirs(base_resultados, exist_ok=True)

    ruta_log = os.path.join(base_resultados, "log_corrida.txt")
    ruta_factibles = os.path.join(base_resultados, "centros_factibles.txt")
    ruta_infactibles = os.path.join(base_resultados, "centros_infactibles.txt")

    with open(ruta_log, "w", encoding="utf-8") as f:
        f.write(f"Inicio: {time.ctime()}\n")
        f.write("pais=chile\n")
        f.write(f"metodo={metodo}\n")
        f.write(f"K={K_centros}\n")
        f.write(f"centros_fijados={len(centros_fijados)}\n")
        f.write(f"k_sampleo={k_sampleo}\n")
        f.write(f"epsilons={EPSILONS}\n\n")

    # ========================================================
    # FASE 1: buscar epsilon mínimo
    # ========================================================

    epsilon_chile, centros_iniciales, modelo_inicial = buscar_epsilon_minimo_chile(
        R=R,
        centros_fijados=centros_fijados,
        comunas_t=comunas_t,
        probabilidades=probabilidades,
        k_sampleo=k_sampleo,
        dict_s=dict_s,
        comunas=comunas,
        metodo=metodo,
        max_intentos_por_eps=10
    )

    if epsilon_chile is None:
        print("No se encontró epsilon factible para Chile.")
        return {
            "pais": "chile",
            "metodo": metodo,
            "epsilon_chile": None,
            "mapas_factibles": 0,
            "resultados": base_resultados,
        }

    with open(ruta_log, "a", encoding="utf-8") as f:
        f.write(f"\nEPSILON_CHILE={epsilon_chile}\n\n")

    # ========================================================
    # FASE 2: generar mapas usando epsilon_chile fijo
    # ========================================================

    t_inicio = time.time()
    tiempo_max = HORAS * 60 * 60

    centros_factibles = []
    centros_infactibles = []

    factibles_set = set()
    infactibles_set = set()
    observados_set = set()

    intentos = 0
    repetidos = 0
    errores = 0
    descartes_region = 0

    regiones_saturadas, tabla_saturacion = regiones_saturadas_por_epsilon(
        comunas,
        centros_fijados,
        K=K_centros,
        eps=epsilon_chile
    )

    with open(ruta_log, "a", encoding="utf-8") as f:
        f.write("\nREGIONES SATURADAS:\n")
        f.write(tabla_saturacion.to_string())
        f.write("\n\n")

    centros_factibles.append(centros_iniciales)
    factibles_set.add(tuple(sorted(centros_iniciales)))

    guardar_resultado_factible(
        base_resultados,
        "t_001",
        modelo_inicial,
        centros_iniciales,
        metadata={
            "pais": "chile",
            "epsilon_chile": epsilon_chile,
            "K_centros": K_centros,
            "cantidad_centros": len(centros_iniciales),
            "intento": 0,
            "metodo_sampleo": metodo,
            "origen": "muestra_que_fijo_epsilon"
        }
    )

    with open(ruta_factibles, "a", encoding="utf-8") as f:
        f.write(
            f"epsilon_chile={epsilon_chile} | "
            + ",".join(centros_iniciales)
            + "\n"
        )

    print(f"[OK] t_001 | epsilon_chile={epsilon_chile}", flush=True)

    while len(centros_factibles) < T_MAPAS:

        if time.time() - t_inicio >= tiempo_max:
            print("Límite de tiempo alcanzado.")
            break

        intentos += 1

        try:
            centros_i = samplear_centros(
                metodo,
                comunas_t,
                probabilidades,
                k_sampleo
            )

            centros_i = list(centros_i)
            centros_total = centros_fijados + centros_i

            cae_saturada, malas = sampleo_cae_en_region_saturada(
                centros_i,
                comunas,
                regiones_saturadas
            )

            if cae_saturada:
                descartes_region += 1

                with open(ruta_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"[DESCARTE REGION] intento={intentos} | "
                        f"regiones={sorted(malas)} | "
                        f"centros_sampleados={centros_i}\n"
                    )

                continue

            centros_key = tuple(sorted(centros_total))
            observados_set.add(centros_key)

            if centros_key in factibles_set or centros_key in infactibles_set:
                repetidos += 1
                continue

            if len(centros_total) != K_centros:
                with open(ruta_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"[ERROR] intento={intentos}: "
                        f"centros={len(centros_total)} != K={K_centros}\n"
                    )
                continue

            modelo_i = modelo_centros_fijos_sin_limite(
                epsilon_chile,
                R,
                centros_total,
                dict_s,
                comunas,
                verbose=False
            )

            if not modelo_i:
                centros_infactibles.append(centros_total)
                infactibles_set.add(centros_key)

                with open(ruta_infactibles, "a", encoding="utf-8") as f:
                    f.write(",".join(centros_total) + "\n")

                continue

            centros_factibles.append(centros_total)
            factibles_set.add(centros_key)

            nombre_resultado = f"t_{len(centros_factibles):03d}"

            guardar_resultado_factible(
                base_resultados,
                nombre_resultado,
                modelo_i,
                centros_total,
                metadata={
                    "pais": "chile",
                    "epsilon_chile": epsilon_chile,
                    "K_centros": K_centros,
                    "cantidad_centros": len(centros_total),
                    "intento": intentos,
                    "metodo_sampleo": metodo,
                }
            )

            with open(ruta_factibles, "a", encoding="utf-8") as f:
                f.write(
                    f"epsilon_chile={epsilon_chile} | "
                    + ",".join(centros_total)
                    + "\n"
                )

            msg = (
                f"[OK] chile {nombre_resultado} | "
                f"epsilon_chile={epsilon_chile} | "
                f"intento={intentos} | "
                f"factibles={len(centros_factibles)} | "
                f"infactibles={len(centros_infactibles)} | "
                f"descartes_region={descartes_region} | "
                f"repetidos={repetidos}"
            )

            print(msg, flush=True)

            with open(ruta_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            if intentos % LOG_CADA == 0:
                estado = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"intentos={intentos} | "
                    f"obs_distintas={len(observados_set)} | "
                    f"factibles={len(centros_factibles)} | "
                    f"infactibles={len(centros_infactibles)} | "
                    f"descartes_region={descartes_region} | "
                    f"repetidos={repetidos}"
                )
                print(estado, flush=True)

        except gp.GurobiError as e:
            errores += 1
            print(f"[GUROBI ERROR] {e}", flush=True)

        except Exception as e:
            errores += 1
            print(f"[ERROR] intento={intentos} | {e}", flush=True)

    resumen = {
        "pais": "chile",
        "metodo": metodo,
        "epsilon_chile": epsilon_chile,
        "mapas_factibles": len(centros_factibles),
        "infactibles": len(centros_infactibles),
        "descartes_region_saturada": descartes_region,
        "observados": len(observados_set),
        "intentos": intentos,
        "repetidos": repetidos,
        "errores": errores,
        "resultados": base_resultados,
    }

    pd.DataFrame([resumen]).to_csv(
        os.path.join(base_resultados, "resumen.csv"),
        index=False
    )

    print("Resumen:", resumen)

    return resumen


# ============================================================
# RUN GLOBAL
# ============================================================

resumenes = []

for metodo in METODOS:

    carpeta_resultado = os.path.join(
        BASE_RESULTADOS,
        f"chile_{metodo}"
    )

    resumen_path = os.path.join(carpeta_resultado, "resumen.csv")

    if os.path.exists(resumen_path):
        df_prev = pd.read_csv(resumen_path)

        if (
            len(df_prev) > 0
            and df_prev.loc[0, "mapas_factibles"] >= T_MAPAS
        ):
            print(f"[SKIP] chile_{metodo} ya completo.")
            resumenes.append(df_prev.iloc[0].to_dict())
            continue

    resumenes.append(
        correr_chile_metodo(CONFIG_CHILE, metodo)
    )

df_resumen = pd.DataFrame(resumenes)

df_resumen.to_csv(
    os.path.join(BASE_RESULTADOS, "resumen_global.csv"),
    index=False
)

print("\nProceso completo.")
print(df_resumen)