import os, json, time
import pandas as pd
import gurobipy as gp

from funciones import obtener_comunas, extraer_prob_centros
from modelos import modelo_centros_fijos_sin_limite
from sampleos import systematic_sampling, pivotal_sampling, sampford_sampling
from funciones_guardado import guardar_resultado_factible


# ============================================================
# CONFIG
# ============================================================

ESTADOS = {
    "ia": {
        "region": "iowa",
        "K": 4,
        "comunas": "DataEEUU/data_eeuu_procesada/comunas_ia.xlsx",
        "s_nuevo": "DataEEUU/data_eeuu_procesada/s_nuevo_ia.txt",
        "modelo_lp": "datos_modelo/modelo_ia.lp",
        "valores_json": "datos_modelo/valores_ia.json",
    },
    "wi": {
        "region": "wisconsin",
        "K": 8,
        "comunas": "DataEEUU/data_eeuu_procesada/comunas_wi.xlsx",
        "s_nuevo": "DataEEUU/data_eeuu_procesada/s_nuevo_wi.txt",
        "modelo_lp": "datos_modelo/modelo_wi.lp",
        "valores_json": "datos_modelo/valores_wi.json",
    },
    "pa": {
        "region": "pennsylvania",
        "K": 17,
        "comunas": "DataEEUU/data_eeuu_procesada/comunas_pa.xlsx",
        "s_nuevo": "DataEEUU/data_eeuu_procesada/s_nuevo_pa.txt",
        "modelo_lp": "datos_modelo/modelo_pa.lp",
        "valores_json": "datos_modelo/valores_pa.json",
    },
}

METODOS = ["sys"]

EPSILONS = (
    [0.00001, 0.0001, 0.001, 0.01] +
    [round(i / 100, 2) for i in range(2, 81)]
)

T_MAPAS = 100
HORAS = 60
LOG_CADA = 50

BASE_RESULTADOS = "resultados_eeuu_eps_fijo"
os.makedirs(BASE_RESULTADOS, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def cargar_dict_s(path):
    with open(path, "r", encoding="utf-8") as f:
        return eval(f.read())


def cargar_valores(path):
    with open(path, "r", encoding="utf-8") as f:
        valores_raw = json.load(f)

    return {k.replace(" ", "_"): v for k, v in valores_raw.items()}


def centros_fijados_desde_modelo(modelo, valores):
    centros = []

    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and valores.get(v.VarName, 0) == 1.0:
            comuna = v.VarName[v.VarName.find("[") + 1:v.VarName.find("]")]
            centros.append(comuna)

    return centros


def samplear_centros(metodo, comunas_t, probabilidades, k_sampleo):
    if metodo == "sys":
        return systematic_sampling(comunas_t, probabilidades, k_sampleo)

    if metodo == "sampford":
        return sampford_sampling(comunas_t, probabilidades, k_sampleo)

    if metodo == "pivotal":
        out = pivotal_sampling(comunas_t, probabilidades)

        if len(out) == len(comunas_t) and all(x in [0, 1] for x in out):
            return [c for c, z in zip(comunas_t, out) if z == 1]

        return out

    raise ValueError(f"Método no reconocido: {metodo}")


def buscar_epsilon_minimo_estado(
    R,
    centros_fijados,
    comunas_t,
    probabilidades,
    k_sampleo,
    dict_s,
    comunas,
    metodo,
    max_intentos_por_eps=200
):
    """
    Busca el menor epsilon del estado que logra al menos una muestra factible.
    Apenas encuentra factibilidad, retorna ese epsilon.
    """

    for eps in EPSILONS:
        print(f"\nBuscando factibilidad con epsilon = {eps}", flush=True)

        vistos = set()

        for intento in range(1, max_intentos_por_eps + 1):

            centros_i = samplear_centros(
                metodo,
                comunas_t,
                probabilidades,
                k_sampleo
            )

            centros_i = list(centros_i)
            key = tuple(sorted(centros_i))

            if key in vistos:
                continue

            vistos.add(key)

            centros_total = centros_fijados + centros_i

            if len(centros_total) != len(centros_fijados) + k_sampleo:
                continue

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
                    f"[EPSILON FIJO ENCONTRADO] epsilon={eps} | "
                    f"intento={intento}",
                    flush=True
                )
                return eps, centros_total, modelo

        print(
            f"No hubo factibilidad con epsilon={eps} "
            f"en {max_intentos_por_eps} muestras distintas.",
            flush=True
        )

    return None, None, None


# ============================================================
# CORRIDA POR ESTADO Y MÉTODO
# ============================================================

def correr_estado_metodo(sigla, config, metodo):

    print("\n" + "=" * 70)
    print(f"ESTADO: {sigla.upper()} | MÉTODO: {metodo}")
    print("=" * 70)

    comunas = pd.read_excel(config["comunas"])
    dict_s = cargar_dict_s(config["s_nuevo"])
    R = obtener_comunas(comunas, config["region"])
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
        raise ValueError(f"{sigla}: hay más centros fijados que K.")

    print(f"Nodos R: {len(R)}")
    print(f"K: {K_centros}")
    print(f"Centros fijados: {len(centros_fijados)}")
    print(f"Centros a samplear: {k_sampleo}")
    print(f"Centros fraccionarios: {len(comunas_t)}")

    base_resultados = os.path.join(
        BASE_RESULTADOS,
        f"{sigla}_{metodo}"
    )
    os.makedirs(base_resultados, exist_ok=True)

    ruta_log = os.path.join(base_resultados, "log_corrida.txt")
    ruta_factibles = os.path.join(base_resultados, "centros_factibles.txt")
    ruta_infactibles = os.path.join(base_resultados, "centros_infactibles.txt")

    with open(ruta_log, "w", encoding="utf-8") as f:
        f.write(f"Inicio: {time.ctime()}\n")
        f.write(f"estado={sigla}\n")
        f.write(f"region={config['region']}\n")
        f.write(f"metodo={metodo}\n")
        f.write(f"K={K_centros}\n")
        f.write(f"centros_fijados={len(centros_fijados)}\n")
        f.write(f"k_sampleo={k_sampleo}\n")
        f.write(f"epsilons={EPSILONS}\n\n")

    # ========================================================
    # FASE 1: buscar epsilon mínimo fijo del estado
    # ========================================================

    epsilon_estado, centros_iniciales, modelo_inicial = buscar_epsilon_minimo_estado(
        R=R,
        centros_fijados=centros_fijados,
        comunas_t=comunas_t,
        probabilidades=probabilidades,
        k_sampleo=k_sampleo,
        dict_s=dict_s,
        comunas=comunas,
        metodo=metodo,
        max_intentos_por_eps=200
    )

    if epsilon_estado is None:
        print(f"No se encontró epsilon factible para {sigla}.")
        return {
            "estado": sigla,
            "metodo": metodo,
            "epsilon_estado": None,
            "mapas_factibles": 0,
            "resultados": base_resultados,
        }

    with open(ruta_log, "a", encoding="utf-8") as f:
        f.write(f"\nEPSILON_ESTADO={epsilon_estado}\n\n")

    # ========================================================
    # FASE 2: generar mapas usando SIEMPRE epsilon_estado
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

    # guardar la muestra que encontró el epsilon
    centros_factibles.append(centros_iniciales)
    factibles_set.add(tuple(sorted(centros_iniciales)))

    guardar_resultado_factible(
        base_resultados,
        "t_001",
        modelo_inicial,
        centros_iniciales,
        metadata={
            "estado": sigla,
            "region": config["region"],
            "epsilon_estado": epsilon_estado,
            "K_centros": K_centros,
            "cantidad_centros": len(centros_iniciales),
            "intento": 0,
            "metodo_sampleo": metodo,
            "origen": "muestra_que_fijo_epsilon"
        }
    )

    with open(ruta_factibles, "a", encoding="utf-8") as f:
        f.write(
            f"epsilon_estado={epsilon_estado} | "
            + ",".join(centros_iniciales)
            + "\n"
        )

    print(f"[OK] t_001 | epsilon_estado={epsilon_estado}", flush=True)

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

            centros_i_key = tuple(sorted(centros_total))
            observados_set.add(centros_i_key)

            if centros_i_key in factibles_set or centros_i_key in infactibles_set:
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
                epsilon_estado,
                R,
                centros_total,
                dict_s,
                comunas,
                verbose=False
            )

            if not modelo_i:
                centros_infactibles.append(centros_total)
                infactibles_set.add(centros_i_key)

                with open(ruta_infactibles, "a", encoding="utf-8") as f:
                    f.write(",".join(centros_total) + "\n")

                continue

            centros_factibles.append(centros_total)
            factibles_set.add(centros_i_key)

            nombre_resultado = f"t_{len(centros_factibles):03d}"

            guardar_resultado_factible(
                base_resultados,
                nombre_resultado,
                modelo_i,
                centros_total,
                metadata={
                    "estado": sigla,
                    "region": config["region"],
                    "epsilon_estado": epsilon_estado,
                    "K_centros": K_centros,
                    "cantidad_centros": len(centros_total),
                    "intento": intentos,
                    "metodo_sampleo": metodo,
                }
            )

            with open(ruta_factibles, "a", encoding="utf-8") as f:
                f.write(
                    f"epsilon_estado={epsilon_estado} | "
                    + ",".join(centros_total)
                    + "\n"
                )

            msg = (
                f"[OK] {sigla} {nombre_resultado} | "
                f"epsilon_estado={epsilon_estado} | "
                f"intento={intentos} | "
                f"factibles={len(centros_factibles)} | "
                f"infactibles={len(centros_infactibles)} | "
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
        "estado": sigla,
        "region": config["region"],
        "metodo": metodo,
        "epsilon_estado": epsilon_estado,
        "mapas_factibles": len(centros_factibles),
        "infactibles": len(centros_infactibles),
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

for sigla, config in ESTADOS.items():
    for metodo in METODOS:
        resumenes.append(
            correr_estado_metodo(sigla, config, metodo)
        )

df_resumen = pd.DataFrame(resumenes)

df_resumen.to_csv(
    os.path.join(BASE_RESULTADOS, "resumen_global.csv"),
    index=False
)

print("\nProceso completo.")
print(df_resumen)