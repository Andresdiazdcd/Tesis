import pandas as pd
import networkx as nx
import numpy as np
from itertools import islice, accumulate
from collections import Counter, defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from gurobipy import GRB
import os, re, json

from sampleos import systematic_sampling, pivotal_sampling

# En este archivo estarán las funciones útiles para trabajar
# Leer datos, ordenar datos, grafos, etc



# CARGAR ARCHIVOS DE QUE SE USARÁN POSTERIORMENTE
# COMUNAS CONTIENE CUT, COMUNA, PROVICINCIA,REGIÓN,SUPERFICIE, POBLACIÓN2017, DENSIDAD, IDH2005, LATITUD Y LONGITUD
# DISTANCIAS ES UNA MATRIZ CON LAS DISTANCIAS ENTRE CADA UNA DE LAS COMUNAS ES DE 347X347
#comunas=pd.read_excel('comunas.xlsx')
#distancias=pd.read_excel('distancias.xlsx')

def codigo_com_cut(comunas, comuna):
  data = comunas.loc[comunas['comuna'] == comuna]
  consulta=data['cut']
  if consulta.empty:
      return None
  else:
      valor = consulta.values[0]
      return valor
  
def codigo_cut_com(comunas, cut):
  data = comunas.loc[comunas['cut'] == cut]
  consulta=data['comuna']
  if consulta.empty:
    return None
  else:
    valor = consulta.values[0]
    return valor
  
# RECIBE UNA REGION Y ENTREGA LAS COMUNAS QUE SON PARTE DE ESA REGION
def obtener_comunas(comunas, region):
    comunasss = comunas.loc[comunas['region'] == region]
    lista_1=[]
    for i in comunasss.comuna:
      lista_1.append(i)
    return lista_1

# RECIBE DOS COMUNAS Y ENTREGA LA DISTANCIA ENTRE ELLAS
def dist(distancias, comuna1, comuna2):
  diss = distancias.loc[distancias['comuna'] == comuna1]
  pregunta= diss[comuna2]
  valor= pregunta.values[0]
  return valor

# RECIBE UNA COMUNA Y ENTREGA LA POBLACIÓN DE ESA COMUNA
def pob(comunas, comuna):
  poblac = comunas.loc[comunas['comuna'] == comuna]
  consulta=poblac['poblacion2017']
  valor= consulta.values[0]
  return valor

# RECIBE UNA LISTA DE COMUNAS Y ENTREGA LA SUMA DE LA POBLACIÓN DE TODAS ESAS COMUNAS
def calcular_poblacion_total(comunas, comunas_lista):
    poblacion_total = 0
    for comuna in comunas_lista:
        pob_total = comunas.loc[comunas['comuna'] == comuna]
        consulta = pob_total['poblacion2017']
        valor = consulta.values[0]
        poblacion_total += valor
    return poblacion_total

# RECIBE UNA COMUNA Y ENTREGA LA REGIÓN A LA CUÁL PERTENECE.
def obtener_region(comunas, comuna):
  data_r= comunas.loc[comunas['comuna'] == comuna]
  preg=data_r['region']
  valor=preg.values[0]
  return valor

# RECIBE UN GRAFO G, UN ORIGEN Y UN DESTINO.
# source y target SON PARTE DE G.
# K ES LA CANTIDAD DE CAMINOS A PEDIR.
# GENERA EL CAMINO MÁS CORTO Y SIMPLE DESDE source a target EN G.
def k_shortest_paths(G, source, target, k, weight=None):
  resultado=list(islice(nx.shortest_simple_paths(G, source, target, weight='weight'),k))
  return resultado

# A PARTIR DEL CAMINO MÁS CORTO, SI EL CAMINO ES MAYOR O IGUAL A 3, 
# DEBE ENTREGARSE SOLAMENTE EL NODO JUSTO ANTES DEL DESTINO
# SINO EL CAMINO SERA VACIO POR QUE LOS NODOS SON ADYACENTES
def Snuevo(grafo_adyacencia, V,u,v):
  H = nx.induced_subgraph(grafo_adyacencia,V)
  for path in k_shortest_paths(H,u,v,1):
      if len(path)>=3:
        return [path[len(path)-2]]
      else:
          return []
      
def extraer_prob_centros(modelo, K):
    centros_frac = []
    centros = []
    count_centros_fijados = 0
    
    
    for v in modelo.getVars():
        if v.VarName.startswith("centros_j") and v.x == 1.0:
            texto_1 = v.VarName
            comuna = texto_1[texto_1.find('[')+1 : texto_1.find(']')]
            centros.append(comuna)
            count_centros_fijados += 1
        elif v.VarName.startswith("centros_j") and v.x > 0.0 and v.x < 1.0:
            texto_1 = v.VarName
            comuna = texto_1[texto_1.find('[')+1 : texto_1.find(']')]
            centros_frac.append((comuna, v.x))

    top_centros_frac = sorted(centros_frac, key=lambda x: x[1], reverse=True)[0:(K-count_centros_fijados)]
    
    print(f"Hay {count_centros_fijados} centros fijados por el Modelo\n")
    print(f"Hay {len(centros_frac)} comunas con peso positivo\n")

    return count_centros_fijados, centros_frac, top_centros_frac

# modelo PL gurobi
# k = número de centros con los que estamos trabajando
# n = número de sampleos
def resultados_systematic(modelo, k, n):

  print(f"----- Resultados para Systematic Sampling -----\n")

  count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(modelo, k)

  # Ordenar de menor a mayor por probabilidad
  probabilidades_ordenadas = sorted(centros_frac, key=lambda x: x[1])
  orden_deseado = [comuna for comuna, _ in probabilidades_ordenadas]

  comunas_top = set([nombre for nombre, _ in top_centros_frac])

  comunas, probabilidades = zip(*centros_frac)
  comunas = list(comunas)
  probabilidades = list(probabilidades)

  comunas_a_samplear = k - count_centros_fijados

  print(f"Se realizarán {n} sampleos para obtener {comunas_a_samplear} comunas en cada sampleo\n")

  sampleos = []

  for i in range(n):
      seleccionados = systematic_sampling(comunas, probabilidades, comunas_a_samplear)
      sampleos.append(seleccionados)

   # Aplanar la lista de comunas seleccionadas (solo queremos contar cuantas veces aparece cada una)
  todas_las_comunas = [comuna for muestra in sampleos for comuna in muestra]
  conteo_systematic = Counter(todas_las_comunas)

  frecuencias_ordenadas_systematic = [conteo_systematic.get(comuna, 0) for comuna in orden_deseado]

  esperanza_empirica_systematic = []

  for comuna, prob in zip(probabilidades_ordenadas, probabilidades_ordenadas):
      comuna_nombre = comuna[0]
      esperanza = conteo_systematic.get(comuna_nombre, 0) / n
      peso_gurobi = prob[1]
      esperanza_empirica_systematic.append((comuna_nombre, esperanza, peso_gurobi))

  esperanza_empirica = pd.DataFrame(esperanza_empirica_systematic,
                                    columns=["Comuna", "Esperanza Empírica", "Peso Gurobi"])


  colores = ['purple' if comuna in comunas_top else 'skyblue' for comuna in orden_deseado]

  plt.figure(figsize=(12, 6))
  bars = plt.bar(orden_deseado, frecuencias_ordenadas_systematic, color=colores)
  plt.xticks(rotation=45, ha='right')
  plt.title("Count comunas Systematic")
  plt.xlabel("Comuna")
  plt.ylabel("Frecuencia")

  top_patch = mpatches.Patch(color='purple', label='Top pesos modelo')
  rest_patch = mpatches.Patch(color='skyblue', label='Resto de comunas')
  plt.legend(handles=[top_patch, rest_patch])

  plt.tight_layout()
  plt.show()

  return frecuencias_ordenadas_systematic, conteo_systematic, esperanza_empirica

   
def resultados_sampleo(modelo, k, n, metodo='systematic'):
    """
    Ejecuta y grafica resultados para sampling sistemático o pivotal.

    Args:
        modelo: salida del modelo que contiene probabilidades
        k: tamaño deseado de muestra total (incluyendo centros fijados)
        n: cantidad de repeticiones del muestreo
        metodo: 'systematic' o 'pivotal'
    """
    print(f"----- Resultados para {metodo.capitalize()} Sampling -----\n")

    count_centros_fijados, centros_frac, top_centros_frac = extraer_prob_centros(modelo, k)

    # Ordenar comunas por probabilidad (de menor a mayor)
    probabilidades_ordenadas = sorted(centros_frac, key=lambda x: x[1])
    orden_deseado = [comuna for comuna, _ in probabilidades_ordenadas]
    comunas_top = set([nombre for nombre, _ in top_centros_frac])

    comunas, probabilidades = zip(*centros_frac)
    comunas = list(comunas)
    probabilidades = list(probabilidades)

    comunas_a_samplear = k - count_centros_fijados
    print(f"Se realizarán {n} sampleos para obtener {comunas_a_samplear} comunas en cada sampleo\n")

    sampleos = []

    for _ in range(n):
        if metodo == 'systematic':
            seleccionados = systematic_sampling(comunas, probabilidades, comunas_a_samplear)
        elif metodo == 'pivotal':
            seleccionados = pivotal_sampling(comunas, probabilidades)
        else:
            raise ValueError("Método no reconocido. Usa 'systematic' o 'pivotal'.")

        sampleos.append(seleccionados)

    # Aplanar lista de todas las comunas seleccionadas
    todas_las_comunas = [comuna for muestra in sampleos for comuna in muestra]
    conteo = Counter(todas_las_comunas)
    frecuencias_ordenadas = [conteo.get(comuna, 0) for comuna in orden_deseado]

    esperanza_empirica = []
    for comuna, prob in zip(probabilidades_ordenadas, probabilidades_ordenadas):
        nombre = comuna[0]
        esperanza = conteo.get(nombre, 0) / n
        peso_gurobi = prob[1]
        esperanza_empirica.append((nombre, esperanza, peso_gurobi))

    esperanza_df = pd.DataFrame(esperanza_empirica,
                                columns=["Comuna", "Esperanza Empírica", "Peso Gurobi"])

    colores = ['purple' if comuna in comunas_top else 'skyblue' for comuna in orden_deseado]

    plt.figure(figsize=(12, 6))
    plt.bar(orden_deseado, frecuencias_ordenadas, color=colores)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Count comunas {metodo.capitalize()}")
    plt.xlabel("Comuna")
    plt.ylabel("Frecuencia")

    top_patch = mpatches.Patch(color='purple', label='Top pesos modelo')
    rest_patch = mpatches.Patch(color='skyblue', label='Resto de comunas')
    plt.legend(handles=[top_patch, rest_patch])

    plt.tight_layout()
    plt.show()

    return frecuencias_ordenadas, conteo, esperanza_df


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_attr(model, name, default=None):
    try:
        return getattr(model, name)
    except Exception:
        return default

# x[i,j]  ->  ("x", "i", "j")  → aquí usamos solo (i, j)
_VAR_PATTERN = re.compile(r'^([A-Za-z_]\w*)\[(.*)\]$')
def parse_x_name(varname: str):
    m = _VAR_PATTERN.match(varname)
    if not m: 
        return None
    base, inside = m.groups()
    if base != 'x': 
        return None
    parts = [p.strip() for p in inside.split(',')]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]

# ==== EXPORTADOR DE UN MODELO =====
def extraer_y_guardar_modelo(
    model, centros, R, comunas, outdir, etiqueta, meta_muestreo=None
):
    """
    Guarda por modelo:
      - .lp, .sol (si hay solución), .ilp (IIS si es infactible)
      - asignaciones x[i,j] completas y x=1 (CSV)
      - cargas de población por centro (CSV)
      - resumen JSON con métricas y metadatos (incluye y*, y^t, K, etc.)
    """
    ensure_dir(outdir)
    prefix = os.path.join(outdir, f"modelo_{etiqueta}")

    # 1) Escribe LP y, si existe, SOL
    try:
        model.write(prefix + ".lp")
    except Exception:
        pass
    try:
        if safe_attr(model, "SolCount", 0) > 0:
            model.write(prefix + ".sol")
    except Exception:
        pass

    # 2) Si es infactible, intenta IIS
    if safe_attr(model, "Status", -1) == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            model.write(prefix + ".ilp")
        except Exception:
            pass

    # 3) Variables x[i,j]
    asignaciones, asignadas_1, xvals = [], [], {}
    for v in model.getVars():
        if not v.VarName.startswith("x["):
            continue
        ij = parse_x_name(v.VarName)
        if ij is None:
            continue
        i, j = ij
        val = float(v.X)
        xvals[(i, j)] = val
        asignaciones.append({"i": i, "j": j, "x": val})
        if abs(val) > 1e-9:
            asignadas_1.append({"i": i, "j": j})

    df_asig  = pd.DataFrame(asignaciones)
    df_asig1 = pd.DataFrame(asignadas_1)
    df_asig.to_csv(prefix + "_asignaciones.csv", index=False)
    df_asig1.to_csv(prefix + "_asignadas_1.csv", index=False)

    # 4) Cargas por centro (usa tu función pob(comunas, i))
    cargas = []
    for j in centros:
        total = 0.0
        for i in R:
            total += pob(comunas, i) * xvals.get((str(i), str(j)), 0.0)
        cargas.append({"centro": str(j), "carga_poblacion": total})
    pd.DataFrame(cargas).to_csv(prefix + "_cargas_por_centro.csv", index=False)

    # 5) Resumen + metadatos del muestreo
    resumen = {
        "etiqueta": etiqueta,
        "status": int(safe_attr(model, "Status", -1)),
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(safe_attr(model, "Status", -1), "OTHER"),
        "obj_val": safe_attr(model, "ObjVal", None),
        "runtime": safe_attr(model, "Runtime", None),
        "num_vars": safe_attr(model, "NumVars", None),
        "num_constrs": safe_attr(model, "NumConstrs", None),
        "centros": list(map(str, centros)),
        "archivos": {
            "lp": prefix + ".lp",
            "sol": prefix + ".sol",
            "iis": prefix + ".ilp",
            "csv_asignaciones": prefix + "_asignaciones.csv",
            "csv_asignadas_1": prefix + "_asignadas_1.csv",
            "csv_cargas": prefix + "_cargas_por_centro.csv",
        },
        "muestreo": meta_muestreo or {},
    }
    with open(prefix + "_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    return resumen

# ==== AGREGADOS PARA EXPERIMENTOS =====
def matriz_X_desde_modelo(model):
    X = {}
    for v in model.getVars():
        if not v.VarName.startswith("x["):
            continue
        ij = parse_x_name(v.VarName)
        if ij is None:
            continue
        val = float(v.X)
        if abs(val) > 1e-9:
            X[ij] = val
    return X

def promedio_X(modelos):
    acc = defaultdict(float)
    T = 0
    for m in modelos:
        if not m or m.Status != GRB.OPTIMAL:
            continue
        X = matriz_X_desde_modelo(m)
        for k, v in X.items():
            acc[k] += v
        T += 1
    if T == 0:
        return {}
    return {k: v / T for k, v in acc.items()}  # \bar X

def comparar_con_baseline(X_bar, X_star):
    claves = set(X_bar.keys()) | set(X_star.keys())
    return {k: X_bar.get(k, 0.0) - X_star.get(k, 0.0) for k in claves}