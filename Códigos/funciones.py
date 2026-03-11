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

def build_matrices_from_gurobi(model, include_bounds=True, core_C_only=True,
                               core_prefixes=('assign[', 'center[', 'centers_total')):
    """
    Extrae matrices del modelo Gurobi y construye:
      - A_leq x <= b_leq  (todas las desigualdades normalizadas a '<=' + bounds opcionales)
      - C x = d           (todas las igualdades, o solo el 'núcleo' si core_C_only=True)

    Parámetros
    ----------
    include_bounds : bool
        Si True, añade límites de variables (LB/UB) como filas en A_leq.
    core_C_only : bool
        Si True, C incluye solo igualdades con nombres que empiezan en core_prefixes.
        Si False, C incluye todas las igualdades del modelo.
    core_prefixes : tuple[str]
        Prefijos para seleccionar el 'núcleo' de C cuando core_C_only=True.

    Retorna
    -------
    A_leq : sp.csr_matrix 
    b_leq : np.ndarray    
    C     : sp.csr_matrix 
    d     : np.ndarray   
    meta  : dict          (metadatos útiles)
    """
    import numpy as np
    import scipy.sparse as sp
    from gurobipy import GRB

    vars_   = model.getVars()
    constrs = model.getConstrs()

    var_names = [v.VarName for v in vars_]
    cnames    = model.getAttr(GRB.Attr.ConstrName, constrs)
    sense     = model.getAttr(GRB.Attr.Sense, constrs)   # '=', '<', '>', 'R'
    rhs       = np.array(model.getAttr(GRB.Attr.RHS, constrs), dtype=float)

    # Rangos si existen
    try:
        ranges = np.array(model.getAttr(GRB.Attr.Range, constrs), dtype=float)
    except Exception:
        ranges = np.zeros_like(rhs)

    A_all = model.getA().tocsr()
    m_all, n = A_all.shape

    # Clasificación por tipo
    idx_E = [i for i, s in enumerate(sense) if s == '=']
    idx_L = [i for i, s in enumerate(sense) if s in ('<', 'L')]
    idx_G = [i for i, s in enumerate(sense) if s in ('>', 'G')]
    idx_R = [i for i, s in enumerate(sense) if s == 'R']

    # ======================
    #  C x = d
    # ======================
    if idx_E:
        C_all = A_all[idx_E, :].tocsr()
        d_all = rhs[idx_E].copy()
        if core_C_only:
            keep = [r for r, k in enumerate(idx_E)
                    if cnames[idx_E[r]] and any(str(cnames[idx_E[r]]).startswith(p) for p in core_prefixes)]
            C = C_all[keep, :].tocsr() if keep else sp.csr_matrix((0, n))
            d = d_all[keep] if keep else np.zeros((0,), dtype=float)
            kept_eq_global_idx = [idx_E[r] for r in keep]
        else:
            C = C_all
            d = d_all
            kept_eq_global_idx = idx_E[:]
    else:
        C = sp.csr_matrix((0, n))
        d = np.zeros((0,), dtype=float)
        kept_eq_global_idx = []

    # ======================
    #  A_leq x <= b_leq
    # ======================
    blocks, b_parts = [], []

    if idx_L:
        blocks.append(A_all[idx_L, :])
        b_parts.append(rhs[idx_L])
    if idx_G:
        blocks.append(-A_all[idx_G, :])
        b_parts.append(-rhs[idx_G])
    if idx_R:
        blocks.append(A_all[idx_R, :])
        b_parts.append(rhs[idx_R])
        blocks.append(-A_all[idx_R, :])
        b_parts.append(-(rhs[idx_R] - ranges[idx_R]))

    A_leq = sp.vstack(blocks, format='csr') if blocks else sp.csr_matrix((0, n))
    b_leq = np.concatenate(b_parts) if b_parts else np.zeros((0,), dtype=float)

    # --- incluir bounds ---
    bound_row_start = A_leq.shape[0]
    added_bound_rows = 0
    if include_bounds:
        bound_rows, bound_rhs = [], []
        INF = 1e100
        for j, v in enumerate(vars_):
            if v.LB is not None and v.LB > -INF:
                bound_rows.append(sp.csr_matrix(([-1.0], ([0], [j])), shape=(1, n)))
                bound_rhs.append(-float(v.LB))
            if v.UB is not None and v.UB < INF:
                bound_rows.append(sp.csr_matrix(([1.0], ([0], [j])), shape=(1, n)))
                bound_rhs.append(float(v.UB))
        if bound_rows:
            B = sp.vstack(bound_rows, format='csr')
            A_leq = sp.vstack([A_leq, B], format='csr')
            b_leq = np.concatenate([b_leq, np.array(bound_rhs, dtype=float)])
            added_bound_rows = B.shape[0]

    # --- mapeo fila original -> A_leq ---
    row_map_leq = [None] * m_all
    pos = 0
    for k in range(m_all):
        if sense[k] in ('<', 'L'):
            row_map_leq[k] = pos; pos += 1
    for k in range(m_all):
        if sense[k] in ('>', 'G'):
            row_map_leq[k] = pos; pos += 1
    for k in range(m_all):
        if sense[k] == 'R':
            row_map_leq[k] = (pos, pos + 1); pos += 2

    # --- meta ---
    meta = dict(
        var_names=var_names,
        constr_names=list(cnames),
        sense=list(sense),
        idx_E=idx_E, idx_L=idx_L, idx_G=idx_G, idx_R=idx_R,
        kept_eq_global_idx=kept_eq_global_idx,
        row_map_leq=row_map_leq,
        A_shape=A_all.shape, Aleq_shape=A_leq.shape, C_shape=C.shape,
        bounds_added=added_bound_rows,
        bounds_start=bound_row_start,
    )

    return A_leq, b_leq, C, d, meta

def delta_b_from_eps(model, b_leq_ref, epsilon_ref, epsilon_new, meta):
    """
    Construye Δb para cambiar el ε de referencia (epsilon_ref) a un nuevo ε (epsilon_new).

    En este modelo:
        pop_up[j]:  sum_i p_i * x[i,j] <= phat * (1+ε) * y[j]
        pop_lo[j]: -sum_i p_i * x[i,j] <= -phat * (1-ε) * y[j]

    Como trabajamos con A_leq x <= b_leq, las filas correspondientes a estas
    restricciones cambian solo en el término independiente (b_leq).

    Al cambiar ε:
        Δb_up_j = phat * (ε_new - ε_ref)
        Δb_lo_j = phat * (ε_new - ε_ref)

    Retorna:
        delta : np.ndarray
            Vector Δb del mismo tamaño que b_leq_ref.
    """

    cnames = meta["constr_names"]
    row_map_leq = meta["row_map_leq"]

    # 1) Detectar filas correspondientes a pop_up / pop_lo
    pop_rows = []
    for k, name in enumerate(cnames):
        if not name:
            continue
        if name.startswith("pop_up[") or name.startswith("pop_lo["):
            r = row_map_leq[k]
            if r is not None:
                pop_rows.append(r)

    delta = np.zeros_like(b_leq_ref, dtype=float)

    if not pop_rows:
        print("No se detectaron restricciones nombradas como pop_up[...] / pop_lo[...].")
        return delta

    # 2) Estimar phat desde el modelo de referencia
    pop_up_vals, pop_lo_vals = [], []
    for k, name in enumerate(cnames):
        if not name:
            continue
        r = row_map_leq[k]
        if r is None:
            continue
        if name.startswith("pop_up["):
            denom = 1.0 + float(epsilon_ref)
            if abs(denom) > 1e-12:
                pop_up_vals.append(float(b_leq_ref[r]) / denom)
        elif name.startswith("pop_lo["):
            denom = 1.0 - float(epsilon_ref)
            if abs(denom) > 1e-12:
                pop_lo_vals.append(-float(b_leq_ref[r]) / denom)

    cand = []
    if pop_up_vals:
        cand.append(np.median(pop_up_vals))
    if pop_lo_vals:
        cand.append(np.median(pop_lo_vals))
    phat_est = float(np.median(cand)) if cand else 0.0

    # 3) Calcular Δb
    delta_val = phat_est * (float(epsilon_new) - float(epsilon_ref))
    for r in pop_rows:
        delta[r] = delta_val

    return delta
