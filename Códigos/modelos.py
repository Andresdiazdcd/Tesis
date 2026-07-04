from gurobipy import *
from gurobipy import Model
from funciones import calcular_poblacion_total, obtener_region, pob, codigo_com_cut, codigo_cut_com

import time
from datetime import datetime
import geopandas as gpd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import scipy as sp
from collections import defaultdict

def modelo_con_limite_con_obj(epsilon, R, K, dict_s, comunas, dist_dict):
    model = Model("Modelo Con Límite Regional")
    model.setParam("Method", 0)
    model.setParam("Threads", 1)

    phat = calcular_poblacion_total(comunas, R) / K

    # Variables
    x = model.addVars([(i, j) for i in R for j in R], vtype=GRB.CONTINUOUS, name="asignaciones_ij")
    y = model.addVars(R, vtype=GRB.CONTINUOUS, name="centros_j")

    # esto no optimiza nada, solamente ve si hay factibilidad y encuentra solución
    model.setObjective(
    quicksum(dist_dict[(i, j)] * x[i, j] for i in R for j in R),
    GRB.MINIMIZE
)

    # Restricciones
    # si no es la misma región, no se puede asignar
    for i in R:
        for j in R:
            if obtener_region(comunas, i) != obtener_region(comunas, j):
                model.addConstr(x[i, j] == 0, name=f"block[{i},{j}]")

    # Balance poblacional 
    for j in R:
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) <= phat * (1 + epsilon) * y[j],
                        name=f"pop_up[{j}]")
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) >= phat * (1 - epsilon) * y[j],
                        name=f"pop_lo[{j}]")
        model.addConstr(x[j, j] == y[j], name=f"center[{j}]")

    # aseguramos la cantidad de centros
    model.addConstr(quicksum(y[j] for j in R) == K, name="centers_total")

    # Cada comuna debe ser asignada completamente a algún centro
    for i in R:
        model.addConstr(quicksum(x[i, j] for j in R) == 1, name=f"assign[{i}]")
        for j in R:
            model.addConstr(x[i, j] <= y[j], name=f"link[{i},{j}]")

            # Restricción de contigüidad
            if obtener_region(comunas, i) == obtener_region(comunas, j):
                aux_s = dict_s[(j, i)]
                while not aux_s == [[]]:
                    for k in aux_s:
                        model.addConstr(quicksum(x[k[0], j] for k in aux_s) >= x[i, j],
                                        name=f"path[{i},{j},{k[0]}]")
                        aux_s = dict_s[(j, k[0])]
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        resultado = []
        for j in R:
            valor = y[j].x
            if valor > 0:
                resultado.append((j, valor))
                print(f"{j}: {valor:.4f}")
        return model
    else:
        print("Modelo infactible")
        return None



def modelo_con_limite(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Con Límite Regional")
    model.setParam("Method", 2)
    #model.setParam("Threads", 1)

    phat = calcular_poblacion_total(comunas, R) / K

    # Variables
    x = model.addVars([(i, j) for i in R for j in R], vtype=GRB.CONTINUOUS, name="asignaciones_ij")
    y = model.addVars(R, vtype=GRB.CONTINUOUS, name="centros_j")

    # esto no optimiza nada, solamente ve si hay factibilidad y encuentra solución
    model.setObjective(0, GRB.MINIMIZE)

    # Restricciones
    # si no es la misma región, no se puede asignar
    for i in R:
        for j in R:
            if obtener_region(comunas, i) != obtener_region(comunas, j):
                model.addConstr(x[i, j] == 0, name=f"block[{i},{j}]")

    # Balance poblacional 
    for j in R:
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) <= phat * (1 + epsilon) * y[j],
                        name=f"pop_up[{j}]")
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) >= phat * (1 - epsilon) * y[j],
                        name=f"pop_lo[{j}]")
        model.addConstr(x[j, j] == y[j], name=f"center[{j}]")

    # aseguramos la cantidad de centros
    model.addConstr(quicksum(y[j] for j in R) == K, name="centers_total")

    # Cada comuna debe ser asignada completamente a algún centro
    for i in R:
        model.addConstr(quicksum(x[i, j] for j in R) == 1, name=f"assign[{i}]")
        for j in R:
            model.addConstr(x[i, j] <= y[j], name=f"link[{i},{j}]")

            # Restricción de contigüidad
            if obtener_region(comunas, i) == obtener_region(comunas, j):
                aux_s = dict_s[(j, i)]

                while aux_s != [[]]:
                    k = aux_s[0][0]

                    model.addConstr(
                        x[k, j] >= x[i, j],
                        name=f"path[{i},{j},{k}]"
                    )

                    aux_s = dict_s[(j, k)]
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        resultado = []
        for j in R:
            valor = y[j].x
            if valor > 0:
                resultado.append((j, valor))
                print(f"{j}: {valor:.4f}")
        return model
    else:
        print("Modelo infactible")
        return None
    


def modelo_con_limite_opti(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Con Límite Regional")

    model.setParam("Method", 1)
    model.setParam("Threads", 6)
    #model.setParam("SoftMemLimit", 15.0)
    model.setParam("OutputFlag", 1)

    region = {i: obtener_region(comunas, i) for i in R}
    pobl = {i: pob(comunas, i) for i in R}

    phat = sum(pobl[i] for i in R) / K

    # Solo pares dentro de la misma región
    Xij = [(i, j) for i in R for j in R if region[i] == region[j]]

    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # Índices inversos
    I_por_j = {j: [] for j in R}
    J_por_i = {i: [] for i in R}

    for i, j in Xij:
        I_por_j[j].append(i)
        J_por_i[i].append(j)

    # Balance poblacional
    for j in R:

        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in I_por_j[j]
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # Asignación
    for i in R:

        model.addConstr(
            quicksum(x[i, j] for j in J_por_i[i]) == 1,
            name=f"assign[{i}]"
        )

        for j in J_por_i[i]:

            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # Contigüidad
    for i in R:
        for j in J_por_i[i]:

            if i == j:
                continue

            aux_s = dict_s[(j, i)]

            while aux_s != [[]]:

                k = aux_s[0][0]

                # Protección por si falta alguna clave
                if (k, j) in x:

                    model.addConstr(
                        x[k, j] >= x[i, j],
                        name=f"path[{i},{j},{k}]"
                    )

                aux_s = dict_s[(j, k)]

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        return model

    if model.status == GRB.Status.MEM_LIMIT:
        print("FALTA MEMORIA RAM")
    else:
        print("Modelo no óptimo. Status:", model.status)

    return None


def modelo_sin_limite_sparse(
    epsilon,
    R,
    K,
    dict_s,
    comunas,
    distancias,
    M=60,
    usar_contiguidad=True
):
    """
    Modelo sin límite regional sparse.

    Idea:
    - Parte con los M centros más cercanos para cada unidad i.
    - Luego hace cierre por caminos:
        si permito x[i,j] y el camino j -> i requiere k,
        entonces también permito x[k,j].
    - Esto evita bloquear artificialmente asignaciones que en el modelo denso sí existirían.
    """

    start_time = time.time()

    model = Model("Modelo Sin Límite Sparse con Cierre")

    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("Threads", 8)
    model.setParam("SoftMemLimit", 20.0)
    model.setParam("OutputFlag", 1)

    print("Modelo: SIN límite regional sparse con cierre")
    print("K:", K)
    print("M inicial:", M)
    print("Usar contigüidad:", usar_contiguidad)

    # -----------------------------------------------------
    # 1. Población
    # -----------------------------------------------------
    pobl = {i: pob(comunas, i) for i in R}
    phat = sum(pobl[i] for i in R) / K

    print("Población total:", sum(pobl.values()))
    print("phat:", phat)
    print("Rango:", phat * (1 - epsilon), phat * (1 + epsilon))

    dist_df = distancias.set_index("comuna")

    # -----------------------------------------------------
    # 2. Candidatos iniciales: M más cercanos
    # -----------------------------------------------------
    J_por_i = {}

    for i in R:
        candidatos = (
            dist_df.loc[i, R]
            .sort_values()
            .index
            .tolist()
        )

        J = candidatos[:M]

        # Necesario para que exista x[i,i]
        if i not in J:
            J.append(i)

        J_por_i[i] = set(J)

    pares_iniciales = sum(len(J_por_i[i]) for i in R)
    print("Pares iniciales:", pares_iniciales)

    # -----------------------------------------------------
    # 3. Cierre por caminos
    # -----------------------------------------------------
    # Diferencia clave con la versión anterior:
    # antes, si faltaba x[k,j], se imponía x[i,j] = 0.
    # ahora agregamos j como candidato de k.
    #
    # Esto mantiene la lógica del modelo denso:
    # si i puede asignarse a j, los intermedios del camino
    # también pueden asignarse a j.
    # -----------------------------------------------------
    if usar_contiguidad:
        print("Aplicando cierre por caminos...")

        cambios = True
        n_agregados = 0
        iter_cierre = 0

        while cambios:
            cambios = False
            iter_cierre += 1
            agregados_iter = 0

            for i in R:
                for j in list(J_por_i[i]):

                    if i == j:
                        continue

                    aux_s = dict_s[(j, i)]

                    while aux_s != [[]]:
                        k = aux_s[0][0]

                        # Si el camino requiere k, debe existir x[k,j].
                        if j not in J_por_i[k]:
                            J_por_i[k].add(j)
                            n_agregados += 1
                            agregados_iter += 1
                            cambios = True

                        aux_s = dict_s[(j, k)]

            print(
                f"  iter cierre {iter_cierre}: "
                f"agregados={agregados_iter}"
            )

        print("Candidatos agregados por cierre:", n_agregados)

    # Volver a listas ordenadas
    J_por_i = {i: sorted(J_por_i[i]) for i in R}

    Xij = [(i, j) for i in R for j in J_por_i[i]]

    print("Variables x:", len(Xij))
    print("Variables y:", len(R))
    print("Pares densos serían:", len(R) * len(R))

    # -----------------------------------------------------
    # 4. Variables
    # -----------------------------------------------------
    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # -----------------------------------------------------
    # 5. Índice inverso
    # -----------------------------------------------------
    I_por_j = {j: [] for j in R}

    for i, j in Xij:
        I_por_j[j].append(i)

    # -----------------------------------------------------
    # 6. Balance poblacional y centro
    # -----------------------------------------------------
    for j in R:
        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in I_por_j[j]
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # -----------------------------------------------------
    # 7. Asignación y link
    # -----------------------------------------------------
    for i in R:
        model.addConstr(
            quicksum(x[i, j] for j in J_por_i[i]) == 1,
            name=f"assign[{i}]"
        )

        for j in J_por_i[i]:
            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # -----------------------------------------------------
    # 8. Contigüidad
    # -----------------------------------------------------
    n_path = 0
    n_block = 0

    if usar_contiguidad:
        print("Agregando restricciones de contigüidad...")

        for i in R:
            for j in J_por_i[i]:

                if i == j:
                    continue

                aux_s = dict_s[(j, i)]

                while aux_s != [[]]:
                    k = aux_s[0][0]

                    if (k, j) in x:
                        model.addConstr(
                            x[k, j] >= x[i, j],
                            name=f"path[{i},{j},{k}]"
                        )
                        n_path += 1
                    else:
                        # Esto idealmente debería ser 0 después del cierre.
                        n_block += 1
                        break

                    aux_s = dict_s[(j, k)]

    print("Restricciones path:", n_path)
    print("Faltantes después del cierre:", n_block)

    # -----------------------------------------------------
    # 9. Diagnóstico
    # -----------------------------------------------------
    model.update()

    print("NumVars:", model.NumVars)
    print("NumConstrs:", model.NumConstrs)

    # -----------------------------------------------------
    # 10. Optimizar
    # -----------------------------------------------------
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(f"[OK] Tiempo: {time.time() - start_time:.2f} segundos")
        return model

    if model.status == GRB.Status.MEM_LIMIT:
        print("FALTA MEMORIA RAM")
    else:
        print("Modelo no óptimo. Status:", model.status)

    return None

def modelo_relajado(epsilon, R, K, comunas, y_star):

    model = Model("modelo_relajado")
    model.Params.OutputFlag = 0

    # Promedio poblacional
    phat = calcular_poblacion_total(comunas, R) / K

    # Variables
    z = model.addVars([(i, j) for i in R for j in R],
                      vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # Objetivo
    model.setObjective(0, GRB.MINIMIZE)

    # 1) Asignación completa
    for i in R:
        model.addConstr(quicksum(z[i, j] for j in R) == 1, name=f"assign[{i}]")

    # 2) Balance poblacional con pesos y*_j
    for j in R:
        yj = y_star.get(j, 0.0)
        model.addConstr(
            quicksum(pob(comunas, i) * z[i, j] for i in R)
            <= phat * (1 + epsilon) * yj,
            name=f"pop_up[{j}]"
        )
        model.addConstr(
            quicksum(pob(comunas, i) * z[i, j] for i in R)
            >= phat * (1 - epsilon) * yj,
            name=f"pop_lo[{j}]"
        )

    # 3) z_ij \leq y*_j
    for i in R:
        for j in R:
            yj = y_star.get(j, 0.0)
            model.addConstr(z[i, j] <= yj, name=f"link[{i},{j}]")

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print("Modelo relajado factible")
        return model
    else:
        print("Modelo relajado infactible")
        return None

    
def modelo_IP(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Sin Límite")

    #model.setParam("Method", 0)
    #model.setParam("Threads", 1)
    start_time = time.time()
    print("La cantidad de centros es",K )
    #Se calcula la población promedio
    phat= calcular_poblacion_total(comunas, R)/K

    # se generan los parametros i,j
    Xij = [(i,j) for i in R for j in R]
    Yj = [(j) for j in R]

    #se crea la variable, xij si es que i pertenece al distrito con centro j
    x= model.addVars(Xij,vtype=GRB.BINARY,name="asignaciones_ij")
    y = model.addVars(Yj, vtype=GRB.BINARY, name="centros_j")

    model.setObjective(0,GRB.MINIMIZE)

    for j in R:
        # Balance de población, permitiendo una diferencia de 1+-epsilon
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) <= phat * (1 + epsilon)*y[j])
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) >= phat * (1 - epsilon)*y[j])
        #Cuando un comuna es centro, debe estar asignada al distrito del cuál es centro
        model.addConstr((x[j, j]) == y[j])
        #Los centros deben ser igual a K (un parametro fijo y dado para el modelo)
        model.addConstr(quicksum(y[j] for j in R) == K)

    for i in R:
        #Todas las comunas deben estar asignadas a algún centro
        model.addConstr(quicksum(x[i,j] for j in R) == 1)
        for j in R:
          #Asigno comunas a un centro, solo si esa comuna esta abierta como centro
          model.addConstr(x[i,j]<= y[j])
          # Restricción de contiguidad
          # Se consulta en el diccionario dict_s (donde esta guardado shortest simple path)
          # Si es que las comunas no son adyacentes, se pide que para que i y j sean asignados
          # i y k deben estar obligatoriamente asignadas, porque para poder de i a j, k está en el camino
          aux_s = dict_s[(j, i)]
          while not aux_s == [[]]:
              for k in aux_s:
                  model.addConstr(quicksum(x[k[0], j] for k in aux_s) >= x[i, j])
                  aux_s = dict_s[(j, k[0])]

    model.optimize()
    end_time = time.time()
    #model.computeIIS()
    #model.write('iismodel.ilp')
    #Si es que el modelo es factible, se imprimen algunos resultados.
    #if model.status == GRB.Status.OPTIMAL:
    #    duration = end_time - start_time
    #    print(f"El código se ejecutó en {duration:.2f} segundos")
    #    asignacion = []
    #    asignacion_value=[]
    #    for i in model.getVars():
    #      if i.x > 0:
    #        print(i.VarName,i.x)
    #        if "asignaciones_ij" in i.VarName:
    #        #EN "asignacion" SE GUARDAN LOS NOMBRES DE LAS VARIABLES
    #          asignacion.append(i.VarName)
    #        #EN "asignacion_value" SE GUARDAN EL NOMBRE SEGUIDO POR EL VALOR DE ASIGNACIÓN
    #          asignacion_value.append(i.VarName)
    #          asignacion_value.append(i.x)
    #else:
    #    print("El modelo es infactible")

    #COMO EL MODELO ES UNA FUNCIÓN SE ENTREGAN ALGUNOS RETURN PARA OCUPARLOS POSTERIORMENTE
    if model.status == GRB.Status.OPTIMAL:
        duration = end_time - start_time
        print(f"El código se ejecutó en {duration:.2f} segundos")
        resultado = []
        for j in R:
            valor = y[j].x
            if valor > 0:
                resultado.append((j, valor))
                # print(f"{j}: {valor:.4f}")
        return model
    else:
        return False

def modelo_sin_limite_1(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Sin Límite")
    start_time = time.time()
    print("La cantidad de centros es",K )
    #Se calcula la población promedio
    phat= calcular_poblacion_total(comunas, R)/K

    # se generan los parametros i,j
    Xij = [(i,j) for i in R for j in R]
    Yj = [(j) for j in R]

    #se crea la variable, xij si es que i pertenece al distrito con centro j
    x= model.addVars(Xij,vtype=GRB.CONTINUOUS,name="asignaciones_ij")
    y = model.addVars(Yj, vtype=GRB.CONTINUOUS, name="centros_j")

    model.setObjective(0,GRB.MINIMIZE)

    for j in R:
        # Balance de población, permitiendo una diferencia de 1+-epsilon
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) <= phat * (1 + epsilon)*y[j])
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) >= phat * (1 - epsilon)*y[j])
        #Cuando un comuna es centro, debe estar asignada al distrito del cuál es centro
        model.addConstr((x[j, j]) == y[j])
        #Los centros deben ser igual a K (un parametro fijo y dado para el modelo)
        model.addConstr(quicksum(y[j] for j in R) == K)

    for i in R:
        #Todas las comunas deben estar asignadas a algún centro
        model.addConstr(quicksum(x[i,j] for j in R) == 1)
        for j in R:
          #Asigno comunas a un centro, solo si esa comuna esta abierta como centro
          model.addConstr(x[i,j]<= y[j])
          # Restricción de contiguidad
          # Se consulta en el diccionario dict_s (donde esta guardado shortest simple path)
          # Si es que las comunas no son adyacentes, se pide que para que i y j sean asignados
          # i y k deben estar obligatoriamente asignadas, porque para poder de i a j, k está en el camino
          aux_s = dict_s[(j, i)]
          while not aux_s == [[]]:
              for k in aux_s:
                  model.addConstr(quicksum(x[k[0], j] for k in aux_s) >= x[i, j])
                  aux_s = dict_s[(j, k[0])]

    model.optimize()
    end_time = time.time()
    #model.computeIIS()
    #model.write('iismodel.ilp')
    #Si es que el modelo es factible, se imprimen algunos resultados.
    if model.status == GRB.Status.OPTIMAL:
        duration = end_time - start_time
        print(f"El código se ejecutó en {duration:.2f} segundos")
        asignacion = []
        asignacion_value=[]
        for i in model.getVars():
          if i.x > 0:
            print(i.VarName,i.x)
            if "asignaciones_ij" in i.VarName:
            #EN "asignacion" SE GUARDAN LOS NOMBRES DE LAS VARIABLES
              asignacion.append(i.VarName)
            #EN "asignacion_value" SE GUARDAN EL NOMBRE SEGUIDO POR EL VALOR DE ASIGNACIÓN
              asignacion_value.append(i.VarName)
              asignacion_value.append(i.x)

        def asignaciones(centro,comuna):
          #valor debe ser i.x de centro, comuna
          lista=[centro,comuna]
          valor=asignacion.find()
          return valor

        coma=","
        corchetefinal = "]"
        corcheteinicio = "["
        lista=[]
        ##PRIMERO SE SEPARA TODO POR CADA CENTRO Y COMUNA
        #LUEGO SE CREA UNA NUEVA LISTA DONDE SE VAN AGREGANDO EN ORDEN  EL CENTRO CON LA COMUNA ASIGNADA
        #LA LISTA ESTA REALIZADA POR PAR
        for i in range(len(asignacion)):
            posicion_coma = asignacion[i].find(coma)
            posicion_corcheteinicio = asignacion[i].find(corcheteinicio)
            # el centro va despues del corchete inicial [ hasta la coma ,
            centro = asignacion[i][posicion_corcheteinicio + 1:posicion_coma]
            posicion_corchetefinal = asignacion[i].find(corchetefinal)
            #la comuna va desde la coma al corchete final
            comuna = asignacion[i][posicion_coma + 1:posicion_corchetefinal]
            lista.append(centro)
            lista.append(comuna)

        #SE CREA UNA LISTA DONDE SE CREAN TUPLAS DE INFORMACIÓN, SE DIVIDE LA LISTA EN SUBLISTAS
        lista_nueva_nombres = []
        for i in range(0, len(lista), 2):
            lista_nueva_nombres.append(lista[i:i+2])

        #SE CREA UNA LISTA DONDE ESTA LA INFORMACION DE LAS ASIGNACIONES Y SU VALOR
        valor_asignacion=[]
        for i in asignacion_value:
          if type(i)==str:
            # ACA SE BORRA EL NOMBRE DE LA VARIABLE PARA POSTERIORMENTE MANIPULAR LOS NOMBRES DE LAS COMUNAS POR SI SOLOS
            new_tupla = i.replace("asignaciones_ij", "")
            valor_asignacion.append(new_tupla)
          else:
            valor_asignacion.append(i)

        ##SE CREA UNA LISTA DONDE SE CREAN TUPLAS DE INFORMACIÓN, SE DIVIDE LA LISTA EN SUBLISTAS
        # SE TIENE LA ASIGNACIÓN CON SU RESPECTIVO VALOR
        # RECORDAR QUE LA ASIGNACIÓN ESTA ORDENADA POR COMUNA, CENTRO
        asignaciones_valores = []
        for i in range(0, len(valor_asignacion), 2):
            asignaciones_valores.append(valor_asignacion[i:i+2])
        diccionario_asignaciones = dict(asignaciones_valores)

        #FUNCION PARA SABER EL VALOR DE LA VARIABLE CENTRO, COMUNA
        def valor_por_asignacion(centro,comuna):
          valor= diccionario_asignaciones["[" + comuna + "," + centro + "]"]
          return valor

        #SE CREA UN DICCIONARIO DONDE SE PUEDE VER CADA CENTRO CON LAS ASIGNACIONES A ÉL
        # dicc_centro_com ES UN DICCIONARIO QUE POSEE CADA CENTRO CON LAS COMUNAS ASIGNADAS
        resumen_centro_comunas = defaultdict(list)
        for par_cen_com in lista_nueva_nombres:
          resumen_centro_comunas[par_cen_com[1]].append(par_cen_com[0])

        dicc_centro_com= dict(resumen_centro_comunas)
        for centro,comuna in dicc_centro_com.items():
          print("Al distrito con centro en", centro,"se le asignaron las comunas", comuna)

        #SE REALIZAN CALCULOS PARA SABER CUANTA ES LA POBLACIÓN DE CADA DISTRITO/CENTRO
        #Y PARA SABER CUAL ES EL RATIO DE LA POBLACIÓN DEL DISTRITO RESPECTO A LA POBLACIÓN PROMEDIO
        total_pob_centro=[]
        ratio_por_centro=[]
        for centro,comuna in dicc_centro_com.items():
          total_pob=0
          for comuna in comuna:
            nombrecentro=centro
            total=comunas.loc[comunas['comuna'] == comuna]
            consulta=total['poblacion2017']
            val=consulta.values[0]
            total_pob=total_pob+val*valor_por_asignacion(nombrecentro,comuna)
            ratio=total_pob/phat
          total_pob_centro.append({nombrecentro:total_pob})
          ratio_por_centro.append({nombrecentro:ratio})
        print("Los ratios poblacion distrito / poblacion promedio es ", ratio_por_centro)
        print("El valor de la poblacion promedio es",phat)
        print("La población por distrito según centro es",total_pob_centro)

        #SE REALIZAN CALCULOS PARA SABER CUÁL ES LA CANTIDAD DE COMUNAS ASIGNADAS POR DISTRITO
        info=[]
        for centro,comuna in dicc_centro_com.items():
          count=0
          for comuna in comuna:
            nombre= centro
            com=comuna[0]
            count=count+ 1*valor_por_asignacion(nombre,comuna)
          info.append({nombre:count})
        print(info)

        #SE CREA UNA LISTA DE ASIGNACIONES EN BASE AL CODIGO DE CADA COMUNA, PARA POSTERIORMENTE PODER GRAFICAR.
        lista_cut = []
        for par in lista_nueva_nombres:
            lista_cut.append(codigo_com_cut(comunas, par[0]))
            lista_cut.append(codigo_com_cut(comunas, par[1]))

        lista_asignaciones_cut = []
        for i in range(0, len(lista_cut), 2):
            lista_asignaciones_cut.append(lista_cut[i:i + 2])
        print(lista_asignaciones_cut)

        #CON PANDAS SE REALIZA EL GRÁFICO PARA VER DE FORMA VISUAL CADA UNO DE LOS DISTRITOS
        comunas_gdf = gpd.read_file('comunas.shp')
        custom_colors = ['#e6194B', '#3cb44b', '#4363d8', '#008b07', '#42d4f4', '#0017FF',
                         '#fabebe', '#469990', '#ff8000', '#9A6324', '#800000', '#aaffc3',
                         '#e24d28', '#ff03db', '#faa43a', '#60bd68', '#f17cb0', '#dcff00',
            '#000075', '#a9a9a9', '#000000', '#25aae2', '#a1d18a', '#edc240',
                         '#b276b2', '#decbe4', '#fddaec', '#ff0000', "#3AF245"]

        custom_colors_metrop = [
            "#1f77b4", "#ff7f0e", "#787926", "#d62728", "#2ca02c", "#FEF52F",
            "#F1B4DF", "#505050", "#9467bd", "#17becf", "#FD00C9", "#35D330",
            '#000000', '#aaffc3']

        comunas_gdf["distritos"] = 'Value'

        for par in lista_asignaciones_cut:
            comunas_gdf["distritos"] = comunas_gdf.apply(
                lambda x: par[1] if (x["cod_comuna"] == par[0]) else x["distritos"], axis=1)
        # district_arbol = list(comunas_gdf['distritos'].unique())
        # districts = list(range(29))
        # custom_cmap = {district_arbol[i]: custom_colors[i] for i in range(len(districts))}
        # cmap = matplotlib.colors.ListedColormap([custom_cmap[b] for b in comunas_gdf['distritos'].unique()])
        distrito_to_color = dict(zip(comunas_gdf['distritos'].unique(), custom_colors))

        # Creamos la figura y los ejes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Graficamos los distritos con colores basados en "distritos"
        comunas_gdf['color'] = comunas_gdf['distritos'].map(distrito_to_color)
        comunas_gdf.plot(ax=ax, color=comunas_gdf['color'])

        # Creamos la leyenda manualmente
        for distrito, color in distrito_to_color.items():
            # Aquí buscamos la "Comuna" correspondiente a cada "distrito"
            label = codigo_cut_com(comunas, distrito)
            ax.plot([], [], color=color, label=label, marker='o', markersize=10, linestyle='')

        # ax.set_position([0.05, 0.1, 0.6, 0.8])
        ax.margins(x=0, y=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        #plt.title("Distritaje Chile")
        plt.tight_layout()
        # Obtiene la hora actual y formatea para usar en el nombre del archivo
        hora_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_chile = f"oneshot_sinlimite_{hora_actual}.png"
        #filename_chile_svg = f"oneshot_sinlimite_{hora_actual}.svg"
        plt.savefig(filename_chile, format="png", bbox_inches='tight')
        #plt.savefig(filename_chile_svg, format="svg")
        # plt.show()

        # Filtrar los datos para incluir solo las filas con 'region' igual a 'metropolitana'
        gdf_metropolitana = comunas_gdf[comunas_gdf['Region'] == 'Región Metropolitana de Santiago']
        # Crear colores
        district_met = list(gdf_metropolitana['distritos'].unique())
        districts_metrop = list(range(len(district_met)))
        custom_cmap_metrop = {district_met[i]: custom_colors_metrop[i] for i in range(len(districts_metrop))}
        cmap_metrop = matplotlib.colors.ListedColormap(
            [custom_cmap_metrop[b] for b in gdf_metropolitana['distritos'].unique()])
        distrito_to_color_met = dict(zip(gdf_metropolitana['distritos'].unique(), custom_colors_metrop))

        # Creamos la figura y los ejes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Graficamos los distritos con colores basados en "distritos"
        gdf_metropolitana['color'] = gdf_metropolitana['distritos'].map(distrito_to_color_met)
        gdf_metropolitana.plot(ax=ax, color=gdf_metropolitana['color'])

        # Creamos la leyenda manualmente
        for distrito, color in distrito_to_color_met.items():
            # Aquí buscamos la "Comuna" correspondiente a cada "distrito"
            label = codigo_cut_com(comunas, distrito)
            ax.plot([], [], color=color, label=label, marker='o', markersize=10, linestyle='')

        # ax.set_position([0.05, 0.1, 0.6, 0.8])
        ax.margins(x=0, y=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        #plt.title("Distritaje Chile Región Metropolitana")
        plt.tight_layout()
        filename_met = f"oneshot_sinlimite_metropolitana_{hora_actual}.png"
        #filename_met_svg = f"oneshot_sinlimite_metropolitana_{hora_actual}.svg"
        plt.savefig(filename_met, format="png", bbox_inches='tight')
        #plt.savefig(filename_met_svg, format="svg")
        # plt.show()
        comunas_a_excluir_n = ['Isla de Pascua', 'Juan Fernández']
        nombres_comunas_a_excluir = comunas_gdf[comunas_gdf['Comuna'].isin(comunas_a_excluir_n)]['Comuna'].unique()

        # Filtra las comunas que no deseas incluir en la visualización
        comunas_gdf_filtrado = comunas_gdf[~comunas_gdf['Comuna'].isin(comunas_a_excluir_n)].copy()
        # Aplica asignaciones a los distritos
        for par in lista_asignaciones_cut:
            comunas_gdf_filtrado.loc[comunas_gdf_filtrado['cod_comuna'] == par[0], 'distritos'] = par[1]

        # Crea el diccionario de colores
        distrito_to_color_filt = dict(zip(comunas_gdf_filtrado['distritos'].unique(), custom_colors))

        # Creamos la figura y los ejes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Graficamos los distritos con colores basados en "distritos"
        comunas_gdf_filtrado['color'] = comunas_gdf_filtrado['distritos'].map(distrito_to_color_filt)
        comunas_gdf_filtrado.plot(ax=ax, color=comunas_gdf_filtrado['color'])

        # Creamos la leyenda manualmente
        for distrito, color in distrito_to_color_filt.items():
            # Aquí buscamos la "Comuna" correspondiente a cada "distrito"
            label = codigo_cut_com(comunas, distrito)
            ax.plot([], [], color=color, label=label, marker='o', markersize=10, linestyle='')

        # ax.set_position([0.05, 0.1, 0.6, 0.8])
        ax.margins(x=0, y=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # Obtiene la hora actual y formatea para usar en el nombre del archivo
        hora_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_chile = f"oneshot_sinlimite_sinislas_{hora_actual}.png"
        plt.savefig(filename_chile, format="png", bbox_inches='tight')
        # plt.show()
    else:
        print("El modelo es infactible")

    #COMO EL MODELO ES UNA FUNCIÓN SE ENTREGAN ALGUNOS RETURN PARA OCUPARLOS POSTERIORMENTE
    if model.status == GRB.Status.OPTIMAL:
        resultado = []
        for j in R:
            valor = y[j].x
            if valor > 0:
                resultado.append((j, valor))
                # print(f"{j}: {valor:.4f}")
        return model
    else:
        return False


# Modelo Centros fijos
# ep= 0.83984375
# R: comunas
# C: centros
def modelo_centros_fijos_con_limite(epsilon, R, C, dict_s, comunas, verbose=True):
    model = Model("Modelo")
    model.setParam("Method", 0)
    model.setParam("Threads", 6)

    if verbose:
        model.Params.LogToConsole = 1
    else:
        model.Params.LogToConsole = 0

    # De la class Model se usarán los atributos y/o funciones:
    # addVars() - setObjective() - addConstr() - optimize() - status()

    start_time = time.time()
    #print("La cantidad de centros es", len(C))
    # Se calcula la población promedio
    phat = calcular_poblacion_total(comunas, R)/len(C)

    # Se generan los parametros i,j
    Xij = [(i,j) for i in R for j in C]

    # Se crea la variable, x_ij si es que i pertenece al distrito con centro j
    x = model.addVars(Xij, vtype = GRB.BINARY, name = "x")

    model.setObjective(0, GRB.MINIMIZE) # REVISAR QUÉ UTILIDAD TIENE ESTA INSTANCIA AQUÍ

    # Se agrega una restricción para evitar asignaciones de comunas de distintas regiones
    # Si las regiones de i y j son distintas obligo que la asignación xij sea 0
    for i in R:
        for j in C:
            if obtener_region(comunas, i) != obtener_region(comunas, j):
                model.addConstr(x[(i, j)] == 0, name=f"block[{i},{j}]")

    # Todas las comunas i deben ser asignadas a algún centro j
    for i in R:
        model.addConstr(quicksum(x[i,j] for j in C) == 1, name=f"assign[{i}]")

    for j in C:
        # Balance de población, permitiendo una diferencia de 1+-epsilon
        model.addConstr(quicksum(pob(comunas,i)*x[i,j] for i in R) <= phat*(1+epsilon), name=f"pop_up[{j}]")
        model.addConstr(quicksum(pob(comunas, i)*x[i,j] for i in R) >= phat*(1-epsilon), name=f"pop_lo[{j}]")
        # Los centros no se pueden dividir
        model.addConstr(x[j,j] == 1.0, name=f"center[{j}]")

    # Restricción de contiguidad
    # Se consulta en el diccionario dict_s (donde esta guardado shortest simple path)
    # Si es que las comunas no son adyacentes, se pide que para que i y j sean asignados
    # i y k deben estar obligatoriamente asignadas, porque para poder de i a j, k está en el camino
    for i in R:
        for j in C:
            if obtener_region(comunas, i) == obtener_region(comunas, j):
                aux_s = dict_s[(j,i)]
                while not aux_s == [[]]:
                   for k in aux_s:
                        model.addConstr(quicksum(x[k[0],j] for k in aux_s) >= x[i,j])
                        aux_s = dict_s[j,(k[0])]


    model.optimize()
    end_time = time.time()
    # Si es que el modelo es factible, se imprimen algunos resultados.
    if model.status == GRB.Status.OPTIMAL:
        duration = end_time - start_time
    #    print(f"El código se ejecutó en {duration:.2f} segundos")
    #    asignacion = []
    #    asignacion_value=[]
    #    for i in model.getVars():
    #      if i.x > 0:
    #        print(i.VarName,i.x)
    #        if i.x >0:
    #          asignacion.append(i.VarName)
    #          asignacion_value.append(i.VarName)
    #          asignacion_value.append(i.x)
    else:
        pass
        #print("El modelo es infactible")

    # COMO EL MODELO ES UNA FUNCIÓN SE ENTREGAN ALGUNOS RETURN PARA OCUPARLOS POSTERIORMENTE
    if model.status == GRB.Status.OPTIMAL:
        return model
    else:
        return False
    
# ep = 0.6796875
def modelo_centros_fijos_sin_limite(epsilon, R, C, dict_s, comunas, verbose = True):
    model = Model("Modelo 1")
    model.setParam("Method", 0)
    model.setParam("Threads", 1)

    if verbose:
        model.Params.LogToConsole = 1
    else:
        model.Params.LogToConsole = 0
    start_time = time.time()
    print("La cantidad de centros es", len(C))
    #Se calcula la población promedio
    phat= calcular_poblacion_total(comunas, R)/len(C)

    # se generan los parametros i,j
    Xij = [(i,j) for i in R for j in C]

    #se crea la variable, xij si es que i pertenece al distrito con centro j
    x= model.addVars(Xij, vtype=GRB.BINARY, name="x")

    model.setObjective(0, GRB.MINIMIZE)

    #Todas las comunas i deben ser asignadas a algún centro j
    for i in R:
        model.addConstr(quicksum(x[i,j] for j in C) == 1)

    for j in C:
        #Balance de población, permitiendo una diferencia de 1+-epsilon
        model.addConstr(quicksum(pob(comunas, i)*x[i,j] for i in R) <=phat*(1+epsilon))
        model.addConstr(quicksum(pob(comunas, i)*x[i,j] for i in R) >=phat*(1-epsilon))
        #Los centros no se pueden dividir
        model.addConstr(x[j,j] == 1.0)

    #Restricción de contiguidad
    #Se consulta en el diccionario dict_s (donde esta guardado shortest simple path)
    #Si es que las comunas no son adyacentes, se pide que para que i y j sean asignados
    #i y k deben estar obligatoriamente asignadas, porque para poder de i a j, k está en el camino
    for i in R:
        for j in C:
            aux_s = dict_s[(j, i)]
            while not aux_s == [[]]:
                for k in aux_s:
                    model.addConstr(quicksum(x[k[0], j] for k in aux_s) >= x[i, j])
                    aux_s = dict_s[j, (k[0])]


    model.optimize()
    end_time=time.time()
    #Si es que el modelo es factible, se imprimen algunos resultados.
    #if model.status == GRB.Status.OPTIMAL:
    #    duration = end_time - start_time
    #    print(f"El código se ejecutó en {duration:.2f} segundos")
    #    asignacion = []
    #    asignacion_value=[]
    #    for i in model.getVars():
    #      if i.x > 0:
    #        print(i.VarName,i.x)
    #        if i.x >0:
    #          asignacion.append(i.VarName)
    #          asignacion_value.append(i.VarName)
    #          asignacion_value.append(i.x)
    #else:
    #    print("El modelo es infactible")

    #COMO EL MODELO ES UNA FUNCIÓN SE ENTREGAN ALGUNOS RETURN PARA OCUPARLOS POSTERIORMENTE
    if model.status == GRB.Status.OPTIMAL:
        duration = end_time - start_time
        print(f"El código se ejecutó en {duration:.2f} segundos")
        return model
    try: 
        if int(model.Status == 17):
            print("FALTA MEMORIA RAM")
    except:
        print("Modelo no óptimo. Status:", model.status)

    return None
    
def modelo_sin_limite(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Sin Límite")

    model.setParam("Method", 2) #-> recomendable según la gente de gurobi para large problems (fueza usar barrier algorithm)
    #model.setParam("Threads", 1)
    start_time = time.time()
    print("La cantidad de centros es",K )
    #Se calcula la población promedio
    phat= calcular_poblacion_total(comunas, R)/K

    # se generan los parametros i,j
    Xij = [(i,j) for i in R for j in R]
    Yj = [(j) for j in R]

    #se crea la variable, xij si es que i pertenece al distrito con centro j
    x= model.addVars(Xij,vtype=GRB.CONTINUOUS,name="asignaciones_ij")
    y = model.addVars(Yj, vtype=GRB.CONTINUOUS, name="centros_j")

    model.setObjective(0,GRB.MINIMIZE)

    for j in R:
        # Balance de población, permitiendo una diferencia de 1+-epsilon
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) <= phat * (1 + epsilon)*y[j])
        model.addConstr(quicksum(pob(comunas, i) * x[i, j] for i in R) >= phat * (1 - epsilon)*y[j])
        #Cuando un comuna es centro, debe estar asignada al distrito del cuál es centro
        model.addConstr((x[j, j]) == y[j])
        #Los centros deben ser igual a K (un parametro fijo y dado para el modelo)
        model.addConstr(quicksum(y[j] for j in R) == K)

    for i in R:
        #Todas las comunas deben estar asignadas a algún centro
        model.addConstr(quicksum(x[i,j] for j in R) == 1)
        for j in R:
          #Asigno comunas a un centro, solo si esa comuna esta abierta como centro
          model.addConstr(x[i,j]<= y[j])
          # Restricción de contiguidad
          # Se consulta en el diccionario dict_s (donde esta guardado shortest simple path)
          # Si es que las comunas no son adyacentes, se pide que para que i y j sean asignados
          # i y k deben estar obligatoriamente asignadas, porque para poder de i a j, k está en el camino
          aux_s = dict_s[(j, i)]
          while not aux_s == [[]]:
              for k in aux_s:
                  model.addConstr(quicksum(x[k[0], j] for k in aux_s) >= x[i, j])
                  aux_s = dict_s[(j, k[0])]

    model.optimize()
    end_time = time.time()
    #model.computeIIS()
    #model.write('iismodel.ilp')
    #Si es que el modelo es factible, se imprimen algunos resultados.
    #if model.status == GRB.Status.OPTIMAL:
    #    duration = end_time - start_time
    #    print(f"El código se ejecutó en {duration:.2f} segundos")
    #    asignacion = []
    #    asignacion_value=[]
    #    for i in model.getVars():
    #      if i.x > 0:
    #        print(i.VarName,i.x)
    #        if "asignaciones_ij" in i.VarName:
    #        #EN "asignacion" SE GUARDAN LOS NOMBRES DE LAS VARIABLES
    #          asignacion.append(i.VarName)
    #        #EN "asignacion_value" SE GUARDAN EL NOMBRE SEGUIDO POR EL VALOR DE ASIGNACIÓN
    #          asignacion_value.append(i.VarName)
    #          asignacion_value.append(i.x)
    #else:
    #    print("El modelo es infactible")

    #COMO EL MODELO ES UNA FUNCIÓN SE ENTREGAN ALGUNOS RETURN PARA OCUPARLOS POSTERIORMENTE
    if model.status == GRB.Status.OPTIMAL:
        duration = end_time - start_time
        print(f"El código se ejecutó en {duration:.2f} segundos")
        resultado = []
        for j in R:
            valor = y[j].x
            if valor > 0:
                resultado.append((j, valor))
                # print(f"{j}: {valor:.4f}")
        return model
    try: 
        if int(model.Status == 17):
            print("FALTA MEMORIA RAM")
    except:
        print("Modelo no óptimo. Status:", model.status)

    return None

    
def modelo_sin_limite_opti(epsilon, R, K, dict_s, comunas):
    """
    Modelo sin límite regional.

    Mantiene la idea original:
    - todos los pares (i,j),
    - balance poblacional proporcional a y[j],
    - x[j,j] = y[j],
    - sum(y) = K,
    - asignación completa,
    - link x[i,j] <= y[j],
    - contigüidad vía dict_s.

    Cambios solo de implementación:
    - cache de población,
    - cotas explícitas 0 <= x,y <= 1,
    - sum(y)=K se agrega una sola vez,
    - protección si dict_s trae un nodo k que no está en R,
    - SoftMemLimit,
    - diagnóstico de variables/restricciones.
    """

    model = Model("Modelo Sin Límite")

    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("Threads", 4)
    #model.setParam("SoftMemLimit", 15.0)
    model.setParam("OutputFlag", 1)

    start_time = time.time()

    print("La cantidad de centros es", K)

    # -----------------------------------------------------
    # 1. Cache de población
    # -----------------------------------------------------
    pobl = {i: pob(comunas, i) for i in R}
    phat = sum(pobl[i] for i in R) / K

    print("Población total:", sum(pobl.values()))
    print("phat:", phat)
    print("Rango permitido:", phat * (1 - epsilon), phat * (1 + epsilon))

    # -----------------------------------------------------
    # 2. Todos los pares, como en el modelo clásico
    # -----------------------------------------------------
    Xij = [(i, j) for i in R for j in R]

    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # -----------------------------------------------------
    # 3. Balance poblacional y centro
    # -----------------------------------------------------
    for j in R:
        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in R
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    # En el original estaba dentro del for j.
    # Es equivalente agregarlo una sola vez.
    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # -----------------------------------------------------
    # 4. Asignación y link
    # -----------------------------------------------------
    for i in R:
        model.addConstr(
            quicksum(x[i, j] for j in R) == 1,
            name=f"assign[{i}]"
        )

        for j in R:
            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # -----------------------------------------------------
    # 5. Contigüidad
    # -----------------------------------------------------
    n_path = 0
    n_skip = 0

    for i in R:
        for j in R:

            if i == j:
                continue

            aux_s = dict_s[(j, i)]

            while aux_s != [[]]:

                # En tu dict_s normalmente aux_s = [[k]]
                # pero dejamos la suma para mantener la lógica original.
                ks_validos = [
                    k[0] for k in aux_s
                    if (k[0], j) in x
                ]

                if ks_validos:
                    model.addConstr(
                        quicksum(x[k, j] for k in ks_validos) >= x[i, j],
                        name=f"path[{i},{j}]"
                    )
                    n_path += 1
                else:
                    # Si dict_s trae un k fuera de R, no hacemos caer el modelo.
                    # Esto no debería pasar si dict_s fue creado con el mismo R.
                    n_skip += 1
                    break

                # Avanzamos por el primer k, igual que en tu versión original.
                k_next = aux_s[0][0]

                if k_next not in R:
                    n_skip += 1
                    break

                aux_s = dict_s[(j, k_next)]

    print("Restricciones path:", n_path)
    print("Paths omitidos por k fuera de R:", n_skip)

    # -----------------------------------------------------
    # 6. Diagnóstico
    # -----------------------------------------------------
    model.update()

    print("Variables:", model.NumVars)
    print("Restricciones:", model.NumConstrs)

    # -----------------------------------------------------
    # 7. Resolver
    # -----------------------------------------------------
    model.optimize()

    end_time = time.time()

    if model.status == GRB.Status.OPTIMAL:
        print(f"El código se ejecutó en {end_time - start_time:.2f} segundos")

        for j in R:
            valor = y[j].X
            if valor > 1e-8:
                print(f"{j}: {valor:.4f}")

        return model

    if model.status == GRB.Status.MEM_LIMIT:
        print("FALTA MEMORIA RAM")
    else:
        print("Modelo no óptimo. Status:", model.status)

    return None


import time
import pickle
from collections import defaultdict
from gurobipy import Model, GRB, quicksum


# =========================================================
# funciones optimizadas
# =========================================================

# =========================================================
# HELPER 2: construir candidatos sparse
# =========================================================

def construir_J_por_i_sparse(R, comunas, distancias, M=60, con_limite=False):
    """
    Construye J_por_i.

    J_por_i[i] = lista de centros j a los que i puede asignarse.

    Caso sin límite regional:
        i puede mirar sus M centros más cercanos dentro de todo R.

    Caso con límite regional:
        i solo puede mirar sus M centros más cercanos dentro de su misma región.

    Esta es la parte que reemplaza el modelo denso:
        antes: j en R para todo i
        ahora: j en M más cercanos para cada i
    """

    dist_df = distancias.set_index("comuna")

    if con_limite:
        region = {i: obtener_region(comunas, i) for i in R}
    else:
        region = None

    J_por_i = {}

    for i in R:

        if con_limite:
            candidatos_base = [
                j for j in R
                if region[j] == region[i]
            ]
        else:
            candidatos_base = list(R)

        candidatos = (
            dist_df.loc[i, candidatos_base]
            .sort_values()
            .index
            .tolist()
        )

        J = candidatos[:M]

        # Importante: asegurar que exista x[i,i],
        # porque usamos la restricción x[j,j] = y[j].
        if i not in J:
            J.append(i)

        J_por_i[i] = J

    return J_por_i


# =========================================================
# MODELO 1: SIN límite regional sparse
# =========================================================

def modelo_sin_limite_sparse(
    epsilon,
    R,
    K,
    dict_s,
    comunas,
    distancias,
    M=60,
    usar_contiguidad=True
):
    """
    Modelo sin límite regional, versión sparse.

    Diferencia con el modelo clásico:
    - Antes se creaban todos los pares (i,j), es decir |R|^2 variables x.
    - Ahora cada unidad i solo puede asignarse a sus M centros más cercanos.

    Si usar_contiguidad=True:
    - Se agregan restricciones de camino usando dict_s desde .pkl.
    - Si el camino exige un nodo k, pero x[k,j] no existe por el sparseo,
      se bloquea x[i,j] = 0.
    """

    start_time = time.time()

    model = Model("Modelo Sin Límite Sparse")

    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("Threads", 8)
    #model.setParam("SoftMemLimit", 20.0)
    model.setParam("OutputFlag", 1)

    print("Modelo: SIN límite regional")
    print("K:", K)
    print("M:", M)
    print("Usar contigüidad:", usar_contiguidad)

    pobl = {i: pob(comunas, i) for i in R}
    phat = sum(pobl[i] for i in R) / K

    print("Población total:", sum(pobl.values()))
    print("phat:", phat)
    print("Rango:", phat * (1 - epsilon), phat * (1 + epsilon))

    # -----------------------------------------------------
    # 1. Candidatos sparse
    # -----------------------------------------------------
    J_por_i = construir_J_por_i_sparse(
        R=R,
        comunas=comunas,
        distancias=distancias,
        M=M,
        con_limite=False
    )

    Xij = [(i, j) for i in R for j in J_por_i[i]]

    print("Variables x:", len(Xij))
    print("Variables y:", len(R))
    print("Pares densos serían:", len(R) * len(R))

    # -----------------------------------------------------
    # 2. Variables
    # -----------------------------------------------------
    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # -----------------------------------------------------
    # 3. Índices inversos
    # -----------------------------------------------------
    I_por_j = {j: [] for j in R}

    for i, j in Xij:
        I_por_j[j].append(i)

    # -----------------------------------------------------
    # 4. Balance poblacional y centro
    # -----------------------------------------------------
    for j in R:
        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in I_por_j[j]
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # -----------------------------------------------------
    # 5. Asignación y link
    # -----------------------------------------------------
    for i in R:
        model.addConstr(
            quicksum(x[i, j] for j in J_por_i[i]) == 1,
            name=f"assign[{i}]"
        )

        for j in J_por_i[i]:
            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # -----------------------------------------------------
    # 6. Contigüidad
    # -----------------------------------------------------
    n_path = 0
    n_block = 0

    if usar_contiguidad:
        print("Agregando contigüidad...")

        for i in R:
            for j in J_por_i[i]:

                if i == j:
                    continue

                aux_s = dict_s[(j, i)]

                while aux_s != [[]]:
                    k = aux_s[0][0]

                    if (k, j) in x:
                        model.addConstr(
                            x[k, j] >= x[i, j],
                            name=f"path[{i},{j},{k}]"
                        )
                        n_path += 1
                    else:
                        model.addConstr(
                            x[i, j] == 0,
                            name=f"path_block[{i},{j},{k}]"
                        )
                        n_block += 1
                        break

                    aux_s = dict_s[(j, k)]

    print("Restricciones path:", n_path)
    print("Asignaciones bloqueadas:", n_block)

    model.update()

    print("NumVars:", model.NumVars)
    print("NumConstrs:", model.NumConstrs)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(f"[OK] Tiempo: {time.time() - start_time:.2f} segundos")
        return model

    try: 
        if int(model.Status == 17):
            print("FALTA MEMORIA RAM")
    except:
        print("Modelo no óptimo. Status:", model.status)
    return None


def modelo_sin_limite_sparse_v2(
    epsilon,
    R,
    K,
    dict_s,
    comunas,
    distancias,
    M=60,
    usar_contiguidad=True,
    max_iter_cierre=10
):
    """
    Modelo sin límite regional, versión sparse v2.

    Diferencia con el sparse anterior:
    - Antes, si el camino j -> i necesitaba un nodo k y no existía x[k,j],
      se bloqueaba x[i,j] = 0.
    - Ahora NO se bloquea.
    - En cambio, antes de crear variables, se agregan al sparse los pares
      necesarios para que los caminos puedan existir.

    Esto se aleja menos del modelo original.
    """

    start_time = time.time()

    model = Model("Modelo Sin Límite Sparse v2")

    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("Threads", 8)
    model.setParam("OutputFlag", 1)

    print("Modelo: SIN límite regional SPARSE v2")
    print("K:", K)
    print("M inicial:", M)
    print("Usar contigüidad:", usar_contiguidad)

    pobl = {i: pob(comunas, i) for i in R}
    phat = sum(pobl[i] for i in R) / K

    print("Población total:", sum(pobl.values()))
    print("phat:", phat)
    print("Rango:", phat * (1 - epsilon), phat * (1 + epsilon))

    # -----------------------------------------------------
    # 1. Candidatos sparse iniciales
    # -----------------------------------------------------
    J_por_i = construir_J_por_i_sparse(
        R=R,
        comunas=comunas,
        distancias=distancias,
        M=M,
        con_limite=False
    )

    pares_iniciales = sum(len(J_por_i[i]) for i in R)

    print("Pares sparse iniciales:", pares_iniciales)
    print("Pares densos serían:", len(R) * len(R))

    # -----------------------------------------------------
    # 2. Cierre por caminos
    # -----------------------------------------------------
    if usar_contiguidad:
        print("\nAplicando cierre por caminos...")

        R_set = set(R)

        for it in range(1, max_iter_cierre + 1):

            cambios = 0
            nuevos = {i: set(J_por_i[i]) for i in R}

            for i in R:
                for j in list(J_por_i[i]):

                    if i == j:
                        continue

                    aux_s = dict_s[(j, i)]

                    while aux_s != [[]]:

                        k = aux_s[0][0]

                        if k not in R_set:
                            break

                        # Si x[i,j] existe, entonces x[k,j] debe existir también.
                        if j not in nuevos[k]:
                            nuevos[k].add(j)
                            cambios += 1

                        aux_s = dict_s[(j, k)]

            J_por_i = {
                i: sorted(nuevos[i])
                for i in R
            }

            total_pares = sum(len(J_por_i[i]) for i in R)

            print(
                f"Iteración cierre {it}: "
                f"pares={total_pares}, "
                f"nuevos={cambios}"
            )

            if cambios == 0:
                print("[OK] Cierre por caminos estabilizado.")
                break

        else:
            print("[WARN] Cierre llegó a max_iter_cierre sin estabilizar.")

    pares_finales = sum(len(J_por_i[i]) for i in R)

    print("\nPares finales:", pares_finales)
    print("Aumento por cierre:", pares_finales - pares_iniciales)
    print("Fracción del denso:", pares_finales / (len(R) * len(R)))

    # -----------------------------------------------------
    # 3. Variables
    # -----------------------------------------------------
    Xij = [
        (i, j)
        for i in R
        for j in J_por_i[i]
    ]

    print("Variables x:", len(Xij))
    print("Variables y:", len(R))

    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # -----------------------------------------------------
    # 4. Índices inversos
    # -----------------------------------------------------
    I_por_j = {j: [] for j in R}

    for i, j in Xij:
        I_por_j[j].append(i)

    # -----------------------------------------------------
    # 5. Balance poblacional y centro
    # -----------------------------------------------------
    for j in R:
        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in I_por_j[j]
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # -----------------------------------------------------
    # 6. Asignación y link
    # -----------------------------------------------------
    for i in R:
        model.addConstr(
            quicksum(x[i, j] for j in J_por_i[i]) == 1,
            name=f"assign[{i}]"
        )

        for j in J_por_i[i]:
            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # -----------------------------------------------------
    # 7. Contigüidad
    # -----------------------------------------------------
    n_path = 0
    n_missing = 0

    if usar_contiguidad:
        print("\nAgregando contigüidad...")

        for i in R:
            for j in J_por_i[i]:

                if i == j:
                    continue

                aux_s = dict_s[(j, i)]

                while aux_s != [[]]:

                    k = aux_s[0][0]

                    if (k, j) in x:
                        model.addConstr(
                            x[k, j] >= x[i, j],
                            name=f"path[{i},{j},{k}]"
                        )
                        n_path += 1
                    else:
                        # En v2 no bloqueamos x[i,j].
                        # Si esto aparece mucho, el cierre no fue suficiente.
                        n_missing += 1
                        break

                    aux_s = dict_s[(j, k)]

    print("Restricciones path:", n_path)
    print("Pares faltantes post-cierre:", n_missing)

    model.update()

    print("NumVars:", model.NumVars)
    print("NumConstrs:", model.NumConstrs)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(f"[OK] Tiempo: {time.time() - start_time:.2f} segundos")
        return model

    if model.status == GRB.Status.MEM_LIMIT:
        print("FALTA MEMORIA RAM")
    else:
        print("Modelo no óptimo. Status:", model.status)

    return None
# =========================================================
# MODELO 2: CON límite regional sparse
# =========================================================

def modelo_con_limite_sparse(
    epsilon,
    R,
    K,
    dict_s,
    comunas,
    distancias,
    M=60,
    usar_contiguidad=True
):
    """
    Modelo con límite regional, versión sparse.

    Diferencia con el modelo sin límite:
    - Cada unidad i solo puede asignarse a centros j de su misma región.
    - Dentro de esa región, se toman los M centros más cercanos.

    Si usar_contiguidad=True:
    - Se usan caminos desde dict_s.
    - Si falta x[k,j] por sparseo, se bloquea x[i,j].
    """

    start_time = time.time()
    model = Model("Modelo Con Límite Regional Sparse")

    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("Threads", 8)
    model.setParam("SoftMemLimit", 20.0)
    model.setParam("OutputFlag", 1)

    print("Modelo: CON límite regional")
    print("K:", K)
    print("M:", M)
    print("Usar contigüidad:", usar_contiguidad)

    pobl = {i: pob(comunas, i) for i in R}
    phat = sum(pobl[i] for i in R) / K

    print("Población total:", sum(pobl.values()))
    print("phat:", phat)
    print("Rango:", phat * (1 - epsilon), phat * (1 + epsilon))

    # -----------------------------------------------------
    # 1. Candidatos sparse con límite regional
    # -----------------------------------------------------
    J_por_i = construir_J_por_i_sparse(
        R=R,
        comunas=comunas,
        distancias=distancias,
        M=M,
        con_limite=True
    )

    Xij = [(i, j) for i in R for j in J_por_i[i]]

    print("Variables x:", len(Xij))
    print("Variables y:", len(R))
    print("Pares densos serían:", len(R) * len(R))

    # -----------------------------------------------------
    # 2. Variables
    # -----------------------------------------------------
    x = model.addVars(
        Xij,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="asignaciones_ij"
    )

    y = model.addVars(
        R,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="centros_j"
    )

    model.setObjective(0, GRB.MINIMIZE)

    # -----------------------------------------------------
    # 3. Índice inverso
    # -----------------------------------------------------
    I_por_j = {j: [] for j in R}

    for i, j in Xij:
        I_por_j[j].append(i)

    # -----------------------------------------------------
    # 4. Balance poblacional y centro
    # -----------------------------------------------------
    for j in R:
        expr_pop = quicksum(
            pobl[i] * x[i, j]
            for i in I_por_j[j]
        )

        model.addConstr(
            expr_pop <= phat * (1 + epsilon) * y[j],
            name=f"pop_up[{j}]"
        )

        model.addConstr(
            expr_pop >= phat * (1 - epsilon) * y[j],
            name=f"pop_lo[{j}]"
        )

        model.addConstr(
            x[j, j] == y[j],
            name=f"center[{j}]"
        )

    model.addConstr(
        quicksum(y[j] for j in R) == K,
        name="centers_total"
    )

    # -----------------------------------------------------
    # 5. Asignación y link
    # -----------------------------------------------------
    for i in R:
        model.addConstr(
            quicksum(x[i, j] for j in J_por_i[i]) == 1,
            name=f"assign[{i}]"
        )

        for j in J_por_i[i]:
            model.addConstr(
                x[i, j] <= y[j],
                name=f"link[{i},{j}]"
            )

    # -----------------------------------------------------
    # 6. Contigüidad
    # -----------------------------------------------------
    n_path = 0
    n_block = 0

    if usar_contiguidad:
        print("Agregando contigüidad...")

        for i in R:
            for j in J_por_i[i]:

                if i == j:
                    continue

                aux_s = dict_s[(j, i)]

                while aux_s != [[]]:
                    k = aux_s[0][0]

                    if (k, j) in x:
                        model.addConstr(
                            x[k, j] >= x[i, j],
                            name=f"path[{i},{j},{k}]"
                        )
                        n_path += 1
                    else:
                        model.addConstr(
                            x[i, j] == 0,
                            name=f"path_block[{i},{j},{k}]"
                        )
                        n_block += 1
                        break

                    aux_s = dict_s[(j, k)]

    print("Restricciones path:", n_path)
    print("Asignaciones bloqueadas:", n_block)

    model.update()

    print("NumVars:", model.NumVars)
    print("NumConstrs:", model.NumConstrs)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(f"[OK] Tiempo: {time.time() - start_time:.2f} segundos")
        return model

    try: 
        if int(model.Status == 17):
            print("FALTA MEMORIA RAM")
    except:
        print("Modelo no óptimo. Status:", model.status)

    return None