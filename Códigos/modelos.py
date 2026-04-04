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




def modelo_con_limite(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Con Límite Regional")
    model.setParam("Method", 4)
    model.setParam("Threads", 8)

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


def modelo_relajado_2(epsilon, R, K, comunas):

    model = Model("modelo_relajado")
    model.Params.OutputFlag = 0

    phat = calcular_poblacion_total(comunas, R) / K

    # Variables
    z = model.addVars([(i, j) for i in R for j in R],
                      vtype=GRB.CONTINUOUS, lb=0.0, name="z")
    y = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")

    # Objetivo
    model.setObjective(0, GRB.MINIMIZE)

    # Asignación completa
    for i in R: model.addConstr(quicksum(z[i, j] for j in R) == 1,
                                name=f"assign[{i}]")

    # Balance poblacional
    for j in R:
        model.addConstr( quicksum(pob(comunas, i) * z[i, j] for i in R) <= phat * (1 + epsilon) * y[j],
                                name=f"pop_up[{j}]" )
        model.addConstr( quicksum(pob(comunas, i) * z[i, j] for i in R) >= phat * (1 - epsilon) * y[j],
                        name=f"pop_lo[{j}]" )
        model.addConstr(z[j, j] == y[j], name=f"center[{j}]")

    # Cantidad de centros
    model.addConstr(quicksum(y[j] for j in R) == K,
                    name="centers_total")

    # z_ij \leq y_j
    for i in R:
        for j in R: model.addConstr(z[i, j] <= y[j],
                                    name=f"link[{i},{j}]")
            
    model.optimize()
    
    if model.status == GRB.Status.OPTIMAL:
        print("Modelo relajado factible")
        return model
    else: 
        print("Modelo relajado infactible")
        return None

def modelo_sin_limite(epsilon, R, K, dict_s, comunas):
    model = Model("Modelo Sin Límite")

    model.setParam("Method", 4)
    model.setParam("Threads", 8)
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
    model.setParam("Method", 4)
    model.setParam("Threads", 8)

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
    model.setParam("Method", 4)
    model.setParam("Threads", 8)

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
    else:
        return False