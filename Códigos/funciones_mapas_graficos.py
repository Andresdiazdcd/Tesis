import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from shapely.geometry import box
def graficar_eeuu_desde_asignaciones_csv(
    ruta_asignaciones,
    ruta_geojson,
    output_png="mapa_eeuu.png",
    tol=1e-6,
    titulo="Distritaje EEUU",
    legend=True,
    cmap="tab20"
):
    df = pd.read_csv(ruta_asignaciones)

    df = df[df["x"] > tol].copy()

    df["comuna"] = df["i"].astype(str).str.strip().str.lower()
    df["centro"] = df["j"].astype(str).str.strip().str.lower()

    df_plot = (
        df.sort_values("x", ascending=False)
          .drop_duplicates("comuna")
          [["comuna", "centro"]]
    )

    gdf = gpd.read_file(ruta_geojson)

    gdf["comuna"] = (
        gdf["county"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    gdf = gdf.merge(df_plot, on="comuna", how="left")

    print("Total polígonos:", len(gdf))
    print("Asignados:", gdf["centro"].notna().sum())
    print("No asignados:", gdf["centro"].isna().sum())

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(
        ax=ax,
        column="centro",
        categorical=True,
        legend=legend,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.15,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "sin asignación"
        }
    )

    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg._loc = 2

    ax.set_axis_off()
    ax.set_title(titulo)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    return gdf


def graficar_mapa_eeuu_gurobi(modelo, nodes_geojson, output_png=None, tol=1e-6):
    """
    Grafica un resultado Gurobi para EEUU.
    
    modelo: modelo Gurobi ya optimizado
    nodes_geojson: ruta a nodes.geojson del estado
    output_png: nombre del archivo de salida
    """

    # =========================
    # Extraer asignaciones x[i,j]
    # =========================
    asignaciones = []

    for v in modelo.getVars():
        if v.VarName.startswith("asignaciones_ij") and v.X > tol:
            contenido = v.VarName[
                v.VarName.find("[") + 1 : v.VarName.find("]")
            ]
            i, j = contenido.split(",")
            asignaciones.append({
                "comuna": i,
                "centro": j,
                "valor": v.X
            })

    df = pd.DataFrame(asignaciones)

    # Si hay fracciones, tomar el centro con mayor peso para graficar
    df_plot = (
        df.sort_values("valor", ascending=False)
          .drop_duplicates("comuna")
          [["comuna", "centro"]]
    )

    # =========================
    # Leer geometría
    # =========================
    gdf = gpd.read_file(nodes_geojson)

    gdf["comuna"] = (
        gdf["county"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    gdf = gdf.merge(df_plot, on="comuna", how="left")

    # =========================
    # Graficar
    # =========================
    print(gdf.total_bounds)

    print(gdf.crs)
    # Pasar a lon/lat
    gdf = gdf.to_crs(epsg=4326)

    # Caja Chile continental aprox.
    bbox_continental = gpd.GeoDataFrame(
        geometry=[box(-76, -56, -66, -17)],
        crs="EPSG:4326"
    )

    # Recortar geometría al continente
    gdf = gpd.clip(gdf, bbox_continental)

    fig, ax = plt.subplots(figsize=(10,10))
    gdf.boundary.plot(ax=ax)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(
        ax=ax,
        column="centro",
        categorical=True,
        legend=False,
        edgecolor="black",
        linewidth=0.15
    )

    ax.set_axis_off()
    ax.set_title("Distritaje EEUU", fontsize=14)
    plt.tight_layout()

    if output_png is None:
        hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_png = f"mapa_eeuu_{hora}.png"

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Mapa guardado en: {output_png}")

    return gdf



def limpiar_nombre_chile(x):
    return (
        str(x).strip().lower()
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ü", "u")
        .replace("ñ", "n").replace("'", "")
        .replace(" ", "_")
    )

def graficar_desde_asignaciones_csv(
    ruta_asignaciones,
    ruta_geojson,
    output_png="mapa.png",
    tol=1e-6,
    titulo="Distritaje",
    clip_continental=False,
    legend=True
):
    df = pd.read_csv(ruta_asignaciones)

    df = df[df["x"] > tol].copy()

    df["comuna"] = df["i"].apply(limpiar_nombre_chile)
    df["centro"] = df["j"].apply(limpiar_nombre_chile)

    df_plot = (
        df.sort_values("x", ascending=False)
          .drop_duplicates("comuna")
          [["comuna", "centro"]]
    )

    gdf = gpd.read_file(ruta_geojson)
    gdf["comuna"] = gdf["county"].apply(limpiar_nombre_chile)

    gdf = gdf.merge(df_plot, on="comuna", how="left")

    print("Total polígonos:", len(gdf))
    print("Asignados:", gdf["centro"].notna().sum())
    print("No asignados:", gdf["centro"].isna().sum())

    if clip_continental:
        gdf = gdf.to_crs(epsg=4326)

        bbox_continental = gpd.GeoDataFrame(
            geometry=[box(-76, -56, -66, -17)],
            crs="EPSG:4326"
        )

        gdf = gpd.clip(gdf, bbox_continental)

    fig, ax = plt.subplots(figsize=(16, 12))

    gdf.plot(
        ax=ax,
        column="centro",
        categorical=True,
        legend=legend,
        cmap="nipy_spectral",
        edgecolor="black",
        linewidth=0.15,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "sin asignación"
        }
    )
    leg = ax.get_legend()

    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg._loc = 2   # upper left

    ax.set_axis_off()
    ax.set_title(titulo)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    return gdf