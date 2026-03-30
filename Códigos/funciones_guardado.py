import os
import json
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def modelo_a_asignaciones_df(modelo):
    filas = []

    for v in modelo.getVars():
        nombre = v.VarName

        if not nombre.startswith("x["):
            continue

        if "[" not in nombre or "]" not in nombre:
            continue

        inside = nombre[nombre.find("[") + 1 : nombre.find("]")]
        partes = [p.strip() for p in inside.split(",")]

        if len(partes) != 2:
            continue

        i, j = partes
        filas.append({
            "i": i,
            "j": j,
            "x": v.X
        })

    return pd.DataFrame(filas)


def guardar_resultado_factible(base_dir, nombre_resultado, modelo, centros, metadata=None):
    carpeta = os.path.join(base_dir, nombre_resultado)
    ensure_dir(carpeta)

    df_asignaciones = modelo_a_asignaciones_df(modelo)
    df_asignaciones.to_csv(
        os.path.join(carpeta, f"{nombre_resultado}_asignaciones.csv"),
        index=False
    )

    with open(os.path.join(carpeta, f"{nombre_resultado}_centros.txt"), "w", encoding="utf-8") as f:
        for centro in centros:
            f.write(f"{centro}\n")

    data = {"centros": centros}
    if metadata is not None:
        data.update(metadata)

    with open(os.path.join(carpeta, f"{nombre_resultado}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)