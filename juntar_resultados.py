"""
CONFIGURACIÓN SIN DESCONOCIDOS: UNIFICACIÓN DE RESULTADOS

Este script recorre los CSV de resultados generados por cada modelo de
reconocimiento facial (Deepface, Empresa, ArcFace, AdaFace, ArcFace small)
sobre los distintos datasets evaluados, y los une en un único fichero
Excel (resultados_completo.xlsx).
Además, convierte las columnas "rank1" y "rank5" de proporción (0-1) a
porcentaje (0-100), y guarda el resultado final en resultados_completo.xlsx.
"""

import pandas as pd
import os

base_out = "/media/nas/carmengcamp/TFG"  #carpeta base 
dataset_names=["lfw_cropped", "lfw_filtrado", "lfw-deepfunneled_cropped", "lfw_noisy_25","lfw_noisy_50","lfw_noisy_75"] #datasets
modelos = ["deepface", "empresa", "arcface", "adaface", "arcface_small"] #modelos
carpetas= ["Deepface", "Empresa", "ArcFace", "AdaFace/resultados", "ArcFace_small"] #carpeta donde está cada modelo

dfs = [] #lista de DataFrames a unir
missing = [] #rutas de ficheros que no se han encontrado

for modelo,nombre in zip(modelos,carpetas):
    for name in dataset_names: #recorre cada combinación de modelo y dataset
        path = f"{base_out}/{nombre}/resultados_{modelo}_{name}.csv" #ruta completa

        if not os.path.exists(path): #si no existe se registra
            missing.append(str(path))
            continue

        df = pd.read_csv(path) #carga el CSV de resultados
        #añade las columnas modelo y dataset
        df.insert(0, "modelo", df["embedding"].str.replace("Embedding_", "", regex=False))
        df.insert(1, "dataset", name)
        dfs.append(df) 

if missing: #si hubo ficheros no encontrados, se informa
    print(f"Ficheros no encontrados ({len(missing)}):")
    for m in missing:
        print(f"   {m}")

if dfs:
    resultado = pd.concat(dfs, ignore_index=True) #une todos los resultados
    for col in ["rank1", "rank5"]:
        if col in resultado.columns: #pasa rank1 y rank5 de proporción (0-1) a porcentaje (0-100)
            resultado[col] = resultado[col] * 100
    salida = f"{base_out}/resultados_completo.xlsx" #ruta de salida
    resultado.to_excel(salida, index=False, sheet_name="resultados")

    print(f"Unificados {len(dfs)} ficheros: {salida}")
    print(f"Filas totales : {len(resultado)}")
    print(f"Columnas      : {list(resultado.columns)}")