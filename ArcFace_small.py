"""
GENERACIÓN DE EMBEDDINGS: ArcFace MobileFaceNet (Entorno Torch)

Este script recorre los datasets de imágenes de caras y genera los embeddings faciales de cada imagen con
el modelo ArcFace MobileFaceNet. Después, los almacena en ficheros .pkl
En segundo lugar, llama a la función "evaluar_modelo" para calcular las métricas de evaluación en la
configuración sin desconocidos.
"""
import os
import cv2

#Configuración de la GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
from insightface.model_zoo import get_model
from funcion_calcular_resultados import evaluar_modelo
import onnx

#comprueba si hay GPU disponible
if torch.cuda.is_available():
    ctx_id = 0 #GPU
else:
    ctx_id = -1 #CPU

#datasets
datasets_paths = ["/media/nas/carmengcamp/TFG/lfw_filtrado","/media/nas/carmengcamp/TFG/lfw_cropped","/media/nas/carmengcamp/TFG/lfw-deepfunneled_cropped","/media/nas/carmengcamp/TFG/lfw_noisy_25", "/media/nas/carmengcamp/TFG/lfw_noisy_50","/media/nas/carmengcamp/TFG/lfw_noisy_75"]
path_plantilla = "/media/nas/carmengcamp/TFG/plantilla.csv" #plantilla de la configuración sin desconocidos
MODEL_PATH     = "/media/nas/carmengcamp/TFG/models/arcface_small/w600k_mbf.onnx" #ruta del modelo


#número de parámetros
print("Precargando modelo en VRAM...")
model = get_model(MODEL_PATH) #carga el modelo ArcFace desde el fichero ONNX
model.prepare(ctx_id=ctx_id)#prepara el modelo para GPU o CPU
try:
    onnx_model   = onnx.load(MODEL_PATH)#carga el grafo ONNX para inspeccionarlo
    total_params = sum(np.prod(t.dims) for t in onnx_model.graph.initializer)#número total de parámetros
    print(f"Modelo       : w600k_r50 (ArcFace)")
    print(f"Parámetros   : {total_params:,}")
    print(f"Dispositivo  : {'GPU (ctx_id=0)' if ctx_id == 0 else 'CPU'}")
except Exception as e:
    print(f"No se pudo obtener el número de parámetros ({e})")
print("¡Modelo cargado! Iniciando extracción...\n")

#generación de embeddings
for dataset_path in datasets_paths:
    dataset_name = os.path.basename(dataset_path)#recorre cada dataset
    print(f"===== GENERANDO EMBEDDINGS DE {dataset_name} =====\n")
    data = [] #lista de embeddings y tiempos
    for person_entry in tqdm(os.scandir(dataset_path), desc="Procesando personas"): #recorre cada persona del dataset
        if not person_entry.is_dir():
            continue
        for img_entry in os.scandir(person_entry.path): #recorre cada imagen de esa persona
            if not img_entry.is_file():
                continue

            img_path = img_entry.path
            img_array = cv2.imread(img_path) #carga la imagen

            if img_array is None:
                print(f"Error al leer la imagen: {img_path}")
                continue

            img_resized = cv2.resize(img_array, (112, 112)) #El modelo espera imágenes de 112x112
            row_data = {
                "ID_Persona":person_entry.name, #identidad de la persona
                "Ruta_Imagen":img_path, #ruta completa de la imagen
                "Nombre_Archivo":img_entry.name, #nombre del archivo de la imagen
            }
            try:
                start=time.time()
                embedding=model.get_feat(img_resized).flatten() #calcula el embedding y lo aplana a vector 1D
                elapsed=time.time()- start #tiempo de inferencia de la imagen
                row_data["Embedding_ArcFace_s"]= embedding.tolist()  #guarda el embedding como lista
                row_data["Tiempo_ArcFace_s"]= elapsed
            except Exception as e: #si falla, se guarda None
                print(f"Error con {img_path}: {e}")
                row_data["Embedding_ArcFace_s"]= None
                row_data["Tiempo_ArcFace_s"]= None
            data.append(row_data)

    df = pd.DataFrame(data)
    pkl_embeddings = f"/media/nas/carmengcamp/TFG/ArcFace_small/embeddings_arcface_small_{dataset_name}.pkl"
    df.to_pickle(pkl_embeddings) #guarda los embeddings en un .pkl

    count = df["Embedding_ArcFace_s"].notna().sum()
    print(f"Embeddings generados para ArcFace: {count}")

    columnas   = {"Embedding_ArcFace_s": "Tiempo_ArcFace_s"}
    salida     = f"/media/nas/carmengcamp/TFG/ArcFace_small/resultados_arcface_small_{dataset_name}.csv"
    #evaluación para la configuración sin desconocidos
    results_df = evaluar_modelo(path_embeddings=pkl_embeddings, path_plantilla=path_plantilla, columnas=columnas, salida=salida)
    print("========================================\n")

