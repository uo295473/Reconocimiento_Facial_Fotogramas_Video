"""
GENERACIÓN DE EMBEDDINGS: AdaFace (Entorno Torch)

Este script recorre los datasets de imágenes de caras y genera los embeddings faciales de cada imagen con
el modelo AdaFace. Después, los almacena en ficheros .pkl
En segundo lugar, llama a la función "evaluar_modelo" para calcular las métricas de evaluación en la
configuración sin desconocidos.

Pasos para utilizar este modelo:
1. Clonar el repositorio https://github.com/mk-minchul/AdaFace
2. Situarse en la carpeta del repositorio clonado y mover a ella este programa
   y el fichero funcion_calcular_resultados.py
3. Crear una carpeta llamada "pretrained" dentro del repositorio
4. Mover el fichero con los pesos del modelo a esa carpeta
5. Ejecutar este programa
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
import pytorch_lightning as pl
from inference import load_pretrained_model, to_input
from funcion_calcular_resultados import evaluar_modelo


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #usa GPU si está disponible, si no CPU

#datasets
datasets_paths = ["/media/nas/carmengcamp/TFG/lfw_filtrado","/media/nas/carmengcamp/TFG/lfw_cropped","/media/nas/carmengcamp/TFG/lfw-deepfunneled_cropped","/media/nas/carmengcamp/TFG/lfw_noisy_25", "/media/nas/carmengcamp/TFG/lfw_noisy_50","/media/nas/carmengcamp/TFG/lfw_noisy_75"]
path_plantilla = "/media/nas/carmengcamp/TFG/plantilla.csv" #plantilla de la configuración sin desconocidos
MODEL_NAME     = "ir_50"

#número de parámetros
print("Precargando modelo en VRAM...")
#para cargar un archivo .ckpt
torch.serialization.add_safe_globals([
    pl.callbacks.model_checkpoint.ModelCheckpoint
])

model = load_pretrained_model(MODEL_NAME).to(device) #carga el modelo preentrenado y lo mueve al dispositivo

#número de parámetros
total_params     = sum(p.numel() for p in model.parameters()) 
print(f"Modelo            : AdaFace {MODEL_NAME}")
print(f"Parámetros totales: {total_params:,}")
print(f"Dispositivo       : {next(model.parameters()).device}")
print("¡Modelo cargado! Iniciando extracción...\n")

#generación de embeddings
for dataset_path in datasets_paths: #recorre cada dataset
    dataset_name = os.path.basename(dataset_path)
    print(f"===== GENERANDO EMBEDDINGS DE {dataset_name} =====\n")
    data = [] #lista de embeddings y tiempos
    with torch.no_grad():
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

                row_data = {
                    "ID_Persona":person_entry.name, #identidad de la persona
                    "Ruta_Imagen":img_path, #ruta completa de la imagen
                    "Nombre_Archivo": img_entry.name, #nombre del archivo de la imagen
                }
                try:
                    img_rgb   = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #convierte a RGB
                    img_resized = cv2.resize(img_rgb, (112, 112)) #El modelo espera imágenes de 112x112
                    bgr_input = to_input(img_resized).to(device) #preprocesa la imagen al formato de entrada del modelo
                    start= time.time()
                    embedding, _ = model(bgr_input) #calcula el embedding
                    elapsed   = time.time() - start #tiempo de inferencia de la imagen
                    row_data["Embedding_AdaFace"]= embedding.squeeze().cpu().numpy().tolist() #guarda el embedding como lista
                    row_data["Tiempo_AdaFace"]= elapsed 
                except Exception as e: #si falla, se guarda None
                    print(f"Error con {img_path}: {e}")
                    row_data["Embedding_AdaFace"]= None
                    row_data["Tiempo_AdaFace"]= None

                data.append(row_data)

    df = pd.DataFrame(data)
    pkl_embeddings = f"/media/nas/carmengcamp/TFG/AdaFace/resultados/embeddings_adaface_{dataset_name}.pkl"
    df.to_pickle(pkl_embeddings) #guarda los embeddings en un .pkl

    count = df["Embedding_AdaFace"].notna().sum()
    print(f"  Embeddings generados para AdaFace: {count}")

    columnas   = {"Embedding_AdaFace": "Tiempo_AdaFace"}
    salida     = f"/media/nas/carmengcamp/TFG/AdaFace/resultados/resultados_adaface_{dataset_name}.csv"
    #evaluación para la configuración sin desconocidos
    results_df = evaluar_modelo(path_embeddings=pkl_embeddings, path_plantilla=path_plantilla, columnas=columnas, salida=salida)
    print("========================================\n")

