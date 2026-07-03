"""
GENERACIÓN DE EMBEDDINGS: DeepFace (Facenet, Facenet512, VGG-Face)

Este script recorre los datasets de imágenes de caras y genera los embeddings faciales de cada imagen con
los tres modelos de DeepFace: Facenet, Facenet512 y VGG-Face. Después, los almacena en ficheros .pkl

En segundo lugar, llama a la función "evaluar_modelo" para calcular las métricas de evaluación en la
configuración sin desconocidos.
"""
import os
import cv2

#Configuración de la GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0' 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" 


from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import time
from funcion_calcular_resultados import evaluar_modelo

#datasets
datasets_paths = ["/media/nas/carmengcamp/TFG/lfw_filtrado","/media/nas/carmengcamp/TFG/lfw_cropped","/media/nas/carmengcamp/TFG/lfw-deepfunneled_cropped","/media/nas/carmengcamp/TFG/lfw_noisy_25", "/media/nas/carmengcamp/TFG/lfw_noisy_50","/media/nas/carmengcamp/TFG/lfw_noisy_75"]


models = [ "Facenet","Facenet512","VGG-Face"]#modelos de DeepFace a evaluar
preprocessing= ["Facenet","Facenet2018","VGGFace"]#normalización correspondiente a cada modelo, en el mismo orden
path_plantilla= "/media/nas/carmengcamp/TFG/plantilla.csv" #plantilla de la configuración sin desconocidos

#número de parámetros
print("Precargando modelos en VRAM...")
for model in models: #recorre cada modelo 
    modelo = DeepFace.build_model(model) #lo carga
    keras_model = getattr(modelo, "model", modelo) #obtiene el modelo de Keras subyacente
    try:
        total = keras_model.count_params() #numéro total de parámetros del modelo
        print(f"  {model}: {total} parámetros")
    except Exception as e:
        print(f"  {model}: no se pudo obtener el número de parámetros ({e})")
print("¡Modelos cargados! Iniciando extracción...\n")

#generación de embeddings
for dataset_path in datasets_paths: #recorre cada dataset
    dataset_name = os.path.basename(dataset_path)
    print(f"===== GENERANDO EMBEDDINGS DE {dataset_name} =====\n")
    data=[] #lista de embeddings y tiempos

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
                    "ID_Persona": person_entry.name, #identidad de la persona
                    "Ruta_Imagen": img_path, #ruta completa de la imagen
                    "Nombre_Archivo": img_entry.name, #nombre del archivo de la imagen
                }
            for model, preprocess in zip(models,preprocessing): #calcula el embedding con cada modelo
                try:
                    start = time.time()
                    emb = DeepFace.represent(
                        img_path=img_array,
                        model_name=model,
                        detector_backend="skip", #no se detecta cara, no se recorta y no se alinea
                        normalization=preprocess,
                        align=False
                        )[0]["embedding"] #vector de embedding de la primera y única cara detectada
                    elapsed = time.time() - start #tiempo de inferencia de la imagen
                    #guardar resultados
                    row_data[f"Embedding_{model}"] = emb
                    row_data[f"Tiempo_{model}"] = elapsed
                except Exception as e: #si falla, se guarda None
                    print(f"Error con {img_path} ({model}): {e}")
                    row_data[f"Embedding_{model}"] = None
                    row_data[f"Tiempo_{model}"] = None
            data.append(row_data)

    df = pd.DataFrame(data)
    pkl_embeddings  = f"/media/nas/carmengcamp/TFG/Deepface/embeddings_deepface_{dataset_name}.pkl"
    df.to_pickle(pkl_embeddings) #guarda los embeddings en un .pkl

    for model in models:
        count = df[f"Embedding_{model}"].notna().sum()
        print(f"  Embeddings generados para {model}: {count}")

    columnas = {f"Embedding_{m}": f"Tiempo_{m}" for m in models} #mapea la columna de embedding a la columna de tiempo para cada modelo
    salida   = f"/media/nas/carmengcamp/TFG/Deepface/resultados_deepface_{dataset_name}.csv"
    #evaluación para la configuración sin desconocidos
    results_df = evaluar_modelo(path_embeddings=pkl_embeddings,path_plantilla=path_plantilla,columnas=columnas,salida=salida)
    print("========================================\n")

