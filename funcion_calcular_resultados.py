"""
CONFIGURACIÓN SIN DESCONOCIDOS: Evaluación de modelos de reconocimiento facial

Este script evalúa el rendimiento de los modelos de reconcocimiento facial en un escenario sin desconocidos.
Para ello, mide la distancia entre cada embedding del conjunto de prueba y todas las imágenes
del conjunto de galería mediante dos funciones de distancia (euclídea o
coseno). Después, calcula el Rank-1 y Rank-5.

El script comienza cargando los embeddings, almacenados en un fichero .pkl y el archivo plantilla.csv.
Después, separa los conjuntos de galería y de prueba, hace comprobaciones
de integridad y realiza el proceso de evaluación. Finalmente, guarda un CSV con los resultados por modelo y métrica.
"""

import numpy as np
import pandas as pd
import time
import os

#distancia euclídea
def euclidean(a, b): 
    return np.linalg.norm(a - b)

#distancia coseno
def cosine_distance(a, b): 
    dot_product = np.dot(a, b)
    source_norm = np.linalg.norm(a)
    test_norm = np.linalg.norm(b)
    distances = 1 - dot_product / (source_norm * test_norm)
    return distances

def evaluate(gallery_df, probe_df, embedding_col, distance_fn):
    """
    Calcula Rank-1 y Rank-5 comparando cada imagen de probe contra toda la galería.
    """
    start= time.time()

    rank1 = 0 #número de veces en las que la identidad correcta queda en la primera posición
    rank5 = 0 #número de veces en las que la identidad correcta queda en la quinta posición
    total = len(probe_df) #número de imágenes evaluadas

    gallery_embeddings = gallery_df[embedding_col].values #embeddings de la galería
    gallery_ids = gallery_df["ID_Persona"].values #identidades correspondientes a esos embeddings

    for _, probe_row in probe_df.iterrows(): #recorre cada imagen de probe
        probe_emb = probe_row[embedding_col] #embedding
        true_id = probe_row["ID_Persona"] #identidad real

        distances = []
        for g_emb, g_id in zip(gallery_embeddings, gallery_ids):
            d = distance_fn(probe_emb, g_emb)#calcular distancias con cada uno de los elementos de gallery
            distances.append((g_id, d))

        distances.sort(key=lambda x: x[1])#ordenar de menor a mayor distancia
        ranked_ids = [x[0] for x in distances] #lista de identidades ordenada por cercanía

        if ranked_ids[0] == true_id:#rank-1: la más cercana es la identidad correcta
            rank1+=1
        if true_id in ranked_ids[:5]:#rank-5: la identidad correcta se encuentra entre las 5 más cercanas
            rank5+=1
    tiempo_metrica = time.time()-start #tiempo total para evaluar esta métrica
    return rank1/total, rank5/total, tiempo_metrica #devuelve rank1, rank5 y el tiempo total

def evaluar_modelo(path_embeddings,path_plantilla,columnas,salida,metrics=None,probe_gallery=False):
  """
  PARÁMETROS DE ENTRADA:
    path_embeddings,        # PKL con los embeddings
    path_plantilla,         # CSV con roles (Gallery/Probe)
    embeddings,             # {columa_embedding_modelo:columna_tiempo_modelo}
    salida,                 # nombre del CSV de resultados
    metrics=None,           # dict {nombre: función}, por defecto distancia euclídea y coseno
    probe_gallery=False,    # descargar PKL de probe y gallery
  """
  if metrics is None:
    metrics = {"euclidean": euclidean, "cosine": cosine_distance}

  df = pd.read_pickle(path_embeddings) #carga los embeddings
  plantilla = pd.read_csv(path_plantilla) #carga la plantilla
  df_final  = pd.merge(df, plantilla, on=["ID_Persona", "Nombre_Archivo"]) #junta ambos DataFrames
  
  for emb_col in columnas:
    df_final[emb_col] = df_final[emb_col].apply(np.array) #asegura que cada embedding sea un array de numpy
  
  #Imprime información del proceso
  print("---CSV EMBEDDINGS")
  print(f"   Número de embeddings (9164): {len(df)}")
  print(f"   Número de personas (1680): {df['ID_Persona'].nunique()}")
  print("---CSV Y PLANTILLA")
  print(f"   Número de embeddings (9164): {len(df_final)}")
  print(f"   Número de personas (1680): {df_final['ID_Persona'].nunique()}")

  gallery = df_final[df_final["Rol"] == "Gallery"].copy() #subconjunto de galería
  probe= df_final[df_final["Rol"] == "Probe"].copy() #subconjunto de prueba
  print("---PROBE Y GALERIA CREADOS")
  print(f"   Número embeddings en galería (1680): {len(gallery)}")
  print(f"   Número embeddings en probe (7484): {len(probe)}")

  set_gallery = set(gallery["Nombre_Archivo"]) #nombres de archivo únicos en galería
  set_probe = set(probe["Nombre_Archivo"]) #nombres de archivo únicos en probe
  
  #comprobaciones:
  errores=[]
  if len(probe)!=len(df_final)-len(gallery):
    errores.append("tamaño probe incorrecto")
  if len(set_gallery)!=len(gallery):
    errores.append("duplicados en gallery")
  if len(set_probe)!=len(probe):
    errores.append("duplicados en probe")
  if set_gallery & set_probe: #si la intersección no está vacía, alguna imagen está en ambos conjuntos
    errores.append("solapamiento probe/gallery")
  if errores: #solo imprime errores si los hay
    print(f"      ERRORES: {', '.join(errores)}")

  if probe_gallery: #si se quiere descargar .pkl de probe y gallery
    print("---CREANDO CSV PROBE Y GALLERY")
    nombre_base = os.path.splitext(os.path.basename(path_embeddings))[0] #nombre del pkl de entrada sin extensión
    gallery.to_pickle(f"gallery_{nombre_base}.pkl")
    probe.to_pickle(f"probe_{nombre_base}.pkl")

  print("---NORMA DE LOS EMBEDDINGS")
  for emb_col in columnas.keys():
    print(f"   {emb_col}: {np.linalg.norm(gallery[emb_col].iloc[0])}") #norma del primer embedding de galería, para ver si tiene norma unitaria o no
  
  print("---EVALUACIÓN")
  results = []
  for emb_col, tiempo_col in columnas.items(): #recorre cada modelo (columna de embedding y columna de tiempo)
      media = df_final[tiempo_col].mean()#tiempo medio de inferencia
      print(f"   Tiempo medio {emb_col}: {media} s/imagen")
      for metric_name, metric_fn in metrics.items(): #recorre cada métrica de distancia
          r1,r5,tiempo_metrica = evaluate(gallery, probe, emb_col, metric_fn) #realiza la evaluación de esta combinación de modelo y distancia
          results.append({"embedding": emb_col,"metric": metric_name,"rank1": r1, "rank5": r5,"Tiempo metrica":tiempo_metrica, "Tiempo medio modelo": media})

  results_df = pd.DataFrame(results)
  results_df.to_csv(salida, index=False) #guarda los resultados en CSV
  return results_df

