"""
CONFIGURACIÓN CON DESCONOCIDOS: Evaluación de modelos de reconocimiento facial

Este script evalúa el rendimiento de los modelos de reconcocimiento facial en un escenario con desconocidos, en
el que el sistema tiene que ser capaz de rechazar identidades no conocidas.
Para ello, mide la distancia entre cada embedding del conjunto de prueba y todas las imágenes
del conjunto de galería mediante dos funciones de distancia (euclídea o
coseno). Después, se compara con un umbral de rechazo, calculado a partir de las distancias mínimas obtenidas. 
  - Si la distancia mínima supera el umbral, se rechaza y la identidad se considera desconocida.
  - Si no, la identidad se considera conocida.
Finalmente, calcula las métricas de esta evaluación: TPIR, FPIR, Rank-5 sobre conocidos y la precisión global

El script comienza cargando los embeddings, almacenados en un fichero .pkl y los archivos plantilla_openset.csv y lista_desconocidos.txt.
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

def cargar_desconocidos(path_unknown: str):
    """
    Lee el fichero de personas desconocidas
    """
    with open(path_unknown, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }

def calcular_umbral(min_distances, k):
    """
    calcula el umbral como media+k·σ sobre las distancias mínimas de cada probe
    """
    media= float(np.mean(min_distances))
    sigma= float(np.std(min_distances))
    umbral= media+k*sigma
    return umbral, media, sigma

def evaluate_open_set(gallery_df, probe_df, embedding_col, distance_fn, ids_desconocidos, k):
    """
    Calcula distancias a toda la galeria y guarda la minima. Después, calcula el umbral de rechazo.
    Finalmente, calcula las métricas correspondientes
    """
    
    tpir = 0 #número de aciertos entre conocidos no rechazados
    rank5 = 0 #número de aciertos rank-5 entre conocidos no rechazados
    rechazo_ok=0 #número de desconocidos correctamente rechazados
    total_conocidos = 0 #número total de conocidos en el conjunto de prueba
    total_desconocidos = 0 #número total de desconocidos en el conjunto de prueba
    aciertos_global = 0 #aciertos totales (identificación correcta y rechazo correcto de desconocidos)

    total = len(probe_df) #número total de imágenes en el conjunto de prueba

    gallery_embeddings = gallery_df[embedding_col].values #embeddings de la galería
    gallery_ids = gallery_df["ID_Persona"].values #identidades correspondientes a esos embeddings

    min_distancias = [] #distancias mínimas
    rankings = [] #ranking completo de identidades ordenado por distancia, por cada probe

    start= time.time()
    for _, probe_row in probe_df.iterrows(): #recorre cada imagen de probe
        probe_emb = probe_row[embedding_col] #embedding

        distances = [] 
        for g_emb, g_id in zip(gallery_embeddings, gallery_ids):
            d = distance_fn(probe_emb, g_emb) #calcular distancias con cada uno de los elementos de gallery
            distances.append((g_id, d))

        distances.sort(key=lambda x: x[1])#ordenar de menor a mayor distancia
        min_distancias.append(distances[0][1]) #guarda la distancia mínima
        rankings.append([x[0] for x in distances]) #guarda el ranking de identidades para esta consulta

    umbral, media, sigma = calcular_umbral(min_distancias, k) #se calcula el umbral de rechazo
    for (_, probe_row), min_dist, ranked_ids in zip(probe_df.iterrows(), min_distancias, rankings):
        true_id = probe_row["ID_Persona"] #identidad real
        es_desconocido= true_id in ids_desconocidos #si la persona está en la lista de desconocidos
        rechazado = min_dist>=umbral #si el sistema decide rechazarla

        if es_desconocido:
            total_desconocidos+=1
            if rechazado: #era desconocida y se rechazó
                rechazo_ok+=1
                aciertos_global+=1
        else:
            total_conocidos+=1
            if not rechazado: #solo se evalúa identificación si el sistema no la rechazó
                if ranked_ids[0] == true_id:#TPIR
                    tpir+=1
                    aciertos_global+=1
                if true_id in ranked_ids[:5]:#rank-5
                    rank5+=1
    tpir_total= tpir / total_conocidos if total_conocidos > 0 else 0
    rank5_conocidos= rank5 / total_conocidos if total_conocidos > 0 else 0
    if total_desconocidos>0:
      fpir= 1-(rechazo_ok / total_desconocidos)
    else:
      fpir=0
    acc_global= aciertos_global / total
    tiempo_metrica = time.time()-start
    return tpir_total, rank5_conocidos, fpir, acc_global,tiempo_metrica

def evaluar_modelo(path_embeddings,path_plantilla,path_unknown, columnas,salida,metrics=None,k=2,probe_gallery=False):
  """
    path_embeddings,        # PKL con los embeddings
    path_plantilla,         # CSV con roles (Gallery/Probe)
    path_unknown,           # TXT con la lista de personas
    embeddings,             # {columa_embedding_modelo:columna_tiempo_modelo}
    salida,                 # nombre del CSV de resultados
    metrics=None,           # dict {nombre: función}, por defecto distancia euclídea y coseno
    k=2,                    # parámetro para calcular el umbral de rechazo
    probe_gallery=False,    # descargar PKL de probe y gallery
  """
  if metrics is None:
    metrics = {"euclidean": euclidean, "cosine": cosine_distance}

  df = pd.read_pickle(path_embeddings) #carga los embeddings
  plantilla = pd.read_csv(path_plantilla) #carga la plantilla
  ids_desconocidos = cargar_desconocidos(path_unknown) #carga la lista de personas desconocidas
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
  print(f"   Número embeddings en galería (1480): {len(gallery)}")
  print(f"   Número embeddings en probe (7684): {len(probe)}")

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
    nombre_base = os.path.splitext(os.path.basename(path_embeddings))[0]
    gallery.to_pickle(f"gallery_{nombre_base}.pkl")
    probe.to_pickle(f"probe_{nombre_base}.pkl")

  print("---NORMA DE LOS EMBEDDINGS")
  for emb_col in columnas.keys():
    print(f"   {emb_col}: {np.linalg.norm(gallery[emb_col].iloc[0])}") #norma del primer embedding de galería, para ver si tiene norma unitaria o no
  
  print("---EVALUACIÓN")
  results = []
  for emb_col, tiempo_col in columnas.items(): #recorre cada modelo (columna de embedding y columna de tiempo)
      media_modelo = df_final[tiempo_col].mean() #tiempo medio de inferencia
      print(f"   Tiempo medio {emb_col}: {media_modelo} s/imagen")
      for metric_name, metric_fn in metrics.items(): #recorre cada métrica de distancia
          r1, r5,fpir,acc_global,tiempo_metrica= evaluate_open_set(gallery, probe, emb_col, metric_fn, ids_desconocidos,k) #realiza la evaluación de esta combinación de modelo y distancia
          results.append({"embedding": emb_col,"metric": metric_name,"TPIR": r1, "rank5_conocidos": r5,"FPIR": fpir, "acc_global":acc_global,"Tiempo metrica":tiempo_metrica,"Tiempo medio modelo": media_modelo})
  results_df = pd.DataFrame(results)
  results_df.to_csv(salida, index=False) #guarda los resultados en CSV
  return results_df

