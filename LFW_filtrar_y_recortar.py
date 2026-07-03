"""
Recorre el dataset LFW (una carpeta por persona, con sus imágenes .jpg dentro) y genera una copia filtrada en función del número mínimo de imágenes por persona (min_images_por_persona).
Opcionalmente, recorta cada imagen alrededor del centro.
  -solo_filtrar=True: solo copia las imágenes de las personas que cumplen el mínimo de fotos. No las modifica
  -solo_filtrar=False: además, recorta cada cara según "scale".
Salida:
  - "_filtrado" si solo_filtrar = True
  - "_cropped"  si solo_filtrar = False
"""

import os
import shutil
import cv2
from tqdm import tqdm
import time


#datasets_paths= ["/media/nas/datasets/face_recognition/lfw"] #para obtener LFW original
datasets_paths= ["/media/nas/datasets/face_recognition/lfw","/media/nas/datasets/face_recognition/lfw-deepfunneled"] #para obtener las versiones recortadas de LFW y LFW-deepfunneled
scale=2.2
min_images_por_persona=2
solo_filtrar=False

#función para contar imágenes de una persona
def count_images(directory): 
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len([f for f in files if f.endswith(".jpg")])
    return total

#función para recortar las imágenes alrededor del centro
def crop_face(img, scale): 
    h, w, _ = img.shape
    half_w = w/(2*scale) #tamaño de la mitad del recorte
    half_h = h/(2*scale)
    cx = w/2.0 #centro de la imagen
    cy = h/2.0
    x1 = int(round(cx - half_w)) #calcular esquinas
    y1 = int(round(cy - half_h))
    x2 = int(round(cx + half_w))
    y2 = int(round(cy + half_h))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


for dataset_path in datasets_paths:
    dataset_name = os.path.basename(dataset_path)
    print("=================================")
    print(f"Procesando dataset: {dataset_name}")
    print(f"Imágenes totales antes del filtro: {count_images(dataset_path)}")
    if solo_filtrar: #nombre del dataset final
        output_dir = f"/media/nas/carmengcamp/TFG/{dataset_name}_filtrado"
    else:
        output_dir = f"/media/nas/carmengcamp/TFG/{dataset_name}_cropped"
    os.makedirs(output_dir, exist_ok=True) #crea la carpeta de salida si no existe
    persons_ok    = 0 #número de personas con >= min_images_por_persona
    persons_skip  = 0 #número de personas descartadas
    imgs_saved    = 0 #número de fotos guardadas
    imgs_skipped  = 0 #número de imágenes que cv2 no pudo leer
    start_time= time.time()
    for person in tqdm(os.listdir(dataset_path), desc=dataset_name): #recorre cada entrada (persona) del dataset
      person_src = os.path.join(dataset_path, person)
      if not os.path.isdir(person_src):
          continue
      images = [f for f in os.listdir(person_src) if f.endswith(".jpg")] #lista de imágenes de esa persona
      if len(images)< min_images_por_persona: #si tiene menos imágenes que el mínimo requerido
        persons_skip+=1 #se descarta
        continue
      persons_ok+=1
      person_dst = os.path.join(output_dir, person) #ruta de destino para esa persona
      os.makedirs(person_dst, exist_ok=True) #crea la carpeta
      for img_name in images:
        if solo_filtrar: 
            shutil.copy2(os.path.join(person_src, img_name), os.path.join(person_dst, img_name)) #copia el archivo
            imgs_saved += 1
        else: #si queremos recortar
            img = cv2.imread(os.path.join(person_src, img_name)) #se carga la imagen
            if img is None:
                imgs_skipped += 1
                continue
            face, coords = crop_face(img, scale) #se le aplica el recorte
            cv2.imwrite(os.path.join(person_dst, img_name), face) #se guarda la imagen
            imgs_saved += 1
    elapsed= time.time()-start_time
    #se imprime información sobre el proceso
    print("=================================")
    print(f"Imágenes totales antes del filtro: {count_images(dataset_path)}")
    print(f"Personas incluidas  : {persons_ok}  |  descartadas: {persons_skip}")
    print(f"Imágenes guardadas  : {imgs_saved}  |  no leídas:   {imgs_skipped}")
    print(f"Tiempo: {elapsed:.1f}s  ({elapsed/60:.2f} min)")