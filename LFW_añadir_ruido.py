"""
Recorre el dataset LFW recortado y genera una copia con ruido gaussiano añadido a cada imagen, con intensidad
controlada por el parámetro "sigma" (recibido por línea de comandos).
Cada imagen usa una semilla derivada de su nombre de archivo, de forma que
el ruido aplicado sea distinto por imagen pero determinado por la semilla
base (semilla = 42).

Uso:
    python script.py <sigma>

Salida:
    /media/nas/carmengcamp/TFG/lfw_noisy_<sigma>
"""

import os
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm

if len(sys.argv) != 2:
    print(f"Uso: python {sys.argv[0]} <sigma>")
    sys.exit(1)

ruta    = "/media/nas/carmengcamp/TFG/lfw_cropped"
semilla = 42 #ruido reproducible
sigma   = int(sys.argv[1]) #intensidad del ruido gaussiano, tomada de la línea de comandos
output_dir = f"/media/nas/carmengcamp/TFG/lfw_noisy_{sigma}" #ruta de salida
os.makedirs(output_dir, exist_ok=True) #crea la carpeta

print("=================================")
print(f"Dataset entrada : {ruta}")
print(f"Dataset salida  : {output_dir}")
print(f"Sigma           : {sigma}  |  Semilla: {semilla}")
print("=================================")

imgs_saved   = 0 #número de imagenes procesadas con éxito
imgs_skipped = 0 #número de imagenes que no se pudieron leer
start_time   = time.time()
for person in tqdm(os.listdir(ruta), desc="Añadiendo ruido"): #recorre cada carpeta de persona

    person_src = os.path.join(ruta, person) #ruta de origen de esa persona
    if not os.path.isdir(person_src):
        continue
    person_dst = os.path.join(output_dir, person) #ruta de destino de esa persona
    os.makedirs(person_dst, exist_ok=True) #crea la carpeta

    images = [f for f in os.listdir(person_src) if f.endswith(".jpg")] #lista de imágenes de esa persona
    for img_name in images:
        img = cv2.imread(os.path.join(person_src, img_name)) #carga la imagen

        if img is None: #si no se puede leer
            imgs_skipped += 1
            continue
        #añadir el ruido
        img_seed = semilla + hash(img_name) % 100000 #semilla única por imagen
        rng = np.random.default_rng(img_seed) #generador de numeros aleatorios
        arr   = np.array(img, dtype=np.float32) #convierte la imagen a float32 para sumar ruido
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape) #ruido gaussiano N(0,sigma) del mismo tamaño que la imagen
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8) #suma el ruido, recorta a [0,255] y vuelve a uint8

        cv2.imwrite(os.path.join(person_dst, img_name), noisy) #guarda la imagen
        imgs_saved += 1 #se cuenta como imagen guardada
        
elapsed = time.time() - start_time #tiempo total transcurrido
#se imprime información sobre el proceso
print("=================================")
print(f"Imágenes guardadas : {imgs_saved}  |  no leídas: {imgs_skipped}")
print(f"Tiempo: {elapsed:.1f}s  ({elapsed/60:.2f} min)")