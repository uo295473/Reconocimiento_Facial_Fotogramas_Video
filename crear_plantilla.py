"""
CONFIGURACIÓN SIN DESCONOCIDOS: Generación de la plantilla
Generación de plantilla de los conjuntos de galería y prueba a partir del dataset LFW recortado

Recorre el dataset LFW recortado y genera un CSV (plantilla.csv) asignando a cada persona con al menos 2 imágenes:
  - 1 imagen aleatoria como "Gallery".
  - El resto de imágenes como "Probe".
"""
import os
import pandas as pd
import random

LFW_PATH = "/media/nas/carmengcamp/TFG/lfw_cropped"
random.seed(42) #para que el proceso sea reproducible

data = []
for person_name in sorted(os.listdir(LFW_PATH)): #recorre cada persona del dataset
    person_dir = os.path.join(LFW_PATH, person_name)
    if os.path.isdir(person_dir):
        imagenes = sorted([f for f in os.listdir(person_dir) if f.endswith(('.jpg'))]) #lista de imágenes de esa persona

        #aunque el dataset ha sido filtrado previamente, comprueba que cada persona cuente con al menos dos imágenes
        if len(imagenes) >= 2:
            random.shuffle(imagenes) #mezcla las fotos

            #la primera formará parte de la galería
            ruta = os.path.join(person_dir, imagenes[0])
            data.append({"ID_Persona": person_name,"Ruta_Imagen": ruta,"Nombre_Archivo": os.path.basename(ruta),"Rol": "Gallery"})

            #el resto irá a probe
            for img in imagenes[1:]:
              ruta = os.path.join(person_dir, img)
              data.append({"ID_Persona": person_name,"Ruta_Imagen": ruta,"Nombre_Archivo": os.path.basename(ruta),"Rol": "Probe"})
        else:
            print(f"{person_name} tiene solo una foto")

df_plantilla = pd.DataFrame(data)
df_plantilla.to_csv("plantilla.csv", index=False) #crea y guarda el CSV

print(f"Plantilla creada: {len(df_plantilla)} imágenes procesadas.")
print(f"Número de personas: {df_plantilla['ID_Persona'].nunique()}")
