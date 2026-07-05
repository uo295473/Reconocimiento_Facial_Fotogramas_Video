"""
CONFIGURACIÓN CON DESCONOCIDOS: Generación de la plantilla
Modifica plantilla.csv para convertir una parte de las personas de galería en "desconocidos".
Mueve la imagen de galeria de 200 personas al conjunto de prueba, de forma que esas personas 
dejan de estar representadas en la galería y no tienen ninguna imagen de referencia para el 
reconocimiento.
Además, crea un archivo TXT que incluye todas las personas desconocidas
"""

import pandas as pd
import random
import os

os.makedirs("/media/nas/carmengcamp/TFG/con_desconocidos", exist_ok=True) #crea una carpeta que guardará todos los resultados de la configuración sin desconocidos

path_plantilla="/media/nas/carmengcamp/TFG/plantilla.csv" #plantilla original
path_salida="/media/nas/carmengcamp/TFG/con_desconocidos/plantilla_openset.csv" #ruta de salida
path_lista_desconocidos="/media/nas/carmengcamp/TFG/con_desconocidos/lista_desconocidos.txt" #lista de desconocidos

n_desconocidos=200 #número de personas a convertir en desconocidas
semilla=42 
random.seed(semilla)#para garantizar que el proceso sea reproducible

plantilla = pd.read_csv(path_plantilla) #se carga la plantilla original en un DataFrame

total_personas = plantilla["ID_Persona"].nunique() #número total de personas distintas
total_gal_orig = len(plantilla[plantilla["Rol"].str.lower() == "gallery"]) #número de fotos en galería
total_probe_orig = len(plantilla[plantilla["Rol"].str.lower() == "probe"]) #número de fotos en el conjunto de prueba
personas_con_gal = plantilla.loc[plantilla["Rol"].str.lower() == "gallery", "ID_Persona"].unique().tolist() #número de personas representadas en la galería
total_fotos = len(plantilla) #número total de fotos

#Estadísticas antes de la generación de desconocidos
print(f"Personas totales    : {total_personas}")
print(f"Personas con galería: {len(personas_con_gal)}")
print(f"Fotos totales       : {total_fotos}")
print(f" · Galería          : {total_gal_orig}")
print(f" · Probe            : {total_probe_orig}")

desconocidos = set(random.sample(personas_con_gal, n_desconocidos)) # elige aleatoriamente n_desconocidos personas representadas en la galería
print(f"  Personas seleccionadas como desconocidas : {n_desconocidos}")
print(f"  ({n_desconocidos / total_personas * 100:.1f}% del total de personas)\n")

fotos_gal_movidas = len(plantilla[plantilla["ID_Persona"].isin(desconocidos) & (plantilla["Rol"].str.lower() == "gallery")]) #total de fotos de desconocidos en galería
fotos_probe_desconocidos = len(plantilla[plantilla["ID_Persona"].isin(desconocidos) & (plantilla["Rol"].str.lower() == "probe")]) #total de fotos de desconocidos en probe
fotos_desconocidas_total = fotos_gal_movidas + fotos_probe_desconocidos #total de fotos de desconocidos

nuevas_personas_conocidas = len(personas_con_gal) - n_desconocidos #número de personas que siguen siendo conocidas
nuevas_fotos_gal = total_gal_orig - fotos_gal_movidas #nuevo tamaño de la galería
nuevas_fotos_probe = total_probe_orig + fotos_gal_movidas #nuevo tamaño del conjunto de prueba
fotos_probe_conocidas = nuevas_fotos_probe - fotos_desconocidas_total #número de fotos conocidas en probe

#Estadísticas después del proceso
print(f"{'─'*60}")
print(f"  RESUMEN DEL NUEVO ESCENARIO")
print(f"{'─'*60}")
print(f"  Personas conocidas: {nuevas_personas_conocidas}")
print(f"  Personas desconocidas: {n_desconocidos}")
print(f"")
print(f"  Fotos en galería: {nuevas_fotos_gal}")
print(f"  Fotos en probe (total): {nuevas_fotos_probe}")
print(f"    · Probe de conocidos: {fotos_probe_conocidas}")
print(f"    · Probe de desconocidos: {fotos_desconocidas_total}")
print(f"       · Fotos de galería movidas: {fotos_gal_movidas}")
print(f"       · Fotos probe originales: {fotos_probe_desconocidos}")
print(f"")
print(f" Porcentaje de desconocidos sobre probe total: {fotos_desconocidas_total / nuevas_fotos_probe * 100:.1f}%")
print(f"{'─'*60}\n")

mask = (plantilla["ID_Persona"].isin(desconocidos) & (plantilla["Rol"].str.lower() == "gallery")) #filas de galería que pertenecen a un desconocido
plantilla.loc[mask, "Rol"] = "Probe" #esas filas pasan a ser probe
plantilla.to_csv(path_salida, index=False) #guarda el nuevo CSV
print(f"CSV generado: {path_salida}")

#Se crea el archivo de texto de desconocidos
with open(path_lista_desconocidos, "w") as f:
    for persona in desconocidos:
        f.write(f"{persona}\n") #escribe una persona desconocida por línea
print("Lista de desconocidos guardada")