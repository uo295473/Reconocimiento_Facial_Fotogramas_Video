"""
CONFIGURACIÓN CON DESCONOCIDOS: Evaluación de modelos de reconocimiento facial

Este script calcula los resultados de evaluación para el escenario con desconocidos, a partir de
embeddings ya calculados y guardados previamente en ficheros .pkl.
"""
from funcion_calcular_resultados_openset import evaluar_modelo 
import os


dataset_names=["lfw_cropped", "lfw_filtrado", "lfw-deepfunneled_cropped", "lfw_noisy_25","lfw_noisy_50","lfw_noisy_75"]
path_plantilla = "/media/nas/carmengcamp/TFG/con_desconocidos/plantilla_openset.csv"
path_unknown= "/media/nas/carmengcamp/TFG/con_desconocidos/lista_desconocidos.txt"


def get_configs(name):
    #rutas a los .pkl creados
    base_dp  = f"/media/nas/carmengcamp/TFG/Deepface"
    base_emp = f"/media/nas/carmengcamp/TFG/Empresa"
    base_arc = f"/media/nas/carmengcamp/TFG/ArcFace"
    base_ada = f"/media/nas/carmengcamp/TFG/AdaFace/resultados"
    base_ars = f"/media/nas/carmengcamp/TFG/ArcFace_small"
    #ruta de salida
    base_out = f"/media/nas/carmengcamp/TFG/con_desconocidos"

    return [
        {"model": "Deepface", "pkl": f"{base_dp}/embeddings_deepface_{name}.pkl", "salida": f"{base_out}/Deepface/resultados_deepface_{name}.csv", "columnas": {"Embedding_Facenet": "Tiempo_Facenet", "Embedding_Facenet512": "Tiempo_Facenet512", "Embedding_VGG-Face": "Tiempo_VGG-Face"}},
        {"model": "Empresa", "pkl": f"{base_emp}/embeddings_empresa_{name}.pkl", "salida": f"{base_out}/Empresa/resultados_empresa_{name}.csv", "columnas": {"Embedding_empresa": "Tiempo_empresa"}},
        {"model": "ArcFace", "pkl": f"{base_arc}/embeddings_arcface_{name}.pkl", "salida": f"{base_out}/ArcFace/resultados_arcface_{name}.csv", "columnas": {"Embedding_ArcFace": "Tiempo_ArcFace"}},
        {"model": "AdaFace", "pkl": f"{base_ada}/embeddings_adaface_{name}.pkl", "salida": f"{base_out}/AdaFace/resultados_adaface_{name}.csv", "columnas": {"Embedding_AdaFace": "Tiempo_AdaFace"}},
        {"model": "ArcFace small", "pkl": f"{base_ars}/embeddings_arcface_small_{name}.pkl", "salida": f"{base_out}/ArcFace_small/resultados_arcface_small_{name}.csv", "columnas": {"Embedding_ArcFace_s": "Tiempo_ArcFace_s"}},
    ]

for name in dataset_names:
    print("==========================")
    print(f"Calculando resultados de: {name}")
    for cfg in get_configs(name):
        print(f"Model: {cfg['model']}")
        os.makedirs(os.path.dirname(cfg["salida"]), exist_ok=True)  # crea la carpeta de salida si no existe
        evaluar_modelo(path_embeddings=cfg["pkl"],path_plantilla=path_plantilla,path_unknown=path_unknown,columnas=cfg["columnas"],salida=cfg["salida"])
        print("==========================")
