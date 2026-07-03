from funcion_calcular_resultados import evaluar_modelo 
import os
"""
programa para calcular los resultados a partir de un .pkl ya creado
"""

dataset_name="lfw_filtrado"
pkl_embeddings = "/media/nas/carmengcamp/TFG/embeddings_deepface_lfw_filtrado.pkl"
path_plantilla = "/media/nas/carmengcamp/TFG/plantilla.csv"
models = ["Facenet", "Facenet512", "VGG-Face"]
salida   = f"/media/nas/carmengcamp/TFG/resultados_deepface_{dataset_name}.csv"

columnas = {f"Embedding_{m}": f"Tiempo_{m}" for m in models}

results_df = evaluar_modelo(path_embeddings=pkl_embeddings,path_plantilla=path_plantilla,columnas=columnas,salida=salida,)
