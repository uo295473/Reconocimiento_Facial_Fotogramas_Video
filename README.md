# RECONOCIMIENTO FACIAL BASADO EN FOTOGRAMAS DE VÍDEOS

Comparación de 7 modelos de reconocimiento facial en la tarea de identificación facial con LFW, en dos configuraciones distintas: sin desconocidos y con desconocidos.

## Modelos evaluados

- Modelo de referencia: `face_reid.pth`
- VGG-Face, FaceNet, FaceNet512 (implementados con [DeepFace](https://github.com/serengil/deepface/tree/master/deepface))
- ArcFace iResNet-50 (modelo `w600k_r50.onnx` de [InsightFace](https://github.com/deepinsight/insightface/tree/master): https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view
- ArcFace MobileFaceNet (modelo `w600k_mbf.onnx` de [InsightFace](https://github.com/deepinsight/insightface/tree/master)): https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view
- AdaFace: https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view

## Conjunto de datos
- LFW original: https://www.kaggle.com/datasets/sayuksh/labeled-faces-in-the-wild-home
- *LFW-deepfunneled*: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data
### Generación de las versiones de LFW
- `LFW_filtrar_y_recortar.py`
- `LFW_añadir_ruido.py`

## Configuración sin desconocidos

1. `crear_plantilla.py`: genera `plantilla.csv` 
2. `funcion_calcular_resultados.py`: implementa la función `evaluar_modelo()`
3. Scripts de embeddings (uno por modelo, llaman a `evaluar_modelo()`):
   - `modelo_de_referencia.py`
   - `deepface.py` (VGG-Face, FaceNet, FaceNet512)
   - `arcface.py`: ArcFace iResNet-50
   - `arcface_small.py`: ArcFace MobileFaceNet
   - `adaface.py`
4. `juntar_resultados.py`: une todos los CSV en `resultados_completo.xlsx`

## Configuración con desconocidos

1. `crear_plantilla_openset.py`: genera `plantilla_openset.csv` y `lista_desconocidos.txt`
2. `funcion_calcular_resultados_openset.py`: función `evaluar_modelo()`
3. `calcular_resultados_openset.py`: reutiliza los `.pkl` ya generados en la configuracion sin desconocidos y calcula los resultados para esta configuración
4. `con_desconocidos/juntar_resultados_openset.py`:  une todos los CSV en `resultados_openset_completo.xlsx`

## Gráficas

`crear_graficas_resultados.ipynb` genera las gráficas analizadas en la memoria a
partir de `resultados_completo.xlsx` y `resultados_openset_completo.xlsx`.

## Requisitos

```bash
pip install -r requirements_entorno_conda.txt #modelos implementados en Deepface.py
pip install -r requirements_entorno_torch.txt #resto de modelos
```

AdaFace requiere clonar su repositorio (https://github.com/mk-minchul/AdaFace),
mover `adaface.py` y `funcion_calcular_resultados.py` dentro, crear una
carpeta `pretrained` y colocar ahí los pesos del modelo.
