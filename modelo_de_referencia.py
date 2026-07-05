"""
GENERACIÓN DE EMBEDDINGS: Modelo de referencia (Entorno Torch)

Este script recorre los datasets de imágenes de caras y genera los embeddings faciales de cada imagen con
el modelo de referencia. Después, los almacena en ficheros .pkl
En segundo lugar, llama a la función "evaluar_modelo" para calcular las métricas de evaluación en la
configuración sin desconocidos.
"""
import os

#Configuración de la GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
from tqdm import tqdm
from funcion_calcular_resultados import evaluar_modelo


#Definición del modelo
class Mish(nn.Module):
    """Mish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish activation function.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return x * torch.tanh(F.softplus(x))
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channel: int, reduction: int) -> None:
        """Initialize SE block.

        Args:
            channel: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with channel attention applied
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlockReID(nn.Module):
    """Residual block with SE attention for face ReID network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        reduction: int,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """Initialize ReID residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            reduction: SE block reduction ratio
            downsample: Optional downsampling module
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-03)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = Mish()
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) if downsample is not None else nn.ZeroPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.se = SEBlock(out_channels, out_channels // reduction)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ReID residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection and SE attention
        """
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return out
class L2Normalization(nn.Module):
    """L2 normalization layer for embeddings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization to input tensor.

        Args:
            x: Input tensor

        Returns:
            L2 normalized tensor
        """
        return F.normalize(x, p=2, dim=1)

class Stem(nn.Module):
    """Initial stem layer for face ReID network."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize stem layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stem layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor after convolution, batch norm, and activation
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class Bottleneck(nn.Module):
    """Bottleneck layer for feature dimension reduction."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize bottleneck layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_features, eps=1e-03)
        self.fc = nn.Linear(in_features * 7 * 7, out_features, bias=True)
        self.bn2 = nn.BatchNorm1d(out_features, eps=1e-03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck layer.

        Args:
            x: Input tensor

        Returns:
            Bottleneck features tensor
        """
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        return x

class FaceReIDEncoder(nn.Module):
    """Face ReID encoder model from hyperion."""

    def __init__(self, embedding_size: int = 128) -> None:
        """Initialize face ReID encoder.

        Args:
            embedding_size: Dimension of output embedding vector
        """
        super().__init__()
        self.stem = Stem(3, 64)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=2, reduction=8)
        self.layer2 = self._make_layer(64, 128, blocks=3, stride=2, reduction=8)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2, reduction=8)
        self.layer4 = self._make_layer(256, 512, blocks=4, stride=2, reduction=8)
        self.bottleneck = Bottleneck(512, embedding_size)
        self.l2_norm = L2Normalization()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        reduction: int,
    ) -> nn.Module:
        layers = []
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, eps=1e-03),
            )
        else:
            downsample = None

        layers.append(ResidualBlockReID(in_channels, out_channels, stride, reduction, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlockReID(out_channels, out_channels, stride=1, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through face ReID encoder.

        Args:
            x: Input face image tensor

        Returns:
            L2 normalized face embedding tensor
        """
        x = (x - 127.5) / 128.0
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bottleneck(x)
        x = self.l2_norm(x)
        return x
    


def preprocess(cv2_img):
    img = cv2.resize(cv2_img, TARGET_SIZE) #Redimensionar al tamaño del modelo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convertir a RGB si viene de cv2 (BGR)
    img = img.astype(np.float32)  
    img = np.transpose(img, (2, 0, 1)) #HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)

#datasets
datasets_paths = ["/media/nas/carmengcamp/TFG/lfw_filtrado","/media/nas/carmengcamp/TFG/lfw_cropped","/media/nas/carmengcamp/TFG/lfw-deepfunneled_cropped","/media/nas/carmengcamp/TFG/lfw_noisy_25", "/media/nas/carmengcamp/TFG/lfw_noisy_50","/media/nas/carmengcamp/TFG/lfw_noisy_75"]
path_plantilla = "/media/nas/carmengcamp/TFG/plantilla.csv" #plantilla de la configuración sin desconocidos
MODEL_PATH     = "/media/nas/carmengcamp/TFG/models/empresa/face_reid/0.1.0/face_reid.pth" #ruta del modelo

TARGET_SIZE    = (112, 112) #tamaño de entrada esperado por el modelo
columnas       = {"Embedding_empresa": "Tiempo_empresa"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #usa GPU si está disponible, si no CPU

if not os.path.exists(MODEL_PATH): #si el fichero de pesos no existe
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

print("Torch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("Torch NO está usando GPU")

print("Precargando modelo en VRAM...")
model = FaceReIDEncoder() #instancia la arquitectura
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) #carga los pesos entrenados
model.to(DEVICE).eval() #mueve el modelo al dispositivo y lo pone en modo evaluación
print("¡Modelo cargado! Iniciando extracción...\n")

#número de parametros
print("============ MODELO FACE ReID ============")
total_params = sum(p.numel() for p in model.parameters())
print(f"Modelo            : FaceReIDEncoder (ResNet custom)")
print(f"Parámetros totales: {total_params:,}")
print(f"Dispositivo       : {next(model.parameters()).device}")
print("==========================================\n")

#generación de embeddings
for dataset_path in datasets_paths: #recorre cada dataset
    dataset_name = os.path.basename(dataset_path)
    print(f"===== Generando embeddings de {dataset_name} =====")
    data = [] #lista de embeddings y tiempos
    for person_entry in tqdm(os.scandir(dataset_path), desc=dataset_name): #recorre cada persona del dataset
        
        if not person_entry.is_dir():
            continue

        for img_entry in os.scandir(person_entry.path): #recorre cada imagen de esa persona
            if not img_entry.is_file():
                continue

            img_path = img_entry.path
            img_array = cv2.imread(img_path) #carga la imagen
            
            if img_array is None:
                print(f"  Error al leer la imagen: {img_path}")
                continue

            try:
                with torch.no_grad(): #desactiva el cálculo de gradientes
                    tensor = preprocess(img_array) #preprocesa la imagen 
                    start = time.time()
                    embedding = model(tensor) #calcula el embedding
                    elapsed = time.time() - start 
                data.append({
                    "ID_Persona":        person_entry.name,  #identidad de la persona
                    "Ruta_Imagen":       img_entry.path, #ruta completa de la imagen
                    "Nombre_Archivo":    img_entry.name, #nombre del archivo de la imagen
                    "Embedding_empresa": embedding.cpu().numpy().flatten(), #embedding
                    "Tiempo_empresa":    elapsed, #tiempo de inferencia de la imagen
                })
            except Exception as e: #si falla, se guarda None
                data.append({
                    "ID_Persona":        person_entry.name,  #identidad de la persona
                    "Ruta_Imagen":       img_entry.path, #ruta completa de la imagen
                    "Nombre_Archivo":    img_entry.name, #nombre del archivo de la imagen
                    "Embedding_empresa": None,
                    "Tiempo_empresa":    None,
                })

    df = pd.DataFrame(data)
    print(f"Embeddings generados : {len(data)}")
    pkl_embeddings = f"/media/nas/carmengcamp/TFG/Empresa/embeddings_empresa_{dataset_name}.pkl"
    df.to_pickle(pkl_embeddings) #guarda los embeddings en un .pkl
    print(f"¡Embeddings guardados en {pkl_embeddings}!")

    #evaluación para la configuración sin desconocidos
    salida = f"/media/nas/carmengcamp/TFG/Empresa/resultados_empresa_{dataset_name}.csv"
    results_df = evaluar_modelo(path_embeddings=pkl_embeddings,path_plantilla=path_plantilla,columnas=columnas,salida=salida)
    print("==========================================\n")


