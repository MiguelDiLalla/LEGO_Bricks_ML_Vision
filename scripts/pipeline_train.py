import os
import optuna
from ultralytics import YOLO
from datetime import datetime
import logging
from tqdm import tqdm
import sys
import shutil
import zipfile

EPOCAS = 4
EXPORT_DIR = "/app/data/exports/"
TRAINING_DIR = "/app/data/training/"

# === Configuración del Logger ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Detección del dispositivo ===
def get_device():
    """
    Detects the appropriate execution device.
    
    Returns:
    - "cpu" if no GPU is available.
    - "cuda:0" if a single GPU is detected (Colab).
    - "cuda:0,1" if multiple GPUs are available (Kaggle).
    """
    if os.path.exists("/proc/driver/nvidia/version"):
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if os.path.exists("/kaggle"):  # Kaggle supports multiple GPUs
            return "cuda:0,1" if cuda_devices is None else f"cuda:{cuda_devices}"
        elif "google.colab" in sys.modules:  # Colab has only 1 GPU
            return "cuda:0"
        else:
            return "cuda" if cuda_devices is None else f"cuda:{cuda_devices}"
    return "cpu"


# === Función para comprimir resultados del entrenamiento ===
def zip_training_results():
    """
    Archiva los resultados del entrenamiento en un archivo zip con timestamp.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"training_results_{timestamp}.zip"
    archive_path = os.path.join(EXPORT_DIR, archive_name)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(TRAINING_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, TRAINING_DIR))
    
    logging.info(f"[INFO] Resultados del entrenamiento archivados en {archive_path}")
    return archive_path

# === Limpieza antes del entrenamiento ===
def pre_training_cleanup():
    """
    Verifica si hay sesiones de entrenamiento previas y pregunta si se deben archivar antes de eliminar.
    """
    if os.path.exists(TRAINING_DIR) and os.listdir(TRAINING_DIR):
        user_input = input("¿Deseas exportar los resultados actuales antes de eliminarlos? (Y/N): ")
        if user_input.strip().lower() == 'y':
            zip_training_results()
        
        shutil.rmtree(TRAINING_DIR)
        logging.info("[INFO] Sesiones anteriores eliminadas.")
    os.makedirs(TRAINING_DIR, exist_ok=True)

# === Callback personalizado para barra de progreso ===
class ProgressBarCallback:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_start(self, trainer, **kwargs):
        # Inicializar barra de progreso
        self.pbar = tqdm(total=self.total_epochs, desc="Progreso del entrenamiento", unit="época")

    def on_epoch_end(self, trainer, **kwargs):
        # Actualizar barra de progreso al final de cada época
        self.pbar.update(1)
        self.pbar.set_postfix({"Última época": kwargs.get('epoch') + 1})

    def on_train_end(self, trainer, **kwargs):
        # Cerrar barra de progreso
        self.pbar.close()

# === Configuración de la Función Objetivo de Optuna ===
def objective(trial):
    """
    Función objetivo para Optuna que entrena el modelo YOLO utilizando hiperparámetros sugeridos.

    Returns:
    - mAP50 (float): Precisión media a IoU 0.5, métrica a optimizar.
    """
    # Definir espacio de búsqueda para hiperparámetros
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 8, 32, step=8)
    momentum = trial.suggest_uniform("momentum", 0.8, 0.99)
    imgsz = trial.suggest_categorical("imgsz", [320, 480, 640, 800])  # Tamaños de imagen

    # Inicializar modelo YOLO
    model = YOLO("yolov8n.pt")

    # Configurar entrenamiento
    project_name = "optuna_yolo_training"
    dataset_yaml = os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=EPOCAS,  # Épocas fijas para experimentos
            batch=batch_size,
            imgsz=imgsz,
            lr0=learning_rate,
            momentum=momentum,
            project=project_name,
            name=f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            device=get_device()
        )

        # Evaluar el modelo
        metrics = model.val()
        return metrics["mAP50"]  # Devolver mAP50 como métrica objetivo
    except Exception as e:
        logging.error(f"[ERROR] Error durante el entrenamiento en el trial {trial.number}: {e}")
        return float("nan")

# === Entrenamiento Regular (Sin Optuna) ===
def train_model(dataset_yaml=None, pretrained_model="yolov8n.pt", epochs=EPOCAS, batch_size=16, learning_rate=0.001, momentum=0.9, imgsz=640):
    """
    Entrena el modelo YOLO con hiperparámetros definidos manualmente.
    """
    pre_training_cleanup()
    dataset_yaml = dataset_yaml or os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")

    if not os.path.exists(dataset_yaml):
        logging.error(f"[ERROR] dataset.yaml no encontrado en {dataset_yaml}.")
        return

    logging.info(f"[INFO] Usando dataset.yaml en: {dataset_yaml}")
    model = YOLO(pretrained_model)
    
    output_dir = os.path.join(TRAINING_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logging.info("[INFO] Iniciando entrenamiento...")
        model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=learning_rate,
            momentum=momentum,
            project=output_dir,
            name="train",
            device=get_device()
        )
        logging.info(f"[INFO] Entrenamiento completado. Resultados guardados en {output_dir}.")
    except Exception as e:
        logging.error(f"[ERROR] Error durante el entrenamiento: {e}")




# === Función Principal ===
def main():
    """
    Ejecuta el entrenamiento con o sin Optuna.

    Parameters:
    - optuna_mode (bool): Si es True, utiliza Optuna para optimizar hiperparámetros.
    """
    if "google.colab" in sys.modules:
        print('google colab')
        dataset_yaml = os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")
        print(dataset_yaml)
    elif os.path.exists("/kaggle"):
        print('kaggle')
        dataset_yaml = os.path.join(os.getcwd(), "output", "dataset", "dataset.yaml")
        print(dataset_yaml)
    else:
        print('local')
        dataset_yaml = os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")
        print(dataset_yaml)


    dataset_yaml = os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")
    train_model(dataset_yaml, imgsz=640)

if __name__ == "__main__":
    main()
