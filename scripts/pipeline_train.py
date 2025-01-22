import os
import optuna
from ultralytics import YOLO
from datetime import datetime
import logging
from tqdm import tqdm

EPOCAS = 4

# === Configuración del Logger ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Detección del dispositivo ===
def get_device():
    """
    Detecta el dispositivo adecuado para la ejecución.

    Returns:
    - str: Dispositivo a usar ("cpu", "0", "0,1").
    """
    if os.environ.get('COLAB_GPU') is not None:
        return "0"  # Colab
    elif os.path.exists("/kaggle"):  # Kaggle
        return "0,1"
    else:
        return "cpu"  # Local

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

    Parameters:
    - dataset_yaml (str): Ruta al archivo dataset.yaml.
    - pretrained_model (str): Modelo YOLO preentrenado.
    - epochs (int): Número de épocas para el entrenamiento.
    - batch_size (int): Tamaño del batch.
    - learning_rate (float): Tasa de aprendizaje inicial.
    - momentum (float): Momento para el optimizador.
    - imgsz (int): Tamaño de las imágenes de entrada.
    """
    dataset_yaml = dataset_yaml or os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")

    if not os.path.exists(dataset_yaml):
        logging.error(f"[ERROR] dataset.yaml no encontrado en {dataset_yaml}. Asegúrate de que el pipeline_setup.py lo haya generado.")
        return

    logging.info(f"[INFO] Usando dataset.yaml en: {dataset_yaml}")

    model = YOLO(pretrained_model)

    output_dir = f"regular_yolo_training/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Registrar el callback de barra de progreso
    progress_bar = ProgressBarCallback(total_epochs=epochs)
    model.add_callback("on_train_start", progress_bar.on_train_start)
    model.add_callback("on_epoch_end", progress_bar.on_epoch_end)
    model.add_callback("on_train_end", progress_bar.on_train_end)

    try:
        logging.info("[INFO] Iniciando entrenamiento regular...")
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

# === Integración de Optuna en el Pipeline ===
def run_optuna_study(dataset_yaml=None, n_trials=20):
    """
    Ejecuta un estudio de Optuna para optimizar los hiperparámetros de YOLO.

    Parameters:
    - dataset_yaml (str): Ruta al archivo dataset.yaml.
    - n_trials (int): Número de pruebas a ejecutar.
    """
    dataset_yaml = dataset_yaml or os.path.join(os.getcwd(), "working", "output", "dataset", "dataset.yaml")

    if not os.path.exists(dataset_yaml):
        logging.error(f"[ERROR] dataset.yaml no encontrado en {dataset_yaml}. Asegúrate de que el pipeline_setup.py lo haya generado.")
        return

    logging.info(f"[INFO] Usando dataset.yaml en: {dataset_yaml}")

    logging.info("[INFO] Iniciando optimización con Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Mostrar resultados
    logging.info(f"[INFO] Mejor conjunto de hiperparámetros: {study.best_params}")
    logging.info(f"[INFO] Mejor mAP50 obtenido: {study.best_value}")

    # Guardar resultados
    study.trials_dataframe().to_csv("optuna_results.csv")
    optuna.visualization.plot_optimization_history(study).write_html("optuna_optimization_history.html")


# === Función Principal ===
def main(optuna_mode=False):
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


    if optuna_mode:
        run_optuna_study(dataset_yaml, n_trials=20)
    else:
        train_model(dataset_yaml, imgsz=640)  # Tamaño de imagen predeterminado

if __name__ == "__main__":
    main(optuna_mode=False)
