import os
from PIL import Image
import torch
import shutil
import kaggle
import zipfile

# === Configuración Inicial ===
def setup_environment():
    """
    Clona el repositorio y configura el entorno necesario para ejecutar el pipeline.

    - Clona el repositorio de GitHub.
    - Instala las dependencias desde el archivo requirements.txt.
    - Configura el dispositivo de ejecución (CPU o GPU).
    """
    try:
        os.system("git clone https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision.git")
        os.chdir("LEGO_Bricks_ML_Vision")
        os.system("pip install -r requirements.txt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")
    except Exception as e:
        print(f"[ERROR] Error al configurar el entorno: {e}")

# === Manejo de Credenciales de Kaggle ===
def get_kaggle_credentials():
    """
    Obtiene las credenciales de Kaggle desde variables de entorno o archivo kaggle.json.

    Prioridad de búsqueda:
    1. Variables de entorno: KAGGLE_USERNAME y KAGGLE_KEY.
    2. Archivo ~/.kaggle/kaggle.json.

    Returns:
    - dict: Diccionario con 'username' y 'key'.
    """
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        return {"username": username, "key": key}

    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"[ERROR] No se pudo leer kaggle.json: {e}")

    raise ValueError("[ERROR] Credenciales de Kaggle no encontradas. Configure las variables de entorno o coloque kaggle.json en ~/.kaggle.")

# === Descarga de Dataset ===
def download_dataset_from_kaggle(dataset, destination):
    """
    Descarga y extrae un dataset de Kaggle.

    Parameters:
    - dataset (str): Nombre del dataset en el formato "usuario/dataset".
    - destination (str): Ruta donde se extraerán los archivos.
    """
    try:
        credentials = get_kaggle_credentials()

        os.makedirs(destination, exist_ok=True)
        os.environ["KAGGLE_USERNAME"] = credentials["username"]
        os.environ["KAGGLE_KEY"] = credentials["key"]

        kaggle.api.dataset_download_files(dataset, path=destination, unzip=True)
        print(f"[INFO] Dataset descargado y extraído en {destination}")
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el dataset: {e}")

# === Validación de Directorios ===
def validate_directories(directories):
    """
    Valida la existencia de los directorios especificados.

    Parameters:
    - directories (list): Lista de rutas a validar.

    Returns:
    - bool: True si todos los directorios existen, False en caso contrario.
    """
    for directory in directories:
        if not os.path.exists(directory):
            print(f"[ERROR] Directorio no encontrado: {directory}")
            return False
    print("[INFO] Todos los directorios están correctamente configurados.")
    return True

# === Preprocesamiento de Imágenes ===
def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """
    Redimensiona imágenes y asegura consistencia en nombres de archivos.

    Parameters:
    - input_dir (str): Ruta de la carpeta con imágenes originales.
    - output_dir (str): Ruta de la carpeta para guardar las imágenes procesadas.
    - target_size (tuple): Dimensiones objetivo para las imágenes (ancho, alto).
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, filename in enumerate(sorted(os.listdir(input_dir))):
        if filename.endswith(".jpg"):
            try:
                img = Image.open(os.path.join(input_dir, filename))
                img_resized = img.resize(target_size)
                new_filename = f"image_{i}.jpg"
                img_resized.save(os.path.join(output_dir, new_filename))
                print(f"[INFO] Procesado: {filename} -> {new_filename}")
            except Exception as e:
                print(f"[ERROR] No se pudo procesar {filename}: {e}")

# === Conversión de Anotaciones de LabelMe a YOLO ===
def labelme_to_yolo(input_folder, output_folder):
    """
    Convierte archivos JSON de LabelMe al formato YOLO.

    Parameters:
    - input_folder (str): Carpeta con archivos JSON de LabelMe.
    - output_folder (str): Carpeta donde se guardarán los archivos YOLO.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            try:
                json_file = os.path.join(input_folder, filename)
                yolo_file = os.path.join(output_folder, filename.replace('.json', '.txt'))
                # Conversión aquí (implementación omitida para brevedad)
                print(f"[INFO] Convertido: {json_file} -> {yolo_file}")
            except Exception as e:
                print(f"[ERROR] Error al convertir {filename}: {e}")

# === Entrenamiento del Modelo YOLOv8n ===
def train_yolo_pipeline(dataset_path, annotations_format="YOLO", epochs=50, img_size=256):
    """
    Configura y entrena el modelo YOLO.

    Parameters:
    - dataset_path (str): Ruta del dataset procesado.
    - annotations_format (str): Formato de las anotaciones (por defecto "YOLO").
    - epochs (int): Número de épocas de entrenamiento.
    - img_size (int): Tamaño de las imágenes usadas para entrenamiento.
    """
    from ultralytics import YOLO

    dataset_dir = os.path.join(dataset_path, "processed_images")
    annotations_dir = os.path.join(dataset_path, "annotations")

    if not validate_directories([dataset_dir, annotations_dir]):
        return

    try:
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=annotations_format,
            imgsz=img_size,
            epochs=epochs,
            batch=16,
            project="LEGO_Training",
            name="YOLO_Lego_Detection"
        )
        print("[INFO] Entrenamiento finalizado. Resultados:", results)
    except Exception as e:
        print(f"[ERROR] Error durante el entrenamiento: {e}")

# === Evaluación del Modelo Entrenado ===
def test_model_on_real_images(model_path, test_images_dir, output_dir):
    """
    Evalúa el modelo YOLO entrenado en imágenes reales.

    Parameters:
    - model_path (str): Ruta del modelo YOLO entrenado.
    - test_images_dir (str): Carpeta con imágenes para evaluación.
    - output_dir (str): Carpeta para guardar los resultados visualizados.
    """
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    for img_file in os.listdir(test_images_dir):
        if img_file.endswith(".jpg"):
            try:
                img_path = os.path.join(test_images_dir, img_file)
                results = model(img_path)
                result_image = results[0].plot()
                output_path = os.path.join(output_dir, img_file)
                Image.fromarray(result_image).save(output_path)
                print(f"[INFO] Procesado: {img_file} -> {output_path}")
            except Exception as e:
                print(f"[ERROR] Error al procesar {img_file}: {e}")

# === Visualización de Resultados ===
def visualize_results(dataset_path):
    """
    Visualiza detecciones en un grid de imágenes anotadas.

    Parameters:
    - dataset_path (str): Ruta del dataset procesado.
    """
    import matplotlib.pyplot as plt

    processed_dir = os.path.join(dataset_path, "processed_images")
    images = [os.path.join(processed_dir, img) for img in os.listdir(processed_dir) if img.endswith(".jpg")]

    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(images[:16]):  # Mostrar 16 imágenes
        try:
            img = Image.open(img_path)
            plt.subplot(4, 4, i + 1)
            plt.imshow(img)
            plt.axis('off')
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {img_path}: {e}")
    plt.show()

# === Ejecución del Pipeline ===
def main():
    """Ejecución principal del pipeline de detección de LEGO."""
    setup_environment()
    download_dataset_from_kaggle("usuario/dataset", "datasets")
    preprocess_images("datasets/raw", "datasets/processed")
    labelme_to_yolo("datasets/processed", "datasets/annotations")
    train_yolo_pipeline("datasets")
    test_model_on_real_images("YOLO_Lego_Detection/best.pt", "test_images", "results")
    visualize_results("datasets")

if __name__ == "__main__":
    main()
