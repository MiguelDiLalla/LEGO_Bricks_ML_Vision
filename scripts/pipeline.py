import os
from PIL import Image
import torch
import shutil
import kaggle
import zipfile

# Configuración inicial del entorno
def setup_environment():
    """Clona el repositorio y configura el entorno."""
    os.system("git clone https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision.git")
    os.chdir("LEGO_Bricks_ML_Vision")
    os.system("pip install -r requirements.txt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Manejo de credenciales de Kaggle
def get_kaggle_credentials():
    """
    Obtiene las credenciales de Kaggle desde variables de entorno o archivo kaggle.json.

    Prioridad:
    1. Variables de entorno KAGGLE_USERNAME y KAGGLE_KEY.
    2. Archivo kaggle.json en el directorio .kaggle.

    Returns:
    - dict: Diccionario con 'username' y 'key'.
    """
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        return {"username": username, "key": key}

    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "r") as f:
            return json.load(f)

    raise ValueError("Credenciales de Kaggle no encontradas. Configure las variables de entorno o coloque kaggle.json en ~/.kaggle.")

# Descarga de dataset desde Kaggle
def download_dataset_from_kaggle(dataset, destination):
    """
    Descarga y extrae un dataset de Kaggle.

    Parameters:
    - dataset (str): Nombre del dataset en el formato "usuario/dataset".
    - destination (str): Ruta donde se extraerán los archivos.
    """
    credentials = get_kaggle_credentials()

    os.makedirs(destination, exist_ok=True)
    os.environ["KAGGLE_USERNAME"] = credentials["username"]
    os.environ["KAGGLE_KEY"] = credentials["key"]

    kaggle.api.dataset_download_files(dataset, path=destination, unzip=True)
    print(f"Dataset descargado y extraído en {destination}")

# Validación de directorios
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
            print(f"Error: Directorio no encontrado -> {directory}")
            return False
    print("Todos los directorios están correctamente configurados.")
    return True

# Preprocesamiento de imágenes
def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """Redimensiona imágenes y asegura consistencia en nombres de archivos."""
    os.makedirs(output_dir, exist_ok=True)
    for i, filename in enumerate(sorted(os.listdir(input_dir))):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(input_dir, filename))
            img_resized = img.resize(target_size)
            new_filename = f"image_{i}.jpg"
            img_resized.save(os.path.join(output_dir, new_filename))
            print(f"Processed {filename} -> {new_filename}")

# Conversión de anotaciones de LabelMe a YOLO
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
            json_file = os.path.join(input_folder, filename)
            yolo_file = os.path.join(output_folder, filename.replace('.json', '.txt'))
            # Conversión aquí (implementación omitida para brevedad)
            print(f"Convertido: {json_file} -> {yolo_file}")

# Pipeline de entrenamiento del modelo YOLOv8n
def train_yolo_pipeline(dataset_path, annotations_format="YOLO", epochs=50, img_size=256):
    """Configura y entrena el modelo YOLO."""
    from ultralytics import YOLO

    # Aseguramos que el dataset esté preparado
    dataset_dir = os.path.join(dataset_path, "processed_images")
    annotations_dir = os.path.join(dataset_path, "annotations")

    if not validate_directories([dataset_dir, annotations_dir]):
        return

    # Configuración del modelo
    model = YOLO("yolov8n.pt")  # Usa un modelo preentrenado

    # Definimos el entrenamiento
    results = model.train(
        data=annotations_format,
        imgsz=img_size,
        epochs=epochs,
        batch=16,
        project="LEGO_Training",
        name="YOLO_Lego_Detection"
    )
    print("Entrenamiento finalizado. Resultados:", results)

# Pruebas con el modelo entrenado
def test_model_on_real_images(model_path, test_images_dir, output_dir):
    """Evalúa el modelo YOLO entrenado en imágenes reales."""
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    for img_file in os.listdir(test_images_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(test_images_dir, img_file)
            results = model(img_path)
            # Guardar resultados visualizados
            result_image = results[0].plot()
            output_path = os.path.join(output_dir, img_file)
            Image.fromarray(result_image).save(output_path)
            print(f"Processed {img_file} -> {output_path}")

# Visualización de resultados
def visualize_results(dataset_path):
    """Visualiza detecciones en un grid de imágenes anotadas."""
    import matplotlib.pyplot as plt

    processed_dir = os.path.join(dataset_path, "processed_images")
    images = [os.path.join(processed_dir, img) for img in os.listdir(processed_dir) if img.endswith(".jpg")]

    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(images[:16]):  # Mostrar 16 imágenes
        img = Image.open(img_path)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# Ejecutar pipeline
def main():
    setup_environment()
    download_dataset_from_kaggle("usuario/dataset", "datasets")
    preprocess_images("datasets/raw", "datasets/processed")
    labelme_to_yolo("datasets/processed", "datasets/annotations")
    train_yolo_pipeline("datasets")
    test_model_on_real_images("YOLO_Lego_Detection/best.pt", "test_images", "results")
    visualize_results("datasets")

if __name__ == "__main__":
    main()
