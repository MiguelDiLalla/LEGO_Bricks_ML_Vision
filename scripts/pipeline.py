import os
from PIL import Image
import torch
import shutil

# Configuración inicial del entorno
def setup_environment():
    """Clona el repositorio y configura el entorno."""
    os.system("git clone https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision.git")
    os.chdir("LEGO_Bricks_ML_Vision")
    os.system("pip install -r requirements.txt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

# Pipeline de entrenamiento del modelo YOLOv8n
def train_yolo_pipeline(dataset_path, annotations_format="YOLO", epochs=50, img_size=256):
    """Configura y entrena el modelo YOLO."""
    from ultralytics import YOLO
    
    # Aseguramos que el dataset esté preparado
    dataset_dir = os.path.join(dataset_path, "processed_images")
    annotations_dir = os.path.join(dataset_path, "annotations")

    if not os.path.exists(dataset_dir) or not os.path.exists(annotations_dir):
        print("El dataset procesado o las anotaciones no existen. Verifique las rutas.")
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
    annotations_dir = os.path.join(dataset_path, "annotations")

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
    preprocess_images("../Spiled_LEGO_Bricks", "processed_images")
    train_yolo_pipeline("processed_images")
    test_model_on_real_images("YOLO_Lego_Detection/best.pt", "test_images", "results")
    visualize_results("processed_images")

if __name__ == "__main__":
    main()
