import os
import random
import logging
import matplotlib.pyplot as plt
from PIL import Image
from scripts.pipeline import test_model_on_real_images

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_grid(input_folder: str, output_folder: str, grid_size=(3, 3)) -> None:
    """
    Genera un grid de imágenes de muestra del dataset y lo guarda en la carpeta especificada.

    Args:
        input_folder (str): Ruta de la carpeta con las imágenes de entrada.
        output_folder (str): Ruta donde se guardará el grid generado.
        grid_size (tuple): Dimensiones del grid (filas, columnas).

    Raises:
        FileNotFoundError: Si la carpeta de entrada no existe o está vacía.
    """
    if not os.path.exists(input_folder):
        logging.error("La carpeta de entrada no existe.")
        raise FileNotFoundError("Carpeta de entrada no encontrada.")

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        logging.error("No se encontraron imágenes en la carpeta de entrada.")
        raise FileNotFoundError("No se encontraron imágenes en la carpeta de entrada.")

    random.shuffle(image_files)
    selected_files = image_files[:grid_size[0] * grid_size[1]]
    fig, axes = plt.subplots(*grid_size, figsize=(grid_size[1] * 4, grid_size[0] * 4))

    for idx, img_file in enumerate(selected_files):
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path)
        row, col = divmod(idx, grid_size[1])
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"{img_file}", fontsize=10)

    for ax in axes.flat[len(selected_files):]:
        ax.axis('off')

    plt.tight_layout()
    grid_path = os.path.join(output_folder, "dataset_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Grid de dataset guardado en {grid_path}.")

def annotate_model_results(model_path: str, input_folder: str, output_folder: str, conf_threshold=0.5) -> None:
    """
    Genera imágenes con anotaciones del modelo y las guarda en la carpeta especificada.

    Args:
        model_path (str): Ruta al modelo YOLO entrenado.
        input_folder (str): Carpeta con imágenes de entrada.
        output_folder (str): Carpeta para guardar las imágenes anotadas.
        conf_threshold (float): Umbral de confianza para las detecciones.

    Raises:
        FileNotFoundError: Si la carpeta de entrada no contiene imágenes.
    """
    if not os.path.exists(input_folder):
        logging.error("La carpeta de entrada no existe.")
        raise FileNotFoundError("Carpeta de entrada no encontrada.")

    os.makedirs(output_folder, exist_ok=True)
    test_model_on_real_images(model_path, input_folder, output_folder)
    logging.info(f"Resultados anotados guardados en {output_folder}.")

def generate_comparison_grid(model_path: str, input_folder: str, output_folder: str, num_samples=5) -> None:
    """
    Crea un grid comparativo de imágenes antes y después de las predicciones del modelo.

    Args:
        model_path (str): Ruta al modelo YOLO entrenado.
        input_folder (str): Carpeta con imágenes de entrada.
        output_folder (str): Carpeta para guardar los grids generados.
        num_samples (int): Número de imágenes a comparar.

    Raises:
        FileNotFoundError: Si la carpeta de entrada no contiene imágenes.
    """
    if not os.path.exists(input_folder):
        logging.error("La carpeta de entrada no existe.")
        raise FileNotFoundError("Carpeta de entrada no encontrada.")

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        logging.error("No se encontraron imágenes en la carpeta de entrada.")
        raise FileNotFoundError("No se encontraron imágenes en la carpeta de entrada.")

    random.shuffle(image_files)
    selected_files = image_files[:num_samples]

    for img_file in selected_files:
        img_path = os.path.join(input_folder, img_file)
        result_path = os.path.join(output_folder, img_file)

        test_model_on_real_images(model_path, img_path, output_folder)
        annotated_img = Image.open(result_path)
        original_img = Image.open(img_path)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title("Original")

        axes[1].imshow(annotated_img)
        axes[1].axis('off')
        axes[1].set_title("Anotado")

        plt.tight_layout()
        comparison_path = os.path.join(output_folder, f"comparison_{img_file}")
        plt.savefig(comparison_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Grid de comparación guardado en {comparison_path}.")

def organize_presentation_folders(base_folder: str) -> None:
    """
    Crea y organiza las subcarpetas necesarias para las visualizaciones.

    Args:
        base_folder (str): Carpeta raíz para las subcarpetas.
    """
    subfolders = ["dataset_samples", "model_results", "before_after"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)
    logging.info(f"Subcarpetas creadas en {base_folder}: {', '.join(subfolders)}")

def main() -> None:
    """
    Ejecución principal del pipeline de visualización.
    """
    base_folder = "presentation"
    organize_presentation_folders(base_folder)

    dataset_folder = "datasets/processed_images"
    model_path = "YOLO_Lego_Detection/best.pt"

    create_dataset_grid(dataset_folder, os.path.join(base_folder, "dataset_samples"))
    annotate_model_results(model_path, dataset_folder, os.path.join(base_folder, "model_results"))
    generate_comparison_grid(model_path, dataset_folder, os.path.join(base_folder, "before_after"))

if __name__ == "__main__":
    main()
