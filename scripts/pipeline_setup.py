import os
import shutil
from sklearn.model_selection import train_test_split
import json
from pprint import pprint
import albumentations as A
import cv2
import yaml
import sys

# === Configuración Inicial ===
def detect_hardware():
    """
    Detects the harware acereation avalaible for the current environment.

    Returns:
    0, 0,1 or cpu
    """
    

def Load_dataset(dataset_name, base_path="/workspace/output"):
    """
    download and unzip the slected dataset
    """
   
def verify_dataset_structure(raw_images_path, raw_labels_path):
    """
    Verifica la existencia de las carpetas requeridas en el dataset y muestra estadísticas iniciales.

    Parameters:
    - raw_images_path (str): Ruta a las imágenes crudas.
    - raw_labels_path (str): Ruta a las etiquetas crudas.
    """
    required_folders = [raw_images_path, raw_labels_path]
    summary = {}
    for folder in required_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"[ERROR] Carpeta requerida no encontrada: {folder}")

        num_files = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        if num_files == 0:
            raise ValueError(f"[ERROR] La carpeta {folder} está vacía.")
        summary[folder] = num_files

    pprint({"Dataset Estructura": summary})

def create_preprocessing_structure(output_dir="/kaggle/working/output"):
    """
    Crea la estructura de carpetas para PREPROCESSING/.

    Parameters:
    - output_dir (str): Ruta base para la carpeta PREPROCESSING/.
    """
    os.makedirs(output_dir, exist_ok=True)
    subfolders = [
        "dataset/images/train", "dataset/images/val", "dataset/images/test",
        "dataset/labels/train", "dataset/labels/val", "dataset/labels/test",
        "test_images"
    ]
    for subfolder in subfolders:
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
    print(f"[INFO] Estructura de carpetas creada en {output_dir}.")

def copy_and_partition_data(input_images, input_labels, output_dir):
    """
    Copia imágenes y etiquetas a las carpetas correspondientes y realiza la partición de datos.

    Parameters:
    - input_images (str): Carpeta de imágenes de entrada.
    - input_labels (str): Carpeta de etiquetas de entrada.
    - output_dir (str): Carpeta base para PREPROCESSING/.
    """
    images = sorted([f for f in os.listdir(input_images) if f.endswith(".jpg")])
    labels = sorted([f for f in os.listdir(input_labels) if f.endswith(".txt")])

    if len(images) != len(labels):
        raise ValueError("[ERROR] Número de imágenes y etiquetas no coincide.")

    image_paths = [os.path.join(input_images, img) for img in images]
    label_paths = [os.path.join(input_labels, lbl) for lbl in labels]

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(temp_imgs, temp_lbls, test_size=0.33, random_state=42)

    partitions = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls)
    }

    for partition, (imgs, lbls) in partitions.items():
        for img, lbl in zip(imgs, lbls):
            shutil.copy(img, os.path.join(output_dir, f"dataset/images/{partition}/"))
            shutil.copy(lbl, os.path.join(output_dir, f"dataset/labels/{partition}/"))

    pprint({"Partición Completada": {partition: len(imgs) for partition, (imgs, _) in partitions.items()}})

def augment_data(input_images, input_labels, output_dir, num_augmentations=2):
    """
    Aplica aumentaciones al dataset y guarda imágenes y etiquetas aumentadas.

    Parameters:
    - input_images (str): Carpeta de imágenes originales.
    - input_labels (str): Carpeta de etiquetas en formato YOLO.
    - output_dir (str): Carpeta donde se guardarán los datos aumentados.
    - num_augmentations (int): Número de versiones aumentadas por imagen.
    """
    aug_images_dir = os.path.join(output_dir, "augmented_images")
    aug_labels_dir = os.path.join(output_dir, "augmented_labels")
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Resize(height=640, width=640),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    images = sorted([f for f in os.listdir(input_images) if f.endswith(".jpg")])
    for img_file in images:
        img_path = os.path.join(input_images, img_file)
        label_path = os.path.join(input_labels, img_file.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Skipping corrupted image: {img_path}")
            continue  # Skip this image and move to the next
        
        bboxes, class_labels = load_labels(label_path)

        for i in range(num_augmentations):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            aug_image_path = os.path.join(aug_images_dir, f"{img_file.split('.')[0]}_aug{i}.jpg")
            cv2.imwrite(aug_image_path, aug_image)

            aug_label_path = os.path.join(aug_labels_dir, f"{img_file.split('.')[0]}_aug{i}.txt")
            save_labels(aug_label_path, aug_bboxes, aug_labels)

    print(f"[INFO] Augmented data saved to {output_dir}.")



def copy_augmented_to_train(augmented_dir, output_path):
    """
    Copia los datos aumentados a las subcarpetas correspondientes de 'train'.

    Parameters:
    - augmented_dir (str): Directorio que contiene imágenes y etiquetas aumentadas.
    - output_path(str): Ruta base para la salida.
    """
    aug_images_dir = os.path.join(augmented_dir, "augmented_images")
    aug_labels_dir = os.path.join(augmented_dir, "augmented_labels")
    train_images_dir = os.path.join(output_path, "dataset/images/train")
    train_labels_dir = os.path.join(output_path, "dataset/labels/train")

    for img_file in os.listdir(aug_images_dir):
        shutil.copy(os.path.join(aug_images_dir, img_file), train_images_dir)

    for label_file in os.listdir(aug_labels_dir):
        shutil.copy(os.path.join(aug_labels_dir, label_file), train_labels_dir)

    print(f"[INFO] Augmented data merged into train set at {output_path}.")

def create_dataset_yaml(output_path, num_classes, class_names):
    """
    Creates a dataset.yaml file with absolute paths for YOLO training.

    Parameters:
    - output_path (str): Base directory where the dataset.yaml file will be saved.
    - num_classes (int): Total number of classes.
    - class_names (list): List of class names.
    """
    # Resolve absolute paths for train and val folders
    dataset_dir = os.path.abspath(output_path)
    train_path = os.path.join(dataset_dir, "images/train")
    val_path = os.path.join(dataset_dir, "images/val")

    # Create the dataset configuration dictionary
    dataset_config = {
        "path": dataset_dir,
        "train": train_path,
        "val": val_path,
        "nc": num_classes,
        "names": {i: name for i, name in enumerate(class_names)}
    }

    # Save the configuration to the dataset.yaml file
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"[INFO] dataset.yaml created at: {yaml_path}")

def validate_final_structure(output_dir="/kaggle/working/output"):
    """
    Valida que las carpetas de imágenes y etiquetas contengan archivos coincidentes.

    Parameters:
    - output_dir (str): Carpeta base para PREPROCESSING/.
    """
    partitions = ["train", "val", "test"]
    summary = {}

    # flag = True

    for partition in partitions:
        images = sorted(os.listdir(os.path.join(output_dir, f"dataset/images/{partition}/")))
        labels = sorted(os.listdir(os.path.join(output_dir, f"dataset/labels/{partition}/")))

        
        # if flag:
        #     print(output_dir, f"dataset/images/{partition}/")
        #     flag = False
        #     #open the folder in file explorer
        #     os.system(f"explorer {os.path.join(output_dir, f'dataset/images/{partition}/').replace('/', '\\')}")
        
        if len(images) != len(labels):
            raise ValueError(f"[ERROR] Desbalance entre imágenes y etiquetas en {partition}.")
        summary[partition] = len(images)
    
    pprint({"Validación Final": summary})

def main(dataset_name="bricks"):
    """
    Ejecución principal del pipeline.
    """
    paths = setup_environment(dataset_name=dataset_name)
    pprint({"Rutas Configuradas": paths})

    verify_dataset_structure(paths["raw_images_path"], paths["raw_labels_path"])

    create_preprocessing_structure(paths["output_path"])

    copy_and_partition_data(paths["raw_images_path"], paths["raw_labels_path"], paths["output_path"])

    augment_data(
        input_images=os.path.join(paths["output_path"], "dataset/images/train"),
        input_labels=os.path.join(paths["output_path"], "dataset/labels/train"),
        output_dir=os.path.join(paths["output_path"], "augmented_dataset"),
        num_augmentations=3
    )

    copy_augmented_to_train(
        augmented_dir=os.path.join(paths["output_path"], "augmented_dataset"),
        output_path=paths["output_path"]
    )

    create_dataset_yaml(
        output_path=os.path.join(paths["output_path"], "dataset"),
        num_classes=1,  # Replace with the actual number of classes
        class_names=[dataset_name[:-1]]  # Add all class names here
    )

    validate_final_structure(paths["output_path"])
    print("\n[INFO] Pipeline setup completed with augmentations and dataset.yaml creation.\n")

if __name__ == "__main__":
    main()
