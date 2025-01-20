import os
import shutil
from sklearn.model_selection import train_test_split
import json
from pprint import pprint
import albumentations as A
import cv2

# === Configuración Inicial ===
def detect_environment():
    """
    Detecta el entorno de ejecución (Kaggle, Google Colab o Local).

    Returns:
    - str: Nombre del entorno detectado.
    """
    if os.path.exists("/kaggle"):
        environment = "kaggle"
    elif "COLAB_GPU" in os.environ:
        environment = "colab"
    else:
        environment = "local"
    pprint({"Detected Environment": environment})
    return environment

def setup_environment(base_path="/kaggle/working/output"):
    """
    Configura el entorno según el sistema detectado y prepara el dataset.

    Parameters:
    - base_path (str): Carpeta base donde se configurará la salida.

    Returns:
    - dict: Rutas configuradas para las imágenes y etiquetas crudas.
    """
    environment = detect_environment()
    print(f"\n[INFO] Entorno detectado: {environment.capitalize()}\n")

    if environment == "kaggle":
        dataset_path = "/kaggle/input/spiled-lego-bricks"
        required_folders = ["Images_600x800", "LabelMe_txt_bricks"]
        for folder in required_folders:
            full_path = os.path.join(dataset_path, folder)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"[ERROR] Carpeta requerida no encontrada: {full_path}")
            print(f"[INFO] Carpeta verificada: {full_path}")

        return {
            "raw_images_path": os.path.join(dataset_path, "Images_600x800"),
            "raw_labels_path": os.path.join(dataset_path, "LabelMe_txt_bricks"),
            "output_path": base_path
        }
    elif environment == "colab":
            from google.colab import userdata
            kaggle_path = "kaggle.json"
            if not os.path.exists(kaggle_path):
                # raise EnvironmentError("[ERROR] Sube tu archivo kaggle.json al entorno Colab en /root/.kaggle/")
                os.makedirs("/root/.kaggle", exist_ok=True)
            
            
                kaggle_user = userdata.get('KaggleUser')
                kaggle_token = userdata.get('KaggleToken')
                if not kaggle_user or not kaggle_token:
                    raise EnvironmentError("[ERROR] No se encontraron las credenciales de Kaggle en Google Colab.")
                kaggle_data = {
                    "username": kaggle_user,
                    "key": kaggle_token
                }
                with open("/root/.kaggle/kaggle.json", "w") as f:
                    json.dump(kaggle_data, f)
                    print("[INFO] Credenciales de Kaggle configuradas en Google Colab.")
            else:
                os.makedirs("/root/.kaggle", exist_ok=True)
                shutil.move(kaggle_path, "/root/.kaggle/kaggle.json")
                print("[INFO] Archivo kaggle.json movido a /root/.kaggle/")
            os.chmod("/root/.kaggle/kaggle.json", 0o600)
            os.makedirs("working", exist_ok=True)
            os.makedirs("working/spiled-lego-bricks", exist_ok=True)
            os.system("kaggle datasets download -d migueldilalla/spiled-lego-bricks -p working/spiled-lego-bricks --unzip")
            os.makedirs("/working/output", exist_ok=True)
            dataset_path = "working/spiled-lego-bricks"

            return {
                "raw_images_path": os.path.join(dataset_path, "Images_600x800"),
                "raw_labels_path": os.path.join(dataset_path, "LabelMe_txt_bricks"),
                "output_path": os.path.join(os.getcwd(), "working", "output")
            }


    elif environment == "local":
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_json_path):
            raise EnvironmentError("[ERROR] Archivo kaggle.json no encontrado en ~/.kaggle/")
        os.makedirs("working", exist_ok=True)
        os.makedirs("working/spiled-lego-bricks", exist_ok=True)
        if not os.listdir("working/spiled-lego-bricks"):
            os.system("kaggle datasets download -d migueldilalla/spiled-lego-bricks -p working/spiled-lego-bricks --unzip")
        os.makedirs("working/output", exist_ok=True)
        dataset_path = "working/spiled-lego-bricks"

        return {
             "raw_images_path": os.path.join(dataset_path, "Images_600x800"),
            "raw_labels_path": os.path.join(dataset_path, "LabelMe_txt_bricks"),
            "output_path": os.path.join(os.getcwd(), "working", "output")
        }
    else:
        while True:
            user_input = input("[PROMPT] No se detectó un entorno. Por favor, escribe 'k' para Kaggle, 'g' para Google Colab, o 'l' para Local: ").strip().lower()
            if user_input in ["k", "g", "l"]:
                return setup_environment_custom(user_input, base_path)
            print("[ERROR] Entrada inválida. Intenta nuevamente.")

def setup_environment_custom(choice, base_path):
    """
    Configura el entorno manualmente basado en la elección del usuario.

    Parameters:
    - choice (str): 'k' para Kaggle, 'g' para Colab, 'l' para Local.
    - base_path (str): Ruta base para la salida.

    Returns:
    - dict: Rutas configuradas para las imágenes y etiquetas crudas.
    """
    if choice == "k":
        return setup_environment()
    elif choice == "g":
        return setup_environment(base_path="working")
    elif choice == "l":
        return setup_environment(base_path="working")
    else:
        raise EnvironmentError("[ERROR] Configuración desconocida.")

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

def load_labels(label_path):
    """
    Carga etiquetas en formato YOLO desde un archivo .txt.

    Parameters:
    - label_path (str): Ruta al archivo de etiquetas en formato YOLO.

    Returns:
    - bboxes (list): Lista de bounding boxes en formato YOLO.
    - class_labels (list): Lista de etiquetas de clase.
    """
    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(class_id))
    return bboxes, class_labels

def save_labels(output_path, bboxes, class_labels):
    """
    Guarda etiquetas en formato YOLO en un archivo .txt.

    Parameters:
    - output_path (str): Ruta donde se guardará el archivo de etiquetas.
    - bboxes (list): Lista de bounding boxes en formato YOLO.
    - class_labels (list): Lista de etiquetas de clase.
    """
    with open(output_path, "w") as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{label} {' '.join(map(str, bbox))}\n")

def main():
    """
    Ejecución principal del pipeline.
    """
    paths = setup_environment()
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

    validate_final_structure(paths["output_path"])
    print("\n[INFO] Pipeline setup completed with augmentations.\n")

if __name__ == "__main__":
    main()
