import os
import shutil
from sklearn.model_selection import train_test_split
import json
from pprint import pprint


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
            # el folder "output" del folder de ejecucion
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
        #make output forder
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
            print(f"[ERROR] Carpeta requerida no encontrada: {folder}")
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

    #make sure PREPROCESSING folder exists
    os.makedirs(output_dir, exist_ok=True)
    pprint({"Estructura de Carpetas": output_dir})

    subfolders = [
        "dataset/images/train", "dataset/images/val", "dataset/images/test",
        "dataset/labels/train", "dataset/labels/val", "dataset/labels/test",
        "test_images"
    ]
    for subfolder in subfolders:
        # print(os.path.join(output_dir, subfolder).replace("\\", "/"))
        # #print if it exists
        # print(os.path.exists(os.path.join(output_dir, subfolder).replace("\\", "/")))
        # #open in file explorer
        # # if subfolder == "test_images":
        # os.system(f"explorer {os.path.join(output_dir, subfolder).replace('/', '\\')}")

        os.makedirs(os.path.join(output_dir, subfolder).replace("\\", "/"), exist_ok=True)
       
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

# === Ejecución del Pipeline ===
def main():
    """Ejecución principal del pipeline."""
    paths = setup_environment()
    pprint({"Rutas Configuradas": paths})

    verify_dataset_structure(paths["raw_images_path"], paths["raw_labels_path"])

    create_preprocessing_structure(paths["output_path"])

    copy_and_partition_data(paths["raw_images_path"], paths["raw_labels_path"], paths["output_path"])

    validate_final_structure(paths["output_path"])

    print("\n[INFO] Pipeline completado exitosamente.\n")

if __name__ == "__main__":
    main()
